import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from encoders import time_external
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer
sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def not_a_tree(original_edges, sparse_edges, nodes):
    num_parents = sparse_edges.sum(axis=-1)
    for i,num_p in enumerate(num_parents):
        if num_p>1:
            print(f'Node {nodes[i]} has parents : {list(np.array(nodes)[(np.argwhere(sparse_edges[i,:] > 0)).squeeze()])}')
            print(f'Node {nodes[i]} originally had parents : {list(np.array(nodes)[(np.argwhere(original_edges[i,:] > 0)).squeeze()])}')

def _densify(edges):
    dense_edges = edges.copy()
    for _ in range(edges.shape[-1]):
        new_edges = np.matmul(dense_edges, edges)
        new_edges = new_edges * (dense_edges==0)
        if (new_edges==0).all():
            break
        dense_edges += new_edges
    return dense_edges

def _sparsify(edges):
    dense_edges = _densify(edges.copy())
    remove = np.matmul(dense_edges, dense_edges)
    sparse_edges = dense_edges * (remove==0).astype(int)
    return sparse_edges

class CustomObjectEmbedder():
    def __init__(self, object_list):
        object_desc = json.load(open("object_desc.json"))
        assert set(object_list) == set(object_desc.keys()), f"Object list does not match object description. {set(object_list) - set(object_desc.keys())} and {set(object_desc.keys()) - set(object_list)}"
        for object_name in object_desc:
            object_desc[object_name]['embedding_description'] = sentence_transformer.encode(object_desc[object_name]['description'])
            object_desc[object_name]['embedding_action'] = sum([sentence_transformer.encode(a) for a in object_desc[object_name]['action_list']])/len(object_desc[object_name]['action_list'])
            object_desc[object_name]['embedding_combined'] = object_desc[object_name]['embedding_description'] + object_desc[object_name]['embedding_action']
        self.embeddings = {object_list.index(object_name):object_desc[object_name]['embedding_combined'] for object_name in object_desc}

    def __call__(self, object_name):
        return torch.stack([self.embeddings[int(o.item())] for o in object_name])

class OneHotEmbedder(): 
    def __init__(self, class_list):
        self.class_list = class_list

    def __call__(self, idxs):
        return F.one_hot(idxs.to(int), num_classes=len(self.class_list))

class IdentityEmbedder():
    def __call__(self, idxs):
        return idxs.float()

class BertEmbedder():
    def __init__(self, map_file, map_type):
        bertmap = torch.load(map_file)
        self.map = bertmap[map_type]
        # self.object_map = bertmap['objects']
        # self.activity_map = bertmap['activity']

    def __call__(self, idxs):
        return torch.stack([self.map[i.to(int).item()] for i in idxs])

    def object_mapper(self):
        return lambda idxs: torch.stack([self.object_map[i.to(int).item()] for i in idxs])
    
    def activity_mapper(self):
        return lambda idxs: torch.stack([self.activity_map[i.to(int).item()] for i in idxs])

    
class ConceptNetEmbedder():

    def __init__(self, file='helpers/numberbatch-en-19.08.txt'):
        '''
        Loads Conceptnet Numberbatch from the text file
        Args:
            file: path to numberbatch_en.txt file (must be on local system)
        Output:
            embeddings_index: dictionary mapping objects to their numberbatch embbs
            num_feats: length of numberbatch embedding
        '''

        # Create dictionary of object: vector
        self.embeddings_index = dict()
        with open(file, 'r', encoding="utf8") as f:
            # Parse text file to populate dictionary
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs

        self.num_feats = len(coefs)
        print('ConceptNet loaded !!!')
        self.synonyms = {
            'tvstand': 'tv_stand',
            'cookingpot': 'cooking_pot',
            'knifeblock': 'knife_block',
        }

    def __call__(self, token):
        token = token.lower()
        if token in self.synonyms: token = self.synonyms[token]
        if token in self.embeddings_index:
            assert len(self.embeddings_index[token]) == self.num_feats
            return self.embeddings_index[token]
        elif '_' in token:
            subtokens = token.split('_')
            try:
                assert all([len(self.embeddings_index[t]) == self.num_feats for t in subtokens])
                return sum([self.embeddings_index[t] for t in subtokens])
            except:
                raise KeyError(f'{subtokens} cannot be embedded through ConceptNet')
        raise KeyError(f'{token} cannot be embedded through ConceptNet')

    def get_func(self, class_list):
        embedding_list = [self(cls) for cls in class_list]
        embedding_tensor = torch.Tensor(np.array(embedding_list))
        embedding_func = lambda idxs : torch.matmul(torch.nn.functional.one_hot(idxs.to(int)).float(), embedding_tensor.float())
        return embedding_func



def collate(tensor_tuple):
    data = {label:torch.Tensor() for label in tensor_tuple[0].keys()}
    for label in data:
        if isinstance(tensor_tuple[0][label], torch.Tensor):
            data[label] = torch.cat([tensors[label].unsqueeze(0) for tensors in tensor_tuple], dim=0)
        elif callable(tensor_tuple[0][label]):
            data[label] = tensor_tuple[0][label]
        else:
            raise Exception(f"DataSplit output must be either tensor or function. Found {type(tensor_tuple[0][label])} for {label}")
    return data


class DataSplit():
    def __init__(self, routines_dir, time_encoder, filerange=(None, None), node_embedder=lambda x:x, activity_embedder=lambda x:x, activity_dropout=0.0):
        self.time_encoder = time_encoder
        self.routines_dir = routines_dir
        self.collate_fn = collate
        self.files = [name for name in os.listdir(self.routines_dir) if os.path.isfile(os.path.join(self.routines_dir, name))]
        self.files.sort()
        if filerange[1] is not None:
            self.files = self.files[:filerange[1]]
        if filerange[0] is not None:
            self.files = self.files[filerange[0]:]
        self.node_embedder = node_embedder
        self.activity_embedder = activity_embedder
        # self.activity_masks = [None for _ in self.files]
        self.activity_dropout = activity_dropout

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        data = torch.load(os.path.join(self.routines_dir, self.files[idx]))
        if data['times'] is None:
            data['times'] = torch.zeros((data['edges'].size()[0]-1, 1))
        movements = (data['edges'][1:,:,:]-data['edges'][:-1,:,:]).max(-1).values
        movements = movements.view(data['edges'].size()[0]-1, data['edges'].size()[1], 1)

        edges = data['edges']
        node_features = self.node_embedder(data['nodes']).unsqueeze(0).repeat(data['edges'].size()[0],1,1)
        node_ids = data['nodes'].unsqueeze(0).repeat(data['edges'].size()[0],1,1)
        activity_feature = self.activity_embedder(data['activity'])
        activity_id = data['activity']
        time_feature = self.time_encoder(data['times'])
        time = data['times']
        dynamic_edges_mask = data['active_edges'].unsqueeze(0).repeat(data['times'].size()[0],1,1).to(int)

        datapoint = {
            'edges': edges, 
            'node_features': node_features, 
            'node_ids': node_ids,
            'activity_features': activity_feature,
            'activity_ids': activity_id,
            'time_features': time_feature,
            'time': time,
            'dynamic_edges_mask': dynamic_edges_mask,
            'activity_mask_drop': torch.ones_like(activity_id).to(bool),
            'node_embedder': self.node_embedder,
            'activity_embedder': self.activity_embedder,
        }

        return datapoint

class RoutinesDataset():
    def __init__(self, data_path, 
                 time_encoder = time_external, 
                 batch_size = 1,
                 use_conceptnet = False,
                 activity_dropout = 0.0):

        with open(os.path.join(data_path, 'common_data.json')) as f:
            self.common_data = json.load(f)

        self.time_encoder = time_encoder
        
        self.params = {}
        self.params['dt'] = self.common_data.get('dt',None)
        self.params['batch_size'] = batch_size
        self.params['multiple_activities'] = self.common_data.get('multiple_activities', False)

        self.node_classes = self.common_data['node_classes']
        self.node_categories = self.common_data.get('node_categories', [None]*len(self.common_data['node_classes']))
        self.edge_keys = self.common_data['edge_keys']
        self.static_nodes = self.common_data.get('static_nodes', [None]*len(self.common_data['node_classes']))

        node_embedder = CustomObjectEmbedder(self.node_classes)
        activity_embedder = IdentityEmbedder()
        
        # Generate train and test loaders
        self.train = DataSplit(os.path.join(data_path,'train'), self.time_encoder, filerange=(None,-1), node_embedder=node_embedder, activity_embedder=activity_embedder, activity_dropout=activity_dropout)
        self.val = DataSplit(os.path.join(data_path,'train'), self.time_encoder, filerange=(-1,None), node_embedder=node_embedder, activity_embedder=activity_embedder, activity_dropout=activity_dropout)
        self.test = DataSplit(os.path.join(data_path,'test'), self.time_encoder, filerange=(None,None), node_embedder=node_embedder, activity_embedder=activity_embedder, activity_dropout=activity_dropout)
        print('Train split has ',len(self.train),' routines')
        print('Val split has ',len(self.val),' routines')
        print('Test split has ',len(self.test),' routines')

        # Infer parameters for the model
        model_data = self.test.collate_fn([self.test[0]])
        self.params['n_nodes'] = model_data['node_features'].size()[-2]
        self.params['n_len'] = model_data['node_features'].size()[-1]
        self.params['act_len'] = model_data['activity_features'].size()[-1]
        self.params['n_stations'] = 5
        print("Inferred model parameters:\n")
        print(f"Number of nodes: {self.params['n_nodes']}")
        print(f"Number of activities: {self.params['act_len']}")
        print(f"Size of node embeddings: {self.params['n_len']}")

    def get_train_loader(self):
        return DataLoader(self.train, num_workers=os.cpu_count(), batch_size=self.params['batch_size'], collate_fn=self.train.collate_fn)

    def get_test_loader(self):
        return DataLoader(self.test, num_workers=os.cpu_count(), batch_size=self.params['batch_size'], collate_fn=self.test.collate_fn)

    def get_val_loader(self):
        return DataLoader(self.val, num_workers=os.cpu_count(), batch_size=self.params['batch_size'], collate_fn=self.test.collate_fn)

    def get_test_split(self):
        return self.test

    def get_single_example_test_loader(self):
        return DataLoader(self.test, num_workers=os.cpu_count(), batch_size=1, collate_fn=self.test.collate_fn)
    
    def get_object_consistency(self):
        obj_moved = None
        kernel = torch.tensor([0,0,0,0,0,0,1,1,1,1,1,1,
                               1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,1,-1).float()
        for datapoint in self.train: # + self.test + self.val:
            changes = (datapoint['edges'][1:].argmax(-1) != datapoint['edges'][:-1].argmax(-1)).to(int)
            obj_moved_around_now = (torch.nn.functional.conv1d(changes.unsqueeze(0).permute(2,0,1).float(), kernel, padding='same').permute(1,2,0).squeeze(0)>0).to(int)

            if obj_moved is None:
                obj_moved = obj_moved_around_now
            else:
                obj_moved += obj_moved_around_now

        return obj_moved
