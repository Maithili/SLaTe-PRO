import json
import os
import torch
from torch.utils.data import DataLoader


class OneHotEmbedder():
    def __init__(self):
        pass

    def get_func(self, class_list):
        return lambda idxs: torch.nn.functional.one_hot(idxs.to(int), num_classes=len(class_list))

class CollateToDict():
    def __init__(self, dict_labels):
        self.dict_labels = dict_labels

    def __call__(self, tensor_tuple):
        data = {label:torch.Tensor() for label in self.dict_labels}
        for i,label in enumerate(self.dict_labels):      
                data[label] = torch.cat([tensors[i].unsqueeze(0) for tensors in tensor_tuple], dim=0)
        return data

class DataSplit_WAH():
    def __init__(self, routines_dir, idx_map, node_embedder=OneHotEmbedder):
        self.routines_dir = routines_dir
        self.collate_fn = CollateToDict(['edges', 'nodes', 'activity', 'dynamic_edges_mask'])
        self.idx_map = idx_map
        self.files = list(set([im[0] for im in self.idx_map]))
        self.files.sort()
        self.node_embedder = node_embedder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        data = torch.load(os.path.join(self.routines_dir, self.files[idx]+'.pt'))
        nodes, edges, active_edges_mask, activity = data
        nodes = self.node_embedder(nodes).unsqueeze(0).repeat(edges.size()[0],1,1)
        active_edges_mask = active_edges_mask.unsqueeze(0).repeat(edges.size()[0],1,1)
        activity = activity.unsqueeze(0).repeat(edges.size()[0],1)
        return edges, nodes, activity, active_edges_mask

class DataSplit_HOMER():
    def __init__(self, routines_dir, idx_map, node_embedder=OneHotEmbedder):
        self.routines_dir = routines_dir
        self.collate_fn = CollateToDict(['edges', 'nodes', 'activity', 'dynamic_edges_mask'])
        self.idx_map = idx_map
        self.node_embedder = node_embedder

    def get_data_file_index(self, index):
        prev_datapoints = 0
        for filename,num_idxs in self.idx_map:
            if prev_datapoints+index < num_idxs:
                return filename, index-prev_datapoints
            prev_datapoints += num_idxs
        return None, None


    def __len__(self):
        return sum([i[1] for i in self.idx_map])

    def __getitem__(self, idx: int):
        filename, dataindex = self.get_data_file_index(idx)
        data = torch.load(os.path.join(self.routines_dir, filename))
        edges, nodes, active_edges_mask, activity = data
        return edges[dataindex,:,:], self.node_embedder(nodes), activity, active_edges_mask


class ActivitiesDataset():
    def __init__(self, data_path):

        with open(os.path.join(data_path, 'common_data.json')) as f:
            self.common_data = json.load(f)

        
        self.params = {}
        self.params['batch_size'] = 1
        self.params['multiple_activities'] = True

        self.node_classes = self.common_data['node_classes']

        node_embedder = OneHotEmbedder()
        

        # Generate train and test loaders
        self.train = DataSplit_HOMER(os.path.join(data_path,'train'), idx_map=self.common_data['index_list']['train'], node_embedder=node_embedder.get_func(self.node_classes))
        self.test = DataSplit_HOMER(os.path.join(data_path,'test'), idx_map=self.common_data['index_list']['test'],node_embedder=node_embedder.get_func(self.node_classes))
        print('Train split has ',len(self.train),' routines')
        print('Test split has ',len(self.test),' routines')

        # Infer parameters for the model
        model_data = self.test.collate_fn([self.test[0]])
        self.params['n_nodes'] = model_data['nodes'].size()[-2]
        self.params['n_len'] = model_data['nodes'].size()[-1]
        print(self.common_data['activities'], len(self.common_data['activities']))
        self.params['n_activities'] = len(self.common_data['activities'])
        if None in self.common_data['activities']:
            self.params['null_activity_idx'] = self.common_data['activities'].index(None)
        else:
            self.params['null_activity_idx'] = -1

    def get_train_loader(self):
        return DataLoader(self.train, num_workers=min(4,os.cpu_count()), batch_size=self.params['batch_size'], collate_fn=self.train.collate_fn)

    def get_test_loader(self):
        return DataLoader(self.test, num_workers=min(4,os.cpu_count()), batch_size=self.params['batch_size'], collate_fn=self.test.collate_fn)

    def get_test_split(self):
        return self.test

    def get_single_example_test_loader(self):
        return DataLoader(self.test, num_workers=min(4,os.cpu_count()), batch_size=1, collate_fn=self.test.collate_fn)
    
