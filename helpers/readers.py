import json
import os
import shutil
from copy import deepcopy
import argparse
import numpy as np
from matplotlib import pyplot as plt
import torch
from transformers import AutoTokenizer, DistilBertModel
from utils import color_map, color_palette


def not_a_tree(original_edges, sparse_edges, nodes):
    num_parents = sparse_edges.sum(axis=-1)
    for i,num_p in enumerate(num_parents):
        if num_p != 1:
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

def multiple_activities(dataset_type):
    if dataset_type in ['HOMER', 'VH']:
        return False
    elif dataset_type in ['WAH']:
        return True
    raise KeyError('Invalid Dataset type')

def data_dir(dataset_type, data_category, datadir):
    if dataset_type == 'HOMER':
        subdirs = {'train': 'routines_train','test': 'routines_test'}
    elif dataset_type in ['WAH', 'VH']:
        subdirs = {'train': 'train','test': 'test'}
    else:
        raise KeyError('Invalid Dataset type')
    return os.path.join(datadir, subdirs[data_category])



class DistillBERTEmbeddingGenerator():
    def __init__ (self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.map = {'objects':{}, 'activity':{}}

    def __call__(self, token, token_type, key=None):
        if key is None: key=token
        assert token_type in self.map.keys()
        if key in self.map[token_type].keys():
            return 
        token_nl = token.replace('_',' ')
        inputs = self.tokenizer(text=token_nl, return_tensors="pt")
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state[0,1:-1,:]
        last_hidden_states = last_hidden_states.mean(dim=0)
        assert last_hidden_states.size() == torch.Size([768])
        self.map[token_type][key] = last_hidden_states.detach()
    
    def add_objects(self, tokenlist):
        for i,token in enumerate(tokenlist):
            if token is not None:
                self(token, 'objects', key=i)
            else:
                self('Unknown Object', 'objects', key=i)


    def add_activities(self, tokenlist):
        for i,token in enumerate(tokenlist):
            if token is not None:
                self(token, 'activity', key=i)
            else:
                self('Unknown Activity', 'activity', key=i)


class readDataFiles():
    def __init__(self, dirpath, common_data, coarse=False):
        self.dirpath = dirpath
        self.common_data = common_data
        getters = self.homer_get
        lengths = self.homer_len
        
        self.get = getters
        self.len = lengths
        self.activity_key = 'activities_coarse' if coarse else 'activities'

    def homer_get(self, index):
        filename = os.listdir(self.dirpath)[index]
        print('Processing file: ', filename)
        data = json.load(open(os.path.join(self.dirpath,filename)))
        graph_sequence = data["graphs"]
        times = torch.Tensor(data["times"])
        activities = torch.Tensor([self.common_data['activities'].index(a) if a is not None else self.common_data['activities'].index("Idle") for a in data[self.activity_key]+["Idle"]]).to(int)
        return graph_sequence, activities, times, filename.split('.')[0]
    
    def homer_len(self):
        return len(os.listdir(self.dirpath))

class readClassFiles():
    def __init__(self, dirpath, coarse=False):
        self.dirpath = dirpath
        self.getter = self.homer_get
        self.activity_key = 'activities_coarse' if coarse else 'activities'
    
    def __call__(self):
        return self.getter()

    def homer_get(self):
        info = json.load(open(os.path.join(self.dirpath, 'info.json'), 'r'))
        classes = json.load(open(os.path.join(self.dirpath, '..', 'classes.json'), 'r'))
        common_data = {}
        for k in ['dt', 'start_time', 'end_time']:
            if k in classes:
                common_data[k] = classes[k]
            else:
                common_data[k] = info[k]
        common_data['node_classes'] = ['home'] + [n['class_name'] for n in classes['nodes']]   # if not ignore_node(n)]
        common_data['node_categories'] = ['home'] + [n['category'] for n in classes['nodes']]  # if not ignore_node(n)]
        common_data['activities'] = {}
        common_data['activities'] = ["Unknown"] + ["Idle"] + classes[self.activity_key]
        common_data['activity_conversion'] = {'to_coarse':{}, 'to_fine':{}}
        if self.activity_key == 'activities_coarse':
            common_data['activity_conversion']['to_coarse'] = {coarse:coarse for coarse, _ in classes['activities_hierarchy'].items()}
            common_data['activity_conversion']['to_fine'] = classes['activities_hierarchy']
        if self.activity_key == 'activities':
            common_data['activity_conversion']['to_coarse'] = {}
            common_data['activity_conversion']['to_fine'] = {}
            for coarse, fine_list in classes['activities_hierarchy'].items():
                for fine in fine_list:
                    common_data['activity_conversion']['to_coarse'][fine] = coarse
                    common_data['activity_conversion']['to_fine'] [fine] = [fine]
        print (common_data['activities'])

        # Diagonal nodes are always irrelevant
        common_data['nonstatic_edges'] = 1 - torch.eye(len(common_data['node_classes']))
        # Rooms, furniture and appliances nodes don't move
        common_data['nonstatic_edges'][np.where(np.array(common_data['node_categories']) == "home"),:] = 0
        common_data['nonstatic_edges'][np.where(np.array(common_data['node_categories']) == "Rooms"),:] = 0
        common_data['nonstatic_edges'][np.where(np.array(common_data['node_categories']) == "Furniture"),:] = 0
        common_data['nonstatic_edges'][np.where(np.array(common_data['node_categories']) == "Decor"),:] = 0
        common_data['nonstatic_edges'][np.where(np.array(common_data['node_categories']) == "Appliances"),:] = 0
        common_data['seen_edges'] = torch.zeros_like(common_data['nonstatic_edges'])
        common_data['home_graph'] = None

        common_data['edge_keys'] = classes['edges']
        static = lambda category : category in ["Furniture", "Room"]
        common_data['static_nodes'] = [n['id'] for n in classes['nodes'] if static(n['category'])]  # and not ignore_node(n)]
        common_data['ignored_node_classes'] = ['wall','light', 'ceiling', 'floor', 'curtain', 'maindoor']
        common_data['static_node_categories'] = ['Rooms', 'Furniture', 'Decor', 'Appliances', 'Home']
        
        return common_data



class ProcessDataset():
    def __init__(self, 
                 datadir, 
                 output_path,
                 dataset_type,
                 overwrite=False,
                 stack_in_time=True,
                 coarse=False,
                 dt=None
                 ):

        self.LMEmbedding = DistillBERTEmbeddingGenerator()

        data_path = (os.path.join(datadir, 'train'), os.path.join(datadir, 'test'))
        metadata_path = os.path.join(datadir, 'metadata.json')
        if not (os.path.exists(data_path[0]) and os.path.exists(data_path[1])):
            print('The data directory must contain both routines_train and routines_test directories')
        if not (os.path.exists(metadata_path)):
            print('The metadata is needed!!!')

        if dt is not None:
            output_path += f'_{dt}'
        if coarse:
            output_path += '_coarse'
        output_train_path = os.path.join(output_path, 'train')
        output_test_path = os.path.join(output_path, 'test')
        if os.path.exists(output_train_path): 
            if not overwrite:
                overwrite = input(f'Dataset seems to already exist at {output_path}!! Overwrite? (y/n)') == 'y'
            if not overwrite: raise InterruptedError('Cancelling data processing...')
            shutil.rmtree(output_path)
        os.makedirs(output_train_path)
        if os.path.exists(output_test_path): 
            if not overwrite:
                overwrite = input(f'Dataset seems to already exist at {output_path}!! Overwrite? (y/n)') == 'y'
            if not overwrite: raise InterruptedError('Cancelling data processing...')
            shutil.rmtree(output_path)
        os.makedirs(output_test_path)

        self.dataset_type = dataset_type

        self.common_data = readClassFiles(self.dataset_type, datadir, coarse=coarse)()
        if dt is not None: self.common_data['dt'] = dt
        self.common_data['dataset_type'] = self.dataset_type
        self.common_data['multiple_activities'] = multiple_activities(self.dataset_type)
        self.common_data['index_list'] = {'train':[], 'test':[]}
        self.LMEmbedding.add_objects(self.common_data['node_classes'])
        self.LMEmbedding.add_activities(self.common_data['activities'])
        
        self.read_data(readDataFiles(self.dataset_type, data_dir(self.dataset_type, 'test', datadir), self.common_data, coarse=coarse), output_test_path, stack_in_time, index_list=self.common_data['index_list']['test'], plot_graphs=True)
        self.read_data(readDataFiles(self.dataset_type, data_dir(self.dataset_type, 'train', datadir), self.common_data, coarse=coarse), output_train_path, stack_in_time, index_list=self.common_data['index_list']['train'])
        
        self.common_edge_data = {}
        for key in ['seen_edges', 'nonstatic_edges', 'home_graph']:
            if key in self.common_data:
                self.common_edge_data[key] = self.common_data[key]
                del self.common_data[key]


        json.dump(self.common_data, open(os.path.join(output_path, 'common_data.json'), 'w'), indent=4)
        torch.save(self.LMEmbedding.map, os.path.join(output_path, 'common_embedding_map.pt'))
        torch.save(self.common_edge_data, os.path.join(output_path, 'common_edge_data.pt'))

    def plot_a_day(self, data, class_names, axs, axsdense):
        obj_mask = data['active_edges'].sum(-1)>0
        dense_gt = data['edges'][1:,:,:].argmax(-1).permute(1,0)[obj_mask].permute(1,0)
        snapshot_gt = deepcopy(data['edges'][1:,:,:].argmax(-1))
        snapshot_gt[snapshot_gt == data['edges'][:-1,:,:].argmax(-1)] = -1
        snapshot_gt = snapshot_gt.permute(1,0)[obj_mask].permute(1,0)
        class_names = [c for c,mask in zip(class_names,obj_mask) if mask]
        activities_gt = data['activity'][:-1]
        num_obj_total = 0
        for i,activity in enumerate(self.common_data['activities']):
            if activity not in color_map:
                color_map[activity] = color_palette[i%len(color_palette)]
        yv, xv = np.meshgrid(np.arange(snapshot_gt.size()[0]), np.arange(snapshot_gt.size()[1]))
        for obj in range(snapshot_gt.size()[1]):
            axs.plot([obj,obj], [0,snapshot_gt.size()[0]], 'k', alpha=0.5, linewidth=0.5)
        axs.scatter([(obj + 1) for _ in range(snapshot_gt.size()[0])], np.arange(snapshot_gt.size()[0]), marker='o', c=[color_map[self.common_data['activities'][a]] for a in activities_gt])
        axs.scatter(xv.reshape(-1), yv.reshape(-1), marker='o', c=[color_palette[c] if c>=0 else [1,1,1,0] for c in snapshot_gt.cpu().numpy().T.reshape(-1)])
        axs.set_xticks(np.arange(int(obj_mask.sum())+1), rotation=90)
        axs.set_xticklabels(class_names+['ACTIVITY'], rotation=90)
        axsdense.scatter([(obj + 1) for _ in range(dense_gt.size()[0])], np.arange(dense_gt.size()[0]), marker='o', c=[color_map[self.common_data['activities'][a]] for a in activities_gt])
        axsdense.scatter(xv.reshape(-1), yv.reshape(-1), marker='o', c=[color_palette[c] for c in dense_gt.cpu().numpy().T.reshape(-1)])
        axsdense.set_xticks(np.arange(int(obj_mask.sum())+1), rotation=90)
        axsdense.set_xticklabels(class_names+['ACTIVITY'], rotation=90)
        num_obj_total += obj_mask.sum()
        width = float(num_obj_total)/5
        height = int(dense_gt.size()[0])/10
        return axs, axsdense, (width, height)

    def read_data(self, datareader, output_dir, stack_in_time, index_list=[], plot_graphs=False):
        if plot_graphs:
            figdense, axsdense = plt.subplots(1, datareader.len())
            fig, axs = plt.subplots(1, datareader.len())
            width, height = 2,0
        for idx in range(datareader.len()):
            graph_sequence, activities, times, filename = datareader.get(idx)
            nodes, edges, active_edges_mask, class_names = self.read_graphs(graph_sequence)
            if stack_in_time:
                edges, activities, times = self.stack_routine(edges, activities, times)
            f_out = os.path.join(output_dir,filename+'.pt')
            index_list.append((filename+'.pt', edges.size()[0]))
            data = {'nodes': nodes, 
                        'edges': edges, 
                        'times': times,
                        'active_edges': active_edges_mask,
                        'activity': activities}
            if plot_graphs:
                axs[idx], axsdense[idx], (dw,dh) =self.plot_a_day(data, class_names, axs[idx], axsdense[idx])
                width += dw
                height = max(dh, height)
            torch.save(data, f_out)
        if plot_graphs:
            fig.set_size_inches(width,height)
            fig.tight_layout()
            fig.savefig(output_dir+f'_movements.png')
            figdense.set_size_inches(width,height)
            figdense.tight_layout()
            figdense.savefig(output_dir+f'_locations.png')
        return

    def get_node_index(self, node, index, temp_data):
        class_index = self.common_data['node_classes'].index(node['class_name'])
        assert len(temp_data['node_ids'])==index
        temp_data['node_idx_from_id'][node['id']] = index
        temp_data['node_ids'].append(node['id'])
        temp_data['node_class_from_idx'].append(node['class_name'])
        return class_index

    def read_graphs(self, graphs):
        temporary_data = {}
        temporary_data['node_idx_from_id'] = {}
        temporary_data['node_ids'] = []
        temporary_data['node_class_from_idx'] = []
        home_node = [{'id':-1, 'class_name':"home", 'category':"Home"}]
        nodes_in_graph = [node for node in graphs[0]['nodes'] if node['class_name'] not in self.common_data['ignored_node_classes']] + home_node
        num_nodes = len(nodes_in_graph)
        node_features = np.zeros((num_nodes))
        nonstatic_edges = torch.Tensor(1 - np.eye(num_nodes))
        for j,n in enumerate(nodes_in_graph):
            node_features[j] = self.get_node_index(n, j, temp_data=temporary_data)
            if n['category'] in self.common_data['static_node_categories']:
                nonstatic_edges[j,:] = 0
        edge_features = np.zeros((len(graphs), num_nodes, num_nodes))
        room_node_idxs = [temporary_data['node_idx_from_id'][n['id']] for n in nodes_in_graph if n['category']=="Rooms"]
        for i,graph in enumerate(graphs):
            for room_node_index in room_node_idxs:
                edge_features[i,room_node_index, temporary_data['node_idx_from_id'][-1]] = 1
            for e in graph['edges']:
                if e['relation_type'] in self.common_data['edge_keys'] and e['from_id'] in temporary_data['node_ids'] and e['to_id'] in temporary_data['node_ids']:
                    edge_features[i,temporary_data['node_idx_from_id'][e['from_id']],temporary_data['node_idx_from_id'][e['to_id']]] = 1
            original_edges = edge_features[i,:,:]
            edge_features[i,:,:] = _sparsify(edge_features[i,:,:])
            edge_features[i, temporary_data['node_idx_from_id'][-1], temporary_data['node_idx_from_id'][-1]] = 1
            if (edge_features[i,:,:].sum(axis=-1)).max() != 1:
                not_a_tree(original_edges, edge_features[i,:,:], temporary_data['node_class_from_idx'])
        
        return torch.Tensor(node_features), torch.Tensor(edge_features), torch.Tensor(nonstatic_edges), temporary_data['node_class_from_idx']


    def stack_routine(self, edges, activities, times):
        times = torch.cat([times, torch.Tensor([float("Inf")])], dim=-1)

        data_idx = -1
        all_edges = []
        all_times = []
        all_activities = []
        for t in range(self.common_data['start_time'], self.common_data['end_time']+1, self.common_data['dt']):
            while t >= times[data_idx+1]:
                ## Read in the first non-idle activity in the time range
                if len(all_activities) > 0 and all_activities[-1] == self.common_data['activities'].index('Idle'):
                    all_activities[-1] = (activities[data_idx]).unsqueeze(0)
                data_idx += 1
            all_edges.append(edges[data_idx].unsqueeze(0))
            all_times.append(torch.Tensor([t]))
            all_activities.append((activities[data_idx]).unsqueeze(0))
        stacked_edges = torch.cat(all_edges)
        stacked_activities = torch.cat(all_activities)
        stacked_times = torch.cat(all_times)

        return stacked_edges, stacked_activities, stacked_times
 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='data/UnitTestsWithActivityConversion0405/AorB_', help='Path where the data lives. Must contain routines, info and classes json files.')
    parser.add_argument('--stack_in_time', action='store_true', help='Set if fixed timestep sequence is desired rather than action-based')
    parser.add_argument('--coarse', action='store_true', help='Set if coarse actions should be read')
    parser.add_argument('--dt', type=int, help='Custom timestep')
    parser.add_argument('-y', action='store_true', help='Overwrite if path exists.')
    args = parser.parse_args()

    args.stack_in_time = True

    if not args.stack_in_time:
        stop = input(f'Are you sure you want stacking in time as {args.stack_in_time}? (y/n)')
        if stop.lower() == 'n': 
            change = input(f'Change stacking in time to {not args.stack_in_time}? (y/n)')
            if change == 'y': 
                args.stack_in_time = not args.stack_in_time
                stop = 'y'
        assert stop.lower() != 'n'

    output_dirname = 'processed_seqLM' if args.stack_in_time else 'processed_obj_actLM'

    ProcessDataset(args.path, output_path=os.path.join(args.path, output_dirname), overwrite=args.y, stack_in_time=args.stack_in_time, coarse=args.coarse, dt=args.dt)
