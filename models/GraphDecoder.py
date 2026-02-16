import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
sys.path.append('helpers')
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning.core.module import LightningModule



def get_masks(gt_tensor, output_tensor, input_tensor):
    masks = {}
    masks['gt_negatives'] = (gt_tensor == input_tensor).cpu()
    masks['gt_positives'] = (gt_tensor != input_tensor).cpu()
    masks['out_negatives'] = (output_tensor == input_tensor).cpu()
    masks['out_positives'] = (output_tensor != input_tensor).cpu()
    masks['tp'] = np.bitwise_and(masks['out_positives'], masks['gt_positives']).to(bool)
    masks['fp'] = np.bitwise_and(masks['out_positives'], masks['gt_negatives']).to(bool)
    masks['tn'] = np.bitwise_and(masks['out_negatives'], masks['gt_negatives']).to(bool)
    masks['fn'] = np.bitwise_and(masks['out_negatives'], masks['gt_positives']).to(bool)
    masks['correct'] = gt_tensor == output_tensor
    masks['wrong'] = gt_tensor != output_tensor
    return masks


class GraphDecoderModule(LightningModule):
    def __init__(self, model_configs):
        
        super().__init__()

        self.cfg = model_configs
        self.n_len = model_configs.n_len
        self.c_len = model_configs.c_len
        self.edge_importance = 'predicted'
        self.hidden_influence_dim = 20

        self.edges_update_input_dim = self.hidden_influence_dim*5 + self.c_len
        self.nodes_update_input_dim = self.hidden_influence_dim*2 + self.n_len + self.c_len
        
        self.mlp_context = nn.Sequential(nn.Linear(self.c_len, self.c_len),
                                             nn.ReLU(),
                                             nn.Linear(self.c_len, self.c_len),
                                             )


        mlp_hidden = model_configs.hidden_layer_size

        self.mlp_influence = nn.Sequential(nn.Linear(2*self.n_len+1, mlp_hidden),
                                                    nn.ReLU(),
                                                    nn.Linear(mlp_hidden, self.hidden_influence_dim),
                                                    )

        self.mlp_update_importance = nn.Sequential(nn.Linear(self.edges_update_input_dim, self.hidden_influence_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(self.hidden_influence_dim, 1)
                                                    )
                                    
        self.mlp_update_edges = nn.Sequential(nn.Linear(self.edges_update_input_dim, self.hidden_influence_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(self.hidden_influence_dim, 1)
                                                    )

    
    def location_loss (self, x,y):
        return (nn.CrossEntropyLoss(reduction='none')(x.squeeze(-1).permute(0,2,1), y.squeeze(-1).long()))
    def inference_location(self, x):
        return x.squeeze(-1).argmax(-1)

    def class_loss (self, xc,yc):
        return nn.CrossEntropyLoss(reduction='none')(xc.permute(0,2,1), yc.long())
    def inference_class (self, xc):
        return xc.argmax(-1)


    def get_time_context(self, t, context_time):
        time_context = context_time        
        time_context = self.mlp_context(time_context)
        return time_context.view(-1, self.c_len)

    def forward(self, edges, nodes, context, mask=None):
        """
        Args:
            adjacency: batch_size x from_nodes x to_nodes x 1
            edge_features: batch_size x from_nodes x to_nodes x edge_feature_len
            nodes: batch_size x num_nodes x node_feature_len
            context_curr: batch_size x context_len
            context_query: batch_size x context_len
            mask (opt): batch_size x num_nodes
        """

        batch_size, num_nodes, node_feature_len = nodes.size()
        batch_size_e, num_f_nodes, num_t_nodes = edges.size()

        if mask is None:
            mask = torch.ones((batch_size, num_nodes)).to('cuda')
        
        # Sanity check input dimensions
        assert batch_size == batch_size_e, "Different edge and node batch sizes"
        assert self.n_len == node_feature_len, (str(self.n_len) +'!='+ str(node_feature_len))
        self.n_nodes = num_nodes
        assert self.n_nodes == num_f_nodes, (str(self.n_nodes) +'!='+ str(num_f_nodes))
        assert self.n_nodes == num_t_nodes, (str(self.n_nodes) +'!='+ str(num_t_nodes))
        batch_size, num_nodes, node_feature_len = nodes.size()

        context = context.view(size=[batch_size, self.c_len])

        x = self.collate_edges(edges=edges.unsqueeze(-1), nodes=nodes)
        x = x.view(
            size=[batch_size * self.n_nodes * self.n_nodes, 
                  2*self.n_len+1])
        x = self.mlp_influence(x)
        x = x.view(
            size=[batch_size, 
                  self.n_nodes, 
                  self.n_nodes, 
                  self.hidden_influence_dim])

        if self.edge_importance == 'predicted':
            ## importance update
            imp = self.message_collection_edges(x, edges.unsqueeze(-1), context, mask)
            imp = imp.view(
                size=[batch_size * self.n_nodes * self.n_nodes, 
                    self.edges_update_input_dim])
            imp = self.mlp_update_importance(imp).view(size=[batch_size, 
                                            self.n_nodes, 
                                            self.n_nodes,
                                            1])
        elif self.edge_importance == 'all':
            imp = torch.ones_like(edges.unsqueeze(-1))
        elif self.edge_importance == 'existing':
            imp = edges.unsqueeze(-1)
        else:
            raise KeyError(f'Edge Importance given as ({self.edge_importance}) is not among predicted, all or existing')

        ## edge message passing
        xe = self.message_collection_edges(x, imp, context, mask)
        
        ## edge update
        xe = xe.view(
            size=[batch_size * self.n_nodes * self.n_nodes, 
                self.edges_update_input_dim])
        xe = self.mlp_update_edges(xe).view(size=[batch_size, 
                                                self.n_nodes, 
                                                self.n_nodes])
        
        imp = imp.view(size=[batch_size, self.n_nodes, self.n_nodes])

        edges_inferred = xe

        pred_activity = torch.ones((batch_size, self.n_nodes, 1)).to('cuda')*0.5
        pred_dynamic = torch.ones((batch_size, self.n_nodes, 1)).to('cuda')*0.5

        return edges_inferred, pred_activity, pred_dynamic


    def step(self, batch):
        edges = batch['edges']
        nodes = batch['nodes']
        y_edges = batch['y_edges']
        y_nodes = batch['y_nodes']
        dyn_edges = batch['dynamic_edges_mask']
        
        time_context = self.get_time_context(batch['time'], batch['context_time'])
        
        context = time_context
 
        edges_pred, nodes_pred, imp = self(edges, nodes, context)

        assert edges_pred.size() == dyn_edges.size(), f'Size mismatch in edges {edges_pred.size()} and dynamic mask {dyn_edges.size()}'
        edges_pred[dyn_edges == 0] = -float('inf')

        edges_inferred = F.softmax(edges_pred, dim=-1)
        edges_inferred[dyn_edges == 0] = edges[dyn_edges == 0]

        evaluate_node = dyn_edges.sum(-1) > 0

        input = {'class':self.inference_class(nodes), 
                 'location':self.inference_location(edges)}
                 
        output_probs = {'class':nodes_pred, 
                        'location':edges_inferred}

        gt = {'class':self.inference_class(y_nodes), 
              'location':self.inference_location(y_edges)}

        losses = {'class':self.class_loss(output_probs['class'], gt['class']),
                  'location':self.location_loss(edges_pred, gt['location']),
                  }

        output = {'class':self.inference_class(output_probs['class']),
                  'location':self.inference_location(edges_inferred)}
 
        loss = {'graph': losses['location'][evaluate_node].mean(), 'context_process': losses['context_process'].mean(), 'context_predict': losses['context_predict'].mean()}

        assert (sum(loss.values()).size() == torch.Size([]))

        details = {'input':input, 
                   'output_probs':output_probs, 
                   'gt':gt, 
                   'losses':losses, 
                   'output':output, 
                   'evaluate_node':evaluate_node,
                   'importance_weights':imp,
                   }

        return loss, details

        
    def training_step(self, batch, batch_idx):
        loss, details = self.step(batch)
        self.log('Train loss',loss)
        self.log('Train',details['logs'])
        assert (sum(loss.values())).size() == torch.Size([])
        return sum(loss.values())

    def test_step(self, batch, batch_idx):
        loss, details = self.step(batch)
        self.log('Test loss',loss)
        self.log('Test',details['logs'])
        return 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def collate_edges(self, edges, nodes):
        # nodes_repeated : batch_size x nodes x repeat dimension x node_feature_len
        nodes_repeated = nodes.unsqueeze(2).repeat([1,1,self.n_nodes,1])
        # concatenated : batch_size x from_nodes x to_nodes x (node_feature * 2 + edge_feature)
        concatenated = torch.cat([nodes_repeated, nodes_repeated.permute(0,2,1,3), edges], dim=-1)
        assert(len(concatenated.size())==4)
        assert(concatenated.size()[1]==self.n_nodes)
        assert(concatenated.size()[2]==self.n_nodes)
        assert(concatenated.size()[3]==self.n_len*2+1)
        return concatenated

    def message_collection_edges(self, edge_influence, topology, context, mask):
        # context = batch_size x context_length
        # edge_influence : batch_size x from_nodes x to_nodes x hidden_influence_dim

        edge_mask = torch.mul(mask.unsqueeze(1), mask.unsqueeze(2)).unsqueeze(-1)
        masked_edge_influence = torch.mul(torch.mul(edge_influence,topology), edge_mask)

        # batch_size x nodes x 1 x hidden_influence_dim
        from_from_influence = (masked_edge_influence).sum(dim=2).unsqueeze(2).repeat([1,1,self.n_nodes,1])
        from_to_influence = (masked_edge_influence).sum(dim=1).unsqueeze(2).repeat([1,1,self.n_nodes,1])
        # batch_size x 1 x nodes x hidden_influence_dim
        to_to_influence = (masked_edge_influence).sum(dim=1).unsqueeze(1).repeat([1,self.n_nodes,1,1])
        to_from_influence = (masked_edge_influence).sum(dim=2).unsqueeze(1).repeat([1,self.n_nodes,1,1])
        
        # all_influences : batch_size x from_nodes x to_nodes x hidden_influence_dim
        all_influences = torch.cat([from_from_influence, from_to_influence, to_to_influence, to_from_influence],dim=-1)
        context_repeated = context.unsqueeze(1).unsqueeze(1).repeat([1,self.n_nodes,self.n_nodes,1])

        # batch_size x from_nodes x to_nodes x self.edges_update_input_dim
        message_to_edge = torch.cat([all_influences,edge_influence,context_repeated],dim=-1)
        
        assert(len(message_to_edge.size())==4)
        assert(message_to_edge.size()[1]==self.n_nodes)
        assert(message_to_edge.size()[2]==self.n_nodes)
        return message_to_edge
