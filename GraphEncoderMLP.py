import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
sys.path.append('helpers')
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
import torch_geometric.nn as geom_nn


class GraphEncoderMLPModule(LightningModule):
    def __init__(self, model_configs):
        
        super().__init__()

        self.cfg = model_configs
        self.embedding_size = self.cfg.c_len

        self.individual_embedding_size = self.embedding_size

        self.graph_cnn = geom_nn.Sequential('x, edge_index',
                                            [(geom_nn.GCNConv(self.cfg.n_len, 300), 'x, edge_index -> x'),
                                            nn.ReLU(inplace=True),
                                            (geom_nn.GCNConv(300, 1000), 'x, edge_index -> x'),
                                            ])
        self.context_from_graph_encodings = nn.Sequential(nn.Linear(1000, 2000),
                                                          nn.ReLU(),
                                                          nn.Linear(2000, 2500)
                                                          )
        self.combine_graph_encodings = nn.Sequential(nn.Linear(5000, 1000),
                                                    nn.ReLU(),
                                                    nn.Linear(1000, 300),
                                                    nn.ReLU(),
                                                    nn.Linear(300, self.individual_embedding_size)
                                                    )

        # self.obj_seq_encoder_transformer_layer = torch.nn.TransformerEncoderLayer(self.embedding_size, nhead=2)
        # self.obj_seq_encoder_transformer = torch.nn.TransformerEncoder(self.obj_seq_encoder_transformer_layer, num_layers=1)


    def forward(self, nodes, edges, time_context=None):
        """
        Args:
            nodes: batch_size x sequence_length+1 x num_nodes x node_feature_len
            edges: batch_size x sequence_length+1 x num_nodes x num_nodes x edge_feature_len
        Returns:
            latents: batch_size x sequence_length x embedding_size
        """
        batch_size = nodes.size()[0]
        sequence_len_plus_one = nodes.size()[1]
        mat2idx = lambda e_mat: (torch.argwhere(e_mat == 1)).transpose(1,0)
        batch = torch.arange(batch_size*sequence_len_plus_one).repeat(self.cfg.n_nodes,1).transpose(1,0).reshape(-1).to(int).to('cuda')
        spatial_edges = torch.cat([mat2idx(edges[b,s,:,:])+(b*(sequence_len_plus_one*self.cfg.n_nodes)+s*self.cfg.n_nodes) for s in range(sequence_len_plus_one) for b in range(batch_size)], dim=-1)
        assert spatial_edges.size()[0] == 2
        # temporal_edges = torch.tensor([[i,self.cfg.n_nodes+i] for i in range((batch_size*sequence_len_plus_one-1)*self.cfg.n_nodes) if i%sequence_len_plus_one!=0], device='cuda').permute(1,0)
        # assert temporal_edges.size()[0] == 2
        # all_edges = torch.cat([spatial_edges], dim=-1)
        # all_edges = torch.cat([spatial_edges, temporal_edges], dim=-1)
        graphs_in = geom_nn.global_mean_pool(self.graph_cnn(nodes.view(batch_size*sequence_len_plus_one*self.cfg.n_nodes, self.cfg.n_len), spatial_edges), batch=batch)
        latent_per_graph = self.context_from_graph_encodings(graphs_in)
        assert latent_per_graph.size()[0] == batch_size*sequence_len_plus_one
        latent_per_graph = latent_per_graph.view(batch_size, sequence_len_plus_one, -1)
        latent = self.combine_graph_encodings(torch.cat([latent_per_graph[:,1:,:], latent_per_graph[:,:-1,:]], dim=-1))
        # latent = (latent_per_graph[:,1:,:] - latent_per_graph[:,:-1,:])*1000
        assert latent.size()[0] == batch_size
        assert latent.size()[1] == sequence_len_plus_one - 1
        assert latent.size()[2] == self.individual_embedding_size
        return latent + time_context

        
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
        return Adam(self.parameters(), lr=1e-3)
