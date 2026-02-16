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
import torch_geometric.nn as geom_nn


class GraphEncoderModule(LightningModule):
    def __init__(self, model_configs):
        
        super().__init__()

        self.cfg = model_configs
        self.embedding_size = self.cfg.c_len

        self.individual_embedding_size = self.embedding_size

        midway = int(round((self.cfg.n_len*3 + self.individual_embedding_size)/2))

        self.embed_node_locations = nn.Sequential(nn.Linear(self.cfg.n_len*3, midway),
                                                  nn.ReLU(),
                                                  nn.Linear(midway, self.individual_embedding_size)
                                                  )

        self.obj_movement_encoder_layer = torch.nn.TransformerEncoderLayer(self.individual_embedding_size, nhead=2, batch_first=True, dropout=0.3)
        self.obj_movement_encoder = torch.nn.TransformerEncoder(self.obj_movement_encoder_layer, num_layers=1)

        self.obj_movement_decoder_layer = torch.nn.TransformerDecoderLayer(self.individual_embedding_size, nhead=2, batch_first=True, dropout=0.3)
        self.obj_movement_decoder = torch.nn.TransformerDecoder(self.obj_movement_decoder_layer, num_layers=3)


    def forward(self, nodes, edges, time_context=None):
        """
        Args:
            nodes: batch_size x sequence_length+1 x num_nodes x node_feature_len
            edges: batch_size x sequence_length+1 x num_nodes x num_nodes
            time_context: batch_size x sequence_length x num_nodes x num_nodes
        Returns:
            latents: batch_size x sequence_length x embedding_size
        """
        batch_size = nodes.size()[0]

        if isinstance(edges, tuple):
            assert len(edges) == 2, "Edges should be a tuple of 2 elements (prev, next) for out-of-sequence evaluation"
            nodes_shortened = nodes.float()
            prev_loc = torch.matmul(edges[0].float(), nodes_shortened)
            new_loc = torch.matmul(edges[1].float(), nodes_shortened)
        else:
            nodes_shortened = nodes[:,1:,:,:].float()
            prev_loc = torch.matmul(edges[:,:-1,:,:].float(), nodes_shortened)
            new_loc = torch.matmul(edges[:,1:,:,:].float(), nodes_shortened)
        sequence_len = nodes_shortened.size()[1]
        nodes_and_locations = torch.cat([nodes_shortened, prev_loc, new_loc], dim=-1)
        assert nodes_and_locations.size()[0] == batch_size, nodes_and_locations.size()
        assert nodes_and_locations.size()[1] == sequence_len, nodes_and_locations.size()
        assert nodes_and_locations.size()[2] == self.cfg.n_nodes, nodes_and_locations.size()
        assert nodes_and_locations.size()[3] == self.cfg.n_len*3, nodes_and_locations.size()

        nodes_and_locations = self.embed_node_locations(nodes_and_locations)
        ## Try embedding node-location pairs per timestep and then passing through the encoder
        nodes_and_locations_latent = self.obj_movement_encoder(nodes_and_locations.view(batch_size*sequence_len, self.cfg.n_nodes, self.individual_embedding_size))
        nodes_and_locations_latent = self.obj_movement_decoder(nodes_and_locations_latent, time_context.view(batch_size*sequence_len, 1, self.individual_embedding_size))
        latent = nodes_and_locations_latent.max(dim=1).values.view(batch_size, sequence_len, self.individual_embedding_size)
        
        assert latent.size()[0] == batch_size
        assert latent.size()[1] == sequence_len
        assert latent.size()[2] == self.individual_embedding_size
        return latent

        
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
