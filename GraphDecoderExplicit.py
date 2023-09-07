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


class GraphDecoderExplicitModule(LightningModule):
    def __init__(self, model_configs):
        
        super().__init__()

        self.cfg = model_configs
        self.n_len = model_configs.n_len
        self.c_len = model_configs.c_len

        self.temperature = nn.Parameter(torch.ones(1)).to('cuda')

        self.cfg = model_configs
        self.embedding_size = self.cfg.c_len

        self.individual_embedding_size = self.embedding_size

        midway = int(round((self.cfg.n_len + self.individual_embedding_size)/2))

        self.mlp_context = nn.Sequential(nn.Linear(self.c_len, self.c_len),
                                             nn.ReLU(),
                                             nn.Linear(self.c_len, self.c_len),
                                             )

        self.embed_node = nn.Sequential(nn.Linear(self.cfg.n_len, midway),
                                                  nn.ReLU(),
                                                  nn.Linear(midway, self.individual_embedding_size)
                                                  )
        self.embed_loc = nn.Sequential(nn.Linear(self.cfg.n_len, midway),
                                                  nn.ReLU(),
                                                  nn.Linear(midway, self.individual_embedding_size)
                                                  )

        midway = int(round(self.individual_embedding_size*1.5))
        self.embed_node_locations = nn.Sequential(nn.Linear(self.individual_embedding_size*2, self.individual_embedding_size),
                                                  nn.ReLU(),
                                                  )
        self.obj_movement_encoder_layer = torch.nn.TransformerEncoderLayer(self.individual_embedding_size, nhead=2, batch_first=True)
        self.obj_movement_encoder = torch.nn.TransformerEncoder(self.obj_movement_encoder_layer, num_layers=2)

        self.obj_movement_decoder_layer = torch.nn.TransformerDecoderLayer(self.individual_embedding_size, nhead=2, batch_first=True)
        self.obj_movement_decoder = torch.nn.TransformerDecoder(self.obj_movement_decoder_layer, num_layers=2)


        self.mlp_predict_edges = nn.Sequential(nn.Linear(self.individual_embedding_size, self.individual_embedding_size),
                                               nn.ReLU(),
                                               nn.Linear(self.individual_embedding_size, self.individual_embedding_size)
                                               )
        self.mlp_predict_activity = nn.Sequential(
                                            #    nn.Linear(self.individual_embedding_size, self.individual_embedding_size),
                                            #    nn.ReLU(),
                                               nn.Linear(self.individual_embedding_size, 1),
                                               nn.Sigmoid()
                                               )
        self.mlp_predict_dynamic = nn.Sequential(
                                            #    nn.Linear(self.individual_embedding_size, self.individual_embedding_size),
                                            #    nn.ReLU(),
                                               nn.Linear(self.individual_embedding_size, 1),
                                               nn.Sigmoid()
                                               )
                                                                                   

        # self.edge_prediction_loss = lambda x,y: (nn.CrossEntropyLoss(reduction='none')(x.squeeze(-1).permute(0,2,1), y.squeeze(-1).long()))
        self.edge_prediction = lambda x: x.squeeze(-1).argmax(-1)


    # def _compute_contrastive_losses(self, node_embeddings, location_embeddings):
    #     logits = (location_embeddings @ node_embeddings.T) / self.temperature
    #     nodes_similarity = node_embeddings @ node_embeddings.T
    #     locations_similarity = location_embeddings @ location_embeddings.T
    #     targets = F.softmax(
    #         (nodes_similarity + locations_similarity) / 2 * self.temperature, dim=-1
    #     )
    #     nodes_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
    #     locations_loss = (-targets * self.log_softmax(logits)).sum(1)
    #     return (nodes_loss + locations_loss) / 2.0

    # def edge_prediction_loss(self, node_embeddings, location_embeddings, y_edges):
    #     logits = (node_embeddings @ location_embeddings.T) / self.temperature
    #     nodes_loss = nn.CrossEntropyLoss(reduction='mean')(logits, y_edges.argmax(-1))
    #     return nodes_loss

    # def edge_prediction_inference(self, node_embeddings, location_embeddings):
    #     logits = (node_embeddings @ location_embeddings.T) / self.temperature
    #     pred_locations = logits.argmax(-1)
    #     return pred_locations


    def forward(self, nodes, edges, context, y_edges=None):
        """
        Args:
            nodes: batch_size * sequence_length x num_nodes x node_feature_len
            edges: batch_size * sequence_length x num_nodes x num_nodes
            context: batch_size * sequence_length x c_len
        Returns:
            pred_edges: batch_size * sequence_length x num_nodes x num_nodes
            nodes: nodes
            imp: None
        """
        sequence_len = nodes.size()[0]
        locations = torch.matmul(edges, nodes)
        # print(f"{movement_mask.sum(-1).reshape(-1)} movements found!!")
        node_embeddings = self.embed_node(nodes)
        target_location_embeddings = self.embed_loc(nodes)
        location_embeddings = self.embed_loc(locations)
        node_and_location_embeddings = self.embed_node_locations(torch.cat([node_embeddings, location_embeddings], dim=-1))
        node_and_location_embeddings = F.normalize(node_and_location_embeddings, dim=-1)
        assert node_embeddings.size()[0] == sequence_len, node_embeddings.size()
        assert node_embeddings.size()[1] == self.cfg.n_nodes, node_embeddings.size()
        assert node_embeddings.size()[2] == self.individual_embedding_size, node_embeddings.size()

        
        # transformed_node_embeddings = self.obj_movement_encoder(torch.cat([node_embeddings, context.unsqueeze(1)], dim=1))
        # transformed_node_embeddings = F.normalize(transformed_node_embeddings[:,:-1,:], dim=-1)
        transformed_node_embeddings = self.obj_movement_encoder(node_and_location_embeddings)
        if context is not None:
            transformed_node_embeddings = self.obj_movement_decoder(transformed_node_embeddings, context.unsqueeze(1))

        ## Predict edges and compare against other nodes
        predicted_location_embeddings = self.mlp_predict_edges(transformed_node_embeddings)
        predicted_edges = (predicted_location_embeddings @ target_location_embeddings.permute(0,2,1)) * torch.exp(self.temperature)

        # predicted_changes = (predicted_location_embeddings @ (node_embeddings.permute(0,2,1) - location_embeddings.permute(0,2,1))) * torch.exp(self.temperature)
        # predicted_edges = predicted_changes + edges
        

        pred_activity = self.mlp_predict_activity(predicted_location_embeddings)
        pred_dynamic = self.mlp_predict_dynamic(predicted_location_embeddings)

        return predicted_edges, pred_activity, pred_dynamic

        
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
