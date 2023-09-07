from argparse import ArgumentError
import os
from copy import deepcopy
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import json
import sys
sys.path.append('helpers')
from random import random
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
from GraphDecoder import GraphDecoderModule
from GraphDecoderExplicit import GraphDecoderExplicitModule
from GraphDecoderTransformerGNN import GraphDecoderTransformerGNNModule
from GraphEncoder import GraphEncoderModule
from GraphEncoderExplicit import GraphEncoderExplicitModule
from GraphEncoderMLP import GraphEncoderMLPModule

GraphEncoder = {
    'explicit': GraphEncoderExplicitModule,
    'mlp': GraphEncoderMLPModule,
    'simple': GraphEncoderModule
}

class ActivityClassifierModule(LightningModule):
    def __init__(self, model_configs, original_model=None):
        
        super().__init__()

        self.cfg = model_configs

        ### Object Encoder ###
        graph_encoder_module = GraphEncoder[self.cfg.encoder_type](model_configs=model_configs)
        self.graph_encoder = graph_encoder_module


        ### Activity Decoder ###
        self.activity_decoder_mlp = nn.Sequential(nn.Linear(self.cfg.c_len, self.cfg.c_len),
                                                nn.ReLU(),
                                                nn.Linear(self.cfg.c_len, self.cfg.n_activities)
                                                )
        if self.cfg.multiple_activities:
            print("So we're doing multiple activities now!!")
            self.activity_prediction_loss = lambda x,y: (nn.MSELoss(reduction='mean')(x.squeeze(-1), y.squeeze(-1)))
            self.activity_inference = lambda x: torch.round(x).long()
            self.activity_accuracy = lambda x,y: (torch.abs(x-y) < 0.5).sum()/torch.numel(y)
        else:
            self.activity_prediction_loss = lambda x,y: (nn.CrossEntropyLoss(reduction='mean')(x.squeeze(-1).permute(0,2,1), y.argmax(-1).long()))
            self.activity_inference = lambda x: F.softmax(x, dim=-1)
            self.activity_accuracy = lambda x,y: (x.argmax(-1) == y.argmax(-1).squeeze(-1)).sum()/torch.numel(y.argmax(-1))



    def encode_graph(self, graph_seq_nodes, graph_seq_edges, graph_dynamic_edges_mask, activity_relevant_edges=None, time_context=None):
        """
        Args:
            graph_seq_nodes          : batch_size x sequence_length+1 x num_nodes x node_feature_len
            graph_seq_edges          : batch_size x sequence_length+1 x num_nodes x num_nodes x edge_feature_len
            graph_dynamic_edges_mask : batch_size x sequence_length+1 x num_nodes x num_nodes x edge_feature_len
        Return:
            graph_latents: batch_size x sequence_length x context_length
            graph_autoenc_loss: [0,1]
            graph_autoenc_accuracy: {'used':[0,1], 'unused':[0,1]}
        """
        if time_context is None:
            print("No time context provided to encode graphs!!")
            time_context = torch.zeros((graph_seq_nodes.size()[0], graph_seq_nodes.size()[1]-1, self.cfg.c_len))
        graph_latents = self.graph_encoder(graph_seq_nodes.float(), graph_seq_edges.float(), time_context=time_context)
        return graph_latents


    def decode_activity(self, latents, ground_truth=None):
        """
        Args:
            latent_vector: batch_size x sequence_length x embedding_size
            ground_truth: batch_size x sequence_length
        Return:
            output_activity: batch_size x sequence_length x n_activities
            activity_pred_loss: batch_size x sequence_length
        """
        
        output_activity = self.activity_decoder_mlp(F.normalize(latents, dim=-1))
        
        activity_pred_loss = None
        activity_pred_acc = None
        if ground_truth is not None:
            activity_pred_loss = self.activity_prediction_loss(output_activity, ground_truth)
            activity_pred_acc = self.activity_accuracy(output_activity, ground_truth)
            
        output_activity = self.activity_inference(output_activity)

        return output_activity, activity_pred_loss, activity_pred_acc


    def forward(self, graph_seq_nodes, graph_seq_edges, graph_dynamic_edges_mask, activity_seq, time_context, graph_seq_dyn_edges=None):
        """
        Args:
            graph_seq_nodes: batch_size x sequence_length+1 x num_nodes x node_feature_len
            graph_seq_edges: batch_size x sequence_length+1 x num_nodes x num_nodes x edge_feature_len
            activity_seq: batch_size x sequence_length
            activity_seq: batch_size x sequence_length x context_length
        """
        if time_context is None:
            print("No time context provided to encode graphs!!")
            time_context = torch.zeros((graph_seq_nodes.size()[0], graph_seq_nodes.size()[1]-1, self.cfg.c_len))
        latents = self.graph_encoder(graph_seq_nodes.float(), graph_seq_edges.float(), time_context=time_context)

        _, act_loss, act_accuracy = self.decode_activity(latents, ground_truth=activity_seq)

        return act_loss, act_accuracy


    def training_step(self, batch, batch_idx):
        loss, accuracy = self(batch['nodes'], batch['edges'], batch['dynamic_edges_mask'], batch['activity'], batch['context_time'])
        self.log('Train loss',loss)
        self.log('Train accuracy',accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self(batch['nodes'], batch['edges'], batch['dynamic_edges_mask'], batch['activity'], batch['context_time'])
        self.log('Test loss',loss)
        self.log('Test accuracy',accuracy)
        return

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

