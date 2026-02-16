from argparse import ArgumentError
import os
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
from pytorch_lightning.core.module import LightningModule
from GraphDecoder import GraphDecoderModule
from GraphEncoder import GraphEncoderModule

EXTRACAREFUL = False

def kl_divergence_analytical(mu1, var1, mu2, var2):
    return 0.5 * (torch.log(var2) - torch.log(var1) + (var1 + (mu1 - mu2)**2) / var2 - 1).sum()

def context_loss(xc, yc):
    temperature = 1
    logits = (yc.view(-1,yc.size()[-1]) @ xc.view(-1,xc.size()[-1]).T) / temperature
    xc_loss = nn.CrossEntropyLoss()(logits, torch.arange(xc.size()[-2]).to('cuda'))
    yc_loss = nn.CrossEntropyLoss()(logits.permute(1,0), torch.arange(xc.size()[-2]).to('cuda'))
    return (xc_loss + yc_loss) / 2.0

class Latent():
    def __init__(self):
        pass

class LatentDeterministic(Latent):
    def __init__(self, data, learn_magnitude):
        super().__init__()
        self.latent = data
        self.learn_magnitude = learn_magnitude
        if not learn_magnitude:
            self.latent = F.normalize(self.latent, dim=-1)
    
    def sample(self):
        return self.latent

    def __getitem__(self, index):
        latent = self.latent[index]
        return LatentDeterministic(latent, self.learn_magnitude)

    def __add__(self, second_latent):
        if isinstance(second_latent, LatentDeterministic):
            return LatentDeterministic(self.latent + second_latent.latent, self.learn_magnitude)
        if isinstance(second_latent, torch.Tensor):
            assert second_latent.size() == self.latent.size(), "latent size must be the same as mu size"
            return LatentDeterministic(self.latent + second_latent, self.learn_magnitude)
        else:
            raise ArgumentError("second_latent for LatentDeterministic must be of type LatentDeterministic or tensor")

    def difference(self, latent_dist2=None, mask=True):
        if not isinstance(mask, bool) and mask.sum() == 0:
            return torch.tensor(0.0).to('cuda')
        
        assert isinstance(latent_dist2, LatentDeterministic)
        v1 = F.normalize(self.latent[mask], dim=-1)
        if latent_dist2 is not None:
            v2 = F.normalize(latent_dist2.latent[mask], dim=-1)
            return context_loss(v1, v2)
        else:
            return v1.norm(dim=-1).mean()

    def append(self, latent2, dim=1):
        assert isinstance(latent2, LatentDeterministic)
        self.latent = torch.cat([self.latent, latent2.unsqueeze(dim)], dim=dim)
        if not self.cfg.learn_latent_magnitude:
            self.latent = F.normalize(self.latent, dim=-1)

    def unsqueeze(self, dim):
        self.latent = self.latent.unsqueeze(dim)
        return self

    def size(self):
        return self.latent.size()
    
    def clone(self):
        return LatentDeterministic(self.latent.clone(), self.learn_magnitude)

class ObjectActivityCoembeddingModule(LightningModule):
    def __init__(self, model_configs, original_model=None):
        
        super().__init__()

        self.cfg = model_configs
        self.embedding_size = self.cfg.c_len

        self.latent_obj = LatentDeterministic

        ### Object AutoEncoder ###
        graph_encoder_module_local = GraphEncoderModule(model_configs=model_configs)
        self.graph_encoder_module = graph_encoder_module_local
        self.graph_encoder = lambda nodes, edges, time_context=None: self.latent_obj(self.graph_encoder_module(nodes, edges, time_context), self.cfg.learn_latent_magnitude)
        # self.graph_encoder = lambda nodes, edges, time_context=None: self.graph_encoder_module(nodes, edges, time_context)

        graph_decoder_module = GraphDecoderModule(model_configs=model_configs)
        self.obj_seq_decoder = graph_decoder_module

        ### Activity AutoEncoder ###
        self.individual_embedding_size = self.embedding_size
        self.embed_context_activity = nn.Linear(self.cfg.act_len, self.individual_embedding_size, bias=False)

        # self.activity_seq_encoder_transformer_layer = torch.nn.TransformerEncoderLayer(self.embedding_size, nhead=2)
        # self.activity_seq_encoder_transformer = torch.nn.TransformerEncoder(self.activity_seq_encoder_transformer_layer, num_layers=1)

        self.activity_decoder_mlp = nn.Sequential(nn.Linear(self.embedding_size, self.embedding_size),
                                                    nn.ReLU(),
                                                    nn.Linear(self.embedding_size, self.cfg.n_activities)
                                                    )

    def activity_prediction_loss(self, x, y):
        if self.cfg.multiple_activities:
            return (nn.MSELoss(reduction='mean')(x.squeeze(-1), y.squeeze(-1)))
        else:
            return ((nn.CrossEntropyLoss(reduction='none')(x.permute(0,2,1), y.long()))[y != 0]).mean()
        
    def activity_inference(self, x):
        if self.cfg.multiple_activities:
            return torch.round(x).long()
        else:
            return F.softmax(x, dim=-1)
                   
    def activity_accuracy(self, x, y):
        if self.cfg.multiple_activities:
            return (torch.abs(x-y) < 0.5).sum()/torch.numel(y)
        else:
            return (x.argmax(-1) == y.squeeze(-1))[y != 0].sum()/(y != 0).sum()
        
    def obj_graph_loss(self, x, y, obj_mask=None):
        if obj_mask is None:
            return nn.CrossEntropyLoss(reduction='mean')(x.permute(0,3,1,2), y.argmax(-1))
        return (nn.CrossEntropyLoss(reduction='none')(x.permute(0,3,1,2), y.argmax(-1))[obj_mask]).mean()
              
    def obj_graph_loss_conf_matrix(self, x, y, ref, obj_mask=None):
        # let softmax be the Heaviside function [Tsoi et al. 2022]
        agreement_prob = x * y
        disagreement_prob = x * (1-y)
        change_gt = y.detach().argmax(-1) != ref.detach().argmax(-1)
        if obj_mask is not None:
            tp_soft = (agreement_prob)[torch.bitwise_and(change_gt, obj_mask.detach())].sum()
            fn_soft = (disagreement_prob)[torch.bitwise_and(change_gt, obj_mask.detach())].sum()
            fp_soft = (disagreement_prob)[torch.bitwise_and(torch.bitwise_not(change_gt), obj_mask.detach())].sum()
        else:
            tp_soft = (agreement_prob)[change_gt]
            fn_soft = (disagreement_prob)[change_gt]
            fp_soft = (disagreement_prob)[torch.bitwise_not(change_gt)]
        prec = tp_soft / (tp_soft + fp_soft)
        recl = tp_soft / (tp_soft + fn_soft)
        f1 = 2 * prec * recl / (prec + recl)
        return -f1

    def activity_encoder(self, activity):
        """
        Args:
            activity: batch_size x sequence_length x n_activities
        Return:
            _: batch_size x sequence_length x n_activities
        """
        return self.latent_obj(self.embed_context_activity(activity.float()), self.cfg.learn_latent_magnitude)
        # return self.embed_context_activity(activity.float())


    def latent_loss(self, latent_obj, latent_act, mask=True, allow_regularization=True):
        """
        Args:
            latent_obj: batch_size x sequence_len x embedding_size
            latent_act: batch_size x sequence_len x embedding_size
            mask: batch_size x sequence_len
        """

        
        latent_loss = latent_obj.difference(latent_act, mask=mask)
        if allow_regularization and self.cfg.latent_regularization is not None:
            latent_loss += self.cfg.latent_regularization * latent_obj.difference(mask=mask)
            latent_loss += self.cfg.latent_regularization * latent_act.difference(mask=mask)       

        return latent_loss

    def autoencode_graph(self, graph_seq_nodes, graph_seq_edges, graph_dynamic_edges_mask, time_context=None):
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

        latent_in = graph_latents + time_context if self.cfg.addtnl_time_context else graph_latents
        _, graph_autoenc_loss, graph_autoenc_accuracy = self.decode_graph(latent_in, 
                                                                          graph_seq_nodes[:,:-1,:,:], 
                                                                          graph_seq_edges[:,:-1,:,:], 
                                                                          graph_dynamic_edges_mask[:,:-1,:,:], 
                                                                          output_edges=graph_seq_edges[:,1:,:,:])

        assert graph_latents.size()[0] == graph_seq_nodes.size()[0] , 'Size Mismatch'
        assert graph_latents.size()[1] == graph_seq_nodes.size()[1] - 1, 'Size Mismatch'
        assert graph_latents.size()[2] == self.cfg.c_len, 'Size Mismatch'

        return graph_latents, graph_autoenc_loss, graph_autoenc_accuracy

    def decode_graph(self, latents, input_nodes, input_edges, dynamic_edges_mask, output_edges=None, activity_relevant_edges=None, activity_mask=None):
        if isinstance(latents, Latent): latents = latents.sample()
        if self.cfg.learn_latent_magnitude:
            latents = F.normalize(latents, dim=-1)
        batch_size, sequence_len, _, _ = input_edges.size()

        pred_edges, pred_activity, pred_dynamic = self.obj_seq_decoder(input_edges.view(batch_size*(sequence_len), self.cfg.n_nodes, self.cfg.n_nodes), 
                                                input_nodes.view(batch_size*(sequence_len), self.cfg.n_nodes, self.cfg.n_len), 
                                                latents.view(batch_size*(sequence_len), self.embedding_size))
        logits = pred_edges.view(batch_size, sequence_len, self.cfg.n_nodes, self.cfg.n_nodes)
        pred_activity = pred_activity.view(batch_size, sequence_len, self.cfg.n_nodes, 1)
        pred_dynamic = pred_dynamic.view(batch_size, sequence_len, self.cfg.n_nodes, 1)
        
        assert (dynamic_edges_mask.to(bool) == (dynamic_edges_mask==1).to(bool)).all(), f"Conversion check {dynamic_edges_mask.to(bool)} {dynamic_edges_mask}"
        pred_edges = F.softmax(logits, dim=-1)
        pred_edges = self.cfg.movement_inertia * input_edges + \
                    (1 - self.cfg.movement_inertia) * pred_edges
        assert not EXTRACAREFUL or torch.allclose(pred_edges.sum(-1), torch.tensor([1.0]).to(pred_edges.device), atol=0.1), f"Sum of edges is not 1.0 {(pred_edges.sum(-1)-1).max()} to {(pred_edges.sum(-1)-1).min()}"
        pred_edges += 1e-8
        pred_edges = pred_edges.masked_fill(dynamic_edges_mask == 0, float(0.0))
        ## Fill in all non-dynamic objects as input edges
        obj_mask = (dynamic_edges_mask.sum(-1) > 0)
        assert not EXTRACAREFUL or torch.allclose(input_edges.sum(-1), torch.tensor([1.0]).to(input_edges.device), atol=0.1), f"Sum of edges is not 1.0 {(input_edges.sum(-1)-1).max()} to {(input_edges.sum(-1)-1).min()}"
        pred_edges[torch.bitwise_not(obj_mask)] = input_edges[torch.bitwise_not(obj_mask)]
        ## Normalize for zeroed out self-edges
        normalizer = pred_edges.sum(-1).unsqueeze(-1)
        pred_edges = pred_edges / normalizer
        assert not EXTRACAREFUL or torch.allclose(pred_edges.sum(-1), torch.tensor([1.0]).to(pred_edges.device), atol=0.1), f"Sum of edges is not 1.0 {(pred_edges.sum(-1)-1).max()} to {(pred_edges.sum(-1)-1).min()}"

        graph_pred_loss = None
        graph_pred_accuracy = {'used':0, 'unused':0}
        self.auxiliary_accuracy = {'activity_recall':0, 'activity_prec':0, 'dynamic_recall':0, 'dynamic_prec':0}
        if output_edges is not None:
            if activity_mask is not None:
                assert activity_mask.size()[0] == obj_mask.size()[0], 'Mismatch in 0th dim '+str(activity_mask.size()[0]) + ' vs '+str(obj_mask.size()[0])
                assert activity_mask.size()[1] == obj_mask.size()[1], 'Mismatch in 1st dim'+str(activity_mask.size()[1]) + ' vs '+str(obj_mask.size()[1])
                obj_mask = torch.bitwise_or(obj_mask, activity_mask.unsqueeze(-1))
            graph_pred_loss = self.obj_graph_loss(logits, output_edges, obj_mask=obj_mask)

            auxiliary_loss_activity = torch.Tensor([0.]).to('cuda')
            if activity_relevant_edges is not None:
                pred_pos = pred_activity.squeeze(-1) > 0.5
                if activity_mask is not None:
                    activity_mask = torch.bitwise_or((activity_relevant_edges > -0.5).max(-1).values, activity_mask)
                else:
                    activity_mask = (activity_relevant_edges > -0.5).max(-1).values
                act_pos = activity_relevant_edges.to(int)
                auxiliary_loss_activity = nn.BCELoss(reduction='mean')(pred_activity.squeeze(-1)[activity_mask], activity_relevant_edges[activity_mask].float())
                self.auxiliary_accuracy['activity_recall'] = torch.bitwise_and(pred_pos[activity_mask], act_pos[activity_mask]).sum()/act_pos[activity_mask].sum()
                self.auxiliary_accuracy['activity_prec'] = torch.bitwise_and(pred_pos[activity_mask], act_pos[activity_mask]).sum()/pred_pos[activity_mask].sum()
                self.auxiliary_accuracy['activity_loss'] = auxiliary_loss_activity
                self.auxiliary_accuracy['activity_accuracy'] = (pred_pos[activity_mask] == act_pos[activity_mask]).sum()/torch.numel(pred_pos[activity_mask])
                self.auxiliary_accuracy['activity_num'] = act_pos[activity_mask].sum()
                self.auxiliary_accuracy['activity_pred_num'] = pred_pos[activity_mask].sum()
            auxiliary_loss_dynamic = nn.BCELoss(reduction='mean')(pred_dynamic.squeeze(-1), dynamic_edges_mask.max(-1).values)
            pred_dyn = pred_dynamic.squeeze(-1) > 0.5
            self.auxiliary_accuracy['dynamic_recall'] = torch.bitwise_and(pred_dyn, dynamic_edges_mask.to(int).max(-1).values).sum()/(dynamic_edges_mask.to(int).max(-1).values).sum()
            self.auxiliary_accuracy['dynamic_prec'] = torch.bitwise_and(pred_dyn, dynamic_edges_mask.to(int).max(-1).values).sum()/(pred_dyn).sum()
            self.auxiliary_accuracy['dynamic_accuracy'] = (pred_dyn == dynamic_edges_mask.to(int).max(-1).values).sum()/torch.numel(pred_dyn)
            self.auxiliary_accuracy['dynamic_loss'] = auxiliary_loss_dynamic
            self.auxiliary_accuracy['dynamic_num'] = (dynamic_edges_mask.to(int).max(-1).values).sum()
            self.auxiliary_accuracy['dynamic_pred_num'] = (pred_dyn).sum()

            graph_pred_loss = graph_pred_loss + self.cfg.aux_loss_weight * (auxiliary_loss_activity + auxiliary_loss_dynamic)

        if output_edges is not None:
            used_mask = torch.bitwise_and((output_edges.argmax(-1) != input_edges.argmax(-1)), obj_mask)
            graph_pred_accuracy['used'] = (pred_edges.argmax(-1) == output_edges.argmax(-1))[used_mask].sum()/(used_mask.sum()+1e-8)
            unused_mask =  torch.bitwise_and((output_edges.argmax(-1) == input_edges.argmax(-1)), obj_mask)
            graph_pred_accuracy['unused'] = (pred_edges.argmax(-1) == output_edges.argmax(-1))[unused_mask].sum()/(unused_mask.sum()+1e-8)

        return pred_edges, graph_pred_loss, graph_pred_accuracy

    def autoencode_activity(self, activity_seq, time_context=None, activity_gt=None):
        activity_latents = self.activity_encoder(activity_seq)
        latent_in = activity_latents + time_context if self.cfg.addtnl_time_context else activity_latents
        _, activity_autoenc_loss, activity_autoenc_accuracy = self.decode_activity(latent_in, ground_truth=activity_gt)

        return activity_latents, activity_autoenc_loss, activity_autoenc_accuracy

    def decode_activity(self, latents, ground_truth=None):
        """
        Args:
            latent_vector: batch_size x sequence_length x embedding_size
            ground_truth: batch_size x sequence_length
        Return:
            output_activity: batch_size x sequence_length x n_activities
            activity_pred_loss: batch_size x sequence_length
        """
   
        if isinstance(latents, Latent): latents = latents.sample()
        if self.cfg.learn_latent_magnitude:
            latents = F.normalize(latents, dim=-1)
        output_activity = self.activity_decoder_mlp(latents)
        
        activity_pred_loss = None
        activity_pred_acc = None
        if ground_truth is not None:
            activity_pred_loss = self.activity_prediction_loss(output_activity, ground_truth)
            activity_pred_acc = self.activity_accuracy(output_activity, ground_truth)
            
        output_activity = self.activity_inference(output_activity)

        return output_activity, activity_pred_loss, activity_pred_acc


    def forward(self, graph_seq_nodes, graph_seq_edges, graph_dynamic_edges_mask, activity_seq, graph_seq_dyn_edges=None):
        """
        Args:
            graph_seq_nodes: batch_size x sequence_length+1 x num_nodes x node_feature_len
            graph_seq_edges: batch_size x sequence_length+1 x num_nodes x num_nodes x edge_feature_len
            activity_seq: batch_size x sequence_length
            activity_seq: batch_size x sequence_length x context_length
        """


        batch_size, num_nodes, node_feature_len = graph_seq_nodes.size()
        batch_size_e, num_f_nodes, num_t_nodes = graph_seq_edges.size()
        batch_size_act, num_act = activity_seq.size()

        self.cfg.n_nodes = num_nodes
        
        # Sanity check input dimensions
        assert batch_size == batch_size_e, "Different edge and node batch sizes"
        assert batch_size == batch_size_act, "Different edge and node batch sizes"
        assert self.cfg.n_len == node_feature_len, (str(self.cfg.n_len) +'!='+ str(node_feature_len))
        assert self.cfg.n_nodes == num_f_nodes, (str(self.cfg.n_nodes) +'!='+ str(num_f_nodes))
        assert self.cfg.n_nodes == num_t_nodes, (str(self.cfg.n_nodes) +'!='+ str(num_t_nodes))
        assert self.cfg.n_activities == num_act, (str(self.cfg.n_activities) +'!='+ str(num_act))

        graph_latents, graph_autoenc_loss, accuracy_object_autoenc = self.autoencode_graph(graph_seq_nodes.unsqueeze(0), graph_seq_edges.unsqueeze(0), graph_dynamic_edges_mask.unsqueeze(0))

        graph_latents = graph_latents.squeeze(0)

        activity_latents, activity_autoenc_loss, accuracy_activity_autoenc = self.autoencode_activity(activity_seq)

        _, cross_graph_pred_loss, cross_accuracy_object = self.decode_graph(latents=activity_latents, 
                                                                    input_nodes=graph_seq_nodes[:,:,:].unsqueeze(0),
                                                                    input_edges=graph_seq_edges[:,:,:].unsqueeze(0),
                                                                    dynamic_edges_mask=graph_dynamic_edges_mask[:,:,:].unsqueeze(0),
                                                                    output_edges=graph_seq_edges[1:,:,:].unsqueeze(0))

        _, cross_activity_pred_loss, cross_accuracy_activity = self.decode_activity(latents=graph_latents, 
                                                                                ground_truth=activity_seq)
                                                                                        
        latent_similarity_loss = self.latent_loss(graph_latents, activity_latents)


        results = {
            'loss' : {
                      'object_autoencoder': graph_autoenc_loss,
                      'object_cross_pred': cross_graph_pred_loss,
                      'activity_autoencoder': activity_autoenc_loss,
                      'activity_cross_pred': cross_activity_pred_loss,
                      'latent_similarity': latent_similarity_loss,
                      },
            'accuracies' : {
                        'object_autoenc_used': accuracy_object_autoenc['used'],
                        'object_autoenc_unused': accuracy_object_autoenc['unused'],
                        'activity_autoenc': accuracy_activity_autoenc,
                        'object_cross_used': cross_accuracy_object['used'],
                        'object_cross_unused': cross_accuracy_object['unused'],
                        'activity_cross': cross_accuracy_activity,
                        }
        }

        return results

        
    def training_step(self, batch, batch_idx):
        results = self(batch['nodes'], batch['edges'], batch['dynamic_edges_mask'], batch['activity'])
        self.log('Train loss',results['loss'])
        self.log('Train accuracy',results['accuracies'])
        res = 0
        res += results['loss']['object_autoencoder']
        res += results['loss']['activity_autoencoder']
        if self.cfg.train_latent_similarity:  
            res += results['loss']['latent_similarity']
        else:
            res += results['loss']['object_cross_pred']
            res += results['loss']['activity_cross_pred']
        return res

    def test_step(self, batch, batch_idx):
        results = self(batch['nodes'], batch['edges'], batch['dynamic_edges_mask'], batch['activity'])
        self.log('Test loss',results['loss'])
        self.log('Test accuracy',results['accuracies'])
        return 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

