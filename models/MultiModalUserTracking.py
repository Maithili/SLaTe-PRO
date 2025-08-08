from argparse import ArgumentError
import os
from copy import deepcopy
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import json
import sys
sys.path.append('helpers')
import random
import math
import numpy as np
from adict import adict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from pytorch_lightning.core.module import LightningModule
from ObjectActivityCoembedding import ObjectActivityCoembeddingModule, Latent, LatentDeterministic, EXTRACAREFUL
from utils import get_metrics

random.seed(23435)
np.random.seed(23435)


def wrap_str(label, l=8):
    wrapped = ''
    if label is None: return 'None'
    if len(label) < l: return label
    for i in range(math.floor(len(label)/l)):
        wrapped += label[l*i:l*(i+1)] + '\n'
    wrapped += label[l*(i+1):]
    return wrapped

class MultiModalUserTrackingModule(LightningModule):
    def __init__(self, model_configs, original_model=False):
        
        super().__init__()

        self.original_model = original_model

        self.cfg = model_configs
        self.embedding_size = self.cfg.c_len
        self.individual_embedding_size = self.embedding_size

        self.object_activity_coembedding_module = ObjectActivityCoembeddingModule(model_configs=model_configs)

        ### Prediction Model ###
        if self.cfg.latent_predictor_type.lower() == 'lstm':
            self.prediction_lstm = torch.nn.LSTM(input_size=self.embedding_size, hidden_size=self.embedding_size, batch_first=True)
        elif self.cfg.latent_predictor_type.lower() == 'transformer':
            self.prediction_transformer_layer = torch.nn.TransformerEncoderLayer(self.embedding_size, nhead=2, batch_first=True)
            self.prediction_transformer = torch.nn.TransformerEncoder(self.prediction_transformer_layer, num_layers=1)

        self.reset_validation()
        self.test_forward=True
        self.object_usage_frequency=None
        

    def reset_validation(self):

        self.num_test_batches = 0
        
        self.results = {
                'reference_locations': torch.tensor([]).to('cuda'),
                'confusion_matrix':{
                    'tp':[0 for _ in range(self.cfg.lookahead_steps)],
                    'fp':[0 for _ in range(self.cfg.lookahead_steps)],
                    'fn':[0 for _ in range(self.cfg.lookahead_steps)],
                    'tn':[0 for _ in range(self.cfg.lookahead_steps)],
                },
                'confusion_matrix_lenient':{
                    'tp_precision':[0 for _ in range(self.cfg.lookahead_steps)],
                    'tp_recall':[0 for _ in range(self.cfg.lookahead_steps)],
                    'fp':[0 for _ in range(self.cfg.lookahead_steps)],
                    'fn':[0 for _ in range(self.cfg.lookahead_steps)],
                },
                'activity':{
                    'differences':[0 for _ in range(self.cfg.lookahead_steps)], 
                },
        }

        self.snapshots = []
        self.snapshots_queries = {}
        self.snapshots_data = []

    def set_object_consistency(self, consistency):
        self.object_usage_frequency = consistency
        self.object_usage_frequency = self.object_usage_frequency.to('cuda')

    def combine_latents(self, pred=None, enc_graph=None, enc_activity=None, time_context=None):
        if pred is not None:
            latent_size = pred.size()
        elif enc_graph is not None:
            latent_size = enc_graph.size()
        elif enc_activity is not None:
            latent_size = enc_activity.size()
        elif time_context is not None:
            latent_size = time_context.size()
        else:
            latent_size = None
                                                        
        assert latent_size is not None, "Combining latents needs at least one input"
        latent = torch.zeros(latent_size).to('cuda')
        if enc_graph is not None:
            latent += enc_graph
        if enc_activity is not None:
            latent += enc_activity
        latent = F.normalize(latent, dim=-1)
        if pred is not None:
            latent *= self.cfg.latent_evidence_weight
            latent += (1-self.cfg.latent_evidence_weight) * pred
        if time_context is not None:
            latent += time_context
        latent = F.normalize(latent, dim=-1)
        return latent


    def seq_encoder(self, latents, time_context, seq_type):
        """
        Args:
            latents: batch_size x sequence_length x embedding_size
        Return:
            encoded: batch_size x sequence_length x embedding_size
        """
        if self.cfg.learn_latent_magnitude:
            latents = F.normalize(latents, dim=-1)
        if seq_type == 'object':
            raise NotImplementedError('Object sequence encoder is not fully implemented!')
        elif seq_type == 'activity':
            raise NotImplementedError('Activity sequence encoder is not fully implemented!')
        elif seq_type == 'predictive':
            batch_size, sequence_length, _ = latents.size()
            if self.cfg.latent_predictor_type.lower() == 'lstm':
                if latents.size()[1] < 3: return latents + time_context
                model = self.prediction_lstm
                encoded, _ = model(latents + time_context)
            elif self.cfg.latent_predictor_type.lower() == 'transformer':
                model = self.prediction_transformer
                mask = (torch.triu(torch.ones(sequence_length, sequence_length)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda')
                encoded = model(latents + time_context, mask=mask)
            else:
                raise ArgumentError(f'Encoder type must be lstm or transformer, not {self.cfg.latent_predictor_type}')
            encoded_latent = LatentDeterministic(encoded, self.cfg.learn_latent_magnitude)
        else:
            raise ArgumentError(f'Sequence type must be object or activity, not {seq_type}')

        return encoded_latent


    def predict(self, latents, time_context, latents_expected=None):
        if isinstance(latents, Latent): latents = latents.sample()

        pred_latents = self.seq_encoder(latents, time_context, seq_type='predictive')
        
        latent_predictive_loss = torch.Tensor([0.]).to('cuda')
        if latents_expected is not None:
            latent_predictive_loss = self.object_activity_coembedding_module.latent_loss(pred_latents, latents_expected, allow_regularization=False)
        
        return pred_latents, latent_predictive_loss


    def forward(self, batch):
        """
        Args:
            graph_seq_nodes          : batch_size x sequence_length+1 x num_nodes x node_feature_len
            graph_seq_edges          : batch_size x sequence_length+1 x num_nodes x num_nodes x edge_feature_len
            graph_dynamic_edges_mask : batch_size x sequence_length+1 x num_nodes x num_nodes x edge_feature_len
            activity_seq             : batch_size x sequence_length+1 x num_activities(+None)
            time_context             : batch_size x sequence_length+1 x context_length
        """
        graph_seq_nodes = batch['node_features'].float()
        graph_seq_edges = batch['edges'].float()
        assert not EXTRACAREFUL or torch.allclose(graph_seq_edges.sum(-1), torch.ones_like(graph_seq_edges.sum(-1)), atol=0.1), "Edges are not normalized!"
        graph_dynamic_edges_mask = batch['dynamic_edges_mask']
        activity_seq = batch['activity_features'][:,:-1,:]
        activity_id_seq = batch['activity_ids'][:,:-1]
        latent_mask = batch['activity_mask_drop'][:,:-1]
        
        time_context = batch.get('time_features', torch.zeros((batch['edges'].size()[0],batch['edges'].size()[1], batch['edges'].size()[2], self.cfg.c_len)))
        time_context = time_context[:,1:,:]
        assert time_context.size()[2] <= self.cfg.c_len, f"Config of size {self.cfg.c_len} does not fit time context of size {time_context.size()[2]}"
        if time_context.size()[-1] > self.cfg.c_len:
            time_context = time_context[:,:,:self.cfg.c_len]
            print("Warning: time context is too long. Truncating it to the first {} values".format(self.cfg.c_len))
            print()
            print("THIS IS ONLY OKAY FOR DEBUGGING!!!!!! YOU ARE NOT TRAINING A MODEL THAT WILL WORK IN REALITY!!!!! THE TIME CONTEXT IS TOO SHORT!!!!!!")
            print()
        time_context = torch.cat([time_context, torch.zeros((time_context.size()[0], time_context.size()[1], self.cfg.c_len - time_context.size()[2])).to('cuda')], dim=-1)
        

        batch_size, sequence_len_plus_one, num_nodes, node_feature_len = graph_seq_nodes.size()
        batch_size_e, sequence_len_plus_one_e, num_f_nodes, num_t_nodes = graph_seq_edges.size()
        batch_size_act, sequence_len, num_act = activity_seq.size()
        
        # Sanity check input dimensions
        assert batch_size == batch_size_e, "Different edge and node batch sizes"
        assert batch_size == batch_size_act, "Different edge and node batch sizes"
        assert sequence_len_plus_one == sequence_len_plus_one_e, "Different edge and node sequence lengths"
        assert sequence_len_plus_one == sequence_len + 1, "Different graph and activity sequence lengths"
        assert self.cfg.n_len == node_feature_len, (str(self.cfg.n_len) +'!='+ str(node_feature_len))
        self.cfg.n_nodes = num_nodes
        assert self.cfg.n_nodes == num_f_nodes, (str(self.cfg.n_nodes) +'!='+ str(num_f_nodes))
        assert self.cfg.n_nodes == num_t_nodes, (str(self.cfg.n_nodes) +'!='+ str(num_t_nodes))

        # Results Initialization for when not populated
        graph_autoenc_loss = torch.Tensor([0.]).to('cuda')
        activity_autoenc_loss = torch.Tensor([0.]).to('cuda')
        latent_similarity_loss = torch.Tensor([0.]).to('cuda')
        latent_predictive_loss = torch.Tensor([0.]).to('cuda')
        latent_predictive_loss_overshoot = torch.Tensor([0.]).to('cuda')
        cross_graph_pred_loss = torch.Tensor([0.]).to('cuda')
        graph_pred_loss_overshoot = torch.Tensor([0.]).to('cuda')
        activity_pred_loss_overshoot = torch.Tensor([0.]).to('cuda')
        cross_accuracy_object = {'used':0, 'unused':0}
        cross_activity_pred_loss = torch.Tensor([0.]).to('cuda')
        cross_accuracy_activity = torch.Tensor([0.]).to('cuda')
        combined_graph_pred_loss = torch.Tensor([0.]).to('cuda')
        combined_activity_pred_loss = torch.Tensor([0.]).to('cuda')
        combined_accuracy_activity = torch.Tensor([0.]).to('cuda')
        graph_pred_loss = torch.Tensor([0.]).to('cuda')
        activity_pred_loss = torch.Tensor([0.]).to('cuda')
        accuracy_object = {'used':0, 'unused':0}
        accuracy_activity = torch.Tensor([0.]).to('cuda')
        latent_magn = {'object' : 0,
                        'activity' : 0,
                        'object_masked' : 0,
                        'activity_masked': 0,
                        'object_wrt_activity': 0}
        

        # Input Graphs for forward predictions
        input_nodes_forward = graph_seq_nodes[:,1:-1,:,:]
        input_edges_forward = graph_seq_edges[:,1:-1,:,:]
        output_edges_forward = graph_seq_edges[:,2:,:,:]

        if not self.original_model:

            # Encode graphs and activities
            graph_latents, graph_autoenc_loss, _ = self.object_activity_coembedding_module.autoencode_graph(graph_seq_nodes, graph_seq_edges, graph_dynamic_edges_mask, time_context=time_context)

            activity_latents, activity_autoenc_loss, accuracy_activity_autoenc = self.object_activity_coembedding_module.autoencode_activity(activity_seq, activity_gt=activity_id_seq, time_context=time_context)

            # Latent training
            latent_in = activity_latents + time_context if self.cfg.addtnl_time_context else activity_latents
            _, cross_graph_pred_loss, cross_accuracy_object = self.object_activity_coembedding_module.decode_graph(
                                                                        latents=latent_in, 
                                                                        input_nodes=graph_seq_nodes[:,:-1,:,:],
                                                                        input_edges=graph_seq_edges[:,:-1,:,:],
                                                                        dynamic_edges_mask=graph_dynamic_edges_mask[:,:-1,:,:],
                                                                        output_edges=graph_seq_edges[:,1:,:,:],
                                                                        )

            latent_in = graph_latents + time_context if self.cfg.addtnl_time_context else graph_latents
            _, cross_activity_pred_loss, cross_accuracy_activity = self.object_activity_coembedding_module.decode_activity(
                                                                                    latents=latent_in, 
                                                                                    ground_truth=activity_id_seq)

            latent_similarity_loss = self.object_activity_coembedding_module.latent_loss(graph_latents, activity_latents) #, mask=latent_mask)## MHC change: activity mask is not a thing
            
            latents =  (graph_latents + activity_latents)

            latent_in = latents + time_context if self.cfg.addtnl_time_context else latents
            _, combined_graph_pred_loss, combined_accuracy_object = self.object_activity_coembedding_module.decode_graph(
                                                                        latents=latent_in, 
                                                                        input_nodes=graph_seq_nodes[:,:-1,:,:],
                                                                        input_edges=graph_seq_edges[:,:-1,:,:],
                                                                        dynamic_edges_mask=graph_dynamic_edges_mask[:,:-1,:,:],
                                                                        output_edges=graph_seq_edges[:,1:,:,:],
                                                                        )
            _, combined_activity_pred_loss, combined_accuracy_activity = self.object_activity_coembedding_module.decode_activity(
                                                                                    latents=latents, 
                                                                                    ground_truth=activity_id_seq)

            # Latent space prediction
            pred_latents, latent_predictive_loss = self.predict(latents[:,:-1,:], 
                                                                time_context[:,:-1,:], 
                                                                latents_expected=latents[:,1:,:])
            next_pred_latents = pred_latents
            if self.cfg.learn_latent_magnitude:
                next_pred_latents_with_correction = pred_latents.clone()

            # Decoding graphs and activities
            latent_in = pred_latents + time_context[:,1:,:] if self.cfg.addtnl_time_context else pred_latents
            pred_edges, graph_pred_loss, accuracy_object = self.object_activity_coembedding_module.decode_graph(
                                                                            latents=latent_in, 
                                                                            input_nodes=input_nodes_forward,
                                                                            input_edges=input_edges_forward,
                                                                            dynamic_edges_mask=graph_dynamic_edges_mask[:,1:-1,:,:],
                                                                            output_edges=output_edges_forward,
                                                                            )

            pred_activity, activity_pred_loss, accuracy_activity = self.object_activity_coembedding_module.decode_activity(
                                                                                        latents=latent_in, 
                                                                                        ground_truth=activity_id_seq[:,1:])


        else:
            pred_edges, graph_pred_loss, accuracy_object = self.object_activity_coembedding_module.decode_graph(
                                                                            latents=time_context[:,1:], 
                                                                            input_nodes=input_nodes_forward,
                                                                            input_edges=input_edges_forward,
                                                                            dynamic_edges_mask=graph_dynamic_edges_mask[:,1:-1,:,:],
                                                                            output_edges=output_edges_forward)
            pred_activity, activity_pred_loss, accuracy_activity = self.object_activity_coembedding_module.decode_activity(
                                                                                        latents=time_context[:,1:], 
                                                                                        ground_truth=activity_id_seq[:,1:])
        
        # Overshoot training
        weighing_factor = 1.0
        assert self.cfg.latent_overshoot >= self.cfg.prediction_overshoot, 'Latent overshoot must be greater or equal to prediction overshoot'
        for i in range(self.cfg.latent_overshoot):
            if sequence_len <= 2+i:
                break
            
            if not self.original_model:
                ## Training for multi-step rollout
                next_pred_latents, additional_latent_predictive_loss = self.predict(next_pred_latents[:,:-1], 
                                                                    time_context[:,i+1:-1], 
                                                                    latents_expected=latents[:,i+2:])
                latent_predictive_loss_overshoot += weighing_factor * additional_latent_predictive_loss

                ## Training for belief propagation; teaches the predictor to manage its magnitude
                if self.cfg.learn_latent_magnitude:
                    next_pred_latents_with_correction, additional_latent_predictive_loss = self.predict(next_pred_latents_with_correction[:,:-1] + latents[:,i+1:-1],
                                                                        time_context[:,i+1:-1],
                                                                        latents_expected=latents[:,i+2:])
                    latent_predictive_loss_overshoot += weighing_factor * additional_latent_predictive_loss

            else:
                next_pred_latents = LatentDeterministic(time_context[:,i+2:], self.cfg.learn_latent_magnitude)

            if i < self.cfg.prediction_overshoot:
                latent_in = next_pred_latents + time_context[:,i+2:] if self.cfg.addtnl_time_context else next_pred_latents
                pred_edges_mixed, additional_graph_pred_loss, additional_accuracy_object = self.object_activity_coembedding_module.decode_graph(
                                                                                                    latents=latent_in, 
                                                                                                    input_nodes=input_nodes_forward[:,1+i:,:,:],
                                                                                                    input_edges=pred_edges[:,:-1,:,:],
                                                                                                    dynamic_edges_mask=graph_dynamic_edges_mask[:,2+i:-1,:,:],
                                                                                                    output_edges= output_edges_forward[:,1+i:,:,:],
                                                                                                    )

                pred_activity, additional_activity_pred_loss, additional_accuracy_activity = self.object_activity_coembedding_module.decode_activity(
                                                                                            latents=latent_in, 
                                                                                            ground_truth=activity_id_seq[:,2+i:])

                pred_edges[graph_dynamic_edges_mask[:,2+i:-1,:,:]] = pred_edges_mixed[graph_dynamic_edges_mask[:,2+i:-1,:,:]]

                graph_pred_loss_overshoot += weighing_factor * additional_graph_pred_loss
                activity_pred_loss_overshoot += weighing_factor * additional_activity_pred_loss
            
            weighing_factor *= 0.9


        results = {
            'loss' : {
                      'object_autoencoder': graph_autoenc_loss,
                      'object_pred': graph_pred_loss,
                      'object_pred_oversht': graph_pred_loss_overshoot,
                      'object_cross_pred': cross_graph_pred_loss,
                      'object_combined_pred': combined_graph_pred_loss,
                      'activity_autoencoder': activity_autoenc_loss,
                      'activity_pred': activity_pred_loss,
                      'activity_pred_oversht': activity_pred_loss_overshoot,
                      'activity_cross_pred': cross_activity_pred_loss,
                      'activity_combined_pred': combined_activity_pred_loss,
                      'latent_similarity': latent_similarity_loss,
                      'latent_pred': latent_predictive_loss,
                      'latent_pred_oversht': latent_predictive_loss_overshoot
                      },
            'accuracies' : {
                        'object': accuracy_object,
                        'activity': accuracy_activity,
            },
            'latents' : latent_magn
        }
        
        return results


    def evaluate_prediction(self, batch, num_steps=1):

        graph_seq_nodes = batch.get('node_features').float()
        graph_seq_edges = batch.get('edges')
        assert not EXTRACAREFUL or torch.allclose(graph_seq_edges.sum(-1), torch.ones_like(graph_seq_edges.sum(-1)), atol=0.1), "Edges are not normalized!"
        graph_dyn_edges = batch.get('dynamic_edges_mask')
        activity_seq = batch.get('activity_features')[:,:-1,:]
        activity_id_seq = batch['activity_ids'][:,:-1]

        time_context = batch.get('time_features', torch.zeros((batch['edges'].size()[0],batch['edges'].size()[1], self.cfg.c_len)))[:,1:,:]
        if time_context.size()[-1] > self.cfg.c_len:
            time_context = time_context[:,:,:self.cfg.c_len]
            print("Warning: time context is too long. Truncating it to the first {} values".format(self.cfg.c_len))
            print()
            print("THIS IS ONLY OKAY FOR DEBUGGING!!!!!! YOU ARE NOT TRAINING A MODEL THAT WILL WORK IN REALITY!!!!!")
            print()
        if time_context.size()[-1] < self.cfg.c_len:
            time_context = torch.cat([time_context, torch.zeros((time_context.size()[0], time_context.size()[1], self.cfg.c_len - time_context.size()[2])).to('cuda')], dim=-1)

        batch_size_act, sequence_len, num_act = activity_seq.size()

        if sequence_len < num_steps+2 : num_steps = sequence_len-2
        if num_steps < 1: return 
        pred_seq_len = sequence_len-num_steps

        self.node_idxs = (batch.get('node_ids')[0,0,0,:])

        input_nodes_forward = graph_seq_nodes[:,1:pred_seq_len+1,:,:].clone().detach()
        input_edges_forward = graph_seq_edges[:,1:pred_seq_len+1,:,:].clone().detach()
        reference_edges = input_edges_forward.clone().detach()
        self.results['reference_locations'] = torch.cat([self.results['reference_locations'], reference_edges], dim=0)
        
        if not self.original_model:
            graph_latents, _, _ = self.object_activity_coembedding_module.autoencode_graph(graph_seq_nodes[:,:pred_seq_len+1,:,:], graph_seq_edges[:,:pred_seq_len+1,:,:], graph_dyn_edges[:,:pred_seq_len+1,:,:], time_context=time_context[:,:pred_seq_len])
            activity_latents, _, _ = self.object_activity_coembedding_module.autoencode_activity(activity_seq[:,:pred_seq_len,:], time_context=time_context[:,:pred_seq_len])
            latents_forward = (graph_latents+activity_latents)

        self.num_test_batches += 1


        for step in range(num_steps):
            if not self.original_model:
                latents_forward, _ = self.predict(latents_forward, time_context[:,step:pred_seq_len+step])
            else:
                latents_forward = LatentDeterministic(time_context[:,step+1:pred_seq_len+step+1], self.cfg.learn_latent_magnitude)
            
            expected_edges = graph_seq_edges[:,2+step:pred_seq_len+2+step,:,:]
            expected_activities = activity_id_seq[:,1+step:pred_seq_len+1+step]

            ## Prediction
            latent_in = latents_forward + time_context[:,step+1:pred_seq_len+step+1] if self.cfg.addtnl_time_context else latents_forward
            pred_activities, _, _ = self.object_activity_coembedding_module.decode_activity(latents=latent_in)
            pred_edges_original, _, _ = self.object_activity_coembedding_module.decode_graph(latents=latent_in,
                                                                                    input_nodes=input_nodes_forward, 
                                                                                    input_edges=input_edges_forward, 
                                                                                    dynamic_edges_mask=graph_dyn_edges[:,1+step:pred_seq_len+1+step,:,:])

            assert not EXTRACAREFUL or torch.all(pred_edges_original[(graph_dyn_edges[:,1+step:pred_seq_len+1+step,:,:] == 0)] == input_edges_forward[(graph_dyn_edges[:,1+step:pred_seq_len+1+step,:,:] == 0)]), \
                    "Not all static edges are the same as in the input graph"

            pred_edges_thresh = pred_edges_original > 0.5
            pred_movements = pred_edges_thresh != reference_edges
            gt_movements = expected_edges != reference_edges
            
            self.results['confusion_matrix']['tp'][step] += int((torch.bitwise_and(pred_movements, gt_movements)).sum())
            self.results['confusion_matrix']['fp'][step] += int((torch.bitwise_and(pred_movements, torch.bitwise_not(gt_movements))).sum())
            self.results['confusion_matrix']['fn'][step] += int((torch.bitwise_and(torch.bitwise_not(pred_movements), gt_movements)).sum())
            self.results['confusion_matrix']['tn'][step] += int((torch.bitwise_and(torch.bitwise_not(pred_movements), torch.bitwise_not(gt_movements))).sum())

            ## TODO Maithili for MHC: Reinstate lenient metric predictions by using 'n' extra steps for gt changes for precision and 'n' fewer steps for pred changes for recall
            # if step >= 2: 
            #     self.results['precision_lenient']['correct'][step] += int(torch.bitwise_and(correct, lenient_changes_pred).sum())
            #     self.results['precision_lenient']['wrong_destination'][step] += int(torch.bitwise_and((torch.bitwise_and(wrong, used_mask)), lenient_changes_pred).sum())
            #     self.results['precision_lenient']['wrong_object'][step] += int(torch.bitwise_and((torch.bitwise_and(wrong, unused_mask)), lenient_changes_pred).sum())
            #     self.results['precision_lenient']['total'][step] += int((lenient_changes_pred).sum())

            self.results['activity']['differences'][step] += (expected_activities - pred_activities).abs()


    def training_step(self, batch, batch_idx):
        batch_training_dropout = (torch.rand_like(batch['activity_ids'].float()) < self.cfg.activity_dropout_train).to(bool)
        final_mask = torch.bitwise_or(batch['activity_mask_drop'], batch_training_dropout)
        
        batch['activity_features'].masked_fill_(final_mask, 0)
        batch['activity_ids'].masked_fill_(batch['activity_mask_drop'], 0)

        results = self(batch)
        for elem in ['loss', 'accuracies', 'latents']:
            for k, v in results[elem].items():
                self.log(f'Train loss -- {k}',results[elem][k])
        res = torch.tensor([0.], requires_grad=True).to('cuda')

        if not self.original_model:   
            res += results['loss']['object_autoencoder']
            res += results['loss']['activity_autoencoder']
            if self.cfg.loss_object_cross:
                res += results['loss']['object_cross_pred']
            if self.cfg.loss_activity_cross:
                res += results['loss']['activity_cross_pred']
            if self.cfg.loss_object_combined:
                res += results['loss']['object_combined_pred']
            if self.cfg.loss_activity_combined:
                res += results['loss']['activity_combined_pred']
            if self.cfg.loss_latent_similarity:
                res += results['loss']['latent_similarity'] * self.cfg.latent_similarity_weight
            if self.cfg.loss_latent_pred:
                res += results['loss']['latent_pred'] + results['loss']['latent_pred_oversht']
            if self.cfg.loss_object_pred:
                res += results['loss']['object_pred'] + results['loss']['object_pred_oversht']
            if self.cfg.loss_activity_pred:
                res += results['loss']['activity_pred'] + results['loss']['activity_pred_oversht']
        else:
            res += results['loss']['object_pred']
            res += results['loss']['activity_pred']
        return res
        
    def validation_step(self, batch, batch_idx):
        batch['activity_features'].masked_fill_(batch['activity_mask_drop'], 0)
        results = self(batch)
        for elem in ['loss', 'accuracies']:
            for k, v in results[elem].items():
                self.log(f'Val {elem} -- {k}',results[elem][k])
        
        self.reset_validation()
        self.evaluate_prediction(batch, num_steps=self.cfg.lookahead_steps)
        
        # Set early stopping metric
        self.log('Val_ES_accuracy',results['accuracies']['object'])

        self.reset_validation()
        return 

    def test_step(self, batch, batch_idx):
        batch['activity_features'].masked_fill_(batch['activity_mask_drop'], 0)
        if self.test_forward:
            results = self(batch)
            for elem in ['loss', 'accuracies']:
                for k, v in results[elem].items():
                    self.log(f'Train {elem} -- {k}',results[elem][k])
        self.evaluate_prediction(batch, num_steps=self.cfg.lookahead_steps)
        return 

    def write_results(self, output_dir, common_data, suffix=''):
        
        node_classes=common_data['node_classes']

        os.makedirs(output_dir, exist_ok=True)
        node_classes_in_order = [node_classes[n.item()] for n in self.node_idxs.int()]
        self.results['node_classes'] = node_classes_in_order
        self.results = get_metrics(self.results)
        results_torch = {k:v for k,v in self.results.items() if k in ['reference_locations', 'activity']}
        results_json = {k:v for k,v in self.results.items() if k not in results_torch}
        torch.save(results_torch, os.path.join(output_dir,f'test_evaluation_splits.pt'))
        json.dump(results_json, open(os.path.join(output_dir,f'test_evaluation_splits.json'),'w'), indent=4)
        self.reset_validation()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

