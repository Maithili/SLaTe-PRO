from argparse import ArgumentError
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from copy import deepcopy
import json
import sys
sys.path.append('helpers')
import random
import numpy as np
from adict import adict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.nn import functional as F
from torch.optim import Adam
from pytorch_lightning.core.module import LightningModule
from ObjectActivityCoembedding import ObjectActivityCoembeddingModule, Latent, LatentDeterministic, EXTRACAREFUL
from utils import color_palette, get_metrics, wrap_str

random.seed(23435)
np.random.seed(23435)

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
        _dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results_with_clarification = {}
        self.result_data = {
            'relocation_distributions':[torch.tensor([]).to(_dev) for _ in range(self.cfg.lookahead_steps)],
            'relocation_locations_gt':[torch.tensor([]).to(_dev) for _ in range(self.cfg.lookahead_steps)],
            'obj_mask':[torch.tensor([]).to(_dev) for _ in range(self.cfg.lookahead_steps)],
            'reference_locations': torch.tensor([]).to(_dev),
            'activity_distributions':[torch.tensor([]).to(_dev) for _ in range(self.cfg.lookahead_steps )],
            'activity_gt':[torch.tensor([]).to(_dev) for _ in range(self.cfg.lookahead_steps)],
            'relocation_distributions_clarified':{},
            'activity_distributions_clarified':{},
            'num_queries':{},
            'query_step': min(self.cfg.lookahead_steps, self.cfg.query_step)-1,
        }
        self.results = {
                'moved_perc':{'correct':[0 for _ in range(self.cfg.lookahead_steps)], 
                            'wrong':[0 for _ in range(self.cfg.lookahead_steps)], 
                            'missed':[0 for _ in range(self.cfg.lookahead_steps)],
                            },
                'unmoved_perc':{'fp':[0 for _ in range(self.cfg.lookahead_steps)], 
                            'tn':[0 for _ in range(self.cfg.lookahead_steps)],
                            },
                'activity_perc':{'correct':[0 for _ in range(self.cfg.lookahead_steps)], 
                            'wrong':[0 for _ in range(self.cfg.lookahead_steps)],
                            },
                'moved':{'correct':[0 for _ in range(self.cfg.lookahead_steps)], 
                            'wrong':[0 for _ in range(self.cfg.lookahead_steps)], 
                            'missed':[0 for _ in range(self.cfg.lookahead_steps)],
                            'total':[0 for _ in range(self.cfg.lookahead_steps)],
                            },
                'unmoved':{'fp':[0 for _ in range(self.cfg.lookahead_steps)], 
                            'tn':[0 for _ in range(self.cfg.lookahead_steps)],
                            'total':[0 for _ in range(self.cfg.lookahead_steps)],
                            },
                'precision':{'correct':[0 for _ in range(self.cfg.lookahead_steps)], 
                         'wrong_destination':[0 for _ in range(self.cfg.lookahead_steps)], 
                         'wrong_object':[0 for _ in range(self.cfg.lookahead_steps)],
                         'total':[0 for _ in range(self.cfg.lookahead_steps)],
                         },
                'precision_lenient':{'correct':[0 for _ in range(self.cfg.lookahead_steps)], 
                         'wrong_destination':[0 for _ in range(self.cfg.lookahead_steps)], 
                         'wrong_object':[0 for _ in range(self.cfg.lookahead_steps)],
                         'total':[0 for _ in range(self.cfg.lookahead_steps)],
                         },
                'precision_perc_lenient':{'with_destination':[0 for _ in range(self.cfg.lookahead_steps)],
                                        'only_object':[0 for _ in range(self.cfg.lookahead_steps)],
                                        },
                'precision_perc':{'with_destination':[0 for _ in range(self.cfg.lookahead_steps)], 
                               'only_object':[0 for _ in range(self.cfg.lookahead_steps)], 
                               },
                'moved_by_activity':[{'correct':0,
                                      'wrong':0,
                                      'missed':0,
                                      'total':0,
                                      'fp':0,
                                      } for _ in range(self.cfg.n_activities)],
                'moved_by_consistency':[],
                'activity':{'correct':[0 for _ in range(self.cfg.lookahead_steps)], 
                            'wrong':[0 for _ in range(self.cfg.lookahead_steps)],
                            'total':[0 for _ in range(self.cfg.lookahead_steps)],
                            },
                'activity_confusion':[[0 for _ in range(self.cfg.n_activities)] for _ in range(self.cfg.n_activities)],
                'queries':[{'num_asked':0,
                            'num_predictions':0,
                            'perc_asked':None} for _ in range(self.cfg.lookahead_steps)],
                'corrects_steps':[],
                'missed_steps':[],
                'wrong_steps':[],
        }

    
    def set_object_consistency(self, consistency):
        self.object_usage_frequency = consistency

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
            _, sequence_length, _ = latents.size()
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
        
        activity_relevant_objects = batch.get('activity_relevant_objects', torch.zeros((graph_seq_nodes.size()[0], graph_seq_nodes.size()[1]-1, graph_seq_nodes.size()[2])))
        assert activity_relevant_objects.size()[0] == graph_seq_nodes.size()[0], f"Wrong auxiliary activity target size {activity_relevant_objects.size()[0]} v.s. {graph_seq_nodes.size()[0]}"
        assert activity_relevant_objects.size()[1] == graph_seq_nodes.size()[1]-1, f"Wrong auxiliary activity target size {activity_relevant_objects.size()[1]} v.s. {graph_seq_nodes.size()[1]-1}"
        assert activity_relevant_objects.size()[2] == graph_seq_nodes.size()[2], f"Wrong auxiliary activity target size {activity_relevant_objects.size()[2]} v.s. {graph_seq_nodes.size()[2]}"
        activity_relevant_objects[latent_mask] = -1


        batch_size, sequence_len_plus_one, num_nodes, node_feature_len = graph_seq_nodes.size()
        batch_size_e, sequence_len_plus_one_e, num_f_nodes, num_t_nodes = graph_seq_edges.size()
        batch_size_act, sequence_len, _ = activity_seq.size()
        
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
        accuracy_object_autoenc = {'used':0, 'unused':0}
        combined_accuracy_object = {'used':0, 'unused':0}
        accuracy_activity_autoenc = torch.Tensor([0.]).to('cuda')
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
            graph_latents, graph_autoenc_loss, accuracy_object_autoenc = self.object_activity_coembedding_module.autoencode_graph(graph_seq_nodes, graph_seq_edges, graph_dynamic_edges_mask, time_context=time_context)

            activity_latents, activity_autoenc_loss, accuracy_activity_autoenc = self.object_activity_coembedding_module.autoencode_activity(activity_seq, activity_gt=activity_id_seq, time_context=time_context)

            # Latent training
            latent_in = activity_latents + time_context if self.cfg.addtnl_time_context else activity_latents
            _, cross_graph_pred_loss, cross_accuracy_object = self.object_activity_coembedding_module.decode_graph(
                                                                        latents=latent_in, 
                                                                        input_nodes=graph_seq_nodes[:,:-1,:,:],
                                                                        input_edges=graph_seq_edges[:,:-1,:,:],
                                                                        dynamic_edges_mask=graph_dynamic_edges_mask[:,:-1,:,:],
                                                                        output_edges=graph_seq_edges[:,1:,:,:],
                                                                        activity_relevant_edges = activity_relevant_objects,
                                                                        activity_mask = latent_mask)

            latent_in = graph_latents + time_context if self.cfg.addtnl_time_context else graph_latents
            _, cross_activity_pred_loss, cross_accuracy_activity = self.object_activity_coembedding_module.decode_activity(
                                                                                    latents=latent_in, 
                                                                                    ground_truth=activity_id_seq)

            latent_similarity_loss = self.object_activity_coembedding_module.latent_loss(graph_latents, activity_latents, mask=latent_mask)
            
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
                                                                            activity_relevant_edges = activity_relevant_objects[:,1:,:])

            _, activity_pred_loss, accuracy_activity = self.object_activity_coembedding_module.decode_activity(
                                                                                        latents=latent_in, 
                                                                                        ground_truth=activity_id_seq[:,1:])


        else:
            pred_edges, graph_pred_loss, accuracy_object = self.object_activity_coembedding_module.decode_graph(
                                                                            latents=time_context[:,1:], 
                                                                            input_nodes=input_nodes_forward,
                                                                            input_edges=input_edges_forward,
                                                                            dynamic_edges_mask=graph_dynamic_edges_mask[:,1:-1,:,:],
                                                                            output_edges=output_edges_forward)
            _, activity_pred_loss, accuracy_activity = self.object_activity_coembedding_module.decode_activity(
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
                                                                                                    activity_relevant_edges = activity_relevant_objects[:,2+i:,:])

                _, additional_activity_pred_loss, _ = self.object_activity_coembedding_module.decode_activity(
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
                        'object_used': accuracy_object['used'],
                        'object_unused': accuracy_object['unused'],
                        'activity': accuracy_activity,
                        'object_autoenc_used': accuracy_object_autoenc['used'],
                        'object_autoenc_unused': accuracy_object_autoenc['unused'],
                        'activity_autoenc': accuracy_activity_autoenc,
                        'object_combined_used': combined_accuracy_object['used'],
                        'object_cross_used': cross_accuracy_object['used'],
                        'object_combined_unused': combined_accuracy_object['unused'],
                        'object_cross_unused': cross_accuracy_object['unused'],
                        'activity_combined': combined_accuracy_activity,
                        'activity_cross': cross_accuracy_activity,
            },
            'latents' : latent_magn
        }
        
        return results

    def rollout_onestep_with_masks(self, latent_series, time_context_series, num_steps, graph_seq_edges, input_nodes_forward, time_context, graph_dyn_edges, query_time, query_type, query_mask, reference_activity, activity_embedding_matrix):
        reference_edges = graph_seq_edges[:,0].argmax(-1).unsqueeze(1)
        input_edges_forward = graph_seq_edges[:,0].unsqueeze(1).float()
        relocations_best_prob = torch.zeros_like(reference_edges).float()
        relocations_probs = torch.zeros_like(input_edges_forward).float()
        query_prob = torch.zeros((query_mask.size()[0],1)).to('cuda') if query_mask is not None else None
        pred_seq_len = 1

        def correct_activity(pred_activity, activity_mask):
            _activities = pred_activity * activity_mask
            _activities_probs = _activities.sum(-1).clone().detach()
            assert (_activities_probs > -0.0001).all() and (_activities_probs < 1.0001).all()
            _corrected_pred_activity = _activities/_activities_probs.unsqueeze(-1)
            _latent_correction_act = (self.object_activity_coembedding_module.activity_encoder((_corrected_pred_activity @ activity_embedding_matrix.T).clone().detach()))
            assert not torch.isnan(_latent_correction_act.sample()).any()
            return _corrected_pred_activity.clone().detach(), _latent_correction_act, _activities_probs.clone().detach()

        pred_activity_next_prob_reference = 1.
        if not self.original_model:
            latent_in = latent_series[:,-1,:] + time_context[:,:1] if self.cfg.addtnl_time_context else latent_series[:,-1,:]
        else:
            latent_in = time_context[:,:1]
        reference_activity_probs = (self.object_activity_coembedding_module.decode_activity(latents=latent_in))[0]
        reference_activity = reference_activity_probs.argmax(-1)
        if len(reference_activity.size()) == 1: 
            reference_activity = reference_activity.unsqueeze(1)
            reference_activity_mask = F.one_hot(reference_activity, reference_activity_probs.size()[-1])
            reference_activity_probs = reference_activity_probs.unsqueeze(1)
        new_pred_activity = False
        activity_probs = -torch.ones_like(reference_activity_probs).to('cuda')
        activity_best_step = -1
        pred_activity_inf = -1

        if query_type == 'activity':
            activity_mask = query_mask
            act_query_applied = torch.zeros((activity_mask.size()[0], activity_mask.size()[1])).to(bool).to('cuda')
            query_prob = torch.zeros_like(act_query_applied).float().to('cuda')
            query_time_act = query_time
        activity_mask_flag = query_type == 'activity'
        graph_mask_flag = query_type == 'graph'

        for step in range(num_steps):
            expected_objects = graph_seq_edges[:,1+step].unsqueeze(1).argmax(-1)
            assert not EXTRACAREFUL or torch.allclose(input_edges_forward.sum(-1), torch.ones_like(input_edges_forward.sum(-1)), atol=0.1), "Edges are not normalized!"
            if not self.original_model:
                latent_series, _ = self.predict(latent_series, time_context_series[:,step:-num_steps+step])
                latents_forward = latent_series[:,-1:,:]
            else:
                latents_forward = LatentDeterministic(time_context[:,step+1:pred_seq_len+step+1], self.cfg.learn_latent_magnitude)
            
            assert not torch.isnan(latents_forward.sample()).any()
            if activity_mask_flag:
                ## Activity Prediction
                latent_in = latents_forward + time_context[:,step+1:pred_seq_len+step+1] if self.cfg.addtnl_time_context else latents_forward
                pred_activity, _, _ = self.object_activity_coembedding_module.decode_activity(latents=latent_in)

                ## Activity based Correction
                if activity_mask is not None and query_time_act == step:
                    _pred_activity_correction, _latent_correction_act, _query_probs = correct_activity(pred_activity, activity_mask)
                    if query_type == 'activity':
                        query_prob = deepcopy(_query_probs)
                    pred_activity = deepcopy(_pred_activity_correction)
                    latents_forward = latents_forward.sample()
                    latents_forward = deepcopy((1 - self.cfg.query_trust) * latents_forward + self.cfg.query_trust * _latent_correction_act.sample())
                    assert (pred_activity.argmax(-1) == activity_mask.argmax(-1)).all()
                    assert not torch.isnan(latents_forward).any()
                    latents_forward = LatentDeterministic(latents_forward, learn_magnitude=self.cfg.learn_latent_magnitude)
                    latent_in = latents_forward + time_context[:,step+1:pred_seq_len+step+1] if self.cfg.addtnl_time_context else latents_forward
                if activity_mask is not None and step < query_time_act:
                    _pred_activity_correction, _latent_correction_act, _ = correct_activity(pred_activity, reference_activity_mask)
                    pred_activity = deepcopy(_pred_activity_correction)
                    latents_forward = latents_forward.sample()
                    latents_forward = deepcopy((1 - self.cfg.query_trust) * latents_forward + self.cfg.query_trust * _latent_correction_act.sample())
                    assert not torch.isnan(latents_forward).any()
                    assert (pred_activity.argmax(-1) == reference_activity_mask.argmax(-1)).all()
                    latents_forward = LatentDeterministic(latents_forward, learn_magnitude=self.cfg.learn_latent_magnitude)
                    latent_in = latents_forward + time_context[:,step+1:pred_seq_len+step+1] if self.cfg.addtnl_time_context else latents_forward
               
                ## Graph Prediction
                pred_edges, _, _ = self.object_activity_coembedding_module.decode_graph(latents=latent_in, 
                                                                                        input_nodes=input_nodes_forward, 
                                                                                        input_edges=input_edges_forward, 
                                                                                        dynamic_edges_mask=graph_dyn_edges[:,1+step:pred_seq_len+1+step,:,:]
                                                                                        )
                
            
            elif graph_mask_flag:
                ## Graph Prediction
                latent_in = latents_forward + time_context[:,step+1:pred_seq_len+step+1] if self.cfg.addtnl_time_context else latents_forward
                pred_edges, _, _ = self.object_activity_coembedding_module.decode_graph(latents=latent_in, 
                                                                                        input_nodes=input_nodes_forward, 
                                                                                        input_edges=input_edges_forward, 
                                                                                        dynamic_edges_mask=graph_dyn_edges[:,1+step:pred_seq_len+1+step,:,:]
                                                                                        )
                ## Graph based Correction
                assert not EXTRACAREFUL or torch.allclose(pred_edges.sum(-1), torch.ones_like(pred_edges.sum(-1)), atol=0.1), "Edges are not normalized!"
                if query_type == 'graph':
                    _edges = ((pred_edges+1e-8)*query_mask)
                    _edges_sum = _edges.sum(-1).clone().detach()
                    if query_time is None:
                        pred_edges = _edges.clone().detach()/(_edges_sum.unsqueeze(-1))
                        assert torch.allclose(pred_edges*(1-query_mask), torch.tensor([0.]).to('cuda')), "Graph mask is not working!!"
                    else:
                        query_prob[query_time==step] = _edges_sum.prod(-1)[query_time==step].float()
                        pred_edges[query_time==step] = (_edges.clone().detach()/(_edges_sum.unsqueeze(-1)))[query_time==step]
                        assert torch.allclose((pred_edges*(1-query_mask))[query_time==step], torch.tensor([0.]).to('cuda')), "Graph mask is not working!!"
                if not self.original_model:
                    _latent_correction_graph = self.object_activity_coembedding_module.graph_encoder(nodes=input_nodes_forward, edges=(input_edges_forward,pred_edges),
                                                                                                     time_context=time_context[:,step+1:pred_seq_len+step+1])
                    _latents = ((1 - self.cfg.query_trust) * latents_forward.sample() + self.cfg.query_trust * _latent_correction_graph.sample())
                    latents_forward = _latents
                    latent_in = latents_forward + time_context[:,step+1:pred_seq_len+step+1] if self.cfg.addtnl_time_context else latents_forward
            
                ## Activity Prediction
                pred_activity, _, _ = self.object_activity_coembedding_module.decode_activity(latents=latent_in)
                assert not EXTRACAREFUL or torch.allclose(pred_edges.sum(-1), torch.ones_like(pred_edges.sum(-1)), atol=0.1), "Edges are not normalized!"
            
            else:
                latent_in = latents_forward + time_context[:,step+1:pred_seq_len+step+1] if self.cfg.addtnl_time_context else latents_forward
                ## Activity Prediction
                pred_activity, _, _ = self.object_activity_coembedding_module.decode_activity(latents=latent_in)
                ## Graph Prediction
                pred_edges, _, _ = self.object_activity_coembedding_module.decode_graph(latents=latent_in, 
                                                                                        input_nodes=input_nodes_forward, 
                                                                                        input_edges=input_edges_forward, 
                                                                                        dynamic_edges_mask=graph_dyn_edges[:,1+step:pred_seq_len+1+step,:,:]
                                                                                        )

            if query_type is None:
                if not new_pred_activity:
                    new_pred_activity = (pred_activity.argmax(-1) != reference_activity)
                    if new_pred_activity:
                        pred_activity_inf = pred_activity.argmax(-1)
                if new_pred_activity and (pred_activity[0,0,reference_activity] > pred_activity_next_prob_reference or pred_activity_inf != pred_activity.argmax(-1)):
                    pred_activity_next_prob_reference = 0
                if pred_activity[0,0,reference_activity] < pred_activity_next_prob_reference:
                    pred_activity_next_prob_reference = pred_activity[0,0,reference_activity]
                    activity_best_step = step
                    activity_probs = pred_activity
            else:
                mask = (activity_probs == -1) & (pred_activity.argmax(-1) != reference_activity).unsqueeze(1)
                activity_probs[mask] = pred_activity[mask]

            if step > 0:
                new_changes_pred = deepcopy(torch.bitwise_and(pred_edges.argmax(-1) != reference_edges, torch.bitwise_not(changes_pred)))
                dest_pred[new_changes_pred] = deepcopy(pred_edges.argmax(-1)[new_changes_pred])
                changes_pred = deepcopy(torch.bitwise_or(changes_pred, new_changes_pred))
                new_changes_gt = deepcopy(torch.bitwise_and(expected_objects != reference_edges, torch.bitwise_not(changes_gt)))
                dest_gt[new_changes_gt] = deepcopy(expected_objects[new_changes_gt])
                changes_gt = deepcopy(torch.bitwise_or(changes_gt, new_changes_gt))
            else:
                dest_pred = deepcopy(pred_edges.argmax(-1))
                changes_pred = deepcopy(dest_pred != reference_edges)
                changes_gt = deepcopy(expected_objects != reference_edges)
                dest_gt = deepcopy(expected_objects)
                new_changes_pred = deepcopy(changes_pred)
                new_changes_gt = deepcopy(changes_gt)

            confidence_in_relocation = deepcopy((pred_edges * F.one_hot(dest_pred, num_classes = pred_edges.size()[-1])).sum(-1))
            continued_changes_pred = pred_edges.argmax(-1) == dest_pred
            better_confidence = torch.bitwise_or(torch.bitwise_and(confidence_in_relocation > relocations_best_prob, continued_changes_pred), new_changes_pred)
            relocations_best_prob[better_confidence] = confidence_in_relocation[better_confidence]
            relocations_probs[better_confidence] = deepcopy(pred_edges[better_confidence])
       
            input_edges_forward = deepcopy(pred_edges).float()

            if not self.original_model:
                latent_series_updated = latent_series.sample()
                if isinstance(latents_forward, Latent):
                    latent_series_updated[:,-1,:] = latents_forward.sample().squeeze(1)
                else:
                    latent_series_updated[:,-1,:] = latents_forward.squeeze(1)
                latent_series = LatentDeterministic(latent_series_updated, latent_series.learn_magnitude)

        mask = (activity_probs == -1)
        activity_probs[mask] = pred_activity[mask]

        assert (changes_pred.to(int) == (reference_edges != relocations_probs.argmax(-1)).to(int)).all(), "Relocation probabilities and changes pred don't match!!"

        return relocations_probs, query_prob, activity_probs, changes_pred, activity_best_step


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

        obj_mask = graph_dyn_edges.sum(-1)>0
        assert obj_mask.size(0) == 1, "This hack doesn't work with >1 batch size"
        assert (obj_mask[0,0,:] == obj_mask[0,5,:]).all(), "Just checking!!"
        assert graph_seq_nodes.size(0) == 1, "This hack doesn't work with >1 batch size"
        self.node_idxs = (batch.get('node_ids')[0,0,0,:])

        input_nodes_forward = graph_seq_nodes[:,1:pred_seq_len+1,:,:].clone().detach()
        input_edges_forward = graph_seq_edges[:,1:pred_seq_len+1,:,:].clone().detach()
        reference_edges = input_edges_forward.argmax(-1).clone().detach()
        reference_activity = activity_seq[:,:pred_seq_len,:].argmax(-1).clone().detach()
        self.result_data['reference_locations'] = torch.cat([self.result_data['reference_locations'], reference_edges], dim=0)
        correct = torch.zeros_like(reference_edges).to('cuda')
        wrong = torch.zeros_like(reference_edges).to('cuda')
        
        initial_latents_forward = None
        activity_embedding_matrix = None
        
        if not self.original_model:
            graph_latents, _, _ = self.object_activity_coembedding_module.autoencode_graph(graph_seq_nodes[:,:pred_seq_len+1,:,:], graph_seq_edges[:,:pred_seq_len+1,:,:], graph_dyn_edges[:,:pred_seq_len+1,:,:], time_context=time_context[:,:pred_seq_len])
            activity_latents, _, _ = self.object_activity_coembedding_module.autoencode_activity(activity_seq[:,:pred_seq_len,:], time_context=time_context[:,:pred_seq_len])
            latents_forward = (graph_latents+activity_latents)
            activity_embedding_matrix = batch['activity_embedder'](torch.arange(self.cfg.n_activities).to('cuda')).float().detach()
            initial_latents_forward = deepcopy(latents_forward)

        self.num_test_batches += 1


        relocations_best_prob = torch.zeros((batch_size_act, pred_seq_len, self.cfg.n_nodes)).to('cuda')
        relocations_best_step = 100 * torch.ones((batch_size_act, pred_seq_len, self.cfg.n_nodes)).to('cuda')
        relocations_probs = torch.zeros((batch_size_act, pred_seq_len, self.cfg.n_nodes, self.cfg.n_nodes)).to('cuda')
        activity_probs = torch.zeros((batch_size_act, pred_seq_len, self.cfg.n_activities)).to('cuda')
        activity_best_step = (num_steps-1) * torch.ones((batch_size_act, pred_seq_len, self.cfg.n_activities)).to('cuda')
        pred_activities_all = torch.tensor([]).to('cuda')
        pred_activity_next = -torch.ones_like(reference_activity).to('cuda')
        expected_activity_next = -torch.ones_like(reference_activity).to('cuda')
        for step in range(num_steps):
            if not self.original_model:
                latents_forward, _ = self.predict(latents_forward, time_context[:,step:pred_seq_len+step])
            else:
                latents_forward = LatentDeterministic(time_context[:,step+1:pred_seq_len+step+1], self.cfg.learn_latent_magnitude)
            
            expected_objects = graph_seq_edges[:,2+step:pred_seq_len+2+step,:,:].argmax(-1)
            expected_activities = activity_id_seq[:,1+step:pred_seq_len+1+step]
            new_expected_activity = (expected_activity_next == -1) & (expected_activities != reference_activity)
            expected_activity_next[new_expected_activity] = expected_activities[new_expected_activity]

            ## Activity Prediction
            latent_in = latents_forward + time_context[:,step+1:pred_seq_len+step+1] if self.cfg.addtnl_time_context else latents_forward
            pred_activity, _, _ = self.object_activity_coembedding_module.decode_activity(latents=latent_in)
            pred_activities = pred_activity.argmax(-1)
            new_pred_activity = (pred_activity_next == -1) & (pred_activities != reference_activity)
            pred_activity_next[new_pred_activity] = pred_activities[new_pred_activity]
            activity_best_step[new_pred_activity] = step
            activity_probs[new_pred_activity] = pred_activity[new_pred_activity]
            
            pred_edges_original, _, _ = self.object_activity_coembedding_module.decode_graph(latents=latent_in,
                                                                                    input_nodes=input_nodes_forward, 
                                                                                    input_edges=input_edges_forward, 
                                                                                    dynamic_edges_mask=graph_dyn_edges[:,1+step:pred_seq_len+1+step,:,:])

            assert not EXTRACAREFUL or torch.all(pred_edges_original[(graph_dyn_edges[:,1+step:pred_seq_len+1+step,:,:] == 0)] == input_edges_forward[(graph_dyn_edges[:,1+step:pred_seq_len+1+step,:,:] == 0)]), \
                    "Not all static edges are the same as in the input graph"

            pred_edges = pred_edges_original.clone().detach()
            pred_edges_if_moved = pred_edges_original.clone().detach()
            pred_edges_if_moved[F.one_hot(reference_edges, num_classes = pred_edges.size()[-1]).bool()] = 0

            obj_mask = graph_dyn_edges[:,1+step:pred_seq_len+1+step,:,:].sum(-1)>0

            if step > 0:
                new_changes_pred = deepcopy(torch.bitwise_and(pred_edges.argmax(-1) != reference_edges, torch.bitwise_not(changes_pred)))
                dest_pred[new_changes_pred] = deepcopy(pred_edges.argmax(-1)[new_changes_pred])
                changes_pred = deepcopy(torch.bitwise_or(changes_pred, new_changes_pred))
                new_changes_gt = deepcopy(torch.bitwise_and(expected_objects != reference_edges, torch.bitwise_not(changes_gt)))
                dest_gt[new_changes_gt] = deepcopy(expected_objects[new_changes_gt])
                changes_gt = deepcopy(torch.bitwise_or(changes_gt, new_changes_gt))
                if step == 2:
                    lenient_changes_pred = deepcopy(torch.bitwise_and(dest_pred != reference_edges, obj_mask))
            else:
                dest_pred = deepcopy(pred_edges.argmax(-1))
                changes_pred = deepcopy(dest_pred != reference_edges)
                new_changes_pred = changes_pred
                changes_gt = deepcopy(expected_objects != reference_edges)
                new_changes_gt = changes_gt
                dest_gt = deepcopy(expected_objects)

            assert new_changes_pred[torch.bitwise_not(obj_mask)].sum() == 0, "New changes predicted in static edges"
            
            confidence_in_relocation = deepcopy((pred_edges * F.one_hot(dest_pred, num_classes = pred_edges.size()[-1])).sum(-1))
            continued_changes_pred = pred_edges.argmax(-1) == dest_pred
            better_confidence = torch.bitwise_or(torch.bitwise_and(confidence_in_relocation > relocations_best_prob, continued_changes_pred), new_changes_pred)
            relocations_best_prob[better_confidence] = confidence_in_relocation[better_confidence]
            relocations_best_step[better_confidence] = step
            relocations_probs[better_confidence] = deepcopy(pred_edges[better_confidence])

            self.result_data['relocation_distributions'][step] = torch.cat([self.result_data['relocation_distributions'][step], relocations_probs], dim=0)
            self.result_data['relocation_locations_gt'][step] = torch.cat([self.result_data['relocation_locations_gt'][step], dest_gt], dim=0)
            self.result_data['obj_mask'][step] = torch.cat([self.result_data['obj_mask'][step], obj_mask], dim=0)
            self.result_data['activity_distributions'][step] = torch.cat([self.result_data['activity_distributions'][step], pred_activity], dim=0)
            self.result_data['activity_gt'][step] = torch.cat([self.result_data['activity_gt'][step], expected_activities], dim=0)

            correct = deepcopy(dest_pred == dest_gt)
            wrong = deepcopy(dest_pred != dest_gt)

            used_mask = deepcopy(torch.bitwise_and(changes_gt, obj_mask))
            unused_mask = deepcopy( torch.bitwise_and(torch.bitwise_not(changes_gt), obj_mask))
            pred_used_mask = deepcopy(torch.bitwise_and(changes_pred, obj_mask))
            used_pred_and_gt = deepcopy(torch.bitwise_and(used_mask, pred_used_mask))
            used_pred_and_not_gt = deepcopy(torch.bitwise_and(torch.bitwise_not(used_mask), pred_used_mask))

            self.results['moved']['correct'][step] += int((torch.bitwise_and(correct, used_pred_and_gt)).sum())
            self.results['moved']['wrong'][step] += int((torch.bitwise_and(wrong, used_pred_and_gt)).sum())
            self.results['moved']['missed'][step] += int((used_mask).sum() - (used_pred_and_gt).sum())
            self.results['moved']['total'][step] += int((used_mask).sum())

            self.results['precision']['correct'][step] += int((torch.bitwise_and(correct, used_pred_and_gt)).sum())
            self.results['precision']['wrong_destination'][step] += int((torch.bitwise_and(wrong, used_pred_and_gt)).sum())
            self.results['precision']['wrong_object'][step] += int(used_pred_and_not_gt.sum())
            self.results['precision']['total'][step] += int((pred_used_mask).sum())

            if step >= 2: 
                self.results['precision_lenient']['correct'][step] += int(torch.bitwise_and(correct, lenient_changes_pred).sum())
                self.results['precision_lenient']['wrong_destination'][step] += int(torch.bitwise_and((torch.bitwise_and(wrong, used_mask)), lenient_changes_pred).sum())
                self.results['precision_lenient']['wrong_object'][step] += int(torch.bitwise_and((torch.bitwise_and(wrong, unused_mask)), lenient_changes_pred).sum())
                self.results['precision_lenient']['total'][step] += int((lenient_changes_pred).sum())

            self.results['unmoved']['fp'][step] += int((torch.bitwise_and(wrong, unused_mask)).sum())
            self.results['unmoved']['tn'][step] += int((unused_mask).sum() - int((torch.bitwise_and(wrong, unused_mask)).sum()))
            self.results['unmoved']['total'][step] += int((unused_mask).sum())
            
            if step == 0:
                for act_num in range(self.cfg.n_activities):
                    act_mask = (activity_id_seq[:,:pred_seq_len] == act_num).unsqueeze(-1)
                    used_and_activity_mask = torch.bitwise_and(used_mask, act_mask)
                    self.results['moved_by_activity'][act_num]['correct'] += int((torch.bitwise_and(correct, used_and_activity_mask)).sum())
                    self.results['moved_by_activity'][act_num]['wrong'] += int((torch.bitwise_and(wrong, used_and_activity_mask)).sum())
                    self.results['moved_by_activity'][act_num]['missed'] += int((used_and_activity_mask).sum() - (torch.bitwise_and(changes_pred, used_and_activity_mask)).sum())
                    self.results['moved_by_activity'][act_num]['total'] += int((used_and_activity_mask).sum())
                    self.results['moved_by_activity'][act_num]['fp'] += int(torch.bitwise_and(used_pred_and_not_gt, act_mask).sum())
            if step == num_steps-1:
                for object_usage_frequency in range(self.object_usage_frequency.max()):
                    if len(self.results['moved_by_consistency']) <= object_usage_frequency:
                        self.results['moved_by_consistency'].append({'correct':0,
                                                                    'wrong':0,
                                                                    'missed':0,
                                                                    'total':0,
                                                                    'fp':0,
                                                                    })
                    consistency_mask = (self.object_usage_frequency[:pred_seq_len,:] == object_usage_frequency).unsqueeze(0).to('cuda')
                    used_and_consistency_mask = torch.bitwise_and(used_mask, consistency_mask)
                    self.results['moved_by_consistency'][object_usage_frequency]['correct'] += int((torch.bitwise_and(correct, used_and_consistency_mask)).sum())
                    self.results['moved_by_consistency'][object_usage_frequency]['wrong'] += int((torch.bitwise_and(wrong, used_and_consistency_mask)).sum())
                    self.results['moved_by_consistency'][object_usage_frequency]['missed'] += int((used_and_consistency_mask).sum() - (torch.bitwise_and(changes_pred, used_and_consistency_mask)).sum())
                    self.results['moved_by_consistency'][object_usage_frequency]['total'] += int((used_and_consistency_mask).sum())
                    self.results['moved_by_consistency'][object_usage_frequency]['fp'] += int(torch.bitwise_and(used_pred_and_not_gt, consistency_mask).sum())

            self.results['activity']['correct'][step] += int((expected_activities == pred_activities).sum())
            self.results['activity']['wrong'][step] += int((expected_activities != pred_activities).sum())
            self.results['activity']['total'][step] += int(torch.numel((expected_activities)))
            for gt_act in range(self.cfg.n_activities):
                gt_act_mask = (expected_activities == gt_act)
                for pred_act in range(self.cfg.n_activities):
                    self.results['activity_confusion'][gt_act][pred_act] += int(((pred_activities == pred_act)[gt_act_mask]).sum())

            pred_activities_all = torch.cat((pred_activities_all, pred_activities.unsqueeze(-1)), dim=-1)
            input_edges_forward = pred_edges.clone().detach()
            
            if len(self.cfg.query_types) > 0 and step == self.result_data['query_step']:
                for query_type in self.cfg.query_types:
                    self.cfg.query_type = query_type
                    if self.cfg.query_type != 'object' and self.original_model: continue
                    data_for_queries = {'batch': batch, 'num_steps': self.result_data['query_step']+1,
                                    'pred_seq_len': pred_seq_len,
                                    'input_nodes_forward': deepcopy(input_nodes_forward),
                                    'input_edges_forward': deepcopy(graph_seq_edges[:,1:pred_seq_len+1,:,:].clone().detach()),
                                    'reference_edges': deepcopy(reference_edges),
                                    'initial_latents_forward' : deepcopy(initial_latents_forward), 
                                    'activity_embedding_matrix' : deepcopy(activity_embedding_matrix),
                                    'relocations_probs' : deepcopy(relocations_probs.clone().detach()),
                                    'relocations_best_prob' : deepcopy(relocations_best_prob.detach()),
                                    'relocations_best_step' : deepcopy(relocations_best_step.detach()),
                                    'activity_probs' : deepcopy(activity_probs.detach()),
                                    'activity_best_step' : deepcopy(activity_best_step.detach()),
                                    'expected_activities' : deepcopy(expected_activities.detach()),
                                    'changes_gt' : deepcopy(changes_gt.detach()),
                                    'dest_gt' : deepcopy(dest_gt.detach()),
                                    'changes_pred' : deepcopy(changes_pred.detach()),
                                    'dest_pred' : deepcopy(dest_pred.detach()),
                                    'pred_activities_all' : deepcopy(pred_activities_all[:,:,-1].detach())}

                    self.evaluate_queries(adict(data_for_queries))
        
    def evaluate_queries(self, data_in):
        graph_seq_nodes = data_in.batch.get('node_features').float()
        graph_seq_edges = data_in.batch.get('edges')
        graph_dyn_edges = data_in.batch.get('dynamic_edges_mask')
        activity_seq = data_in.batch.get('activity_features')[:,:-1,:]
        activity_id_seq = data_in.batch['activity_ids'][:,:-1]
        visible_activity_id_seq = deepcopy(activity_id_seq)
        visible_activity_id_seq.masked_fill_(data_in.batch['activity_mask_drop'][:,:-1], -1)

        pred_seq_len = data_in['pred_seq_len']
        time_context = data_in.batch.get('time_features', torch.zeros((data_in.batch['edges'].size()[0],data_in.batch['edges'].size()[1], self.cfg.c_len)))[:,1:,:]
        if time_context.size()[-1] > self.cfg.c_len:
            time_context = time_context[:,:,:self.cfg.c_len]
            print("Warning: time context is too long. Truncating it to the first {} values".format(self.cfg.c_len))
            print()
            print("THIS IS ONLY OKAY FOR DEBUGGING!!!!!! YOU ARE NOT TRAINING A MODEL THAT WILL WORK IN REALITY!!!!!")
            print()
        if time_context.size()[-1] < self.cfg.c_len:
            time_context = torch.cat([time_context, torch.zeros((time_context.size()[0], time_context.size()[1], self.cfg.c_len - time_context.size()[2])).to('cuda')], dim=-1)

        _, sequence_len, _ = activity_seq.size()

        if sequence_len < data_in.num_steps+2 : data_in.num_steps = sequence_len-2
        if data_in.num_steps < 1: return 

        obj_mask = graph_dyn_edges[:,1:pred_seq_len+1,:,:].sum(-1)>0

        inconsistency_masks = {thresh:torch.bitwise_and(self.object_usage_frequency[:pred_seq_len,:].unsqueeze(0).to('cuda') > thresh, 
                                                        self.object_usage_frequency[:pred_seq_len,:].unsqueeze(0).to('cuda') < (60-thresh)) for thresh in self.cfg.consistency_thresholds}
 
        ## Based on relocation_probs and activity_probs, decide which queries to make
        def usefuleness_metric(pred_edges_original, pred_edges_new):
            if self.cfg.query_usefulness_metric == 'expected_changes':
                return (pred_edges_original.argmax(-1) != pred_edges_new.argmax(-1)).sum(-1)
            elif self.cfg.query_usefulness_metric == 'information_gain':
                def entropy(pedg):
                    return -(pedg*(pedg+1e-8).log()).sum(-1).sum(-1)
                return torch.clip(entropy(pred_edges_original) - entropy(pred_edges_new), min=0)

        relocations_probs_wo_query = deepcopy(data_in.relocations_probs)
        pred_activities_wo_query = deepcopy(data_in.pred_activities_all).to(int)
        relocation_correction = [deepcopy(relocations_probs_wo_query[0,i:i+1,:,:]) for i in range(pred_seq_len)]
        relocation_corrected = [deepcopy(relocations_probs_wo_query[0,i:i+1,:,:]) for i in range(pred_seq_len)]
        pred_act_correction = [(deepcopy(pred_activities_wo_query[0,i:i+1])).to(int) for i in range(pred_seq_len)]
        pred_activities_corrected = [(deepcopy(pred_activities_wo_query[0,i:i+1])) for i in range(pred_seq_len)]
        tensor_for_size = (pred_activities_wo_query[0,:1])
        best_queries_act_viz = [(torch.zeros_like(tensor_for_size).to('cuda')) for _ in range(pred_seq_len)]
        queries_act_viz_carryover = [(torch.zeros_like(tensor_for_size).to('cuda')) for _ in range(pred_seq_len)]
        best_queries_obj_viz = [(torch.zeros_like(tensor_for_size).to('cuda').to(int)) for _ in range(pred_seq_len)]
        queries_obj_viz_carryover = [(torch.zeros((1,obj_mask.size()[-1])).to('cuda').to(int)) for _ in range(pred_seq_len)]
        value = [(torch.zeros_like(tensor_for_size).to('cuda')) for _ in range(pred_seq_len)]
        value_act = [(torch.zeros_like(tensor_for_size).to('cuda')) for _ in range(pred_seq_len)]
        value_obj = [(torch.zeros_like(tensor_for_size).to('cuda')) for _ in range(pred_seq_len)]
        oracle_positive_obj = [(torch.zeros_like(tensor_for_size).to('cuda').to(bool)) for _ in range(pred_seq_len)]
        query_type = [-(torch.ones_like(tensor_for_size)) for _ in range(pred_seq_len)]
        correction_masks = []
        num_carried_obj_queries = 0
        num_carried_act_queries = 0

        if self.cfg.only_confused_queries:
            confused = lambda p: torch.bitwise_and(p>0.1, p<0.9)
        else:
            confused = lambda p: torch.ones_like(p).to(bool).to('cuda') 

        def soften_mask(mask):
            mask += 1e-8
            normalizer = mask.sum(-1).clone().detach()
            mask = mask/normalizer
            return mask.to('cuda')

        changes_at_step_wo_query = data_in.changes_pred[:,:1,:]
        for curr_step in range(pred_seq_len):
            activity_masks = None
            activity_best_step = data_in.activity_best_step[:,curr_step:1+curr_step,0].to(int).item()
            if data_in.initial_latents_forward is not None:
                latent_series = deepcopy(data_in.initial_latents_forward[:,:curr_step+1,:])
            else:
                latent_series = None 

            def evaluate_obj_query(mask, time_mask, ask_mask):
                graph_mask_pos = mask[0]
                graph_mask_neg = mask[1]
                assert not EXTRACAREFUL or torch.allclose(graph_seq_edges.sum(-1), torch.ones_like(graph_seq_edges.sum(-1)), atol=0.1), "Edges are not normalized!"

                ## pos_query
                pos_query_time_mask = time_mask
                relocatn_probs_pos, prob_of_query_pos, pred_activity_pos, changes_pred_pos, _ = self.rollout_onestep_with_masks(
                                                latent_series=LatentDeterministic(deepcopy(latent_series.sample().repeat(mask[0].size()[0],1,1)), learn_magnitude=latent_series.learn_magnitude) if latent_series is not None else None,
                                                time_context_series=deepcopy(time_context[:,:curr_step+1+data_in.num_steps,:].repeat(mask[0].size()[0],1,1)),
                                                num_steps=data_in.num_steps, 
                                                graph_seq_edges=deepcopy(graph_seq_edges[:,curr_step+1:curr_step+data_in.num_steps+2].repeat(mask[0].size()[0],1,1,1)),
                                                input_nodes_forward=deepcopy(graph_seq_nodes[:,curr_step+1:2+curr_step,:,:].repeat(mask[0].size()[0],1,1,1)), 
                                                time_context=deepcopy(time_context[:,curr_step:curr_step+data_in.num_steps+2].repeat(mask[0].size()[0],1,1)), 
                                                graph_dyn_edges=deepcopy(graph_dyn_edges[:,curr_step+1:curr_step+data_in.num_steps+2,:,:].repeat(mask[0].size()[0],1,1,1)), 
                                                query_time = pos_query_time_mask, 
                                                query_type = 'graph', 
                                                query_mask = graph_mask_pos,
                                                reference_activity = activity_id_seq[:,curr_step:curr_step+1],
                                                activity_embedding_matrix=data_in.activity_embedding_matrix)
            
                ## neg_query
                neg_query_time_mask = None if self.cfg.query_negative_at_all_steps else time_mask
                relocatn_probs_neg, prob_of_query_neg, pred_activity_neg, changes_pred_neg, _ = self.rollout_onestep_with_masks(
                                                latent_series=LatentDeterministic(deepcopy(latent_series.sample().repeat(mask[0].size()[0],1,1)), learn_magnitude=latent_series.learn_magnitude) if latent_series is not None else None,
                                                time_context_series=deepcopy(time_context[:,:curr_step+1+data_in.num_steps,:].repeat(mask[0].size()[0],1,1)),
                                                num_steps=data_in.num_steps, 
                                                graph_seq_edges=deepcopy(graph_seq_edges[:,curr_step+1:curr_step+data_in.num_steps+2].repeat(mask[0].size()[0],1,1,1)),
                                                input_nodes_forward=deepcopy(graph_seq_nodes[:,curr_step+1:2+curr_step,:,:].repeat(mask[0].size()[0],1,1,1)), 
                                                time_context=deepcopy(time_context[:,curr_step:curr_step+data_in.num_steps+2].repeat(mask[0].size()[0],1,1)), 
                                                graph_dyn_edges=deepcopy(graph_dyn_edges[:,curr_step+1:curr_step+data_in.num_steps+2,:,:].repeat(mask[0].size()[0],1,1,1)), 
                                                query_time = neg_query_time_mask, 
                                                query_type = 'graph', 
                                                query_mask = graph_mask_neg,
                                                reference_activity = activity_id_seq[:,curr_step:curr_step+1],
                                                activity_embedding_matrix=data_in.activity_embedding_matrix)

                query_value = prob_of_query_pos * usefuleness_metric(relocations_probs_wo_query[:,curr_step:1+curr_step,:,:].repeat(relocatn_probs_pos.size()[0],1,1,1), relocatn_probs_pos) + \
                        (1 - prob_of_query_pos) * usefuleness_metric(relocations_probs_wo_query[:,curr_step:1+curr_step,:,:].repeat(relocatn_probs_pos.size()[0],1,1,1), relocatn_probs_neg)
            
                assert (prob_of_query_pos>=-0.1).all()
                assert (prob_of_query_pos<=1.1).all()
                assert not torch.isnan(query_value).any()

                query_value *= ask_mask

                best_query = query_value.argmax(0)
                all_relocation_probs_pos = relocatn_probs_pos[best_query,:,:,:].unsqueeze(0)
                all_relocation_probs_neg = relocatn_probs_neg[best_query,:,:,:].unsqueeze(0)
                all_act_pred_pos = pred_activity_pos[best_query,:,:].argmax(-1)
                all_act_pred_neg = pred_activity_neg[best_query,:,:].argmax(-1)

                return query_value.max(0).values, \
                    all_relocation_probs_pos, \
                    all_relocation_probs_neg, \
                    best_query, \
                    all_act_pred_pos, \
                    all_act_pred_neg
        
            def oracle_activity(activity_id_seq):
                current_activity = activity_id_seq[0,curr_step:1+curr_step]
                oracle_next_activity = deepcopy(activity_id_seq[0,1+curr_step:2+curr_step])
                oracle_next_activity[oracle_next_activity == current_activity] = -1
                for ostep in range(data_in.num_steps):
                    oracle_next_activity[oracle_next_activity < 2] = activity_id_seq[0,1+curr_step+ostep:2+curr_step+ostep][oracle_next_activity < 2]
                    oracle_next_activity[oracle_next_activity == current_activity] = -1
                ## Only return current activity if no other activity will be performed
                oracle_next_activity[oracle_next_activity == -1] = current_activity[oracle_next_activity == -1]
                return oracle_next_activity
        
            def oracle_object(query_objects, changes_gt):
                oracle_positive = (changes_gt.squeeze(0) * F.one_hot(query_objects, num_classes=changes_gt.size()[-1])).sum(-1)
                oracle_negative = ((1-changes_gt.to(int).squeeze(0)) * F.one_hot(query_objects, num_classes=changes_gt.size()[-1])).sum(-1)
                assert oracle_positive.max() <= 1 and oracle_positive.min() >= 0
                assert oracle_negative.max() <= 1 and oracle_negative.min() >= 0
                assert (oracle_negative == 1 - oracle_positive).all()
                return oracle_positive.to(bool), oracle_negative.to(bool)
        
            no_activity_queries = False

            if self.cfg.query_type in ['activity','both']:
                ## Exact activity queries
                value_local = torch.zeros((1,1)).float().to('cuda')
                relocations_by_activity = []
                pred_act_by_activity = []
                activities_checked = []
                query_activity_mask = torch.tensor([]).to('cuda')
                if activity_masks is None:
                    for activity in range(self.cfg.n_activities):
                        activity_mask = F.one_hot((torch.ones((1,1))*activity).to(int).to('cuda'), num_classes=self.cfg.n_activities).clone().detach()
                        query_activity_mask = torch.cat([query_activity_mask, activity_mask])
                        activities_checked.append(activity)
                    relocations_by_activity, prob_of_query, pred_act_by_activity, changes_pred, _ = self.rollout_onestep_with_masks(
                                                    latent_series=LatentDeterministic(deepcopy(latent_series.sample().repeat(len(activities_checked),1,1)), learn_magnitude=latent_series.learn_magnitude) if latent_series is not None else None,
                                                    time_context_series=deepcopy(time_context[:,:curr_step+1+data_in.num_steps,:].repeat(len(activities_checked),1,1)),
                                                    num_steps=data_in.num_steps, 
                                                    graph_seq_edges=deepcopy(graph_seq_edges[:,curr_step+1:curr_step+data_in.num_steps+2].repeat(len(activities_checked),1,1,1)),
                                                    input_nodes_forward=deepcopy(graph_seq_nodes[:,curr_step+1:2+curr_step,:,:].repeat(len(activities_checked),1,1,1)), 
                                                    time_context=deepcopy(time_context[:,curr_step:curr_step+data_in.num_steps+2].repeat(len(activities_checked),1,1)), 
                                                    graph_dyn_edges=deepcopy(graph_dyn_edges[:,curr_step+1:curr_step+data_in.num_steps+2,:,:].repeat(len(activities_checked),1,1,1)), 
                                                    query_time = activity_best_step, 
                                                    query_type = 'activity', 
                                                    query_mask = query_activity_mask,
                                                    reference_activity = activity_id_seq[:,curr_step:curr_step+1],
                                                    activity_embedding_matrix=data_in.activity_embedding_matrix)
                    value_local = prob_of_query.T @ usefuleness_metric(relocations_probs_wo_query[:,curr_step:1+curr_step,:,:].repeat(len(activities_checked),1,1,1), relocations_by_activity)
                    oracle_labels = oracle_activity(activity_id_seq)
                    actidx = activities_checked.index(oracle_labels.item())
                    relocation_correction[curr_step] = relocations_by_activity[actidx]
                    pred_act_correction[curr_step] = pred_act_by_activity[actidx].argmax(-1)
                    best_queries_act_viz[curr_step] = oracle_labels

                    assert value_local.squeeze(0).size()[0] == 1
                    value[curr_step] = deepcopy(value_local.squeeze(0))
                    query_type[curr_step] = torch.ones_like(value_local.squeeze(0))
                    value_act[curr_step] = deepcopy(value_local.squeeze(0))
                
                else:
                    no_activity_queries = True

            if self.cfg.query_type in ['object','both']:
                object_queries_list_lenient = [torch.argwhere(confused(relprob) & dynamic).squeeze(-1) for relprob,dynamic in zip(data_in.relocations_best_prob[0,curr_step:curr_step+1,:], obj_mask[0,curr_step:curr_step+1,:])]
                object_queries_list = [torch.tensor([o for o in object_queries_list_lenient_step]).to('cuda') for object_queries_list_lenient_step in object_queries_list_lenient]

                num_queries = [len(oq) for oq in object_queries_list]
                if max(num_queries) > 0:
                    object_queries = torch.stack([torch.cat([aq.to(torch.long), torch.zeros(max(num_queries)-nq, dtype=torch.long).to(aq.device)]) for aq, nq in zip(object_queries_list, num_queries)], dim=-1)
                    ask_mask = torch.stack([torch.cat([torch.ones(nq, dtype=torch.long).to(object_queries.device), torch.zeros(max(num_queries)-nq, dtype=torch.long).to(object_queries.device)]) for nq in num_queries], dim=-1)
                    assert changes_at_step_wo_query.size()[0] == 1, "Batch size should be 1 for query hack, else add a dimension to all of this!!!"
                    object_mask = (F.one_hot(object_queries, num_classes=self.cfg.n_nodes))
                    query_time_mask = (object_mask * data_in.relocations_best_step[:,curr_step:curr_step+1,:].repeat(object_mask.size()[0],1,1)).sum(-1)
                    query_time_mask[ask_mask == 0] = 0
                    object_mask = object_mask.unsqueeze(-1).repeat(1,1,1,self.cfg.n_nodes)
                    _reference_edges = deepcopy(graph_seq_edges[:,curr_step+1:curr_step+2,:,:].argmax(-1))
                    reference_edge_mask = F.one_hot(_reference_edges, num_classes=self.cfg.n_nodes).clone().detach()
                    pos_mask = (object_mask*(1-reference_edge_mask))+(1-object_mask)
                    neg_mask = (object_mask*reference_edge_mask)+(1-object_mask)
                    value_obj_at_step, relocation_pos_obj, relocation_neg_obj, best_queries_obj, pred_act_pos_obj, pred_act_neg_obj = evaluate_obj_query((pos_mask, neg_mask), query_time_mask, ask_mask)
                    assert not EXTRACAREFUL or relocation_pos_obj.argmax(-1).squeeze()[object_queries_list[0][best_queries_obj]] != _reference_edges[0,0,object_queries_list[0][best_queries_obj]]
                    assert not EXTRACAREFUL or relocation_neg_obj.argmax(-1).squeeze()[object_queries_list[0][best_queries_obj]] == _reference_edges[0,0,object_queries_list[0][best_queries_obj]]
                    oracle_positive_obj_at_step, _ = oracle_object(object_queries[best_queries_obj, torch.arange(len(best_queries_obj))], data_in.changes_gt[:,curr_step:curr_step+1,:])
                    if EXTRACAREFUL:
                        multiple_locations_for_object = (len(graph_seq_edges[:,curr_step+1:curr_step+2+data_in.num_steps,object_queries_list[0][best_queries_obj],:].argmax(-1).unique()) > 1)
                        assert multiple_locations_for_object == oracle_positive_obj_at_step, f"Object oracle is wrong!! {multiple_locations_for_object} != {oracle_positive_obj_at_step}"
                    oracle_positive_obj[curr_step] = oracle_positive_obj_at_step
                    best_queries_obj_viz[curr_step] = object_queries[best_queries_obj, torch.arange(len(best_queries_obj))]
                    relocation_correction_obj = relocation_neg_obj
                    relocation_correction_obj[oracle_positive_obj_at_step] = relocation_pos_obj[oracle_positive_obj_at_step]
                    pred_act_correction_obj = pred_act_neg_obj
                    pred_act_correction_obj[oracle_positive_obj_at_step] = pred_act_pos_obj[oracle_positive_obj_at_step]
                    value_obj[curr_step] = deepcopy(value_obj_at_step)
                    if self.cfg.query_type == 'both' and not no_activity_queries:
                        if (value_obj_at_step > value_act[curr_step]).sum() > 0:
                            query_type[curr_step][value_obj_at_step>value_act[curr_step]] = 2
                            value[curr_step][value_obj_at_step>value_act[curr_step]] = value_obj_at_step[value_obj_at_step>value_act[curr_step]]
                            relocation_correction[curr_step][value_obj_at_step>value_act[curr_step]] = relocation_correction_obj[value_obj_at_step>value_act[curr_step]]
                            pred_act_correction[curr_step][value_obj_at_step>value_act[curr_step]] = pred_act_correction_obj[value_obj_at_step>value_act[curr_step]]
                    else:
                        query_type[curr_step] = torch.ones_like(value_obj_at_step)*2
                        value[curr_step] = deepcopy(value_obj_at_step)
                        relocation_correction[curr_step] = relocation_correction_obj
                        pred_act_correction[curr_step] = pred_act_correction_obj

            correction_mask = (value[curr_step] > self.cfg.query_thresh).to(bool)
            act_correction = correction_mask & (query_type[curr_step] == 1)
            obj_correction = correction_mask & (query_type[curr_step] == 2)
            value_act[curr_step][torch.bitwise_not(act_correction)] = 0
            value_obj[curr_step][torch.bitwise_not(obj_correction)] = 0

            if correction_mask.sum() > 0:
                relocation_corrected[curr_step][correction_mask] = deepcopy(relocation_correction[curr_step][correction_mask])
                pred_activities_corrected[curr_step][correction_mask] = deepcopy(pred_act_correction[curr_step][correction_mask])

            correction_masks.append(correction_mask)

        relocation_corrected = torch.stack(relocation_corrected, dim=1)
        pred_activities_corrected = torch.stack(pred_activities_corrected, dim=1)
        best_queries_act_viz = torch.cat(best_queries_act_viz, dim=0)
        queries_act_viz_carryover = torch.cat(queries_act_viz_carryover, dim=0)
        best_queries_obj_viz = torch.cat(best_queries_obj_viz, dim=0)
        queries_obj_viz_carryover = torch.cat(queries_obj_viz_carryover, dim=0)
        value = torch.cat(value, dim=0)
        value_act = torch.cat(value_act, dim=0)
        value_obj = torch.cat(value_obj, dim=0)
        oracle_positive_obj = torch.cat(oracle_positive_obj, dim=0)
        query_type = torch.cat(query_type, dim=0)
        correction_masks = torch.cat(correction_masks, dim=0)

        assert ((num_carried_act_queries == 0) and (num_carried_obj_queries == 0))
        
        changes_pred = deepcopy(relocation_corrected.argmax(-1) != graph_seq_edges[0,1:pred_seq_len+1,:,:].argmax(-1))
        dest_pred = deepcopy(relocation_corrected.argmax(-1))

        assert not EXTRACAREFUL or (changes_pred[0,torch.arange(changes_pred.size()[1]), best_queries_obj_viz][oracle_positive_obj & correction_masks & (value_obj>0)]).all()
        assert not EXTRACAREFUL or not (changes_pred[0,torch.arange(changes_pred.size()[1]), best_queries_obj_viz][torch.bitwise_not(oracle_positive_obj) & correction_masks & (value_obj>0)]).any()

        assert not EXTRACAREFUL or (value_act > 0).sum()+(value_obj > 0).sum() == correction_masks.sum()

        if self.cfg.query_type not in self.result_data['relocation_distributions_clarified']:
            self.result_data['relocation_distributions_clarified'][self.cfg.query_type] = deepcopy(relocation_corrected)
            self.result_data['activity_distributions_clarified'][self.cfg.query_type] = deepcopy(pred_activities_corrected)
            self.result_data['num_queries'][self.cfg.query_type] = int(correction_masks.sum())
        else:
            self.result_data['relocation_distributions_clarified'][self.cfg.query_type] = torch.cat([self.result_data['relocation_distributions_clarified'][self.cfg.query_type], 
                                                                                                                    deepcopy(relocation_corrected)], dim=0)
            self.result_data['activity_distributions_clarified'][self.cfg.query_type] = torch.cat([self.result_data['activity_distributions_clarified'][self.cfg.query_type], 
                                                                                                                    deepcopy(pred_activities_corrected)], dim=0)
            self.result_data['num_queries'][self.cfg.query_type] += int(correction_masks.sum())
        if self.cfg.query_type not in self.results_with_clarification:
            self.results_with_clarification[self.cfg.query_type] = {'num_queries': 0,
                                                            #  'queries_spread': 0,
                                                                'tp': 0, 
                                                                'fn': 0,
                                                                'fp': 0,
                                                                'act_correct': 0,
                                                                'act_total': 0,
                                                                'num_activity_queries': 0,
                                                                'num_object_queries': 0,
                                                                'num_activity_queries_carriedover': 0,
                                                                'num_object_queries_carriedover': 0,
                                                                'consistent_tp': {k:0 for k in inconsistency_masks.keys()},
                                                                'consistent_fn': {k:0 for k in inconsistency_masks.keys()},
                                                                'consistent_fp': {k:0 for k in inconsistency_masks.keys()},
                                                                'inconsistent_tp': {k:0 for k in inconsistency_masks.keys()},
                                                                'inconsistent_fn': {k:0 for k in inconsistency_masks.keys()},
                                                                'inconsistent_fp': {k:0 for k in inconsistency_masks.keys()}}
        self.results_with_clarification[self.cfg.query_type]['tp'] += int(torch.bitwise_and(torch.bitwise_and((dest_pred == data_in.dest_gt), data_in.changes_gt), obj_mask).sum())
        self.results_with_clarification[self.cfg.query_type]['fp'] += int(torch.bitwise_and(torch.bitwise_and((dest_pred != data_in.dest_gt), changes_pred), obj_mask).sum())
        self.results_with_clarification[self.cfg.query_type]['fn'] += int(torch.bitwise_and(torch.bitwise_and((dest_pred != data_in.dest_gt), data_in.changes_gt), obj_mask).sum())
        self.results_with_clarification[self.cfg.query_type]['num_queries'] += int(correction_masks.sum())
        self.results_with_clarification[self.cfg.query_type]['act_correct'] += int((pred_activities_corrected == data_in.expected_activities).sum())
        self.results_with_clarification[self.cfg.query_type]['num_activity_queries_carriedover'] += num_carried_act_queries
        self.results_with_clarification[self.cfg.query_type]['num_object_queries_carriedover'] += num_carried_obj_queries
        self.results_with_clarification[self.cfg.query_type]['act_total'] += int(torch.numel(pred_activities_corrected))
        self.results_with_clarification[self.cfg.query_type]['num_activity_queries'] += int(torch.bitwise_and(correction_masks, query_type==1).sum())
        self.results_with_clarification[self.cfg.query_type]['num_object_queries'] += int(torch.bitwise_and(correction_masks, query_type==2).sum())
        for consistency_thresh, inconsistency_mask in inconsistency_masks.items():
            consistent_obj_mask = torch.bitwise_and(obj_mask, torch.bitwise_not(inconsistency_mask))
            self.results_with_clarification[self.cfg.query_type]['consistent_tp'][consistency_thresh] += int(torch.bitwise_and(torch.bitwise_and((dest_pred == data_in.dest_gt), data_in.changes_gt), consistent_obj_mask).sum())
            self.results_with_clarification[self.cfg.query_type]['consistent_fp'][consistency_thresh] += int(torch.bitwise_and(torch.bitwise_and((dest_pred != data_in.dest_gt), changes_pred), consistent_obj_mask).sum())
            self.results_with_clarification[self.cfg.query_type]['consistent_fn'][consistency_thresh] += int(torch.bitwise_and(torch.bitwise_and((dest_pred != data_in.dest_gt), data_in.changes_gt), consistent_obj_mask).sum())
            inconsistent_obj_mask = torch.bitwise_and(obj_mask, inconsistency_mask)
            self.results_with_clarification[self.cfg.query_type]['inconsistent_tp'][consistency_thresh] += int(torch.bitwise_and(torch.bitwise_and((dest_pred == data_in.dest_gt), data_in.changes_gt), inconsistent_obj_mask).sum())
            self.results_with_clarification[self.cfg.query_type]['inconsistent_fp'][consistency_thresh] += int(torch.bitwise_and(torch.bitwise_and((dest_pred != data_in.dest_gt), changes_pred), inconsistent_obj_mask).sum())
            self.results_with_clarification[self.cfg.query_type]['inconsistent_fn'][consistency_thresh] += int(torch.bitwise_and(torch.bitwise_and((dest_pred != data_in.dest_gt), data_in.changes_gt), inconsistent_obj_mask).sum())
            assert self.results_with_clarification[self.cfg.query_type]['tp'] == self.results_with_clarification[self.cfg.query_type]['consistent_tp'][consistency_thresh] + self.results_with_clarification[self.cfg.query_type]['inconsistent_tp'][consistency_thresh], "TPs are not consistent"
            assert self.results_with_clarification[self.cfg.query_type]['fp'] == self.results_with_clarification[self.cfg.query_type]['consistent_fp'][consistency_thresh] + self.results_with_clarification[self.cfg.query_type]['inconsistent_fp'][consistency_thresh], "FPs are not consistent"
            assert self.results_with_clarification[self.cfg.query_type]['fn'] == self.results_with_clarification[self.cfg.query_type]['consistent_fn'][consistency_thresh] + self.results_with_clarification[self.cfg.query_type]['inconsistent_fn'][consistency_thresh], "FNs are not consistent"
        
        assert obj_mask.size(0) == 1, "This hack doesn't work with >1 batch size"
        assert (obj_mask[0,0,:] == obj_mask[0,5,:]).all(), "Just checking!!"
        

    def training_step(self, batch, batch_idx):
        batch_training_dropout = (torch.rand_like(batch['activity_ids'].float()) < self.cfg.activity_dropout_train).to(bool)
        final_mask = torch.bitwise_or(batch['activity_mask_drop'], batch_training_dropout)
        
        batch['activity_features'].masked_fill_(final_mask.unsqueeze(-1).repeat(1,1,batch['activity_features'].size()[-1]), 0)
        batch['activity_ids'].masked_fill_(batch['activity_mask_drop'], 0)

        results = self(batch)
        self.log_dict({f"Train loss/{k}": v for k, v in results['loss'].items()})
        self.log_dict({f"Train accuracy/{k}": v for k, v in results['accuracies'].items()})
        if isinstance(results['latents'], dict):
            self.log_dict({f"Train latents/{k}": v for k, v in results['latents'].items()})
        else:
            self.log('Train latents', results['latents'])
        try:
            self.log('Aux',self.object_activity_coembedding_module.auxiliary_accuracy)
        except Exception as e:
            print(e)
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
        batch['activity_features'].masked_fill_(batch['activity_mask_drop'].unsqueeze(-1).repeat(1,1,batch['activity_features'].size()[-1]), 0)
        results = self(batch)
        self.log_dict({f"Val accuracy/{k}": v for k, v in results['accuracies'].items()})
        
        self.reset_validation()
        self.evaluate_prediction(batch, num_steps=self.cfg.lookahead_steps)
        
        # Set early stopping metric
        self.log('Val_ES_accuracy',results['accuracies']['object_used'])

        try:
            self.log('Aux',self.object_activity_coembedding_module.auxiliary_accuracy)
        except Exception as e:
            print(e)

        self.reset_validation()
        return 

    def test_step(self, batch, batch_idx):
        batch['activity_features'].masked_fill_(batch['activity_mask_drop'].unsqueeze(-1).repeat(1,1,batch['activity_features'].size()[-1]), 0)
        if self.test_forward:
            results = self(batch)
            self.log_dict({f"Test loss/{k}": v for k, v in results['loss'].items()})
            self.log_dict({f"Test accuracy/{k}": v for k, v in results['accuracies'].items()})
        self.evaluate_prediction(batch, num_steps=self.cfg.lookahead_steps)
        return 

    def write_results(self, output_dir, common_data, suffix=''):
        
        activity_names=common_data['activities']
        node_classes=common_data['node_classes']
        std_for_activity = common_data['activity_stdev']

        os.makedirs(output_dir, exist_ok=True)
        self.results['f1_score'] = [None for _ in range(self.cfg.lookahead_steps)]
        quer_asked, quer_tot = 0,0 
        for step in range(self.cfg.lookahead_steps):
            self.results['moved_perc']['correct'][step] = self.results['moved']['correct'][step]/(self.results['moved']['total'][step]+1e-8) 
            self.results['moved_perc']['wrong'][step] = self.results['moved']['wrong'][step]/(self.results['moved']['total'][step]+1e-8) 
            self.results['moved_perc']['missed'][step] = self.results['moved']['missed'][step]/(self.results['moved']['total'][step]+1e-8) 
            
            self.results['precision_perc']['with_destination'][step] = self.results['precision']['correct'][step]/(self.results['precision']['total'][step]+1e-8) 
            self.results['precision_perc']['only_object'][step] = (self.results['precision']['correct'][step] + self.results['precision']['wrong_destination'][step])/(self.results['precision']['total'][step]+1e-8) 
            if step >= 2:
                self.results['precision_perc_lenient']['with_destination'][step] = self.results['precision_lenient']['correct'][step]/(self.results['precision_lenient']['total'][step]+1e-8) 
                self.results['precision_perc_lenient']['only_object'][step] = (self.results['precision_lenient']['correct'][step] + self.results['precision_lenient']['wrong_destination'][step])/(self.results['precision_lenient']['total'][step]+1e-8) 
            
            self.results['f1_score'][step] = 2*self.results['moved_perc']['correct'][step]*self.results['precision_perc']['with_destination'][step]/(self.results['moved_perc']['correct'][step]+self.results['precision_perc']['with_destination'][step]+1e-8)

            self.results['unmoved_perc']['fp'][step] = self.results['unmoved']['fp'][step]/(self.results['unmoved']['total'][step]+1e-8) 
            self.results['unmoved_perc']['tn'][step] = self.results['unmoved']['tn'][step]/(self.results['unmoved']['total'][step]+1e-8) 
            self.results['activity_perc']['correct'][step] = self.results['activity']['correct'][step]/(self.results['activity']['total'][step]+1e-8)
            self.results['activity_perc']['wrong'][step] = self.results['activity']['wrong'][step]/(self.results['activity']['total'][step]+1e-8)
        
            quer_asked += int(self.results['queries'][step]['num_asked'])
            quer_tot += int(self.results['queries'][step]['num_predictions'])
            self.results['queries'][step]['num_asked'] = quer_asked
            self.results['queries'][step]['num_predictions'] = quer_tot
            self.results['queries'][step]['perc_asked'] = quer_asked/(self.results['queries'][0]['num_predictions']+1e-8)

        assert len(activity_names) == self.cfg.n_activities
        moved_by_activity_list = deepcopy(self.results['moved_by_activity'])
        self.results['moved_by_activity'] = {k:{n:{'correct':0, 'wrong':0, 'missed':0, 'total':0, 'fp':0} for n in activity_names} for k in ['original']}
        for act_num in range(self.cfg.n_activities):
            act_name = activity_names[act_num]
            self.results['moved_by_activity']['original'][act_name] = moved_by_activity_list[act_num]
        
        activity_confusion_list = deepcopy(self.results['activity_confusion'])
        fig, axs = plt.subplots()
        sums = np.expand_dims(np.array(activity_confusion_list).sum(-1), -1)
        axs.imshow(np.array(activity_confusion_list)/(sums+1e-8))
        axs.set_xticks(np.arange(self.cfg.n_activities))
        axs.set_xticklabels([wrap_str(a) for a in activity_names])
        axs.set_yticks(np.arange(self.cfg.n_activities))
        axs.set_yticklabels([wrap_str(a) for a in activity_names])
        fig.tight_layout()
        fig.set_size_inches(3+self.cfg.n_activities, 3+self.cfg.n_activities)
        fig.savefig(os.path.join(output_dir,f'activity_confusion.png'))

        self.results['activity_confusion'] = {}
        for gt_act_num in range(self.cfg.n_activities):
            gt_act_name = activity_names[gt_act_num]
            self.results['activity_confusion'][gt_act_name] = {}
            for pred_act_num in range(self.cfg.n_activities):
                self.results['activity_confusion'][gt_act_name][activity_names[pred_act_num]] = int(activity_confusion_list[gt_act_num][pred_act_num])

        json.dump(self.results, open(os.path.join(output_dir,f'test_evaluation_{suffix}.json'),'w'), indent=4)
        node_classes_in_order = [node_classes[n.item()] for n in self.node_idxs.int()]
        obj_time_inconsistency_masks = torch.bitwise_and(self.object_usage_frequency.unsqueeze(0).to('cuda') > self.cfg.consistency_thresholds[0], self.object_usage_frequency.unsqueeze(0).to('cuda') < (60-self.cfg.consistency_thresholds[0]))
        torch.save({'activity_consistencies':std_for_activity, 
                    'node_classes':node_classes_in_order, 
                    'obj_time_inconsistency':obj_time_inconsistency_masks,
                    'data':self.result_data}, os.path.join(output_dir,f'raw_results_{suffix}.pt'))
        json.dump(get_metrics(self.result_data, node_classes=node_classes_in_order, activity_consistencies=std_for_activity), open(os.path.join(output_dir,f'test_evaluation_splits.json'),'w'), indent=4)
        for qt in self.results_with_clarification.keys():
            self.results_with_clarification[qt]['precision'] = (self.results_with_clarification[qt]['tp'])/(self.results_with_clarification[qt]['tp']+self.results_with_clarification[qt]['fp']+1e-8)
            self.results_with_clarification[qt]['recall'] = (self.results_with_clarification[qt]['tp'])/(self.results_with_clarification[qt]['tp']+self.results_with_clarification[qt]['fn']+1e-8)
            self.results_with_clarification[qt]['f1'] = 2*self.results_with_clarification[qt]['precision']*self.results_with_clarification[qt]['recall']/(self.results_with_clarification[qt]['precision']+self.results_with_clarification[qt]['recall']+1e-8)
        json.dump(self.results_with_clarification, open(os.path.join(output_dir,f'test_evaluation_clarifications_{suffix}.json'),'w'), indent=4)

        plt.close('all')
        self.reset_validation()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

