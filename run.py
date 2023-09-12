import sys
import yaml
import json
import os
import argparse
import time
sys.path.append('helpers')
sys.path.append('models')

from adict import adict
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from MultiModalUserTracking import MultiModalUserTrackingModule
from loader_sequential import RoutinesDataset
from encoders import TimeEncodingOptions

import random
from numpy import random as nrandom
random.seed(23435)
nrandom.seed(23435)

def compare_models(model_1, model_2, num_params_expected=None):
    models_differ = 0
    num_params = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            num_params += torch.numel(key_item_1[1])
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('!!! Mismatch found at', key_item_1[0])
            else:
                raise Exception
    assert num_params_expected == num_params, f"Expected {num_params_expected} got {num_params}"
    if models_differ == 0:
        print(f'\nModels match perfectly! :) All {num_params} parameters!!!\n')

def run(data, group=None, cfg = {}, tags=[], logs_dir='logs', original_model=False, model_generator=MultiModalUserTrackingModule, checkpoint_dir=None, train_only=False):
    
    t=time.localtime()
    timestr='{:02d}{:02d}{:02d}{:02d}'.format(t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min)

    cfg.update(data.params)
    model_configs = adict(cfg)
    output_dir = logs_dir
    os.makedirs(output_dir, exist_ok=True)

    wandb_logger = WandbLogger(name=cfg['NAME'], group = group, tags = tags, settings=wandb.Settings(start_method="fork"), mode='disabled')
    wandb_logger.experiment.config.update(cfg)
    
    if checkpoint_dir is None:
    
        epochs = cfg['epochs']

        ckpt_callback = ModelCheckpoint(dirpath=output_dir)

        train_loader = data.get_train_loader()
        val_loader = data.get_val_loader()        

        if model_configs.phased_training:
            ckpt_callback = ModelCheckpoint(dirpath=output_dir+'_pretr')
            model_configs.loss_latent_pred = False
            model_configs.loss_object_pred = False
            model_configs.loss_activity_pred = False

        early_stop_callback = EarlyStopping(monitor="Val_ES_accuracy", patience=40, verbose=False, mode="max")
        trainer = Trainer(accelerator='gpu', devices = torch.cuda.device_count(), logger=wandb_logger, max_epochs=epochs, log_every_n_steps=1, callbacks=[ckpt_callback, early_stop_callback], check_val_every_n_epoch=5)
        model = model_generator(model_configs = model_configs, original_model = original_model)
        model.set_object_consistency(data.get_object_consistency())
        model.cfg.query_types = []
        trainer.fit(model, train_loader, val_loader)

        if model_configs.phased_training:
            model_configs.loss_latent_pred = True
            model_configs.loss_object_pred = True
            model_configs.loss_activity_pred = True

            ckpt_callback = ModelCheckpoint(dirpath=output_dir)
            early_stop_callback = EarlyStopping(monitor="Val_ES_accuracy", patience=40, verbose=False, mode="max")
            trainer = Trainer(accelerator='gpu', devices = torch.cuda.device_count(), logger=wandb_logger, max_epochs=epochs, log_every_n_steps=1, callbacks=[ckpt_callback, early_stop_callback], check_val_every_n_epoch=5)
            model = model_generator(model_configs = model_configs, original_model = original_model)
            model.set_object_consistency(data.get_object_consistency())
            trainer.fit(model, train_loader, val_loader)

        torch.save(model.state_dict(), os.path.join(output_dir,'weights.pt'))
        model_check = model_generator(model_configs = model_configs, original_model = original_model)
        model_check.load_state_dict(torch.load(os.path.join(output_dir,'weights.pt')))
        compare_models(model, model_check, num_params_expected=sum(param.numel() for param in model.parameters()))
        print('Outputs saved at ',output_dir)

    else:
        assert not train_only, "Cannot train only with --read_ckpt"
        checkpoint_file = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        assert len(checkpoint_file) == 1, f"None or many checkpoint files in directory {checkpoint_dir}: {checkpoint_file}"
        checkpoint_file = checkpoint_file[0]
        trainer = Trainer(accelerator='gpu', devices = torch.cuda.device_count(), logger=wandb_logger)
        config_file = os.path.join(checkpoint_dir, 'config.json')
        if os.path.exists(config_file):
            model_configs.update(json.load(open(config_file)))
        model = model_generator.load_from_checkpoint(os.path.join(checkpoint_dir, checkpoint_file), model_configs = model_configs, original_model = original_model)
        json.dump(model_configs, open(os.path.join(output_dir, f'config_{timestr}.json'), 'w'), indent=4)

    if not train_only:
        model.set_object_consistency(data.get_object_consistency())
        eval_dir = os.path.join(output_dir,'test_evals_'+timestr)
        model.test_forward = False
        model.cfg.query_types = cfg['query_types']
        for query_usefulness_metric in ['information_gain']:
            print(f"Starting evaluation for with {query_usefulness_metric} metric")
            model.cfg.query_usefulness_metric = query_usefulness_metric

            os.makedirs(eval_dir, exist_ok=True)
            model.cfg.query_usefullness_thresh = cfg['query_usefullness_thresh'][query_usefulness_metric]
            trainer.test(model, data.get_test_loader())
            model.write_results(eval_dir,
                                common_data = data.common_data,
                                suffix=query_usefulness_metric,
                                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='data/HouseholdVariations/persona1_', help='Path where the data lives. Must contain routines, info and classes json files.')
    parser.add_argument('--cfg', type=str, help='Name of config file.')
    parser.add_argument('--train_days', type=int, help='Number of routines to train on.')
    parser.add_argument('--name', type=str, default='default_100', help='Name of run.')
    parser.add_argument('--tags', type=str, help='Tags for the run separated by a comma \',\'')
    parser.add_argument('--ckpt_dir', type=str, help='Path to checkpoint file')
    parser.add_argument('--read_ckpt', action='store_true')
    parser.add_argument('--original_model', action='store_true')
    parser.add_argument('--use_bert', action='store_true')
    parser.add_argument('--train_only', action='store_true')
    parser.add_argument('--logs_dir', type=str, default='./logs', help='Path to store putputs.')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--activity_availability', default=100, type=int)
    parser.add_argument('--batch_size', type=int)

    args = parser.parse_args()
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    
    with open('config/default.yaml') as f:
        cfg = yaml.safe_load(f)

    if args.cfg is not None:
        with open(os.path.join('config',args.cfg)+'.yaml') as f:
            cfg.update(yaml.safe_load(f))
    if args.name is not None:
        cfg['NAME'] = args.name
    if args.epochs is not None:
        cfg['epochs'] = args.epochs

    for k in args.__dict__: 
        if args.__dict__[k] is not None:
            cfg[k] = args.__dict__[k]
    if args.activity_availability is not None:
        cfg['activity_dropout_prob'] = 1-(args.activity_availability/100)

    cfg['MAX_TRAINING_SAMPLES'] = args.train_days
    args.tags = args.tags.split(',') if args.tags is not None else []

    if cfg['NAME'] is None:
        cfg['NAME'] = os.path.basename(args.path)+'_trial'

    if 'processed_seqLM' not in args.path.split('/')[-1]: args.path = os.path.join(args.path, 'processed_seqLM_coarse')
    cfg['DATA_INFO'] = json.load(open(os.path.join(args.path, 'common_data.json')))

    time_options = TimeEncodingOptions(cfg['DATA_INFO']['weeekend_days'] if 'weeekend_days' in cfg['DATA_INFO'].keys() else None)
    time_encoding = time_options('sine_informed')

    data = RoutinesDataset(data_path=args.path,
                            time_encoder=time_encoding, 
                            batch_size=cfg['batch_size'],
                            use_bert=args.use_bert,
                            activity_dropout=cfg['activity_dropout_prob'])

    group = args.path.split('/')[-2]
    output_dir = os.path.join(args.logs_dir, group, cfg['NAME'])
    if args.read_ckpt:
        run(data, group=group, cfg=cfg, tags = args.tags, logs_dir=output_dir, original_model=cfg['original_model'], model_generator=MultiModalUserTrackingModule, checkpoint_dir=args.ckpt_dir if args.ckpt_dir is not None else output_dir, train_only=args.train_only)

    else:
        run(data, group=group, cfg=cfg, tags = args.tags, logs_dir=output_dir, original_model=cfg['original_model'], model_generator=MultiModalUserTrackingModule, train_only=args.train_only)
