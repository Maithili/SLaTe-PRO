import os
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from adict import adict

stdev_threshes = [2.0, 1.0, 0.5, 0.1]

color_red_to_green = ['k', 'r', 'y', 'g', 'g', 'c']

def make_adict(diction):
    adict_ = {}
    for key, value in diction.items():
        if isinstance(value, dict):
            adict_[key] = make_adict(value)
        else:
            adict_[key] = value
    return adict(adict_)


def wrap_str(label, l=8):
    wrapped = ''
    if label is None: return 'None'
    if len(label) < l: return label
    for i in range(math.floor(len(label)/l)):
        wrapped += label[l*i:l*(i+1)] + '\n'
    wrapped += label[l*(i+1):]
    return wrapped


objects_by_activity = {
    "brushing_teeth": [
        "tooth_paste"
    ],
    "showering": [
        "hairbrush",
        "hairdryer"
    ],
    "watching_tv": [
        "remote_control"
    ],
    "breakfast": [
        "bowl",
        "coffee_filter",
        "cup",
        "food_cereal",
        "food_oatmeal",
        "ground_coffee",
        "juice",
        "milk",
        "mug",
        "spoon"
    ],
    "wash_dishes_breakfast": [
        "bowl",
        "coffee_filter",
        "cup",
        "mug",
        "spoon"
    ],
    "computer_work": [
        "headset",
        "notebook",
        "pen"
    ],
    "lunch": [
        "food_bread",
        "food_cheese",
        "food_jam",
        "food_peanut_butter",
        "knife",
        "plate"
    ],
    "wash_dishes_lunch": [
        "knife",
        "plate"
    ],
    "listening_to_music": [
        "cd",
        "radio"
    ],
    "playing_music": [
        "instrument_guitar"
    ],
    "reading": [
        "book"
    ],
    "dinner": [
        "cookingpot",
        "dry_pasta",
        "food_chicken",
        "food_rice",
        "food_vegetable",
        "fork",
        "fryingpan",
        "oil",
        "plate",
        "spoon"
    ],
    "wash_dishes_dinner": [
        "cookingpot",
        "fryingpan",
        "plate",
        "spoon"
    ],
    "socializing": [
        "chessboard",
        "coffee",
        "coffee_cup",
        "cup",
        "cutting_board",
        "deck_of_cards",
        "food_cheese",
        "food_donut",
        "tea",
        "wine",
        "wine_glass"
    ],
    "getting_dressed": [
        "razor",
        "shaving_cream"
    ],
    "cleaning": [
        "cleaning_solution",
        "vacuum_cleaner",
        "washcloth"
    ],
    "leave_home": [
        "clothes_jacket"
    ],
    "come_home": [
        "clothes_jacket",
        "groceries",
        "mail"
    ],
    "take_out_trash": [
        "food_apple",
        "food_donut",
        "trashbag"
    ],
    "taking_medication": [
        "drinking_glass",
        "painkillers"
    ]
}

color_map = {
 "sleeping" : sns.color_palette(palette='pastel')[0], 
 "sleep" : sns.color_palette(palette='pastel')[0], 
 "nap" : sns.color_palette(palette='pastel')[0], 
 "wake_up" : sns.color_palette()[0], 
 "breakfast" : sns.color_palette()[1], 
 "lunch" : sns.color_palette()[2], 
 "computer_work" : sns.color_palette()[3], 
 "reading" : sns.color_palette()[4],
 "cleaning" : sns.color_palette()[5], 
 "laundry" : sns.color_palette()[6], 
 "leave_home" : sns.color_palette()[7], 
 "come_home" : sns.color_palette()[8], 
 "socializing" : sns.color_palette(palette='dark')[0], 
 "taking_medication" : sns.color_palette(palette='dark')[1], 
 "vaccuum_cleaning" : sns.color_palette(palette='dark')[2], 
 "getting_dressed" : sns.color_palette(palette='dark')[3],
 "dinner" : sns.color_palette(palette='dark')[4], 
 "kitchen_cleaning" : sns.color_palette(palette='dark')[5],
 "take_out_trash" : sns.color_palette(palette='dark')[6],
 "wash_dishes" : sns.color_palette(palette='dark')[7],
 "wash_dishes_breakfast" : sns.color_palette(palette='dark')[7],
 "wash_dishes_lunch" : sns.color_palette(palette='dark')[7],
 "wash_dishes_dinner" : sns.color_palette(palette='dark')[7],
 "playing_music" : sns.color_palette(palette='dark')[8],
 "diary_logging" : sns.color_palette(palette='dark')[9],
 "brushing_teeth" : sns.color_palette(palette='pastel')[1], 
 "showering" : sns.color_palette(palette='pastel')[2], 
 "leaving_home_fast" : sns.color_palette(palette='pastel')[3], 
 "watching_tv" : sns.color_palette(palette='pastel')[4],
 "talk_on_phone" : sns.color_palette(palette='pastel')[5], 
 "online_meeting" : sns.color_palette(palette='pastel')[6], 
 "going_to_the_bathroom" : sns.color_palette(palette='pastel')[7],
 "listening_to_music" : sns.color_palette(palette='pastel')[8],
 "Idle" : [1,1,1,0.5],
 "Unknown" : [0,0,0,0],
}

color_palette = sns.color_palette()+sns.color_palette(palette='dark')
color_palette = color_palette * 10

def get_metrics(results, node_classes=None, activity_consistencies=[]):

    difficult_objects = {}

    stdev_for_node_idx = [0 for _ in node_classes]
    for activity, stdev in activity_consistencies.items():
        if activity in objects_by_activity:
            for object_with_std in objects_by_activity[activity]:
                if stdev_for_node_idx[node_classes.index(object_with_std)] < stdev:
                    stdev_for_node_idx[node_classes.index(object_with_std)] = stdev
    print('inconsistent_activity_objects', node_classes, stdev_for_node_idx)


    name_thresh = [(n,t) for n,t in zip(['_parts_a','_parts_b','_parts_c','_parts_d'], stdev_threshes)]
    for name, thresh in name_thresh:
        difficult_objects[name] = torch.tensor([0 for _ in node_classes]).to(bool).to('cuda')
    for nidx, stdev in enumerate(stdev_for_node_idx):
        for name, thresh in name_thresh:
            if stdev > thresh:
                difficult_objects[name][nidx] = True
                break
            
    name_thresh = [(n,t) for n,t in zip(['_cumul_a','_cumul_b','_cumul_c','_cumul_d'], stdev_threshes)]
    for name, thresh in name_thresh:
        difficult_objects[name] = torch.tensor([0 for _ in node_classes]).to(bool).to('cuda')
        for nidx, stdev in enumerate(stdev_for_node_idx):
            if stdev > thresh:
                difficult_objects[name][nidx] = True

    query_types = results['num_queries'].keys()

    results = make_adict(results)
    data_steps = len(results.relocation_distributions)
    metrics = {'relocation':{m:{'tp':{s:0 for s in range(data_steps)}, 'fp':{s:0 for s in range(data_steps)}, 'fn':{s:0 for s in range(data_steps)}, 'result':{s:0 for s in range(data_steps)}} 
                    for m in ['precision', 'recall']},
                'activity':{'correct':{s:0 for s in range(data_steps)}, 'total':{s:0 for s in range(data_steps)}, 'correct_fraction':{s:0 for s in range(data_steps)}},
                'relocation_clarified':{qt:{'precision':{'tp':0, 'fp':0, 'fn':0, 'result':0}, 'recall':{'tp':0, 'fp':0, 'fn':0, 'result':0}, 'num_queries':results.num_queries[qt], 'f1':0} for qt in query_types},
                'activity_clarified':{qt:{'correct':0, 'total':0, 'correct_fraction':0} for qt in query_types}}
    metrics.update({'relocation_difficult'+suffix:{m:{'tp':{s:0 for s in range(data_steps)}, 'fp':{s:0 for s in range(data_steps)}, 'fn':{s:0 for s in range(data_steps)}, 'result':{s:0 for s in range(data_steps)}} 
                    for m in ['precision', 'recall']} for suffix in difficult_objects.keys()})
    metrics.update({'relocation_easy'+suffix:{m:{'tp':{s:0 for s in range(data_steps)}, 'fp':{s:0 for s in range(data_steps)}, 'fn':{s:0 for s in range(data_steps)}, 'result':{s:0 for s in range(data_steps)}} 
                    for m in ['precision', 'recall']} for suffix in difficult_objects.keys()})
    metrics.update({'relocation_clarified_difficult'+suffix:{qt:{'precision':{'tp':0, 'fp':0, 'fn':0, 'result':0}, 'recall':{'tp':0, 'fp':0, 'fn':0, 'result':0}, 'num_queries':results.num_queries[qt], 'f1':0} 
                                                      for qt in query_types} for suffix in difficult_objects.keys()})
    metrics.update({'relocation_clarified_easy'+suffix:{qt:{'precision':{'tp':0, 'fp':0, 'fn':0, 'result':0}, 'recall':{'tp':0, 'fp':0, 'fn':0, 'result':0}, 'num_queries':results.num_queries[qt], 'f1':0} 
                                                      for qt in query_types} for suffix in difficult_objects.keys()})

    metrics['relocation_clarified']['query_step'] = results.query_step

    assert data_steps <= len(results.relocation_locations_gt), "Not enough ground truth steps"

    for pred_step in range(data_steps):
        gt_step = {'recall': pred_step, 'precision': pred_step}
        
        obj_mask = results.obj_mask[pred_step].to(bool)
        dest_pred = results.relocation_distributions[pred_step].argmax(dim=-1)
        changes_pred = (dest_pred != results.reference_locations)
        combination_metrics = ['relocation']
        for m in ['recall', 'precision']:
            dest_gt = results.relocation_locations_gt[gt_step[m]]
            assert torch.numel(dest_gt) > 0, f"No ground truth for step{gt_step[m]} among {results.relocation_locations_gt}"
            changes_gt = (dest_gt != results.reference_locations)
            metrics['relocation'][m]['tp'][pred_step] += ((dest_pred == dest_gt) & changes_gt & changes_pred & obj_mask).sum().item()
            metrics['relocation'][m]['fp'][pred_step] += ((dest_pred != dest_gt) & changes_pred & obj_mask).sum().item()
            metrics['relocation'][m]['fn'][pred_step] += ((dest_pred != dest_gt) & changes_gt & obj_mask).sum().item()

            for suffix in difficult_objects.keys():
                metrics['relocation_difficult' + suffix][m]['tp'][pred_step] += ((dest_pred == dest_gt) & changes_gt & changes_pred & obj_mask & difficult_objects[suffix]).sum().item()
                metrics['relocation_difficult' + suffix][m]['fp'][pred_step] += ((dest_pred != dest_gt) & changes_pred & obj_mask & difficult_objects[suffix]).sum().item()
                metrics['relocation_difficult' + suffix][m]['fn'][pred_step] += ((dest_pred != dest_gt) & changes_gt & obj_mask & difficult_objects[suffix]).sum().item()
                metrics['relocation_easy' + suffix][m]['tp'][pred_step] += ((dest_pred == dest_gt) & changes_gt & changes_pred & obj_mask & torch.bitwise_not(difficult_objects[suffix])).sum().item()
                metrics['relocation_easy' + suffix][m]['fp'][pred_step] += ((dest_pred != dest_gt) & changes_pred & obj_mask & torch.bitwise_not(difficult_objects[suffix])).sum().item()
                metrics['relocation_easy' + suffix][m]['fn'][pred_step] += ((dest_pred != dest_gt) & changes_gt & obj_mask & torch.bitwise_not(difficult_objects[suffix])).sum().item()
                if m == 'recall':
                    combination_metrics.append('relocation_difficult' + suffix)
                    combination_metrics.append('relocation_easy' + suffix)

        for r in combination_metrics:
            if 'f1' not in metrics[r]:
                metrics[r]['f1'] = {s:0 for s in range(data_steps)}
            metrics[r]['precision']['result'][pred_step] = metrics[r]['precision']['tp'][pred_step] / \
                                                                    (metrics[r]['precision']['tp'][pred_step] + metrics[r]['precision']['fp'][pred_step] + 1e-8)
            metrics[r]['recall']['result'][pred_step] = metrics[r]['recall']['tp'][pred_step] / \
                                                                (metrics[r]['recall']['tp'][pred_step] + metrics[r]['recall']['fn'][pred_step] + 1e-8)
            metrics[r]['f1'][pred_step] = 2 * metrics[r]['precision']['result'][pred_step] * metrics[r]['recall']['result'][pred_step] / \
                                                        (metrics[r]['precision']['result'][pred_step] + metrics[r]['recall']['result'][pred_step] + 1e-8)

        activities_pred = results.activity_distributions[pred_step].argmax(dim=-1)
        activities_gt = results.activity_gt[pred_step]
        metrics['activity']['correct'][pred_step] += (activities_pred == activities_gt).sum().item()
        metrics['activity']['total'][pred_step] += torch.numel(activities_gt)
        metrics['activity']['correct_fraction'][pred_step] = metrics['activity']['correct'][pred_step] / metrics['activity']['total'][pred_step]
        
        if pred_step == results.query_step:
            for qt in query_types:
                dest_pred_clarified = results.relocation_distributions_clarified[qt].argmax(dim=-1)
                changes_pred_clarified = (dest_pred_clarified != results.reference_locations)
                combination_metrics = ['relocation_clarified']
                for m in ['recall', 'precision']:
                    dest_gt = results.relocation_locations_gt[gt_step[m]]
                    changes_gt = (dest_gt != results.reference_locations)
                    metrics['relocation_clarified'][qt][m]['tp'] += ((dest_pred_clarified == dest_gt) & changes_gt & changes_pred_clarified & obj_mask).sum().item()
                    metrics['relocation_clarified'][qt][m]['fp'] += ((dest_pred_clarified != dest_gt) & changes_pred_clarified & obj_mask).sum().item()
                    metrics['relocation_clarified'][qt][m]['fn'] += ((dest_pred_clarified != dest_gt) & changes_gt & obj_mask).sum().item()
                    
                    for suffix in difficult_objects.keys():
                        metrics['relocation_clarified_difficult'+suffix][qt][m]['tp'] += ((dest_pred_clarified == dest_gt) & changes_gt & changes_pred_clarified & obj_mask & difficult_objects[suffix]).sum().item()
                        metrics['relocation_clarified_difficult'+suffix][qt][m]['fp'] += ((dest_pred_clarified != dest_gt) & changes_pred_clarified & obj_mask & difficult_objects[suffix]).sum().item()
                        metrics['relocation_clarified_difficult'+suffix][qt][m]['fn'] += ((dest_pred_clarified != dest_gt) & changes_gt & obj_mask & difficult_objects[suffix]).sum().item()
                        
                        metrics['relocation_clarified_easy'+suffix][qt][m]['tp'] += ((dest_pred_clarified == dest_gt) & changes_gt & changes_pred_clarified & obj_mask & torch.bitwise_not(difficult_objects[suffix])).sum().item()
                        metrics['relocation_clarified_easy'+suffix][qt][m]['fp'] += ((dest_pred_clarified != dest_gt) & changes_pred_clarified & obj_mask & torch.bitwise_not(difficult_objects[suffix])).sum().item()
                        metrics['relocation_clarified_easy'+suffix][qt][m]['fn'] += ((dest_pred_clarified != dest_gt) & changes_gt & obj_mask & torch.bitwise_not(difficult_objects[suffix])).sum().item()
                        
                        if m == 'recall':
                            combination_metrics.append('relocation_clarified_difficult' + suffix)
                            combination_metrics.append('relocation_clarified_easy' + suffix)
                        
                for r in combination_metrics:
                    metrics[r][qt]['precision']['result'] = metrics[r][qt]['precision']['tp'] / \
                                                                            (metrics[r][qt]['precision']['tp'] + metrics[r][qt]['precision']['fp'] + 1e-8)
                    metrics[r][qt]['recall']['result'] = metrics[r][qt]['recall']['tp'] / \
                                                                        (metrics[r][qt]['recall']['tp'] + metrics[r][qt]['recall']['fn'] + 1e-8)
                    metrics[r][qt]['f1'] = 2 * metrics[r][qt]['precision']['result'] * metrics[r][qt]['recall']['result'] / \
                                                                (metrics[r][qt]['precision']['result'] + metrics[r][qt]['recall']['result'] + 1e-8)
                activities_pred_clarified = results.activity_distributions_clarified[qt]
                metrics['activity_clarified'][qt]['correct'] += (activities_pred_clarified == activities_gt).sum().item()
                metrics['activity_clarified'][qt]['total'] += torch.numel(activities_gt)
                metrics['activity_clarified'][qt]['correct_fraction'] = metrics['activity_clarified'][qt]['correct'] / metrics['activity_clarified'][qt]['total']

    return metrics
