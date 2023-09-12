import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from adict import adict

# color_red_to_green = ["#ffffff", "#bc4b51", "#fbc4ab", "#90a955", "#31572c"]

# stdev_threshes = [2.0, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
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

## All objects even if used across variations
obj_in_multiple_variations_all_all = [
        "bowl",
        "coffee_filter",
        "book", 
        "cup",
        "food_cereal",
        "food_oatmeal",
        "ground_coffee",
        "juice",
        "milk",
        "mug",
        "spoon",
        "headset",
        "notebook",
        "pen",
        "food_cheese",
        "food_peanut_butter",
        "plate",
        "knife",
        "food_jam",
        "food_bread",
        "cd",
        "radio",
        "oil",
        "fryingpan",
        "cookingpot",
        "dry_pasta",
        "food_chicken",
        "fork",
        "food_rice",
        "food_vegetable",
        "food_donut",
        "chessboard",
        "deck_of_cards",
        "cup",
        "coffee_cup",
        "coffee",
        "wine_glass",
        "wine",
        "cutting_board",
        "cleaning_solution",
        "washcloth",
        "vacuum_cleaner",
]

## Only ones that differ between variations
obj_in_multiple_variations = [
        "coffee_cup", 
        "dry_pasta", 
        "book", 
        "juice", 
        "food_chicken", 
        "ground_coffee", 
        "food_oatmeal", 
        "cutting_board", 
        "notebook", 
        "tea", 
        "radio", 
        "coffee_filter", 
        "washcloth", 
        "vacuum_cleaner", 
        "food_jam", 
        "food_donut", 
        "cup", 
        "wine", 
        "mug", 
        "food_rice", 
        "food_cereal", 
        "pen", 
        "food_vegetable", 
        "coffee", 
        "shaving_cream", 
        "wine_glass", 
        "food_peanut_butter", 
        "cleaning_solution", 
        "razor", 
        "headset", 
        "cd", 
        "food_cheese"
    ]

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

def get_metrics(results, node_classes=None, activity_consistencies=[], obj_time_inconsistency=None, split_objects_by=None):

    s_batch, s_sequence, s_obj = results['reference_locations'].size()

    # query_threshes = list(results['num_queries'].keys())
    # query_thresh = min(query_threshes)
    # query_thresh_control = max(query_threshes)
    
    difficult_objects = {}

    difficult_objects1 = [1 if obj in obj_in_multiple_variations else 0 for obj in node_classes]
    difficult_objects1 = torch.tensor(difficult_objects1).bool().to('cuda')
    difficult_objects['_1'] = difficult_objects1


    stdev_for_node_idx = [0 for _ in node_classes]
    for activity, stdev in activity_consistencies.items():
        if activity in objects_by_activity:
            for object_with_std in objects_by_activity[activity]:
                if stdev_for_node_idx[node_classes.index(object_with_std)] < stdev:
                    stdev_for_node_idx[node_classes.index(object_with_std)] = stdev
    print('inconsistent_activity_objects', node_classes, stdev_for_node_idx)


    name_thresh = [(n,t) for n,t in zip(['_2a','_2b','_2c','_2d'], stdev_threshes)]
    for name, thresh in name_thresh:
        difficult_objects[name] = torch.tensor([0 for _ in node_classes]).to(bool).to('cuda')
    for nidx, stdev in enumerate(stdev_for_node_idx):
        for name, thresh in name_thresh:
            if stdev > thresh:
                difficult_objects[name][nidx] = True
                break
            
    name_thresh = [(n,t) for n,t in zip(['_3a','_3b','_3c','_3d'], stdev_threshes)]
    for name, thresh in name_thresh:
        difficult_objects[name] = torch.tensor([0 for _ in node_classes]).to(bool).to('cuda')
        for nidx, stdev in enumerate(stdev_for_node_idx):
            if stdev > thresh:
                difficult_objects[name][nidx] = True
                
    difficult_objects4 = torch.tensor([True])
    if obj_time_inconsistency is not None:
        difficult_objects4 = obj_time_inconsistency.squeeze(0)[:s_sequence,:]
        print(f"Object Time Inconsistency: {difficult_objects4.sum()}/{torch.numel(difficult_objects4)}")
        difficult_objects['_4'] = difficult_objects4


    difficult_objects5 = [1 if obj in obj_in_multiple_variations_all_all else 0 for obj in node_classes]
    difficult_objects5 = torch.tensor(difficult_objects5).bool().to('cuda')
    difficult_objects['_5'] = difficult_objects5

    difficult_objects['_all'] = difficult_objects1 & difficult_objects['_3b'] & difficult_objects4 & difficult_objects5
    difficult_objects['_any'] = difficult_objects1 | difficult_objects['_3b'] | difficult_objects4 | difficult_objects5

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

    # assert all([v==0 for v in results.num_queries[query_thresh_control].values()]), "Control query thresh should have no queries"
    assert data_steps <= len(results.relocation_locations_gt), "Not enough ground truth steps"

    for pred_step in range(data_steps):
        gt_step = {'recall': pred_step, 'precision': pred_step}
        
        # split_objects_by in 'taking_out' ,'putting_away', None
        taking_out_mask = (results.reference_locations == results.reference_locations[:,0,:].unsqueeze(1))
        putting_away_mask = (results.reference_locations != results.reference_locations[:,0,:].unsqueeze(1))
        obj_mask = results.obj_mask[pred_step].to(bool)
        if split_objects_by == 'taking_out': obj_mask=obj_mask & taking_out_mask
        if split_objects_by == 'putting_away': obj_mask=obj_mask & putting_away_mask
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



def print_snapshots(node_classes, activity_names, output_dir, snapshots_data, snapshots, snapshots_queries, query_steps=0, std_range_by_activity=None, activity_consistencies=None):

    os.makedirs(output_dir, exist_ok=True)

    for i,act in enumerate(activity_names):
        if act not in color_map: color_map[act] = color_palette[i%len(color_palette)]

    stdev_for_node_idx = [0 for _ in node_classes]
    for activity, stdev in activity_consistencies.items():
        if activity in objects_by_activity:
            for object_with_std in objects_by_activity[activity]:
                if stdev_for_node_idx[node_classes.index(object_with_std)] < stdev:
                    stdev_for_node_idx[node_classes.index(object_with_std)] = stdev

    stdev_for_node_idx = [n for n,om in zip(stdev_for_node_idx, snapshots_data[0][-2]) if om ]

    figdense, axsdense = plt.subplots(1, len(snapshots_data))
    fig, axs = plt.subplots(1, len(snapshots_data))
    num_obj_total = 0
    for i,(dense_gt, snapshot_gt, snapshot_gt_alpha, activities_gt, obj_mask, node_idxs) in enumerate(snapshots_data):
        yv, xv = np.meshgrid(np.arange(snapshot_gt.size()[0]), np.arange(snapshot_gt.size()[1]))
        if activity_consistencies is not None:
            for obj in range(snapshot_gt.size()[1]):
                color = 'w'
                for icol,thresh in enumerate(stdev_threshes):
                    if stdev_for_node_idx[obj] > thresh:
                        color = color_red_to_green[icol]
                        break
                axs[i].plot([obj,obj], [0,snapshot_gt.size()[0]], color=color, alpha=0.5, linewidth=0.5)
        else:
            for obj in range(snapshot_gt.size()[1]):
                axs[i].plot([obj,obj], [0,snapshot_gt.size()[0]], color='k', alpha=0.5, linewidth=0.5)
        assert activities_gt.size(0) == 1, "This hack doesn't work with >1 batch size"
        axs[i].scatter([(obj + 1) for _ in range(snapshot_gt.size()[0])], np.arange(snapshot_gt.size()[0]), marker='o', c=[color_map[activity_names[a]] for a in activities_gt[0,:]])
        axs[i].scatter(xv.reshape(-1), yv.reshape(-1), marker='o', c=[color_palette[c] if c>=0 else [1,1,1,0] for c in snapshot_gt.cpu().numpy().T.reshape(-1)], alpha=snapshot_gt_alpha.cpu().numpy().T.reshape(-1))
        axs[i].set_xticks(np.arange(int(obj_mask.sum())+1), rotation=90)
        axs[i].set_xticklabels([n for n,om in zip(node_classes, obj_mask) if om ]+['ACTIVITY'], rotation=90)
        axsdense[i].scatter([(obj + 1) for _ in range(dense_gt.size()[0])], np.arange(dense_gt.size()[0]), marker='o', c=[color_map[activity_names[a]] for a in activities_gt[0,:]])
        axsdense[i].scatter(xv.reshape(-1), yv.reshape(-1), marker='o', c=[color_palette[c] for c in dense_gt.cpu().numpy().T.reshape(-1)])
        axsdense[i].set_xticks(np.arange(int(obj_mask.sum())+1), rotation=90)
        axsdense[i].set_xticklabels([n for n,om in zip(node_classes, obj_mask) if om ]+['ACTIVITY'], rotation=90)
        num_obj_total += obj_mask.sum()
    fig.set_size_inches(float(num_obj_total)/5+2,int(dense_gt.size()[0])/12+2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'movements_data.png'))
    figdense.set_size_inches(float(num_obj_total)/5+2,int(dense_gt.size()[0])/12+2)
    figdense.tight_layout()
    figdense.savefig(os.path.join(output_dir, f'locations_data.png'))

    plt.close('all')

    for num_steps,snapshot in enumerate(snapshots):
        figdense, axsdense = plt.subplots(1, len(snapshot))
        fig, axs = plt.subplots(1, len(snapshot))
        num_obj_total = 0
        for i,(dense_pred, snapshot_pred, activities_pred, dense_gt, snapshot_gt, snapshot_gt_alpha, activities_gt, obj_mask, act_alpha, node_idxs, potential_loc, potential_alp) in enumerate(snapshot):
            yv, xv = np.meshgrid(np.arange(snapshot_pred.size()[0]), np.arange(snapshot_pred.size()[1]))
            if activity_consistencies is not None:
                for obj in range(snapshot_gt.size()[1]):
                    color = 'w'
                    for icol,thresh in enumerate(stdev_threshes):
                        if stdev_for_node_idx[obj] > thresh:
                            color = color_red_to_green[icol]
                            break
                    axs[i].plot([obj,obj], [0,snapshot_gt.size()[0]], color=color, alpha=0.5, linewidth=0.5)
            else:
                for obj in range(snapshot_gt.size()[1]):
                    axs[i].plot([obj,obj], [0,snapshot_gt.size()[0]], color='k', alpha=0.5, linewidth=0.5)
            assert activities_gt.size(0) == 1, "This hack doesn't work with >1 batch size"
            axs[i].scatter([(obj + 1)-0.1 for _ in range(snapshot_pred.size()[0])], np.arange(snapshot_pred.size()[0]), marker='o', c=[color_map[activity_names[a]] for a in activities_gt[0,:]], alpha=act_alpha)
            axs[i].scatter([(obj + 1)+0.4 for _ in range(snapshot_pred.size()[0])], np.arange(snapshot_pred.size()[0]), marker='x', c=[color_map[activity_names[a]] for a in activities_pred[0,:]])
            axs[i].scatter(xv.reshape(-1)-0.3, yv.reshape(-1), marker='o', c=[color_palette[c] if c>=0 else [1,1,1,0] for c in snapshot_gt.cpu().numpy().T.reshape(-1)])
            axs[i].scatter(xv.reshape(-1), yv.reshape(-1), marker='x', c=[color_palette[c] if c>=0 else [1,1,1,0] for c in snapshot_pred.cpu().numpy().T.reshape(-1)])
            axs[i].scatter(xv.reshape(-1)+0.2, yv.reshape(-1), marker='.', c=[color_palette[c] if c>=0 else [1,1,1,0] for c in potential_loc.cpu().numpy().T.reshape(-1)], alpha=potential_alp.cpu().numpy().T.reshape(-1))
            axs[i].set_xticks(np.arange(int(obj_mask.sum())+1), rotation=90)
            axs[i].set_xticklabels([n for n,om in zip(node_classes, obj_mask) if om ]+['ACTIVITY'], rotation=90)
            axsdense[i].scatter([(obj + 1)-0.1 for _ in range(dense_pred.size()[0])], np.arange(dense_pred.size()[0]), marker='o', c=[color_map[activity_names[a]] for a in activities_gt[0,:]], alpha=act_alpha)
            axsdense[i].scatter([(obj + 1)+0.4 for _ in range(dense_pred.size()[0])], np.arange(dense_pred.size()[0]), marker='x', c=[color_map[activity_names[a]] for a in activities_pred[0,:]])
            axsdense[i].scatter(xv.reshape(-1)-0.1, yv.reshape(-1), marker='o', c=[color_palette[c] for c in dense_gt.cpu().numpy().T.reshape(-1)], alpha=snapshot_gt_alpha.cpu().numpy().T.reshape(-1))
            axsdense[i].scatter(xv.reshape(-1)+0.1, yv.reshape(-1), marker='x', c=[color_palette[c] for c in dense_pred.cpu().numpy().T.reshape(-1)])
            axsdense[i].set_xticks(np.arange(int(obj_mask.sum())+1), rotation=90)
            axsdense[i].set_xticklabels([n for n,om in zip(node_classes, obj_mask) if om ]+['ACTIVITY'], rotation=90)
            num_obj_total += obj_mask.sum()
        fig.set_size_inches(float(num_obj_total)/3+2,int(dense_gt.size()[0])/12+2)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir,f'movements_{num_steps}.png'))
        figdense.set_size_inches(float(num_obj_total)/5+2,int(dense_pred.size()[0])/12+2)
        figdense.tight_layout()
        figdense.savefig(os.path.join(output_dir,f'locations_{num_steps}.png'))
        plt.close('all')

    for query_type, snapshot_queries_for_type in snapshots_queries.items():
        figdense, axsdense = plt.subplots(1, len(snapshot_queries_for_type))
        fig, axs = plt.subplots(1, len(snapshot_queries_for_type))
        # for i,(dense_pred, pred_alpha, snapshot_pred, activities_pred, dense_gt, snapshot_gt, snapshot_gt_alpha, activities_gt, obj_mask, node_idxs, potential_loc, potential_alp,best_queries_obj_viz, value_obj, oracle_positive_obj, best_queries_act_viz, value_act, oracle_positive_act) in enumerate(snapshot_queries_for_type):
        for i,query_data in enumerate(snapshot_queries_for_type):
            yv, xv = np.meshgrid(np.arange(query_data.pred.snapshot.size()[0]), np.arange(query_data.pred.snapshot.size()[1]))
            if activity_consistencies is not None:
                for obj in range(snapshot_gt.size()[1]):
                    color = 'w'
                    for icol,thresh in enumerate(stdev_threshes):
                        if stdev_for_node_idx[obj] > thresh:
                            color = color_red_to_green[icol]
                            break
                    axs[i].plot([obj,obj], [0,snapshot_gt.size()[0]], color=color, alpha=0.5, linewidth=0.5)
            for time in range(0,query_data.pred.snapshot.size()[0]):
                if std_range_by_activity is not None:
                    axs[i].plot([0,query_data.pred.snapshot.size()[1]], [time,time], color=color_red_to_green[std_range_by_activity[query_data.gt.activities[0,time]]], alpha=0.1, linewidth=0.5)
                else:
                    axs[i].plot([0,query_data.pred.snapshot.size()[1]], [time,time], 'k', alpha=0.1, linewidth=0.5)
                
            assert query_data.gt.activities.size(0) == 1, "This hack doesn't work with >1 batch size"
            axs[i].scatter([(obj + 1)-0.1 for _ in range(query_data.pred.snapshot.size()[0])], np.arange(query_data.pred.snapshot.size()[0]), marker='o', c=[color_map[activity_names[a.to(int).item()]] for a in query_data.gt.activities.reshape(-1)], alpha=act_alpha)
            axs[i].scatter([(obj + 1)-0.2 for _ in range(query_data.pred.snapshot.size()[0])], np.arange(query_data.pred.snapshot.size()[0]), marker='.', c=[color_map[activity_names[a]] for a in snapshots_data[i][3][0,1:query_data.pred.snapshot.size()[0]+1]])
            axs[i].scatter([(obj + 1)+0.4 for _ in range(query_data.pred.snapshot.size()[0])], np.arange(query_data.pred.snapshot.size()[0]), marker='x', c=[color_map[activity_names[a.to(int).item()]] for a in query_data.pred.activities.reshape(-1)])
            # activity_alpha = (query_data.queries.act_value.cpu().numpy().T.reshape(-1)/(max(query_data.queries.act_value.cpu())+1e-8))
            activity_alpha = ((query_data.queries.act_value.cpu().numpy().T.reshape(-1))>0)
            axs[i].scatter([(obj + 1)+1.2 for _ in range(query_data.pred.snapshot.size()[0])], np.arange(query_data.pred.snapshot.size()[0]), marker='v', c=[color_map[activity_names[a]] if a>0 else [1,1,1,0] for a in query_data.queries.act_queries_carryover.reshape(-1).to(int).cpu().numpy()], alpha=0.5) # alpha=query_data.queries.act_oracle.cpu().numpy()*0.5+0.5)
            axs[i].scatter([(obj + 1)+0.8 for _ in range(query_data.pred.snapshot.size()[0])], np.arange(query_data.pred.snapshot.size()[0]), marker='^', c=[color_map[activity_names[a]] if a>0 else [1,1,1,0] for a in query_data.queries.act_queries.to(int).cpu().numpy()], alpha=activity_alpha*0.8+0.2) # alpha=query_data.queries.act_oracle.cpu().numpy()*0.5+0.5)
            axs[i].scatter([(obj + 1)+1.6 for _ in range(query_data.pred.snapshot.size()[0])], np.arange(query_data.pred.snapshot.size()[0]), marker='x', c=[color_map[activity_names[a]] for a in query_data.pred.activities_wo_query.to(int).reshape(-1).cpu().numpy()], s=10)
            axs[i].scatter(xv.reshape(-1)-0.3, yv.reshape(-1), marker='o', c=[color_palette[c] if c>=0 else [1,1,1,0] for c in query_data.gt.snapshot.cpu().numpy().T.reshape(-1)])
            axs[i].scatter(xv.reshape(-1), yv.reshape(-1), marker='x', c=[color_palette[c] if c>=0 else [1,1,1,0] for c in query_data.pred.snapshot.cpu().numpy().T.reshape(-1)], alpha=query_data.pred.alpha.cpu().numpy().T.reshape(-1))
            axs[i].scatter(xv.reshape(-1)+0.2, yv.reshape(-1), marker='.', c=[color_palette[c] if c>=0 else [1,1,1,0] for c in query_data.probs.location.cpu().numpy().T.reshape(-1)], alpha=query_data.probs.alpha.cpu().numpy().T.reshape(-1))
            query_data_obj_value_alpha = ((query_data.queries.obj_value.unsqueeze(-1).repeat(1,query_data.queries.obj_queries.size()[-1]).reshape(-1))>0).to(float)
            query_data_obj_value_alpha = query_data_obj_value_alpha.cpu().numpy().T.reshape(-1)*0.99+0.01
            axs[i].scatter(xv.reshape(-1)+0.4, yv.reshape(-1), marker='v', c=['k' if c>0 else [1,1,1,0] for c in (query_data.queries.obj_queries_carryover).cpu().numpy().T.reshape(-1)], alpha=0.5)
            axs[i].scatter(xv.reshape(-1)+0.4, yv.reshape(-1), marker='^', c=[color_palette[(c+p)] if c>0 else [1,1,1,0] for c,p in zip((query_data.queries.obj_queries).cpu().numpy().T.reshape(-1), (query_data.queries.obj_oracle).unsqueeze(1).to(int).repeat(1,query_data.queries.obj_queries.size()[1]).cpu().numpy().T.reshape(-1))], alpha=(query_data_obj_value_alpha))
            axs[i].scatter(xv.reshape(-1)+0.4, yv.reshape(-1), marker='x', c=[color_palette[(c)] if c>=0 else [1,1,1,0] for c in query_data.pred.snapshot_wo_query.cpu().numpy().T.reshape(-1)], s=10)
            axs[i].set_xticks(np.arange(int(query_data.obj_mask.sum())+1), rotation=90)
            axs[i].set_xticklabels([n for n,om in zip(node_classes, query_data.obj_mask) if om ]+['ACTIVITY'], rotation=90)
            axsdense[i].scatter([(obj + 1)-0.1 for _ in range(query_data.pred.dense.size()[0])], np.arange(query_data.pred.dense.size()[0]), marker='o', c=[color_map[activity_names[a.to(int).item()]] for a in query_data.gt.activities.reshape(-1)], alpha=act_alpha)
            axsdense[i].scatter([(obj + 1)+0.4 for _ in range(query_data.pred.dense.size()[0])], np.arange(query_data.pred.dense.size()[0]), marker='x', c=[color_map[activity_names[a.to(int).item()]] for a in query_data.pred.activities.reshape(-1)])
            axsdense[i].scatter(xv.reshape(-1)-0.1, yv.reshape(-1), marker='o', c=[color_palette[c] for c in query_data.gt.dense.cpu().numpy().T.reshape(-1)], alpha=query_data.gt.alpha.cpu().numpy().T.reshape(-1))
            axsdense[i].scatter(xv.reshape(-1)+0.1, yv.reshape(-1), marker='x', c=[color_palette[c] for c in query_data.pred.dense.cpu().numpy().T.reshape(-1)], alpha=query_data.pred.alpha.cpu().numpy().T.reshape(-1))
            axsdense[i].set_xticks(np.arange(int(query_data.obj_mask.sum())+1), rotation=90)
            axsdense[i].set_xticklabels([n for n,om in zip(node_classes, query_data.obj_mask) if om ]+['ACTIVITY'], rotation=90)
        fig.set_size_inches(float(num_obj_total)/3+2,int(query_data.gt.dense.size()[0])/12+2)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir,f'query_movements_{query_type}_{query_steps}.png'))
        figdense.set_size_inches(float(num_obj_total)/5+2,int(query_data.pred.dense.size()[0])/12+2)
        figdense.tight_layout()
        figdense.savefig(os.path.join(output_dir,f'query_locations_{query_type}_{query_steps}.png'))


    plt.close('all')
