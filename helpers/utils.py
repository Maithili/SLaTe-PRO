import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from adict import adict

# color_red_to_green = ["#ffffff", "#bc4b51", "#fbc4ab", "#90a955", "#31572c"]

def make_adict(diction):
    adict_ = {}
    for key, value in diction.items():
        if isinstance(value, dict):
            adict_[key] = make_adict(value)
        else:
            adict_[key] = value
    return adict(adict_)

color_map = {
}

color_palette = sns.color_palette()+sns.color_palette(palette='dark')
color_palette = color_palette * 10

def get_metrics(results):

    def stats_from_confusion_matrix(conf_mat_dict):
        stats = {'precision':[], 'recall':[], 'f1':[]}
        for lookahead_step in range(len(conf_mat_dict['tp'])):
            tp = conf_mat_dict['tp'][lookahead_step]
            fp = conf_mat_dict['fp'][lookahead_step]
            fn = conf_mat_dict['fn'][lookahead_step]
            stats['precision'].append(tp / (tp + fp + 1e-10))
            stats['recall'].append(tp / (tp + fn + 1e-10))
            stats['f1'].append(2 * tp / (2 * tp + fp + fn + 1e-10))
        return stats

    results.update(stats_from_confusion_matrix(results['confusion_matrix']))

    results['num_movements_fetched'] = [[{'mean':np.mean(data), 'std':np.std(data)} for data in lookahed_step_data] for lookahed_step_data in results['num_movements_fetched']]
    results['num_movements_returned'] = [[{'mean':np.mean(data), 'std':np.std(data)} for data in lookahed_step_data] for lookahed_step_data in results['num_movements_returned']]
    ## TODO Maithili for MHC: Split metrics by object type. Add other metrics here.

    return results

