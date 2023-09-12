import os
import shutil
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from math import isnan, floor, ceil, sqrt
import torch
import torch.nn.functional as F
from copy import deepcopy
from encoders import human_readable_from_external
import seaborn as sns
from utils import color_palette

METRIC_TO_PLOT = 'f1'
# METRIC_TO_PLOT = 'precision'
# METRIC_TO_PLOT = 'recall'

LENIENCY=0

def get_eval_file(m, output_dir, leniency=LENIENCY):
    eval_dirs = [os.path.join(output_dir,m,d) for d in os.listdir(os.path.join(output_dir,m)) if d.startswith('test_evals_')]
    if len(eval_dirs) == 0: return None
    eval_dirs.sort(reverse=True)
    eval_dir = eval_dirs[0]
    for file in os.listdir(eval_dir):
        if file.startswith('test_evaluation') and \
        'splits' in file and \
        os.path.exists(os.path.join(eval_dir,file)):
            return json.load(open(os.path.join(eval_dir,file)))
    else:
        return None
    
method_labels = {
    'original': 'STOT',
    'default_0': '(No Act)',
    'default_25': '(25% Act)',
    'default_50': '(50% Act)',
    'default_75': '(75% Act)',
    'default_100': '(100% Act)',
}
    
nice_red = np.array([153.,48.,63.])/255
nice_plain =  np.array([204,183,174])/255 #np.array([95,89,78])/255
nice_green =  np.array([92,137,62])/255

color_palette = {
    'object': '#C4D473',
    'activity': '#FFEB85',
    'both': '#F19C79',
    'none': '#A5243D',
    'None': '#A5243D',
    '0': sns.color_palette()[0],
    '1': sns.color_palette()[1],
    '2': sns.color_palette()[2],
    '3': sns.color_palette()[3],
    '4': sns.color_palette()[4],
    '5': sns.color_palette()[5],
    '6': sns.color_palette()[6],
    '7': sns.color_palette()[7],
    '8': sns.color_palette()[8],
    '9': sns.color_palette()[9],
    'original': '#0667B2',
    'default_0': nice_plain,
    'default_25': nice_plain*.75+nice_red*.25,  # '#AA4B95',
    'default_50': nice_plain*.5+nice_red*.5,  # '#E57C04',
    'default_75': nice_plain*.25+nice_red*.75,  # '#4D8B31',
    'default_100': nice_red,
    'default_100_2': '#F19C79', # '#C77F8F',
    'original_2': '#77C7E4',
    
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model on routines.')
    parser.add_argument('--path', type=str, default='logs', help='Path where the data lives. Must contain routines, info and classes json files.')
    parser.add_argument('--methods', type=str, default='original,default_0,default_25,default_50,default_75,default_100', help='Path where the data lives. Must contain routines, info and classes json files.')
    parser.add_argument('--log_dir_prefix', type=str, default='0529_Final_0.8trust_0.5thresh_6step_3query_6len', help='Path where the data lives. Must contain routines, info and classes json files.')
    
    args = parser.parse_args()
    order = args.methods.split(',')
    query_step = None
    dirpath=args.path

    ob_difftypes_labels = [
        ('_3c','f1_easy',"Most \n Consistent",1),
        ('_2c','f1_difficult',"",2),
        # ('_2b','f1_difficult',"",3),
        ('_3b','f1_difficult',"Least \n Consistent",3)
        # ('_3n','f1_easy',"Most \n Consistent",1),
        # ('_2n','f1_difficult',"",2),
        # ('_2m','f1_difficult',"",3),
        # ('_2l','f1_difficult',"",4),
        # ('_2k','f1_difficult',"",5),
        # ('_2j','f1_difficult',"",6),
        # ('_2i','f1_difficult',"",7),
        # ('_2h','f1_difficult',"",8),
        # ('_2g','f1_difficult',"",9),
        # ('_2f','f1_difficult',"",10),
        # ('_2e','f1_difficult',"",11),
        # ('_2d','f1_difficult',"",12),
        # ('_2c','f1_difficult',"",13),
        # ('_2b','f1_difficult',"",14),
        # ('_2a','f1_difficult',"Least \n Consistent",15),
    ]

    fig_st_all, ax_st_all = plt.subplots(1,2,figsize=(25,7))
    firstdir = True

    log_dirs = [p for p in os.listdir(args.path) if p.startswith(args.log_dir_prefix)]
    log_dirs.sort()

    for dirpath in [os.path.join(args.path, d) for d in log_dirs]:

        print("Processing directory ", dirpath)

        fig_ob, ax_ob = plt.subplots(2,2,figsize=(18,12))
        ax_ob = ax_ob.reshape(-1)

        fig_ob_final1, ax_ob_final1 = plt.subplots(figsize=(8,5))
        fig_ob_final2, ax_ob_final2 = plt.subplots(figsize=(3,5))
        fig_ob_final3, ax_ob_final3 = plt.subplots(figsize=(7,5))

        fig_st, ax_st = plt.subplots(1,2,figsize=(25,7))
        # ob_difftypes_labels = [
        #     ('_2','f1_easy',"Easy",1),
        #     ('_2','f1_difficult',"Difficult",2),
        # ]
        
        labels_ob_1 = lambda qt: qt
        labels_ob_0 = ["STOT", "STOT + F", "SLaTe-PRO", "SLaTe-PRO + F"]
        num_gt_in_splits = {}

        diff_types = [k.replace('relocation_difficult','') for k in get_eval_file('default_100', os.path.join(dirpath,'persona0_')).keys() if k.startswith('relocation_difficult')]
        for difficulty_type in diff_types:
            num_gt_in_splits[difficulty_type] = {}
            
            labels_step_1 = ["STOT", "SLaTe-PRO"]

            metrics = {method:{'relocation':{},
                    'activity':{},
                    'relocation_clarified':{},
                    'activity_clarified':{}} for method in ['original','default_0','default_25','default_50','default_75','default_100']}

            for dataset in os.listdir(dirpath):
                if 'plots' in dataset: continue
                if not os.path.isdir(os.path.join(dirpath,dataset)): continue
                print("Dataset", dataset)
                methods_and_eval = {m: get_eval_file(m, os.path.join(dirpath,dataset)) for m in order}
                for method, eval in methods_and_eval.items():
                    if eval is None:
                        print('Missing eval for', method)
                        continue
                    eval_in = methods_and_eval[method]
                    for s in eval_in['relocation']['f1']:
                        if s not in metrics[method]['relocation']:
                            metrics[method]['relocation'][s] = {'f1':[], 'f1_difficult':[], 'f1_easy':[]}
                            metrics[method]['activity'][s] = []
                        if METRIC_TO_PLOT == 'f1':
                            metrics[method]['relocation'][s]['f1'].append(eval_in['relocation']['f1'][s])
                            metrics[method]['relocation'][s]['f1_difficult'].append(eval_in['relocation_difficult'+difficulty_type]['f1'][s])
                            metrics[method]['relocation'][s]['f1_easy'].append(eval_in['relocation_easy'+difficulty_type]['f1'][s])
                        else:
                            metrics[method]['relocation'][s]['f1'].append(eval_in['relocation'][METRIC_TO_PLOT]['result'][s])
                            metrics[method]['relocation'][s]['f1_difficult'].append(eval_in['relocation_difficult'+difficulty_type][METRIC_TO_PLOT]['result'][s])
                            metrics[method]['relocation'][s]['f1_easy'].append(eval_in['relocation_easy'+difficulty_type][METRIC_TO_PLOT]['result'][s])
                        metrics[method]['activity'][s].append(eval_in['activity']['correct_fraction'][s])
                        if method == 'default_100':
                            if s not in num_gt_in_splits[difficulty_type]: num_gt_in_splits[difficulty_type][s] = {'f1_difficult':0, 'f1_easy':0}
                            num_gt_in_splits[difficulty_type][s]['f1_difficult'] += eval_in['relocation_difficult'+difficulty_type]["recall"]["tp"][s] + eval_in['relocation_difficult'+difficulty_type]["recall"]["fn"][s]
                            num_gt_in_splits[difficulty_type][s]['f1_easy'] += eval_in['relocation_easy'+difficulty_type]["recall"]["tp"][s] + eval_in['relocation_easy'+difficulty_type]["recall"]["fn"][s]
                    for qt in eval_in['relocation_clarified']:
                        if qt in 'query_step':
                            query_step = eval_in['relocation_clarified'][qt]
                            continue
                        if qt in ['f1']: continue
                        if qt not in metrics[method]['relocation_clarified']:
                            metrics[method]['relocation_clarified'][qt] = {'f1':[], 'f1_difficult':[], 'f1_easy':[], 'num_queries':[]}
                            metrics[method]['activity_clarified'][qt] = []
                        metrics[method]['relocation_clarified'][qt]['num_queries'].append(eval_in['relocation_clarified'][qt]['num_queries'])
                        if METRIC_TO_PLOT == 'f1':
                            metrics[method]['relocation_clarified'][qt]['f1'].append(eval_in['relocation_clarified'][qt]['f1'])
                            metrics[method]['relocation_clarified'][qt]['f1_difficult'].append(eval_in['relocation_clarified_difficult'+difficulty_type][qt]['f1'])
                            metrics[method]['relocation_clarified'][qt]['f1_easy'].append(eval_in['relocation_clarified_easy'+difficulty_type][qt]['f1'])
                        else:
                            metrics[method]['relocation_clarified'][qt]['f1'].append(eval_in['relocation_clarified'][qt][METRIC_TO_PLOT]['result'])
                            metrics[method]['relocation_clarified'][qt]['f1_difficult'].append(eval_in['relocation_clarified_difficult'+difficulty_type][qt][METRIC_TO_PLOT]['result'])
                            metrics[method]['relocation_clarified'][qt]['f1_easy'].append(eval_in['relocation_clarified_easy'+difficulty_type][qt][METRIC_TO_PLOT]['result'])
                        metrics[method]['activity_clarified'][qt].append(eval_in['activity_clarified'][qt]['correct_fraction'])

                
            metrics_lenient = {method:{'relocation':{},
                    'activity':{},
                    'relocation_clarified':{},
                    'activity_clarified':{}} for method in ['original','default_0','default_25','default_50','default_75','default_100']}

            eval_in = None
            for dataset in os.listdir(dirpath):
                if 'plots' in dataset: continue
                if not os.path.isdir(os.path.join(dirpath,dataset)): continue
                print("Dataset", dataset)
                methods_and_eval = {m: get_eval_file(m, os.path.join(dirpath,dataset), leniency=2) for m in order}
                for method, eval in methods_and_eval.items():
                    if eval is None:
                        print('Missing eval for', method)
                        continue
                    eval_in = methods_and_eval[method]
                    for s in eval_in['relocation']['f1']:
                        if s not in metrics_lenient[method]['relocation']:
                            metrics_lenient[method]['relocation'][s] = {'f1':[], 'f1_difficult':[], 'f1_easy':[]}
                            metrics_lenient[method]['activity'][s] = []
                        metrics_lenient[method]['relocation'][s]['f1'].append(eval_in['relocation']['f1'][s])
                        metrics_lenient[method]['relocation'][s]['f1_difficult'].append(eval_in['relocation_difficult'+difficulty_type]['f1'][s])
                        metrics_lenient[method]['relocation'][s]['f1_easy'].append(eval_in['relocation_easy'+difficulty_type]['f1'][s])
                        metrics_lenient[method]['activity'][s].append(eval_in['activity']['correct_fraction'][s])
                    for qt in eval_in['relocation_clarified']:
                        if qt in 'query_step':
                            query_step = eval_in['relocation_clarified'][qt]
                            continue
                        if qt in ['f1']: continue
                        if qt not in metrics_lenient[method]['relocation_clarified']:
                            metrics_lenient[method]['relocation_clarified'][qt] = {'f1':[], 'f1_difficult':[], 'f1_easy':[]}
                            metrics_lenient[method]['activity_clarified'][qt] = []
                        metrics_lenient[method]['relocation_clarified'][qt]['f1'].append(eval_in['relocation_clarified'][qt]['f1'])
                        metrics_lenient[method]['relocation_clarified'][qt]['f1_difficult'].append(eval_in['relocation_clarified_difficult'+difficulty_type][qt]['f1'])
                        metrics_lenient[method]['relocation_clarified'][qt]['f1_easy'].append(eval_in['relocation_clarified_easy'+difficulty_type][qt]['f1'])
                        metrics_lenient[method]['activity_clarified'][qt].append(eval_in['activity_clarified'][qt]['correct_fraction'])
            
            if eval_in is None:
                print(f"Skipping directory without data {dirpath}")
                continue

            steps = list(eval_in['relocation']['f1'].keys())
            steps.sort(key = lambda x: int(x))
            query_types = list(eval_in['relocation_clarified'].keys())
            query_types.remove('query_step')

            os.makedirs(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','plots'+difficulty_type), exist_ok=True)

            fig, ax = plt.subplots(1,3,figsize=(20,10))
            res_str = ''
            for i, s in enumerate(steps):
                for j, metric in enumerate(['f1','f1_difficult','f1_easy']):
                    # for mi,m in enumerate(order):
                    #     ax[j].bar(mi+i/len(steps), np.mean(metrics[m]['relocation'][s][metric]), yerr=np.std(metrics[m]['relocation'][s][metric]), label=f"{m}", width=1/len(steps)-0.1, color=color_palette[s])
                    ax[j].errorbar(np.arange(len(order)),[np.mean(metrics[m]['relocation'][s][metric]) if len(metrics[m]['relocation'])>0 else np.nan for m in order], yerr=[np.std(metrics[m]['relocation'][s][metric]) if len(metrics[m]['relocation'])>0 else np.nan for m in order], label=f"{s}-step")
                    ax[j].set_title(metric)
                    ax[j].legend()
                    if i == len(steps)-1:  
                        res_str += f"\n{difficulty_type}_{metric}:{np.mean(metrics['default_100']['relocation'][s][metric])} +- {np.std(metrics['default_100']['relocation'][s][metric])}"
            open(os.path.join(dirpath,'compare_plots_lenient.txt'), 'a').write(res_str+'\n')
            # ax[0].get_shared_y_axes().join(ax[0], ax[1])
            # ax[0].get_shared_x_axes().join(ax[0], ax[1])
            # ax[0].get_shared_y_axes().join(ax[0], ax[2])
            # ax[0].get_shared_x_axes().join(ax[0], ax[2])
            fig.tight_layout()
            fig.savefig(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','plots'+difficulty_type,'relocation.png'))
            plt.close(fig)

            fig,ax = plt.subplots(1,1,figsize=(5,4))
            for mi, m in enumerate(order):
                if len(metrics[m]['relocation']) > 0:
                    p = ax.bar(mi, np.mean(metrics[m]['relocation'][steps[query_step]]['f1']), yerr= np.std(metrics[m]['relocation'][steps[query_step]]['f1']), color=color_palette[m])
                    ax.bar_label(p, fmt='%.2f')
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels([method_labels[o] for o in order])
            ax.set_ylabel('F1-score')
            # ax.legend(loc='lower right')
            fig.tight_layout()
            fig.savefig(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','plots'+difficulty_type,'relocation_bw_methods.png'))

            fig,ax = plt.subplots(1,1,figsize=(5,4))
            for mi, m in enumerate(order):
                if len(metrics[m]['relocation']) > 0:
                    p = ax.bar(mi-0.2, np.mean(metrics[m]['relocation'][steps[query_step]]['f1']), yerr= np.std(metrics[m]['relocation'][steps[query_step]]['f1']), color=color_palette[m], width=0.35)
                    ax.bar_label(p, fmt='%.3f')
                    if 'both' in metrics[m]['relocation_clarified']:
                        p = ax.bar(mi+0.2, np.mean(metrics[m]['relocation_clarified']['both']['f1']), yerr= np.std(metrics[m]['relocation_clarified']['both']['f1']), color=color_palette[m], width=0.35)
                        print(f"using both: {m}")
                    else:
                        print(metrics[m]['relocation_clarified'])
                        p = ax.bar(mi+0.2, np.mean(metrics[m]['relocation_clarified']['object']['f1']), yerr= np.std(metrics[m]['relocation_clarified']['object']['f1']), color=color_palette[m], width=0.35)
                        print(f"using object: {m}")
                    ax.bar_label(p, fmt='%.3f')
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels([method_labels[o] for o in order])
            ax.set_ylabel('F1-score')
            # ax.legend(loc='lower right')
            fig.tight_layout()
            fig.savefig(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','plots'+difficulty_type,'relocation_clarified_bw_methods.png'))


            fig,ax = plt.subplots(1,1,figsize=(7,5))
            for i,s in enumerate(steps):
                m='original'
                if len(metrics[m]['relocation']) > 0:
                    p = ax.bar(i-0.2, np.mean(metrics[m]['relocation'][s]['f1']), yerr= np.std(metrics[m]['relocation'][s]['f1']), width=0.35, label=labels_step_1[0], color=color_palette[m])
                    ax.bar_label(p, fmt='%.2f')
                    labels_step_1[0] = None
                m='default_100'
                if len(metrics[m]['relocation']) > 0:
                    p = ax.bar(i+0.2, np.mean(metrics[m]['relocation'][s]['f1']), yerr= np.std(metrics[m]['relocation'][s]['f1']), width=0.35, label=labels_step_1[1], color=color_palette[m])
                    ax.bar_label(p, fmt='%.2f')
                    labels_step_1[1] = None
            ax.set_xticks(np.arange(len(steps)))
            ax.set_xticklabels([f"{int(s)+1}-step" for s in steps])
            ax.set_ylabel('F1-score')
            ax.legend(loc='lower right')
            fig.tight_layout()
            fig.savefig(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','plots'+difficulty_type,'relocation_over_steps.png'))

            fig, ax = plt.subplots(1,3,figsize=(20,10))
            for j, metric in enumerate(['f1','f1_difficult','f1_easy']):
                # for mi,m in enumerate(order):
                for i, qt in enumerate(query_types):
                    if qt in ['f1']: continue
                        # if len(metrics[m]['relocation_clarified'])>0:
                        #     ax[j].bar(mi+((i+1)/(len(query_types)+2)), np.mean(metrics[m]['relocation_clarified'][qt][metric]), yerr=np.std(metrics[m]['relocation_clarified'][qt][metric]), label=qt, width=(1/(len(query_types)+2))-0.1, color=color_palette[qt])
                        # ax[j].bar(mi, np.mean(metrics[m]['relocation'][str(query_step)][metric]), yerr=np.std(metrics[m]['relocation'][str(query_step)][metric]), width=(1/(len(query_types)+2))-0.1, label="None", color=color_palette[qt])
                    ax[j].errorbar(np.arange(len(order)),[np.mean(metrics[m]['relocation_clarified'][qt][metric]) if qt in metrics[m]['relocation_clarified'] else np.nan for m in order], yerr=[np.std(metrics[m]['relocation_clarified'][qt][metric]) if qt in metrics[m]['relocation_clarified'] else np.nan for m in order], label=f"{qt}")
                    ob_pick = [o for o in ob_difftypes_labels if o[0]==difficulty_type]
                    if len(ob_pick) > 0:
                        if qt == 'both':
                            for _, easyhard, label, xidx in ob_pick:
                                if metric == easyhard:
                                    x = np.array(metrics['original']['relocation'][str(query_step)][metric])
                                    x = x[x>0] if x.max()>0 else x
                                    p = ax_ob_final1.bar(xidx-0.3, np.mean(x), yerr=np.std(x), width=0.15, label=labels_ob_0[0], color=color_palette['original'])
                                    ax_ob_final1.bar_label(p, fmt='%.2f')
                                    x = np.array(metrics['original']['relocation_clarified']['object'][metric])
                                    x = x[x>0] if x.max()>0 else x
                                    p = ax_ob_final1.bar(xidx-0.1, np.mean(x), yerr=np.std(x), width=0.15, label=labels_ob_0[1], color=color_palette['original_2'])
                                    ax_ob_final1.bar_label(p, fmt='%.2f')
                                    x = np.array(metrics['default_100']['relocation'][str(query_step)][metric])
                                    x = x[x>0] if x.max()>0 else x
                                    p = ax_ob_final1.bar(xidx+0.1, np.mean(x), yerr=np.std(x), width=0.15, label=labels_ob_0[2], color=color_palette['default_100'])
                                    ax_ob_final1.bar_label(p, fmt='%.2f')
                                    x = np.array(metrics['default_100']['relocation_clarified'][qt][metric])
                                    x = x[x>0] if x.max()>0 else x
                                    p = ax_ob_final1.bar(xidx+0.3, np.mean(x), yerr=np.std(x), width=0.15, label=labels_ob_0[3], color=color_palette['default_100_2'])
                                    ax_ob_final1.bar_label(p, fmt='%.2f')
                                    labels_ob_0 = [None, None, None, None]
                                    width = 0.8/len(steps)
                                    for stpidx,s in enumerate(steps):
                                        p = ax_st[0].bar(xidx-0.4+width+stpidx*width, np.mean(metrics['default_100']['relocation'][s][metric]), yerr=np.std(metrics['default_100']['relocation'][s][metric]), width=width-0.01, color=((nice_red*stpidx+nice_plain*(len(steps)-stpidx))/len(steps)))
                                        ax_st[0].bar_label(p, fmt='%.3f')
                                        p = ax_st[1].bar(xidx-0.4+width+stpidx*width, np.mean(metrics_lenient['default_100']['relocation'][s][metric]), yerr=np.std(metrics_lenient['default_100']['relocation'][s][metric]), width=width-0.01, color=((nice_red*stpidx+nice_plain*(len(steps)-stpidx))/len(steps)))
                                        ax_st[1].bar_label(p, fmt='%.3f')
                                        if firstdir:
                                            p = ax_st_all[0].bar(xidx-0.4+width/2+stpidx*width, np.mean(metrics['default_100']['relocation'][s][metric]), yerr=np.std(metrics['default_100']['relocation'][s][metric]), width=width/2-0.01, color=((nice_red*stpidx+nice_plain*(len(steps)-stpidx))/len(steps)))
                                            ax_st_all[0].bar_label(p, fmt='%.3f')
                                            p = ax_st_all[1].bar(xidx-0.4+width/2+stpidx*width, np.mean(metrics_lenient['default_100']['relocation'][s][metric]), yerr=np.std(metrics_lenient['default_100']['relocation'][s][metric]), width=width/2-0.01, color=((nice_red*stpidx+nice_plain*(len(steps)-stpidx))/len(steps)))
                                            ax_st_all[1].bar_label(p, fmt='%.3f')
                                    stpidx = query_step
                                    p = ax_st_all[0].bar(xidx-0.4+width+stpidx*width, np.mean(metrics['default_100']['relocation_clarified']['both'][metric]), yerr=np.std(metrics['default_100']['relocation_clarified']['both'][metric]), width=width/2-0.01, color=((nice_green*stpidx+nice_plain*(len(steps)-stpidx))/len(steps)))
                                    ax_st_all[0].bar_label(p, fmt='%.3f')
                                    p = ax_st_all[1].bar(xidx-0.4+width+stpidx*width, np.mean(metrics_lenient['default_100']['relocation_clarified']['both'][metric]), yerr=np.std(metrics_lenient['default_100']['relocation_clarified']['both'][metric]), width=width/2-0.01, color=((nice_green*stpidx+nice_plain*(len(steps)-stpidx))/len(steps)))
                                    ax_st_all[1].bar_label(p, fmt='%.3f')
                            if metric=='f1':
                                x = np.array(metrics['original']['relocation'][str(query_step)][metric])
                                x = x[x>0] if x.max()>0 else x
                                p = ax_ob_final2.bar(-0.3, np.mean(x), yerr=np.std(x), width=0.15, color=color_palette['original'])
                                ax_ob_final2.bar_label(p, fmt='%.2f')
                                x = np.array(metrics['original']['relocation_clarified']['object'][metric])
                                x = x[x>0] if x.max()>0 else x
                                p = ax_ob_final2.bar(-0.1, np.mean(x), yerr=np.std(x), width=0.15, color=color_palette['original_2'])
                                ax_ob_final2.bar_label(p, fmt='%.2f')
                                x = np.array(metrics['default_100']['relocation'][str(query_step)][metric])
                                x = x[x>0] if x.max()>0 else x
                                p = ax_ob_final2.bar(0.1, np.mean(x), yerr=np.std(x), width=0.15, color=color_palette['default_100'])
                                ax_ob_final2.bar_label(p, fmt='%.2f')
                                x = np.array(metrics['default_100']['relocation_clarified'][qt][metric])
                                x = x[x>0] if x.max()>0 else x
                                p = ax_ob_final2.bar(0.3, np.mean(x), yerr=np.std(x), width=0.15, color=color_palette['default_100_2'])
                                ax_ob_final2.bar_label(p, fmt='%.2f')
                                width = 0.8/len(steps)
                                for stpidx,s in enumerate(steps):
                                    p = ax_st[0].bar(-0.4+width+stpidx*width, np.mean(metrics['default_100']['relocation'][s][metric]), yerr=np.std(metrics['default_100']['relocation'][s][metric]), width=width-0.01, color=((nice_red*stpidx+nice_plain*(len(steps)-stpidx))/len(steps)))
                                    ax_st[0].bar_label(p, fmt='%.3f')
                                    p = ax_st[1].bar(-0.4+width+stpidx*width, np.mean(metrics_lenient['default_100']['relocation'][s][metric]), yerr=np.std(metrics_lenient['default_100']['relocation'][s][metric]), width=width-0.01, color=((nice_red*stpidx+nice_plain*(len(steps)-stpidx))/len(steps)))
                                    ax_st[1].bar_label(p, fmt='%.3f')
                                    if firstdir:
                                        p = ax_st_all[0].bar(-0.4+width/2+stpidx*width, np.mean(metrics['default_100']['relocation'][s][metric]), yerr=np.std(metrics['default_100']['relocation'][s][metric]), width=width/2-0.01, color=((nice_red*stpidx+nice_plain*(len(steps)-stpidx))/len(steps)))
                                        ax_st_all[0].bar_label(p, fmt='%.3f')
                                        p = ax_st_all[1].bar(-0.4+width/2+stpidx*width, np.mean(metrics_lenient['default_100']['relocation'][s][metric]), yerr=np.std(metrics_lenient['default_100']['relocation'][s][metric]), width=width/2-0.01, color=((nice_red*stpidx+nice_plain*(len(steps)-stpidx))/len(steps)))
                                        ax_st_all[1].bar_label(p, fmt='%.3f')
                                stpidx = query_step
                                p = ax_st_all[0].bar(-0.4+width+stpidx*width, np.mean(metrics['default_100']['relocation_clarified']['both'][metric]), yerr=np.std(metrics['default_100']['relocation_clarified']['both'][metric]), width=width/2-0.01, color=((nice_green*stpidx+nice_plain*(len(steps)-stpidx))/len(steps)))
                                ax_st_all[0].bar_label(p, fmt='%.3f')
                                p = ax_st_all[1].bar(-0.4+width+stpidx*width, np.mean(metrics_lenient['default_100']['relocation_clarified']['both'][metric]), yerr=np.std(metrics_lenient['default_100']['relocation_clarified']['both'][metric]), width=width/2-0.01, color=((nice_green*stpidx+nice_plain*(len(steps)-stpidx))/len(steps)))
                                ax_st_all[1].bar_label(p, fmt='%.3f')
                        for _, easyhard, label, xidx in ob_pick:
                            if metric == easyhard:
                                qid = ['activity','object','both'].index(qt)
                                x = np.array(metrics['default_100']['relocation_clarified'][qt][metric])
                                x = x[x>0] if x.max() > 0 else x
                                p = ax_ob[1].bar(xidx-0.1+0.2*qid, np.mean(x), yerr=np.std(x), width=0.15, label=labels_ob_1(qt), color=color_palette[qt])
                                ax_ob[1].bar_label(p, fmt='%.3f')
                                if qt == 'both':
                                    x = np.array(metrics['default_100']['relocation'][str(query_step)][metric])
                                    x = x[x>0] if x.max() > 0 else x
                                    p = ax_ob[1].bar(xidx-0.3, np.mean(x), yerr=np.std(x), width=0.15, label=labels_ob_1("None"), color=color_palette['none'])
                                    ax_ob[1].bar_label(p, fmt='%.3f')
                        if metric=='f1':
                            qid = ['activity','object','both'].index(qt)
                            x = np.array(metrics['default_100']['relocation_clarified'][qt][metric])
                            x = x[x>0] if x.max() > 0 else x
                            p = ax_ob[1].bar(-0.1+0.2*qid, np.mean(x), yerr=np.std(x), width=0.15, color=color_palette[qt])
                            ax_ob[1].bar_label(p, fmt='%.3f')
                            if qt == 'both':
                                x = np.array(metrics['default_100']['relocation'][str(query_step)][metric])
                                x = x[x>0] if x.max() > 0 else x
                                p = ax_ob[1].bar(-0.3, np.mean(x), yerr=np.std(x), width=0.15, color=color_palette['none'])
                                ax_ob[1].bar_label(p, fmt='%.3f')
                            
                        


                ax[j].errorbar(np.arange(len(order)),[np.mean(metrics[m]['relocation'][str(query_step)][metric]) if len(metrics[m]['relocation'])>0 else np.nan for m in order], yerr=[np.std(metrics[m]['relocation'][str(query_step)][metric]) if len(metrics[m]['relocation'])>0 else np.nan for m in order], label=f"None")
                ax[j].legend()
                ax[j].set_title(metric)
            # ax[0].get_shared_y_axes().join(ax[0], ax[1])
            # ax[0].get_shared_x_axes().join(ax[0], ax[1])
            # ax[0].get_shared_y_axes().join(ax[0], ax[2])
            # ax[0].get_shared_x_axes().join(ax[0], ax[2])
            fig.tight_layout()
            fig.savefig(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','plots'+difficulty_type,'relocation_clarified.png'))
            plt.close(fig)
            labels_ob_1 = lambda qt: None

            fig, ax = plt.subplots(1,2,figsize=(20,10))
            for j, metric in enumerate(['f1','f1_easy','f1_difficult']):
                if qt in ['f1']: continue
                x = np.array(metrics['original']['relocation'][str(query_step)][metric])
                x = x[x>0] if x.max() > 0 else x
                p = ax[0].bar(j-0.3, np.mean(x), yerr=np.std(x), width=0.15, color=color_palette['original'])
                ax[0].bar_label(p, fmt='%.3f')
                x = np.array(metrics['original']['relocation_clarified']['object'][metric])
                x = x[x>0] if x.max() > 0 else x
                p = ax[0].bar(j-0.1, np.mean(x), yerr=np.std(x), width=0.15, color=color_palette['original_2'])
                ax[0].bar_label(p, fmt='%.3f')
                x = np.array(metrics['default_100']['relocation'][str(query_step)][metric])
                x = x[x>0] if x.max() > 0 else x
                p = ax[0].bar(j+0.1, np.mean(x), yerr=np.std(x), width=0.15, color=color_palette['default_100'])
                ax[0].bar_label(p, fmt='%.3f')
                x = np.array(metrics['default_100']['relocation_clarified']['both'][metric])
                x = x[x>0] if x.max() > 0 else x
                p = ax[0].bar(j+0.3, np.mean(x), yerr=np.std(x), width=0.15, color=color_palette['default_100_2'])
                ax[0].bar_label(p, fmt='%.3f')
                x = np.array(metrics['default_100']['relocation'][str(query_step)][metric])
                x = x[x>0] if x.max() > 0 else x
                p = ax[1].bar(j-0.3, np.mean(x), yerr=np.std(x), width=0.15, color=color_palette['default_100'])
                ax[1].bar_label(p, fmt='%.3f')
                for i, qt in enumerate(query_types):
                    x = np.array(metrics['default_100']['relocation_clarified'][qt][metric])
                    x = x[x>0] if x.max() > 0 else x
                    p = ax[1].bar(j-0.1+0.2*i, np.mean(x), yerr=np.std(x), width=0.15, color=color_palette[qt])
                    ax[1].bar_label(p, fmt='%.3f')
                        
            ax[0].set_xticks(np.arange(3))
            ax[0].set_xticklabels(['Overall', 'Consistent', 'Inconcsistent'])
            ax[0].set_ylabel('F-1 Score')
            ax[1].set_xticks(np.arange(3))
            ax[1].set_xticklabels(['Overall', 'Consistent', 'Inconcsistent'])
            ax[1].set_ylabel('F-1 Score')
            ax[0].get_shared_y_axes().join(ax[0], ax[1])
            ax[0].legend()
            ax[1].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','plots'+difficulty_type,'relocation_by_difficulty.png'))
            plt.close(fig)
            
            
            fig, ax = plt.subplots(1,2,figsize=(20,10))
            for i, s in enumerate(steps):
                # for i,m in enumerate(order):
                #     if len(metrics[m]['activity'])>0:
                #         ax[0].bar(i+int(s)/(len(steps)+1), np.mean(metrics[m]['activity'][s]), yerr=np.std(metrics[m]['activity'][s]), label=f"{s}-step", width=1/(len(steps)+1)-0.1, color=color_palette[s])
                ax[0].errorbar(np.arange(len(order)),[np.mean(metrics[m]['activity'][s]) if len(metrics[m]['activity'])>0 else np.nan for m in order], yerr=[np.std(metrics[m]['activity'][s]) if len(metrics[m]['activity'])>0 else np.nan for m in order], label=f"{s}-step")
                ax[0].set_title('Stepwise Activity')
                ax[0].legend()

            for i, qt in enumerate(query_types):
                if qt in ['f1']: continue
                # for mi,m in enumerate(order):
                #     if len(metrics[m]['activity_clarified'])>0:
                #         ax[1].bar(mi+(i/(len(query_types)+1)), np.mean(metrics[m]['activity_clarified'][qt]), yerr=np.std(metrics[m]['activity_clarified'][qt]), label=qt, width=(1/(len(query_types)+1))-0.1, color=color_palette[qt])
                ax[1].errorbar(np.arange(len(order)),[np.mean(metrics[m]['activity_clarified'][qt]) if qt in metrics[m]['activity_clarified'] else np.nan for m in order], yerr=[np.std(metrics[m]['activity_clarified'][qt]) if qt in metrics[m]['activity_clarified'] else np.nan for m in order], label=f"{qt}")
                ax[1].set_title('Clarified Activity')
                ax[1].legend()

            # ax[0].get_shared_y_axes().join(ax[0], ax[1])
            # ax[0].get_shared_x_axes().join(ax[0], ax[1])
            fig.tight_layout()
            fig.savefig(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','plots'+difficulty_type,'activity.png'))
            plt.close(fig)
            for i_ob_idx,(category, easyhard, label, index) in enumerate(ob_difftypes_labels):
                if category == difficulty_type:
                    def xoff(midx):
                        width = 0.8/len(order)
                        off = -0.4+width/2+width*midx
                        return off, width-0.02
                    for mi, method in enumerate(order):
                        qt = 'object' if method in ['original', 'default_0'] else 'both'
                        off, width = xoff(mi)
                        x = np.array(metrics[method]['relocation_clarified'][qt][easyhard])
                        x = x[x>0] if x.max() > 0 else x
                        label = method_labels[method].replace('\n',' ') if i_ob_idx==0 else None
                        p = ax_ob[2].bar(index-1+off+width/4, np.mean(x), \
                                                yerr=np.std(x), \
                                                color=color_palette[method], alpha=0.5, width=width/2, label=label)
                        ax_ob[2].bar_label(p, fmt='%.3f')
                        x = np.array(metrics[method]['relocation'][str(query_step)][easyhard])
                        x = x[x>0] if x.max() > 0 else x
                        p = ax_ob[2].bar(index-1+off-width/4, np.mean(x), \
                                                yerr=np.std(x), \
                                                color=color_palette[method], alpha=1.0, width=width/2, label=label)
                        ax_ob[2].bar_label(p, fmt='%.3f')
                        p = ax_ob_final3.bar(index-1+off, np.mean(x), \
                                                yerr=np.std(x), \
                                                color=color_palette[method], alpha=1.0, width=width, label=label)
                        ax_ob_final3.bar_label(p, rotation=90, fmt='%.2f')

        ax_ob_final1.set_xticks(np.arange(len(ob_difftypes_labels))+1)
        # ax_ob_final1.set_xticklabels(['Overall']+[(label+str(num_gt_in_splits[cat][str(query_step)][eh])) for cat,eh,label,_ in ob_difftypes_labels])
        ax_ob_final1.set_xticklabels([label for _,_,label,_ in ob_difftypes_labels])
        ax_ob_final1.set_ylabel('F-1 Score')

        ax_ob_final2.set_xticks([0])
        # ax_ob_final2.set_xticklabels(['Overall']+[(label+str(num_gt_in_splits[cat][str(query_step)][eh])) for cat,eh,label,_ in ob_difftypes_labels])
        ax_ob_final2.set_xticklabels(['Overall'])
        ax_ob_final2.set_ylabel('F-1 Score')
        ax_ob[1].set_xticks(np.arange(len(ob_difftypes_labels)+1))
        # ax_ob[1].set_xticklabels(['Overall']+[(label+str(num_gt_in_splits[cat][str(query_step)][eh])) for cat,eh,label,_ in ob_difftypes_labels])
        ax_ob[1].set_xticklabels(['Overall']+[label for _,_,label,_ in ob_difftypes_labels])
        ax_ob[1].set_ylabel('F-1 Score')
        ax_ob[2].set_xticks(np.arange(len(ob_difftypes_labels)))
        # ax_ob[2].set_xticklabels([(label+str(num_gt_in_splits[cat][str(query_step)][eh])) for cat,eh,label,_ in ob_difftypes_labels])
        ax_ob[2].set_xticklabels([label for _,_,label,_ in ob_difftypes_labels])
        ax_ob[2].set_ylabel('F-1 Score')
        ax_ob_final3.set_xticks(np.arange(len(ob_difftypes_labels)))
        # ax_ob_final3.set_xticklabels([(label+str(num_gt_in_splits[cat][str(query_step)][eh])) for cat,eh,label,_ in ob_difftypes_labels])
        ax_ob_final3.set_xticklabels([label for _,_,label,_ in ob_difftypes_labels])
        ax_ob_final3.set_ylabel('F-1 Score')
        ax_ob_final1.legend()
        ax_ob[1].legend()
        # ax_ob[2].legend()
        ax_ob_final3.legend()
        ax_ob_final1.set_ylim([0,0.8])
        ax_ob_final2.set_ylim([0,0.8])
        ax_ob_final3.set_ylim([0,0.85])
        ax_ob[2].get_shared_y_axes().join(ax_ob[2], ax_ob[1])
        ax_ob[3].get_shared_y_axes().join(ax_ob[3], ax_ob[1])
        ax_st[0].set_xticks(np.arange(len(ob_difftypes_labels)+1))
        ax_st[0].set_xticklabels(['Overall']+[(label+str(num_gt_in_splits[cat][str(query_step)][eh])) for cat,eh,label,_ in ob_difftypes_labels])
        ax_st[0].set_ylabel('F-1 Score across proactivity windows with Activity')
        ax_st[1].set_xticks(np.arange(len(ob_difftypes_labels)+1))
        ax_st[1].set_xticklabels(['Overall']+[label for _,_,label,_ in ob_difftypes_labels])
        ax_st[1].set_ylabel('Lenient F-1 Score across proactivity windows')
        ax_st[0].set_ylim((0,1))
        ax_st[1].set_ylim((0,1))

        fig_ob.tight_layout()
        fig_ob.savefig(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','object_types.png'))

        fig_ob_final1.tight_layout()
        fig_ob_final1.savefig(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','final1.png'))
        fig_ob_final2.tight_layout()
        fig_ob_final2.savefig(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','final2.png'))
        fig_ob_final3.tight_layout()
        fig_ob_final3.savefig(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','final3.png'))
        fig_st.tight_layout()
        fig_st.savefig(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','steps.png'))
        plt.close(fig_ob)
        plt.close(fig_st)

        firstdir=False


        query_counts = {m: {qt: (sum(metrics[m]['relocation_clarified'][qt]['num_queries']), metrics[m]['relocation_clarified'][qt]['num_queries']) 
                            for qt in metrics[m]['relocation_clarified']}
                        for m in order}
            
        json.dump({'query_counts': query_counts, 'metrics':metrics, 'metrics_lenient':metrics_lenient, 'num_in_splits':num_gt_in_splits}, open(os.path.join(dirpath,f'plots_{METRIC_TO_PLOT}_{LENIENCY}len','consolidated_metrics.json'),'w'), indent=4)
    
    ax_st_all[0].set_xticks(np.arange(len(ob_difftypes_labels)+1))
    ax_st_all[0].set_xticklabels(['Overall']+[label for _,_,label,_ in ob_difftypes_labels])
    ax_st_all[0].set_ylabel('F-1 Score across proactivity windows with Activity')
    ax_st_all[1].set_xticks(np.arange(len(ob_difftypes_labels)+1))
    ax_st_all[1].set_xticklabels(['Overall']+[label for _,_,label,_ in ob_difftypes_labels])
    ax_st_all[1].set_ylabel('Lenient F-1 Score across proactivity windows')
    ax_st_all[0].set_ylim((0,1))
    ax_st_all[1].set_ylim((0,1))
    fig_st_all.tight_layout()
    os.makedirs(os.path.join(args.path,f'plots_{args.log_dir_prefix}'), exist_ok=True)
    fig_st_all.savefig(os.path.join(args.path,f'plots_{args.log_dir_prefix}','steps.png'))
