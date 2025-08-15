import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logs_dir_list = [os.path.join(d,os.listdir(d)[0],'default_100') for d in os.listdir('.') if 'logs_0813_variations' in d]
logs_dir_list.sort()

categories = {'real_only','real_and_gpt','gpt_only','gpt_valid'}

colors_by_category = {
    'real_only': '#a4133c',
    'real_and_gpt': '#40916c',
    'gpt_only': '#0077b6',
    'gpt_valid': '#005f73',
}

colors_by_variation = {
    'gpt_only_split_0': '#03045e',
    'gpt_only_split_1': '#0077b6',
    'gpt_only_split_2': '#00b4d8',
    'gpt_only_split_3': '#90e0ef',
    'gpt_only_split_4': '#caf0f8',
    'gpt_valid_split_6': '#005f73',
    'real_and_gpt_split_3': '#1b4332',
    'real_and_gpt_split_4': '#40916c',
    'real_and_gpt_split_5': '#74c69d',
    'real_and_gpt_split_6': '#b7e4c7',
    'real_only_split_0': '#590d22',
    'real_only_split_1': '#800f2f',
    'real_only_split_2': '#a4133c',
    'real_only_split_3': '#c9184a',
    'real_only_split_4': '#ff4d6d',
    'real_only_split_5': '#ff758f',
    'real_only_split_6': '#ffb3c1',
}

config = json.load(open(os.path.join(logs_dir_list[0], 'config.json')))

fig, axs = plt.subplots(3, 1, figsize=(15, 15))
results = {'precision': {c:{ep:[] for ep in config['epochs']} for c in categories}, 
           'recall': {c:{ep:[] for ep in config['epochs']} for c in categories}, 
           'f1': {c:{ep:[] for ep in config['epochs']} for c in categories}}
axs = {key: axs[i] for i, key in enumerate([k for k in results.keys() if k != 'epochs'])}

for logs_dir in logs_dir_list:
    category = [c for c in categories if c in logs_dir.replace('logs_0813_variations_', '').split('/')[0]][0]
    results_each = {'precision': {ep:[] for ep in config['epochs']}, 
                    'recall': {ep:[] for ep in config['epochs']}, 
                    'f1': {ep:[] for ep in config['epochs']}}
    for epoch in config['epochs']:
        json_file = os.path.join(logs_dir, f'Epoch{epoch}_test_evaluation_splits.json')
        if not os.path.exists(json_file): continue
        with open(json_file, 'r') as f:
            data = json.load(f)
        for key in results.keys():
            if key == 'epochs': continue
            results[key][category][epoch].append(data[key])

    labelname = logs_dir.replace('logs_0813_variations_', '').split('/')[0]
    for key in results.keys():
        if key == 'epochs': continue
        axs[key].plot(results_each[key].keys(), results_each[key].values(), label=labelname, color=colors_by_variation[labelname])
        axs[key].set_xlabel('Epoch')

for key in results.keys():
    if key == 'epochs': continue
    axs[key].legend(loc='lower right')
    axs[key].set_title(key)
plt.tight_layout()
plt.show()
plt.close()

fig, axs = plt.subplots(3, 1, figsize=(15, 15))
axs = {key: axs[i] for i, key in enumerate([k for k in results.keys()])}

for key in results.keys():
    for category in categories:
        axs[key].plot(results[key][category].keys(), results[key][category].values(), label=category, color=colors_by_category[category])
    axs[key].set_xlabel('Epoch')

for key in results.keys():
    axs[key].legend(loc='lower right')
    axs[key].set_title(key)
plt.tight_layout()
plt.show()
plt.close()



