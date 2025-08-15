import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize(logs_dir, config):
    results = {'epochs':[], 'precision': [], 'recall': [], 'f1': []}

    for epoch in config['epochs']:
        # if epoch > 90: continue
        json_file = os.path.join(logs_dir, f'Epoch{epoch}_test_evaluation_splits.json')
        with open(json_file, 'r') as f:
            data = json.load(f)
        results['epochs'].append(epoch)
        for key in results.keys():
            if key == 'epochs': continue
            results[key].append(data[key])

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    for i, key in enumerate(results.keys()):
        if key == 'epochs': continue
        ax.plot(results['epochs'], results[key], label=key)
        ax.set_xlabel('Epoch')

    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(logs_dir, 'f1_over_epochs.png'))

