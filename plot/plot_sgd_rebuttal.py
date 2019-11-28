import wandb
import matplotlib.pyplot as plt
import ipdb
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--redo', type=int, default=0)
args = parser.parse_args()


def get_value(config, key):
    value = config[key]
    if type(value) is dict:
        value = value['value']  # happens when value is 0 for some reason
    return value


experiment_type = 'sgd'

if not os.path.isfile(f'./data/{experiment_type}.json') or args.redo:
    api = wandb.Api()
    runs = api.runs(
        'kylehsu/pacbayes_opt',
        {
            '$and': [
                {'tags': 'rebuttal'},
                {'tags': experiment_type},
            ]
        }
    )


    def extract_summary_from_history(run, metric='error_bound'):
        df = run.history()
        idx = df.idxmin(axis=0)[metric]
        series = df.iloc[idx]
        return series


    config_keys = [
        'seed',
        'dataset',
        'oracle_prior_variance',
        'posterior_mean_stopping_error_train',
        'alpha',
        'net_type'
    ]
    series_list = []
    for run in tqdm(runs):
        config = run.config
        series = extract_summary_from_history(run, 'error_bound').append(pd.Series({key: get_value(config, key)
                                                                                    for key in config_keys}))
        series_list.append(series)

    df = pd.DataFrame(series_list)

    df.to_json(f'./data/{experiment_type}.json')
else:
    df = pd.read_json(f'./data/{experiment_type}.json')

df['error_bound'] = df['error_bound'].clip(upper=1)
# for o in [0, 1]:
for o in [0]:
    df_o = df.loc[df['oracle_prior_variance'] == o]

    colors = ['tab:orange', 'tab:blue', 'tab:green']
    markers = ['o', 'v', 's']
    plt.rcParams.update({'font.size': 16, 'font.family': 'serif'})

    for dataset, net_type in [('mnist', 'lenet'), ('mnist', 'mlp'), ('fashion_mnist', 'lenet')]:
        fig = plt.figure()
        fig.set_size_inches(w=10, h=6, forward=True)
        fig.set_dpi(100)
        ax = plt.gca()

        df_dn = df_o.loc[(df['dataset'] == dataset) & (df['net_type'] == net_type)]
        ps = np.sort(df_dn['posterior_mean_stopping_error_train'].unique())
        for p, c, m in zip(ps, colors[:len(ps)], markers[:len(ps)]):
            df_p = df_dn.loc[df_dn['posterior_mean_stopping_error_train'] == p]
            gb = df_p.groupby(['alpha'])['error_bound']
            mean = gb.mean()
            std = gb.std()

            ax.plot(mean.index, mean, linestyle='-', color=c, marker=m, markersize=10, label=f'{p:.3f}')
            ax.fill_between(mean.index, mean - 2 * std, mean + 2 * std, alpha=0.2, color=c)

            gb = df_p.groupby(['alpha'])['error_test']
            mean = gb.mean()
            std = gb.std()

            ax.plot(mean.index, mean, linestyle='--', color=c, marker=m, markersize=10)
            ax.fill_between(mean.index, mean - 2 * std, mean + 2 * std, alpha=0.2, color=c)

        ax.set_xticks([0, 0.1, 0.2, 0.5, 0.7, 0.9])
        ax.set_ylabel('0-1 error')
        ax.set_xlabel(r'$\alpha$')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.grid()
        ax.legend()

        file_name = f'./output/{experiment_type}_{dataset}_{net_type}'
        if o:
            file_name += '_oracle'

        plt.savefig(file_name + '_rebuttal.pdf', bbox_inches='tight')
        plt.close()

