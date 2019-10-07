import wandb
import matplotlib.pyplot as plt
import ipdb
import pandas as pd

api = wandb.Api()
experiment_type = 'direct'
runs = api.runs(
    'kylehsu/pacbayes_opt',
    {
        '$and': [
            {'tags': 'final'},
            {'tags': experiment_type}
        ]
    }
)
ipdb.set_trace()
df = pd.DataFrame.from_dict()


def extract_summary_from_history(run, metric='error_bound'):
    df = run.history()
    idx = df.idxmin(axis=0)[metric]
    series = df.iloc[idx]
    return series

