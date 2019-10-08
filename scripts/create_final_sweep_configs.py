import wandb_summarizer.download
import ipdb
import wandb
import yaml
n_seeds = 10

api = wandb.Api()
# experiment_type = 'direct'
# runs = api.runs(
#     'kylehsu/pacbayes_opt',
#     {
#         '$and': [
#             {'tags': 'best'},
#             {'tags': experiment_type}
#         ]
#     }
# )

exclude_keys = [
    '_wandb',
    'wandb_version',
    'debug',
    # 'hidden_layer_sizes',
    'dataset_path'
]

def get_value(config, key):
    value = config[key]
    if type(value) is dict:
        value = value['value']  # happens when value is 0 for some reason
    return value

# for run in runs:
#     run_config = run.config
#     config = {}
#     config.update({'parameters': {key: {'value': get_value(run_config, key)} for key in run_config.keys() if key not in exclude_keys}})
#     config['parameters'].update({'seed': {'values': [i + 1 for i in range(n_seeds)]}})
#     config['parameters'].update({'bound_optimization_patience': {'value': 10}})
#     alpha = get_value(run_config, 'alpha')
#     dataset = get_value(run_config, 'dataset')
#     sweep_name = f"exp-{experiment_type}__dataset-{dataset}__alpha-{alpha:.2f}"
#     file_name = f"./sweeps/final/{sweep_name}.yaml"
#     config.update({
#         'name': sweep_name,
#         'program': f'scripts/data_dependent_prior_{experiment_type}.py',
#         'method': 'grid',
#     })
#
#     with open(file_name, 'w') as f:
#         yaml.dump(config, f)
#     print(file_name)


experiment_type = 'sgd'
runs = api.runs(
    'kylehsu/pacbayes_opt',
    {
        '$and': [
            {'tags': 'best'},
            {'tags': experiment_type},
        ]
    }
)
for run in runs:
    run_config = run.config
    if get_value(run_config, 'net_type') != 'mlp':
        continue
    config = {}
    config.update({'parameters': {key: {'value': get_value(run_config, key)} for key in run_config.keys() if key not in exclude_keys}})
    config['parameters'].update({'seed': {'values': [i + 1 for i in range(n_seeds)]}})
    # config['parameters'].update({'bound_optimization_patience': {'value': 10}})
    config['parameters'].update({'oracle_prior_variance': {'values': [0, 1]}})
    net_type = get_value(run_config, 'net_type')
    alpha = get_value(run_config, 'alpha')
    dataset = get_value(run_config, 'dataset')
    posterior_mean_stopping_error_train = get_value(run_config, 'posterior_mean_stopping_error_train')
    sweep_name = f"exp-{experiment_type}__dataset-{dataset}__alpha-{alpha:.1f}__net_type-{net_type}__posterior_mean_stopping_error_train-{posterior_mean_stopping_error_train}"
    file_name = f"./sweeps/final/{sweep_name}.yaml"
    config.update({
        'name': sweep_name,
        'program': f'scripts/data_dependent_prior_{experiment_type}.py',
        'method': 'grid',
    })

    with open(file_name, 'w') as f:
        yaml.dump(config, f)
    print(file_name)



