import os
import json

config = {
    # basic setting
    "num_clients": 2,
    "seed": 2180279,
    # dataset
    "data": {
        "dataset_name": "codon",
        "test_size": 0.1,
        "data_partition": 'iid-even2',
    },
    # "missing_simulate": 'fixed2@mr=0.5-mm=mnar_quantile-mfunc=lr-k=0.9',
    "missing_simulate": 'fixed2@mr=0.5-mm=mnar_quantile-mfunc=lr-k=0.5',
    # imputation models
    "imputation": {
        "imp_model": 'not-miwae',
        "imp_model_params": {
            "latent_size": 16,
            "n_hidden": 128,
            "K": 20
        },
        "imp_params": {
            "global_epochs": 1,
            "local_epochs": 5000,
            "batch_size": 128,
            "weight_decay": 0.001,
            "lr": 1e-3,
            "L": 1000,
            "imp_interval": 1,
            "verbose": 3,
            "verbose_interval": 1,
            "eval_interval": 1,
            "optimizer": "adam"
        },
    },
    # client
    "client": {
        "client_type": "ice",
        "client_config": {},
    },
    # "aggregation strategy
    "agg_strategy": {
        "agg_strategy_name": 'local',
        "agg_strategy_params": {},
    },
    # federated server
    "server": {
        "server_type": 'ice',
        "server_config": {},
        "server_imp_workflow_params": {
            "imp_iterations": 20,
            "initial_imp_num": "mean",
            "initial_imp_cat": "mode",
        },
    },
    # experiment settings
    "experiment": {
        "num_rounds": 1,
        "mtp": False
    }
}

exp_dir_name = '/'.join([config["dataset_name"], config['data_partition'], config['missing_simulate']])
exp_file_name = f"{config['imp_model']}_{config['server']}2"

experiment = Experiment(name='experiment')
rets = experiment.run(config)

# save results
results_dir = f"./results/raw_results/{exp_dir_name}/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
with open(f"{results_dir}/{exp_file_name}.json", 'w') as f:  # TODO: add config to files
    json.dump(rets, f)
