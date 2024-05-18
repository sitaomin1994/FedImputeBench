import json
import warnings
import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
from omegaconf import DictConfig, OmegaConf
import hydra
from emf.reproduce_utils import setup_seeds
from src.loaders.load_scenario import simulate_scenario
from src.loaders.load_data import load_data
import gc
from config import settings
import numpy as np


@hydra.main(version_base=None, config_path="config", config_name="scenario_config")
def my_app(cfg: DictConfig) -> None:
    config_dict: dict = OmegaConf.to_container(cfg, resolve=True)

    scenario_name = config_dict['scenario_name']
    scenario_version = config_dict['scenario_version']
    seed = config_dict['seed']
    num_rounds = config_dict['n_rounds']
    num_clients = config_dict['num_clients']
    dataset_name = config_dict['dataset_name']

    # generate seeds for each round
    seeds = setup_seeds(seed, num_rounds)
    print(scenario_name, scenario_version, seeds)
    meta_info = {
        'scenario_name': scenario_name,
        'scenario_version': scenario_version,
        'seed': seed,
        'num_rounds': num_rounds,
        'num_clients': num_clients,
        'dataset_name': dataset_name,
        'data_partition': config_dict['data_partition'],
        'missing_scenario': config_dict['missing_scenario'],
        'seeds': seeds,
    }
    scenario_dir_path = os.path.join(settings['scenario_dir'], scenario_version, dataset_name, scenario_name)
    if not os.path.exists(scenario_dir_path):
        os.makedirs(scenario_dir_path)

    with open(os.path.join(scenario_dir_path, 'meta_info.json'), 'w') as f:
        json.dump(meta_info, f)

    for round_id, seed in enumerate(seeds):
        ###########################################################################################################
        data, data_config = load_data(dataset_name)

        ###########################################################################################################
        # Scenario setup
        data_partition_params = config_dict['data_partition']['partition_params']
        missing_scenario_params = config_dict['missing_scenario']['params']

        clients_data, global_test_data, client_seeds, stats = simulate_scenario(
            data.values, data_config, num_clients, data_partition_params, missing_scenario_params, seed
        )

        save_scenario_data(
            scenario_dir_path, round_id, seed, clients_data, global_test_data, client_seeds, stats
        )

        del data, data_config, clients_data, global_test_data, client_seeds, stats
        gc.collect()


def save_scenario_data(
        scenario_dir_path: str, round_num: int, seed: int,
        clients_data: list, global_test_data: np.ndarray, client_seeds: list, stats: list
):

    dir_path = os.path.join(scenario_dir_path, str(round_num))
    print(dir_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # data
    clients_train_data, clients_test_data, clients_ms_data = {}, {}, {}
    for client_id, client_data in enumerate(clients_data):
        clients_train_data[str(client_id)] = client_data[0]
        clients_test_data[str(client_id)] = client_data[1]
        clients_ms_data[str(client_id)] = client_data[2]

    np.savez_compressed(os.path.join(dir_path, 'clients_train_data.npz'), **clients_train_data)
    np.savez_compressed(os.path.join(dir_path, 'clients_test_data.npz'), **clients_test_data)
    np.savez_compressed(os.path.join(dir_path, 'clients_train_data_ms.npz'), **clients_ms_data)
    np.savez_compressed(os.path.join(dir_path, 'global_test_data.npz'), global_test=global_test_data)

    # information
    stats_dict = {
        'scenario_path': scenario_dir_path,
        'round_num': round_num,
        'seed': seed,
        'client_seeds': [int(item) for item in client_seeds],
        'num_clients': len(clients_data),
        'class_distribution': stats,
    }

    with open(os.path.join(dir_path, 'stats.json'), 'w') as f:
        json.dump(stats_dict, f)


if __name__ == "__main__":
    my_app()
