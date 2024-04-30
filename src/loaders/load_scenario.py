from typing import Tuple, List

import numpy as np
from src.loaders.load_data_partition import load_data_partition
from src.loaders.load_missing_simulation_dist import add_missing
from src.loaders.load_missing_simulation_cent import add_missing_central
from src.utils.setup_seeds import setup_clients_seed


def setup_scenario(
        train_data: np.array, data_config:dict, data_partition_strategy: str, ms_simulate_strategy: str,
        num_clients: int, seed: int
) -> Tuple[List[np.array], List[np.array], List[int]]:

    # ========================================================================================
    # setup clients seeds
    client_seeds = setup_clients_seed(num_clients, seed=seed)

    # ========================================================================================
    # data partition
    clients_train_data_list = load_data_partition(
        data_partition_strategy, train_data, num_clients, {}, seed=seed
    )

    # ========================================================================================
    # simulate missing data
    if 'ms_cols' in data_config:
        cols = data_config['ms_cols']
    else:
        cols = np.arange(train_data.shape[1] - 1)
    client_train_data_ms_list = add_missing(
        clients_train_data_list, ms_simulate_strategy, cols, seeds=client_seeds
    )

    return clients_train_data_list, client_train_data_ms_list, client_seeds


def load_scenario(scenario_name: str, train_data: np.ndarray, num_clients: int, seed: int = 201030) -> dict:
    ###################################################################################################################
    # Main benchmark scenarios
    # - missing ratio 0.3 - 0.7 truncated normal distribution sigma = 0.15
    # - noniid using direchilet distribution
    # - number of missing feature depends on datasets
    ###################################################################################################################
    # MCAR scenarios
    if scenario_name == 'mcar_dataiid':
        client_train_data_list, client_train_data_ms_list, client_seeds = setup_scenario(
            train_data=train_data,
            data_partition_strategy='iid-iid-uneven10dir',
            ms_simulate_strategy='mcar',
            num_clients=num_clients,
            seed=seed
        )

        return {
            'client_train_data_list': client_train_data_list,
            'client_train_data_ms_list': client_train_data_ms_list,
            'client_seeds': client_seeds
        }

    elif scenario_name == 'mcar_dataniid':
        raise NotImplementedError

    ###################################################################################################################
    # MAR scenarios
    elif scenario_name == 'mar_homo_dataiid':
        raise NotImplementedError
    elif scenario_name == 'mar_heter_dataiid':
        raise NotImplementedError
    elif scenario_name == 'mar_homo1_dataniid':
        # simulate on centralized data then split
        raise NotImplementedError
    elif scenario_name == 'mar_homo2_dataniid':
        # simulate on local data after splitting
        raise NotImplementedError
    elif scenario_name == 'mar_heter_dataniid':
        raise NotImplementedError

    ####################################################################################################################
    # MNAR scenarios
    elif scenario_name == 'mnar_homo_dataiid':
        raise NotImplementedError
    elif scenario_name == 'mnar_heter_dataiid':
        raise NotImplementedError
    elif scenario_name == 'mnar_homo1_dataniid':
        # simulate on centralized data then split
        raise NotImplementedError
    elif scenario_name == 'mnar_homo2_dataniid':
        # simulate on local data after splitting
        raise NotImplementedError
    elif scenario_name == 'mnar_heter_dataniid':
        raise NotImplementedError
    else:
        raise NotImplementedError(f"{scenario_name} not supported")
