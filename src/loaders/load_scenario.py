from typing import Tuple, List

import numpy as np
from src.loaders.load_data_partition import load_data_partition
from src.modules.missing_simulate.add_missing import add_missing
from src.utils.setup_seeds import setup_clients_seed


def simulate_scenario(
        data: np.array, data_config:dict, num_clients,
        data_partition_params: dict,
        missing_simulate_params: dict,
        seed: int = 100330201
) -> Tuple[List[Tuple[np.array, np.ndarray, np.ndarray]], np.array, List[int], List[List[Tuple[int, int]]]]:
    """
    Simulate missing data scenario
    :param data: data
    :param data_config: data configuration
    :param num_clients: number of clients
    :param data_partition_params: data partition parameters
        - partition_strategy: partition strategy - iid, niid_dir
        - size_strategy: size strategy - even, even2, dir, hs
        - size_niid_alpha: size niid alpha
        - min_samples: minimum samples
        - max_samples: maximum samples
        - niid_alpha: non-iid alpha dirichlet
        - even_sample_size: even sample size
        - local_test_size: local test ratio
        - global_test_size: global test ratio
        - reg_bins: regression bins
    :param missing_simulate_params: missing data simulation parameters
        - global_missing: whether simulate missing data globally or locally
        - mf_strategy: missing features strategy - all
        - mr_dist: missing ratio distribution - fixed, uniform, uniform_int, gaussian, gaussian_int
        - mr_lower: missing ratio lower bound
        - mr_upper: missing ratio upper bound
        - mm_funcs_dist: missing mechanism functions distribution - identity, random, random2,
        - mm_funcs_bank: missing mechanism functions banks - None, 'lr', 'mt', 'all'
        - mm_mech: missing mechanism - 'mcar', 'mar_quantile', 'mar_sigmoid', 'mnar_quantile', 'mnar_sigmoid'
        - mm_strictness: missing adding probailistic or deterministic
        - mm_obs:  missing adding based on observed data
        - mm_feature_option: missing mechanism associated with which features - self, all, allk=0.1
        - mm_beta_option: mechanism beta coefficient option - (mnar) self, sphere, randu, (mar) fixed, randu, randn
    :param seed:
    :return:
    """

    # ========================================================================================
    # setup clients seeds
    client_seeds = setup_clients_seed(num_clients, seed=seed)

    # ========================================================================================
    # data partition
    clients_train_data_list, clients_test_data_list, global_test_data, stats = load_data_partition(
        data, data_config, num_clients, seed=seed, **data_partition_params
    )

    # ========================================================================================
    # simulate missing data globally
    if 'ms_cols' in data_config:
        cols = data_config['ms_cols']
    else:
        cols = np.arange(data.shape[1] - 1)
    client_train_data_ms_list = add_missing(
        clients_train_data_list, cols, client_seeds, seed=seed, **missing_simulate_params
    )

    # ========================================================================================
    # organize results
    clients_data = []
    for i in range(num_clients):
        clients_data.append(
            (clients_train_data_list[i], clients_test_data_list[i], client_train_data_ms_list[i])
        )

    return clients_data, global_test_data, client_seeds, stats
