from typing import List, Tuple

from emf.params_utils import parse_strategy_params
from src.modules.data_paritition.partition_data import (
    separate_data_niid
)
from src.modules.data_paritition.partition_data_utils import (
    calculate_data_partition_stats, generate_local_test_data, generate_samples_iid, noniid_sample_dirichlet,
    generate_global_test_data
)
import numpy as np
import random
import gc


def load_data_partition(
        data, data_config, num_clients, partition_strategy,
        size_strategy: str = 'even',
        size_niid_alpha: float = 0.2,
        min_samples: int = 100,
        max_samples: int = 2000,
        niid_alpha: float = 0.2,
        even_sample_size: int = 1000,
        local_test_size: float = 0.1,
        global_test_size: float = 0.1,
        reg_bins: int = 50,
        rng: np.random.Generator = np.random.default_rng(42),
        seed=201031
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, List[List[Tuple[int, int]]]]:
    """
    Load data partition
    :param rng: numpy random generator
    :param data: data
    :param data_config: data configuration
    :param num_clients: number of clients
    :param partition_strategy: partition strategy
    :param size_strategy: size strategy
    :param size_niid_alpha: size niid alpha
    :param min_samples: minimum samples
    :param max_samples: maximum samples
    :param niid_alpha: non-iid alpha dirichlet distribution parameter
    :param even_sample_size: even sample size
    :param local_test_size: local test ratio
    :param global_test_size: global test ratio
    :param reg_bins: regression bins
    :param seed: random seed
    :return: List of training data, List of test data, global test data, statistics
    """


    #############################################################################################################
    # split a global test data
    train_data, test_data = generate_global_test_data(data, data_config, test_size=global_test_size, seed=seed)

    #############################################################################################################
    # partition data
    # iid partition

    if partition_strategy == 'iid':
        # evenly partition data
        if size_strategy == 'even':
            sample_fracs = [1 / num_clients for _ in range(num_clients)]
        # even partition with fixed sample size
        elif size_strategy == 'even2':
            sample_fracs = [even_sample_size / data.shape[0] for _ in range(num_clients)]
        elif size_strategy == 'random_uniform':
            sample_fracs = rng.uniform(min_samples / data.shape[0], max_samples / data.shape[0], num_clients).tolist()
        # dirichlet distribution
        elif size_strategy == 'dir':
            if max_samples == -1:
                max_samples = data.shape[0]
            sizes = noniid_sample_dirichlet(
                data.shape[0], num_clients, size_niid_alpha, min_samples, max_samples, rng = rng
            )
            sample_fracs = [size / data.shape[0] for size in sizes]
        # hub and spoke
        elif size_strategy == 'hs':
            sample_fracs = [0.5] + [0.05 for _ in range(num_clients - 1)]
            rng.shuffle(sample_fracs)
        else:
            raise NotImplementedError

        regression = data_config['task_type'] == 'regression'
        datas = generate_samples_iid(train_data, sample_fracs, regression=regression, reg_bins=reg_bins, seed=seed)

    # non-iid partition
    elif partition_strategy == 'niid_dir':
        datas = separate_data_niid(
            train_data, data_config, num_clients, niid=True, partition='dir', balance=False, class_per_client=None,
            niid_alpha=niid_alpha, min_samples=min_samples, reg_bins=reg_bins, seed=seed, rng = rng
        )
    else:
        raise ValueError('Strategy not found')

    del train_data
    gc.collect()

    #############################################################################################################
    # calculate statistics
    regression = data_config['task_type'] == 'regression'
    statistics = calculate_data_partition_stats(datas, regression=regression)

    #############################################################################################################
    # generate local test data
    train_datas, test_datas = generate_local_test_data(
        datas, local_test_size, regression=regression, seed=seed
    )

    return train_datas, test_datas, test_data, statistics
