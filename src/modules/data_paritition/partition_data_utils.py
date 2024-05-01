from typing import List, Union, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer


def binning_target(y, reg_bins, seed):
    """
    binning target variable
    :param y: target variable
    :param reg_bins: number of bins
    :param seed: random seed
    :return: binned target variable
    """
    assert reg_bins > 1, "reg_bins should be greater than 1"
    y = y.copy().reshape(-1, 1)
    est = KBinsDiscretizer(
        n_bins=reg_bins, encode='ordinal', strategy='uniform',
        subsample=None, random_state=seed
    )
    y = est.fit_transform(y).flatten()
    return y


def calculate_data_partition_stats(
        datas: List[np.ndarray], regression: bool, reg_bins: int = 50
):
    """
    Calculate the statistics of the data partition
    :param datas: list of data partitions
    :param regression: regression task
    :param reg_bins: regression bins
    :return: list of statistics
    """
    if regression:
        label = np.concatenate([data[:, -1] for data in datas]).reshape(-1, 1)
        binner = KBinsDiscretizer(
            n_bins=reg_bins, encode='ordinal', strategy='uniform', subsample=None, random_state=0
        )
        binner.fit(label)

        num_clients = len(datas)
        statistic = [[] for _ in range(num_clients)]
        for client, data in enumerate(datas):
            target = binner.transform(data[:, -1].reshape(-1, 1)).flatten()
            for i in np.unique(target):
                statistic[client].append((int(i), int(sum((target == i).tolist()))))

        for client in range(num_clients):
            print(f"Client {client}\t Size of data: {len(datas[client])}\t Labels: ",
                  np.unique(binner.transform(datas[client].reshape(-1, 1)).flatten()))
            print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
            print("-" * 50)
    else:
        num_clients = len(datas)
        statistic = [[] for _ in range(num_clients)]
        for client, data in enumerate(datas):
            for i in np.unique(data[:, -1]):
                statistic[client].append((int(i), int(sum((data[:, -1] == i).tolist()))))

        for client in range(num_clients):
            print(f"Client {client}\t Size of data: {len(datas[client])}\t Labels: ",
                  np.unique(datas[client][:, -1]))
            print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
            print("-" * 50)

    return statistic


def noniid_sample_dirichlet(
        num_population, n_clients, alpha, min_samples, max_samples, max_repeat_times=5e6, seed=None
):
    """
    Perform non-iid sampling using dirichlet distribution non-iidness control by alpha,
    larger alpha, more uniform distributed, smaller alpha, more skewed distributed
    :param num_population: number of samples in the population
    :param n_clients: number of clients
    :param alpha: dirichlet distribution parameter
    :param min_samples: minimum number of samples in each client
    :param max_samples: maximum number of samples in each client
    :param max_repeat_times: maximum number of times to repeat the sampling
    :return: list of number of samples in each client
    """

    min_size = 0
    max_size = np.inf
    repeat_times = 0
    sizes = None

    while min_size < min_samples or max_size > max_samples:
        repeat_times += 1
        np.random.seed(seed + repeat_times)
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        sizes = num_population * proportions
        min_size = min(sizes)
        max_size = max(sizes)
        if repeat_times > max_repeat_times:
            print('max repeat times reached')
            raise ValueError('max repeat times reached')

    print(repeat_times, sizes)
    return list(sizes)


def generate_samples_iid(
        data: np.ndarray, sample_fracs: List[float], seed: int, regression: bool = False,
        reg_bins: Union[None, int] = None,
) -> List[np.ndarray]:
    """
    generate samples iid
    :param data: global data array
    :param sample_fracs: sample size for each client
    :param seed: randon seed
    :param regression: whether regression task
    :param reg_bins: bins for regression target
    :return: list of data array for each client
    """

    ret = []

    # set features and target
    X, y = data[:, :-1], data[:, -1]
    if regression and reg_bins is not None:
        y = binning_target(y, reg_bins)

    # split using sample_fracs and train_test_split
    for idx, sample_frac in enumerate(sample_fracs):
        new_seed = (seed + idx * 990983) % (2 ^ 32)
        if sample_frac == 1.0:
            ret.append(data.copy())
        else:
            # new_seed = seed
            if regression and reg_bins is None:
                _, X_test, _, y_test = train_test_split(
                    X, y, test_size=sample_frac, random_state=(new_seed) % (2 ** 32)
                )
            else:
                _, X_test, _, y_test = train_test_split(
                    X, y, test_size=sample_frac, random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
                )
            ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())

    return ret


def generate_local_test_data(
        datas: np.ndarray, local_test_size: float, regression: bool, seed: int = 201030
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate local test data
    :param datas: list of data
    :param local_test_size: local test data ratio
    :param regression: regression task or not
    :param seed: randomness
    :return: tuple of train and test datas
    """

    train_datas, test_datas = [], []
    for data in datas:
        if not regression:
            train_data, test_data = train_test_split(
                data, test_size=local_test_size, random_state=seed, stratify=data[:, -1]
            )
        else:
            train_data, test_data = train_test_split(
                data, test_size=local_test_size, random_state=seed
            )

        train_datas.append(train_data)
        test_datas.append(test_data)

    return train_datas, test_datas


def generate_global_test_data(
        data: np.ndarray, data_config: dict, test_size: float = 0.2, seed: int = 42):

    """
    Split data into train and test set
    """

    task_type = data_config['task_type']
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    if task_type == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    train_data = np.concatenate([X_train, y_train], axis=1)
    test_data = np.concatenate([X_test, y_test], axis=1)

    return train_data, test_data
