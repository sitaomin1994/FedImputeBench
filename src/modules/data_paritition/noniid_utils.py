import numpy as np


def noniid_sample_dirichlet(num_population, n_clients, alpha, min_samples, max_samples, max_repeat_times=5e6, seed = None):
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