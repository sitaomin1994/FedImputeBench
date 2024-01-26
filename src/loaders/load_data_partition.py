from emf.params_utils import parse_strategy_params
from src.modules.data_paritition.sample_data import generate_samples_iid
from src.modules.data_paritition.noniid_utils import noniid_sample_dirichlet
import numpy as np
import random


def load_data_partition(strategy, data, n_clients, params, seed=201030):
    strategy, strategy_params = parse_strategy_params(strategy)

    if strategy == 'iid-full':
        return [data.copy() for _ in range(n_clients)]

    elif strategy == 'iid-even':
        sample_fracs = [1 / n_clients for _ in range(n_clients)]
        return generate_samples_iid(data, sample_fracs, seed)

    elif strategy == 'iid-even2':
        sample_fracs = [0.1 for _ in range(n_clients)]
        return generate_samples_iid(data, sample_fracs, seed)

    elif strategy == 'iid-uneven10dir':
        sizes = noniid_sample_dirichlet(
            data.shape[0], 10, 0.2, 400, data.shape[0] * 0.5, seed=seed
        )
        sample_fracs = [size / data.shape[0] for size in sizes]
        return generate_samples_iid(data, sample_fracs, seed)

    elif strategy == 'iid-uneven10range':
        random.seed(seed)
        np.random.seed(seed)
        sample_fracs = []
        for i in range(3):
            sample_fracs.append(random.randint(3000, 4000))
        for i in range(3):
            n1 = random.randint(1500, 2000)
            sample_fracs.append(n1)
        for i in range(4):
            n1 = random.randint(500, 1000)
            sample_fracs.append(n1)
        np.random.shuffle(sample_fracs)

        sample_fracs = [size / data.shape[0] for size in sample_fracs]
        return generate_samples_iid(data, sample_fracs, seed)

    elif strategy == 'iid-uneven10hs':
        sample_fracs = [9000 / 18000] + [1000 / 18000 for _ in range(n_clients - 1)]
        np.random.seed(seed)
        np.random.shuffle(sample_fracs)
        return generate_samples_iid(data, sample_fracs, seed)

    elif strategy == 'iid-uneven10hsl':
        sample_fracs = [9000 / 18000] + [1000 / 18000 for _ in range(n_clients - 1)]
        np.random.seed(seed)
        return generate_samples_iid(data, sample_fracs, seed)

    elif strategy == 'iid-uneven10hsr':
        sample_fracs = [1000 / 18000 for _ in range(n_clients - 1)] + [9000 / 18000]
        np.random.seed(seed)
        return generate_samples_iid(data, sample_fracs, seed)

    elif strategy == 'iid-uneven10hsdir':
        ratio = 0.2
        n1_size = int(ratio * data.shape[0])
        nrest_sizes = noniid_sample_dirichlet(data.shape[0] - n1_size, n_clients - 1, 0.1, 50, n1_size)
        sample_fracs = [n1_size / data.shape[0]] + [size / data.shape[0] for size in nrest_sizes]
        np.random.seed(seed)
        np.random.shuffle(sample_fracs)
        return generate_samples_iid(data, sample_fracs, seed)

    else:
        raise ValueError('Strategy not found')
