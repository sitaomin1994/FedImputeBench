import numpy as np
import torch
import os
import random


def setup_clients_seed(num_clients, rng: np.random.Generator):
    """
    Setup seeds for each client
    """
    # Set seeds for each client
    client_seeds = rng.integers(0, 10000, num_clients)
    print(client_seeds)
    return list(client_seeds)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)
