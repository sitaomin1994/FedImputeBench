import numpy as np


def setup_clients_seed(num_clients, seed=0):
    """
    Setup seeds for each client
    """
    # Set seeds for each client
    client_seeds = np.random.randint(0, 10000, num_clients)
    return list(client_seeds)
