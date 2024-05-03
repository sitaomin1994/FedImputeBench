from typing import List

import numpy as np

from src.client import Client
from copy import deepcopy
import gc

def formulate_centralized_client(clients: List[Client]) -> Client:
    """
    Formulate a centralized client
    :param clients: list of clients
    :return: centralized client
    """
    centralized_client = deepcopy(clients[0])

    # reset client id and datas
    centralized_client.client_id = len(clients)
    centralized_train_datas, centralized_test_datas, centralized_ms_datas, centralized_masks = [], [], [], []
    for client in clients:
        centralized_train_datas.append(np.concatenate([client.X_train, client.y_train.reshape(-1, 1)], axis=1))
        centralized_test_datas.append(np.concatenate([client.X_test, client.y_test.reshape(-1, 1)], axis=1))
        centralized_ms_datas.append(client.X_train_ms)
        centralized_masks.append(client.X_train_mask)

    centralized_train_data = np.concatenate(centralized_train_datas, axis=0)
    centralized_test_data = np.concatenate(centralized_test_datas, axis=0)
    centralized_ms_data = np.concatenate(centralized_ms_datas, axis=0)
    centralized_mask = np.concatenate(centralized_masks, axis=0)

    centralized_client.X_train = centralized_train_data[:, :-1]
    centralized_client.y_train = centralized_train_data[:, -1]
    centralized_client.X_test = centralized_test_data[:, :-1]
    centralized_client.y_test = centralized_test_data[:, -1]
    centralized_client.X_train_ms = centralized_ms_data
    centralized_client.X_train_mask = centralized_mask
    centralized_client.X_train_imp = centralized_ms_data.copy()

    print(
        f"Centralized Client - Train data shape: {centralized_client.X_train.shape}, Test data shape: {centralized_client.X_test.shape}"
        f"Missing data shape: {centralized_client.X_train_ms.shape}, Missing data mask shape: {centralized_client.X_train_mask.shape}"
    )

    return centralized_client
