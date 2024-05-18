from sklearn.model_selection import train_test_split
import numpy as np
from emf.reproduce_utils import set_seed
from typing import List, Union, Tuple
from src.modules.data_paritition.partition_data_utils import (
    binning_target, binning_features, noniid_sample_dirichlet, generate_samples_iid
)

import gc


# def separate_data_iid(
#         data: np.ndarray, data_config: dict, num_clients: int, sample_size_option: str = 'even',
#         reg_bins: Union[None, int] = None, size_alpha: float = 0.2, min_samples: int = 400, max_samples: int = 2000,
#         even_sample_size: int = 500, seed: int = 201030,
# ) -> List[np.ndarray]:
#     if sample_size_option == 'even':
#         sample_fracs = [1 / num_clients for _ in range(num_clients)]
#     elif sample_size_option == 'even2':
#         sample_fracs = [even_sample_size / data.shape[0] for _ in range(num_clients)]
#     elif sample_size_option == 'dir':
#         if max_samples == -1:
#             max_samples = data.shape[0]
#         sizes = noniid_sample_dirichlet(
#             data.shape[0], num_clients, size_alpha, min_samples, max_samples, seed=seed
#         )
#         sample_fracs = [size / data.shape[0] for size in sizes]
#     elif sample_size_option == 'hs':
#         sample_fracs = [0.5] + [0.05 for _ in range(num_clients - 1)]
#         np.random.seed(seed)
#         np.random.shuffle(sample_fracs)
#     else:
#         raise NotImplementedError
#
#     regression = data_config['task'] == 'regression'
#     datas = generate_samples_iid(data, sample_fracs, regression=regression, reg_bins=reg_bins, seed=seed)
#
#     return datas


def separate_data_niid(
        data: np.ndarray, data_config: dict, num_clients: int, split_col_idx: Union[int, list] = -1,
        niid: bool = True, partition: str = 'dir', balance: bool = False,
        class_per_client: Union[None, int] = None, niid_alpha: float = 0.1,
        min_samples: int = 50, reg_bins: int = 20, seed: int = 201030
):
    rng = np.random.default_rng(seed)

    # split based on target
    if split_col_idx == -1:
        dataset_label = data[:, -1]
        if data_config['task_type'] == 'regression':  # if regression task, bin the target # TODO: refactor this
            dataset_label = binning_target(dataset_label, reg_bins, seed)
    # split based on feature
    else:
        # split based on one feature
        if not isinstance(split_col_idx, list):
            dataset_label = data[:, split_col_idx]
            if np.unique(dataset_label).shape[0] > reg_bins:
                dataset_label = binning_target(dataset_label, reg_bins, seed)
        # split based on multiple features (feature clustering)
        else:
            X = data[: split_col_idx]
            dataset_label = binning_features(X, reg_bins=10, seed=seed)

    dataset_content, target = data[:, :-1], data[:, -1]

    print(dataset_label)
    num_classes = len(np.unique(dataset_label))
    # guarantee that each client must have at least one batch of data for testing.
    min_samples = int(min(min_samples, int(len(dataset_label) / num_clients / 2)))  # ?
    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = rng.integers(
                    int(max(num_per / 10, min_samples / num_classes)), int(num_per), num_selected_clients - 1
                ).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(
                        dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample], axis=0
                    )
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        N = len(dataset_label)

        try_cnt = 1
        idx_clients = [[] for _ in range(num_clients)]
        # class_condition = False
        while (min_size < min_samples):
            # if try_cnt > 1:
            #     print(f'Client data size does not meet the minimum requirement {min_samples}. '
            #           f'Try allocating again for the {try_cnt}-th time.')

            idx_clients = [[] for _ in range(num_clients)]
            # all_class_condition = np.zeros(num_classes, dtype=bool)
            for class_id in range(num_classes):
                class_indices = np.where(dataset_label == class_id)[0]
                # split classes indices into num_clients parts
                rng.shuffle(class_indices)
                alphas = np.repeat(niid_alpha, num_clients)
                proportions = rng.dirichlet(alphas)
                proportions = np.array(
                    [p * (len(idx_client) < N / num_clients) for p, idx_client in zip(proportions, idx_clients)])
                proportions = proportions / proportions.sum()

                # limited numbers
                # num_array = (proportions * len(class_indices)).astype(int)
                # all_class_condition[class_id] = (num_array == 1).any()
                # print(class_id, num_array, all_class_condition[class_id])

                # [100, 110, 113, 135, 235, ..., 100]
                proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
                splited_idx = np.split(class_indices, proportions)

                # filter out classes only with one sample
                splited_idx_new = []
                for idx in splited_idx:
                    if len(idx) == 1:
                        splited_idx_new.append(np.array([], dtype=int))
                    else:
                        splited_idx_new.append(idx)

                idx_clients = [idx_client + idx.tolist() for idx_client, idx in zip(idx_clients, splited_idx_new)]
                min_size = min([len(item) for item in idx_clients])

            try_cnt += 1
            # class_condition = ~(all_class_condition.any())

        for j in range(num_clients):
            dataidx_map[j] = idx_clients[j]
    else:
        raise NotImplementedError

    # assign data
    datas = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        idxs = dataidx_map[client]
        datas[client] = np.concatenate([dataset_content[idxs], target[idxs].reshape(-1, 1)], axis=1).copy()

    return datas

#  elif strategy == 'iid-uneven10range':
#     random.seed(seed)
#     np.random.seed(seed)
#     sample_fracs = []
#     for i in range(3):
#         sample_fracs.append(random.randint(3000, 4000))
#     for i in range(3):
#         n1 = random.randint(1500, 2000)
#         sample_fracs.append(n1)
#     for i in range(4):
#         n1 = random.randint(500, 1000)
#         sample_fracs.append(n1)
#     np.random.shuffle(sample_fracs)
#
#     sample_fracs = [size / data.shape[0] for size in sample_fracs]
#     return generate_samples_iid(data, sample_fracs, seed)
#
# elif strategy == 'iid-uneven10hs':
#     sample_fracs = [9000 / 18000] + [1000 / 18000 for _ in range(n_clients - 1)]
#     np.random.seed(seed)
#     np.random.shuffle(sample_fracs)
#     return generate_samples_iid(data, sample_fracs, seed)
#
# elif strategy == 'iid-uneven10hsl':
#     sample_fracs = [9000 / 18000] + [1000 / 18000 for _ in range(n_clients - 1)]
#     np.random.seed(seed)
#     return generate_samples_iid(data, sample_fracs, seed)
#
# elif strategy == 'iid-uneven10hsr':
#     sample_fracs = [1000 / 18000 for _ in range(n_clients - 1)] + [9000 / 18000]
#     np.random.seed(seed)
#     return generate_samples_iid(data, sample_fracs, seed)
#
# elif strategy == 'iid-uneven10hsdir':
#     ratio = 0.2
#     n1_size = int(ratio * data.shape[0])
#     nrest_sizes = noniid_sample_dirichlet(data.shape[0] - n1_size, n_clients - 1, 0.1, 50, n1_size)
#     sample_fracs = [n1_size / data.shape[0]] + [size / data.shape[0] for size in nrest_sizes]
#     np.random.seed(seed)
#     np.random.shuffle(sample_fracs)
#     return generate_samples_iid(data, sample_fracs, seed)
