from emf.params_utils import parse_strategy_params
from src.modules.data_paritition.sample_data import generate_samples_iid
from src.modules.data_paritition.noniid_utils import noniid_sample_dirichlet
import numpy as np
import random


def load_data_partition(strategy, data, n_clients, params, seed=201030):
    strategy, strategy_params = parse_strategy_params(strategy)

    ####################################################################################################################
    # IID data partition
    ####################################################################################################################

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
    
    
def separate_data(
    data, num_clients, niid=False, balance=False, partition='dir', class_per_client=None, 
    alpha = 0.1, least_samples = 50, local_test_ratio = 0.1, seed = 201030
):
    
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data[:, :-1], data[:, -1]
    num_classes = len(np.unique(dataset_label))
    # guarantee that each client must have at least one batch of data for testing. 
    least_samples = int(min(least_samples / (1-local_test_ratio), len(dataset_label) / num_clients / 2)) # ?
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
            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_clients = [[] for _ in range(num_clients)]
            for class_id in range(num_classes):
                class_indices = np.where(dataset_label == class_id)[0]
                
                # split classes indices into num_clients parts
                np.random.seed(seed)
                np.random.shuffle(class_indices)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_client)<N/num_clients) for p, idx_client in zip(proportions, idx_clients)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(class_indices)).astype(int)[:-1]   # [100, 110, 113, 135, 235, ..., 100]
                splited_idx = np.split(class_indices,proportions)
                idx_clients = [idx_client + idx.tolist() for idx_client, idx in zip(idx_clients, splited_idx)]
                min_size = min([len(item) for item in idx_clients])
            
            seed = (seed + 1)%(2^31-1)
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_clients[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
            
    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic
