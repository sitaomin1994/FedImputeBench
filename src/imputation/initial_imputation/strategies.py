import numpy as np


########################################################################################################################
# Local Initial Imputation
########################################################################################################################
def local(clients_data_utils, key='mean', col_type='num'):
    if col_type == 'num':
        if key not in ['mean', 'median', 'zero']:
            raise ValueError("key must be one of 'mean', 'std', 'min', 'max'")

        if key == 'zero':
            clients_value_array = np.zeros((len(clients_data_utils), clients_data_utils[0]['num_cols']))
        else:
            clients_value_array = np.array(
                [[item['col_stats'][col][key] for col in range(item['num_cols'])] for item in clients_data_utils]
            )

    elif col_type == 'cat':
        if key not in ['mode']:
            raise ValueError("key must be one of 'mode'")

        if key == 'zero':
            clients_value_array = np.zeros(
                (len(clients_data_utils), clients_data_utils[0]['n_features'] - clients_data_utils[0]['num_cols'])
            )
        else:
            clients_value_array = np.array(
                [[item['col_stats'][col][key] for col in range(item['num_cols'], item['n_features'])] for item in
                 clients_data_utils]
            )
    else:
        raise ValueError("col_type must be one of 'num', 'cat'")

    return [item for item in clients_value_array]


########################################################################################################################
# FedAvg Initial Imputation
########################################################################################################################
def fedavg(clients_data_utils, key='mean'):
    if key not in ['mean', 'median']:
        raise ValueError("key must be one of 'mean', 'median'")

    # fetch sample sizes and mean values
    sample_sizes = [client_data_utils['n_samples'] for client_data_utils in clients_data_utils]  # (n_clients, )
    clients_value_array = np.array(
        [[item['col_stats'][col][key] for col in range(item['num_cols'])] for item in clients_data_utils]
    )  # (n_clients, num_cols)

    # calculate weighted average
    imp_value_avg = np.average(
        clients_value_array, axis=0, weights=sample_sizes
    )

    # assign imputation values
    clients_impute_values = [imp_value_avg] * len(clients_data_utils)
    return clients_impute_values