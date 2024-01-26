from typing import List
import numpy as np
from .strategies import local, fedavg


########################################################################################################################
# Initial Imputation for Numerical Columns
def initial_imputation_num(strategy, clients_data_utils: List[dict]) -> List[np.ndarray]:
    if strategy == 'local_mean':
        return local(clients_data_utils, key='mean', col_type='num')
    elif strategy == 'local_median':
        return local(clients_data_utils, key='median', col_type='num')
    elif strategy == 'local_zero':
        raise local(clients_data_utils, key='zero', col_type='num')
    elif strategy == 'fedavg_mean':
        return fedavg(clients_data_utils, key='mean')
    elif strategy == 'fedavg_median':
        raise fedavg(clients_data_utils, key='median')
    elif strategy == 'complement_mean':
        raise NotImplemented
    elif strategy == 'complement_median':
        raise NotImplemented
    else:
        raise ValueError("strategy must be one of 'local_mean', 'local_median', 'local_zero', 'fedavg_mean', "
                         "'fedavg_median', 'complement_mean', 'complement_median'")


########################################################################################################################
# Initial Imputation for Categorical Columns
def initial_imputation_cat(strategy, clients_data_utils: List[dict]) -> List[np.ndarray]:
    if strategy == 'local_mode':
        return local(clients_data_utils, key='mode', col_type='cat')
    else:
        raise ValueError("strategy must be one of 'local_mode'")
