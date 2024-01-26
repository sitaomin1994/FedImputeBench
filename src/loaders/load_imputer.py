from src.imputation.miwae_imputer import MIWAEImputer
from src.imputation.notmiwae_imputer import NOTMIWAEImputer


def load_imputer(name, model_params):
    if name == 'miwae':
        return MIWAEImputer(model_params)
    elif name == 'not-miwae':
        return NOTMIWAEImputer(model_params)
    elif name == 'mean':
        raise NotImplementedError
    elif name == 'mice_ridge':
        raise NotImplementedError
    else:
        raise NotImplementedError


def load_imputation_models(client_train_data_ms_list, imp_model_name, imp_model_params, seeds):
    num_clients = len(client_train_data_ms_list)
    imputers = []
    for i in range(num_clients):
        imp_model_params['num_features'] = client_train_data_ms_list[i].shape[1] - 1
        imp_model_params['seed'] = seeds[i]
        imputers.append(load_imputer(imp_model_name, imp_model_params))
    return imputers
