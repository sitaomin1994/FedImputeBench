from src.imputation.imputers import (
    MIWAEImputer,
    NOTMIWAEImputer,
    LinearICEImputer
)


def load_imputer(name, imputer_params):
    if name == 'miwae':
        return MIWAEImputer(imputer_params)
    elif name == 'not-miwae':
        return NOTMIWAEImputer(imputer_params)
    elif name == 'linear_ice':
        return LinearICEImputer(**imputer_params)
    elif name == 'mean':
        raise NotImplementedError
    else:
        raise NotImplementedError
