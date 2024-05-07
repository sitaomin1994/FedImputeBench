from src.imputation.imputers import (
    MIWAEImputer,
    LinearICEImputer,
    SimpleImputer,
    LinearICEGradImputer,
    MLPICEImputer,
)


def load_imputer(name, imputer_params):
    if name == 'miwae':
        return MIWAEImputer(name='miwae', **imputer_params)
    elif name == 'notmiwae':
        return MIWAEImputer(name='notmiwae', **imputer_params)
    elif name == 'linear_ice':
        return LinearICEImputer(**imputer_params)
    elif name == 'simple':
        return SimpleImputer(**imputer_params)
    elif name == 'linear_ice_sgd':
        return LinearICEGradImputer(**imputer_params)
    elif name == 'mlp_ice':
        return MLPICEImputer(**imputer_params)
    elif name == 'linear_sgd_ice':
        return MLPICEImputer(**imputer_params)
    else:
        raise NotImplementedError
