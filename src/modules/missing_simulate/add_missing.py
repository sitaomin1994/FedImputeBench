import numpy as np
import src.modules.missing_simulate.mcar_simulate as mcar_simulate
import src.modules.missing_simulate.mar_simulate as mar_simulate
import src.modules.missing_simulate.mnar_simulate as mnar_simulate
from typing import List, Union
from emf.params_utils import parse_strategy_params


########################################################################################################################
# Simulate missing for one client
########################################################################################################################
def simulate_nan(
        X_train: np.ndarray, y_train: np.ndarray, cols: list, mech_type: str, missing_ratio: Union[float, list, dict],
        mechanism_func: Union[str, list, dict], seed: int = 201030
) -> np.ndarray:

    # mar_quantile@obs=True-strict=True
    # mar_sigmoid@obs=True-strict=True-association=all-beta=random_uniform
    '''
    params:
        - missing_mech: mcar, mar_quantile, mar_sigmoid, mnar_quantile, mnar_sigmoid
        - obs: True, False
        - strict: True, False
        - association: all, random, random_uniform
        - beta: random, random_uniform

    '''
    mech_type, mech_params = parse_strategy_params(mech_type)

    if mech_type == 'mcar':
        X_train_ms = mcar_simulate.simulate_nan_mcar(X_train, cols, missing_ratio, seed=seed)
    elif mech_type == 'mar_quantile':
        X_train_ms = mar_simulate.simulate_nan_mar_quantile(
            X_train, cols, missing_ratio=missing_ratio, missing_func=mechanism_func, seed=seed, **mech_params
        )

    elif mech_type == 'mnar_quantile':
        X_train_ms = mnar_simulate.simulate_nan_mnar_quantile(
            X_train, cols, missing_ratio=missing_ratio, missing_func=mechanism_func, seed=seed
        )
    elif mech_type == 'mnar_sigmoid':
        X_train_ms = mnar_simulate.simulate_nan_mnar_sigmoid(
            X_train, cols, missing_ratio=missing_ratio, missing_func=mechanism_func, seed=seed
        )
    else:
        raise NotImplementedError

    return X_train_ms

    # if isinstance(mechanism, list):
    #     if mechanism[0].startswith('mnar_quantile'):
    #         mechanism_truncated = [item.split('_')[-1] for item in mechanism]
    #         data_ms = mnar_simulate.simulate_nan_mnar_quantile(
    #             X_train, cols, missing_ratios=missing_ratio, missing_funcs=mechanism_truncated, seed=seed)
    #         X_train_ms = data_ms
    #     else:
    #         raise NotImplementedError
    # else:
    #     if mechanism == 'mcar':
    #         X_train_ms = mcar_simulate.simulate_nan_mcar(X_train, cols, missing_ratio, seed)
    #     elif mechanism == 'mar_quantile_left':
    #         if len(cols) == X_train.shape[1]:
    #             cols = np.arange(0, X_train.shape[1] - 1)
    #         X_train_ms = mar_simulate.simulate_nan_mar_quantile(
    #             X_train, cols, missing_ratio, missing_func='left', obs=True, seed=seed
    #         )
    #     elif mechanism == 'mar_quantile_right':
    #         if len(cols) == X_train.shape[1]:
    #             cols = np.arange(0, X_train.shape[1] - 1)
    #         X_train_ms = mar_simulate.simulate_nan_mar_quantile(
    #             X_train, cols, missing_ratio, missing_func='right', obs=True, seed=seed
    #         )
    #     elif mechanism == 'mar_quantile_mid':
    #         if len(cols) == X_train.shape[1]:
    #             cols = np.arange(0, X_train.shape[1] - 1)
    #         X_train_ms = mar_simulate.simulate_nan_mar_quantile(
    #             X_train, cols, missing_ratio, missing_func='mid', obs=True, seed=seed
    #         )
    #     elif mechanism == 'mar_quantile_tail':
    #         if len(cols) == X_train.shape[1]:
    #             cols = np.arange(0, X_train.shape[1] - 1)
    #         X_train_ms = mar_simulate.simulate_nan_mar_quantile(
    #             X_train, cols, missing_ratio, missing_func='tail', obs=True, seed=seed
    #         )
    #     elif mechanism == 'mar_sigmoid_left':
    #         if len(cols) == X_train.shape[1]:
    #             cols = np.arange(0, X_train.shape[1] - 1)
    #         X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
    #             X_train, cols, missing_ratio, missing_func='right', obs=True, k='all', seed=seed
    #         )
    #     elif mechanism == 'mar_sigmoid_right':
    #         if len(cols) == X_train.shape[1]:
    #             cols = np.arange(0, X_train.shape[1] - 1)
    #         X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
    #             X_train, cols, missing_ratio, missing_func='right', obs=True, k='all', seed=seed
    #         )
    #     elif mechanism == 'mar_sigmoid_mid':
    #         if len(cols) == X_train.shape[1]:
    #             cols = np.arange(0, X_train.shape[1] - 1)
    #         X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
    #             X_train, cols, missing_ratio, missing_func='mid', obs=True, k='all', seed=seed
    #         )
    #     elif mechanism == 'mar_sigmoid_tail':
    #         if len(cols) == X_train.shape[1]:
    #             cols = np.arange(0, X_train.shape[1] - 1)
    #         X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
    #             X_train, cols, missing_ratio, missing_func='tail', obs=True, k='all', seed=seed
    #         )
    #     elif mechanism == 'mary_left':
    #         data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
    #         data_ms = mar_simulate.simulate_nan_mary_quantile(data, cols, missing_ratio, missing_func='left', seed=seed)
    #         X_train_ms = data_ms[:, :-1]
    #     elif mechanism == 'mary_right':
    #         data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
    #         data_ms = mar_simulate.simulate_nan_mary_quantile(data, cols, missing_ratio, missing_func='right',
    #                                                           seed=seed)
    #         X_train_ms = data_ms[:, :-1]
    #     elif mechanism == 'mary_sigmoid_left':
    #         data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
    #         data_ms = mar_simulate.simulate_nan_mary_sigmoid(data, cols, missing_ratio, missing_func='left', seed=seed)
    #         X_train_ms = data_ms[:, :-1]
    #     elif mechanism == 'mary_sigmoid_right':
    #         data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
    #         data_ms = mar_simulate.simulate_nan_mary_sigmoid(data, cols, missing_ratio, missing_func='right', seed=seed)
    #         X_train_ms = data_ms[:, :-1]
    #     elif mechanism == 'mary_mid':
    #         data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
    #         data_ms = mar_simulate.simulate_nan_mary_quantile(data, cols, missing_ratio, missing_func='mid', seed=seed)
    #         X_train_ms = data_ms[:, :-1]
    #     elif mechanism == 'mary_tail':
    #         data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
    #         data_ms = mar_simulate.simulate_nan_mary_quantile(data, cols, missing_ratio, missing_func='tail', seed=seed)
    #         X_train_ms = data_ms[:, :-1]
    #     elif mechanism == 'mnar_sigmoid_left':
    #         data_ms = mnar_simulate.simulate_nan_mnar_sigmoid(X_train, cols, missing_ratio, missing_func='left',
    #                                                           seed=seed)
    #         X_train_ms = data_ms
    #     elif mechanism == 'mnar_sigmoid_right':
    #         data_ms = mnar_simulate.simulate_nan_mnar_sigmoid(
    #             X_train, cols, missing_ratio, missing_func='right', seed=seed)
    #         X_train_ms = data_ms
    #     elif mechanism == 'mnar_quantile_left':
    #         data_ms = mnar_simulate.simulate_nan_mnar_quantile(
    #             X_train, cols, missing_ratios=missing_ratio, missing_funcs='left', seed=seed)
    #         X_train_ms = data_ms
    #     elif mechanism == 'mnar_quantile_right':
    #         data_ms = mnar_simulate.simulate_nan_mnar_quantile(
    #             X_train, cols, missing_ratios=missing_ratio, missing_funcs='right', seed=seed)
    #         X_train_ms = data_ms
    #     elif mechanism == 'mnar_quantile_mid':
    #         data_ms = mnar_simulate.simulate_nan_mnar_quantile(
    #             X_train, cols, missing_ratios=missing_ratio, missing_funcs='mid', seed=seed)
    #         X_train_ms = data_ms
    #     elif mechanism == 'mnar_quantile_tail':
    #         data_ms = mnar_simulate.simulate_nan_mnar_quantile(
    #             X_train, cols, missing_ratios=missing_ratio, missing_funcs='tail', seed=seed)
    #         X_train_ms = data_ms
    #     else:
    #         raise NotImplementedError
    #
    # return X_train_ms
