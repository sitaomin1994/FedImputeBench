import numpy as np
import src.modules.missing_simulate.mcar_simulate as mcar_simulate
import src.modules.missing_simulate.mar_simulate as mar_simulate
import src.modules.missing_simulate.mnar_simulate as mnar_simulate
from typing import List
from src.modules.missing_simulate.add_missing_utils import (
    generate_missing_cols, generate_missing_ratios, generate_missing_mech_funcs
)


########################################################################################################################
def add_missing_central(
        data: np.ndarray, cols: List[int], mf_strategy: str, mr_dist: str, mr_lower: float, mr_upper: float,
        mm_funcs_bank: str, mm_mech: str, mm_strictness: bool, mm_obs: bool, mm_feature_option: str,
        mm_beta_option: str, seed: int
) -> np.ndarray:
    """
    Add missing data for global dataset
    :param data: data array
    :param cols: columns to add missing values
    :param mm_funcs_bank: missing mechanism functions banks
    :param mf_strategy: missing features strategy
    :param mr_dist: missing ratio distribution
    :param mr_lower: missing ratio lower bound
    :param mr_upper: missing ratio upper bound
    :param mm_mech: missing mechanism
    :param mm_strictness: missing adding probailistic or deterministic
    :param mm_obs:  missing adding based on observed data
    :param mm_feature_option: missing mechanism associated with which features
    :param mm_beta_option: missing mechanism beta coefficient option
    :param seed: randonness
    :return: dataset with missing values
    """

    # missing features - e.g. 'all'
    # missing_cols_strategy, missing_cols_params = parse_strategy_params(missing_cols_strategy)
    missing_cols = generate_missing_cols(mf_strategy, 1, cols, seed=seed)
    missing_cols = missing_cols[0]
    num_cols = len(missing_cols)

    # missing ratios - 'uniform@mrl=0.3-mrr=0.7'
    # mr_dist, mr_params = parse_strategy_params(mr_strategy)
    mr_range = (mr_lower, mr_upper)
    missing_ratios = generate_missing_ratios(
        mr_dist, ms_range=mr_range, num_clients=1, num_cols=num_cols, seed=seed
    )
    missing_ratios = missing_ratios[0]

    # missing mechanism funcs - e.g. 'mcar'
    # TODO: currently only support all cols with same misisng mechs
    # mm_mech, mm_mech_params = parse_strategy_params(missing_mech)
    # missing_mechs = generate_missing_mech(mm_mech, num_clients=1, num_cols=num_cols, seed=seed)
    # missing mechanism functions - e.g. 'lr'
    mm_funcs = generate_missing_mech_funcs(
        'homo', mm_funcs_bank, num_clients=1, num_cols=num_cols, seed=seed
    )
    mm_funcs = mm_funcs[0]

    # Simulate missing data
    print(missing_cols, missing_ratios, mm_funcs)
    X_train, y_train = data[:, :-1], data[:, -1]
    X_train_ms = simulate_nan(
        X_train, y_train, mm_mech=mm_mech, missing_features=missing_cols, missing_ratios=missing_ratios,
        mechanism_funcs=mm_funcs, mm_obs=mm_obs, mm_strictness=mm_strictness,
        mm_feature_option=mm_feature_option, mm_beta_option=mm_beta_option, seed=seed
    )

    return np.concatenate([X_train_ms, y_train.reshape(-1, 1)], axis=1)


########################################################################################################################
def add_missing_dist(
        data: np.ndarray, cols: List[int], num_clients: int, mf_strategy: str,
        mr_dist: str, mr_lower: float, mr_upper: float, mm_funcs_dist: str, mm_funcs_bank: str,
        mm_mech: str, mm_strictness: bool, mm_obs: bool, mm_feature_option: str, mm_beta_option: str, seed: int
) -> List[np.ndarray]:
    """
        Add missing data for each client's dataset
        :param data: data array
        :param cols: columns to add missing values
        :param num_clients: number of clients
        :param mm_funcs_dist: missing mechanism functions distribution across clients and features
        :param mm_funcs_bank: missing mechanism functions banks
        :param mf_strategy: missing features strategy
        :param mr_dist: missing ratio distribution
        :param mr_lower: missing ratio lower bound
        :param mr_upper: missing ratio upper bound
        :param mm_mech: missing mechanism
        :param mm_strictness: missing adding probailistic or deterministic
        :param mm_obs:  missing adding based on observed data
        :param mm_feature_option: missing mechanism associated with which features
        :param mm_beta_option: missing mechanism beta coefficient option
        :param seed: randonness
        :return: list of datasets with missing values
        """

    # missing features - e.g. 'all'
    clients_missing_cols = generate_missing_cols(mf_strategy, num_clients, cols, seed=seed)
    num_cols = len(clients_missing_cols[0])

    # missing ratios - 'uniform@mrl=0.3-mrr=0.7'
    # mr_dist, mr_params = parse_strategy_params(mr_strategy)
    mr_range = (mr_lower, mr_upper)
    clients_missing_ratios = generate_missing_ratios(
        mr_dist, ms_range=mr_range, num_clients=num_clients, num_cols=num_cols, seed=seed
    )

    # missing mechanism funcs - 'homo', 'random'
    clients_mm_funcs = generate_missing_mech_funcs(
        mm_funcs_dist, mm_funcs_bank, num_clients=num_clients, num_cols=num_cols, seed=seed
    )

    # Simulate missing data
    assert len(clients_missing_cols) == len(clients_missing_ratios) == len(
        clients_mm_funcs), "error when generate missing data"

    clients_data_ms = []
    for i in range(num_clients):
        missing_cols = clients_missing_cols[i]
        missing_ratios = clients_missing_ratios[i]
        mm_funcs = clients_mm_funcs[i]

        print(i, missing_cols, missing_ratios, mm_funcs, "\n")

        X_train, y_train = data[:, :-1], data[:, -1]
        X_train_ms = simulate_nan(
            X_train, y_train, mm_mech=mm_mech, missing_features=missing_cols, missing_ratios=missing_ratios,
            mechanism_funcs=mm_funcs, mm_obs=mm_obs, mm_strictness=mm_strictness,
            mm_feature_option=mm_feature_option, mm_beta_option=mm_beta_option, seed=seed
        )

        clients_data_ms.append(np.concatenate([X_train_ms, y_train.reshape(-1, 1)], axis=1))

    return clients_data_ms


########################################################################################################################
# Simulate missing for one client
########################################################################################################################
def simulate_nan(
        X_train: np.ndarray, y_train: np.ndarray, mm_mech: str,
        missing_features: List[int], missing_ratios: List[float], mechanism_funcs: List[str],
        mm_strictness: bool, mm_obs: bool, mm_feature_option: str, mm_beta_option: str, seed: int = 201030
) -> np.ndarray:
    """
    Simulate missing values for one client
    :param X_train: X_train data
    :param y_train: y_train data
    :param mm_mech: missing mechanism
    :param missing_features: missing features
    :param missing_ratios: missing ratios for each feature
    :param mechanism_funcs: missing mechanism functions for each feature
    :param mm_strictness: missing strictness
    :param mm_obs: missing based on observed data
    :param mm_feature_option: missing mechanism associated with which features
    :param mm_beta_option: missing mechanism beta coefficient option
    :param seed: randomness
    :return: data with missing values
    """

    # TODO: add mary
    if mm_mech == 'mcar':
        X_train_ms = mcar_simulate.simulate_nan_mcar(
            X_train, missing_features, missing_ratios, seed=seed
        )
    elif mm_mech == 'mar_quantile':
        X_train_ms = mar_simulate.simulate_nan_mar_quantile(
            X_train, missing_features, missing_ratio=missing_ratios, missing_func=mechanism_funcs,
            obs=mm_obs, strict=mm_strictness, seed=seed
        )
    elif mm_mech == 'mar_sigmoid':
        X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
            X_train, missing_features, missing_ratio=missing_ratios, missing_func=mechanism_funcs,
            obs=mm_obs, strict=mm_strictness, mm_feature_option=mm_feature_option, mm_beta_option=mm_beta_option,
            seed=seed
        )
    elif mm_mech == 'mnar_quantile':
        X_train_ms = mnar_simulate.simulate_nan_mnar_quantile(
            X_train, missing_features, missing_ratio=missing_ratios, missing_func=mechanism_funcs, strict=mm_strictness,
            seed=seed
        )
    elif mm_mech == 'mnar_sigmoid':
        X_train_ms = mnar_simulate.simulate_nan_mnar_sigmoid(
            X_train, missing_features, missing_ratio=missing_ratios, missing_func=mechanism_funcs,
            strict=mm_strictness, mm_feature_option=mm_feature_option, mm_beta_option=mm_beta_option, seed=seed
        )
    else:
        raise NotImplementedError

    return X_train_ms
