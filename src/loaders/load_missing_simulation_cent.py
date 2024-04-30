from typing import Tuple, List
import numpy as np
from scipy.stats import stats

from src.modules.missing_simulate.add_missing import simulate_nan
from src.modules.missing_simulate.missing_scenario_utils import (
    generate_missing_ratios, generate_missing_mech, generate_missing_mech_funcs, generate_missing_cols
)
from emf.params_utils import parse_strategy_params


def add_missing_central(
        data: np.ndarray, cols: List[int], missing_mech: str, missing_mech_funcs: str, mr_strategy: str,
        missing_cols_strategy: str,
        seed: int
) -> np.ndarray:

    # missing features - e.g. 'all'
    missing_cols_strategy, missing_cols_params = parse_strategy_params(missing_cols_strategy)
    missing_cols = generate_missing_cols(missing_cols_strategy, 1, cols, seed=seed)
    missing_cols = missing_cols[0]
    num_cols = len(missing_cols)

    # missing ratios - 'uniform@mrl=0.3-mrr=0.7'
    mr_dist, mr_params = parse_strategy_params(mr_strategy)
    try:
        mr_range = (float(mr_params['mrl']), float(mr_params['mrr']))
    except KeyError as e:
        raise KeyError(f"Missing ratio strategy should have 'mrl' and 'mrr' keys: {e}")
    except ValueError as e:
        raise ValueError(f"Missing ratio should be float: {e}")

    missing_ratios = generate_missing_ratios(
        mr_dist, ms_range=mr_range, num_clients=1, num_cols=num_cols, seed=seed
    )
    missing_ratios = missing_ratios[0]

    # missing mechanism - e.g. 'mcar'
    mm_mech, mm_mech_params = parse_strategy_params(missing_mech)
    missing_mechs = generate_missing_mech(mm_mech, num_clients=1, num_cols=num_cols, seed=seed)
    missing_mechs = missing_mechs[0][0]  # TODO: currently only support all cols with same misisng mechs

    # missing mechanism functions - e.g. 'lr'
    mm_func, mm_func_params = parse_strategy_params(missing_mech_funcs)
    missing_mech_funcs = generate_missing_mech_funcs(
        'homo', mm_func, num_clients=1, num_cols=num_cols, seed=seed
    )
    missing_mech_funcs = missing_mech_funcs[0]

    print(missing_cols, missing_mechs, missing_ratios, missing_mech_funcs)
    X_train, y_train = data[:, :-1], data[:, -1]
    X_train_ms = simulate_nan(
        X_train, y_train, cols=missing_cols, mech_type=missing_mechs, missing_ratio=missing_ratios,
        mechanism_func=missing_mech_funcs, seed=seed
    )

    return np.concatenate([X_train_ms, y_train.reshape(-1, 1)], axis=1)
