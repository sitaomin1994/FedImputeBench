import pandas as pd
from sklearn.decomposition import PCA
from dython.nominal import correlation_ratio
from loguru import logger
from sklearn.datasets import fetch_openml
import numpy as np
from .utils import (
    normalization, move_target_to_end, convert_gaussian, drop_unique_cols, one_hot_categorical,
)


def process_diabetes(normalize=True, verbose=False, threshold=None):
    if threshold is None:
        threshold = 0.15
    from sklearn.datasets import load_diabetes
    data_obj = load_diabetes(as_frame=True)
    data = data_obj['frame']
    data = data.dropna()
    target_col = 'target'

    # # move target to the end of the dataframe
    data = move_target_to_end(data, target_col)
    correlation_ret = data.corrwith(data[target_col]).abs().sort_values(ascending=False)
    # # correlation
    important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
    important_features.remove(target_col)
    data_config = {
        'target': target_col,
        'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
        'important_features': important_features,
        'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
        'num_cols': data.shape[1] - 1,
        'task_type': 'regression',
    }

    if verbose:
        logger.debug("Important features {}".format(important_features))
        logger.debug("Data shape {}".format(data.shape, data.shape))
        logger.debug(data_config)
    return data, data_config


########################################################################################################################
# California Housing
########################################################################################################################
def process_california_housing(normalize=True, verbose=False, threshold=0.1):
    if threshold is None:
        threshold = 0.1
    from sklearn.datasets import fetch_california_housing
    data_obj = fetch_california_housing(data_home='./data/california_housing', as_frame=True)
    data = data_obj['frame']
    sample_size = 20000
    data = data.sample(sample_size, random_state=42)
    data = data.dropna()
    target_col = 'MedHouseVal'

    if normalize:
        data = normalization(data, target_col)

    # # move target to the end of the dataframe
    data = move_target_to_end(data, target_col)
    correlation_ret = data.corrwith(data[target_col]).abs().sort_values(ascending=False)
    # # correlation
    important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
    important_features.remove(target_col)
    data_config = {
        'target': target_col,
        'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
        'important_features': important_features,
        'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
        'num_cols': data.shape[1] - 1,
        'task_type': 'regression',
    }
    if verbose:
        logger.debug("Important features {}".format(important_features))
        logger.debug("Data shape {}".format(data.shape, data.shape))
        logger.debug(data_config)
    return data, data_config


########################################################################################################################
# Housing
########################################################################################################################
def process_housing(normalize=True, verbose=False, threshold=0.2):
    if threshold is None:
        threshold = 0.2

    data = pd.read_csv("./data/housing/housing.csv", delimiter='\s+', header=None)
    data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
    # sample_size = 1000
    # data = data.sample(sample_size, random_state=42)
    data = data.dropna()
    # data
    target_col = '14'

    if normalize:
        data = normalization(data, target_col)

    # # # move target to the end of the dataframe
    data = move_target_to_end(data, target_col)
    correlation_ret = data.corrwith(data[target_col]).abs().sort_values(ascending=False)
    # # # correlation
    important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
    important_features.remove(target_col)
    data_config = {
        'target': target_col,
        'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
        'important_features': important_features,
        'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
        'num_cols': data.shape[1] - 1,
        'task_type': 'regression',
    }
    if verbose:
        logger.debug("Important features {}".format(important_features))
        logger.debug("Data shape {}".format(data.shape, data.shape))
        logger.debug(data_config)
    return data, data_config


########################################################################################################################
# red wine regression
########################################################################################################################
def process_red_reg(normalize=True, verbose=False, threshold=0.15):
    if threshold is None:
        threshold = 0.15
    data = pd.read_csv("./data/wine/winequality-red.csv", delimiter=';')
    # data = data.drop(["id", "Unnamed: 32"], axis=1)
    data = data.dropna()
    target_col = 'quality'
    if normalize:
        data = normalization(data, target_col)
    data = move_target_to_end(data, target_col)

    # print("Wine Red data loaded. Train size {}, Test size {}".format(train.shape, test.shape))
    correlation_ret = data.corrwith(data[target_col]).abs().sort_values(ascending=False)
    important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
    important_features.remove(target_col)
    print(important_features)

    data_config = {
        'target': target_col,
        'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
        'important_features': important_features,
        'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
        "num_cols": data.shape[1] - 1,
        'task_type': 'regression',
    }

    if verbose:
        logger.debug("Important features {}".format(important_features))
        logger.debug("Data shape {}".format(data.shape, data.shape))
        logger.debug(data_config)
    return data, data_config


########################################################################################################################
# red wine regression
########################################################################################################################
def process_white_reg(normalize=True, verbose=False, threshold=0.15):
    if threshold is None:
        threshold = 0.15
    data = pd.read_csv("./data/wine/winequality-white.csv", delimiter=';')
    # data = data.drop(["id", "Unnamed: 32"], axis=1)
    data = data.dropna()
    target_col = 'quality'
    if normalize:
        data = normalization(data, target_col)
    data = move_target_to_end(data, target_col)

    # print("Wine Red data loaded. Train size {}, Test size {}".format(train.shape, test.shape))
    correlation_ret = data.corrwith(data[target_col]).abs().sort_values(ascending=False)
    important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
    important_features.remove(target_col)
    print(important_features)

    data_config = {
        'target': target_col,
        'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
        'important_features': important_features,
        'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
        "num_cols": data.shape[1] - 1,
        'task_type': 'regression',
    }

    if verbose:
        logger.debug("Important features {}".format(important_features))
        logger.debug("Data shape {}".format(data.shape, data.shape))
        logger.debug(data_config)
    return data, data_config
