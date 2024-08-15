import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, PowerTransformer, LabelEncoder, StandardScaler, RobustScaler
)
from dython.nominal import correlation_ratio
from loguru import logger
from sklearn.datasets import fetch_openml
import numpy as np
from .utils import (
    convert_gaussian, normalization, drop_unique_cols, one_hot_categorical, move_target_to_end,
)
from sklearn.feature_selection import mutual_info_classif
from imblearn.under_sampling import RandomUnderSampler
import json
from collections import OrderedDict


def avg_correlation(df):
    avg_correlation_cols = list(OrderedDict(df.corr().abs().mean().sort_values(ascending=False).to_dict()).items())
    features = set(df.columns.tolist()[:-1])
    avg_correlation_cols = [col for col in avg_correlation_cols if col[0] in features]
    return avg_correlation_cols


def process_codrna(normalize=True, verbose=False, threshold=None, sample=True, gaussian=True):
    if threshold is None:
        threshold = 0.1

    # data_obj = fetch_openml(data_id=351, as_frame='auto', parser='auto')
    # X = pd.DataFrame(data_obj.data.todense(), columns=data_obj.feature_names)
    # y = pd.DataFrame(data_obj.target, columns=data_obj.target_names)
    # data = pd.concat([X, y], axis=1)
    # data.to_csv('./data/codrna/codrna.csv', index=False)
    data = pd.read_csv('./data/codrna/codrna.csv')

    target_col = 'Y'
    data[target_col] = pd.factorize(data[target_col])[0]
    data = data.dropna()

    if gaussian:
        data = convert_gaussian(data, target_col)

    if normalize:
        data = normalization(data, target_col)

    data = move_target_to_end(data, target_col)
    correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
    important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
    important_features.remove(target_col)

    if sample:
        data_y0 = data[data[target_col] == 0]
        data_y1 = data[data[target_col] == 1]
        if data_y0.shape[0] > data_y1.shape[0]:
            data_y0 = data_y0.sample(n=data_y1.shape[0], random_state=0)
        elif data_y0.shape[0] < data_y1.shape[0]:
            data_y1 = data_y1.sample(n=data_y0.shape[0], random_state=0)
        data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)

    if len(data) >= 20000:
        data = data.sample(n=20000, random_state=0).reset_index(drop=True)

    avg_cols = avg_correlation(data)
    avg_cols = [col[0] for col in avg_cols]
    split_col_idx = [data.columns.tolist().index(col) for col in avg_cols]

    data_config = {
        'target': target_col,
        'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
        'split_col_idx': split_col_idx,
        'ms_col_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
        'obs_col_idx': [1, 4, 7],
        "num_cols": data.shape[1] - 1,
        'task_type': 'classification',
        'clf_type': 'binary-class',
        'data_type': 'tabular'
    }

    if verbose:
        logger.debug("Important features {}".format(important_features))
        logger.debug("Data shape {}".format(data.shape))
        logger.debug(data_config)

    return data, data_config


def process_hhip(verbose=False):
    data = pd.read_csv('./data/HHP_herritage_health/data_cleaned.csv')
    with open('./data/HHP_herritage_health/data_config.json') as f:
        data_config = json.load(f)
    data = data.astype(float)

    if verbose:
        logger.debug("Data shape {}".format(data.shape))
        logger.debug(data_config)

    return data, data_config

def process_hhip_sp(verbose=False):
    data = pd.read_csv('./data/HHP_herritage_health/data_cleaned_sp.csv')
    with open('./data/HHP_herritage_health/data_config_sp.json') as f:
        data_config = json.load(f)
    data = data.astype(float)

    if verbose:
        logger.debug("Data shape {}".format(data.shape))
        logger.debug(data_config)

    return data, data_config

def process_hhip_np(verbose=False):
    data = pd.read_csv('./data/HHP_herritage_health/data_cleaned_np.csv')
    with open('./data/HHP_herritage_health/data_config_np.json') as f:
        data_config = json.load(f)
    data = data.astype(float)

    if verbose:
        logger.debug("Data shape {}".format(data.shape))
        logger.debug(data_config)

    return data, data_config


def process_california(verbose=False):
    data = pd.read_csv('./data/california/data_cleaned.csv')
    with open('./data/california/data_config.json') as f:
        data_config = json.load(f)
    data = data.astype(float)

    if verbose:
        logger.debug("Data shape {}".format(data.shape))
        logger.debug(data_config)

    return data, data_config


def process_dvisits(verbose=False):
    data = pd.read_csv('./data/dvisits/data_cleaned.csv')
    with open('./data/dvisits/data_config.json') as f:
        data_config = json.load(f)
    data = data.astype(float)

    if verbose:
        logger.debug("Data shape {}".format(data.shape))
        logger.debug(data_config)

    return data, data_config


def process_vehicle(verbose=False):
    data = pd.read_csv('./data/vehicle/data_cleaned.csv')
    with open('./data/vehicle/data_config.json') as f:
        data_config = json.load(f)
    data = data.astype(float)

    if verbose:
        logger.debug("Data shape {}".format(data.shape))
        logger.debug(data_config)

    return data, data_config

def process_vehicle_np(verbose=False):
    data = pd.read_csv('./data/vehicle/data_cleaned_natural.csv')
    with open('./data/vehicle/data_config_natural.json') as f:
        data_config = json.load(f)
    data = data.astype(float)

    if verbose:
        logger.debug("Data shape {}".format(data.shape))
        logger.debug(data_config)

    return data, data_config


def process_school(verbose=False, pca = True):
    if pca:
        data = pd.read_csv('./data/school/data_cleaned_pca.csv')
        with open('./data/school/data_config_pca.json') as f:
            data_config = json.load(f)
    else:
        data = pd.read_csv('./data/school/data_cleaned.csv')
        with open('./data/school/data_config.json') as f:
            data_config = json.load(f)
        data = data.astype(float)

    if verbose:
        logger.debug("Data shape {}".format(data.shape))
        logger.debug(data_config)

    return data, data_config


def process_codon(verbose=False):
    data = pd.read_csv('./data/codon/data_cleaned.csv')
    with open('./data/school/data_config.json') as f:
        data_config = json.load(f)
    data = data.astype(float)

    if verbose:
        logger.debug("Data shape {}".format(data.shape))
        logger.debug(data_config)

    return data, data_config