import pandas as pd
from sklearn.decomposition import PCA
from dython.nominal import correlation_ratio
from loguru import logger
from sklearn.datasets import fetch_openml
import numpy as np
from .utils import (
    normalization, move_target_to_end, convert_gaussian, drop_unique_cols, one_hot_categorical,
)


def process_adult(normalize=True, verbose=False, threshold=None, sample=False, pca=False, gaussian=False):
    if threshold is None:
        threshold = 0.1
    # retain_threshold = 0.05

    columns = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
               "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
               "Hours per week", "Country", "Target"]
    types = {
        0: int, 1: str, 2: int, 3: str, 4: int, 5: str, 6: str, 7: str, 8: str, 9: str, 10: int,
        11: int, 12: int, 13: str, 14: str
    }

    data_train = pd.read_csv(
        "./data/adult/adult_train.csv", names=columns, na_values=['?'], sep=r'\s*,\s*', engine='python',
        dtype=types
    )
    data_test = pd.read_csv(
        "./data/adult/adult_test.csv", names=columns, na_values=['?'], sep=r'\s*,\s*', engine='python',
        dtype=types
    )
    data = pd.concat([data_train, data_test], axis=0)
    data = data.dropna()
    col_drop = ["Country", "Education", "fnlwgt"]
    data = data.drop(col_drop, axis=1)
    # target
    target_col = 'Target'
    data[target_col] = data[target_col].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})

    # convert categorical to numerical
    if pca:
        cat_cols = ["Sex", "Martial Status", "Relationship", "Race", "Workclass", "Occupation"]
        data_oh = pd.get_dummies(data.drop(target_col, axis=1), columns=cat_cols, drop_first=True)
        data = pd.concat([data[target_col], data_oh], axis=1)
        data.reset_index(drop=True, inplace=True)
        #print(data.shape)

        pca = PCA(n_components=20)
        pca.fit(data.drop(target_col, axis=1))
        data = pd.concat([data[target_col], pd.DataFrame(pca.transform(data.drop(target_col, axis=1)))], axis=1)
    else:
        for col in ["Sex", "Martial Status", "Relationship", "Race", "Workclass", "Occupation"]:
            values = data[col].value_counts().index.tolist()
            corr_y = []
            for value in values:
                corr_y_data = data[data[col] == value][target_col].value_counts(normalize=True)
                corr = corr_y_data[0] / corr_y_data[1]
                corr_y.append(corr)

            sorted_values = sorted(values, key=lambda x: corr_y[values.index(x)])
            np.random.seed(31)
            np.random.shuffle(sorted_values)
            mapping = {value: idx for idx, value in enumerate(sorted_values)}
            data[col] = data[col].map(mapping)

    if gaussian:
        data = convert_gaussian(data, target_col)

    if normalize:
        data = normalization(data, target_col)

    data = move_target_to_end(data, target_col)
    data[target_col] = pd.factorize(data[target_col])[0]

    # sample balance
    if sample:
        data_y0 = data[data[target_col] == 0]
        data_y1 = data[data[target_col] == 1]
        data_y0 = data_y0.sample(n=data_y1.shape[0], random_state=0)
        data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)

    correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
    # retained_features = correlation_ret[correlation_ret >= retain_threshold].index.tolist()
    # new_cols = []
    # new_num_cols = 0
    # for idx, feature in enumerate(data.columns.tolist()):
    #     if feature in retained_features:
    #         new_cols.append(feature)
    #         if idx < num_cols:
    #             new_num_cols += 1

    # data = data[new_cols]

    important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
    important_features.remove(target_col)

    data_config = {
        'target': target_col,
        'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
        'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
        "num_cols": data.shape[1] - 1,
        'task_type': 'classification',
        'clf_type': 'binary-class',
        'data_type': 'tabular'
    }

    if verbose:
        logger.debug("Important features {}".format(important_features))
        logger.debug("Data shape {}".format(data.shape, data.shape))
        logger.debug(data_config)
    return data, data_config


def process_default_credit(normalize=True, verbose=False, threshold=None):
    if threshold is None:
        threshold = 0.1

    retain_threshold = 0.0

    data = pd.read_csv("./data/default_credit/default_creidt.csv")
    data = data.dropna()
    data = data.drop('ID', axis=1)

    # target
    target_col = 'default payment next month'
    if normalize:
        data = normalization(data, target_col)

    data = move_target_to_end(data, target_col)
    data[target_col] = pd.factorize(data[target_col])[0]

    # sample balance
    data_y0 = data[data[target_col] == 0]
    data_y1 = data[data[target_col] == 1]
    data_y1 = data_y1.sample(n=data_y0.shape[0], random_state=0)
    data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)

    correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
    important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
    important_features.remove(target_col)

    retained_features = correlation_ret[correlation_ret >= retain_threshold].index.tolist()
    new_cols = []
    for idx, feature in enumerate(data.columns.tolist()):
        if feature in retained_features:
            new_cols.append(feature)

    data = data[new_cols]

    data_config = {
        'target': target_col,
        'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
        'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
        "num_cols": data.shape[1] - 1,
        'task_type': 'classification',
        'clf_type': 'binary-class',
        'data_type': 'tabular'
    }

    if verbose:
        logger.debug("Important features {}".format(important_features))
        logger.debug("Data shape {}".format(data.shape, data.shape))
        logger.debug(data_config)
    return data, data_config


def process_bank_market(normalize=True, verbose=False, threshold=None, sample=False, pca=False, gaussian=False):
    if threshold is None:
        threshold = 0.1
    data = pd.read_csv("./data/bank_market/bank-full.csv", sep=';')
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    target_col = 'y'
    if pca:
        data = pd.get_dummies(data, columns=cat_cols)
        print(data.shape)
        pca = PCA(n_components=10)
        pca.fit(data.drop(target_col, axis=1))
        data = pd.concat([pd.DataFrame(pca.transform(data.drop(target_col, axis=1))), data[target_col]], axis=1)
    else:
        for col in cat_cols:
            cats = data[col].value_counts().index.tolist()
            np.random.seed(0)
            np.random.shuffle(cats)
            mapping = {cat: idx for idx, cat in enumerate(cats)}
            data[col] = data[col].map(mapping)

    # target_col = 'Y'
    data[target_col] = pd.factorize(data[target_col])[0]
    data = data.dropna()
    if gaussian:
        data = convert_gaussian(data, target_col)
    if normalize:
        data = normalization(data, target_col)

    data = move_target_to_end(data, target_col)

    # sample balance
    if sample:
        data_y0 = data[data[target_col] == 0]
        data_y1 = data[data[target_col] == 1]
        data_y0 = data_y0.sample(n=data_y1.shape[0], random_state=0)
        data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)

    correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
    important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
    important_features.remove(target_col)

    data_config = {
        'target': target_col,
        'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
        'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
        "num_cols": data.shape[1] - 1,
        'task_type': 'classification',
        'clf_type': 'binary-class',
        'data_type': 'tabular'
    }

    if verbose:
        logger.debug("Important features {}".format(important_features))
        logger.debug("Data shape {}".format(data.shape, data.shape))
        logger.debug(data_config)

    return data, data_config
