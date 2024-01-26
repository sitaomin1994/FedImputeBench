import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer
from dython.nominal import correlation_ratio
from loguru import logger
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split


def split_train_test(data, data_config, test_size=0.2, stratify=True, seed=42, output_format='dataframe_merge'):
    """
    Split data into train and test set
    """

    target = data_config['target']
    task_type = data_config['task_type']
    X = data.drop(columns=target)
    y = data[target]
    if task_type == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=seed, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    if output_format == 'dataframe_merge':
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        return train_data, test_data
    elif output_format == 'dataframe':
        return X_train, y_train, X_test, y_test
    elif output_format == 'nparray':
        return X_train.values, y_train.values, X_test.values, y_test.values


#######################################################################################################################
# Utility
#######################################################################################################################
def normalization(data, target_col, categorical_cols=None):
    if categorical_cols is None:
        categorical_cols = []
    data_X = data.drop([target_col] + categorical_cols, axis=1).values
    scaler = MinMaxScaler()
    new_data_X = scaler.fit_transform(data_X)
    data.loc[:, ~data.columns.isin([target_col] + categorical_cols)] = new_data_X

    return data


def move_target_to_end(data, target_col):
    target_col_series = data[target_col]
    data = data.drop(target_col, axis=1)
    data = pd.concat([data, target_col_series], axis=1)
    # data.insert(len(data.columns), target_col, target_col_series)
    return data


def drop_unique_cols(data, target_col):
    col_drop = []
    for col in data.columns:
        if col != target_col:
            if data[col].value_counts(normalize=True)[0] > 0.98:
                col_drop.append(col)

    data = data.drop(col_drop, axis=1)
    return data


def one_hot_categorical(data, categorical_cols):
    one_hot_encoder = OneHotEncoder(
        categories='auto', handle_unknown='ignore', drop='first', max_categories=5
    )
    one_hot_encoder.fit(data[categorical_cols])
    one_hot_encoded = one_hot_encoder.transform(data[categorical_cols]).toarray()
    one_hot_encoded = pd.DataFrame(one_hot_encoded)
    data = data.drop(categorical_cols, axis=1).reset_index(drop=True)
    num_cols = data.shape[1] - 1
    data = pd.concat([data, one_hot_encoded], axis=1)
    return data, num_cols


def convert_gaussian(data, target_col):
    for col in data.columns:
        if col != target_col:
            pt = PowerTransformer()
            data[col] = pt.fit_transform(data[col].values.reshape(-1, 1)).flatten()
    return data
