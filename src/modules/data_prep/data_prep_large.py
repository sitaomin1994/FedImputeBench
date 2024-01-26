import pandas as pd
from dython.nominal import correlation_ratio
from loguru import logger
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

from .utils import (
    normalization, move_target_to_end, convert_gaussian, )


def process_ijcnn(normalize=True, verbose=False, threshold=None, sample=False, pca=False, gaussian=False):
    if threshold is None:
        threshold = 0.1

    data_obj = fetch_openml(data_id=1575, as_frame='auto', parser='auto')
    X = pd.DataFrame(data_obj.data.todense(), columns=data_obj.feature_names)
    y = pd.DataFrame(data_obj.target, columns=data_obj.target_names)
    data = pd.concat([X, y], axis=1)

    target_col = 'class'
    data[target_col] = pd.factorize(data[target_col])[0]
    data = data.dropna()
    data = move_target_to_end(data, target_col)

    if pca:
        category_cols = [data.columns[idx] for idx in list(range(0, 8))]

        pca = PCA(n_components=20)
        pca.fit(data.iloc[:, :-1])
        data_pca = pca.transform(data.iloc[:, :-1])
        data_pca = pd.DataFrame(data_pca)
        data_pca = pd.concat([data_pca, data[target_col]], axis=1)
        data = data_pca

    if gaussian:
        data = convert_gaussian(data, target_col)
    if normalize:
        data = normalization(data, target_col)

    correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
    important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
    important_features.remove(target_col)

    # # # sample balance
    if sample:
        data_y0 = data[data[target_col] == 0]
        data_y1 = data[data[target_col] == 1]
        data_y0 = data_y0.sample(n=data_y1.shape[0], random_state=0)
        data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)

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


def process_susy(normalize=True, verbose=False, threshold=0.15, gaussian=False):
    if threshold is None:
        threshold = 0.15

    data = pd.read_csv("./data/susy/SUSY.csv", sep=',', header=None)

    data = data.dropna()
    data.columns = [str(i) for i in range(data.shape[1])]
    target_col = '0'
    data[target_col] = pd.factorize(data[target_col])[0]
    data[target_col].value_counts()

    data = data.sample(frac=0.01, random_state=42)
    if gaussian:
        data = convert_gaussian(data, target_col)
    if normalize:
        data = normalization(data, target_col)

    # #data = convert_gaussian(data, target_col)
    data = move_target_to_end(data, target_col)

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


def process_higgs(verbose=False, threshold=0.15):
    if threshold is None:
        threshold = 0.15

    data = pd.read_csv("./data/higgs/higgs_new.csv", sep=',')

    data = data.dropna()
    target_col = '0'
    # #data = convert_gaussian(data, target_col)

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


def process_statlog(normalize=True, verbose=False, threshold=None, pca=False, gaussian=False):
    if threshold is None:
        threshold = 0.1
    data_train = pd.read_csv("./data/statlog/shuttle.trn.trn", sep='\s+', header=None)
    data_test = pd.read_csv("./data/statlog/shuttle.tst.tst", sep='\s+', header=None)
    data = pd.concat([data_train, data_test], axis=0).reset_index(drop=True)

    data = data.dropna()
    data.columns = [str(i) for i in range(data.shape[1])]
    target_col = '9'
    data[target_col] = pd.factorize(data[target_col])[0]

    # sample balance
    data = data[data[target_col].isin([2, 1, 3])]
    data_rest = data[data[target_col].isin([1, 3])]
    data_more = data[data[target_col].isin([2])]
    data_more = data_more.sample(n=data_rest.shape[0], random_state=42)
    data = pd.concat([data_rest, data_more], axis=0).reset_index(drop=True)

    if pca:
        pca = PCA(n_components=10)
        pca.fit(data.drop(target_col, axis=1))
        data = pd.concat([pd.DataFrame(pca.transform(data.drop(target_col, axis=1))), data[target_col]], axis=1)

    if gaussian:
        data = convert_gaussian(data, target_col)

    data[target_col] = pd.factorize(data[target_col])[0]

    if normalize:
        data = normalization(data, target_col)

    # #data = convert_gaussian(data, target_col)

    correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
    important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
    important_features.remove(target_col)

    data_config = {
        'target': target_col,
        'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
        'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
        "num_cols": data.shape[1] - 1,
        'task_type': 'classification',
        'clf_type': 'multi-class',
        'data_type': 'tabular'
    }

    if verbose:
        logger.debug("Important features {}".format(important_features))
        logger.debug("Data shape {}".format(data.shape, data.shape))
        logger.debug(data_config)

    return data, data_config


def process_svm(normalize=True, verbose=False, threshold=None, gaussian=False):
    if threshold is None:
        threshold = 0.1
    data_train = pd.read_csv("./data/svm1/svm_p.csv", sep=',', header=None)
    data_test = pd.read_csv("./data/svm1/svm_pt.csv", sep=',', header=None)
    data = pd.concat([data_train, data_test], axis=0).reset_index(drop=True)
    data = data.dropna()
    data.columns = [str(i) for i in range(data.shape[1])]
    target_col = '0'
    data[target_col] = pd.factorize(data[target_col])[0]
    if gaussian:
        data = convert_gaussian(data, target_col)
    if normalize:
        data = normalization(data, target_col)

    data = move_target_to_end(data, target_col)

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
