import numpy as np
import pytest


def test_func_kcorr():
    seed = 102020
    np.random.seed(seed)
    data = np.random.randn(20, 10)
    cols = [0, 1, 3, 4, 8]

    keep_mask = np.ones(data.shape[1], dtype=bool)
    keep_mask[list(cols)] = False
    X_rest = data[:, keep_mask]
    cols_selection_option = 'allk0.5'
    col = 8
    print(np.corrcoef(data, rowvar=False)[8])
    X = np.concatenate([data[:, col].reshape(-1, 1), X_rest], axis=1)
    print(X.shape, X_rest.shape)
    k = max(int(float(cols_selection_option.split('allk')[-1]) * X_rest.shape[1]), 1)
    k = min(k, X_rest.shape[1])
    print(k)
    mi = np.abs(np.corrcoef(X, rowvar=False)[0])
    print(mi)
    mi_idx = np.argsort(mi)[::-1][1:k + 1]
    print(mi_idx)
    print(X[:, mi_idx], data[:, [2, 6]])
    assert np.array_equal(X[:, mi_idx], data[:, [2, 6]])