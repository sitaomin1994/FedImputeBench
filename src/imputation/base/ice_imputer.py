from typing import Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from src.imputation.pred_models.load_estimator import get_estimator
from abc import ABC, abstractmethod
from .base_imputer import BaseImputer


class ICEImputer(BaseImputer):

    def __init__(
            self, estimator_num, estimator_cat, initial_strategy_cat: str = "mode",
            initial_strategy_num: str = 'mean', clip: bool = True, use_y: bool = False, seed: int = 0):

        super().__init__()
        self.estimator_num = estimator_num
        self.estimator_cat = estimator_cat
        self.seed = seed
        self.initial_strategy_cat = initial_strategy_cat
        self.initial_strategy_num = initial_strategy_num
        self.clip = clip
        self.min_values = None
        self.max_values = None
        self.use_y = use_y

        # estimators
        self.estimators = []

        # missing information
        self.missing_information = []

        # utils_info
        self.data_utils_info = {}

    @abstractmethod
    def initialize(self, X, missing_mask, data_config):
        pass

        # # get basic information from data
        # self.data_utils_info['num_cols'] = data_config['num_cols'] if 'num_cols' in data_config else X.shape[1]
        # self.data_utils_info['num_class_dict'] = data_config[
        #     'num_class_dict'] if 'num_class_dict' in data_config else None
        # self.data_utils_info['task_type'] = data_config['task_type']
        #
        # # initialized imputation model
        # for i in range(X.shape[1]):
        #     if i < data_config['num_cols']:
        #         estimator = self.estimator_num
        #     else:
        #         estimator = self.estimator_cat
        #     self.estimators.append(get_estimator(estimator))
        #
        # # calculate missing data information
        # for i in range(X.shape[1]):
        #     self.missing_information.append(self.get_missing_info(missing_mask, i))

    @abstractmethod
    def fit(self, X, y, missing_mask, feature_idx) -> Tuple[dict, dict]:
        pass

    @abstractmethod
    def impute(self, X, y, missing_mask, feature_idx):
        pass

    @abstractmethod
    def get_imp_model_params(self, feature_idx):
        pass
        #return self.estimators[feature_idx].get_params()  # TODO: make it to be compatible with all estimators

    @abstractmethod
    def update_imp_model(self, updated_model, feature_idx):
        pass
        #self.estimators[feature_idx].set_params(**updated_model)

    def initial_imputation(self, X, missing_mask):
        X_copy = X.copy()
        X_copy[missing_mask] = np.nan
        num_cols = self.data_utils_info['num_cols']

        # initial imputation for numerical columns
        X_num = X_copy[:, :num_cols]
        if self.initial_strategy_num == 'mean':
            simple_imp = SimpleImputer(strategy='mean')
            X_num_t = simple_imp.fit_transform(X_num)
        elif self.initial_strategy_num == 'median':
            simple_imp = SimpleImputer(strategy='median')
            X_num_t = simple_imp.fit_transform(X_num)
        elif self.initial_strategy_num == 'zero':
            simple_imp = SimpleImputer(strategy='constant', fill_value=0)
            X_num_t = simple_imp.fit_transform(X_num)
        else:
            raise ValueError("initial_strategy_num must be one of 'mean', 'median', 'zero'")

        if num_cols == X.shape[1]:
            return X_num_t

        # initial imputation for categorical columns
        X_cat = X_copy[:, num_cols:]
        if self.initial_strategy_cat == 'mode':
            simple_imp = SimpleImputer(strategy='most_frequent')
            X_cat_t = simple_imp.fit_transform(X_cat)
        elif self.initial_strategy_cat == 'other':
            simple_imp = SimpleImputer(strategy='constant', fill_value=-1)
            X_cat_t = simple_imp.fit_transform(X_cat)
        else:
            raise ValueError("initial_strategy_cat must be one of 'mode', 'other'")

        Xt = np.concatenate((X_num_t, X_cat_t), axis=1)
        return Xt

    @staticmethod
    def get_clip_thresholds(X, clip=False):
        if clip:
            min_values = X.min(axis=0)
            max_values = X.max(axis=0)
        else:
            min_values = np.full((X.shape[1],), -np.inf)
            max_values = np.full((X.shape[1],), np.inf)

        return min_values, max_values

    def set_clip_thresholds(self, min_values, max_values):
        self.min_values = min_values
        self.max_values = max_values

    @staticmethod
    def get_visit_indices(visit_sequence, missing_mask):
        frac_of_missing_values = missing_mask.mean(axis=0)
        missing_values_idx = np.flatnonzero(frac_of_missing_values)

        if visit_sequence == 'roman':
            ordered_idx = missing_values_idx
        elif visit_sequence == 'arabic':
            ordered_idx = missing_values_idx[::-1]
        elif visit_sequence == 'random':
            ordered_idx = missing_values_idx.copy()
            np.random.shuffle(ordered_idx)
        elif visit_sequence == 'ascending':
            n = len(frac_of_missing_values) - len(missing_values_idx)
            ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:]
        elif visit_sequence == 'descending':
            n = len(frac_of_missing_values) - len(missing_values_idx)
            ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:][::-1]
        else:
            raise ValueError("Invalid choice for visit order: %s" % visit_sequence)

        return ordered_idx
