from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from scipy import stats
from src.imputation.base import BaseImputer
from src.loaders.load_imputer import load_imputer


class Client:

    def __init__(
            self,
            client_id: int,
            train_data: np.ndarray, test_data: np.ndarray, X_train_ms: np.ndarray, data_config: dict,
            imp_model_name, imp_model_params, client_config: dict, seed=0,
    ) -> None:

        # client id
        self.client_id = client_id

        # data
        self.X_train, self.y_train = train_data[:, :-1], train_data[:, -1]  # training data
        self.X_test, self.y_test = test_data[:, :-1], test_data[:, -1]  # testing data
        self.X_train_ms = X_train_ms  # missing data
        self.X_train_mask = np.isnan(self.X_train_ms)  # missing data mask
        self.X_train_imp = self.X_train_ms.copy()  # imputed data

        # calculate data stats
        self.data_utils = self.calculate_data_utils(data_config)

        # imputation model
        self.imp_model = load_imputer(imp_model_name, imp_model_params)
        self.imp_model.initialize(self.data_utils, seed)

        # others
        self.seed = seed
        self.client_config = client_config

    def fit_local_imp_model(self, params: dict) -> Tuple[dict, dict]:
        """
        Fit local imputation model
        """
        fit_res = self.imp_model.fit(
            self.X_train_imp, self.y_train, self.X_train_mask, params
        )
        model_parameters = self.imp_model.get_imp_model_params(params)
        fit_res.update(self.data_utils)
        # fit_res['sample_size'] = self.data_utils['missing_stats_cols'][feature_idx]['sample_size_obs'] todo: add sample size
        # TODO: design a consistent structure for fit_res, fit_params, imp_res, imp_params and document it

        return model_parameters, fit_res

    def update_local_imp_model(self, updated_local_model: dict, params: dict):
        """
        Fit local imputation model
        """
        self.imp_model.update_imp_model(updated_local_model, params)

    def local_imputation(self, params: dict):
        """
        Imputation
        """
        feature_idx = params['feature_idx']
        self.imp_model.impute(self.X_train_imp, self.y_train, self.X_train_mask, params)

    def initial_impute(self, imp_values: np.ndarray, col_type: str = 'num'):
        """
        Initial imputation
        """
        num_cols = self.data_utils['num_cols']
        if col_type == 'num':
            for i in range(num_cols):
                self.X_train_imp[:, i][self.X_train_mask[:, i]] = imp_values[i]
        elif col_type == 'cat':
            for i in range(num_cols, self.X_train.shape[1]):
                self.X_train_imp[:, i][self.X_train_mask[:, i]] = imp_values[i]

    def calculate_data_utils(self, data_config: dict):
        """
        Calculate data statistic
        # TODO: add VGM for numerical columns for modeling mixture of clusters
        """
        data_utils = {
            'task_type': data_config['task_type'],
            'n_features': self.X_train.shape[1],
            'num_cols': data_config['num_cols'] if 'num_cols' in data_config else self.X_train.shape[1]
        }

        #########################################################################################################
        # column statistics
        col_stats_dict = {}
        for i in range(self.X_train.shape[1]):
            # numerical stats
            if i < data_utils['num_cols']:
                col_stats_dict[i] = {
                    'min': np.nanmin(self.X_train_ms[:, i]),
                    'max': np.nanmax(self.X_train_ms[:, i]),
                    'mean': np.nanmean(self.X_train_ms[:, i]),
                    'std': np.nanstd(self.X_train_ms[:, i]),
                    'median': np.nanmedian(self.X_train_ms[:, i]),
                }
            # categorical stats
            else:
                col_stats_dict[i] = {
                    'num_class': len(np.unique(self.X_train_ms[:, i][~np.isnan(self.X_train_ms[:, i])])),
                    "mode": stats.mode(self.X_train_ms[:, i][~np.isnan(self.X_train_ms[:, i])], keepdims=False)[0]
                    # TODO: add frequencies
                }

        data_utils['col_stats'] = col_stats_dict

        #########################################################################################################
        # local data and missing data statistics
        data_utils['sample_size'] = self.X_train.shape[0]
        data_utils['missing_rate_cell'] = np.sum(self.X_train_mask) / (self.X_train.shape[0] * self.X_train.shape[1])
        data_utils['missing_rate_rows'] = np.sum(self.X_train_mask, axis=1) / self.X_train.shape[1]
        data_utils['missing_rate_cols'] = np.sum(self.X_train_mask, axis=0) / self.X_train.shape[0]

        missing_stats_cols = {}
        for col_idx in range(self.X_train.shape[1]):
            row_mask = self.X_train_mask[:, col_idx]
            x_obs_mask = self.X_train_mask[~row_mask][:, np.arange(self.X_train_mask.shape[1]) != col_idx]
            missing_stats_cols[col_idx] = {
                'sample_size_obs': x_obs_mask.shape[0],
                'sample_size_obs_pct': x_obs_mask.shape[0] / self.X_train.shape[0],
                'missing_rate_rows': x_obs_mask.any(axis=1).sum() / x_obs_mask.shape[0],
                'missing_rate_cell': x_obs_mask.sum().sum() / (x_obs_mask.shape[0] * x_obs_mask.shape[1]),
                'missing_rate_obs': x_obs_mask.sum() / (x_obs_mask.shape[0] * x_obs_mask.shape[1]),
            }
        data_utils['missing_stats_cols'] = missing_stats_cols

        #########################################################################################################
        # label stats
        if data_utils['task_type'] == 'regression':
            data_utils['label_stats'] = {
                'min': float(np.nanmin(self.y_train)),
                'max': float(np.nanmax(self.y_train)),
                'mean': float(np.nanmean(self.y_train)),
                'std': float(np.nanstd(self.y_train)),
            }
        else:
            data_utils['label_stats'] = {
                'num_class': len(np.unique(self.y_train))
                # TODO: add frequencies
            }

        return data_utils
