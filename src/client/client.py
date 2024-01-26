import numpy as np
from src.imputation.base import BaseImputer


class Client:

    def __init__(
            self,
            client_id: int,
            train_data: np.ndarray, test_data: np.ndarray, X_train_ms: np.ndarray, data_config: dict,
            imp_model: BaseImputer,
            seed=0
    ) -> None:
        # client id
        self.client_id = client_id
        self.seed = seed

        # data
        self.data_config = data_config
        self.X_train, self.y_train = train_data[:, :-1], train_data[:, -1]  # training data
        self.X_test, self.y_test = test_data[:, :-1], test_data[:, -1]  # testing data
        self.X_train_ms = X_train_ms  # missing data
        self.X_train_mask = np.isnan(self.X_train_ms)  # missing data mask
        self.X_train_imp = self.X_train_ms.copy()  # imputed data

        # local imputation model
        self.imp_model = imp_model

        # evaluation result
        self.eval_ret = None

        # data utils
        self.data_utils = {}
        self.calculate_data_utils()

    def initial_impute(self):
        """
        Initial imputation using mean imputation # TODO: add other imputation methods
        """
        self.X_train_imp[self.X_train_mask] = 0

    def get_sample_size(self):
        return self.X_train.shape[0]

    def calculate_data_utils(self):
        """
        Calculate data statistic
        # TODO: add VGM for numerical columns for modeling mixture of clusters
        """

        #########################################################################################################
        # task type
        self.data_utils['task_type'] = self.data_config['task_type']
        # numerical columns
        self.data_utils['num_cols'] = \
            self.data_config['num_cols'] if 'num_cols' in self.data_config else self.X_train.shape[1]

        #########################################################################################################
        # column statistics
        col_stats_dict = {}
        for i in range(self.X_train.shape[1]):
            # numerical stats
            if i < self.data_utils['num_cols']:
                col_stats_dict[i] = {
                    'min': np.nanmin(self.X_train_ms[:, i]),
                    'max': np.nanmax(self.X_train_ms[:, i]),
                    'mean': np.nanmean(self.X_train_ms[:, i]),
                    'std': np.nanstd(self.X_train_ms[:, i]),
                }
            # categorical stats
            else:
                col_stats_dict[i] = {
                    'num_class': len(np.unique(self.X_train_ms[:, i][~np.isnan(self.X_train_ms[:, i])]))
                    # TODO: add frequencies
                }

        self.data_utils['col_stats'] = col_stats_dict

        #########################################################################################################
        # local data and missing data statistics
        self.data_utils['sample_size'] = self.X_train.shape[0]
        self.data_utils['missing_rate_cell'] = np.sum(self.X_train_mask) / (self.X_train.shape[0] * self.X_train.shape[1])
        self.data_utils['missing_rate_rows'] = np.sum(self.X_train_mask, axis=1) / self.X_train.shape[1]
        self.data_utils['missing_rate_cols'] = np.sum(self.X_train_mask, axis=0) / self.X_train.shape[0]

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
        self.data_utils['missing_stats_cols'] = missing_stats_cols

        #########################################################################################################
        # label stats
        if self.data_utils['task_type'] == 'regression':
            self.data_utils['label_stats'] = {
                'min': np.nanmin(self.y_train),
                'max': np.nanmax(self.y_train),
                'mean': np.nanmean(self.y_train),
                'std': np.nanstd(self.y_train),
            }
        else:
            self.data_utils['label_stats'] = {
                'num_class': len(np.unique(self.y_train))
                # TODO: add frequencies
            }