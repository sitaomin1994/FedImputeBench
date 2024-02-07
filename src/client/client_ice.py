from typing import Tuple

from .client import Client
import numpy as np
from src.imputation.base import ICEImputer


class ICEClient(Client):

    def __init__(
            self,
            client_id: int,
            train_data: np.ndarray, test_data: np.ndarray, X_train_ms: np.ndarray, data_config: dict,
            imp_model: ICEImputer, client_config:dict, seed=0,
    ) -> None:

        # initialize data and initial imputation
        super().__init__(
            client_id, train_data, test_data, X_train_ms, data_config,
            imp_model, client_config, seed,
        )

        # initialize imputation model
        self.imp_model = imp_model
        self.imp_model.initialize(self.data_utils, seed)

    def fit_local_imputation_model(self, feature_idx: int, imp_params: dict) -> Tuple[dict, dict]:
        """
        Fit local imputation model
        """
        fit_res = self.imp_model.fit(
            self.X_train_imp, self.y_train, self.X_train_mask, feature_idx
        )

        model_parameters = self.imp_model.get_imp_model_params(feature_idx)
        fit_res.update(self.data_utils)
        fit_res['sample_size'] = self.data_utils['missing_stats_cols'][feature_idx]['sample_size_obs']
        # TODO: design a consistent structure for fit_res, fit_params, imp_res, imp_params and document it

        return model_parameters, fit_res

    def imputation(self, updated_local_model:dict, feature_idx: int):
        """
        Imputation
        """
        self.imp_model.update_imp_model(updated_local_model, feature_idx)
        self.imp_model.impute(self.X_train_imp, self.y_train, self.X_train_mask, feature_idx)
