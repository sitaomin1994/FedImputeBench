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
            initial_imp_num: str = 'mean', initial_imp_cat: str = 'mode',
    ) -> None:

        # initialize data and initial imputation
        super().__init__(
            client_id, train_data, test_data, X_train_ms, data_config,
            imp_model, client_config, seed,
            initial_imp_num, initial_imp_cat
        )

        # initialize imputation model
        self.imp_model = imp_model
        self.imp_model.initialize(self.data_utils, seed)

    def fit_local_imputation_model(self, feature_idx: int, imp_params: dict) -> Tuple[dict, dict]:
        """
        Fit local imputation model
        """
        model_parameter, fit_res = self.imp_model.fit(
            self.X_train_ms, self.y_train, self.X_train_mask, feature_idx
        )

        fit_res = self.data_utils
        return model_parameter, fit_res

    def imputation(self, updated_local_model:dict, feature_idx: int):
        """
        Imputation
        """
        self.imp_model.update_imp_model(updated_local_model, feature_idx)
        self.imp_model.impute(self.X_train_imp, self.y_train, self.X_train_mask, feature_idx)
