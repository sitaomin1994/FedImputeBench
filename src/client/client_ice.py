from typing import Tuple

from .client import Client
import numpy as np
from src.imputation.base import ICEImputer, BaseImputer


class ICEClient(Client):

    def __init__(
            self, client_id: int,
            train_data: np.ndarray, test_data: np.ndarray, X_train_ms: np.ndarray, data_config: dict,
            imp_model: ICEImputer, seed=0
    ) -> None:

        # Super call
        super().__init__(client_id, train_data, test_data, X_train_ms, data_config, imp_model, seed)

        # initial imputation
        self.imp_model.initial_imputation(self.X_train_imp, self.X_train_mask)

    def fit_local_imputation_model(self, feature_idx: int, imp_params: dict) -> Tuple[dict, dict]:
        """
        Fit local imputation model
        """
        model_parameter, fit_res = self.imp_model.fit(
            self.X_train_ms, self.y_train, self.X_train_mask, feature_idx
        )

        sample_size = self.get_sample_size()
        fit_res['sample_size'] = sample_size

        return model_parameter, fit_res

    def imputation(self, updated_local_model:dict, feature_idx: int):
        """
        Imputation
        """
        self.imp_model.update(updated_local_model, feature_idx)
        self.imp_model.impute(self.X_train_imp, self.X_train_mask, feature_idx)
