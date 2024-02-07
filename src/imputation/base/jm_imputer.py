from abc import ABC, abstractmethod

import numpy as np
from src.imputation.base import BaseImputer


class JMImputer(BaseImputer):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_imp_model_params(self) -> dict:
        pass

    @abstractmethod
    def update_imp_model(self, updated_model: dict) -> None:
        pass

    @abstractmethod
    def initialize(self, data_utils, seed) -> None:
        pass

    @abstractmethod
    def fit_local_imp_model(
            self, X_train_imp: np.ndarray, X_train_mask: np.ndarray, X_train_full: np.ndarray, y_train: np.ndarray,
            params: dict
    ) -> dict:
        pass

    @abstractmethod
    def imputation(self, X_train_ms: np.ndarray, X_train_mask: np.ndarray, params: dict) -> np.ndarray:
        pass
