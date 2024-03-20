from abc import ABC, abstractmethod

import numpy as np
from src.imputation.base import BaseImputer


class JMImputer(BaseImputer):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_imp_model_params(self, params) -> dict:
        pass

    @abstractmethod
    def update_imp_model(self, updated_model: dict, params) -> None:
        pass

    @abstractmethod
    def initialize(self, data_utils, params, seed) -> None:
        pass

    @abstractmethod
    def fit(
            self, X, y: np.ndarray, missing_mask: np.ndarray, params: dict
    ) -> dict:
        pass

    @abstractmethod
    def impute(self, X: np.ndarray, y: np.ndarray, missing_mask:np.ndarray, params: dict) -> np.ndarray:
        pass
