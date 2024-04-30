
from abc import ABC, abstractmethod


class BaseImputer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_imp_model_params(self, params) -> dict:
        pass

    @abstractmethod
    def set_imp_model_params(self, updated_model, params):
        pass

    @abstractmethod
    def initialize(self, data_utils, params, seed):
        pass

    @abstractmethod
    def fit(self, X, y, missing_mask, params) -> dict:
        pass

    @abstractmethod
    def impute(self, X, y, missing_mask, params) -> np.ndarray:
        pass