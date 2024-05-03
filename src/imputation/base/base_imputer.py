
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np


class BaseImputer(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def get_imp_model_params(self, params: dict) -> OrderedDict:
        """
        Return model parameters
        :param params: dict contains parameters for get_imp_model_params
        :return: OrderedDict - model parameters dictionary
        """
        pass

    @abstractmethod
    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:
        """
        Set model parameters
        :param updated_model_dict: global model parameters dictionary
        :param params: parameters for set parameters function
        :return: None
        """
        pass

    @abstractmethod
    def initialize(self, data_utils: dict, params: dict, seed: int) -> None:
        """
        Initialize imputer - statistics imputation models etc.
        :param data_utils: data utils dictionary - contains information about data
        :param params: params for initialization
        :param seed: int - seed for randomization
        :return: None
        """
        pass

    @abstractmethod
    def fit(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> dict:
        """
        Fit imputer to train local imputation models
        :param X: features - float numpy array
        :param y: target
        :param missing_mask: missing mask
        :param params: parameters for local training
        :return: fit results of local training
        """
        pass

    @abstractmethod
    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:
        """
        Impute missing values using imputation model
        :param X: numpy array of features
        :param y: numpy array of target
        :param missing_mask: missing mask
        :param params: parameters for imputation
        :return: imputed data - numpy array - same dimension as X
        """
        pass
