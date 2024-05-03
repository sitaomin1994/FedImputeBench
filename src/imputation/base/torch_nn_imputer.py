from abc import ABCMeta, abstractmethod
from typing import Tuple
import torch
from src.imputation.base.base_imputer import BaseImputer


class TorchNNImputer(BaseImputer, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def fetch_model(
            self, params, X_train_imp, y_train, X_train_mask
    ) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader]:
        pass
