from abc import ABC, abstractmethod
import torch
from typing import Tuple

class StrategyClient(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit_local_prox(
            self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[torch.nn.Module, dict]:
        pass
