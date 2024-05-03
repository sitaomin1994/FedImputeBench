from src.fed_strategy.fed_strategy_client import StrategyClient
import torch
from typing import Tuple

class FedAvgStrategyClient(StrategyClient):

    def __init__(self, strategy_params):
        self.strategy_params = strategy_params
        super().__init__('fedavg')

    def fit_local_prox(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Tuple[
        torch.nn.Module, dict]:
        raise NotImplementedError
