from src.fed_strategy.fed_strategy_client.base_strategy import StrategyClient
import torch
from typing import Tuple


class LocalStrategyClient(StrategyClient):

    def __init__(self, strategy_params: dict):
        self.strategy_params = strategy_params
        super().__init__('local')

    def fit_local_prox(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Tuple[
        torch.nn.Module, dict]:
        raise NotImplementedError
