from src.fed_strategy.fed_strategy_client.base_strategy import StrategyClient, fit_local_model_base
import torch
from typing import Tuple


class LocalStrategyClient(StrategyClient):

    def __init__(self, strategy_params: dict):
        self.strategy_params = strategy_params
        super().__init__('local')

    def fit_local_model(
            self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, params: dict
    ) -> Tuple[torch.nn.Module, dict]:
        return fit_local_model_base(model, dataloader, params)
