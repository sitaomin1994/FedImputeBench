from src.fed_strategy.fed_strategy_client import StrategyClient
import torch
from typing import Tuple
from src.fed_strategy.fed_strategy_client.base_strategy import fit_local_model_base


class FedAvgStrategyClient(StrategyClient):

    def __init__(self, strategy_params):
        self.strategy_params = strategy_params
        super().__init__('fedavg')

    def fit_local_model(
            self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, params: dict
    ) -> Tuple[torch.nn.Module, dict]:

        return fit_local_model_base(model, dataloader, params)
