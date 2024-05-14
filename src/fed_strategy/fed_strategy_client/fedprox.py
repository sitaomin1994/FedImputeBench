from src.fed_strategy.fed_strategy_client.base_strategy import StrategyClient
import torch
from typing import Tuple
from .utils import trainable_params

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FedProxStrategyClient(StrategyClient):

    def __init__(self, strategy_params: dict):
        self.strategy_params = strategy_params
        super().__init__('fedprox')

        self.mu = strategy_params.get('mu', 0.01)
        self.global_model_params = None

    def pre_training_setup(self, model: torch.nn.Module, params: dict):
        self.global_model_params = trainable_params(model)

    def fed_updates(self, model: torch.nn.Module):
        if self.global_model_params is None:
            raise ValueError(
                "Global model parameters not initialized, please persist global model in pretraining_setup first"
        )

        for w, w_t in zip(trainable_params(model), self.global_model_params):
            w.grad.data += self.mu * (w.data - w_t.data)

    def post_training_setup(self, model: torch.nn.Module):
        self.global_model_params = None
