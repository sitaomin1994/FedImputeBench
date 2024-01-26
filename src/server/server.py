from abc import ABC, abstractmethod
from typing import Dict, Union, List, Tuple
from src.client import Client
from src.agg_strategy import Strategy


class Server(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def run_fed_imputation(
            self, clients:List[Client], agg_strategy: Strategy, workflow_params:dict
    ) -> dict:
        pass
