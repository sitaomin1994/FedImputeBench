from abc import ABC, abstractmethod
from typing import Dict, Union, List, Tuple
from src.client import Client
from src.agg_strategy import Strategy
from src.evaluation.evaluator import Evaluator
from src.utils.tracker import Tracker


class Server(ABC):

    def __init__(self):
        self.evaluator: Evaluator = Evaluator()
        self.tracker: Tracker = Tracker()

    @abstractmethod
    def run_fed_imputation(
            self, clients: List[Client], agg_strategy: Strategy, workflow_params: dict
    ) -> dict:
        pass

    def save_results(self):
        return self.tracker.to_dict()
