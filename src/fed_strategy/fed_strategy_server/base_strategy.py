from abc import ABC, abstractmethod
from typing import List, OrderedDict, Tuple


class StrategyServer(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def aggregate_parameters(
            self, local_model_parameters: List[OrderedDict], fit_res: List[dict], *args, **kwargs
    ) -> Tuple[List[OrderedDict], dict]:
        pass

    @abstractmethod
    def fit_instruction(self, params_list: List[dict]) -> List[dict]:
        pass

    @abstractmethod
    def update_instruction(self, params: dict) -> dict:
        pass