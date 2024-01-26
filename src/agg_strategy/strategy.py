from abc import ABC, abstractmethod
from src.client import Client
from typing import Dict, Union, List, Tuple


class Strategy(ABC):

    @abstractmethod
    def aggregate(
            self, local_model_parameters: List[dict], fit_res: List[dict], *args, **kwargs
    ) -> Tuple[List[dict], dict]:
        pass

    @abstractmethod
    def update_local_model(
            self, global_model: dict, local_model:dict, client: Client, *args, **kwargs
    ) -> dict:
        pass
