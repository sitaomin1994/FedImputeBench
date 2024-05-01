from abc import ABC, abstractmethod
from src.client import Client
from typing import Dict, Union, List, Tuple
from collections import OrderedDict
from src.client.client import Client


class AggStrategy(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def local_train_instruction(self, Clients: List[Client]) -> List[dict]:
        """
        Local training instructions
        :param Clients: List of clients
        :return: List of dictionaries containing local training instructions
        """
        pass

    @abstractmethod
    def aggregate(
            self, local_model_parameters: List[OrderedDict], fit_res: List[dict], *args, **kwargs
    ) -> Tuple[List[OrderedDict], dict]:
        """
        Aggregate local models
        :param local_model_parameters: List of local model parameters
        :param fit_res: List of fit results of local training
        :param args: other params list
        :param kwargs: other params dict
        :return: List of aggregated model parameters, dict of aggregated results
        """
        pass

    # @abstractmethod
    # def update_local_model(
    #         self, global_model: dict, local_model: dict, client: Client, *args, **kwargs
    # ) -> dict:
    #     pass
