from collections import OrderedDict

from src.client import Client
from typing import Dict, Union, List, Tuple

from .agg_strategy import AggStrategy
from copy import deepcopy


class LocalStrategy(AggStrategy):

    def __init__(self, **kwargs):
        super().__init__(name='local')

    def local_train_instruction(self, Clients: List[Client]) -> List[dict]:
        """
        Local training instructions
        :return:
        """
        local_instructions = [deepcopy({'fit_local': True}) for _ in range(len(Clients))]

        return local_instructions

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

        return local_model_parameters, {}

    # def update_local_model(
    #         self, global_model: dict, local_model: dict, client: BaseClient, *args, **kwargs
    # ) -> dict:
    #     return global_model
