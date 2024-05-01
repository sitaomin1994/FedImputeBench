from copy import deepcopy
from typing import List, OrderedDict, Tuple
from src.agg_strategy.agg_strategy import AggStrategy
from src.client.client import Client


class CentralStrategy(AggStrategy):

    def __init__(self, name: str):
        super().__init__(name='central')
        self.name = name

    def local_train_instruction(self, Clients: List[Client]) -> List[dict]:
        """
        Local training instructions
        :return:
        """
        local_instructions = [deepcopy({'fit_local': False}) for _ in range(len(Clients))]
        local_instructions[-1]['fit_local'] = True  # only last client (contains all data) fits local model

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
        central_model_params = local_model_parameters[-1]

        agg_model_parameters = [deepcopy(central_model_params) for _ in range(len(local_model_parameters))]
        agg_res = {}

        return agg_model_parameters, agg_res
