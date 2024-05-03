from copy import deepcopy
from typing import List, Tuple
from collections import OrderedDict
from src.fed_strategy.fed_strategy_server.base_strategy import StrategyServer


class LocalStrategyServer(StrategyServer):

    def __init__(self, strategy_params: dict):
        super(LocalStrategyServer, self).__init__('local')
        self.strategy_params = strategy_params

    def aggregate_parameters(
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


    def fit_instruction(self, params_list: List[dict]) -> List[dict]:

        return [{'fit_model': True} for _ in range(len(params_list))]