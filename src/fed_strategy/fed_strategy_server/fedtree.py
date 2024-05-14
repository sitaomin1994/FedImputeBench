from copy import deepcopy
from typing import List, Tuple
from collections import OrderedDict

from src.fed_strategy.fed_strategy_server.base_strategy import StrategyServer


class FedAvgStrategyServer(StrategyServer):

    def __init__(self, strategy_params):
        super(FedAvgStrategyServer, self).__init__('fedavg')
        self.strategy_params = strategy_params
        self.fine_tune_epochs = strategy_params.get('fine_tune_steps', 0)

    def aggregate_parameters(
            self, local_model_parameters: List[OrderedDict], fit_res: List[dict], params: dict, *args, **kwargs
    ) -> Tuple[List[OrderedDict], dict]:
        """
        Aggregate local models
        :param local_model_parameters: List of local model parameters
        :param fit_res: List of fit results of local training
            - sample_size: int - number of samples used for training
        :param params: dictionary for information
        :param args: other params list
        :param kwargs: other params dict
        :return: List of aggregated model parameters, dict of aggregated results
        """

        # federated averaging implementation
        averaged_model_state_dict = OrderedDict([])  # global parameters
        sample_sizes = [item['sample_size'] for item in fit_res]
        normalized_coefficient = [size / sum(sample_sizes) for size in sample_sizes]

        for it, local_model_state_dict in enumerate(local_model_parameters):
            for key in local_model_state_dict.keys():
                if it == 0:
                    averaged_model_state_dict[key] = normalized_coefficient[it] * local_model_state_dict[key]
                else:
                    averaged_model_state_dict[key] += normalized_coefficient[it] * local_model_state_dict[key]

        # copy parameters for each client
        agg_model_parameters = [deepcopy(averaged_model_state_dict) for _ in range(len(local_model_parameters))]
        agg_res = {}

        return agg_model_parameters, agg_res

    def fit_instruction(self, params_list: List[dict]) -> List[dict]:

        return [{'fit_model': True} for _ in range(len(params_list))]

    def update_instruction(self, params: dict) -> dict:

        return {}