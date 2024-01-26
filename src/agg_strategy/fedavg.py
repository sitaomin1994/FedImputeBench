from abc import ABC, abstractmethod
from collections import OrderedDict

from src.client import Client
from typing import Dict, Union, List, Tuple
from .strategy import Strategy
from copy import deepcopy


class FedAvg(Strategy):

    def aggregate(
            self, local_model_parameters: List[dict], fit_res: List[dict], *args, **kwargs
    ) -> Tuple[List[dict], dict]:

        # federated averaging implementation
        averaged_model_state_dict = OrderedDict()  # global parameters
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

    def update_local_model(
            self, global_model: dict, local_model:dict, client: Client, *args, **kwargs
    ) -> dict:

        return global_model

