from collections import OrderedDict

from src.client import Client
from typing import Dict, Union, List, Tuple
from .strategy import Strategy
from copy import deepcopy


class Local(Strategy):

    def aggregate(
            self, local_model_parameters: List[dict], fit_res: List[dict], *args, **kwargs
    ) -> Tuple[List[dict], dict]:

        return local_model_parameters, {}

    def update_local_model(
            self, global_model: dict, local_model:dict, client: Client, *args, **kwargs
    ) -> dict:

        return global_model

