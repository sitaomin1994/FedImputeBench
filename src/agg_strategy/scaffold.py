from collections import OrderedDict

from src.client import Client
from typing import Dict, Union, List, Tuple

from . import FedAvgStrategy
from .agg_strategy import AggStrategy
from copy import deepcopy


class ScaffoldStrategy(FedAvgStrategy):

    def __init__(self, **kwargs):
        super().__init__(name='scaffold')

    def local_train_instruction(self, Clients: List[Client]) -> List[dict]:
        """
        Local training instructions
        :return:
        """
        local_instructions = [deepcopy({'fit_local': True, 'correction_fit': True}) for _ in range(len(Clients))]

        return local_instructions

