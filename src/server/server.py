from typing import Dict, Union, List, Tuple
from src.evaluation.evaluator import Evaluator
from src.loaders.load_agg_strategy import load_agg_strategy
from src.utils.tracker import Tracker


class Server:

    def __init__(
            self,
            agg_strategy_name: str,
            agg_strategy_params: Dict[str, Union[str, int, float]],
            server_config: Dict[str, Union[str, int, float]],
    ):
        self.aggregation_strategy = load_agg_strategy(agg_strategy_name, agg_strategy_params)
        self.evaluator: Evaluator = Evaluator()
        self.tracker: Tracker = Tracker()
        self.server_config = server_config

    def aggregate(self, local_model_parameters: List[dict], fit_res: List[dict]) -> Tuple[List[dict], dict]:
        """
        Aggregate local imputation models
        """
        raise NotImplementedError

    def global_evaluation(self, eval_res: dict) -> dict:
        # global evaluation of imputation models
        raise NotImplementedError
