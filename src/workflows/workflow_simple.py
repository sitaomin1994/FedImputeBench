from .workflow import BaseWorkflow
from src.server import Server
from typing import List
from src.client import Client
from src.imputation.initial_imputation.initial_imputation import initial_imputation
from ..evaluation.evaluator import Evaluator
from src.utils.tracker import Tracker
from .utils import formulate_centralized_client, update_clip_threshold


class WorkflowSimple(BaseWorkflow):

    def __init__(
            self,
            workflow_params: dict
    ):
        super().__init__()
        self.workflow_params = workflow_params
        self.tracker = None

    def fed_imp_sequential(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker, train_params: dict
    ) -> Tracker:

        """
        Imputation workflow for MICE Sequential Version
        """
        ############################################################################################################
        # Workflow Parameters
        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))

        ############################################################################################################
        # Initial Imputation
        clients = initial_imputation(server.fed_strategy.strategy_params['initial_impute'], clients)

        # initial evaluation and tracking
        self.eval_and_track(evaluator, tracker, clients, phase='initial')

        ############################################################################################################
        # federated imputation
        params_list, fit_rest_list = [], []
        fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
        for client in clients:
            train_params.update(fit_instruction[client.client_id])
            params, fit_res = client.fit_local_imp_model(train_params)
            params_list.append(params)
            fit_rest_list.append(fit_res)

        global_models, agg_res = server.fed_strategy.aggregate_parameters(params_list, fit_rest_list, {})

        for global_model, client in zip(global_models, clients):
            client.update_local_imp_model(global_model, params={})
            client.local_imputation(params={})

        ########################################################################################################
        # Final Evaluation and Tracking
        self.eval_and_track(evaluator, tracker, clients, phase='final', iterations=1)

        return tracker

    def fed_imp_parallel(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker, train_params: dict
    ) -> Tracker:
        pass
