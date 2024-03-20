from src.server import Server
from typing import List
from src.client import Client
from src.evaluation.evaluator import Evaluator

from src.imputation.initial_imputation import initial_imputation_num, initial_imputation_cat
from .workflow import BaseWorkflow
from tqdm.auto import trange

from ..utils.tracker import Tracker


class WorkflowJM(BaseWorkflow):

    def __init__(
            self,
            workflow_params: dict
    ):
        super().__init__()
        self.workflow_params = workflow_params
        self.tracker = None

    def fed_imp_sequential(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker
    ) -> Tracker:

        """
        Imputation workflow for Joint Modeling Imputation
        """
        initial_imp_num = self.workflow_params['initial_imp_num']
        initial_imp_cat = self.workflow_params['initial_imp_cat']
        iterations = self.workflow_params['imp_iterations']
        evaluation_interval = self.workflow_params['evaluation_interval']

        ############################################################################################################
        # Initial Imputation
        initial_data_num = initial_imputation_num(initial_imp_num, [client.data_utils for client in clients])
        initial_data_cat = initial_imputation_cat(initial_imp_cat, [client.data_utils for client in clients])
        for client_idx, client in enumerate(clients):
            client.initial_impute(initial_data_num[client_idx], col_type='num')
            client.initial_impute(initial_data_cat[client_idx], col_type='cat')

        self.eval_and_track(evaluator, tracker, clients, phase='initial')

        ############################################################################################################
        # Federated Imputation Workflow
        for epoch in trange(iterations, desc='Global Epoch', colour='blue'):

            ########################################################################################################
            # Evaluation
            self.eval_and_track(
                evaluator, tracker, clients, phase='round', epoch=epoch, iterations=iterations,
                evaluation_interval=evaluation_interval
            )

            ########################################################################################################
            # local training of imputation model
            local_models, clients_fit_res = [], []
            for client_idx in trange(len(clients), desc='Client idx', colour='blue', leave=False):
                client = clients[client_idx]
                model_parameter, fit_res = client.fit_local_imp_model(params={})
                local_models.append(model_parameter)
                clients_fit_res.append(fit_res)

            # aggregate local imputation model
            global_models, agg_res = server.aggregate(
                local_model_parameters=local_models, fit_res=clients_fit_res
            )

            # updates local imputation model and do imputation
            for global_model, client in zip(global_models, clients):
                # updated_local_model = agg_strategy.update_local_model(global_model, local_model, client)
                client.update_local_imp_model(global_model, params={})
                client.local_imputation(params={})

        ########################################################################################################
        # Evaluation
        self.eval_and_track(evaluator, tracker, clients, phase='final', iterations=iterations)

        return tracker

    def fed_imp_parallel(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker
    ) -> Tracker:
        return tracker
