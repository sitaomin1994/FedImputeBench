import numpy as np

from .utils import formulate_centralized_client, update_clip_threshold
from .workflow import BaseWorkflow
from src.server import Server
from typing import List
from src.client import Client
from src.imputation.initial_imputation import initial_imputation_num, initial_imputation_cat
from ..evaluation.evaluator import Evaluator

from tqdm.auto import trange
from src.utils.tracker import Tracker
from ..imputation.initial_imputation.initial_imputation import initial_imputation


class WorkflowICE(BaseWorkflow):

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
        data_dim = clients[0].X_train.shape[1]
        iterations = self.workflow_params['imp_iterations']
        evaluation_interval = self.workflow_params['evaluation_interval']

        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))

        ############################################################################################################
        # Update Global clip thresholds
        if server.fed_strategy.name != 'local':
            update_clip_threshold(clients)

        ############################################################################################################
        # Initial Imputation
        clients = initial_imputation(server.fed_strategy.strategy_params['initial_impute'], clients)

        # initial evaluation and tracking
        self.eval_and_track(evaluator, tracker, clients, phase='initial')

        ############################################################################################################
        # Federated Imputation Sequential Workflow
        for epoch in trange(iterations, desc='ICE Iterations', colour='blue'):

            ########################################################################################################
            # Evaluation
            if epoch % evaluation_interval == 0 or epoch >= iterations - 3:
                self.eval_and_track(
                    evaluator, tracker, clients, phase='round', epoch=epoch, iterations=iterations,
                    evaluation_interval=evaluation_interval
                )

            ########################################################################################################
            # federated imputation for each feature
            for feature_idx in trange(data_dim, desc='Feature_idx', leave=False, colour='blue'):
                # client local train imputation model
                local_models, clients_fit_res = [], []
                fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
                for client in clients:
                    fit_params = {'feature_idx': feature_idx}
                    fit_params.update(train_params)
                    fit_params.update(fit_instruction[client.client_id])
                    model_parameter, fit_res = client.fit_local_imp_model(params=fit_params)
                    local_models.append(model_parameter)
                    clients_fit_res.append(fit_res)

                # server aggregate local imputation model
                global_models, agg_res = server.fed_strategy.aggregate_parameters(
                    local_model_parameters=local_models, fit_res=clients_fit_res, params = {}
                )

                # client update local imputation model and do imputation
                for global_model, client in zip(global_models, clients):
                    client.update_local_imp_model(global_model, params={'feature_idx': feature_idx})
                    client.local_imputation(params={'feature_idx': feature_idx})

        ########################################################################################################
        # Final Evaluation and Tracking
        self.eval_and_track(evaluator, tracker, clients, phase='final', iterations=iterations)

        return tracker

    def fed_imp_parallel(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker, train_params: dict
    ) -> Tracker:
        pass
