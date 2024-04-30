import numpy as np

from .workflow import BaseWorkflow
from src.server import Server
from typing import List
from src.client import Client
from src.imputation.initial_imputation import initial_imputation_num, initial_imputation_cat
from ..evaluation.evaluator import Evaluator

from tqdm.auto import trange
from src.utils.tracker import Tracker


class WorkflowICE(BaseWorkflow):

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
        Imputation workflow for MICE Sequential Version
        """
        ############################################################################################################
        # Workflow Parameters
        data_dim = clients[0].X_train.shape[1]
        iterations = self.workflow_params['imp_iterations']
        initial_imp_num = self.workflow_params['initial_imp_num']
        initial_imp_cat = self.workflow_params['initial_imp_cat']
        evaluation_interval = self.workflow_params['evaluation_interval']

        ############################################################################################################
        # Update Global clip thresholds
        if server.aggregation_strategy.name == 'local':
            initial_values_min, initial_values_max = [], []
            for client_id, client in enumerate(clients):
                initial_values_min.append(client.imp_model.min_values)  # encapsulate these
                initial_values_max.append(client.imp_model.max_values)  # encapsulate these
            global_min_values = np.min(np.array(initial_values_min), axis=0, initial=0)
            global_max_values = np.max(np.array(initial_values_max), axis=0, initial=1)
            for client_id, client in enumerate(clients):
                client.imp_model.set_clip_thresholds(
                    global_min_values, global_max_values
                )  # encapsulate these interfaces

        ############################################################################################################
        # Initial Imputation
        initial_data_num = initial_imputation_num(initial_imp_num, [client.data_utils for client in clients])
        initial_data_cat = initial_imputation_cat(initial_imp_cat, [client.data_utils for client in clients])
        for client_idx, client in enumerate(clients):
            client.initial_impute(initial_data_num[client_idx], col_type='num')
            client.initial_impute(initial_data_cat[client_idx], col_type='cat')

        # initial evaluation and tracking
        self.eval_and_track(evaluator, tracker, clients, phase='initial')

        ############################################################################################################
        # Federated Imputation Sequential Workflow
        for epoch in trange(iterations, desc='ICE Iterations', colour='blue'):

            ########################################################################################################
            # Evaluation
            self.eval_and_track(
                evaluator, tracker, clients, phase='round', epoch=epoch, iterations=iterations,
                evaluation_interval=evaluation_interval
            )

            ########################################################################################################
            # federated imputation for each feature
            for feature_idx in trange(data_dim, desc='Feature_idx', leave=False, colour='blue'):
                # client local train imputation model
                local_models, clients_fit_res = [], []
                for client in clients:
                    # TODO: dynamic local training
                    model_parameter, fit_res = client.fit_local_imp_model(params={'feature_idx': feature_idx})
                    local_models.append(model_parameter)
                    clients_fit_res.append(fit_res)

                # server aggregate local imputation model
                global_models, agg_res = server.aggregate(
                    local_model_parameters=local_models, fit_res=clients_fit_res
                )

                # client update local imputation model and do imputation
                for global_model, client in zip(global_models, clients):
                    # TODO: see how to do this when using new strategies
                    # updated_local_model = self.server.update_local_model(global_model, local_model, client)
                    client.set_imp_model_params(global_model, params={'feature_idx': feature_idx})
                    client.local_imputation(params={'feature_idx': feature_idx})

        ########################################################################################################
        # Final Evaluation and Tracking
        self.eval_and_track(evaluator, tracker, clients, phase='final', iterations=iterations)

        return tracker

    def fed_imp_parallel(self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker):
        pass
