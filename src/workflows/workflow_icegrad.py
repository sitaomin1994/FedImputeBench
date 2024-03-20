import numpy as np

from src.server import Server
from typing import List
from src.client import Client
from src.imputation.initial_imputation import initial_imputation_num, initial_imputation_cat
from .workflow import BaseWorkflow
from ..evaluation.evaluator import Evaluator
from tqdm.auto import trange

from ..utils.tracker import Tracker


class WorkflowICEGrad(BaseWorkflow):

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
        data_dim = self.workflow_params['data_dim']
        iterations = self.workflow_params['imp_iterations']
        model_epochs = self.workflow_params['model_epochs']
        model_converge_tol = self.workflow_params['model_converge_tol']  # TODO: not used
        model_converge_patience = self.workflow_params['model_converge_patience']  # TODO: not used

        initial_imp_num = self.workflow_params['initial_imp_num']
        initial_imp_cat = self.workflow_params['initial_imp_cat']

        evaluation_interval = self.workflow_params['evaluation_interval']

        ############################################################################################################
        # Update Global clip thresholds
        if server.aggregation_strategy.name == 'local':
            initial_values_min, initial_values_max = [], []
            for client_id, client in enumerate(clients):
                initial_values_min.append(client.imp_model.min_values)
                initial_values_max.append(client.imp_model.max_values)
            global_min_values = np.min(np.array(initial_values_min), axis=0, initial=0)
            global_max_values = np.max(np.array(initial_values_max), axis=0, initial=1)
            for client_id, client in enumerate(clients):
                client.imp_model.set_clip_thresholds(global_min_values, global_max_values)

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
            # Federated imputation for each feature
            for feature_idx in trange(data_dim, desc='Feature_idx', leave=False, colour='blue'):

                # Collaboratively training imputation model using gradient-based method
                for model_epoch in trange(model_epochs, desc='Model Epochs', leave=False, colour='blue'):

                    # local training of imputation model
                    local_models, clients_fit_res = [], []
                    for client in clients:
                        # TODO: dynamic local training
                        model_parameter, fit_res = client.fit_local_imp_model(params={'feature_idx': feature_idx})
                        local_models.append(model_parameter)
                        clients_fit_res.append(fit_res)

                    # aggregate local imputation model
                    global_models, agg_res = server.aggregate(
                        local_model_parameters=local_models, fit_res=clients_fit_res
                    )

                    # updates local imputation model and do imputation
                    for global_model, client in zip(global_models, clients):
                        # updated_local_model = agg_strategy.update_local_model(global_model, local_model, client)
                        client.update_imp_model(global_model, params={'feature_idx': feature_idx})

                    # TODO: how about some clients are already converged? -> client need to be check convergence locally

                # Local imputation using updated imputation models
                for client in clients:
                    client.local_imputation(params={'feature_idx': feature_idx})

        ########################################################################################################
        # Final Evaluation and Tracking
        self.eval_and_track(evaluator, tracker, clients, phase='final', iterations=iterations)

        return tracker

    def fed_imp_parallel(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker
    ) -> Tracker:
        return tracker
