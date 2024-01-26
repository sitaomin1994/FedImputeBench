import numpy as np

from .server import Server
from typing import Dict, Union, List, Tuple
from src.client import Client, ICEClient
from src.agg_strategy import Strategy
from src.imputation.initial_imputation import initial_imputation_num, initial_imputation_cat
from ..evaluation.evaluator import Evaluator


class ServerICE(Server):

    def __init__(self):
        super().__init__()

    def run_fed_imputation(
            self, clients: List[ICEClient], agg_strategy: Strategy, workflow_params: dict
    ) -> dict:
        """
        Imputation workflow for MICE
        """
        ############################################################################################################
        # Workflow Parameters
        data_dim = workflow_params['data_dim']
        iterations = workflow_params['iterations']

        ############################################################################################################
        # Update Global clip thresholds
        if agg_strategy.name == 'local':
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
        initial_imp_num = workflow_params['initial_imp_num']
        initial_imp_cat = workflow_params['initial_imp_cat']
        initial_imputation_num(initial_imp_num, [client.data_utils for client in clients])
        initial_imputation_cat(initial_imp_cat, [client.data_utils for client in clients])

        ############################################################################################################
        # Federated Imputation Workflow
        ret = {}
        for epoch in range(iterations):

            ########################################################################################################
            # Evaluation
            self.tracker.imp_quality.append((0, self.evaluator.evaluate_imputation(clients)))

            ########################################################################################################
            # federated imputation for each feature
            for feature_idx in range(data_dim):

                # local training of imputation model
                local_models, clients_fit_res = [], []
                for client in clients:
                    model_parameter, fit_res = client.fit_local_imputation_model(
                        feature_idx=feature_idx, imp_params={})
                    local_models.append(model_parameter)
                    clients_fit_res.append(fit_res)

                # aggregate local imputation model
                global_models, agg_res = agg_strategy.aggregate(
                    local_model_parameters=local_models, fit_res=clients_fit_res
                )

                # updates local imputation model and do imputation
                for global_model, local_model, client in zip(global_models, local_models, clients):
                    updated_local_model = agg_strategy.update_local_model(global_model, local_model, client)
                    client.imputation(updated_local_model, feature_idx=feature_idx)

        ########################################################################################################
        # Final Evaluation
        self.tracker.imp_quality.append((iterations, self.evaluator.evaluate_imputation(clients)))

        return ret
