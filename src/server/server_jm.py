from .server import Server
from typing import Dict, Union, List, Tuple
from src.client import Client, JMClient
from src.agg_strategy import Strategy
from ..imputation.initial_imputation import initial_imputation_num, initial_imputation_cat
from tqdm.auto import tqdm, trange


class ServerJM(Server):

    def __init__(self):
        super().__init__()

    def run_fed_imputation(
            self, clients: List[JMClient], agg_strategy: Strategy, workflow_params: dict
    ) -> dict:

        """
        Imputation workflow for Joint Modeling Imputation
        """
        ############################################################################################################
        # Initial Imputation
        initial_imp_num = workflow_params['initial_imp_num']
        initial_imp_cat = workflow_params['initial_imp_cat']
        initial_data_num = initial_imputation_num(initial_imp_num, [client.data_utils for client in clients])
        initial_data_cat = initial_imputation_cat(initial_imp_cat, [client.data_utils for client in clients])
        for client_idx, client in enumerate(clients):
            client.initial_impute(initial_data_num[client_idx], col_type='num')
            client.initial_impute(initial_data_cat[client_idx], col_type='cat')

        ############################################################################################################
        # Federated Imputation Workflow
        iterations = workflow_params['imp_iterations']
        ret = {}
        for epoch in trange(iterations, desc='Global Epoch', colour='blue'):

            ########################################################################################################
            # Evaluation
            evaluation_results = self.evaluator.evaluate_imputation(clients)
            self.tracker.imp_quality.append((0, evaluation_results))
            tqdm.write(f"Evaluation: rmse - {evaluation_results['imp_rmse_avg']} ws - {evaluation_results['imp_ws_avg']}")

            ########################################################################################################
            # local training of imputation model
            local_models, clients_fit_res = [], []
            for client_idx in trange(len(clients), desc='Client idx', colour='blue', leave = False):
                client = clients[client_idx]
                model_parameter, fit_res = client.fit_local_imputation_model(params = {})
                local_models.append(model_parameter)
                clients_fit_res.append(fit_res)

            # aggregate local imputation model
            global_models, agg_res = agg_strategy.aggregate(
                local_model_parameters=local_models, fit_res=clients_fit_res
            )

            # updates local imputation model and do imputation
            for global_model, local_model, client in zip(global_models, local_models, clients):
                updated_local_model = agg_strategy.update_local_model(global_model, local_model, client)
                client.imputation(updated_local_model)

        ########################################################################################################
        # Evaluation
        evaluation_results = self.evaluator.evaluate_imputation(clients)
        self.tracker.imp_quality.append((iterations, evaluation_results))
        tqdm.write(
            f"Final Evaluation: rmse - {evaluation_results['imp_rmse_avg']} ws - {evaluation_results['imp_ws_avg']}")

        return ret
