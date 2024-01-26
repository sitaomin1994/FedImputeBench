from .server import Server
from typing import Dict, Union, List, Tuple
from src.client import Client, JMClient
from src.agg_strategy import Strategy


class ServerJM(Server):

    def __init__(self):
        super().__init__()

    def run_fed_imputation(
            self, clients: List[JMClient], agg_strategy: Strategy, workflow_params: dict
    ) -> dict:
        """
        Imputation workflow for MICE
        """
        ############################################################################################################
        # Workflow Parameters
        iterations = workflow_params['iterations']

        ############################################################################################################
        # Federated Imputation Workflow
        ret = {}
        for epoch in range(iterations):

            ########################################################################################################

            # local training of imputation model
            local_models, clients_fit_res = [], []
            for client in clients:
                model_parameter, fit_res = client.fit_local_imputation_model()
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
        # evaluation
        for client in clients:
            pass  # todo: evaluation

        return ret
