from src.server import Server
from typing import List
from src.client import Client
from src.evaluation.evaluator import Evaluator

from src.imputation.initial_imputation import initial_imputation_num, initial_imputation_cat
from .utils import formulate_centralized_client
from .workflow import BaseWorkflow
from tqdm.auto import trange

from ..imputation.initial_imputation.initial_imputation import initial_imputation
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
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker, train_params: dict
    ) -> Tracker:

        """
        Imputation workflow for Joint Modeling Imputation
        """
        evaluation_interval = self.workflow_params['evaluation_interval']
        imp_interval = self.workflow_params['imp_interval']

        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))

        ############################################################################################################
        # Initial Imputation
        clients = initial_imputation(server.fed_strategy.strategy_params['initial_impute'], clients)

        self.eval_and_track(evaluator, tracker, clients, phase='initial')

        ############################################################################################################
        # Federated Imputation Workflow
        model_converge_tol = train_params['model_converge_tol']
        model_converge_patience = train_params['model_converge_patience']
        best_significant_loss_per_client = {client.client_id: None for client in clients}
        patience_counter_per_client = {client.client_id: 0 for client in clients}
        all_clients_converged = False

        global_model_epochs = train_params['global_epoch']
        for epoch in trange(global_model_epochs, desc='Global Epoch', colour='blue'):

            ########################################################################################################
            # Evaluation
            if epoch % evaluation_interval == 0 or epoch >= global_model_epochs - 3:
                self.eval_and_track(
                    evaluator, tracker, clients, phase='round', epoch=epoch, iterations=global_model_epochs,
                    evaluation_interval=evaluation_interval
                )

            ########################################################################################################
            # local training of imputation model
            local_models, clients_fit_res = [], []
            fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
            for client_idx in range(len(clients)):
                client = clients[client_idx]
                fit_params = train_params
                fit_params.update(fit_instruction[client_idx])
                model_parameter, fit_res = client.fit_local_imp_model(params=fit_params)
                local_models.append(model_parameter)
                clients_fit_res.append(fit_res)

            ########################################################################################################
            # check convergence
            for client, fit_res in zip(clients, clients_fit_res):
                current_loss = fit_res['loss']  # Assuming 'loss' is reported in fit_res

                if best_significant_loss_per_client[client.client_id] is None:
                    best_significant_loss_per_client[client.client_id] = current_loss
                    improvement = float('inf')  # Assume large improvement to initialize
                else:
                    # Calculate improvement
                    improvement = best_significant_loss_per_client[client.client_id] - current_loss

                # Update the best loss and reset patience if there's significant improvement
                if improvement > model_converge_tol:
                    best_significant_loss_per_client[client.client_id] = current_loss
                    patience_counter_per_client[client.client_id] = 0
                else:
                    patience_counter_per_client[client.client_id] += 1  # Increment patience counter

            if all(
                    patience_counter_per_client[client.client_id] >= model_converge_patience
                    for client in clients
            ):
                print("All clients have converged. Stopping training.")
                all_clients_converged = True
                break

            ########################################################################################################
            # aggregate local imputation model
            global_models, agg_res = server.fed_strategy.aggregate_parameters(
                local_model_parameters=local_models, fit_res=clients_fit_res
            )

            ########################################################################################################
            # updates local imputation model and do imputation
            for global_model, client in zip(global_models, clients):
                client.update_local_imp_model(global_model, params={})

            if epoch % imp_interval == 0:
                for client in clients:
                    client.local_imputation(params={})

        for client in clients:
            client.local_imputation(params={})

        if not all_clients_converged:
            print("Training completed without early stopping.")

        ########################################################################################################
        # Evaluation
        self.eval_and_track(evaluator, tracker, clients, phase='final', iterations=global_model_epochs)

        return tracker

    def fed_imp_parallel(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker, train_params: dict
    ) -> Tracker:
        return tracker
