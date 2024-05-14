from .utils import formulate_centralized_client, update_clip_threshold
from .workflow import BaseWorkflow
from src.server import Server
from typing import List
from src.client import Client
from ..evaluation.evaluator import Evaluator

from tqdm.auto import trange
from src.utils.tracker import Tracker
from ..imputation.initial_imputation.initial_imputation import initial_imputation


class WorkflowEM(BaseWorkflow):

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
        evaluation_interval = train_params['evaluation_interval']
        max_iterations = train_params['max_iterations']

        ############################################################################################################
        # Centralized Initialization
        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))


        ############################################################################################################
        # Initial Imputation and evaluation
        clients = initial_imputation(server.fed_strategy.strategy_params['initial_impute'], clients)
        self.eval_and_track(evaluator, tracker, clients, phase='initial')
        # Update Global clip thresholds
        if server.fed_strategy.name != 'local':
            update_clip_threshold(clients)

        ########################################################################################################
        # federated EM imputation
        clients_converged_signs = [False for _ in range(len(clients))]
        for iteration in trange(max_iterations, desc='Iterations', leave=False, colour='blue'):

            #####################################################################################################
            # client local train imputation model
            local_models, clients_fit_res = [], []
            fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
            for client in clients:
                fit_params = train_params
                fit_params.update(fit_instruction[client.client_id])
                model_parameter, fit_res = client.fit_local_imp_model(params=fit_params)
                local_models.append(model_parameter)
                clients_fit_res.append(fit_res)
                clients_converged_signs[client.client_id] = fit_res['converged']  # convergence sign

            # all converged
            if all(clients_converged_signs):
                print(f"All clients converged, iteration {iteration}")
                break

            # server aggregate local imputation models
            global_models, agg_res = server.fed_strategy.aggregate_parameters(
                local_model_parameters=local_models, fit_res=clients_fit_res, params={}
            )

            for global_model, client in zip(global_models, clients):
                client.update_local_imp_model(global_model, params={})
                client.local_imputation(params={})

            ########################################################################################################
            # Impute and Evaluation
            if iteration % evaluation_interval == 0:
                self.eval_and_track(
                    evaluator, tracker, clients, phase='round', epoch=iteration, iterations=max_iterations,
                    evaluation_interval=evaluation_interval
                )

        ########################################################################################################
        # Final Evaluation and Tracking
        self.eval_and_track(evaluator, tracker, clients, phase='final', iterations=max_iterations)

        return tracker

    def fed_imp_parallel(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker, train_params: dict
    ) -> Tracker:
        pass
