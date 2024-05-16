import delu
import numpy as np
from tqdm import tqdm

import src.utils.nn_utils as nn_utils

from src.server import Server
from typing import List
from src.client import Client
from src.evaluation.evaluator import Evaluator
from .utils import formulate_centralized_client, update_clip_threshold
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

        ############################################################################################################
        # Initial Imputation and Evaluation
        if server.fed_strategy.name == 'central':
            clients.append(formulate_centralized_client(clients))

        #clients = initial_imputation(server.fed_strategy.strategy_params['initial_impute'], clients)
        if train_params['initial_zero_impute']:
            clients = initial_imputation('zero', clients)
        else:
            clients = initial_imputation(server.fed_strategy.strategy_params['initial_impute'], clients)
        if server.fed_strategy.name != 'local':
            update_clip_threshold(clients)

        self.eval_and_track(
            evaluator, tracker, clients, phase='initial', central_client=server.fed_strategy.name == 'central'
        )

        ############################################################################################################
        # Federated Imputation Workflow
        use_early_stopping_global = train_params['use_early_stopping']
        model_converge_tol = train_params['model_converge']['tolerance']
        model_converge_patience = train_params['model_converge']['patience']
        model_converge_window_size = train_params['model_converge']['window_size']
        model_converge_steps = train_params['model_converge']['check_steps']
        model_converge_back_steps = train_params['model_converge']['back_steps']

        early_stoppings = [
            nn_utils.EarlyStopping(
                patience=model_converge_patience, tolerance=model_converge_tol, window_size=model_converge_window_size,
                check_steps=model_converge_steps, backward_window_size=model_converge_back_steps
            ) for _ in clients
        ]
        all_clients_converged_sign = [False for _ in clients]

        ################################################################################################################
        # Federated Training
        global_model_epochs = train_params['global_epoch']
        log_interval = train_params['log_interval']
        imp_interval = train_params['imp_interval'] if 'imp_interval' in train_params else 1e8

        fit_params_list = [train_params.copy() for _ in clients]

        for epoch in trange(global_model_epochs, desc='Global Epoch', colour='blue'):

            ###########################################################################################
            # Local training of an imputation model
            local_models, clients_fit_res = [], []

            fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
            for client_idx in range(len(clients)):
                client = clients[client_idx]
                fit_params = fit_params_list[client_idx]
                fit_params.update(fit_instruction[client_idx])
                # if it is converged, do not fit the model
                if all_clients_converged_sign[client_idx]:
                    fit_params.update({'fit_model': False})
                model_parameter, fit_res = client.fit_local_imp_model(params=fit_params)
                local_models.append(model_parameter)
                clients_fit_res.append(fit_res)

            #############################################################################################
            # Early Stopping
            if epoch % log_interval == 0:
                self.logging_loss(clients_fit_res)
            if use_early_stopping_global:
                self.early_stopping_step(clients_fit_res, early_stoppings, all_clients_converged_sign)
                if all(all_clients_converged_sign):
                    print("All clients have converged. Stopping training at {}.".format(epoch))
                    break

            ############################################################################################
            # Aggregate local imputation model
            global_models, agg_res = server.fed_strategy.aggregate_parameters(
                local_model_parameters=local_models, fit_res=clients_fit_res, params={
                    'current_epoch': epoch, 'global_epoch': global_model_epochs
                }
            )

            ###########################################################################################
            # Updates local imputation model and do imputation
            for client_idx, (global_model, client) in enumerate(zip(global_models, clients)):
                if not all_clients_converged_sign[client_idx]:
                    client.update_local_imp_model(global_model, params={})

            ###########################################################################################
            # if epoch == 0:
            #     for client in clients:
            #         client.local_imputation(params={})

            if epoch > 0 and epoch % imp_interval == 0:
                for client in clients:
                    client.local_imputation(params={})
                self.eval_and_track(
                    evaluator, tracker, clients, phase='round', epoch=epoch,
                    central_client=server.fed_strategy.name == 'central'
                )

        ################################################################################################################
        print("start fine tuning ...")
        ################################################################################################################
        # local training of an imputation model
        early_stoppings = [
            nn_utils.EarlyStopping(
                patience=model_converge_patience, tolerance=model_converge_tol, window_size=model_converge_window_size,
                check_steps=model_converge_steps, backward_window_size=model_converge_back_steps
            ) for _ in clients
        ]
        all_clients_converged_sign = [False for _ in clients]

        fine_tune_epochs = server.fed_strategy.fine_tune_epochs
        for epoch in trange(fine_tune_epochs, desc='Fine Tuning Epoch', colour='blue'):

            clients_fit_res = []
            fit_instruction = server.fed_strategy.fit_instruction([{} for _ in range(len(clients))])
            for client_idx in range(len(clients)):
                client = clients[client_idx]
                fit_params = train_params.copy()
                fit_params.update(fit_instruction[client_idx])
                fit_params.update({'freeze_encoder': False})
                # if it is converged, do not fit the model
                if all_clients_converged_sign[client_idx]:
                    fit_params.update({'fit_model': False})
                _, fit_res = client.fit_local_imp_model(params=fit_params)
                clients_fit_res.append(fit_res)

            ####################################################################################################
            # Early Stopping and Logging and Evaluation
            if epoch % log_interval == 0:
                self.logging_loss(clients_fit_res)

            if epoch > 0 and epoch % imp_interval == 0:
                for client in clients:
                    client.local_imputation(params={})
                self.eval_and_track(
                    evaluator, tracker, clients, phase='round', epoch=epoch + global_model_epochs,
                    central_client=server.fed_strategy.name == 'central'
                )

            if use_early_stopping_global:
                self.early_stopping_step(clients_fit_res, early_stoppings, all_clients_converged_sign)
                if all(all_clients_converged_sign):
                    print("All clients have converged. Stopping training at {}.".format(epoch))
                    break

        #########################################################################################################
        # Final imputation and Evaluation
        for client in clients:
            client.local_imputation(params={})

        self.eval_and_track(
            evaluator, tracker, clients, phase='final', central_client=server.fed_strategy.name == 'central'
        )

        return tracker

    def fed_imp_parallel(
            self, clients: List[Client], server: Server, evaluator: Evaluator, tracker: Tracker, train_params: dict
    ) -> Tracker:
        return tracker

    @staticmethod
    def early_stopping_step(clients_fit_res: List, early_stoppings: List, all_clients_converged_sign: List):

        for idx, (client_fit_res, early_stopping) in enumerate(zip(clients_fit_res, early_stoppings)):
            if 'loss' not in client_fit_res:
                continue
            early_stopping.update(client_fit_res['loss'])
            if early_stopping.check_convergence():
                all_clients_converged_sign[idx] = True

    @staticmethod
    def logging_loss(clients_fit_res: List):
        losses = np.array([client_fit_res['loss'] for client_fit_res in clients_fit_res if 'loss' in client_fit_res])
        if len(losses) == 0:
            tqdm.write("\nLoss: {:.2f} ({:2f})".format(0, 0))
        else:
            tqdm.write("\nLoss: {:.4f} ({:4f})".format(losses.mean(), losses.std()))
