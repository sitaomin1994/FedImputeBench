import json
import numpy as np

from emf.base_experiment import BaseExperiment
from src.evaluation.evaluator import Evaluator
from src.loaders.load_environment import setup_clients, setup_server
from src.utils.tracker import Tracker
from src.utils.result_analyzer import ResultAnalyzer
from src.utils.consistency_checker import check_consistency
from src.loaders.load_data import load_data
from src.loaders.load_workflow import load_workflow
from src.loaders.load_scenario import simulate_scenario
from config import settings
import os


class Experiment(BaseExperiment):

    def __init__(self, mode: str):
        self.exp_name = 'exp'
        self.exp_type = 'federated_prediction'
        assert mode in ['origin', 'local', 'fed'], f"Invalid mode: {mode}"
        self.mode = mode

        super().__init__(self.exp_name, self.exp_type)

    def run(self, config: dict, experiment_meta: dict) -> dict:

        ###############################################################################
        # Experiment Setup from laoding data to running federated prediction
        n_rounds = config['experiment']['n_rounds']
        seed = config['experiment']['seed']
        mtp = config['experiment']['mtp']
        if n_rounds == 1:
            return self.single_run(config, seed)
        else:
            results = self.multiple_runs(config, seed, n_rounds, mtp=mtp)
            return {'results': results, 'meta': experiment_meta}

    def single_run(self, config: dict, seed) -> dict:
        """
        Run the single round experiment
        :param seed: seed for running experiment
        :param config: configuration of the experiment
        :return: results dictionary
        """

        ###########################################################################################################
        # Experiment configuration
        # - model and params - linear, neural network
        # - strategy and params - local, fedavg, fedprox, fedavg_ft
        # - data - origin, imputed
        # - scenario - dataset, dp, missing, imp, fed, round
        # - prediction workflow params
        # - evaluator params
        # - result tracker
        #

        ###########################################################################################################
        # Data loading
        dataset_name = config['dataset_name']
        data, data_config = load_data(dataset_name)

        ###########################################################################################################
        # Scenario setup
        num_clients = config['num_clients']
        data_partition_params = config['data_partition']['partition_params']
        missing_scenario_params = config['missing_scenario']['params']
        clients_data, global_test_data, client_seeds, stata = simulate_scenario(
            data.values, data_config, num_clients, data_partition_params, missing_scenario_params, seed
        )

        ###########################################################################################################
        # Setup Clients and Server (imputation models and agg strategy)
        imputer_name = config['imputer']['imp_name']
        imputer_params = config['imputer']['imp_params']
        imp_model_train_params = config['imputer']['model_train_params']

        fed_strategy_name = config['fed_strategy']['fed_strategy_name']
        fed_strategy_client_params = config['fed_strategy']['fed_strategy_client_params']
        fed_strategy_server_params = config['fed_strategy']['fed_strategy_server_params']

        workflow_name = config['imp_workflow']['workflow_name']
        workflow_params = config['imp_workflow']['workflow_params']
        workflow_run_type = config['imp_workflow']['workflow_runtype']

        client_config = config['client_config']
        client_config['local_dir_path'] = self.get_experiment_out_dir(config['experiment'])

        # check the consistency of imputer, fed_strategy and workflow
        check_consistency(imputer_name, fed_strategy_name, workflow_name)

        clients = setup_clients(
            clients_data, client_seeds, data_config, imputer_name, imputer_params,
            fed_strategy_name, fed_strategy_client_params, client_config
        )

        server = setup_server(
            fed_strategy_name, fed_strategy_server_params, server_config={}
        )

        workflow = load_workflow(workflow_name, workflow_params)

        ###########################################################################################################
        # Evaluator and Tracker
        evaluator_params = config['evaluator_params']
        tracker_params = config['tracker']['params']
        ret_analyze_params = config['ret_analyze_params']
        evaluator = Evaluator(evaluator_params)  # initialize evaluator
        tracker = Tracker(tracker_params)  # initialize tracker
        result_analyzer = ResultAnalyzer(ret_analyze_params)  # initialize result analyzer

        ###########################################################################################################
        # Run Federated Imputation
        tracker = workflow.run_fed_imp(
            clients, server, evaluator, tracker, workflow_run_type, imp_model_train_params
        )

        ret = result_analyzer.clean_and_analyze_results(tracker)  # TODO: result analyzer

        return ret


