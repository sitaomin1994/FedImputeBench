import json
from copy import deepcopy

import numpy as np

from emf.base_experiment import BaseExperiment
from FedImpute.evaluation.evaluator import Evaluator
from FedImpute.loaders.load_environment import setup_clients, setup_server
from FedImpute.utils.tracker import Tracker
from FedImpute.utils.result_analyzer import ResultAnalyzer
from FedImpute.utils.consistency_checker import check_consistency
from FedImpute.loaders.load_data import load_data
from FedImpute.loaders.load_workflow import load_workflow
from FedImpute.loaders.load_scenario import simulate_scenario
from config import settings
import os
from FedImpute.utils.data_persistence import deep_update


class Experiment(BaseExperiment):

    def __init__(self, scenario_setup: bool = False):
        self.exp_name = 'exp'
        self.exp_type = 'federated_imputation'
        self.scenario_setup = scenario_setup
        super().__init__(self.exp_name, self.exp_type)

    def run(self, config: dict, experiment_meta: dict) -> dict:

        ###############################################################################
        # Scenario already setup just read data and run federated imputation
        if self.scenario_setup:
            # setup scenario
            # hyper parameter settings
            if 'hyper_tune' in experiment_meta and experiment_meta['hyper_tune']:
                core_param_dict = {
                    'miwae': {
                        'lr': ['imputer', 'model_train_params', 'learning_rate'],
                        'le': ['imputer', 'model_train_params', 'local_epoch'],
                        'opt': ['imputer', 'model_train_params', 'optimizer'],
                    },
                    'gain': {
                        'lr': ['imputer', 'model_train_params', 'learning_rate'],
                        'le': ['imputer', 'model_train_params', 'local_epoch'],
                        'opt': ['imputer', 'model_train_params', 'optimizer'],
                    },
                }

                imputer = config['imputer']['imp_name']
                if imputer in core_param_dict:
                    hypers = []
                    for key, value in core_param_dict[imputer].items():
                        hyper_name = key
                        config_dict = config
                        for v in value:
                            print(v, config_dict)
                            config_dict = config_dict[v]
                        hyper_value = str(config_dict)
                        hyper_item = '='.join([hyper_name, hyper_value])
                        hypers.append(hyper_item)
                    hyper_str = '_'.join(hypers)
                    experiment_meta['output_path'] = experiment_meta['output_path'] + '/' + hyper_str
                    print(experiment_meta['output_path'])
                else:
                    raise ValueError(f"Imputer {imputer} is not supported for hyper-parameter tuning")
            return self.single_run_scenario(config)
        ###############################################################################
        # Experiment Setup from laoding data to running federated imputation
        else:
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

    def single_run_scenario(self, config: dict) -> dict:
        """
        Run the single round experiment
        :param config: configuration of the experiment
        :return: results dictionary
        """

        ###########################################################################################################
        # Parameters
        data_partition_name = config['data_partition_name']
        missing_data_scenario = config['missing_scenario_name']
        round_id = config['round_id']
        dataset_name = config['dataset_name']
        scenario_version = config['scenario_version']

        ###########################################################################################################
        # Load scenario setting
        scenario_data_dir = settings['scenario_dir']
        scenario_name = f"{data_partition_name}_{missing_data_scenario}"
        scenario_dir_path = os.path.join(
            scenario_data_dir, scenario_version, dataset_name, scenario_name, str(round_id)
        )

        with open(os.path.join(scenario_dir_path, 'stats.json'), 'r') as f:
            stats_dict = json.load(f)

        seed = stats_dict['seed']
        client_seeds = stats_dict['client_seeds']
        data_config = stats_dict['data_config']

        ##########################################################################################################
        # Load clients data
        clients_data = []
        clients_train_data = np.load(os.path.join(scenario_dir_path, 'clients_train_data.npz'))
        clients_test_data = np.load(os.path.join(scenario_dir_path, 'clients_test_data.npz'))
        clients_train_data_ms = np.load(os.path.join(scenario_dir_path, 'clients_train_data_ms.npz'))
        for client_id in clients_train_data.keys():
            clients_data.append(
                (clients_train_data[client_id], clients_test_data[client_id], clients_train_data_ms[client_id])
            )

        ###########################################################################################################
        # Setup Clients and Server
        use_default_hyper_params = config['use_default_hyper_params']
        # Imputers
        imputer_name = config['imputer']['imp_name']
        imputer_params = config['imputer']['imp_params']
        imp_model_train_params = config['imputer']['model_train_params']
        if 'hyper_params' in config['imputer'] and use_default_hyper_params:
            imputer_hyper_params = config['imputer']['hyper_params'][dataset_name]
            imputer_params = deep_update(imputer_params, imputer_hyper_params['imp_params'])
            imp_model_train_params = deep_update(
                imp_model_train_params, imputer_hyper_params['model_train_params']
            )

        # federated strategy
        fed_strategy_name = config['fed_strategy']['fed_strategy_name']
        fed_strategy_client_params = config['fed_strategy']['fed_strategy_client_params']
        fed_strategy_server_params = config['fed_strategy']['fed_strategy_server_params']
        if 'hyper_params' in config['fed_strategy'] and use_default_hyper_params:
            fed_strategy_hyper_params = config['fed_strategy']['hyper_params'][dataset_name]
            if 'fed_strategy_client_params' in fed_strategy_hyper_params:
                fed_strategy_client_params = deep_update(
                    fed_strategy_client_params,fed_strategy_hyper_params['fed_strategy_client_params']
                )
            if 'fed_strategy_server_params' in fed_strategy_hyper_params:
                fed_strategy_server_params = deep_update(
                    fed_strategy_server_params,fed_strategy_hyper_params['fed_strategy_server_params']
             )

        workflow_name = config['imp_workflow']['workflow_name']
        workflow_params = config['imp_workflow']['workflow_params']
        workflow_run_type = config['imp_workflow']['workflow_runtype']

        client_config = config['client_config']
        client_config['local_dir_path'] = self.get_experiment_out_dir(config['experiment'])

        # check the consistency of imputer, fed_strategy and workflow
        check_consistency(imputer_name, fed_strategy_name, workflow_name)

        clients = setup_clients(
            clients_data, client_seeds, data_config, imputer_name, imputer_params, fed_strategy_name,
            fed_strategy_client_params, client_config
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
            clients, server, evaluator, tracker, workflow_run_type, deepcopy(imp_model_train_params)
        )

        ret = result_analyzer.clean_and_analyze_results(tracker)  # TODO: result analyzer

        return ret
