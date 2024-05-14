import numpy as np
from emf.base_experiment import BaseExperiment
from src.evaluation.evaluator import Evaluator
from src.loaders.load_environment import setup_clients, setup_server
from src.utils.tracker import Tracker
from src.utils.result_analyzer import ResultAnalyzer
from src.utils.consistency_checker import check_consistency
from src.loaders.load_data import load_data
from src.server import Server
from src.client import Client
from src.loaders.load_workflow import load_workflow
from src.loaders.load_scenario import simulate_scenario


class Experiment(BaseExperiment):

    def __init__(self):
        self.exp_name = 'exp'
        self.exp_type = 'federated_imputation'
        super().__init__(self.exp_name, self.exp_type)

    def run(self, config: dict, experiment_meta: dict) -> dict:

        n_rounds = config['experiment']['n_rounds']
        seed = config['experiment']['seed']
        mtp = config['experiment']['mtp']
        if n_rounds == 1:
            return self.single_run(config, seed)
        else:
            results = self.multiple_runs(config, seed, n_rounds, mtp=mtp)
            return self.merge_results(results)

    def single_run(self, config: dict, seed) -> dict:
        """
        Run the single round experiment
        :param config: configuration of the experiment
        :return: results dictionary
        """
        print("aaa", seed)

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

        # check the consistency of imputer, fed_strategy and workflow
        check_consistency(imputer_name, fed_strategy_name, workflow_name)

        clients = setup_clients(
            clients_data, client_seeds, data_config,
            imputer_name, imputer_params, fed_strategy_name, fed_strategy_client_params
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

    @staticmethod
    def setup_workflow(
            clients_train_data_list, client_train_data_ms_list, client_seeds,
            imp_model_name, imp_model_params,
            agg_strategy_name, agg_strategy_params,
            workflow_name, workflow_params,
            test_data, data_config,
    ):

        clients = []
        for i in range(len(clients_train_data_list)):
            clients.append(
                Client(
                    client_id=i,
                    train_data=clients_train_data_list[i],
                    test_data=test_data,
                    X_train_ms=client_train_data_ms_list[i][:, :-1],
                    data_config=data_config,
                    imp_model_name=imp_model_name,
                    imp_model_params=imp_model_params,
                    client_config={},
                    seed=client_seeds[i]
                )
            )

        # ========================================================================================
        server = Server(agg_strategy_name, agg_strategy_params, server_config={})

        # ========================================================================================
        workflow = load_workflow(workflow_name, workflow_params)

        return clients, server, workflow
