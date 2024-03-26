import numpy as np
from emf.base_experiment import BaseExperiment
from src.evaluation.evaluator import Evaluator
from src.utils.tracker import Tracker
from src.utils.result_analyzer import ResultAnalyzer
from src.loaders.load_data import load_data
from src.utils.setup_seeds import setup_clients_seed

from src.loaders.load_data_partition import load_data_partition
from src.loaders.load_missing_simulation import add_missing
from src.server import Server
from src.client import Client
from src.loaders.load_workflow import load_workflow


class Experiment(BaseExperiment):

    def __init__(self):
        self.exp_name = 'exp'
        self.exp_type = 'federated_imputation'
        super().__init__(self.exp_name, self.exp_type)

    def run(self, config: dict) -> dict:
        """
        Run the single round experiment
        :param config: configuration of the experiment
        :return: results dictionary
        """

        seed = config['experiment']['seed']

        ###########################################################################################################
        # Data loading
        dataset_name = config["data"]['dataset_name']
        test_size = config["data"]['test_size']
        train_data, test_data, data_config = load_data(dataset_name, test_size, seed)

        ###########################################################################################################
        # Scenario setup
        num_clients = config['num_clients']
        data_partition_strategy = config['data_partition']
        ms_simulate_strategy = config["missing_simulate"]
        clients_train_data_list, client_train_data_ms_list, client_seeds = self.setup_scenario(
            train_data, data_partition_strategy, ms_simulate_strategy, num_clients, seed
        )

        ###########################################################################################################
        # Setup Clients and Server (imputation models and agg strategy)
        imp_model_name = config['imputer']['imp_name']
        imp_model_params = config['imputer']['imp_params']

        agg_strategy_name = config['agg_strategy']['agg_strategy_name']
        agg_strategy_params = config['agg_strategy']['agg_strategy_params']

        workflow_name = config['server']['server_imp_workflow']  # todo
        workflow_params = config['server']['server_imp_workflow_params']  # todo
        workflow_run_type = config['server']['server_imp_workflow_run_type']  # todo

        clients, server, workflow = self.setup_workflow(
            clients_train_data_list, client_train_data_ms_list, client_seeds,
            imp_model_name, imp_model_params,
            agg_strategy_name, agg_strategy_params,
            workflow_name, workflow_params,
            test_data, data_config,
        )

        ###########################################################################################################
        # Evaluator and Tracker
        evaluator_params = config['evaluator_params']
        tracker_params = config['tracker_params']
        ret_analyze_params = config['ret_analyze_params']
        evaluator = Evaluator(evaluator_params)  # initialize evaluator
        tracker = Tracker(tracker_params)  # initialize tracker
        result_analyzer = ResultAnalyzer(ret_analyze_params)  # initialize result analyzer

        ###########################################################################################################
        # Run Federated Imputation
        tracker = workflow.run_fed_imp(clients, server, evaluator, tracker, workflow_run_type)
        ret = result_analyzer.clean_and_analyze_results(tracker)  # TODO: result analyzer

        return ret

    def multiple_rounds_run(self, config: dict) -> dict:  # todo
        """
        Run multiple rounds of the experiment
        :param config: configuration files
        :return: multiple runs results dictionary
        """
        num_rounds = config["experiment"]['num_rounds']
        return {}

    @staticmethod
    def setup_scenario(
            train_data, data_partition_strategy, ms_simulate_strategy, num_clients, seed
    ):
        # ========================================================================================
        # setup clients seeds
        client_seeds = setup_clients_seed(num_clients, seed=seed)

        # ========================================================================================
        # data partition
        clients_train_data_list = load_data_partition(
            data_partition_strategy, train_data, num_clients, {}, seed=seed
        )

        # ========================================================================================
        # simulate missing data
        cols = np.arange(train_data.shape[1] - 1)
        client_train_data_ms_list = add_missing(
            clients_train_data_list, ms_simulate_strategy, cols, seeds=client_seeds
        )

        return clients_train_data_list, client_train_data_ms_list, client_seeds

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
