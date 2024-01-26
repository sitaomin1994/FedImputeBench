import numpy as np

from src.loaders.load_agg_strategy import load_agg_strategy
from src.loaders.load_data import load_data
from src.modules.data_prep.utils import split_train_test
from src.utils.setup_seeds import setup_clients_seed

from src.loaders.load_data_partition import load_data_partition
from src.loaders.load_missing_simulation import add_missing
from src.loaders.load_imputer import load_imputer
from src.loaders.load_clients import load_client
from src.loaders.load_server import load_server


class Experiment:

    def __init__(self, name='experiment'):
        self.name = name

    def run(self, config):
        num_clients = config['num_clients']
        seed = config['seed']
        num_rounds = config["experiment"]['num_rounds']

        ##########################################################################################
        # setup seeds for each client
        client_seeds = setup_clients_seed(num_clients, seed=seed)

        ##########################################################################################
        # load data and split train/test
        dataset_name = config["data"]['dataset_name']
        test_size = config["data"]['test_size']
        data, data_config = load_data(dataset_name)
        train_data, test_data = split_train_test(
            data, data_config, test_size=test_size, seed=seed, output_format='dataframe_merge'
        )

        ##########################################################################################
        # data partition
        data_partition_strategy = config["data"]['data_partition']
        clients_train_data_list = load_data_partition(
            data_partition_strategy, train_data.values, num_clients, {}, seed=seed
        )

        ##########################################################################################
        # simulate missing data
        ms_simulate_strategy = config["missing_simulate"]
        cols = np.arange(data.shape[1] - 1)
        client_train_data_ms_list = add_missing(clients_train_data_list, ms_simulate_strategy, cols, seeds=client_seeds)

        ##########################################################################################
        # setup imputation models
        imp_model_name = config['imp_model']
        imp_model_params = config['imp_model_params']
        imputers = [
            load_imputer(imp_model_name, imp_model_params) for _ in range(num_clients)
        ]

        ##########################################################################################
        # setup clients
        client_type = config['client']['client_type']
        client_config = config['client']['client_config']
        clients = []
        for i in range(num_clients):
            clients.append(
                load_client(
                    client_type = client_type, client_id=i, train_data=clients_train_data_list[i], test_data=test_data,
                    X_train_ms=client_train_data_ms_list[i][:, :-1], data_config=data_config, imp_model=imputers[i],
                    client_config=client_config.copy(), seed=client_seeds[i]
                )
            )

        ##########################################################################################
        # setup federated aggregation strategy
        agg_strategy = config['agg_strategy']['agg_strategy_name']
        agg_strategy_params = config['agg_strategy']['agg_strategy_params']
        agg_strategy = load_agg_strategy(agg_strategy, agg_strategy_params)

        ##########################################################################################
        # setup federated workflow and aggregation strategy
        server = config['server']['server_type']
        server_config = config['server']['server_config']
        server = load_server(server, server_config)
        imp_workflow_params = config['server']['server_imp_workflow_params']

        server.run_fed_imputation(clients, agg_strategy, imp_workflow_params)

        ##########################################################################################
        # evaluation
        # rets = {}
        # for client in clients:
        #     client.local_evaluate({})
        #     rets[client.client_id] = client.eval_ret

        # return {
        #     'rets': rets,
        #     'history': history,
        #     'eval_history': eval_history,
        # }
