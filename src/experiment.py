from src.loaders.load_data import load_data
from src.modules.data_prep.utils import split_train_test
from src.loaders.load_data_partition import load_data_partition
from src.loaders.load_missing_simulation import add_missing
from src.loaders.load_imputer import load_imputation_models
from src.loaders.load_clients import setup_clients
from src.loaders.load_server import load_server
from src.modules.setup_seeds import setup_clients_seed
import numpy as np


class Experiment:

    def __init__(self, name='experiment'):
        self.name = name

    def run(self, config):
        num_clients = config['num_clients']
        num_rounds = config['num_rounds']
        seed = config['seed']

        ##########################################################################################
        # setup seeds for each client
        client_seeds = setup_clients_seed(num_clients, seed=seed)

        ##########################################################################################
        # load data and split train/test
        dataset_name = config['dataset_name']
        test_size = config['test_size']
        data, data_config = load_data(dataset_name)
        train_data, test_data = split_train_test(
            data, data_config, test_size=test_size, seed=seed, output_format='dataframe_merge'
        )

        ##########################################################################################
        # data partition
        data_partition_strategy = config['data_partition']
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
        imp_models = load_imputation_models(client_train_data_ms_list, imp_model_name, imp_model_params, client_seeds)

        ##########################################################################################
        # setup clients
        clients = setup_clients(
            clients_train_data_list, client_train_data_ms_list, test_data.values, data_config, imp_models, client_seeds
        )

        ##########################################################################################
        # setup federated workflow and aggregation strategy
        server_config = config['server_config']
        server = config['server']
        server = load_server(server, clients, server_config)
        imp_params = config['imp_params']
        history, eval_history, clients = server.run_imputation(imp_params)

        ##########################################################################################
        # evaluation
        rets = {}
        for client in clients:
            client.local_evaluate({})
            rets[client.client_id] = client.eval_ret

        return {
            'rets': rets,
            'history': history,
            'eval_history': eval_history,
        }
