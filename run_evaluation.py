import json
import warnings
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from config import settings, ROOT_DIR
import numpy as np
from src.loaders.load_environment import setup_clients, setup_server
from src.evaluation.evaluator2 import Evaluator
import timeit
import loguru
from emf.logging import setup_logger

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="config", config_name="evaluation_config")
def my_app(cfg: DictConfig) -> None:

    config_dict: dict = OmegaConf.to_container(cfg, resolve=True)

    ####################################################################################################################
    # Result directory
    scenario_version = config_dict['scenario_version']
    dataset_name = config_dict['dataset_name']
    data_partition = config_dict['data_partition_name']
    missing_scenario = config_dict['missing_scenario_name']
    scenario_name = data_partition + '_' + missing_scenario
    imputer = config_dict['imputer_name']
    fed_strategy = config_dict['fed_strategy_name']
    round_idx = config_dict['round_idx']
    eval_dir_name = config_dict['eval_dir_name']

    eval_ret_dir = os.path.join(
        ROOT_DIR, settings['result_dir']['base'], settings['result_dir']['raw'], eval_dir_name, scenario_version,
        dataset_name, scenario_name, imputer, fed_strategy, str(round_idx)
    )
    if not os.path.exists(eval_ret_dir):
        os.makedirs(eval_ret_dir)

    log_to_file = config_dict['log_to_file']
    log_level = config_dict['log_level']
    setup_logger(eval_ret_dir, to_file=log_to_file, level=log_level)

    ####################################################################################################################
    # Read scenario data
    scenario_version = config_dict['scenario_version']
    dataset_name = config_dict['dataset_name']
    scenario_dir_path = os.path.join(
        ROOT_DIR, settings['scenario_dir'], scenario_version, dataset_name, scenario_name, str(round_idx)
    )

    loguru.logger.debug(scenario_dir_path)

    clients_train_data = np.load(os.path.join(scenario_dir_path, 'clients_train_data.npz'))
    clients_test_data = np.load(os.path.join(scenario_dir_path, 'clients_test_data.npz'))
    clients_train_data_ms = np.load(os.path.join(scenario_dir_path, 'clients_train_data_ms.npz'))
    clients_data = []
    for client_id in clients_train_data.keys():
        clients_data.append(
            (clients_train_data[client_id], clients_test_data[client_id], clients_train_data_ms[client_id])
        )

    with open(os.path.join(scenario_dir_path, 'stats.json'), 'r') as f:
        stats_dict = json.load(f)

    seed = stats_dict['seed']
    client_seeds = stats_dict['client_seeds']
    data_config = stats_dict['data_config']

    loguru.logger.debug(data_config)

    ####################################################################################################################
    # Loader Environments (client)
    experiment_name = config_dict['experiment_name']
    ret_dir = os.path.join(
        ROOT_DIR, settings['result_dir']['base'], settings['result_dir']['raw'], experiment_name, scenario_version,
        dataset_name, scenario_name, imputer, fed_strategy, str(round_idx)
    )

    if not os.path.exists(ret_dir):
        raise FileNotFoundError(f"Results directory not found: {ret_dir}")

    # latest results
    dates = os.listdir(ret_dir)
    date = max([int(item) for item in dates])
    date = str(date)
    date = '0' + date if len(date) == 3 else date
    ret_dir = os.path.join(ret_dir, date)

    loguru.logger.debug(ret_dir)

    with open(os.path.join(ret_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    imputer_name = config['config']['imputer']['imp_name']
    imputer_params = config['config']['imputer']['imp_params']

    fed_strategy_name = config['config']['fed_strategy']['fed_strategy_name']
    fed_strategy_client_params = config['config']['fed_strategy']['fed_strategy_client_params']

    clients = setup_clients(
        clients_data, client_seeds, data_config, imputer_name, imputer_params, fed_strategy_name,
        fed_strategy_client_params, {'local_dir_path': ret_dir}
    )

    for client in clients:
        client.X_train_imp[client.X_train_mask] = 0
        client.imputer.initialize(client.X_train_imp, client.X_train_mask, client.data_utils, {}, client.seed)
        client.load_imp_model(version='final')

    ####################################################################################################################
    # Evaluation
    evaluation_params = config_dict['eval_params']
    evaluator = Evaluator(**evaluation_params)

    X_train_imps = [client.X_train_imp for client in clients]
    X_train_origins = [client.X_train for client in clients]
    X_train_masks = [client.X_train_mask for client in clients]
    y_trains = [client.y_train for client in clients]
    X_tests = [client.X_test for client in clients]
    y_tests = [client.y_test for client in clients]

    start = timeit.default_timer()
    ret = evaluator.run_evaluation(
        X_train_imps, X_train_origins, X_train_masks, y_trains, X_tests, y_tests, data_config
    )
    end = timeit.default_timer()

    ####################################################################################################################
    # Save to Result
    eval_ret = {}
    with open(os.path.join(eval_ret_dir, 'eval_results.json'), 'w') as f:
        eval_ret['execution_time'] = end - start
        eval_ret['results'] = ret
        json.dump(eval_ret, f)

    loguru.logger.info(ret['agg_stats'])
    loguru.logger.info(f"Time taken: {end - start}")


if __name__ == "__main__":
    my_app()
