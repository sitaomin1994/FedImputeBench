from loguru import logger
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="exp_config_imp_amarel2")
def main(cfg: DictConfig) -> None:
    #print(OmegaConf.to_yaml(cfg))
    # debug = cfg.debug
    # # debug mode
    # if debug:
    #     logger.remove(0)
    #     logger.add(sys.stderr, level="DEBUG")
    # else:
    #     logger.remove(0)
    #     logger.add(sys.stderr, level="INFO")
    #
    # # initialize fed_imp manager and fed_imp class
    # experiment_manager = ExperimentManager()
    # experiment_manager.set_experiment(Experiment)
    #
    # # load configuration files and fed_imp type
    # configs, exp_meta = load_configs_raw(OmegaConf.to_container(cfg, resolve=True))
    #
    # # run experiments and persist results
    # experiment_manager.run_experiments(configs, exp_meta, use_db=False, debug=debug)
    return 0

if __name__ == '__main__':
    main()
