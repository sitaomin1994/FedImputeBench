import loguru
from loguru import logger
import sys
import warnings
import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
from omegaconf import DictConfig, OmegaConf
import hydra
from src.exp_manager import FedMissExpManager


@hydra.main(version_base=None, config_path="config", config_name="imp_config_p")
def my_app(cfg: DictConfig) -> None:

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    exp_manager = FedMissExpManager()
    exp_manager.execute_experiment(
        'federated_imputation_scenario', config_dict, experiment_meta=config_dict['experiment']
    )


if __name__ == "__main__":
    try:
        my_app()
    except Exception as e:
        logger.error(e)