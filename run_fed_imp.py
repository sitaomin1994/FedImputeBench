from loguru import logger
import sys
import warnings
import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from src.exp_manager import FedMissExpManager


@hydra.main(version_base=None, config_path="config", config_name="imp_config")
def my_app(cfg: DictConfig) -> None:
    print(cfg.experiment.output_path)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    print(config_dict)

    exp_manager = FedMissExpManager()
    exp_manager.execute_experiment(
        'federated_imputation', config_dict, experiment_meta=config_dict['experiment']
    )

if __name__ == "__main__":

    my_app()
