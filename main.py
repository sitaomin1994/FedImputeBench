from loguru import logger
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from omegaconf import DictConfig, OmegaConf
import hydra

from src.exp_manager import FedMissExpManager


@hydra.main(version_base=None, config_path="config", config_name="experiment_config_imp_pc2")
def my_app(cfg: DictConfig) -> None:
    print(cfg.experiment.output_path)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    print(config_dict)

    exp_manager = FedMissExpManager()
    exp_manager.execute_experiment(
        'federated_imputation', config_dict, experiment_meta = config_dict['experiment']
    )


if __name__ == "__main__":
    my_app()
