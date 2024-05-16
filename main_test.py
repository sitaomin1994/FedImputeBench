import os
import json
import numpy as np
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="test_config")
def main(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    print("job id: {} gpu available: {} gpu: {} seed: {}".format(
        HydraConfig.get().job.num, torch.cuda.is_available(), torch.cuda.get_device_name(), config_dict['seed']
    ))


if __name__ == '__main__':
    main()

