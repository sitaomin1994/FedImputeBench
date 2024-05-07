import os
import json
import numpy as np
import hydra


@hydra.main(version_base=None, config_path="config", config_name="experiment_config")
def main(seed=10):
    rng = np.random.default_rng(seed=seed)

    a = 0
    while a <= 10:
        ret = rng.dirichlet([0.1, 0.1, 0.1])
        a = a + 1
        print(ret)


if __name__ == '__main__':
    main(10)

