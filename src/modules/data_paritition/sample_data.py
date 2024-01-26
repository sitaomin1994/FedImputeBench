from sklearn.model_selection import train_test_split
import numpy as np
import random
from typing import List


def generate_samples_iid(data, sample_fracs, seed, regression=False) -> List[np.ndarray]:
    ret = []
    for idx, sample_frac in enumerate(sample_fracs):
        new_seed = seed + idx * seed + idx*990983
        if sample_frac == 1.0:
            ret.append(data.copy())
        else:
            # new_seed = seed
            if regression:
                _, X_test, _, y_test = train_test_split(
                    data[:, :-1], data[:, -1], test_size=sample_frac,
                    random_state=(new_seed) % (2 ** 32)
                )
            else:
                _, X_test, _, y_test = train_test_split(
                    data[:, :-1], data[:, -1], test_size=sample_frac,
                    random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
                )
            ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())

    return ret