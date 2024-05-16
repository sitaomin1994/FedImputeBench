from typing import List

import numpy as np


class Tracker:
    """
    Tracker class to track the imputation results along iterations
    tracker_params: {
        "track_data": bool,          # whether to track imputed data along iterations
        "track_model_params": bool,  # whether to track imputation model parameters along iterations
        "track_misc": bool,          # whether to track other parameters along iterations
        "persist": str - one of {'none', 'final', 'all'}  # options persist imputed data and model parameters
    }
    """

    def __init__(self, tracker_params: dict):

        # options
        if 'track_data' not in tracker_params:
            self.track_data = False
        else:
            assert isinstance(tracker_params['track_data'], bool), "track_data is not a boolean"
            self.track_data = tracker_params['track_data']

        if 'track_model_params' not in tracker_params:
            self.track_model_params = False
        else:
            assert isinstance(tracker_params['track_model_params'], bool), "track_model_params is not a boolean"
            self.track_model_params = tracker_params['track_model_params']

        if 'track_misc' not in tracker_params:
            self.track_misc = False
        else:
            assert isinstance(tracker_params['track_misc'], bool), "track_misc is not a boolean"
            self.track_misc = tracker_params['track_misc']

        if 'persist' not in tracker_params:
            self.persist = 'none'
        else:
            assert tracker_params['persist'] in ['none', 'final', 'all'], "persist is not a valid option"
            self.persist = tracker_params['persist']

        # internal data structures
        self.rounds = []
        self.imp_quality = []  # tracking history results of imputation quality
        self.imp_data = []  # tracking final imputed data
        self.model_params = []  # tracking final imputation model parameters
        self.misc = []  # tracking other parameters

        self.origin_data = None  # tracking original data
        self.mask = None  # tracking missing mask
        self.split_indices = None  # tracking split indices
        self.imp_data_final = None  # tracking final imputed data

    def record_initial(self, data: List[np.ndarray], mask: List[np.ndarray], imp_quality: dict):

        self.origin_data = np.concatenate(data)
        self.mask = np.concatenate(mask)
        self.split_indices = np.cumsum([item.shape[0] for item in data])[:-1]
        self.rounds.append(0)
        self.imp_quality.append(imp_quality)

    def record_round(
            self, round_num: int, imp_quality: dict,
            data: List[np.ndarray], model_params: List[dict], other_info: List[dict]
    ):

        self.rounds.append(round_num)
        self.imp_quality.append(imp_quality)

        if self.track_data and data is not None:
            self.imp_data.append(data)

        if self.track_model_params and model_params is not None:
            self.model_params.append(model_params)

        if self.track_misc and other_info is not None:
            self.misc.append(other_info)

    def record_final(
            self,
            imp_quality: dict,
            data: List[np.ndarray], model_params: List[dict], other_info: List[dict]
    ):

        self.rounds.append(len(self.rounds) + 1)
        self.imp_quality.append(imp_quality)
        self.imp_data_final = np.concatenate(data)

        if self.track_model_params and model_params is not None:
            self.model_params.append(model_params)

        if self.track_misc and other_info is not None:
            self.misc.append(other_info)

    def to_dict(self) -> dict:

        ret = {
            "results": {
                'rounds': self.rounds,
                "imp_quality": self.imp_quality,
            }
        }

        if self.persist == 'final':
            raise NotImplementedError("Final persist is not implemented yet")
        elif self.persist == 'all':
            raise NotImplementedError("All persist is not implemented yet")
        elif self.persist == 'none':
            ret['persist'] = {}
        else:
            raise ValueError("Invalid persist option")

        return ret
