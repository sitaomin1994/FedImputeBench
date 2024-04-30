from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import json
from pathlib import Path
from config import settings, ROOT_DIR
import base64


class Backend(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def save(self, experiment_name, experiment_result):
        pass


class FileBackend:

    def __init__(self):
        self.result_dir_path = settings['result_dir']["base"] + '/' + settings['result_dir']["raw"]
        self.config_dir_path = settings['exp_config_dir']

    def consolidate_save_path(self, experiment_name: str, output_path: str):
        """
        Consolidate the save path for the experiment results
        :param experiment_name: experiment name
        :param output_path: output path in configuration
        :return: following format -> ROOT_DIR/result_dir_path/experiment_name/output_path/date/
        """
        date = datetime.now().strftime("%m%d")
        return f"{ROOT_DIR}/{self.result_dir_path}/{experiment_name}/{output_path}/{date}/"

    @staticmethod
    def save(dir_path: str, experiment_results: dict):
        """
        Save the experiment results to the directory
        :param dir_path: experiment directory path
        :param experiment_results: experiment results dictionary
        :return:
        """

        # create directory if not exists
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        ################################################################################################
        # save experiment meta
        assert "experiment_meta" in experiment_results, "experiment meta is not found in the experiment results"
        assert "config" in experiment_results, "configurations are not found in the experiment results"
        with open(dir_path + '/config.json', 'w') as f:
            json.dump({
                "experiment": experiment_results['experiment_meta'],
                "config":experiment_results['config']
            }, f)

        ################################################################################################
        # save experiment results
        assert "results" in experiment_results, "results are not found in the experiment results"
        if "results" in experiment_results:
            with open(dir_path + '/results.json', 'w') as f:
                json.dump(experiment_results['results'], f)

        ##############################################################################################
        # save images
        if "plots" in experiment_results and experiment_results["plots"] is not None:
            for img_name, img_content in experiment_results["plots"].items():
                with open(dir_path + '/' + img_name + '.png', 'wb') as f:
                    f.write(base64.b64decode(img_content))

        ##############################################################################################
        # save data
        if "data" in experiment_results and experiment_results["data"] is not None:
            data_dir = dir_path + '/data/'
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            for key, value in experiment_results["data"].items():
                if isinstance(value, np.ndarray):
                    data_file_name = "{}.npy".format(key)
                    np.save(data_dir + '/' + data_file_name, value)
                elif isinstance(value, list):
                    for idx, item in enumerate(value):
                        data_file_name = "{}_{}.npy".format(key, idx)
                        np.save(data_dir + '/' + data_file_name, item)
