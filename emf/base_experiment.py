from abc import ABC, abstractmethod
from typing import List
import joblib
from .file_backend import FileBackend
from .reproduce_utils import setup_seeds


class BaseExperiment(ABC):

    def __init__(self, exp_name, exp_type):
        self.exp_name = exp_name
        self.exp_type = exp_type
        self.exp_results = {}
        self.save_backend = FileBackend()

    @abstractmethod
    def single_run(self, config, seed) -> dict:
        """
        Run the single round experiment
        :param config: configuration for experiment
        :param seed: random seed for experiment
        :return: dictionary of results
        """
        pass

    @abstractmethod
    def run(self, config, experiment_meta) -> dict:
        """
        Run the experiment
        :param config: configuration for experiment
        :return: dictionary of results
        """
        pass

    def multiple_runs(self, config, seed, n_rounds, mtp = False) -> List[dict]:  # todo: result analyzer
        """
        Run multiple rounds of the experiment
        :param config: configuration for experiment
        :param seeds: list of random seeds for each run
        :return: dictionary of results
        """
        results = []
        seeds = setup_seeds(seed, n_rounds)
        if mtp is False:
            for seed in seeds:
                results.append(self.single_run(config, seed))
        else:
            r = joblib.Parallel(n_jobs=-1, backend='loky')(
                joblib.delayed(self.single_run)(config, seed) for seed in seeds
            )

            results = r  # todo check how to use this

        return results

    def save(self, results: dict, config: dict):
        exp_result_dict = {
            "config": config,
            "results": results['results'],
            "plots": results['plots'] if "plots" in results else None,
            "data": results['data'] if "data" in results else None
        }
        save_path = self.save_backend.consolidate_save_path(
            config['experiment']['experiment_name'], config['experiment']['output_path']
        )
        print(f"Saving experiment results to {save_path}")
        self.save_backend.save(save_path, exp_result_dict)

    def merge_results(self, results: List[dict]) -> dict:   # TODO
        """
        Merge results from multiple rounds
        :param results: list of results from multiple rounds
        :return: merged results
        """
        pass

