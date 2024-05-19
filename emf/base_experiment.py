from abc import ABC, abstractmethod
from typing import List
import joblib
import loguru

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
        :param experiment_meta: experiment meta configuration
        :param config: experiment configuration
        :return: dictionary of results
        """
        pass

    def multiple_runs(self, config, seed, n_rounds, mtp=False) -> List[dict]:
        """
        Run multiple rounds of the experiment
        :param mtp: whether to run multiple rounds experiment in parallel
        :param n_rounds: number of rounds to run
        :param config: configuration for experiment
        :param seed: list of random seeds for each run
        :return: dictionary of results
        """
        results = []
        seeds = setup_seeds(seed, n_rounds)
        if mtp is False:
            for seed in seeds:
                results.append(self.single_run(config, seed))
        else:
            # todo check how to use this
            r = joblib.Parallel(n_jobs=-1, backend='loky')(
                joblib.delayed(self.single_run)(config, seed) for seed in seeds
            )

            results = r

        return results

    def save(self, results: dict, config: dict, experiment_meta: dict):
        """
        Save the experiment results
        :param experiment_meta: experiment meta configuration
        :param results: results dictionary
        :param config: experiment configuration dictionary
        :return: None - results will be written to file using file backend
        """
        exp_result_dict = {
            "experiment_meta": experiment_meta,
            "config": config,
            "results": results['results'],
            "plots": results['plots'] if "plots" in results else None,
            "data": results['data'] if "data" in results else None
        }
        save_path = self.save_backend.consolidate_save_path(
            experiment_meta['experiment_name'], experiment_meta['output_path']
        )
        loguru.logger.debug(f"Saving experiment results to {save_path}")
        self.save_backend.save(save_path, exp_result_dict)

    def get_experiment_out_dir(self, experiment_meta: dict) -> str:
        """
        Get the experiment output directory
        :param experiment_meta: experiment meta-configuration
        :return: output directory path
        """
        return self.save_backend.consolidate_save_path(
            experiment_meta['experiment_name'], experiment_meta['output_path']
        )

    def merge_results(self, results: List[dict]) -> dict:  # TODO
        """
        Merge results from multiple rounds
        :param results: list of results from multiple rounds
        :return: merged results
        """
        pass
