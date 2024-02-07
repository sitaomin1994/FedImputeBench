from abc import ABC, abstractmethod
from .file_backend import FileBackend


class BaseExperiment(ABC):

    def __init__(self, exp_name, exp_type):
        self.exp_name = exp_name
        self.exp_type = exp_type
        self.exp_results = {}
        self.save_backend = FileBackend()

    @abstractmethod
    def run(self, config) -> dict:
        pass

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
