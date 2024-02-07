from abc import ABC, abstractmethod
from .base_experiment import BaseExperiment
import timeit


class ExperimentManager(ABC):

    @abstractmethod
    def load_experiment_obj(self, exp_type) -> BaseExperiment:
        raise NotImplementedError

    def execute_experiment(self, exp_type: str, config: dict):
        experiment = self.load_experiment_obj(exp_type)
        # TODO: add logging
        start = timeit.default_timer()
        result = experiment.run(config)
        end = timeit.default_timer()

        result['results']['execution_time'] = end - start
        print(f"Execution time: {result['results']['execution_time']}")
        experiment.save(result, config)
        print("Storing experiment results ... Experiment completed!")
