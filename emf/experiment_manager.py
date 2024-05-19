from abc import ABC, abstractmethod
from .base_experiment import BaseExperiment
import timeit
import loguru
from emf.logging import setup_logger


class ExperimentManager(ABC):

    @abstractmethod
    def load_experiment_obj(self, exp_type) -> BaseExperiment:
        pass

    def execute_experiment(self, exp_type: str, config: dict, experiment_meta: dict):
        # load experiment object
        experiment = self.load_experiment_obj(exp_type)

        # set up logging system
        setup_logger(
            experiment.get_experiment_out_dir(experiment_meta), to_file=experiment_meta['log_to_file'],
            level=experiment_meta['logging_level']
        )

        # execute experiment
        start = timeit.default_timer()
        result = experiment.run(config, experiment_meta)
        end = timeit.default_timer()

        # saving results
        result['results']['execution_time'] = end - start
        loguru.logger.info(f"Experiment finished execution time: {result['results']['execution_time']}")
        experiment.save(result, config, experiment_meta)
        loguru.logger.debug(
            f"Storing experiment results {experiment.get_experiment_out_dir(experiment_meta)} Experiment completed!"
        )
