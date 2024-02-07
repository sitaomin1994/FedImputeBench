# This class is responsible for loading the experiment object

from emf.experiment_manager import ExperimentManager
from src.experiment_fedimp import Experiment


class FedMissExpManager(ExperimentManager):

    def load_experiment_obj(self, exp_type):
        if exp_type == 'federated_imputation':
            return Experiment()
        else:
            raise NotImplementedError
