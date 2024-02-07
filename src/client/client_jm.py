# third party
import numpy as np
from .client import Client
from src.imputation.base import BaseImputer, JMImputer
# local model and reproducible
from emf.reproduce_utils import set_seed


class JMClient(Client):

    def __init__(
            self, client_id: int,
            train_data: np.ndarray, test_data: np.ndarray, X_train_ms: np.ndarray, data_config: dict,
            imp_model: JMImputer, client_config:dict, seed=0,
    ) -> None:

        # super call
        # initialize data and initial imputation
        super().__init__(
            client_id, train_data, test_data, X_train_ms, data_config,
            imp_model, client_config, seed,
        )

        # initialize imputation model
        self.imp_model = imp_model
        self.imp_model.initialize(self.data_utils, seed)

        # pytorch dataloader
        self.train_dataset = None
        self.train_dataloader = None
        self.val_dataset = None
        self.val_data_loader = None

        # evaluation result
        self.eval_ret = None

    def fit_local_imputation_model(self, params):
        """
        Local training of imputation model for local epochs
        # TODO: add parameters to dynamic control the training
        """
        fit_res = self.imp_model.fit_local_imp_model(
            self.X_train_imp, self.X_train_mask, self.X_train, self.y_train, params
        )

        # get model parameters
        model_parameters = self.imp_model.get_imp_model_params()
        fit_res.update(self.data_utils)

        return model_parameters, fit_res

    def imputation(self, params):
        """
        Imputation using local trained imputation model
        """
        self.X_train_imp = self.imp_model.imputation(self.X_train_ms, self.X_train_mask, params)
