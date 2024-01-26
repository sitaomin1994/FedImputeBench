# third party
import numpy as np
from .client import Client
from src.imputation.base import BaseImputer, JMImputer
# local model and reproducible
from emf.reproduce_utils import set_seed
import src.modules.evaluation.imp_quality_metrics as imp_quality_metrics


class JMClient(Client):

    def __init__(
            self, client_id: int,
            train_data: np.ndarray, test_data: np.ndarray, X_train_ms: np.ndarray, data_config: dict,
            imp_model: JMImputer, seed=0
    ) -> None:

        # super call
        super().__init__(client_id, train_data, test_data, X_train_ms, data_config, imp_model, seed)

        # pytorch dataloader
        self.train_dataset = None
        self.train_dataloader = None
        self.val_dataset = None
        self.val_data_loader = None

        # evaluation result
        self.eval_ret = None

        # initial imputation
        self.initial_impute()

    def initial_impute(self):
        """
        Initial imputation using mean imputation
        """
        self.X_train_imp[self.X_train_mask] = 0

    def fit_local_imputation_model(
            self, params, init=True, global_z=None, global_decoder=None, global_encoder=None
    ):
        """
        Local training of imputation model for local epochs
        """
        model_parameters = {}
        local_train_ret_dict = self.imp_model.local_train(
            self.X_train_imp, self.X_train_mask, self.X_train, self.y_train, params, init, global_z=global_z,
            global_decoder=global_decoder, global_encoder=global_encoder
        )

        return model_parameters, local_train_ret_dict

    def imputation(self, params):
        """
        Imputation using local trained imputation model
        """
        self.X_train_imp = self.imp_model.imputation(self.X_train_ms, self.X_train_mask, params)

    # def local_evaluate(self, params, pred_eval = False):
    #     """
    #     Local evaluation of imputation model
    #     """
    #     eval_result = {}
    #
    #     # imputation quality
    #     eval_result['imp_rmse'] = imp_quality_metrics.rmse(self.X_train_imp, self.X_train, self.X_train_mask)
    #     eval_result['imp_sliced-ws'] = imp_quality_metrics.sliced_ws(self.X_train_imp, self.X_train)
    #
    #     # downstream task evaluation
    #     if pred_eval:
    #         raise NotImplementedError # TODO: implement this
    #         # eval_params = {
    #         #     "clf": params['clf'], "task_type": self.data_config["task_type"], "seed": self.seed,
    #         #     "n_rounds": params["n_rounds"]
    #         # }
    #         # eval_result.update(
    #         #     model_performance_evaluation(
    #         #         self.X_train, self.y_train, self.X_test, self.y_test, self.X_train_imp, params=eval_params
    #         #     )
    #         # )
    #
    #     self.eval_ret = eval_result
    #     return eval_result

    def get_sample_size(self):
        return self.X_train.shape[0]
