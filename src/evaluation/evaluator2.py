from typing import List, Dict, Union
import numpy as np

from src.evaluation.imp_quality_metrics import rmse, sliced_ws
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from .twonn import TwoNNRegressor, TwoNNClassifier
from .pred_model_metrics import task_eval


class Evaluator:

    def __init__(
            self,
            imp_quality_metrics: Union[None, List[str]] = None,
            imp_fairness_metrics: Union[None, List[str]] = None,
            downstream: bool = False,
            model: str = 'linear',
            model_params: Union[None, dict] = None,
    ):

        if model_params is None:
            model_params = {}
        if imp_fairness_metrics is None:
            imp_fairness_metrics = ['variance', 'jain-index']
        if imp_quality_metrics is None:
            imp_quality_metrics = ['rmse', 'sliced-ws']

        self.imp_quality_metrics = imp_quality_metrics
        self.imp_fairness_metrics = imp_fairness_metrics
        self.model = model
        self.model_params = model_params
        self.downstream = downstream

    def run_evaluation(
            self, X_train_imps: List[np.ndarray], X_train_origins: List[np.ndarray], X_train_masks: List[np.ndarray],
            y_trains: List[np.ndarray], X_tests: List[np.ndarray], y_tests: List[np.ndarray], data_config: dict
    ):

        # imputation quality
        imp_qualities = self._evaluate_imp_quality(
            self.imp_quality_metrics, X_train_imps, X_train_origins, X_train_masks
        )

        # imputation fairness
        imp_fairness = self._evaluation_imp_fairness(self.imp_fairness_metrics, imp_qualities)

        # downstream task
        if self.downstream:
            pred_performance = self._evaluation_downstream_prediction(
                self.model, self.model_params,
                X_train_imps, X_train_origins, y_trains, X_tests, y_tests, data_config
            )
        else:
            pred_performance = {}

        # clean results
        for key, value in imp_qualities.items():
            imp_qualities[key] = list(value)

        results = {
            'imp_quality': imp_qualities,
            'imp_fairness': imp_fairness,
            'pred_performance': pred_performance
        }

        return results

    @staticmethod
    def _evaluate_imp_quality(
            metrics: List[str], X_train_imps: List[np.ndarray], X_train_origins: List[np.ndarray],
            X_train_masks: List[np.ndarray]
    ) -> dict:
        ret_all = {metric: [] for metric in metrics}
        for metric in metrics:
            if metric == 'rmse':
                ret = []
                for X_train_imp, X_train_origin, X_train_mask in zip(X_train_imps, X_train_origins, X_train_masks):
                    ret.append(rmse(X_train_imp, X_train_origin, X_train_mask))
            elif metric == 'nrmse':
                ret = []
                for X_train_imp, X_train_origin, X_train_mask in zip(X_train_imps, X_train_origins, X_train_masks):
                    rmse_ = np.sqrt(np.mean((X_train_imp - X_train_origin) ** 2))
                    std = np.std(X_train_origin)
                    ret.append(rmse_ / std)
            elif metric == 'mae':
                ret = []
                for X_train_imp, X_train_origin, X_train_mask in zip(X_train_imps, X_train_origins, X_train_masks):
                    ret.append(np.mean(np.abs(X_train_imp - X_train_origin)))
            elif metric == 'nmae':
                ret = []
                for X_train_imp, X_train_origin, X_train_mask in zip(X_train_imps, X_train_origins, X_train_masks):
                    mae_ = np.mean(np.abs(X_train_imp - X_train_origin))
                    std = np.std(X_train_origin)
                    ret.append(mae_ / std)
            elif metric == 'sliced-ws':
                ret = []
                for X_train_imp, X_train_origin in zip(X_train_imps, X_train_origins):
                    ret.append(sliced_ws(X_train_imp, X_train_origin, N=100))
            else:
                raise ValueError(f"Invalid metric: {metric}")

            ret_all[metric] = ret

        return ret_all

    @staticmethod
    def _evaluation_imp_fairness(metrics, imp_qualities: Dict[str, List[float]]) -> dict:

        ret = {metric: {} for metric in metrics}
        for metric in metrics:
            for quality_metric, imp_quality in imp_qualities.items():
                if metric == 'variance':
                    ret[metric][quality_metric] = np.var(imp_quality)
                elif metric == 'cosine-similarity':
                    imp_quality = np.array(imp_quality)
                    ret[metric][quality_metric] = np.dot(imp_quality, imp_quality) / (np.linalg.norm(imp_quality) ** 2)
                elif metric == 'jain-index':
                    ret[metric][quality_metric] = np.sum(imp_quality) ** 2 / (
                            len(imp_quality) * np.sum([x ** 2 for x in imp_quality]))
                elif metric == 'entropy':
                    imp_quality = np.array(imp_quality)
                    imp_quality = imp_quality / np.sum(imp_quality)
                    ret[metric][quality_metric] = -np.sum(imp_quality * np.log(imp_quality))
                else:
                    raise ValueError(f"Invalid metric: {metric}")

        return ret

    @staticmethod
    def _evaluation_downstream_prediction(
            model: str, model_params: dict,
            X_train_imps: List[np.ndarray], X_train_origins: List[np.ndarray], y_trains: List[np.ndarray],
            X_tests: List[np.ndarray], y_tests: List[np.ndarray], data_config: dict
    ):
        try:
            task_type = data_config['task_type']
            clf_type = data_config['clf_type']
        except KeyError:
            raise KeyError("task_type is not defined in data_config")

        assert task_type in ['classification', 'regression'], f"Invalid task_type: {task_type}"
        if task_type == 'classification':
            assert clf_type in ['binary', 'multi-class'], f"Invalid clf_type: {clf_type}"

        ################################################################################################################
        # Loader classification model
        if model == 'linear':
            if task_type == 'classification':
                if clf_type == 'binary':
                    clf = LogisticRegressionCV(Cs=5, class_weight='balanced', solver='saga', **model_params)
                else:
                    clf = LogisticRegressionCV(Cs=5, class_weight='balanced', solver='saga', **model_params)
            else:
                clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], **model_params)
        elif model == 'tree':
            if task_type == 'classification':
                if clf_type == 'binary':
                    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', **model_params)
                else:
                    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', **model_params)
            else:
                clf = RandomForestRegressor(n_estimators=100, **model_params)
        elif model == 'nn':
            if task_type == 'classification':
                clf = TwoNNClassifier(**model_params)
            else:
                clf = TwoNNRegressor(**model_params)
        else:
            raise ValueError(f"Invalid model: {model}")

        ################################################################################################################
        # Evaluation
        if task_type == 'classification':
            eval_metrics = ['accuracy', 'f1', 'auc', 'prc']
        else:
            eval_metrics = ['mse', 'mae', 'r2', 'mlse']

        ret = {eval_metric: [] for eval_metric in eval_metrics}
        for idx, (X_train_imp, X_train_origin, y_train, X_test, y_test) in enumerate(zip(
                X_train_imps, X_train_origins, y_trains, X_tests, y_tests
        )):
            clf.fit(X_train_imp, y_train)
            y_pred = clf.predict(X_test)
            if task_type == 'classification':
                y_pred_proba = clf.predict_proba(X_test)
            else:
                y_pred_proba = None

            for eval_metric in eval_metrics:
                ret[eval_metric][idx] = task_eval(
                    eval_metric, task_type, clf_type, y_pred, y_test, y_pred_proba
                )

        return ret

    def _eval_downstream_fed_prediction(self):
        pass
