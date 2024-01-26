from typing import List
import numpy as np
from src.client import Client
from src.evaluation.imp_quality_metrics import rmse, sliced_ws


class Evaluator:

    def __init__(self):
        pass

    @staticmethod
    def evaluate_imputation(clients: List[Client]):
        evaluation_results = {
            'imp_rmse_clients': [],
            'imp_ws_clients': [],
            'imp_rmse_avg': 0,
            'imp_ws_avg': 0,
            'imp_rmse_global': 0,
            'imp_ws_global': 0
        }

        # imputation quality evaluation
        for client_id, client in enumerate(clients):
            imp_rmse = rmse(client.X_train_imp, client.X_train, client.X_train_mask)
            imp_ws = sliced_ws(client.X_train_imp, client.X_train)
            evaluation_results['imp_rmse_clients'].append(imp_rmse)
            evaluation_results['imp_ws_clients'].append(imp_ws)

        evaluation_results['imp_rmse_avg'] = float(np.mean(evaluation_results['imp_rmse_clients']))
        evaluation_results['imp_ws_avg'] = float(np.mean(evaluation_results['imp_ws_clients']))

        # global imputation quality evaluation
        merged_X_imp = np.concatenate([client.X_train_imp for client in clients], axis=0)
        merged_X_origin = np.concatenate([client.X_train for client in clients], axis=0)
        merged_X_mask = np.concatenate([client.X_train_mask for client in clients], axis=0)
        imp_rmse = rmse(merged_X_imp, merged_X_origin, merged_X_mask)
        imp_ws = sliced_ws(merged_X_imp, merged_X_origin)
        evaluation_results['imp_rmse_global'] = imp_rmse
        evaluation_results['imp_ws_global'] = imp_ws

        return evaluation_results

    @staticmethod
    def evaluation_prediction(clients: List[Client]):
        pass
