from abc import ABC
from copy import deepcopy
from typing import Dict, Union, List, Tuple

import numpy as np
from torch.utils.data import DataLoader
from ..models.vae_models.miwae import MIWAE
import torch
from src.evaluation.imp_quality_metrics import rmse
from src.imputation.base import JMImputer
from tqdm.auto import tqdm, trange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MIWAEImputer(JMImputer):

    def __init__(self, imp_model_params: dict, imp_params: dict):

        super().__init__()
        self.model = None
        self.name = "miwae"

        # imputation model parameters
        self.imp_model_params = imp_model_params

        # imputation parameters
        self.lr = imp_params['lr']
        self.weight_decay = imp_params['weight_decay']
        self.local_epochs = imp_params['local_epochs']
        self.imp_interval = imp_params['imp_interval']
        self.batch_size = imp_params['batch_size']
        self.verbose = imp_params['verbose']
        self.optimizer_name = imp_params['optimizer']

    def get_imp_model_params(self) -> dict:
        """
        Return model parameters
        """
        return deepcopy(self.model.state_dict())

    def update_imp_model(self, updated_model: dict) -> None:
        params = self.model.state_dict()
        params.update(updated_model)
        self.model.load_state_dict(params)

    def initialize(self, data_utils, seed) -> None:
        # TODO: handling categorical data
        self.model = MIWAE(num_features=data_utils['n_features'], **self.imp_model_params)
        self.model.init(seed)

    def fit_local_imp_model(
            self, X_train_imp: np.ndarray, X_train_mask: np.ndarray, X_train_full: np.ndarray,
            y_train: np.ndarray, params: dict
    ) -> Dict[str, float]:
        """
        Local training of imputation model for local epochs
        """
        self.model.to(DEVICE)

        # initialization weights
        # if init:
        # 	self.model.init()

        # optimizer and params
        lr = self.lr
        weight_decay = self.weight_decay
        local_epochs = self.local_epochs
        imputation_interval = self.imp_interval
        batch_size = self.batch_size
        verbose = self.verbose
        optimizer_name = self.optimizer_name

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            raise NotImplementedError

        # data
        n = X_train_imp.shape[0]
        X_imp = X_train_imp.copy()
        X_mask = X_train_mask.copy()
        bs = min(batch_size, n)

        # training
        final_loss = 0
        rmses = []
        for ep in trange(local_epochs, desc='Client Local Epoch', leave=False, colour='blue'):

            # evaluation
            if ep % imputation_interval == 0:
                with torch.no_grad():
                    X_imp_new = self.model.impute(
                        torch.from_numpy(X_imp).float().to(DEVICE), torch.from_numpy(~X_mask).float().to(DEVICE)
                    )
                    rmse_value = rmse(X_imp_new.detach().clone().cpu().numpy(), X_train_full, X_mask)
                    rmses.append(rmse_value)

            # shuffle data
            perm = np.random.permutation(n)  # We use the "random reshuffling" version of SGD
            batches_data = np.array_split(X_imp[perm,], int(n / bs), )
            batches_mask = np.array_split(X_mask[perm,], int(n / bs), )
            batches_y = np.array_split(y_train[perm,], int(n / bs), )
            total_loss, total_iters = 0, 0
            kd_loss = 0
            self.model.train()
            for it in range(len(batches_data)):
                optimizer.zero_grad()
                self.model.encoder.zero_grad()
                self.model.decoder.zero_grad()
                b_data = torch.from_numpy(batches_data[it]).float().to(DEVICE)
                b_mask = torch.from_numpy(~batches_mask[it]).float().to(DEVICE)
                b_y = torch.from_numpy(batches_y[it]).long().to(DEVICE)
                data = [b_data, b_mask]

                # TODO: check how to integrate these
                # global_z_, global_decoder_, global_encoder_ = None, None, None
                # if global_z is not None:
                #     global_z_ = torch.from_numpy(global_z).float().to(DEVICE)
                #
                # if global_decoder is not None:
                #     global_decoder_ = []
                #     for item in global_decoder:
                #         global_decoder_.append(torch.from_numpy(item).float().to(DEVICE))
                #
                # if global_encoder is not None:
                #     global_encoder_ = []
                #     for item in global_encoder:
                #         global_encoder_.append(torch.from_numpy(item).float().to(DEVICE))

                loss, ret_dict = self.model.compute_loss(data)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_iters += 1

            # print loss
            if (ep + 1) % verbose == 0:
                tqdm.write('Epoch %s/%s, Loss = %s RMSE = %s' % (
                ep, local_epochs, total_loss / total_iters, rmses[-1]))

            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            final_loss = total_loss / total_iters

            if (ep + 1) % 10000 == 0:
                with torch.no_grad():
                    X_imp_new = self.model.impute(
                        torch.from_numpy(X_imp).float().to(DEVICE), torch.from_numpy(~X_mask).float().to(DEVICE)
                    )
                    X_imp = X_imp_new.detach().clone().cpu().numpy()

        self.model.to("cpu")

        # evaluation on the end of the training
        with torch.no_grad():
            X_imp_new = self.model.impute(
                torch.from_numpy(X_imp).float().to(DEVICE), torch.from_numpy(~X_mask).float().to(DEVICE)
            )
            rmse_value = rmse(X_imp_new.detach().clone().cpu().numpy(), X_train_full, X_mask)
            rmses.append(rmse_value)

        # self.imputation(X_train_imp, X_train_mask, {'L': L})

        return {'loss': final_loss, 'rmse': rmses}

    def imputation(
            self, X_train_ms: np.ndarray, X_train_mask: np.ndarray, params: dict
    ) -> np.ndarray:
        """
        Impute missing values of client data
        """
        # make complete
        X_train_imp = X_train_ms.copy()
        X_train_imp[X_train_mask] = 0
        self.model.to(DEVICE)
        x = torch.from_numpy(X_train_imp.copy()).float().to(DEVICE)
        mask = torch.from_numpy(~X_train_mask.copy()).float().to(DEVICE)
        with torch.no_grad():
            x_imp = self.model.impute(x, mask)

        x_imp = x_imp.detach().cpu().numpy()
        self.model.to("cpu")

        return x_imp
