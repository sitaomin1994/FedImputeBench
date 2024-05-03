from copy import deepcopy
from typing import Dict, Union, List, Tuple, OrderedDict

import numpy as np
from torch.utils.data import DataLoader

from ..base.torch_nn_imputer import TorchNNImputer
from ..models.vae_models.miwae import MIWAE
import torch
from src.evaluation.imp_quality_metrics import rmse
from src.imputation.base import JMImputerMixin, BaseImputer
from tqdm.auto import tqdm, trange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MIWAEImputer(TorchNNImputer, JMImputerMixin):

    def __init__(self, imp_model_params: dict):

        super().__init__()
        self.model = None
        self.name = "miwae"

        # imputation model parameters
        self.imp_model_params = imp_model_params
        self.model_type = 'torch_nn'

    def get_imp_model_params(self, params: dict) -> OrderedDict:
        """
        Return model parameters
        :param params: dict contains parameters for get_imp_model_params
        :return: OrderedDict - model parameters dictionary
        """
        """
        Return model parameters
        """
        return deepcopy(self.model.state_dict())

    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:
        """
        Set model parameters
        :param updated_model_dict: global model parameters dictionary
        :param params: parameters for set parameters function
        :return: None
        """
        params = self.model.state_dict()
        params.update(deepcopy(updated_model_dict))
        self.model.load_state_dict(params)

    def initialize(self, data_utils: dict, params: dict, seed: int) -> None:
        """
        Initialize imputer - statistics imputation models etc.
        :param data_utils: data utils dictionary - contains information about data
        :param params: params for initialization
        :param seed: int - seed for randomization
        :return: None
        """
        # TODO: handling categorical data
        self.model = MIWAE(num_features=data_utils['n_features'], **self.imp_model_params)
        self.model.init(seed)

    def fetch_model(
            self, params: dict, X_train_imp: np.ndarray, y_train: np.ndarray, X_train_mask: np.ndarray
    ) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader]:
        """
        Fetch model for training
        :param params: parameters for training
        :param X_train_imp: imputed data
        :param y_train: target
        :param X_train_mask: missing mask
        :return: model, data loader
        """
        """
        Fetch model for training
        """
        # set up train and test data for training imputation model
        raise NotImplementedError

    def fit(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> dict:
        """
        Fit imputer to train local imputation models
        :param X: features - float numpy array
        :param y: target
        :param missing_mask: missing mask
        :param params: parameters for local training
        :return: fit results of local training
        """
        """
        Local training of imputation model for local epochs
        """
        self.model.to(DEVICE)

        # initialization weights
        # if init:
        # 	self.model.init()

        try:
            lr = params['lr']
            weight_decay = params['weight_decay']
            local_epochs = params['local_epoch']
            batch_size = params['batch_size']
            #verbose = params['verbose']
            optimizer_name = params['optimizer']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            raise NotImplementedError

        # data
        n = X.shape[0]
        X_imp = X.copy()
        X_mask = missing_mask.copy()
        bs = min(batch_size, n)

        # training
        final_loss = 0
        rmses = []
        for ep in range(local_epochs):

            # shuffle data
            perm = np.random.permutation(n)  # We use the "random reshuffling" version of SGD
            batches_data = np.array_split(X_imp[perm,], int(n / bs), )
            batches_mask = np.array_split(X_mask[perm,], int(n / bs), )
            batches_y = np.array_split(y[perm,], int(n / bs), )
            total_loss, total_iters = 0, 0
            self.model.train()
            for it in range(len(batches_data)):
                optimizer.zero_grad()
                self.model.encoder.zero_grad()
                self.model.decoder.zero_grad()
                b_data = torch.from_numpy(batches_data[it]).float().to(DEVICE)
                b_mask = torch.from_numpy(~batches_mask[it]).float().to(DEVICE)
                b_y = torch.from_numpy(batches_y[it]).long().to(DEVICE)
                data = [b_data, b_mask]

                loss, ret_dict = self.model.compute_loss(data)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_iters += 1

            # print loss
            # if (ep + 1) % 1 == 0:
            #     tqdm.write('Epoch %s/%s, Loss = %s' % (
            #     ep, local_epochs, total_loss / total_iters))

            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            final_loss = total_loss / total_iters

            # if (ep + 1) % 10000 == 0:
            #     with torch.no_grad():
            #         X_imp_new = self.model.impute(
            #             torch.from_numpy(X_imp).float().to(DEVICE), torch.from_numpy(~X_mask).float().to(DEVICE)
            #         )
            #         X_imp = X_imp_new.detach().clone().cpu().numpy()

        self.model.to("cpu")

        return {
            'loss': final_loss, 'rmse': rmses, 'sample_size': X.shape[0]
        }

    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:
        """
        Impute missing values using imputation model
        :param X: numpy array of features
        :param y: numpy array of target
        :param missing_mask: missing mask
        :param params: parameters for imputation
        :return: imputed data - numpy array - same dimension as X
        """
        # make complete
        X_train_imp = X.copy()
        X_train_imp[missing_mask] = 0
        self.model.to(DEVICE)
        self.model.eval()
        x = torch.from_numpy(X_train_imp.copy()).float().to(DEVICE)
        mask = torch.from_numpy(~missing_mask.copy()).float().to(DEVICE)
        with torch.no_grad():
            x_imp = self.model.impute(x, mask)

        x_imp = x_imp.detach().cpu().numpy()
        self.model.to("cpu")

        return x_imp
