import gc
from copy import deepcopy
from typing import Dict, Union, List, Tuple, OrderedDict

import numpy as np
from torch.utils.data import DataLoader

from ..models.vae_models.gnr import GNR
from ..models.vae_models.miwae import MIWAE
from ..models.vae_models.notmiwae import NOTMIWAE
import torch
from src.evaluation.imp_quality_metrics import rmse
from src.imputation.base import JMImputerMixin, BaseNNImputer
from tqdm.auto import tqdm, trange

from src.utils.nn_utils import load_optimizer, load_lr_scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MIWAEImputer(BaseNNImputer, JMImputerMixin):

    def __init__(self, name, imp_model_params: dict, clip: bool = True):

        super().__init__()
        self.model = None
        self.name = name

        # imputation model parameters
        self.clip = clip
        self.min_values = None
        self.max_values = None
        self.imp_model_params = imp_model_params
        self.model_type = 'torch_nn'
        self.train_dataloader = None
        self.model_persistable = True

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
        params_dict = self.model.state_dict()
        params_dict.update(deepcopy(updated_model_dict))
        self.model.load_state_dict(params_dict)

    def initialize(
            self, X: np.array, missing_mask: np.array, data_utils: dict, params: dict, seed: int
    ) -> None:
        """
        Initialize imputer - statistics imputation models etc.
        :param X: data with intial imputed values
        :param missing_mask: missing mask of data
        :param data_utils:  utils dictionary - contains information about data
        :param params: params for initialization
        :param seed: int - seed for randomization
        :return: None
        """
        if self.name == 'miwae':
            self.model = MIWAE(num_features=data_utils['n_features'], **self.imp_model_params)
        elif self.name == 'notmiwae':
            self.model = NOTMIWAE(num_features=data_utils['n_features'], **self.imp_model_params)
        elif self.model == 'gnr':
            self.model = GNR(num_features=data_utils['n_features'], **self.imp_model_params)
        else:
            raise ValueError(f"Model {self.name} not supported")

        self.model.init(seed)
        #self.model = torch.compile(self.model)
        self.min_values, self.max_values = self.get_clip_thresholds(data_utils)

    def configure_model(
            self, params: dict, X: np.ndarray, y: np.ndarray, missing_mask: np.ndarray
    ) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader]:

        try:
            batch_size = params['batch_size']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        # if self.train_dataloader is not None:
        #     return self.model, self.train_dataloader
        # else:
        n = X.shape[0]
        X_imp = X.copy()
        X_mask = missing_mask.copy()
        bs = min(batch_size, n)

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_imp).float(), torch.from_numpy(~X_mask).float()
        )
        train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)

        return self.model, train_dataloader

    def configure_optimizer(
            self, params: dict, model: torch.nn.Module
    ) -> tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:

        try:
            learning_rate = params['learning_rate']
            weight_decay = params['weight_decay']
            optimizer_name = params['optimizer']
            scheduler_name = params['scheduler']
            scheduler_params = params['scheduler_params']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        optimizer = load_optimizer(optimizer_name, model.parameters(), learning_rate, weight_decay)
        lr_scheduler = load_lr_scheduler(scheduler_name, optimizer, scheduler_params)

        return [optimizer], [lr_scheduler]

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
            lr = params['learning_rate']
            weight_decay = params['weight_decay']
            local_epochs = params['local_epoch']
            batch_size = params['batch_size']
            # verbose = params['verbose']
            optimizer_name = params['optimizer']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            raise NotImplementedError

        # data
        n = X.shape[0]
        X_imp = X.copy()
        X_mask = missing_mask.copy()
        bs = min(batch_size, n)
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_imp).float(), torch.from_numpy(~X_mask).float()
        )
        train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        # training
        final_loss = 0
        rmses = []
        for ep in range(local_epochs):

            # shuffle data
            # perm = np.random.permutation(n)  # We use the "random reshuffling" version of SGD
            # batches_data = np.array_split(X_imp[perm,], int(n / bs), )
            # batches_mask = np.array_split(X_mask[perm,], int(n / bs), )
            # batches_y = np.array_split(y[perm,], int(n / bs), )
            total_loss, total_iters = 0, 0
            self.model.train()
            # for it in range(len(batches_data)):
            for it, inputs in enumerate(train_dataloader):
                optimizer.zero_grad()
                self.model.encoder.zero_grad()
                self.model.decoder.zero_grad()
                # b_data = torch.from_numpy(batches_data[it]).float().to(DEVICE)
                # b_mask = torch.from_numpy(~batches_mask[it]).float().to(DEVICE)
                b_data, b_mask = inputs
                b_data = b_data.to(DEVICE)
                b_mask = b_mask.to(DEVICE)
                # b_y = torch.from_numpy(batches_y[it]).long().to(DEVICE)
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
        Impute missing values using an imputation model
        :param X: numpy array of features
        :param y: numpy array of target
        :param missing_mask: missing mask
        :param params: parameters for imputation
        :return: imputed data - numpy array - same dimension as X
        """
        # make complete
        X_train_imp = X
        X_train_imp[missing_mask] = 0
        x = torch.from_numpy(X_train_imp.copy()).float().to(DEVICE)
        mask = torch.from_numpy(~missing_mask.copy()).float().to(DEVICE)

        self.model.to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            x_imp = self.model.impute(x, mask)

        x_imp = x_imp.detach().cpu().numpy()
        self.model.to("cpu")

        del X
        gc.collect()

        if self.clip:
            for i in range(x_imp.shape[1]):
                x_imp[:, i] = np.clip(x_imp[:, i], self.min_values[i], self.max_values[i])

        return x_imp
