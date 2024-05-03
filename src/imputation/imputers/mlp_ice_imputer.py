from copy import deepcopy
from typing import Tuple

from sklearn.model_selection import StratifiedKFold
from src.imputation.base.ice_imputer import ICEImputerMixin
import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from ..base.torch_nn_imputer import TorchNNImputer
from ..model_loader_utils import load_pytorch_model
from collections import OrderedDict
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPICEImputer(TorchNNImputer, ICEImputerMixin):

    def __init__(
            self,
            estimator_num: str,
            estimator_cat: str,
            mm_model: str,
            mm_model_params: dict,
            imp_model_params: dict,
            clip: bool = True,
            use_y: bool = False,
    ):
        super().__init__()

        # estimator for numerical and categorical columns
        self.estimator_num = estimator_num
        self.estimator_cat = estimator_cat
        self.mm_model_name = mm_model
        self.mm_model_params = mm_model_params
        self.clip = clip
        self.min_values = None
        self.max_values = None
        self.use_y = use_y

        self.hidden_dim_min = imp_model_params.get('hidden_dim_min', 16)
        self.hidden_dim_max = imp_model_params.get('hidden_dim_max', 64)
        self.hidden_dim_factor = imp_model_params.get('hidden_dim_factor', 2)

        # Imputation models
        self.imp_models = None
        self.mm_model = None
        self.data_utils_info = None
        self.seed = None
        self.model_type = 'torch_nn'
        self.data_loaders = {}

    def initialize(self, data_utils, params, seed):

        # initialized imputation models
        self.imp_models = []
        for i in range(data_utils['n_features']):
            if i < data_utils['num_cols']:
                estimator = self.estimator_num
            else:
                estimator = self.estimator_cat

            if self.use_y:
                model_params = {
                    'input_dim': data_utils['n_features'],  # TODO: check whether need to one hot encode y or not
                    'hidden_dim': min(
                        max(data_utils['n_features'] * self.hidden_dim_factor, self.hidden_dim_max),
                        self.hidden_dim_min
                    )
                }
            else:
                model_params = {
                    'input_dim': data_utils['n_features'] - 1,
                     'hidden_dim': min(
                        max(data_utils['n_features'] * self.hidden_dim_factor, self.hidden_dim_max),
                        self.hidden_dim_min
                    )
                }
            self.imp_models.append(load_pytorch_model(estimator, model_params))

        # Missing Mechanism Model
        if self.mm_model_name == 'logistic':     # TODO: make mechanism model as a separate component
            self.mm_model = LogisticRegressionCV(
                Cs=self.mm_model_params['Cs'], class_weight=self.mm_model_params['class_weight'],
                cv=StratifiedKFold(self.mm_model_params['cv']), random_state=seed, max_iter=1000, n_jobs=-1
            )
        else:
            raise ValueError("Invalid missing mechanism model")

        # initialize min max values for clipping threshold
        self.min_values, self.max_values = self.get_clip_thresholds(data_utils)

        # seed same as client
        self.seed = seed
        self.data_utils_info = data_utils

    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:
        """
        Set model parameters
        :param updated_model_dict: global model parameters dictionary
        :param params: parameters for set parameters function
        :return: None
        """
        if 'feature_idx' not in params:
            raise ValueError("Feature index not provided")
        feature_idx = params['feature_idx']
        self.imp_models[feature_idx].load_state_dict(deepcopy(updated_model_dict))

    def get_imp_model_params(self, params: dict) -> OrderedDict:
        """
        Return model parameters
        :param params: dict contains parameters for get_imp_model_params
        :return: OrderedDict - model parameters dictionary
        """
        if 'feature_idx' not in params:
            raise ValueError("Feature index not provided")
        feature_idx = params['feature_idx']
        return deepcopy(self.imp_models[feature_idx].state_dict())

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
        feature_idx = params['feature_idx']
        model = self.imp_models[feature_idx]

        # set up train and test data for training imputation model
        row_mask = X_train_mask[:, feature_idx]
        X_train = X_train_imp[~row_mask][:, np.arange(X_train_imp.shape[1]) != feature_idx]
        y_train = X_train_imp[~row_mask][:, feature_idx]

        # make X_train and y_train as torch tensors and torch data loader
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)

        return model, train_loader

    def fit(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> dict:
        """
        Fit imputer to train local imputation models
        :param X: features - float numpy array
        :param y: target
        :param missing_mask: missing mask
        :param params: parameters for local training
            - feature_idx
            - local_epochs
            - learning_rate
            - batch_size
            - weight_decay
            - optimizer
        :return: fit results of local training
        """

        try:
            feature_idx = params['feature_idx']
            local_epochs = params['local_epoch']
            learning_rate = params['learning_rate']
            batch_size = params['batch_size']
            weight_decay = params['weight_decay']
            optimizer_name = params['optimizer']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        # set up train and test data for training imputation model
        if 'feature_idx' not in self.data_loaders:
            row_mask = missing_mask[:, feature_idx]
            X_train = X[~row_mask][:, np.arange(X.shape[1]) != feature_idx]
            y_train = X[~row_mask][:, feature_idx]

            # make X_train and y_train as torch tensors and torch data loader
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            train_data = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            self.data_loaders['feature_idx'] = train_loader
        else:
            train_loader = self.data_loaders['feature_idx']

        model = self.imp_models[feature_idx]
        model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # model training
        losses = []
        for epoch in range(local_epochs):
            loss_epoch = 0
            for i, inputs in enumerate(train_loader):
                inputs = [i.to(DEVICE) for i in inputs]
                model.zero_grad()
                loss, _ = model.compute_loss(inputs)
                loss.backward()
                optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                loss_epoch += loss.item()

            losses.append(loss_epoch / len(train_loader))

        model.to('cpu')
        # Fit mechanism models
        # if row_mask.sum() == 0:
        #     mm_coef = np.zeros(X.shape[1]) + 0.001
        # else:
        #     self.mm_model.fit(X, row_mask)
        #     mm_coef = np.concatenate([self.mm_model.coef_[0], self.mm_model.intercept_])

        return {
            #'mm_coef': mm_coef,
            'loss': np.array(losses).mean(),
            'sample_size': len(train_loader.dataset)
        }

    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:
        """
        Impute missing values using imputation model
        :param X: numpy array of features
        :param y: numpy array of target
        :param missing_mask: missing mask
        :param params: parameters for imputation
            - feature_idx
        :return: imputed data - numpy array - same dimension as X
        """

        if 'feature_idx' not in params:
            raise ValueError("Feature index not provided")
        feature_idx = params['feature_idx']

        # clip the imputed values
        if self.clip:
            min_values = self.min_values
            max_values = self.max_values
        else:
            min_values = np.full((X.shape[1],), 0)
            max_values = np.full((X.shape[1],), 1)

        # x_test for imputation
        row_mask = missing_mask[:, feature_idx]
        if np.sum(row_mask) == 0:
            return X

        # convert data to tensor
        X_test = X[row_mask][:, np.arange(X.shape[1]) != feature_idx]
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

        # get imputed value and do imputation
        model = self.imp_models[feature_idx]
        model.to(DEVICE)
        model.eval()
        imputed_values = model(X_test_tensor).detach().cpu().numpy()

        # convert to binary if categorical
        if feature_idx >= self.data_utils_info['num_cols']:
            imputed_values = (imputed_values >= 0.5).float()

        imputed_values = np.clip(imputed_values, min_values[feature_idx], max_values[feature_idx])
        X[row_mask, feature_idx] = np.squeeze(imputed_values)

        return X
