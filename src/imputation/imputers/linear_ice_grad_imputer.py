from copy import deepcopy

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from src.imputation.base.ice_imputer import ICEImputer
from src.imputation.base.base_imputer import BaseImputer
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from ..model_loader_utils import load_pytorch_model
from collections import OrderedDict
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPICEImputer(BaseImputer, ICEImputer):

    def __init__(
            self,
            estimator_num,
            estimator_cat,
            mm_model,
            mm_model_params,
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

        # Imputation models
        self.imp_models = None
        self.mm_model = None
        self.data_utils_info = None
        self.seed = None

    def initialize(self, data_utils, params, seed):

        # initialized imputation models
        self.imp_models = []
        for i in range(data_utils['n_features']):
            if i < data_utils['num_cols']:
                estimator = self.estimator_num
            else:
                estimator = self.estimator_cat

            model_params = {}
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

    def set_imp_model_params(self, updated_model: OrderedDict, feature_idx):
        self.imp_models[feature_idx].load_state_dict(updated_model)

    def get_imp_model_params(self, feature_idx):
        return deepcopy(self.imp_models[feature_idx].state_dict())

    def fit(self, X, y, missing_mask, feature_idx):

        # TODO: see where to get this params from
        local_epochs = 5
        learning_rate = 0.01
        batch_size = 32
        weight_decay = 0.01
        optimizer = 'adam'
        regression = self.data_utils_info['task_type'] == 'regression'

        # get feature based train test
        num_cols = self.data_utils_info['num_cols']
        regression = self.data_utils_info['task_type'] == 'regression'

        # set up train and test data for training imputation model
        row_mask = missing_mask[:, feature_idx]
        X_cat = X[:, num_cols:]
        if X_cat.shape[1] > 0:
            onehot_encoder = OneHotEncoder(max_categories=5, drop="if_binary")
            X_cat = onehot_encoder.fit_transform(X_cat)
            X = np.concatenate((X[:, :num_cols], X_cat), axis=1)

        if self.use_y:
            if regression:
                oh = OneHotEncoder(drop='first')
                y = oh.fit_transform(y.reshape(-1, 1)).toarray()
            X = np.concatenate((X, y.reshape(-1, 1)), axis=1)

        X_train = X[~row_mask][:, np.arange(X.shape[1]) != feature_idx]
        y_train = X[~row_mask][:, feature_idx]

        # make X_train and y_train as torch tensors and torch data loader
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        model = self.imp_models[feature_idx]
        model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # model training
        losses = []
        for epoch in range(local_epochs):
            loss_epoch = 0
            for i, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                model.train()
                model.zero_grad()
                y_pred = model(X_batch)
                if regression:
                    loss = torch.nn.functional.mse_loss(y_pred, y_batch)
                else:
                    loss = torch.nn.functional.cross_entropy(y_pred, y_batch)  # what if it is multi-class

                loss.backward()
                optimizer.step()
                loss_epoch += loss.item().detach().cpu().numpy()
                torch.cuda.empty_cache()

            losses.append(loss_epoch/len(train_loader))

        model.to('cpu')
        # Fit mechanism models
        if row_mask.sum() == 0:
            mm_coef = np.zeros(X.shape[1]) + 0.001
        else:
            self.mm_model.fit(X, row_mask)
            mm_coef = np.concatenate([self.mm_model.coef_[0], self.mm_model.intercept_])

        return {
            'mm_coef': mm_coef,
            'loss': np.array(losses).mean()
        }

    def impute(self, X, y, missing_mask, feature_idx):

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

        num_cols = self.data_utils_info['num_cols']
        regression = self.data_utils_info['task_type'] == 'regression'
        X_cat = X[:, num_cols:]
        if X_cat.shape[1] > 0:
            onehot_encoder = OneHotEncoder(sparse=False, max_categories=10, drop="if_binary")
            X_cat = onehot_encoder.fit_transform(X_cat)
            X = np.concatenate((X[:, :num_cols], X_cat), axis=1)
        else:
            X = X[:, :num_cols]

        if self.use_y:
            if regression:
                oh = OneHotEncoder(drop='first')
                y = oh.fit_transform(y.reshape(-1, 1)).toarray()
            X = np.concatenate((X, y.reshape(-1, 1)), axis=1)

        # convert data to tensor
        X_test = X[row_mask][:, np.arange(X.shape[1]) != feature_idx]
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

        # get imputed value and do imputation
        model = self.imp_models[feature_idx]
        model.to(DEVICE)
        model.eval()
        imputed_values = model(X_test_tensor).detach().cpu().numpy()
        imputed_values = np.clip(imputed_values, min_values[feature_idx], max_values[feature_idx])
        X[row_mask, feature_idx] = np.squeeze(imputed_values)

        return X
