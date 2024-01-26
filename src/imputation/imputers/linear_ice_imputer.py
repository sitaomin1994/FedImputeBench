from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from src.imputation.base.ice_imputer import ICEImputer
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from ..model_loader_utils import load_linear_model


class LinearICEImputer(ICEImputer):

    def __init__(
            self,
            estimator_num,
            estimator_cat,
            clip: bool = True,
            use_y: bool = False,
    ):
        super().__init__(estimator_num, estimator_cat, clip, use_y)

        # MM model
        self.mm_model = LogisticRegressionCV(
            Cs=[1e-1], cv=StratifiedKFold(3), random_state=0, max_iter=1000, n_jobs=-1, class_weight='balanced'
        )  # TODO: add more options

        # Imputation models
        self.imp_models = None
        self.data_utils_info = None
        self.seed = None

    def initialize(self, data_utils, seed):

        # initialized imputation models
        self.imp_models = []
        for i in range(data_utils['n_features']):
            if i < data_utils['num_cols']:
                estimator = self.estimator_num
            else:
                estimator = self.estimator_cat

            self.imp_models.append(load_linear_model(estimator))

        # initialize min max values for clipping threshold
        self.min_values, self.max_values = self.get_clip_thresholds(data_utils)

        # seed same as client
        self.seed = seed
        self.data_utils_info = data_utils

    def update_imp_model(self, updated_model: dict, feature_idx):
        updated_model['w_b'] = np.array(updated_model['w_b'])
        # TODO: make imp model as a class that has get_params() interface so it can using non-sklearn models
        self.imp_models[feature_idx].coef_ = updated_model['w_b'][:-1]
        self.imp_models[feature_idx].intercept_ = updated_model['w_b'][-1]

    def get_imp_model_params(self, feature_idx):
        imp_model = self.imp_models[feature_idx]
        # TODO: make imp model as a class that has get_params() interface so it can using non-sklearn models
        parameters = np.concatenate([imp_model.coef_, np.expand_dims(imp_model.intercept_, 0)])
        return {"w_b": parameters}

    def fit(self, X, y, missing_mask, feature_idx):

        # get feature based train test
        num_cols = self.data_utils_info['num_cols']
        regression = self.data_utils_info['task_type'] == 'regression'
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

        # fit linear imputation models
        estimator = self.imp_models[feature_idx]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_train)
        coef = np.concatenate([estimator.coef_, np.expand_dims(estimator.intercept_, 0)])

        # Fit mechanism models
        if row_mask.sum() == 0:
            mm_coef = np.zeros(X.shape[1]) + 0.001
        else:
            self.mm_model.fit(X, row_mask)
            mm_coef = np.concatenate([self.mm_model.coef_[0], self.mm_model.intercept_])

        return {
            'coef': coef,
            'mm_coef': mm_coef,
            'loss': {},  # TODO: add loss
        }

    def impute(self, X, y, missing_mask, feature_idx):

        if self.clip:
            min_values = self.min_values[feature_idx]
            max_values = self.max_values[feature_idx]
        else:
            min_values = np.full((X.shape[1],), 0)
            max_values = np.full((X.shape[1],), 1)

        row_mask = missing_mask[:, feature_idx]
        if np.sum(row_mask) == 0:
            return X

        # one hot encoding for categorical columns
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

        # impute missing values
        X_test = X[row_mask][:, np.arange(X.shape[1]) != feature_idx]
        estimator = self.imp_models[feature_idx]
        imputed_values = estimator.predict(X_test)
        imputed_values = np.clip(imputed_values, min_values[feature_idx], max_values[feature_idx])
        X[row_mask, feature_idx] = np.squeeze(imputed_values)

        return X
