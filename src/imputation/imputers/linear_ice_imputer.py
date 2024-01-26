from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from src.imputation.base.ice_imputer import ICEImputer
import numpy as np
from sklearn.linear_model import LogisticRegressionCV


class LinearICEImputer(ICEImputer):

    def __init__(
            self,
            estimator_num,
            estimator_cat,
            initial_strategy_cat: str = "mode",
            initial_strategy_num: str = 'mean',
            clip: bool = True,
            use_y: bool = False,
            seed: int = 0,
            data_utils_info: dict = None,
    ):
        super().__init__(estimator_num, estimator_cat, initial_strategy_cat, initial_strategy_num, clip, use_y, seed)

        # MM model
        self.mm_model = LogisticRegressionCV(
            Cs=[1e-1], cv=StratifiedKFold(3), random_state=0, max_iter=1000, n_jobs=-1, class_weight='balanced'
        )  # TODO: add more options

        # Imputation models
        self.estimators = []
        for i in range(X.shape[1]):
            if i < data_config['num_cols']:
                estimator = self.estimator_num
            else:
                estimator = self.estimator_cat
            self.estimators.append(get_estimator(estimator))

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
        estimator = self.estimators[feature_idx]
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
            "missing_info": self.missing_information[feature_idx],
            'loss': {},  # TODO: add loss
        }

    def impute(self, X, y,  missing_mask, feature_idx):

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
        estimator = self.estimators[feature_idx]
        imputed_values = estimator.predict(X_test)
        imputed_values = np.clip(imputed_values, min_values[feature_idx], max_values[feature_idx])
        X[row_mask, feature_idx] = np.squeeze(imputed_values)

        return X
