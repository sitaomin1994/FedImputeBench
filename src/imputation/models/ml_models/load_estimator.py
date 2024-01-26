from sklearn.linear_model import (
    BayesianRidge, LinearRegression, RidgeCV, LassoCV, LogisticRegressionCV,
    LogisticRegression, Ridge, TheilSenRegressor, HuberRegressor, Lasso
)

from sklearn.neural_network import MLPRegressor


def get_estimator(estimator_name):
    # TODO: ADD SEEDED RANDOM STATE
    if estimator_name == 'bayesian_ridge':
        return BayesianRidge()
    elif estimator_name == 'linear_regression':
        return LinearRegression(n_jobs=-1)
    elif estimator_name == 'ridge':
        return Ridge(alpha=1.0, random_state=0)
    elif estimator_name == 'lasso':
        return Lasso(alpha=0.1, random_state=0)
    elif estimator_name == 'theilsen':
        return TheilSenRegressor(random_state=0, n_jobs=-1)
    elif estimator_name == 'huber':
        return HuberRegressor()
    elif estimator_name == 'ridge_cv':
        return RidgeCV(alphas=[0.1, 1.0, 10.0])
    elif estimator_name == 'lasso_cv':
        return LassoCV(alphas=[0.1, 1.0, 10.0])
    elif estimator_name == 'logistic':
        return LogisticRegression(penalty='l1', n_jobs=-1)
    elif estimator_name == 'logistic_cv':
        return LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], penalty='l1', solver='saga')
    elif estimator_name == 'mlp':
        return MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=1000, random_state=0)
    else:
        raise ValueError('Unknown estimator name: {}'.format(estimator_name))
