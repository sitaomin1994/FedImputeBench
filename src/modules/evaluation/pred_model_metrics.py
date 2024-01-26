import numpy as np
#from .eval_models import load_evaluation_model, eval_metrics


def model_performance_evaluation(
        X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
        X_train_imp: np.ndarray, params: dict
) -> dict:
    task_type = params['task_type']
    clf = params['clf']
    seed = params['seed']
    n_rounds = params['n_rounds']
    # tune_params = params['tune_params']

    eval_ret_imp_all = []
    eval_ret_ori_all = []
    for i in range(n_rounds):
        seed = i * 1029390 + seed

        # load evaluation model
        model, param_grids = load_evaluation_model(clf, seed)

        model.fit(X_train_imp, y_train)
        y_pred = model.predict(X_test)
        eval_ret_imp = eval_metrics(y_test, y_pred, task_type)

        # train model on original data
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        eval_ret_ori = eval_metrics(y_test, y_pred, task_type)

        # merge results
        eval_ret_imp_all.append(eval_ret_imp)
        eval_ret_ori_all.append(eval_ret_ori)

    return {
        'clf_imp': summarize_dict(eval_ret_imp_all),
        'clf_ori': summarize_dict(eval_ret_ori_all)
    }


def summarize_dict(list_dict):
    # list_dict: list of dictionaries with same keys

    # get keys
    keys = list_dict[0].keys()

    # initialize summary dict
    sum_dict = {}
    for key in keys:
        sum_dict[key] = []

    # summarize
    for dict_ in list_dict:
        for key in keys:
            sum_dict[key].append(dict_[key])

    # average
    for key in keys:
        sum_dict[key] = np.mean(sum_dict[key])

    return sum_dict
