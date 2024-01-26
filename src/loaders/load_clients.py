import src.client.client_jm as client_jm


def setup_clients(
    clients_train_data_list, clients_train_data_ms_list, test_data, data_config, imp_models, seeds
):
    num_clients = len(clients_train_data_list)
    clients = []
    for i in range(num_clients):
        clients.append(client_jm.Client(client_id=i, seed=seeds[i]))

    for i, client in enumerate(clients):
        client.setup_data(
            train_data=clients_train_data_list[i], test_data=test_data,
            X_train_ms=clients_train_data_ms_list[i][:, :-1], data_config=data_config
        )
        client.setup_imputation_model(imp_models[i])

    return clients
