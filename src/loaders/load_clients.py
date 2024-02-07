from src.client import (
    Client,
    ICEClient,
    JMClient
)


def load_client(
        client_type, client_id, train_data, test_data, X_train_ms, data_config, imp_model, client_config, seed,
) -> Client:
    if client_type == 'ice':
        return ICEClient(
            client_id, train_data, test_data, X_train_ms, data_config, imp_model, client_config, seed
        )
    elif client_type == 'jm':
        return JMClient(
            client_id, train_data, test_data, X_train_ms, data_config, imp_model, client_config, seed
        )
    else:
        raise ValueError(f'client type {client_type} is not supported')
