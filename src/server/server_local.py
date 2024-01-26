from .agg_strategy.fedavg import fedavg


class LocalServer:

    def __init__(self, clients, server_config: dict):
        self.clients = clients

    def run_imputation(self, imp_params):
        """
        Run imputation for all clients
        """
        history, eval_history = [], []

        # federated learning
        epoch_ret = {}
        # local training
        for client in self.clients:
            client_training_ret_dict = client.local_train(imp_params)
            epoch_ret[client.client_id] = client_training_ret_dict
            history.append(epoch_ret)

        # imputation
        for client in self.clients:
            client.local_imputation(imp_params)

        return history, eval_history, self.clients
