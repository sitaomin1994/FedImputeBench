from .agg_strategy.fedavg import fedavg


class FedAvgServer:

    def __init__(self, clients, server_config: dict, layer_keys = None):
        self.clients = clients
        self.layer_keys = layer_keys

    def run_imputation(self, imp_params):
        """
        Run imputation for all clients
        """
        history, eval_history = [], []

        # federated learning
        global_epochs = imp_params['global_epochs']
        for epoch in range(global_epochs):
            if epoch % imp_params['verbose_interval'] == 0:
                print(f"Imputation round {epoch}")

            # if epoch % imp_params['eval_interval'] == 0:
            #     # evaluation
            #     eval_ret = {}
            #     for client in self.clients:
            #         eval_ret[client.client_id] = client.local_evaluate({})
            #     eval_history.append((epoch, eval_ret))

            epoch_ret = {}
            # local training
            for client in self.clients:
                init = True if epoch == 0 else False
                client_training_ret_dict = client.local_train(imp_params, init=init)
                epoch_ret[client.client_id] = client_training_ret_dict
            history.append(epoch_ret)

            # aggregate global model
            averaged_model_weights = fedavg(self.clients, layer_keys=self.layer_keys)

            # update global model
            for client in self.clients:
                local_weights = client.imp_model.model.state_dict()
                local_weights.update(averaged_model_weights)
                client.imp_model.model.load_state_dict(local_weights)

        # imputation
        for client in self.clients:
            client.local_imputation(imp_params)

        return history, eval_history, self.clients
