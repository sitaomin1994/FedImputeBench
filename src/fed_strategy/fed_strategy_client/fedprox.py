from src.fed_strategy.fed_strategy_client.base_strategy import StrategyClient
import torch
from typing import Tuple
from .utils import trainable_params

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FedProxStrategyClient(StrategyClient):

    def __init__(self, strategy_params: dict):
        self.strategy_params = strategy_params
        super().__init__('fedprox')
        self.mu = strategy_params.get('mu', 0.01)

    def fit_local_model(
            self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, params: dict
    ) -> Tuple[torch.nn.Module, dict]:

        try:
            local_epochs = params['local_epoch']
            learning_rate = params['learning_rate']
            weight_decay = params['weight_decay']
            optimizer_name = params['optimizer']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")

        # Initialization - model to device and copy global model parameters
        model.to(DEVICE)
        global_model_params = trainable_params(model)
        model.train()
        dataloader.train()

        total_loss, total_iters = 0, 0
        for ep in range(local_epochs):
            loss_epoch = 0
            for i, inputs in enumerate(dataloader):

                # training step
                inputs = tuple(input_.to(DEVICE) for input_ in inputs)
                optimizer.zero_grad()
                loss, train_res_dict = model.compute_loss(inputs)
                loss.backward()
                loss_epoch += loss.item().detach().cpu().numpy()

                # Add FedProx regularization term
                for w, w_t in zip(trainable_params(model), global_model_params):
                    w.grad.data += self.mu * (w.data - w_t.data)

                # update parameters
                optimizer.step()

            final_loss = loss_epoch / len(dataloader)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            total_loss += final_loss
            total_iters += 1

        model.to('cpu')
        final_loss = total_loss / total_iters

        return model, {'loss': final_loss, 'sample_size': len(dataloader.dataset)}
