from abc import ABC, abstractmethod
import torch
from typing import Tuple
import torch.optim.lr_scheduler as lr_scheduler

class StrategyClient(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit_local_model(
            self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, params: dict
    ) -> Tuple[torch.nn.Module, dict]:
        pass


def fit_local_model_base(
        model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, params: dict
) -> Tuple[torch.nn.Module, dict]:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        local_epochs = params['local_epoch']
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        optimizer_name = params['optimizer']
        model_converge_tol = params['model_converge_tol']
        model_converge_patience = params['model_converge_patience']
        schedule_step = params['schedule_step_size']
        schedule_gamma = params['schedule_gamma']
        schedule_last_epoch = params['schedule_last_epoch']
    except KeyError as e:
        raise ValueError(f"Parameter {str(e)} not found in params")

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")

    scheduler = lr_scheduler.StepLR(optimizer, step_size=schedule_step, gamma=schedule_gamma)
    # Initialization - model to device and copy global model parameters
    model.to(DEVICE)
    model.train()

    total_loss, total_iters = 0, 0
    early_stopping_triggered = False
    best_loss = float('inf')
    patience_counter = 0
    for ep in range(local_epochs):
        loss_epoch = 0
        for i, inputs in enumerate(dataloader):
            # training step
            inputs = tuple(input_.to(DEVICE) for input_ in inputs)
            optimizer.zero_grad()
            loss, train_res_dict = model.compute_loss(inputs)
            loss.backward()
            loss_epoch += loss.item()
            # update parameters
            optimizer.step()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        final_loss = loss_epoch / len(dataloader)
        # if ep % 20 == 0:
        #     print(f"Epoch {ep} - Loss: {final_loss}")
        #
        # if final_loss < best_loss - model_converge_tol:
        #     best_loss = final_loss
        #     patience_counter = 0  # Reset patience counter
        # else:
        #     patience_counter += 1  # Increment patience counter

            # Early stopping check
        # if patience_counter >= model_converge_patience:
        #     print(f"Early stopping triggered after {ep} epochs. Loss: {final_loss}")
        #     early_stopping_triggered = True
        #     break

        scheduler.step()

        total_loss += final_loss
        total_iters += 1

    # if not early_stopping_triggered:
    #     print(f"Training completed without early stopping. Final loss: {final_loss}")

    model.to('cpu')
    final_loss = total_loss / total_iters

    return model, {'loss': final_loss, 'sample_size': len(dataloader.dataset)}
