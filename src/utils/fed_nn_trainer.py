from typing import Tuple
import torch
import numpy as np
from tqdm import trange, tqdm

from src.fed_strategy.fed_strategy_client import StrategyClient
from src.fed_strategy.fed_strategy_client.utils import trainable_params
from src.imputation.base import BaseNNImputer
import delu
import src.utils.nn_utils as nn_utils
from torch.cuda.amp import autocast, GradScaler


DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'


def fit_fed_nn_model(
        imputer: BaseNNImputer, training_params: dict, fed_strategy: StrategyClient,
        X_train_imp: np.ndarray, y_train: np.ndarray, X_train_mask: np.ndarray
) -> Tuple[torch.nn.Module, dict]:
    ######################################################################################
    # training params
    try:
        local_epochs = training_params['local_epoch']
    except KeyError as e:
        raise ValueError(f"Parameter {str(e)} not found in params")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ######################################################################################
    # model and dataloader
    model, train_dataloader = imputer.configure_model(training_params, X_train_imp, y_train, X_train_mask)

    # optimizer and scheduler
    optimizers, lr_schedulers = imputer.configure_optimizer(training_params, model)
    #scaler = GradScaler()
    model.to(DEVICE)

    ######################################################################################
    # pre-training setup
    fed_strategy.pre_training_setup(model, training_params)

    ######################################################################################
    # training loop
    total_loss, total_iters = 0, 0

    # for ep in trange(local_epochs, desc='Local Epoch', colour='blue'):
    for ep in range(local_epochs):

        #################################################################################
        # training one epoch
        losses_epoch, ep_iters = [0 for _ in range(len(optimizers))], 0
        for batch_idx, batch in enumerate(train_dataloader):
            for optimizer_idx, optimizer in enumerate(optimizers):
                #########################################################################
                # training step
                model.train()
                optimizer.zero_grad()
                #with autocast(dtype=torch.float16):
                loss, res = model.train_step(batch, batch_idx, optimizers, optimizer_idx=optimizer)

                losses_epoch[optimizer_idx] += loss
                #print('===================================================================')
                #print(torch.norm(model.state_dict()['encoder.hidden_layers.model.0.weight']))
                ########################################################################
                # fed updates
                fed_strategy.fed_updates(model)

                #########################################################################
                # backpropagation
                optimizer.step()
                # scaler.step(optimizer)
                # scaler.update()
                #print(torch.norm(model.state_dict()['encoder.0.weight']))
                #print(torch.norm(model.state_dict()['encoder.hidden_layers.model.0.weight']))
                #print('===================================================================')

            ep_iters += 1

        #################################################################################
        # epoch end - update loss, early stopping, evaluation, garbage collection etc.
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        losses_epoch = np.array(losses_epoch) / len(train_dataloader)
        epoch_loss = losses_epoch.mean()

        # update lr scheduler
        # for scheduler in lr_schedulers:
        #     scheduler.step()

        total_loss += epoch_loss  # average loss
        total_iters += 1

    #########################################################################################
    # post-training setup
    fed_strategy.post_training_setup(model)

    model.to('cpu')
    final_loss = total_loss / total_iters

    return model, {'loss': final_loss, 'sample_size': len(train_dataloader.dataset)}



