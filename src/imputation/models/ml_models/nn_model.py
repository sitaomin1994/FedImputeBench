from typing import List, Dict, Tuple

import torch.nn as nn
import torch

from emf.reproduce_utils import set_seed
from src.imputation.models.common_blocks import weights_init


class TwoNNRegressor(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super(TwoNNRegressor, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def init(self, seed):
        set_seed(seed)
        self.linear.apply(weights_init)

    def forward(self, x):
        return self.linear(x)

    def compute_loss(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:

        x, y_true = inputs
        y_pred = self.forward(x)
        return torch.nn.MSELoss()(y_pred, y_true), {}


class TwoNNClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super(TwoNNClassifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def init(self, seed):
        set_seed(seed)
        self.linear.apply(weights_init)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def compute_loss(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        x, y_true = inputs
        y_pred = self.forward(x)
        loss = nn.BCELoss()(y_pred, y_true.view_as(y_pred))  # Ensure y_true is the same shape as y_pred
        return loss, {}

