from typing import List, Tuple, Dict

import torch.nn as nn
import torch


class RidgeRegression(nn.Module):

    def __init__(self, input_dim):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

    def compute_loss(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        x, y_true = inputs
        y_pred = self.forward(x)
        return torch.nn.MSELoss()(y_pred, y_true), {}


class Logistic(nn.Module):

    def __init__(self, input_dim, C=0.1):
        super(Logistic, self).__init__()
        self.C = C
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def compute_loss(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        x, y_true = inputs
        y_pred = self.forward(x)
        loss = nn.BCELoss()(y_pred, y_true.view_as(y_pred))  # Ensure y_true is the same shape as y_pred
        return loss, {}
