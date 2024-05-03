from typing import List, Dict, Tuple

import torch.nn as nn
import torch


class TwoNNRegressor(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(TwoNNRegressor, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

    def compute_loss(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:

        x, y_true = inputs
        y_pred = self.forward(x)
        return torch.nn.MSELoss()(y_pred, y_true), {}


class TwoNNClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(TwoNNClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear2(torch.relu(self.linear1(x))))

    def compute_loss(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        x, y_true = inputs
        y_pred = self.forward(x)
        loss = nn.BCELoss()(y_pred, y_true.view_as(y_pred))  # Ensure y_true is the same shape as y_pred
        l2_reg = self.C * torch.norm(self.linear.weight, 2)
        return loss + l2_reg, {}

