import torch.nn as nn
import torch


class Ridge(nn.Module):

    def __init__(self, input_dim, output_dim, alpha=0.1):
        super(Ridge, self).__init__()
        self.alpha = alpha
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    def loss(self, y_pred, y_true):
        return torch.nn.MSELoss(y_pred, y_true) + self.alpha * torch.norm(self.linear.weight, 2)


class Logistic(nn.Module):

    def __init__(self, input_dim, output_dim, C=0.1):
        super(Logistic, self).__init__()
        self.C = C
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    def loss(self, y_pred, y_true):
        return torch.nn.CrossEntropyLoss(y_pred, y_true) + self.C * torch.norm(self.linear.weight, 2)
