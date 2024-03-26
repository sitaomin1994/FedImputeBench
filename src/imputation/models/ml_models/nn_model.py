import torch.nn as nn
import torch


class TwoNNRegressor(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoNNRegressor, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

    def loss(self, y_pred, y_true):
        return torch.nn.MSELoss(y_pred, y_true)


class TwoNNClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoNNClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))  # TODO: check if this is the right way to do it

    def loss(self, y_pred, y_true):
        return torch.nn.CrossEntropyLoss(y_pred, y_true)

