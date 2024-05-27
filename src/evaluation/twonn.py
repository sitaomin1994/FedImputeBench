from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import loguru

class TwoLayerNNBase(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNNBase, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)


class TwoNNRegressor(nn.Module):
    def __init__(
        self, hidden_size = 32, epochs=500, lr=0.001, batch_size=32, early_stopping_rounds=30, weight_decay=0.001
    ):
        super(TwoNNRegressor, self).__init__()

        self.hidden_size = hidden_size
        self.epoch = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.weight_decay = weight_decay

        self.network = None
        self.dataset = None
        self.dataloader = None
        self.criterion = nn.MSELoss()

    def _build_network(self, input_size):
        self.network = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_atribute_size),
            nn.ReLU(),
            nn.Linear(self.hidden_atribute_size, self.hidden_atribute_size),
            nn.ReLU(),
            nn.Linear(self.hidden_atribute_size, 1)  # Output is a single value for regression
        )

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Ensure y_tensor is 2D for MSE Loss

        # Prepare dataset for DataLoader
        if self.dataset is None:
            self.dataset = TensorDataset(X_tensor, y_tensor)
            self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Build the network on first call to fit
        if self.network is None:
            self._build_network(input_size = X.shape[1])

        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss = float('inf')
        patience = 0

        final_loss = 0
        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0
            for X_batch, y_batch in self.data_loader:
                outputs = self(X_batch)
                loss = self.criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.data_loader)
            final_loss = avg_epoch_loss

            if self.early_stopping_rounds:
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.early_stopping_rounds:
                        loguru.logger.debug(f"Early stopping at epoch {epoch}")
                        break

            loguru.logger.debug(f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_epoch_loss}')

        return {
            'loss': final_loss,
            'sample_size': len(X),
        }

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(torch.tensor(X, dtype=torch.float32))
        return outputs.numpy().flatten()

    def get_parameters(self):
        return deepcopy(self.network.state_dict())

    def update_parameters(self, new_params):
        self.network.load_state_dict(new_params)
        return self


# Example Usage:
# regressor = TwoNNRegressor(input_size=10, hidden_size=50)
# regressor.fit(X_train, y_train, epochs=50, lr=0.001, batch_size=64)
# predictions = regressor.predict(X_test)


class TwoNNClassifier(nn.Module):
    def __init__(
            self, hidden_size = 32, epochs=500, lr=0.001, batch_size=32, early_stopping_rounds=30, weight_decay=0.001
    ):
        super(TwoNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.epoch = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.weight_decay = weight_decay
        self.network = None
        self.dataset = None
        self.dataloader = None
        self.criterion = nn.CrossEntropyLoss()

    def _build_network(self, input_size, output_size):
        self.network = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, output_size)
        )

    def fit(self, X, y):

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Prepare dataset for DataLoader
        if self.dataset is None:
            self.dataset = TensorDataset(X_tensor, y_tensor)
            self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Determine the number of unique classes to set output size
        if self.network is None:
            unique_classes = np.unique(y)
            self._build_network(input_size = X.shape[1], output_size=len(unique_classes))

        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        best_loss = float('inf')
        patience = 0

        final_loss = 0
        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0
            for X_batch, y_batch in self.data_loader:
                outputs = self(X_batch)
                loss = self.criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.data_loader)
            final_loss = avg_epoch_loss

            if self.early_stopping_rounds:
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.early_stopping_rounds:
                        loguru.logger.debug(f"Early stopping at epoch {epoch}")
                        break

            loguru.logger.debug(f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_epoch_loss}')

        return {
            'loss': final_loss,
            'sample_size': len(X),
        }

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(torch.tensor(X, dtype=torch.float32))
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(torch.tensor(X, dtype=torch.float32))
            probabilities = nn.functional.softmax(outputs, dim=1)
        return probabilities.numpy()

    def get_parameters(self):
        return deepcopy(self.network.state_dict())

    def update_parameters(self, new_params):
        self.network.load_state_dict(new_params)
        return self

