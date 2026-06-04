"""Baseline methods for comparison.

The paper's Figure 1 (left panel) shows the common heuristic of training an
ensemble of plain networks with MSE and using the *empirical* variance of
their point predictions as an uncertainty estimate. The paper demonstrates
that this is inferior to learning the variance with the NLL loss.

This module isolates that baseline so the main :mod:`ensemble` module stays
focused on the paper's actual method.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import PlainMLP
from .losses import mse_loss


class MSEEnsemble:
    """Ensemble of plain MLPs trained with MSE (uncertainty = empirical var)."""

    def __init__(self, input_dim, hidden_dims=(100,), M=5):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.M = M
        self.members = []

    def fit(self, x_train, y_train, *, epochs=40, batch_size=100, lr=0.1,
            device="cpu", base_seed=0):
        x_train = torch.as_tensor(x_train, dtype=torch.float32)
        y_train = torch.as_tensor(y_train, dtype=torch.float32)

        self.members = []
        for m in range(self.M):
            torch.manual_seed(base_seed + m)
            model = PlainMLP(self.input_dim, self.hidden_dims).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loader = DataLoader(TensorDataset(x_train, y_train),
                                batch_size=batch_size, shuffle=True)
            model.train()
            for _ in range(epochs):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = mse_loss(model(xb.to(device)), yb.to(device))
                    loss.backward()
                    optimizer.step()
            model.eval()
            self.members.append(model)
        return self

    @torch.no_grad()
    def predict(self, x, device="cpu"):
        """Return the empirical mean and variance of the members' outputs."""
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        preds = torch.stack([model(x) for model in self.members])  # (M,N,1)
        mean = preds.mean(dim=0)
        # Empirical variance across the M point predictions, the heuristic
        # the paper compares against.
        variance = preds.var(dim=0, unbiased=False)
        return mean.cpu().numpy(), variance.cpu().numpy()
