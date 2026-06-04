"""Neural network architectures for Deep Ensembles.

Implements the probabilistic networks described in Lakshminarayanan et al.,
"Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
(NeurIPS 2017).

For regression, the network outputs *two* values per input (Section 2.2.1):
the predicted mean mu(x) and the predicted variance sigma^2(x). The variance
is constrained to be positive via a softplus, with a small floor added for
numerical stability (footnote 2 of the paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMLP(nn.Module):
    """MLP that parametrises a heteroscedastic Gaussian predictive distribution.

    The final layer produces two outputs: the mean and the (pre-softplus)
    variance. We map the second output through ``softplus`` and add
    ``min_variance`` so that sigma^2(x) > 0 everywhere (paper footnote 2).

    Parameters
    ----------
    input_dim : int
        Dimensionality D of the input features.
    hidden_dims : list[int]
        Sizes of the hidden layers. The toy experiment uses a single
        hidden layer; the UCI experiments use one layer of 50 or 100 units.
    min_variance : float
        Variance floor for numerical stability (1e-6 in the paper).
    """

    def __init__(self, input_dim, hidden_dims=(100,), min_variance=1e-6):
        super().__init__()
        self.min_variance = min_variance

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.backbone = nn.Sequential(*layers)
        # One head for the mean, one for the (raw) variance.
        self.head = nn.Linear(prev, 2)

    def forward(self, x):
        """Return (mean, variance), each of shape (batch, 1)."""
        features = self.backbone(x)
        out = self.head(features)
        mean = out[:, 0:1]
        raw_var = out[:, 1:2]
        # Softplus guarantees positivity; the floor avoids division by ~0
        # in the NLL loss (see losses.gaussian_nll).
        variance = F.softplus(raw_var) + self.min_variance
        return mean, variance


class PlainMLP(nn.Module):
    """Standard regression MLP that outputs a single value (the mean only).

    Used purely as a baseline: an ensemble of these trained with MSE, whose
    *empirical* variance is used as an uncertainty proxy. The paper shows in
    Figure 1 (left panel) that this is inferior to learning the variance.
    """

    def __init__(self, input_dim, hidden_dims=(100,)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
