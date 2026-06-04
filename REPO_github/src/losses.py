"""Training criteria for Deep Ensembles.

The paper argues for training probabilistic networks with a *proper scoring
rule*. For regression with a Gaussian predictive distribution, the negative
log-likelihood (NLL) is such a rule. It corresponds to Equation (1):

    -log p_theta(y | x) = log(sigma^2(x)) / 2
                          + (y - mu(x))^2 / (2 * sigma^2(x))
                          + constant
"""

import math
import torch


def gaussian_nll(mean, variance, target):
    """Negative log-likelihood of a heteroscedastic Gaussian (Equation 1).

    Parameters
    ----------
    mean, variance : torch.Tensor
        Predicted mean mu(x) and variance sigma^2(x), shape (batch, 1).
    target : torch.Tensor
        Observed targets y, shape (batch, 1).

    Returns
    -------
    torch.Tensor
        The mean NLL over the batch (a scalar).
    """
    # The "+ constant" term is 0.5 * log(2*pi); we keep it so that the
    # reported NLL matches the values in Table 1 of the paper.
    const = 0.5 * math.log(2.0 * math.pi)
    nll = 0.5 * torch.log(variance) \
        + 0.5 * (target - mean) ** 2 / variance \
        + const
    return nll.mean()


def mse_loss(prediction, target):
    """Plain mean squared error, used only for the MSE baseline."""
    return ((prediction - target) ** 2).mean()
