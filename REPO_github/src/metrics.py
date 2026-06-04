"""Evaluation metrics for regression (Section 3.1).

The paper evaluates regression with two metrics:

* **RMSE**: root mean squared error of the predictive mean.
* **NLL** : negative log-likelihood of the Gaussian predictive
  distribution; this is the metric that actually depends on the predicted
  *uncertainty* and is a proper scoring rule.

Both are reported on the *original* target scale in Table 1. When the model is
trained on standardised targets, the predictive mean/variance must be mapped
back. For the NLL this is a change of variables: if y = s * y_std + m, then

    log p(y) = log p_std(y_std) - log(s)   ==>   NLL_orig = NLL_std + log(s).
"""

import math
import numpy as np


def rmse(mean_pred, y_true):
    """Root mean squared error between the predicted mean and the targets.

    Both arguments must be on the same (original) scale.
    """
    mean_pred = np.asarray(mean_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    return float(np.sqrt(np.mean((mean_pred - y_true) ** 2)))


def gaussian_nll_numpy(mean_pred, var_pred, y_true):
    """Mean Gaussian NLL on whatever scale the inputs are given in.

    Parameters
    ----------
    mean_pred, var_pred : array-like
        Predicted mean and variance.
    y_true : array-like
        Observed targets.
    """
    mean_pred = np.asarray(mean_pred).ravel()
    var_pred = np.asarray(var_pred).ravel()
    y_true = np.asarray(y_true).ravel()

    const = 0.5 * math.log(2.0 * math.pi)
    nll = 0.5 * np.log(var_pred) \
        + 0.5 * (y_true - mean_pred) ** 2 / var_pred \
        + const
    return float(np.mean(nll))


def nll_original_scale(mean_std, var_std, y_true_std, y_std):
    """NLL on the original target scale, given predictions on the std scale.

    Parameters
    ----------
    mean_std, var_std : array-like
        Predicted mean and variance, in standardised target units.
    y_true_std : array-like
        Standardised true targets.
    y_std : float
        Standard deviation used to standardise the targets.

    Returns
    -------
    float
        The mean NLL expressed on the original target scale, directly
        comparable to the values in Table 1 of the paper.
    """
    nll_std = gaussian_nll_numpy(mean_std, var_std, y_true_std)
    return nll_std + math.log(y_std)
