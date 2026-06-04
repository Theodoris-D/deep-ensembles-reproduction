"""Deep Ensembles reproduction package.

Reproduction of Lakshminarayanan, Pritzel & Blundell, "Simple and Scalable
Predictive Uncertainty Estimation using Deep Ensembles" (NeurIPS 2017).
"""

from .models import GaussianMLP, PlainMLP
from .losses import gaussian_nll, mse_loss
from .adversarial import fgsm_perturbation, per_dimension_epsilon
from .ensemble import DeepEnsemble
from .baselines import MSEEnsemble
from .metrics import rmse, gaussian_nll_numpy, nll_original_scale
from .evaluate import run_kfold_experiment, summarise
from .data import make_toy_dataset, toy_ground_truth, load_uci_dataset

__all__ = [
    "GaussianMLP",
    "PlainMLP",
    "gaussian_nll",
    "mse_loss",
    "fgsm_perturbation",
    "per_dimension_epsilon",
    "DeepEnsemble",
    "MSEEnsemble",
    "rmse",
    "gaussian_nll_numpy",
    "nll_original_scale",
    "run_kfold_experiment",
    "summarise",
    "make_toy_dataset",
    "toy_ground_truth",
    "load_uci_dataset",
]
