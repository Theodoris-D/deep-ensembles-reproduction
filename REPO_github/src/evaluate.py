"""K-fold evaluation of Deep Ensembles on regression benchmarks (Section 3.3).

The paper evaluates each UCI dataset over 20 random train/test splits and
reports the mean +/- standard error of RMSE and NLL (Table 1). This module
provides :func:`run_kfold_experiment`, which runs that protocol for a given
dataset and returns the per-fold metrics.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .ensemble import DeepEnsemble
from .adversarial import per_dimension_epsilon
from .metrics import rmse, nll_original_scale


def run_kfold_experiment(X, y, *, n_folds=20, test_size=0.1, M=5,
                         hidden_dims=(50,), epochs=40, batch_size=32,
                         lr=0.1, adversarial=False, adv_fraction=0.01,
                         device="cpu", verbose=True):
    """Run the Table-1 protocol for one dataset.

    Parameters
    ----------
    X, y : np.ndarray
        Features (N, D) and targets (N, 1).
    n_folds : int
        Number of random train/test splits (20 in the paper; reduce for speed).
    test_size : float
        Fraction held out for testing in each split (0.1 in the paper).
    M : int
        Ensemble size.
    hidden_dims : tuple[int]
        Hidden layer sizes (50 for small datasets, 100 for large ones).
    epochs, batch_size, lr : training hyper-parameters.
    adversarial : bool
        Whether to use adversarial training.
    adv_fraction : float
        Epsilon as a fraction of each feature's range.

    Returns
    -------
    dict
        {'rmse': [...], 'nll': [...]} with one entry per fold.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    results = {"rmse": [], "nll": []}

    for fold in range(n_folds):
        # Each fold is a different random split (seed = fold).
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=fold)

        # Standardise using statistics from the training fold only.
        x_mean, x_std = X_tr.mean(0), X_tr.std(0) + 1e-8
        y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-8

        Xs_tr = (X_tr - x_mean) / x_std
        Xs_te = (X_te - x_mean) / x_std
        ys_tr = (y_tr - y_mean) / y_std
        ys_te = (y_te - y_mean) / y_std

        epsilon = 0.01
        if adversarial:
            epsilon = per_dimension_epsilon(torch.tensor(Xs_tr),
                                            fraction=adv_fraction)

        ensemble = DeepEnsemble(input_dim=X.shape[1], hidden_dims=hidden_dims,
                                M=M, adversarial=adversarial, epsilon=epsilon)
        ensemble.fit(Xs_tr, ys_tr, epochs=epochs, batch_size=batch_size,
                     lr=lr, device=device, base_seed=1000 * fold)

        mean_s, var_s = ensemble.predict(Xs_te, device=device)

        # RMSE on the original target scale.
        mean_orig = mean_s.ravel() * y_std + y_mean
        fold_rmse = rmse(mean_orig, y_te)

        # NLL on the original target scale (change-of-variables correction).
        fold_nll = nll_original_scale(mean_s, var_s, ys_te, y_std)

        results["rmse"].append(fold_rmse)
        results["nll"].append(fold_nll)

        if verbose:
            print(f"  fold {fold + 1:2d}/{n_folds}  "
                  f"RMSE={fold_rmse:7.3f}  NLL={fold_nll:7.3f}")

    return results


def summarise(results):
    """Return (mean, standard error) for each metric in a results dict."""
    summary = {}
    for metric, values in results.items():
        values = np.asarray(values)
        mean = values.mean()
        # The paper reports the standard error of the mean.
        stderr = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 \
            else 0.0
        summary[metric] = (float(mean), float(stderr))
    return summary
