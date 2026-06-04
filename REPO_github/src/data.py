"""Datasets used in the reproduction.

* :func:`make_toy_dataset` reproduces the 1-D toy regression task of
  Section 3.2 (originally from Hernandez-Lobato & Adams, 2015): 20 points
  drawn as y = x^3 + epsilon with epsilon ~ N(0, 3^2).

* :func:`load_uci_dataset` downloads / loads the small UCI regression
  benchmarks used in Section 3.3 and Table 1.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Toy 1-D regression (Section 3.2, Figure 1)
# ---------------------------------------------------------------------------
def make_toy_dataset(n_points=20, noise_std=3.0, x_range=(-4.0, 4.0), seed=1):
    """Generate the 1-D toy dataset y = x^3 + N(0, noise_std^2).

    Parameters
    ----------
    n_points : int
        Number of training points (20 in the paper).
    noise_std : float
        Standard deviation of the additive Gaussian noise (3.0 in the paper).
    x_range : tuple(float, float)
        Range from which the training inputs are drawn uniformly.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    x_train, y_train : np.ndarray, shape (n_points, 1)
    """
    rng = np.random.RandomState(seed)
    x = rng.uniform(x_range[0], x_range[1], size=(n_points, 1))
    noise = rng.normal(0.0, noise_std, size=(n_points, 1))
    y = x ** 3 + noise
    return x.astype(np.float32), y.astype(np.float32)


def toy_ground_truth(x):
    """The noise-free target function y = x^3 (the blue curve in Figure 1)."""
    return x ** 3


# ---------------------------------------------------------------------------
# UCI regression benchmarks (Section 3.3, Table 1)
# ---------------------------------------------------------------------------
# Each entry: a loader function from scikit-learn or a small fetch helper.
# We keep this minimal: only datasets that load without manual downloads.

def load_uci_dataset(name):
    """Load a UCI regression dataset as (X, y) float arrays.

    Supported names (others can be added by the student):
        'diabetes': scikit-learn built-in, no download (offline demo)
        'boston'  : Boston housing (via a static copy)
        'concrete': Concrete compressive strength
        'energy'  : Energy efficiency
        'wine'    : Wine quality (red)
        'yacht'   : Yacht hydrodynamics

    The function returns numpy arrays; standardisation is left to the caller
    so that it can be fit on the training fold only.
    """
    import pandas as pd

    urls = {
        "concrete": ("https://archive.ics.uci.edu/ml/machine-learning-"
                     "databases/concrete/compressive/Concrete_Data.xls"),
        "yacht": ("https://archive.ics.uci.edu/ml/machine-learning-"
                  "databases/00243/yacht_hydrodynamics.data"),
        "energy": ("https://archive.ics.uci.edu/ml/machine-learning-"
                   "databases/00242/ENB2012_data.xlsx"),
        "wine": ("https://archive.ics.uci.edu/ml/machine-learning-"
                 "databases/wine-quality/winequality-red.csv"),
    }

    name = name.lower()

    if name == "diabetes":
        # Ships with scikit-learn; no download needed. Handy as an offline
        # demonstration dataset for the k-fold pipeline.
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X, y = data.data, data.target

    elif name == "boston":
        # The Boston dataset was removed from scikit-learn; load from the
        # original CMU StatLib copy.
        url = "http://lib.stat.cmu.edu/datasets/boston"
        raw = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
        data = np.hstack([raw.values[::2, :], raw.values[1::2, :2]])
        X, y = data[:, :-1], data[:, -1]

    elif name == "concrete":
        df = pd.read_excel(urls["concrete"])
        X, y = df.values[:, :-1], df.values[:, -1]

    elif name == "yacht":
        df = pd.read_csv(urls["yacht"], sep=r"\s+", header=None)
        X, y = df.values[:, :-1], df.values[:, -1]

    elif name == "energy":
        df = pd.read_excel(urls["energy"])
        # Two possible targets (Y1 heating load); use the first as in [24].
        X, y = df.values[:, :-2], df.values[:, -2]

    elif name == "wine":
        df = pd.read_csv(urls["wine"], sep=";")
        X, y = df.values[:, :-1], df.values[:, -1]

    else:
        raise ValueError(
            f"Dataset '{name}' not available in this minimal loader. "
            "Add it following the pattern above."
        )

    return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)
