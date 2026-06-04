"""Deep Ensembles for regression (Algorithm 1 of the paper).

A :class:`DeepEnsemble` trains ``M`` :class:`GaussianMLP` networks
independently. Each network is randomly initialised and sees the data in a
different (shuffled) order, the paper found that this randomisation alone is
enough, and that bagging actually *hurt* performance (Section 2.4).

At prediction time the ensemble is treated as a uniformly-weighted mixture of
Gaussians. For convenience this mixture is approximated by a single Gaussian
whose mean and variance match the mixture's (formulas at the end of
Section 2.4):

    mu_*(x)      = (1/M) * sum_m  mu_m(x)
    sigma_*^2(x) = (1/M) * sum_m ( sigma_m^2(x) + mu_m(x)^2 )  -  mu_*(x)^2
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import GaussianMLP
from .losses import gaussian_nll
from .adversarial import fgsm_perturbation


def _train_single_network(model, x_train, y_train, *, epochs, batch_size,
                           lr, adversarial, epsilon, device, seed):
    """Train one GaussianMLP with NLL (and optionally adversarial training)."""
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            if adversarial:
                # Need gradients w.r.t. the input to build x'.
                xb = xb.clone().detach().requires_grad_(True)

            mean, var = model(xb)
            loss = gaussian_nll(mean, var, yb)

            if adversarial:
                # Equation in Algorithm 1, line 5-6: augment the loss with
                # the loss evaluated at the adversarial example x'.
                x_adv = fgsm_perturbation(xb, loss, epsilon)
                mean_adv, var_adv = model(x_adv)
                loss = loss + gaussian_nll(mean_adv, var_adv, yb)

            loss.backward()
            optimizer.step()

    model.eval()
    return model


class DeepEnsemble:
    """An ensemble of ``M`` GaussianMLPs for probabilistic regression.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the inputs.
    hidden_dims : tuple[int]
        Hidden layer sizes passed to every member network.
    M : int
        Number of networks in the ensemble (paper default: 5).
    adversarial : bool
        Whether to use adversarial training (Algorithm 1 is optional here).
    epsilon : float or torch.Tensor
        FGSM perturbation size. Ignored if ``adversarial`` is False.
    min_variance : float
        Variance floor passed to each GaussianMLP.
    """

    def __init__(self, input_dim, hidden_dims=(100,), M=5,
                 adversarial=False, epsilon=0.01, min_variance=1e-6):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.M = M
        self.adversarial = adversarial
        self.epsilon = epsilon
        self.min_variance = min_variance
        self.members = []

    def fit(self, x_train, y_train, *, epochs=40, batch_size=100, lr=0.1,
            device="cpu", base_seed=0, verbose=False):
        """Train the M member networks independently (Algorithm 1).

        Each member uses a different random seed, which controls both the
        weight initialisation and the data shuffling order.
        """
        x_train = torch.as_tensor(x_train, dtype=torch.float32)
        y_train = torch.as_tensor(y_train, dtype=torch.float32)

        self.members = []
        for m in range(self.M):
            model = GaussianMLP(self.input_dim, self.hidden_dims,
                                min_variance=self.min_variance)
            model = _train_single_network(
                model, x_train, y_train,
                epochs=epochs, batch_size=batch_size, lr=lr,
                adversarial=self.adversarial, epsilon=self.epsilon,
                device=device, seed=base_seed + m,
            )
            self.members.append(model)
            if verbose:
                print(f"  trained member {m + 1}/{self.M}")
        return self

    @torch.no_grad()
    def predict(self, x, device="cpu"):
        """Return the ensemble predictive mean and variance.

        Returns
        -------
        mean : np.ndarray, shape (N, 1)
            mu_*(x), the mixture mean.
        variance : np.ndarray, shape (N, 1)
            sigma_*^2(x), the mixture variance (Section 2.4 formula).
        """
        x = torch.as_tensor(x, dtype=torch.float32).to(device)

        means, variances = [], []
        for model in self.members:
            mu, var = model(x)
            means.append(mu)
            variances.append(var)

        means = torch.stack(means)        # (M, N, 1)
        variances = torch.stack(variances)  # (M, N, 1)

        mean_star = means.mean(dim=0)
        # Var of a mixture = E[var] + E[mu^2] - (E[mu])^2.
        var_star = (variances + means ** 2).mean(dim=0) - mean_star ** 2

        return mean_star.cpu().numpy(), var_star.cpu().numpy()

    @torch.no_grad()
    def member_predictions(self, x, device="cpu"):
        """Return per-member means and variances, shape (M, N, 1) each.

        Useful for plotting individual networks (e.g. Figure 1).
        """
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        means, variances = [], []
        for model in self.members:
            mu, var = model(x)
            means.append(mu.cpu().numpy())
            variances.append(var.cpu().numpy())
        return np.stack(means), np.stack(variances)
