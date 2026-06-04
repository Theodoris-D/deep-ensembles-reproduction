"""Adversarial training to smooth the predictive distributions (Section 2.3).

Goodfellow et al.'s fast gradient sign method (FGSM) generates an adversarial
example by perturbing the input along the sign of the loss gradient:

    x' = x + epsilon * sign( grad_x  loss(theta, x, y) )

The paper uses this as an *optional* ingredient: the adversarial examples
augment the training set, which increases the likelihood of the target in an
epsilon-neighbourhood of each training point and thus smooths the predictive
distribution. Following Section 3.1, epsilon is set per-dimension to a small
fraction (1%) of that input dimension's range.
"""

import torch


def fgsm_perturbation(x, loss, epsilon):
    """Compute the FGSM adversarial perturbation for inputs ``x``.

    Parameters
    ----------
    x : torch.Tensor
        Input batch. Must have ``requires_grad=True`` and be the same tensor
        used to compute ``loss``.
    loss : torch.Tensor
        Scalar loss already computed from ``x``.
    epsilon : float or torch.Tensor
        Perturbation size. A scalar applies the same size to every dimension;
        a tensor of shape (input_dim,) applies a per-dimension size.

    Returns
    -------
    torch.Tensor
        The adversarial input x' = x + epsilon * sign(grad), detached from
        the graph (it is treated as a fresh training example).
    """
    grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
    x_adv = x + epsilon * grad.sign()
    return x_adv.detach()


def per_dimension_epsilon(x_train, fraction=0.01):
    """Return a per-dimension epsilon equal to ``fraction`` of each range.

    The paper notes that a single epsilon for all dimensions is unsatisfying
    when input dimensions have different scales (Section 3.1), so epsilon is
    scaled by the range (max - min) of each feature in the training data.
    """
    ranges = x_train.max(dim=0).values - x_train.min(dim=0).values
    return fraction * ranges
