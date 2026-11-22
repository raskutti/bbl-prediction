"""Unit tests for the NB2 likelihood used by the powerplay model."""

import numpy as np
from scipy.optimize import check_grad

from bbl.negbin import logpmf, nb_nll


def _toy_problem(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.integers(0, 2, size=n)])
    beta = np.array([3.7, 0.15, -0.1])
    mu = np.exp(X @ beta)
    alpha = 0.04
    r = 1 / alpha
    y = rng.negative_binomial(r, r / (r + mu)).astype(float)
    w = rng.uniform(0.5, 1.5, size=n)
    lam = np.array([0.0, 2.0, 2.0])
    return X, y, w, lam


def test_gradient_matches_finite_differences():
    X, y, w, lam = _toy_problem()
    f = lambda p: nb_nll(p, X, y, w, lam)[0]
    g = lambda p: nb_nll(p, X, y, w, lam)[1]
    for p0 in (np.array([3.5, 0.0, 0.0, -3.0]), np.array([3.7, 0.2, -0.1, -3.2])):
        assert check_grad(f, g, p0) < 1e-3 * max(1.0, np.linalg.norm(g(p0)))


def test_optimiser_recovers_parameters():
    from scipy.optimize import minimize

    X, y, w, lam = _toy_problem(n=4000)
    sol = minimize(
        nb_nll, np.append(np.zeros(3), 0.0), args=(X, y, w, np.zeros(3)), jac=True, method="L-BFGS-B"
    ).x
    assert np.allclose(sol[:3], [3.7, 0.15, -0.1], atol=0.05)
    assert abs(np.exp(sol[3]) - 0.04) < 0.015


def test_pmf_is_a_distribution_with_the_stated_moments():
    mu, alpha = 44.0, 0.04
    grid = np.arange(0, 600)
    p = np.exp(logpmf(grid, mu, alpha))
    assert abs(p.sum() - 1) < 1e-9
    mean = (grid * p).sum()
    var = ((grid - mean) ** 2 * p).sum()
    assert abs(mean - mu) < 1e-6
    assert abs(var - (mu + alpha * mu**2)) < 1e-4
