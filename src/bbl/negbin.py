"""Negative-binomial (NB2) regression likelihood with ridge penalty and observation weights.

Parameterisation: y ~ NB(mean=mu, dispersion alpha) with Var(y) = mu + alpha * mu**2.
`par = [beta..., log_alpha]`, mu = exp(X @ beta). The optimiser works with la = log(alpha)
so that alpha stays positive; r = 1/alpha = exp(-la) is scipy's `n` parameter.
"""

import numpy as np
from scipy.special import digamma, gammaln
from scipy.stats import nbinom


def nb_nll(par, X, yv, w, lam_vec):
    k = X.shape[1]
    beta, la = par[:k], par[k]
    eta = X @ beta
    mu = np.exp(np.clip(eta, -5, 8))
    r = np.exp(-la)
    ll_ = (
        gammaln(yv + r) - gammaln(r) - gammaln(yv + 1) + r * np.log(r / (r + mu)) + yv * np.log(mu / (r + mu))
    )
    nll = -(w * ll_).sum() + 0.5 * (lam_vec * beta**2).sum()
    g_eta = w * (yv - mu) / (1 + mu / r)
    g_beta = -(X.T @ g_eta) + lam_vec * beta
    dr = digamma(yv + r) - digamma(r) + np.log(r / (r + mu)) + (mu - yv) / (r + mu)
    g_la = -(w * dr * (-r)).sum()
    return nll, np.append(g_beta, g_la)


def logpmf(yv, mu, a):
    r = 1 / a
    return nbinom.logpmf(yv, r, r / (r + mu))
