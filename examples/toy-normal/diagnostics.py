"""Reference posterior for the toy Normal model.

This provides a **reference posterior** computed on a (mu, sigma) grid. Because
this toy is low-dimensional, we can evaluate the exact (unnormalised) posterior
directly and normalise it numerically. This gives a ground truth to compare a
trained BayesFlow posterior against — the kind of check that is impossible for
real models and precisely why a toy reference model is worth having. It is also
the one diagnostic here that BayesFlow cannot provide, because it is specific to
this model.

Recovery, calibration (SBC) and the z-score/contraction sensitivity diagnostic are
**not** hand-rolled: they come from ``bayesflow.diagnostics`` (see
``run_validation.py``), in keeping with the "prefer validated libraries" principle.

All probability densities come from ``scipy.stats`` rather than being hand-coded,
so the prior and likelihood are correct (including normalising constants) by
construction. Priors must match ``simulator.py`` exactly:
mu ~ Normal(0, 1), sigma ~ HalfNormal(1).

Requires NumPy and SciPy (no deep-learning backend).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from simulator import MU_LOC, MU_SCALE, SIGMA_SCALE


# --------------------------------------------------------------------------- #
# Reference posterior on a grid
# --------------------------------------------------------------------------- #
def _log_prior(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """log p(mu) + log p(sigma) for Normal(0,1) and HalfNormal(1).

    ``halfnorm.logpdf`` returns -inf for sigma <= 0 automatically.
    """
    return (
        stats.norm.logpdf(mu, loc=MU_LOC, scale=MU_SCALE)
        + stats.halfnorm.logpdf(sigma, scale=SIGMA_SCALE)
    )


def _log_likelihood(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """sum_i log Normal(x_i | mu, sigma), broadcast over a grid of (mu, sigma).

    x has shape (n,); mu and sigma are 2-D grids of matching shape. The result
    sums the per-observation log-densities over the observation axis.
    """
    x = np.asarray(x).reshape(-1)
    per_obs = stats.norm.logpdf(
        x[:, None, None], loc=mu[None, :, :], scale=sigma[None, :, :]
    )
    return per_obs.sum(axis=0)


def reference_posterior(
    x: np.ndarray,
    mu_lim: tuple[float, float] = (-4.0, 4.0),
    sigma_lim: tuple[float, float] = (1e-3, 6.0),
    n_grid: int = 200,
) -> dict:
    """Normalised posterior over a (mu, sigma) grid for observed data ``x``.

    Returns a dict with the grid axes, the normalised posterior density, and the
    posterior means of mu and sigma (useful as point summaries for recovery).
    """
    mu_axis = np.linspace(*mu_lim, n_grid)
    sigma_axis = np.linspace(*sigma_lim, n_grid)
    mu_grid, sigma_grid = np.meshgrid(mu_axis, sigma_axis, indexing="ij")

    log_post = _log_prior(mu_grid, sigma_grid) + _log_likelihood(x, mu_grid, sigma_grid)
    log_post -= log_post.max()                     # stabilise before exp
    post = np.exp(log_post)

    d_mu = mu_axis[1] - mu_axis[0]
    d_sigma = sigma_axis[1] - sigma_axis[0]
    post /= post.sum() * d_mu * d_sigma            # normalise to integrate to 1

    weight = post * d_mu * d_sigma
    mu_mean = float((mu_grid * weight).sum())
    sigma_mean = float((sigma_grid * weight).sum())

    return dict(
        mu_axis=mu_axis,
        sigma_axis=sigma_axis,
        posterior=post,
        mu_mean=mu_mean,
        sigma_mean=sigma_mean,
    )


if __name__ == "__main__":
    # Sanity demo (pure NumPy): the grid posterior should recover a known mu/sigma
    # when N is reasonably large.
    np.random.seed(0)
    true_mu, true_sigma = 1.5, 0.8
    x = np.random.normal(true_mu, true_sigma, size=20)
    ref = reference_posterior(x)
    print(f"true  mu={true_mu:.2f} sigma={true_sigma:.2f}")
    print(f"grid  mu={ref['mu_mean']:.2f} sigma={ref['sigma_mean']:.2f}")
