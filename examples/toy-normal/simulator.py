"""Toy reference model: infer the mean and standard deviation of a Normal.

Model (approved priors):

    mu    ~ Normal(0, 1)            # location
    sigma ~ HalfNormal(1)           # scale, > 0   (|Normal(0, 1)|)
    N     ~ randint(5, 20)          # observations per dataset (variable)
    x_i   ~ Normal(mu, sigma)       # i = 1..N, i.i.d.

The observations `x` are exchangeable and variable-length, which is what makes
this a useful smoke test for BayesFlow: the summary network must map a set of
5-20 numbers to a fixed embedding (see ``train.py``, which uses ``.as_set`` +
``DeepSet``).

This module is deliberately pure NumPy so it can be imported, run, and tested
without a deep-learning backend. The only BayesFlow dependency is confined to
``make_simulator`` below, which imports lazily.
"""

from __future__ import annotations

import numpy as np

# --- Model constants (single source of truth; imported by tests & diagnostics)
MU_LOC, MU_SCALE = 0.0, 1.0        # Normal(0, 1) prior on mu
SIGMA_SCALE = 1.0                  # HalfNormal(1) prior on sigma
N_MIN, N_MAX = 5, 20               # inclusive range for the number of observations


def prior() -> dict:
    """Draw one (mu, sigma) from the priors. Uses global NumPy RNG state."""
    return dict(
        mu=np.random.normal(MU_LOC, MU_SCALE),
        sigma=np.abs(np.random.normal(0.0, SIGMA_SCALE)),   # HalfNormal(1)
    )


def meta() -> dict:
    """Draw the (variable) number of observations for one dataset."""
    return dict(N=int(np.random.randint(N_MIN, N_MAX + 1)))


def likelihood(mu: float, sigma: float, N: int) -> dict:
    """Draw N i.i.d. observations given the parameters and sample size."""
    return dict(x=np.random.normal(mu, sigma, size=N))


def make_simulator():
    """Assemble the BayesFlow simulator from the functions above.

    Imported lazily so that importing this module (e.g. from tests) does not
    require BayesFlow or a backend to be installed.
    """
    import bayesflow as bf

    return bf.simulators.make_simulator([prior, likelihood], meta_fn=meta)


if __name__ == "__main__":
    # Tiny prior-predictive smoke check (pure NumPy).
    np.random.seed(0)
    for _ in range(3):
        p = prior()
        m = meta()
        d = likelihood(p["mu"], p["sigma"], m["N"])
        print(f"mu={p['mu']:+.3f}  sigma={p['sigma']:.3f}  N={m['N']:2d}  "
              f"x[:3]={np.round(d['x'][:3], 3)}")
