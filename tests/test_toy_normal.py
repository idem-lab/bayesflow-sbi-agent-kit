"""Backend-free smoke tests for the toy Normal example.

These cover the parts that do not need a deep-learning backend: the simulator
(prior-predictive behaviour) and the reference-posterior diagnostics. The
BayesFlow training path is exercised separately by ``examples/toy-normal/train.py
--smoke`` in an environment that has a backend installed.

Runnable two ways:
    pytest tests/test_toy_normal.py
    python3 tests/test_toy_normal.py      # no pytest required
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make the example importable without packaging/install.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "examples", "toy-normal"),
)

import simulator as sim          # noqa: E402

# diagnostics.py needs SciPy; keep the simulator tests runnable without it.
try:
    import diagnostics as diag   # noqa: E402
    HAVE_DIAG = True
except ImportError:
    HAVE_DIAG = False


class _Skip(Exception):
    """Raised to skip a test when an optional dependency is missing."""


def _skip_if_no_diag():
    if not HAVE_DIAG:
        try:
            import pytest
            pytest.skip("diagnostics.py requires scipy")
        except ImportError:
            raise _Skip("scipy not installed")


def test_prior_ranges_and_moments():
    np.random.seed(0)
    draws = [sim.prior() for _ in range(5000)]
    mus = np.array([d["mu"] for d in draws])
    sigmas = np.array([d["sigma"] for d in draws])

    assert np.all(sigmas > 0), "sigma must be positive (HalfNormal)"
    # mu ~ Normal(0, 1): mean near 0, sd near 1.
    assert abs(mus.mean()) < 0.1
    assert abs(mus.std() - 1.0) < 0.1
    # HalfNormal(1) has mean sqrt(2/pi) ~ 0.798.
    assert abs(sigmas.mean() - np.sqrt(2.0 / np.pi)) < 0.1


def test_meta_sample_size_in_range():
    np.random.seed(1)
    Ns = np.array([sim.meta()["N"] for _ in range(2000)])
    assert Ns.min() >= sim.N_MIN
    assert Ns.max() <= sim.N_MAX
    assert set(np.unique(Ns)).issubset(set(range(sim.N_MIN, sim.N_MAX + 1)))


def test_likelihood_shapes():
    np.random.seed(2)
    for _ in range(20):
        m = sim.meta()
        x = sim.likelihood(0.0, 1.0, m["N"])["x"]
        assert x.shape == (m["N"],)


def test_prior_density_is_normalised():
    # Guards against dropped normalising constants (e.g. the -0.5*log(2*pi) that
    # was missing from the earlier hand-rolled priors). The joint prior density
    # must integrate to ~1 over a wide grid.
    _skip_if_no_diag()
    mu_axis = np.linspace(-8, 8, 400)
    sigma_axis = np.linspace(1e-4, 10, 400)
    mu_grid, sigma_grid = np.meshgrid(mu_axis, sigma_axis, indexing="ij")
    density = np.exp(diag._log_prior(mu_grid, sigma_grid))
    d_mu = mu_axis[1] - mu_axis[0]
    d_sigma = sigma_axis[1] - sigma_axis[0]
    mass = density.sum() * d_mu * d_sigma
    assert abs(mass - 1.0) < 1e-3, f"prior does not integrate to 1 (got {mass:.4f})"


def test_reference_posterior_is_normalised():
    _skip_if_no_diag()
    np.random.seed(3)
    x = np.random.normal(1.0, 0.5, size=15)
    ref = diag.reference_posterior(x)
    d_mu = ref["mu_axis"][1] - ref["mu_axis"][0]
    d_sigma = ref["sigma_axis"][1] - ref["sigma_axis"][0]
    mass = ref["posterior"].sum() * d_mu * d_sigma
    assert abs(mass - 1.0) < 1e-6


def test_reference_posterior_recovers_parameters():
    # With a reasonably large sample the posterior mean should sit near truth.
    _skip_if_no_diag()
    np.random.seed(4)
    true_mu, true_sigma = 1.5, 0.8
    x = np.random.normal(true_mu, true_sigma, size=20)
    ref = diag.reference_posterior(x)
    assert abs(ref["mu_mean"] - true_mu) < 0.4
    assert abs(ref["sigma_mean"] - true_sigma) < 0.4


def test_recovery_and_sbc_helpers():
    _skip_if_no_diag()
    truth = np.array([0.0, 1.0, -1.0, 2.0])
    est = truth + np.array([0.05, -0.05, 0.1, -0.1])
    summary = diag.recovery_summary(truth, est)
    assert summary["rmse"] < 0.2
    assert summary["correlation"] > 0.99

    # Rank of the true value among draws it sits in the middle of ~ L/2.
    draws = np.linspace(-3, 3, 101)
    assert diag.sbc_rank(0.0, draws) == 50


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = skipped = 0
    for t in tests:
        try:
            t()
        except _Skip as exc:
            print(f"[SKIP] {t.__name__} ({exc})")
            skipped += 1
            continue
        print(f"[PASS] {t.__name__}")
        passed += 1
    print(f"\n{passed} passed, {skipped} skipped, {len(tests)} total")


if __name__ == "__main__":
    _run_all()
