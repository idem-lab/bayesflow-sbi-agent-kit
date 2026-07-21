"""Backend-free tests for the influenza SIR example.

Cover the pure-NumPy pieces — simulator (dynamics + reporting + trajectory/masking) and
data loader — without a deep-learning backend (mirroring tests/test_toy_normal.py).
Training/inference-network checks live in run_validation.py / reliability.py /
posterior_predictive.py / forecast.py, which need a backend.

    .venv/bin/python tests/test_flu_sir.py      # or: pytest tests/test_flu_sir.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "flu-sir"))

import simulator as sim          # noqa: E402
import data_flu                  # noqa: E402


def test_prior_draws_valid():
    rng = np.random.default_rng(0)
    p = sim.sample_prior(2000, rng)
    assert set(p) == set(sim.PARAM_NAMES)
    for k in sim.PARAM_NAMES:
        assert np.all(np.isfinite(p[k])) and np.all(p[k] > 0), k
    assert np.all(p["rho"] < 1.0)                       # reported fraction is a probability
    # medians land near the literature-anchored priors
    assert 1.15 < np.median(p["R0"]) < 1.4
    assert 3 < 1 / np.median(p["gamma"]) < 9            # generation interval in days


def test_simulate_shapes_and_nonneg():
    rng = np.random.default_rng(1)
    for W in (4, 8, 16, sim.T_MAX):
        c = sim.simulate(sim.sample_prior(32, rng), W, rng)["cases"]
        assert c.shape == (32, W)
        assert np.all(c >= 0) and c.dtype.kind in "iu"
    # default length is the full season
    assert sim.simulate(sim.sample_prior(4, rng), rng=rng)["cases"].shape[1] == sim.T_MAX


def test_reporting_censors_and_thins():
    # cases can never exceed true infections (thinning by rho<=1 + delay censoring)
    rng = np.random.default_rng(2)
    p = sim.sample_prior(200, rng)
    out = sim.simulate(p, sim.T_MAX, rng, return_latent=True)
    assert np.all(out["cases"].sum(1) <= out["weekly_inf"].sum(1) + 1e-9)


def test_simulate_reproducible_with_seed():
    a = sim.simulate(sim.sample_prior(8, np.random.default_rng(5)), 12, np.random.default_rng(9))
    b = sim.simulate(sim.sample_prior(8, np.random.default_rng(5)), 12, np.random.default_rng(9))
    assert np.array_equal(a["cases"], b["cases"])


def test_observed_and_mask():
    """The condition builder: weeks 1..L observed, rest zero-padded and masked out."""
    rng = np.random.default_rng(4)
    cases_full = sim.simulate(sim.sample_prior(16, rng), sim.T_MAX, rng)["cases"]
    for L in (4, 12, 36):
        cases_obs, mask = sim.observed_and_mask(cases_full.astype("float32"), L)
        assert cases_obs.shape == (16, sim.T_MAX, 1)
        assert mask.shape == (16, sim.T_MAX, 1)
        assert np.all(mask[:, :L, 0] == 1.0) and np.all(mask[:, L:, 0] == 0.0)
        assert np.array_equal(cases_obs[:, :L, 0], cases_full[:, :L].astype("float32"))
        assert np.all(cases_obs[:, L:, 0] == 0.0)


def test_monthly_cutoffs():
    cuts = data_flu.monthly_cutoffs()
    assert cuts == [4, 8, 12, 16, 20, 24, 28, 32, 36]
    assert all(0 < L < data_flu.WINDOW_WEEKS for L in cuts)


def test_data_loads_and_target_window():
    df = data_flu.load_clinical()
    states = data_flu.list_states(df)
    assert 45 <= len(states) <= 52                      # 50 states + DC, territories excluded
    _, cases = data_flu.target_series("Illinois", df=df)
    assert len(cases) == data_flu.WINDOW_WEEKS == 39
    assert np.all(cases >= 0)


def test_pretarget_library_nonempty():
    lib = data_flu.pretarget_windows()
    assert len(lib) > 50
    assert all(len(c) == data_flu.WINDOW_WEEKS for c in lib)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
