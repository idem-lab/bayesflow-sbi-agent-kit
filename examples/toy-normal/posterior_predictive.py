"""Posterior predictive check report for the toy Normal model.

This is the worked example of **stage 9** of the workflow (the `posterior-predictive-check`
skill): condition on an observed dataset, draw parameters from the posterior, simulate
**replicated** datasets from the model's OWN likelihood, and compare those replicates to
the observation.

Design choices, matching the skill and the repo's "use validated libraries" principle:

- **Replicated datasets come from the toy's own** ``likelihood`` (``simulator.py``), not a
  hand-rolled sampler. The posterior predictive check *is* "simulate from the fitted model";
  the model is the simulator we already have.
- **All diagnostics come from ArviZ** — ``plot_ppc_dist`` (observed-vs-replicated overlay),
  ``plot_ppc_tstat`` (Bayesian p-value for a test statistic), and ``plot_ppc_pit`` (a PIT
  calibration check). ArviZ is the standard, validated Bayesian-workflow diagnostics
  package; we do not reimplement these. The only number we compute directly is the
  definitional posterior predictive p-value P(T(y_rep) >= T(y_obs)) for the printed table.
- **The test statistics deliberately include features the model was NOT directly fit on.**
  The toy fits the mean (``mu``) and sd (``sigma``), so ``mean``/``std`` are near-trivially
  reproduced and are only a sanity check; ``min``/``max``/``iqr``/``skew`` probe features that
  are not directly parameterised. This is the skill's core instruction — and, as
  ``--misspecify`` shows below, the statistic has to *target the feature that is wrong* to
  catch a misfit.

On this **correctly specified** toy the check is EXPECTED to pass — Bayesian p-values sit
away from 0 and 1 and the observation lies inside the replicated cloud. That is itself the
validation that the report works. Pass ``--misspecify`` to draw the observation from a
**skewed** process (a shifted, scaled exponential — skewness 2 — matched in mean and sd):
a Normal cannot represent skew, so ``mean``/``std`` (the fitted moments) still look fine,
and — importantly — the generic ``min``/``max``/``iqr`` checks stay quiet too, while the
**skewness** statistic flags it. That is the sharper lesson of the skill: a not-directly-fit
statistic only catches a misfit if it *targets the feature that is wrong*, so choose test
statistics for the failure modes you care about.

A toy has no real data, so a held-out draw from the prior predictive stands in for the
observation (seeded, documented). For a real model this observation would be your data.
Keep the observation size within the trained N range (5-20): a larger set would itself be
out-of-distribution for the summary network — exactly what the stage-8 reliability check
guards against — so it would never legitimately reach this stage.

Run (needs a backend):

    KERAS_BACKEND=jax .venv/bin/python examples/toy-normal/posterior_predictive.py
    KERAS_BACKEND=jax .venv/bin/python examples/toy-normal/posterior_predictive.py --misspecify

Plots (if matplotlib + arviz are available) are written to ``outputs/`` next to this file.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy import stats as sstats

from simulator import prior, likelihood
from train import build_workflow

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")


def _iqr(a: np.ndarray, axis=None) -> np.ndarray:
    q75, q25 = np.percentile(a, [75, 25], axis=axis)
    return q75 - q25


def _skew_np(a: np.ndarray, axis=None) -> np.ndarray:
    return sstats.skew(a, axis=axis)


def _skew_da(da):
    """Skewness reducer for ArviZ ``plot_ppc_tstat`` (a callable t_stat must reduce the
    non-``sample`` dimension and keep the sample dimension)."""
    import xarray as xr

    reduce = [d for d in da.dims if d != "sample"]
    return xr.apply_ufunc(sstats.skew, da, input_core_dims=[reduce], kwargs={"axis": -1})


_skew_da.__name__ = "skew"

# label -> (ArviZ t_stat argument, NumPy reducer for the printed p-value table).
# mean/std are directly fit (sanity); min/max/iqr are generic not-directly-fit checks;
# skew directly targets asymmetry — the feature a Normal cannot reproduce (see --misspecify).
T_STATS = {
    "mean": ("mean", np.mean),
    "std": ("std", np.std),
    "min": ("min", np.min),
    "max": ("max", np.max),
    "iqr": ("iqr", _iqr),
    "skew": (_skew_da, _skew_np),
}
DIRECTLY_FIT = {"mean": "mu", "std": "sigma"}


def make_observation(obs_n: int, seed: int, misspecify: bool):
    """A stand-in 'observed' dataset. Well-specified = drawn from the model itself;
    misspecified = a shifted, scaled exponential (skewness 2) with the SAME mean and sd, so
    the moments the model fits (mean, sd) still match and only the skewness statistic flags it."""
    np.random.seed(seed)
    p = prior()
    if not misspecify:
        x = likelihood(p["mu"], p["sigma"], obs_n)["x"]
    else:
        # skewed: a shifted, scaled exponential (skewness 2) standardised to the SAME
        # mean and sd as the model would fit. A Normal cannot reproduce this asymmetry,
        # so the moments match but the skewness statistic flags it.
        z = np.random.exponential(1.0, size=obs_n) - 1.0  # mean 0, sd 1, skew 2
        x = p["mu"] + p["sigma"] * z
    return p, np.asarray(x, dtype="float32")


def posterior_predictive(workflow, x_obs: np.ndarray, n_samples: int, seed: int):
    """Posterior draws given the observation, and replicated datasets simulated from the
    model's own likelihood at each posterior draw."""
    s = workflow.sample(conditions={"x": x_obs[None, :]}, num_samples=n_samples)
    mu = np.asarray(s["mu"]).reshape(-1)
    sigma = np.asarray(s["sigma"]).reshape(-1)

    N = x_obs.shape[0]
    np.random.seed(seed + 1)
    y_rep = np.stack([likelihood(m, sd, N)["x"] for m, sd in zip(mu, sigma)])  # (S, N)
    return mu, sigma, y_rep


def build_idata(mu, sigma, y_rep, x_obs):
    """Assemble an ArviZ DataTree in the conventional (chain, draw, obs) layout."""
    import arviz as az

    return az.from_dict(
        {
            "posterior": {"mu": mu[None, :], "sigma": sigma[None, :]},
            "posterior_predictive": {"x": y_rep[None, :, :]},
            "observed_data": {"x": x_obs},
        }
    )


def p_value_table(y_rep: np.ndarray, x_obs: np.ndarray) -> dict:
    """Definitional posterior predictive p-value P(T(y_rep) >= T(y_obs)) per statistic.
    (The plots come from ArviZ; this is just the number for the printed report.)"""
    out = {}
    for name, (_, fn) in T_STATS.items():
        t_rep = fn(y_rep, axis=1)
        t_obs = fn(x_obs)
        out[name] = float(np.mean(t_rep >= t_obs))
    return out


def _maybe_plots(idata, prefix: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import arviz as az
    except ImportError:
        print("(matplotlib / arviz not available — skipping plots)")
        return

    os.makedirs(OUTDIR, exist_ok=True)

    def save(pc, name):
        path = os.path.join(OUTDIR, f"{prefix}{name}.png")
        pc.savefig(path)
        print(f"  saved {name:28s} -> {path}")

    save(az.plot_ppc_dist(idata, kind="kde"), "posterior_predictive_dist")
    for name, (t_stat, _) in T_STATS.items():
        save(az.plot_ppc_tstat(idata, t_stat=t_stat), f"posterior_predictive_tstat_{name}")
    save(az.plot_ppc_pit(idata), "posterior_predictive_pit")


def main() -> None:
    parser = argparse.ArgumentParser(description="Posterior predictive check report.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--obs-n", type=int, default=20,
                        help="size of the observed dataset; keep within the trained N range "
                             "(5-20) so it is in-distribution for the summary network")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--misspecify", action="store_true",
                        help="draw the observation from a skewed process to show the "
                             "check catching a misfit")
    args = parser.parse_args()

    print(f"Training: epochs={args.epochs} batch_size={args.batch_size} "
          f"num_batches={args.num_batches}")
    workflow = build_workflow()
    workflow.fit_online(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_batches_per_epoch=args.num_batches,
    )

    p_true, x_obs = make_observation(args.obs_n, args.seed, args.misspecify)
    mu, sigma, y_rep = posterior_predictive(workflow, x_obs, args.n_samples, args.seed)

    tag = "MISSPECIFIED (skewed)" if args.misspecify else "well-specified"
    print(f"\n=== Posterior predictive check ({tag}) ===")
    print(f"  observation: N={args.obs_n}  true mu={p_true['mu']:+.3f}  "
          f"sigma={p_true['sigma']:.3f}")
    print(f"  posterior mean: mu={mu.mean():+.3f}  sigma={sigma.mean():.3f}  "
          f"({args.n_samples} draws, {y_rep.shape[0]} replicated datasets)")

    print("\n  Bayesian posterior predictive p-values  P(T(y_rep) >= T(y_obs)):")
    print("  (near 0 or 1 = the model cannot reproduce that feature; ~0.5 = fine)")
    pvals = p_value_table(y_rep, x_obs)
    for name in T_STATS:
        fit = f"directly fit via {DIRECTLY_FIT[name]}" if name in DIRECTLY_FIT else "NOT directly fit"
        print(f"    {name:5s}  p={pvals[name]:.3f}   ({fit})")

    print("\n=== Plots (ArviZ) ===")
    prefix = "misspec_" if args.misspecify else ""
    idata = build_idata(mu, sigma, y_rep, x_obs)
    _maybe_plots(idata, prefix)


if __name__ == "__main__":
    main()
