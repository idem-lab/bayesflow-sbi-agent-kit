"""Stage 9 — posterior predictive check on the real 2018/2019 fits.

Runs only after the stage-8 reliability check (reliability.py). Conditions the fitted model
on real observed weeks and asks whether it reproduces features of that data — especially
features NOT directly fit. Replicated datasets are the **in-sample posterior predictive from
the inferred trajectory** (pure SBI): draw the joint posterior over θ and the latent weekly
infections, then push those infections through the reporting model → reported-case
replicates for the observed weeks. No particle filter.

Test statistics target epidemic *shape* features a smooth single-wave SIR may not
reproduce — **week-to-week roughness** (the feature the stage-8 OOD check implicated),
peak, total, and lag-1 autocorrelation. Diagnostics via **ArviZ** (`plot_ppc_dist`,
`plot_ppc_tstat`); the printed p-value is the definitional P(T(y_rep) ≥ T(y_obs)).

Aggregated across all states, the roughness p-value shows whether the shape misfit is
systematic (connecting stage 9 back to the stage-8 finding) — a coherent misspecification
story, not a single number.

    KERAS_BACKEND=jax .venv/bin/python examples/flu-sir/posterior_predictive.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import simulator as sim
from data_flu import list_states, target_series
from train import load_trained, sample_posterior, infections_from_draws

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")


def _roughness(c, axis=-1):
    c = np.asarray(c, float); lc = np.log(c + 1.0)
    k = np.ones(3) / 3.0
    sm = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, lc)
    return np.mean(np.abs(lc - sm), axis=axis)


def _autocorr1(c, axis=-1):
    c = np.asarray(c, float)
    a = c - c.mean(axis=axis, keepdims=True)
    num = np.sum(a[..., 1:] * a[..., :-1], axis=axis)
    den = np.sum(a * a, axis=axis)
    return num / np.maximum(den, 1e-9)

T_STATS = {
    "roughness":  _roughness,     # week-to-week jaggedness (stage-8-implicated shape feature)
    "peak":       lambda c, axis=-1: np.max(c, axis=axis),
    "total":      lambda c, axis=-1: np.sum(c, axis=axis),
    "autocorr1":  _autocorr1,
}


def replicates(estimator, y_full, L, n_theta, n_rep, seed):
    """In-sample posterior-predictive case replicates for observed weeks 1..L: infer the
    joint posterior at cutoff L, push the posterior infection trajectories through the
    reporting model, and keep the observed-window columns. Returns (n_theta*n_rep, L)."""
    post = sample_posterior(estimator, y_full, L, num_samples=n_theta)
    inf = np.rint(infections_from_draws(post)).astype(np.int64)      # (n_theta, T_MAX)
    rho, delay = post["rho"], post["delay_mean"]
    rng = np.random.default_rng(seed)
    reps = [sim.report(inf, rho, delay, rng)[:, :L] for _ in range(n_rep)]
    return np.concatenate(reps, axis=0)                             # (n_theta*n_rep, L)


def p_values(y_rep, y_obs):
    return {name: float(np.mean(fn(y_rep, axis=1) >= fn(y_obs)[()])) for name, fn in T_STATS.items()}


def build_idata(y_rep, y_obs):
    import arviz as az
    return az.from_dict({"posterior_predictive": {"cases": y_rep[None, :, :]},
                         "observed_data": {"cases": np.asarray(y_obs, float)}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-theta", type=int, default=200)
    ap.add_argument("--n-rep", type=int, default=6)
    ap.add_argument("--L", type=int, default=20, help="observed cutoff (weeks) for the in-sample PPC")
    ap.add_argument("--featured", nargs="*", default=["Illinois", "California", "Vermont"])
    args = ap.parse_args()

    estimator = load_trained()
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"=== Stage 9: posterior predictive check (real 2018/19, first {args.L} weeks) ===")
    print("  Bayesian p-values P(T(y_rep) >= T(y_obs)); near 0/1 => model can't reproduce that feature\n")
    print(f"  {'state':12s} " + "  ".join(f"{n:>10s}" for n in T_STATS))
    # featured states: table + ArviZ plots
    for st in args.featured:
        _, y_full = target_series(st)
        if len(y_full) < args.L:
            continue
        y_obs = y_full[:args.L]
        y_rep = replicates(estimator, y_full, args.L, args.n_theta, args.n_rep, seed=1)
        pv = p_values(y_rep, y_obs)
        print(f"  {st:12s} " + "  ".join(f"{pv[n]:10.3f}" for n in T_STATS))
        try:
            import matplotlib; matplotlib.use("Agg"); import arviz as az
            idata = build_idata(y_rep, y_obs)
            az.plot_ppc_dist(idata, kind="kde").savefig(os.path.join(OUTDIR, f"ppc_dist_{st.replace(' ','_')}.png"))
        except Exception as e:  # pragma: no cover
            print(f"    (plot skipped: {e})")

    # aggregate across all states: is the roughness misfit systematic?
    print("\n=== Aggregate roughness check across all states (connects to stage 8) ===")
    rough_p = []
    for st in list_states():
        _, y_full = target_series(st)
        if len(y_full) < args.L or y_full[:args.L].max() < 20:
            continue
        y_obs = y_full[:args.L]
        y_rep = replicates(estimator, y_full, args.L, 100, 4, seed=2)
        rough_p.append(p_values(y_rep, y_obs)["roughness"])
    rough_p = np.array(rough_p)
    print(f"  states checked: {len(rough_p)}")
    print(f"  roughness p-value < 0.05 (observed jaggier than model): {np.mean(rough_p < 0.05)*100:.0f}% of states")
    print(f"  median roughness p-value: {np.median(rough_p):.3f}  (0.5 = well matched; <<0.5 = model too smooth)")
    print(f"\nplots -> {OUTDIR}/ppc_dist_*.png")


if __name__ == "__main__":
    main()
