"""The final analysis — per-state, per-month influenza forecasts for 2018/2019 (pure SBI).

For each state and each monthly forecast cutoff ``L ∈ {4, 8, …, 36}`` (an analyst
re-forecasting once a month across the full Oct–Jun season):

1. Apply the *one* trained amortised estimator to the observed weeks 1..L (padded + masked)
   → a **joint posterior over the globals θ and the full latent weekly-infection
   trajectory** ``i₁..i₃₉`` (train.py). No particle filter, no MCMC — the latent states are
   inference targets the network learned to extrapolate.
2. Push every posterior trajectory through the reporting model (thin by ρ, geometric delay)
   → posterior-predictive **case** draws for all 39 weeks. Weeks ≤ L are the in-sample fit
   (data-pinned → tight); weeks > L are the forecast (dynamics-driven → uncertainty grows
   with horizon).

Outputs:
- ``outputs/forecast_<state>.png`` — for featured states, a 3-panel figure (forecasts issued
  early/mid/late in the season) overlaying real data, the in-sample posterior-predictive fit,
  the forecast band, and a vertical line at the forecast date.
- an **all-states forecast-calibration summary**: how often the held-out real forecast weeks
  fall inside the 90% forecast band (a genuine out-of-sample check of the whole pipeline),
  overall and per horizon week.

    KERAS_BACKEND=jax .venv/bin/python examples/flu-sir/forecast.py            # featured + summary
    KERAS_BACKEND=jax .venv/bin/python examples/flu-sir/forecast.py --all      # every state
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import simulator as sim
from data_flu import list_states, target_series, monthly_cutoffs
from train import load_trained, sample_posterior, infections_from_draws

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
FEATURED = ["Illinois", "California", "Vermont", "New York", "Texas"]
PLOT_CUTOFFS = (8, 16, 24)      # early / mid / late season, for the featured-state figures
HORIZON = 4                     # forecast the next 4 weeks (one month)


def case_bands(post: dict, seed: int, n_rep: int = 4) -> dict:
    """Push posterior infection trajectories through the reporting model → posterior-
    predictive weekly **case** bands (5/50/95) over all T_MAX weeks. Each trajectory is
    reported ``n_rep`` times to average over reporting (thinning+delay) stochasticity."""
    inf = np.rint(infections_from_draws(post)).astype(np.int64)      # (S, T_MAX)
    rho = post["rho"]; delay = post["delay_mean"]
    rng = np.random.default_rng(seed)
    reps = [sim.report(inf, rho, delay, rng) for _ in range(n_rep)]  # each (S, T_MAX)
    cases = np.concatenate(reps, axis=0)                             # (n_rep*S, T_MAX)
    lo, mid, hi = np.percentile(cases, [5, 50, 95], axis=0)
    return dict(lo=lo, mean=mid, hi=hi)


def fit_and_forecast(estimator, y_full: np.ndarray, L: int, n_theta: int, seed: int = 0) -> dict:
    """Observe weeks 1..L, infer the joint posterior, and derive in-sample + forecast case
    bands from the posterior trajectory. Returns bands, the held-out truth, and horizon."""
    post = sample_posterior(estimator, y_full, L, num_samples=n_theta)
    bands = case_bands(post, seed=seed)
    h = min(HORIZON, len(y_full) - L)
    return dict(L=L, y_full=y_full, bands=bands, horizon=h,
                truth_fore=y_full[L:L + h], post=post)


def coverage(res: dict) -> np.ndarray:
    """Per-horizon-week indicator: is the real forecast value inside the 90% band?"""
    L, h = res["L"], res["horizon"]
    lo, hi = res["bands"]["lo"], res["bands"]["hi"]
    t = res["truth_fore"]
    return np.array([(lo[L + i] <= t[i] <= hi[L + i]) for i in range(h)], dtype=float)


def plot_state(state: str, results: list, path: str) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.8), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        L = res["L"]; b = res["bands"]; y = res["y_full"]
        wk = np.arange(1, sim.T_MAX + 1)
        # in-sample fit (weeks 1..L) and forecast (weeks L+1..end) from the one trajectory
        ax.fill_between(wk[:L], b["lo"][:L], b["hi"][:L], color="C0", alpha=0.25, label="in-sample 90%")
        ax.plot(wk[:L], b["mean"][:L], color="C0", lw=1.3)
        ax.fill_between(wk[L:], b["lo"][L:], b["hi"][L:], color="C3", alpha=0.22, label="forecast 90%")
        ax.plot(wk[L:], b["mean"][L:], color="C3", lw=1.3)
        ax.plot(wk[:L], y[:L], "ko", ms=2.5, label="real (fit)")
        ax.plot(wk[L:], y[L:], "o", color="0.4", mfc="white", ms=3, label="real (held out)")
        ax.axvline(L + 0.5, color="k", ls="--", lw=1)
        ax.set_title(f"{state} — forecast from week {L} (month {L//4})")
        ax.set_xlabel("epidemic week (from ~1 Oct)")
    axes[0].set_ylabel("weekly positive specimens")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=115); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run every state for the summary")
    ap.add_argument("--n-theta", type=int, default=400)
    args = ap.parse_args()

    estimator = load_trained()
    os.makedirs(OUTDIR, exist_ok=True)
    cutoffs = monthly_cutoffs()      # [4, 8, ..., 36]

    # Featured states: figures at a few representative cutoffs.
    print("=== Featured-state forecasts ===")
    for st in FEATURED:
        _, y_full = target_series(st)
        if len(y_full) < sim.T_MAX:
            print(f"  {st}: insufficient data ({len(y_full)} wk), skipping"); continue
        results = [fit_and_forecast(estimator, y_full, L, args.n_theta, seed=L) for L in PLOT_CUTOFFS]
        path = os.path.join(OUTDIR, f"forecast_{st.replace(' ', '_')}.png")
        plot_state(st, results, path)
        covs = [coverage(r).mean() for r in results]
        print(f"  {st:12s} saved {os.path.basename(path)}  | forecast-in-band by cutoff: "
              + ", ".join(f"L{r['L']}={c*100:.0f}%" for r, c in zip(results, covs)))

    # All-states: out-of-sample forecast calibration across every monthly cutoff.
    states = list_states() if args.all else FEATURED
    print(f"\n=== Forecast calibration over {len(states)} states x {len(cutoffs)} monthly cutoffs ===")
    per_h = [[] for _ in range(HORIZON)]
    n_fits = 0
    for st in states:
        _, y_full = target_series(st)
        if len(y_full) < sim.T_MAX:
            continue
        for L in cutoffs:
            res = fit_and_forecast(estimator, y_full, L, args.n_theta, seed=L)
            cov = coverage(res); n_fits += 1
            for i in range(len(cov)):
                per_h[i].append(cov[i])
    allc = np.concatenate([np.array(h) for h in per_h])
    print(f"  fits: {n_fits} | overall 90% forecast-interval coverage: {allc.mean()*100:.0f}% (target ~90%)")
    for i in range(HORIZON):
        h = np.array(per_h[i])
        print(f"    horizon +{i+1} wk: coverage {h.mean()*100:.0f}%  (n={len(h)})")


if __name__ == "__main__":
    main()
