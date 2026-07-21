"""Stage 3 — prior predictive check for the influenza SIR model.

Two reads, both against the committed stage-2 targets and the 137 real pre-2018
epidemic windows (the "plausible real data" a late-2018 analyst has):

1. **Plausibility** — do prior-predictive curves look like flu epidemics (not all-zero,
   not exploding, sensible shapes)?
2. **Coverage** — do the prior-predictive summary distributions *cover* the real ones?
   This is the guard against the stage-8 out-of-distribution failure: if the training
   distribution doesn't span plausible real data, the amortised posterior will be asked
   to extrapolate on the real 2018/19 season.

Backend-free (NumPy only): the pushforward is just the simulator, no network.

    .venv/bin/python examples/flu-sir/prior_predictive.py [--n 4000]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_flu import pretarget_windows, _summary          # noqa: E402
from simulator import sample_prior, simulate, PARAM_NAMES, T_MAX  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")

# Library takeoff threshold — matches data_flu.pretarget_windows(min_peak=10), which drops
# near-empty state-seasons. The prior predictive is compared to the library *conditional on
# takeoff*, so simulated curves are filtered the same way (a fraction fizzle, as in reality).
MIN_PEAK = 10

# Committed stage-2 targets (v4, full 39-week season): (central=50%, low5, high95) of the
# 138 real pre-2018 epidemic windows. Coverage = simulated 5-95% band must span the real
# IQR; shape/timing summaries must in addition be centred.
TARGETS = {
    "peak":       (198, 22, 1300),
    "peak_week":  (19, 12, 24),
    "total":      (1700, 131, 11741),
    "log_growth": (0.23, 0.04, 0.35),
}
KEYS = ["peak", "peak_week", "total", "log_growth"]


def summarise_batch(cases: np.ndarray) -> dict:
    """Compute the four committed summaries for a (B, W) batch of case curves."""
    return {k: np.array([_summary(c)[k] for c in cases], float) for k in KEYS}


# Scale summaries legitimately span widely (a prior covering many state sizes); we ask
# only that the simulated band spans the real IQR. Shape/timing summaries must in
# addition be CENTERED right — the simulated median must fall within the real IQR —
# otherwise the prior predictive wastes capacity on the wrong epidemic regime (the v1
# failure: correct support but a too-fast, too-early centre).
SCALE_KEYS = {"peak", "total"}
SHAPE_KEYS = {"peak_week", "log_growth"}


def coverage_report(sim: dict, real: dict) -> bool:
    """Print per-summary coverage/centering vs targets and the real library. Returns True
    if every summary passes its criterion (span for scale; span+centre for shape)."""
    print(f"\n{'summary':11s} {'target 5/50/95':>22s} {'real 5/50/95':>22s} {'sim 5/50/95':>24s}  verdict")
    ok_all = True
    for k in KEYS:
        c, lo, hi = TARGETS[k]
        rp = np.percentile(real[k], [5, 50, 95])
        sp = np.percentile(sim[k], [5, 50, 95])
        r_iqr = np.percentile(real[k], [25, 75])
        spans = (sp[0] <= r_iqr[0]) and (sp[2] >= r_iqr[1])
        centred = r_iqr[0] <= sp[1] <= r_iqr[1]
        ok = spans and (centred or k in SCALE_KEYS)
        ok_all &= ok
        tag = "OK" if ok else ("OFF-CENTRE" if (spans and k in SHAPE_KEYS) else "MISS")
        fmt = (lambda a: "/".join(f"{x:.2f}" for x in a)) if k == "log_growth" else (lambda a: "/".join(f"{x:.0f}" for x in a))
        print(f"{k:11s} {fmt([lo,c,hi]):>22s} {fmt(rp):>22s} {fmt(sp):>24s}  {tag}")
    return ok_all


def plausibility_report(cases: np.ndarray) -> None:
    peak = cases.max(axis=1)
    total = cases.sum(axis=1)
    took = cases[peak >= MIN_PEAK]                       # timing only meaningful once it takes off
    pw = np.argmax(took, axis=1)
    print("\n=== Plausibility screens ===")
    print(f"  fizzled curves (peak<10) : {np.mean(peak < MIN_PEAK)*100:5.1f}%  (some real state-seasons too)")
    print(f"  exploding (peak > 1e5)   : {np.mean(peak > 1e5)*100:5.1f}%  (want ~0)")
    print(f"  took-off peaks wks 1-8   : {np.mean(pw < 8)*100:5.1f}%  (real ~1%; early = too-fast)")
    print(f"  took-off peaks wks 12-24 : {np.mean((pw >= 12) & (pw < 24))*100:5.1f}%  (real ~93%; Dec–Mar)")
    print(f"  peak weekly cases q50    : {np.median(peak):.0f}   total q50: {np.median(total):.0f}")


def maybe_plot(sim: dict, real: dict, cases: np.ndarray, real_lib: list) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable — skipping plots)")
        return
    os.makedirs(OUTDIR, exist_ok=True)
    # Summary distributions: simulated vs real
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, k in zip(axes, KEYS):
        lo = min(sim[k].min(), real[k].min()); hi = max(np.percentile(sim[k], 99), real[k].max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(sim[k], bins=bins, density=True, alpha=0.5, label="prior-pred (sim)")
        ax.hist(real[k], bins=bins, density=True, alpha=0.5, label="real pre-2018")
        c, tlo, thi = TARGETS[k]
        ax.axvline(c, color="k", ls="--", lw=1)
        ax.set_title(k); ax.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle("Stage 3: prior-predictive vs real summary distributions")
    fig.tight_layout(); p = os.path.join(OUTDIR, "prior_predictive_summaries.png")
    fig.savefig(p, dpi=110); print(f"  saved {p}")
    # Example curves: simulated vs real overlay
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(cases), 40, replace=False):
        axes[0].plot(cases[i], color="C0", alpha=0.25, lw=0.8)
    for c in [real_lib[i] for i in rng.choice(len(real_lib), 40, replace=False)]:
        axes[1].plot(c, color="C1", alpha=0.3, lw=0.8)
    axes[0].set_title("40 prior-predictive curves"); axes[1].set_title("40 real pre-2018 curves")
    for ax in axes: ax.set_xlabel("week"); ax.set_ylabel("weekly positives")
    fig.tight_layout(); p = os.path.join(OUTDIR, "prior_predictive_curves.png")
    fig.savefig(p, dpi=110); print(f"  saved {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    params = sample_prior(args.n, rng)
    cases = simulate(params, n_weeks=T_MAX, rng=rng)["cases"]
    # Coverage/centring is judged conditional on takeoff, matching the library filter.
    took = cases[cases.max(axis=1) >= MIN_PEAK]
    sim = summarise_batch(took)

    real_lib = pretarget_windows()
    real = summarise_batch(np.array(real_lib))

    print(f"Prior-predictive check: {args.n} draws, {T_MAX}-week full season "
          f"({took.shape[0]} took off ≥ peak {MIN_PEAK})")
    plausibility_report(cases)
    ok = coverage_report(sim, real)
    print(f"\n=== Coverage verdict: {'PASS' if ok else 'FAIL — priors do not cover real data (route to stage 2)'} ===")
    maybe_plot(sim, real, cases, real_lib)


if __name__ == "__main__":
    main()
