"""Stages 6-7 — parameter recovery + SBC calibration on SIMULATED data.

Loads the trained estimator (train.py) and, on held-out prior-predictive datasets, asks
the method-validation questions the workflow requires *before* real data:

- **Recovery (stage 6):** does the posterior locate the truth *relative to its own
  uncertainty*? Read via posterior contraction and posterior z-score (not point accuracy).
- **Global calibration / SBC (stage 7):** are the 7 global posteriors calibrated? Read via
  ECDF-based SBC rank plots + calibration error.
- **Trajectory calibration / SBC (stage 7, the key new check for the pure-SBI redesign):**
  is the inferred latent **weekly-infection trajectory** calibrated *per week* — both for
  the in-sample weeks (≤ L, pinned by data → tight) and the forecast weeks (> L, driven by
  the learned dynamics → wide)? A correctly amortised posterior is calibrated at every week
  and every cutoff. This is what replaces the old particle-filter reconstruction.

All metrics/plots come straight from ``bayesflow.diagnostics`` (no hand-rolled diagnostics).
Globals are scored on the **inference scale** (log for positives, logit for ``rho``); the
trajectory is scored on its ``log(infections+1)`` scale. Amortisation is checked across the
monthly cutoffs L = 8, 16, 24 (one estimator, all forecast dates).

    KERAS_BACKEND=jax .venv/bin/python examples/flu-sir/run_validation.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import simulator as sim
from train import load_trained, TRAJ_KEY

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")


def to_inference_scale(d: dict) -> dict:
    """log the positive params, logit rho — the scale the flow infers on."""
    out = {}
    for k in sim.PARAM_NAMES:
        v = np.asarray(d[k], float)
        if k == "rho":
            v = np.clip(v, 1e-6, 1 - 1e-6); out[k] = np.log(v / (1 - v))
        else:
            out[k] = np.log(np.clip(v, 1e-12, None))
    return out


def collect(workflow, n_datasets: int, L: int, n_samples: int, seed: int):
    """Simulate n_datasets full-season datasets, observe each to cutoff L (mask beyond),
    sample posteriors in one batched call, and return diagnostics-ready arrays for BOTH the
    globals (inference scale) and the latent log-infection trajectory."""
    rng = np.random.default_rng(seed)
    truth = sim.sample_prior(n_datasets, rng)
    full = sim.simulate(truth, sim.T_MAX, rng, return_latent=True)
    cases, mask = sim.observed_and_mask(full["cases"].astype("float32"), L)  # (D, T, 1) each
    draws = workflow.sample(conditions={"cases": cases, "mask": mask}, num_samples=n_samples)

    # --- globals (inference scale): estimates (D,S,1), targets (D,1) ---
    est_s = to_inference_scale({k: np.asarray(draws[k]).reshape(n_datasets, n_samples)
                                for k in sim.PARAM_NAMES})
    tgt_s = to_inference_scale(truth)
    g_est = {k: est_s[k][:, :, None] for k in sim.PARAM_NAMES}
    g_tgt = {k: tgt_s[k][:, None] for k in sim.PARAM_NAMES}

    # --- trajectory (log(inf+1) scale): estimates (D,S,T), targets (D,T) ---
    t_est = np.asarray(draws[TRAJ_KEY]).reshape(n_datasets, n_samples, sim.T_MAX)
    t_tgt = np.log(full["weekly_inf"].astype(float) + 1.0)
    return g_est, g_tgt, t_est, t_tgt


def main() -> None:
    import bayesflow.diagnostics as bfd

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-datasets", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=300)
    args = ap.parse_args()

    workflow = load_trained()
    os.makedirs(OUTDIR, exist_ok=True)
    CUTOFFS = (8, 16, 24)

    # --- Amortisation across cutoffs: global recovery correlation + contraction ---
    print("=== Global recovery across monthly cutoffs (one estimator, all forecast dates) ===")
    cache = {}
    rows = {k: [] for k in sim.PARAM_NAMES}
    for L in CUTOFFS:
        g_est, g_tgt, t_est, t_tgt = collect(workflow, args.n_datasets, L, args.n_samples, seed=100 + L)
        cache[L] = (g_est, g_tgt, t_est, t_tgt)
        corr = bfd.metrics.correlation(estimates=g_est, targets=g_tgt)["values"]
        contr = bfd.metrics.posterior_contraction(estimates=g_est, targets=g_tgt)["values"]
        for i, k in enumerate(sim.PARAM_NAMES):
            rows[k].append(f"{corr[i]:.2f}/{contr[i]:.2f}")
    print(f"{'param':11s} " + "   ".join(f"L={L:<2d}(corr/contr)" for L in CUTOFFS))
    for k in sim.PARAM_NAMES:
        print(f"{k:11s} " + "     ".join(rows[k]))

    # --- Global stage 6/7 verdict at L=16 ---
    g_est, g_tgt, t_est, t_tgt = cache[16]
    zsc = bfd.metrics.posterior_z_score(estimates=g_est, targets=g_tgt)["values"]
    cal = bfd.metrics.calibration_error(estimates=g_est, targets=g_tgt)["values"]
    contr = bfd.metrics.posterior_contraction(estimates=g_est, targets=g_tgt)["values"]
    corr = bfd.metrics.correlation(estimates=g_est, targets=g_tgt)["values"]
    print("\n=== Stage 6 (recovery) + Stage 7 (SBC) — globals @ L=16, inference scale ===")
    print(f"{'param':11s} {'corr':>6} {'contraction':>12} {'mean z':>8} {'calib_err':>10}  read")
    for i, k in enumerate(sim.PARAM_NAMES):
        read = "well-identified" if contr[i] > 0.5 else "weakly-identified (expected)"
        print(f"{k:11s} {corr[i]:6.2f} {contr[i]:12.2f} {float(np.mean(zsc[i])):8.2f} "
              f"{cal[i]:10.3f}  {read}")

    # --- Trajectory SBC @ L=12: per-week calibration, split in-sample vs forecast ---
    L = 12
    _, _, t_est, t_tgt = collect(workflow, args.n_datasets, L, args.n_samples, seed=777)
    t_cal = bfd.metrics.calibration_error(estimates=t_est, targets=t_tgt)["values"]     # (T,)
    t_contr = bfd.metrics.posterior_contraction(estimates=t_est, targets=t_tgt)["values"]
    wk = np.arange(sim.T_MAX)
    ins, fwd = wk < L, wk >= L
    print(f"\n=== Trajectory SBC — latent weekly infections @ cutoff L={L} ===")
    print(f"  in-sample weeks 1-{L}:   mean calib_err {t_cal[ins].mean():.3f}   "
          f"mean contraction {t_contr[ins].mean():.2f}  (data-pinned → tight)")
    print(f"  forecast weeks {L+1}-{sim.T_MAX}:  mean calib_err {t_cal[fwd].mean():.3f}   "
          f"mean contraction {t_contr[fwd].mean():.2f}  (dynamics-driven → wide)")
    print(f"  worst-week calib_err {t_cal.max():.3f} at week {int(t_cal.argmax())+1} "
          f"(SBC-calibrated if all ≲ 0.05)")

    # --- Plots (bayesflow.diagnostics) ---
    print("\n=== Plots ===")
    for name, fn, est, tgt, names in [
        ("recovery", bfd.recovery, g_est, g_tgt, sim.PARAM_NAMES),
        ("calibration_ecdf", bfd.calibration_ecdf, g_est, g_tgt, sim.PARAM_NAMES),
        ("z_score_contraction", bfd.z_score_contraction, g_est, g_tgt, sim.PARAM_NAMES),
    ]:
        try:
            fig = fn(estimates=est, targets=tgt, variable_names=names)
            p = os.path.join(OUTDIR, f"validation_{name}.png")
            fig.savefig(p, dpi=110); print(f"  saved {p}")
        except Exception as e:  # pragma: no cover
            print(f"  ({name} plot skipped: {e})")
    # Trajectory SBC ECDF for a few representative weeks (in-sample + forecast).
    try:
        wk_sel = [3, 7, 11, 15, 23, 35]      # weeks 4,8,12(=L),16,24,36 (0-indexed)
        est_sel = {f"wk{w+1:02d}": t_est[:, :, w:w + 1] for w in wk_sel}
        tgt_sel = {f"wk{w+1:02d}": t_tgt[:, w:w + 1] for w in wk_sel}
        fig = bfd.calibration_ecdf(estimates=est_sel, targets=tgt_sel,
                                   variable_names=list(est_sel.keys()))
        p = os.path.join(OUTDIR, "validation_trajectory_sbc.png")
        fig.savefig(p, dpi=110); print(f"  saved {p}")
    except Exception as e:  # pragma: no cover
        print(f"  (trajectory SBC plot skipped: {e})")


if __name__ == "__main__":
    main()
