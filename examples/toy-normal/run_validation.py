"""Fuller training run + inference check for the toy Normal model.

Trains the amortised posterior for a meaningful number of epochs, then checks that
inference actually works, using **BayesFlow's built-in diagnostics** rather than
hand-rolled metrics:

1. **Parameter recovery** — do posterior means track the true mu / sigma?
   (normalised RMSE and correlation, plus a recovery plot.)
2. **Calibration (SBC)** — is the posterior calibrated? (ECDF-based calibration
   error, plus a calibration-ECDF plot — the modern SBC variant.)
3. **Sensitivity** — posterior z-score vs. posterior contraction. This is what
   distinguishes a genuinely *poorly-identified* parameter (low contraction, small
   |z|) from a *biased* inference engine (large |z|). See the
   ``workflow-orchestration`` skill, stage 6.
4. **Grid cross-check** — on a few datasets, does the BayesFlow posterior mean
   agree with the exact grid-reference posterior from ``diagnostics.py``? That grid
   reference is the one, bespoke, model-specific ground truth BayesFlow cannot
   provide.

Run (needs a backend):

    KERAS_BACKEND=jax .venv/bin/python examples/toy-normal/run_validation.py

Sampling is batched by set size N so JAX compiles at most once per N. Plots (if
matplotlib is available) are written to ``outputs/`` next to this file.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

import bayesflow.diagnostics as bfd
from simulator import prior, meta, likelihood
import diagnostics as diag
from train import build_workflow

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")


def collect(workflow, n_test: int, n_samples: int, seed: int = 7):
    """Sample posteriors over many fresh test datasets.

    Returns ``(estimates, targets)`` in BayesFlow's diagnostics format:
    ``estimates[name]`` has shape (num_datasets, num_samples, 1) and
    ``targets[name]`` has shape (num_datasets, 1). Datasets are bucketed by set
    size N so JAX compiles at most once per N.
    """
    np.random.seed(seed)
    buckets: dict[int, list] = defaultdict(list)
    for _ in range(n_test):
        p, m = prior(), meta()
        obs = likelihood(p["mu"], p["sigma"], m["N"])
        buckets[m["N"]].append((p["mu"], p["sigma"], obs["x"]))

    mu_e, sig_e, mu_t, sig_t = [], [], [], []
    for N, items in sorted(buckets.items()):
        X = np.stack([it[2] for it in items]).astype("float32")   # (k, N)
        s = workflow.sample(conditions={"x": X}, num_samples=n_samples)
        k = len(items)
        mu_e.append(np.asarray(s["mu"]).reshape(k, n_samples, 1))
        sig_e.append(np.asarray(s["sigma"]).reshape(k, n_samples, 1))
        mu_t += [it[0] for it in items]
        sig_t += [it[1] for it in items]

    estimates = {"mu": np.concatenate(mu_e), "sigma": np.concatenate(sig_e)}
    targets = {"mu": np.array(mu_t).reshape(-1, 1),
               "sigma": np.array(sig_t).reshape(-1, 1)}
    return estimates, targets


def _metric(fn, estimates, targets) -> dict:
    """Reduce a BayesFlow metric function to {variable_name: value}."""
    r = fn(estimates, targets)
    return dict(zip(r["variable_names"], np.asarray(r["values"]).ravel()))


def report_metrics(estimates, targets) -> None:
    rmse = _metric(bfd.metrics.root_mean_squared_error, estimates, targets)
    corr = _metric(bfd.metrics.correlation, estimates, targets)
    cal = _metric(bfd.metrics.calibration_error, estimates, targets)
    zsc = _metric(bfd.metrics.posterior_z_score, estimates, targets)
    con = _metric(bfd.metrics.posterior_contraction, estimates, targets)
    print("  (lower NRMSE / higher corr = better recovery; calibration error ~0 = "
          "calibrated;\n   contraction near 1 = informative, near 0 = poorly "
          "identified; mean z near 0 = unbiased)")
    for name in ("mu", "sigma"):
        print(f"  {name:5s}  NRMSE={rmse[name]:.3f}  corr={corr[name]:.3f}  "
              f"cal_err={cal[name]:.3f}  | contraction={con[name]:.3f}  "
              f"mean z={zsc[name]:+.3f}")


def grid_crosscheck(workflow, n_datasets: int = 4, n_samples: int = 1000, seed: int = 11):
    np.random.seed(seed)
    rows = []
    for _ in range(n_datasets):
        p, m = prior(), meta()
        obs = likelihood(p["mu"], p["sigma"], m["N"])
        s = workflow.sample(conditions={"x": obs["x"][None, :].astype("float32")},
                            num_samples=n_samples)
        bf_mu = float(np.asarray(s["mu"]).reshape(-1).mean())
        bf_sigma = float(np.asarray(s["sigma"]).reshape(-1).mean())
        ref = diag.reference_posterior(obs["x"])
        rows.append(dict(N=m["N"], true_mu=p["mu"], true_sigma=p["sigma"],
                         bf_mu=bf_mu, grid_mu=ref["mu_mean"],
                         bf_sigma=bf_sigma, grid_sigma=ref["sigma_mean"]))
    return rows


def _maybe_plot(estimates, targets) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401  (ensures a backend is set)
    except ImportError:
        print("(matplotlib not available — skipping plots)")
        return

    os.makedirs(OUTDIR, exist_ok=True)
    specs = [
        ("recovery", bfd.recovery, "recovery.png"),
        ("calibration (SBC-ECDF)", bfd.calibration_ecdf, "calibration_ecdf.png"),
        ("z-score vs contraction", bfd.z_score_contraction, "z_score_contraction.png"),
    ]
    for label, fn, fname in specs:
        fig = fn(estimates, targets)
        path = os.path.join(OUTDIR, fname)
        fig.savefig(path, dpi=110)
        print(f"  saved {label:26s} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuller training + inference check.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()

    print(f"Training: epochs={args.epochs} batch_size={args.batch_size} "
          f"num_batches={args.num_batches}")
    workflow = build_workflow()
    workflow.fit_online(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_batches_per_epoch=args.num_batches,
    )

    print("\n=== Recovery, calibration & sensitivity (bayesflow.diagnostics) ===")
    estimates, targets = collect(workflow, n_test=args.n_test, n_samples=args.n_samples)
    report_metrics(estimates, targets)

    print("\n=== Grid cross-check (BayesFlow vs exact grid posterior mean) ===")
    for r in grid_crosscheck(workflow):
        print(f"  N={r['N']:2d}  mu: true={r['true_mu']:+.2f} bf={r['bf_mu']:+.2f} "
              f"grid={r['grid_mu']:+.2f}   sigma: true={r['true_sigma']:.2f} "
              f"bf={r['bf_sigma']:.2f} grid={r['grid_sigma']:.2f}")

    print("\n=== Plots ===")
    _maybe_plot(estimates, targets)


if __name__ == "__main__":
    main()
