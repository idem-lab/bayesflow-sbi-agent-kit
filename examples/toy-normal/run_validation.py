"""Fuller training run + inference check for the toy Normal model.

Trains the amortised posterior for a meaningful number of epochs, then checks
that inference actually works:

1. **Parameter recovery** — over many fresh test datasets, do posterior means
   track the true mu / sigma? (RMSE and correlation, plus a scatter plot.)
2. **Simulation-based calibration (SBC)** — are the rank statistics roughly
   uniform? (Systematic non-uniformity means the posteriors are mis-calibrated.)
3. **Grid cross-check** — on a few datasets, does the BayesFlow posterior mean
   agree with the exact grid-reference posterior from ``diagnostics.py``?

Run (needs a backend):

    KERAS_BACKEND=jax .venv/bin/python examples/toy-normal/run_validation.py

Evaluation is batched by set size N so that JAX compiles at most once per N.
Plots (if matplotlib is available) are written to ``outputs/`` next to this file.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

from simulator import prior, meta, likelihood
import diagnostics as diag
from train import build_workflow

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")


def _draws(samples_key) -> np.ndarray:
    """Flatten one dataset's posterior draws to 1-D."""
    return np.asarray(samples_key).reshape(-1)


def evaluate(workflow, n_test: int, n_samples: int, seed: int = 7) -> dict:
    np.random.seed(seed)

    # Draw the test set, then bucket by N so we can sample each shape in one call.
    buckets: dict[int, list] = defaultdict(list)
    for _ in range(n_test):
        p, m = prior(), meta()
        obs = likelihood(p["mu"], p["sigma"], m["N"])
        buckets[m["N"]].append((p["mu"], p["sigma"], obs["x"]))

    true = {"mu": [], "sigma": []}
    est = {"mu": [], "sigma": []}
    rank = {"mu": [], "sigma": []}

    for N, items in sorted(buckets.items()):
        X = np.stack([it[2] for it in items]).astype("float32")   # (k, N)
        samples = workflow.sample(conditions={"x": X}, num_samples=n_samples)
        mu_s = np.asarray(samples["mu"])       # (k, n_samples[, 1])
        sig_s = np.asarray(samples["sigma"])
        for i, (mu_t, sig_t, _) in enumerate(items):
            mud, sgd = mu_s[i].reshape(-1), sig_s[i].reshape(-1)
            true["mu"].append(mu_t);   est["mu"].append(mud.mean())
            true["sigma"].append(sig_t); est["sigma"].append(sgd.mean())
            rank["mu"].append(diag.sbc_rank(mu_t, mud))
            rank["sigma"].append(diag.sbc_rank(sig_t, sgd))

    out = {"n_test": n_test, "n_samples": n_samples, "recovery": {}, "sbc": {}}
    for name in ("mu", "sigma"):
        t, e = np.array(true[name]), np.array(est[name])
        out["recovery"][name] = diag.recovery_summary(t, e)
        r = np.array(rank[name]) / n_samples          # normalise ranks to [0, 1]
        # For a calibrated posterior these are ~Uniform(0,1): mean ~0.5, std ~0.289.
        out["sbc"][name] = dict(mean=float(r.mean()), std=float(r.std()))
        out[f"_scatter_{name}"] = (t, e)
        out[f"_ranks_{name}"] = r
    return out


def grid_crosscheck(workflow, n_datasets: int = 4, n_samples: int = 1000, seed: int = 11):
    np.random.seed(seed)
    rows = []
    for _ in range(n_datasets):
        p, m = prior(), meta()
        obs = likelihood(p["mu"], p["sigma"], m["N"])
        s = workflow.sample(conditions={"x": obs["x"][None, :].astype("float32")},
                            num_samples=n_samples)
        bf_mu = float(_draws(s["mu"]).mean())
        bf_sigma = float(_draws(s["sigma"]).mean())
        ref = diag.reference_posterior(obs["x"])
        rows.append(dict(N=m["N"], true_mu=p["mu"], true_sigma=p["sigma"],
                         bf_mu=bf_mu, grid_mu=ref["mu_mean"],
                         bf_sigma=bf_sigma, grid_sigma=ref["sigma_mean"]))
    return rows


def _maybe_plot(results) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available — skipping plots)")
        return

    os.makedirs(OUTDIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for col, name in enumerate(("mu", "sigma")):
        t, e = results[f"_scatter_{name}"]
        ax = axes[0, col]
        lo, hi = min(t.min(), e.min()), max(t.max(), e.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.scatter(t, e, s=10, alpha=0.5)
        ax.set(title=f"Recovery: {name}", xlabel="true", ylabel="posterior mean")
        ax = axes[1, col]
        ax.hist(results[f"_ranks_{name}"], bins=20, range=(0, 1), color="C0", alpha=0.8)
        ax.axhline(results["n_test"] / 20, color="k", ls="--", lw=1)
        ax.set(title=f"SBC ranks: {name}", xlabel="normalised rank", ylabel="count")
    fig.tight_layout()
    path = os.path.join(OUTDIR, "validation.png")
    fig.savefig(path, dpi=110)
    print(f"\nSaved plots to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuller training + inference check.")
    parser.add_argument("--epochs", type=int, default=30)
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

    print("\n=== Parameter recovery & SBC ===")
    results = evaluate(workflow, n_test=args.n_test, n_samples=args.n_samples)
    for name in ("mu", "sigma"):
        rec, sbc = results["recovery"][name], results["sbc"][name]
        print(f"  {name:5s}  RMSE={rec['rmse']:.3f}  corr={rec['correlation']:.3f}  "
              f"| SBC rank mean={sbc['mean']:.3f} (~0.50)  std={sbc['std']:.3f} (~0.289)")

    print("\n=== Grid cross-check (BayesFlow vs exact grid posterior mean) ===")
    for r in grid_crosscheck(workflow):
        print(f"  N={r['N']:2d}  mu: true={r['true_mu']:+.2f} bf={r['bf_mu']:+.2f} "
              f"grid={r['grid_mu']:+.2f}   sigma: true={r['true_sigma']:.2f} "
              f"bf={r['bf_sigma']:.2f} grid={r['grid_sigma']:.2f}")

    _maybe_plot(results)


if __name__ == "__main__":
    main()
