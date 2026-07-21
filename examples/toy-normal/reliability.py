"""Reliability / out-of-distribution (OOD) check for the toy Normal model (stage 8).

This is the worked example of **stage 8** of the workflow (the `real-data-reliability`
skill): before trusting an amortised posterior, check that the real data is *typical*
under the training / prior-predictive distribution. An amortised network only ever
learned inference over the region the prior samples; on data unlike anything it trained
on it extrapolates and returns a confidently wrong posterior with **no error** — the
failure mode specific to amortised SBI, and the reason this check gates stage 9.

How it works, and how it uses BayesFlow:

- **Embeddings and the distance metric are BayesFlow's.** Each dataset is embedded with
  the trained summary network via ``approximator.summarize``, and datasets are compared
  in that summary space by Maximum Mean Discrepancy
  (``bayesflow.metrics.functional.maximum_mean_discrepancy``). This is the Schmitt et al.
  (2023) misspecification signal.
- **The null distribution is built from the simulator (a Monte Carlo null).** We draw
  many *fresh, independent* in-distribution samples from the model and record each one's
  MMD to a fixed prior-predictive reference. Because a genuine in-distribution
  observation is itself just another independent prior-predictive sample, this null is
  **calibrated**: a typical observation gives a p-value that is not small, and only a
  genuinely OOD observation is flagged. Building the null from the simulator is the
  natural, correct choice in an SBI setting, where the simulator is available by
  construction. (BayesFlow ships a convenience wrapper, ``summary_space_comparison``,
  that instead bootstraps the null by *resampling the reference set*; that null shares
  points with the reference and is mildly anti-conservative, so we build the simulator
  null here and use BayesFlow's MMD directly.)

Two 'observed' samples are checked against the same reference and null:

- **in-distribution** — fresh draws from the model itself. Its MMD sits within the null
  (p not small): the data is typical, the posterior can be trusted, proceed to stage 9.
- **out-of-distribution** — draws whose mean is far outside the Normal(0, 1) prior on
  ``mu``. Its MMD lands far beyond the null (p = 0): the training distribution did not
  cover the real data, so route back to **stage 2** (widen the prior / simulator range)
  — a *scientific* decision for the human — then retrain. Do NOT interpret the OOD posterior.

Note on single observations: the MMD test needs a *sample* of datasets. With a single
real dataset, use the Mahalanobis-distance-to-the-cloud variant instead (see the
`real-data-reliability` skill) — the embeddings still come from ``approximator.summarize``.

The observation set size is fixed within the trained N range (5-20) so datasets stack
into one array; a set size outside that range would itself be OOD for the summary network.

Run (needs a backend):

    KERAS_BACKEND=jax .venv/bin/python examples/toy-normal/reliability.py

Plots (if matplotlib is available) are written to ``outputs/`` next to this file.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from keras.ops import convert_to_numpy, convert_to_tensor

import bayesflow.diagnostics as bfd
from bayesflow.metrics.functional import maximum_mean_discrepancy
from train import build_workflow

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
OBS_N = 15  # fixed set size, within the trained 5-20 range


def sample_datasets(n_sets: int, seed: int, ood_shift: float = 0.0) -> dict:
    """``n_sets`` datasets of OBS_N i.i.d. Normal draws. ``ood_shift`` = 0 reproduces the
    model's own prior (in-distribution: mu ~ N(0,1), sigma ~ HalfNormal(1)); a large shift
    moves the location far outside the prior on mu (out-of-distribution)."""
    rng = np.random.default_rng(seed)
    xs = []
    for _ in range(n_sets):
        mu = rng.normal(0.0, 1.0) + ood_shift
        sigma = abs(rng.normal(0.0, 1.0))
        xs.append(rng.normal(mu, sigma, size=OBS_N))
    return {"x": np.asarray(xs, dtype="float32")}


def summaries(approximator, data: dict) -> np.ndarray:
    """Embed each dataset with the trained summary network."""
    return convert_to_numpy(approximator.summarize(data))


def mmd(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Maximum Mean Discrepancy between two sets of summary embeddings (BayesFlow's)."""
    return float(
        convert_to_numpy(
            maximum_mean_discrepancy(
                convert_to_tensor(emb_a, dtype="float32"),
                convert_to_tensor(emb_b, dtype="float32"),
            )
        )
    )


def calibrated_null(approximator, ref_summaries, n_observed, num_null, seed0):
    """Monte Carlo null: MMD-to-reference for many FRESH in-distribution samples drawn
    from the simulator. Calibrated, because a real in-distribution observation is itself
    just another independent prior-predictive sample."""
    null = np.empty(num_null, dtype=float)
    for i in range(num_null):
        fresh = sample_datasets(n_observed, seed=seed0 + i, ood_shift=0.0)
        null[i] = mmd(summaries(approximator, fresh), ref_summaries)
    return null


def _maybe_plot(d_null, d_obs, name: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
        return
    os.makedirs(OUTDIR, exist_ok=True)
    fig = bfd.mmd_hypothesis_test(mmd_null=d_null, mmd_observed=d_obs, alpha_level=0.05)
    path = os.path.join(OUTDIR, f"reliability_mmd_{name}.png")
    fig.savefig(path, dpi=110)
    print(f"  saved reliability MMD ({name:8s}) -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliability / OOD check (stage 8).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--n-reference", type=int, default=800, help="prior-predictive reference sample")
    parser.add_argument("--n-observed", type=int, default=200, help="observed / per-null sample size")
    parser.add_argument("--num-null", type=int, default=200, help="Monte Carlo null draws (from the simulator)")
    parser.add_argument("--ood-shift", type=float, default=6.0,
                        help="location shift for the OOD sample (mu is ~this far outside the prior)")
    args = parser.parse_args()

    print(f"Training: epochs={args.epochs} batch_size={args.batch_size} "
          f"num_batches={args.num_batches}")
    workflow = build_workflow()
    workflow.fit_online(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_batches_per_epoch=args.num_batches,
    )
    approximator = workflow.approximator

    reference = sample_datasets(args.n_reference, seed=1, ood_shift=0.0)
    ref_summaries = summaries(approximator, reference)

    print(f"\n=== Reliability / OOD check (summary-space MMD, simulator-calibrated null) ===")
    print(f"  reference: {args.n_reference} prior-predictive datasets | "
          f"observed: {args.n_observed} each | null: {args.num_null} fresh draws | N={OBS_N}")
    print("  (p = P(null MMD >= observed MMD); small p => atypical => OOD)")

    null = calibrated_null(approximator, ref_summaries, args.n_observed, args.num_null, seed0=1000)
    obs_in = sample_datasets(args.n_observed, seed=2, ood_shift=0.0)
    obs_ood = sample_datasets(args.n_observed, seed=3, ood_shift=args.ood_shift)

    d_in = mmd(summaries(approximator, obs_in), ref_summaries)
    d_ood = mmd(summaries(approximator, obs_ood), ref_summaries)
    p_in = float(np.mean(null >= d_in))
    p_ood = float(np.mean(null >= d_ood))
    null95 = float(np.percentile(null, 95))

    def verdict(p):
        return "OOD — do NOT trust the posterior (route to stage 2)" if p < 0.05 else "typical — trustworthy"

    print(f"\n  null MMD 95th pct = {null95:.4f}")
    print(f"  in-distribution : MMD={d_in:.4f}  p={p_in:.3f}  -> {verdict(p_in)}")
    print(f"  out-of-dist     : MMD={d_ood:.4f}  p={p_ood:.3f}  -> {verdict(p_ood)}")
    print(f"\n  MMD separation: OOD is {d_ood / d_in:.0f}x the in-distribution MMD.")

    print("\n=== Plots (BayesFlow mmd_hypothesis_test) ===")
    _maybe_plot(null, d_in, "in")
    _maybe_plot(null, d_ood, "ood")


if __name__ == "__main__":
    main()
