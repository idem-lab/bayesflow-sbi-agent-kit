"""Reliability / out-of-distribution (OOD) check for the toy Normal model (stage 8).

This is the worked example of **stage 8** of the workflow (the `real-data-reliability`
skill): before trusting an amortised posterior, check that the real data is *typical*
under the training / prior-predictive distribution. An amortised network only ever
learned inference over the region the prior samples; on data unlike anything it trained
on it extrapolates and returns a confidently wrong posterior with **no error** — the
failure mode specific to amortised SBI, and the reason this check gates stage 9.

Design choices, matching the skill and the "use BayesFlow functionality" principle:

- **The OOD metric is BayesFlow's own** ``summary_space_comparison`` (Schmitt et al.
  2023): it embeds each dataset with the trained summary network and compares an
  observed sample against a prior-predictive reference sample by Maximum Mean
  Discrepancy (MMD), building a bootstrap null distribution for a hypothesis test.
  ``mmd_hypothesis_test`` plots the observed MMD against that null. We do not hand-roll
  the metric.

  *Reading the result.* BayesFlow's bootstrap null resamples the reference set itself,
  so it shares points with the reference and is mildly **anti-conservative**: an
  independent in-distribution sample can sit just past its 95th percentile even though
  it is perfectly typical. So judge by the **MMD magnitude relative to the null's
  scale** (and the plot), not the raw p-value alone: a truly OOD sample lands *orders of
  magnitude* beyond the null (here ~50×), which no bootstrap artefact can produce, while
  an in-distribution sample stays at the null's scale (~1×). The OOD signal is
  unambiguous either way.

Two 'observed' samples are checked against the same prior-predictive reference:

- **in-distribution** — fresh draws from the model itself. Its MMD sits with the null
  (the data is typical); the posterior can be trusted, so proceed to stage 9.
- **out-of-distribution** — draws whose mean is far outside the Normal(0, 1) prior on
  ``mu``. Its MMD lands far in the null's tail (flagged). This is the stage-8 failure:
  the training distribution did not cover the real data, so route back to **stage 2**
  (widen the prior / simulator range) — a *scientific* decision for the human — then
  retrain. Do NOT interpret the OOD posterior.

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

import bayesflow.diagnostics as bfd
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


def check(approximator, observed: dict, reference: dict, num_null: int):
    """MMD in summary space between observed and reference, with a bootstrap null."""
    d_obs, d_null = bfd.summary_space_comparison(
        observed, reference, approximator, num_null_samples=num_null
    )
    p = float(np.mean(d_null >= d_obs))
    return d_obs, d_null, p


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
    parser.add_argument("--n-observed", type=int, default=200, help="observed sample (<= reference)")
    parser.add_argument("--num-null", type=int, default=200, help="bootstrap null samples")
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
    obs_in = sample_datasets(args.n_observed, seed=2, ood_shift=0.0)
    obs_ood = sample_datasets(args.n_observed, seed=3, ood_shift=args.ood_shift)

    print(f"\n=== Reliability / OOD check (summary-space MMD vs prior-predictive reference) ===")
    print(f"  reference: {args.n_reference} prior-predictive datasets | "
          f"observed: {args.n_observed} each | N={OBS_N}")
    print("  Verdict by MMD magnitude vs the null's scale (ratio = MMD / null 95th pct);")
    print("  the bootstrap null is mildly anti-conservative, so p alone over-flags in-dist.")

    d_in, null_in, p_in = check(approximator, obs_in, reference, args.num_null)
    d_ood, null_ood, p_ood = check(approximator, obs_ood, reference, args.num_null)

    def report(label, d_obs, d_null, p):
        null95 = float(np.percentile(d_null, 95))
        ratio = d_obs / null95
        # Orders of magnitude beyond the null => OOD; at the null's scale => typical.
        verdict = ("OOD — do NOT trust the posterior (route to stage 2)" if ratio > 5.0
                   else "typical — trustworthy")
        print(f"  {label:15s}: MMD={d_obs:.4f}  null 95th pct={null95:.4f}  "
              f"ratio={ratio:5.1f}x  p={p:.3f}  -> {verdict}")

    print()
    report("in-distribution", d_in, null_in, p_in)
    report("out-of-dist", d_ood, null_ood, p_ood)
    print(f"\n  MMD separation: OOD is {d_ood / d_in:.0f}x the in-distribution MMD.")

    print("\n=== Plots (BayesFlow mmd_hypothesis_test) ===")
    _maybe_plot(null_in, d_in, "in")
    _maybe_plot(null_ood, d_ood, "ood")


if __name__ == "__main__":
    main()
