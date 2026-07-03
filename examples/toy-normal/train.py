"""Train an amortised posterior for the toy Normal model with BayesFlow 2.

This is the ONLY file in the example that requires BayesFlow and a Keras 3
backend. It is written against the BayesFlow 2.x API (mirroring the official
``One_Sample_TTest`` and ``Two_Moons_Starter`` examples). Run it inside an
isolated environment as described in ``README.md``.

What it demonstrates (the toy-benchmark checklist in
``validation/benchmark-matrix.md``):

- prior-predictive simulation via the simulator in ``simulator.py``;
- a training smoke test (a few short epochs);
- a posterior-sampling smoke test;
- structure for parameter recovery and posterior-predictive checks
  (see ``diagnostics.py`` for the backend-free reference posterior).

Keep the network small and epochs low: this is a smoke test, not a benchmark of
accuracy. Increase ``--epochs`` / network size for a serious run.
"""

from __future__ import annotations

import argparse

import numpy as np

from simulator import make_simulator


def build_workflow():
    """Assemble simulator + adapter + summary/inference networks."""
    import bayesflow as bf

    simulator = make_simulator()

    # The observations `x` are an exchangeable, variable-length set (N = 5..20).
    # `.as_set("x")` tells BayesFlow to treat them as a set so the DeepSet
    # summary network can map them to a fixed-size embedding regardless of N.
    # `N` (from the meta function) only sizes each simulation and is not used as
    # a condition here, so we drop it. `.to_array()` first so every field is a
    # NumPy array before dtype conversion.
    #
    # `sigma` is a positive scale parameter, but the CouplingFlow models the
    # inference variables in unconstrained R. Without a constraint the flow can
    # place mass that maps to sigma <= 0 and is biased near the sigma=0 boundary
    # (where the HalfNormal prior has its mode). `.constrain("sigma", lower=0)`
    # maps sigma to unconstrained space for the flow and inverts posterior draws
    # back to sigma > 0. (Inference variables are already standardized by the
    # BasicWorkflow default, which is only an affine rescale and does not fix the
    # boundary — hence the constraint is what removes the sigma bias.)
    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .drop("N")
        .convert_dtype("float64", "float32")
        .as_set("x")
        .rename("x", "summary_variables")
        .constrain("sigma", lower=0)
        .concatenate(["mu", "sigma"], into="inference_variables")
    )

    summary_network = bf.networks.DeepSet(summary_dim=16, depth=1)
    inference_network = bf.networks.CouplingFlow()

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_network,
        inference_network=inference_network,
    )
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the toy Normal posterior.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run for CI: 2 epochs, small batches.")
    args = parser.parse_args()

    if args.smoke:
        args.epochs, args.num_batches, args.batch_size = 2, 10, 32

    workflow = build_workflow()

    workflow.fit_online(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_batches_per_epoch=args.num_batches,
    )

    # Posterior-sampling smoke test on one freshly simulated dataset.
    from simulator import prior, meta, likelihood

    np.random.seed(0)
    p, m = prior(), meta()
    obs = likelihood(p["mu"], p["sigma"], m["N"])
    samples = workflow.sample(conditions={"x": obs["x"][None, :]}, num_samples=500)

    mu_draws = np.asarray(samples["mu"]).reshape(-1)
    sigma_draws = np.asarray(samples["sigma"]).reshape(-1)
    print(f"true   mu={p['mu']:+.3f}  sigma={p['sigma']:.3f}  (N={m['N']})")
    print(f"post   mu={mu_draws.mean():+.3f}  sigma={sigma_draws.mean():.3f}")


if __name__ == "__main__":
    main()
