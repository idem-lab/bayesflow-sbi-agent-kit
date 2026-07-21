"""Stage 4-5 — build, train, and persist the amortised estimator (BayesFlow 2).

**Pure-SBI, trajectory target.** One estimator maps an observed weekly case series
(truncated at a monthly forecast cutoff ``L``, padded to ``T_MAX`` with a mask channel) →
a full joint posterior over the 7 global parameters θ **and** the latent weekly
log-infection trajectory ``log(iₜ+1)`` for every week ``t = 1..T_MAX``. The latent epidemic
states are inference *targets* — there is no particle filter and no MCMC. It is trained
once and reused for every state and every forecast date; the only expensive artefact, so it
is **saved to disk** (``outputs/flu_sir.keras``) for the recovery, calibration,
reliability, posterior-predictive, and forecasting steps to load without retraining.

From the one posterior:
  * in-sample fit (weeks ≤ L) = reporting-model pushforward of the posterior infections;
  * forecast (weeks > L) = reporting-model pushforward of the posterior *future* infections
    — which the flow has learned to extrapolate dynamically-consistently from the simulator.

Design (CPU-only machine — see ENGINEERING_LOG.md §1):

- **Condition / summary network:** a 2-channel weekly series ``[log1p-standardized cases,
  observed-week mask]`` → ``TimeSeriesNetwork`` (LSTNet: bidirectional GRU + skip
  convolutions) → fixed ``summary_dim`` embedding. The mask channel tells the network which
  weeks are observed vs. padded, so one estimator serves every cutoff ``L``.
- **Inference network:** ``CouplingFlow`` over the 46-dim target (full posterior samples,
  not quantiles). The approximator standardizes the concatenated inference variables.
- **Adapter:** log1p + standardize the case counts for the GRU; log-transform the 6 positive
  globals (esp. ``S0`` spanning 1e4-1e6) and logit ``rho``∈(0,1); the log-infection
  trajectory is already on a ~log scale and is concatenated after the globals.

    KERAS_BACKEND=jax .venv/bin/python examples/flu-sir/train.py --smoke   # pilot
    KERAS_BACKEND=jax .venv/bin/python examples/flu-sir/train.py           # full run + save
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import simulator as sim

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
# Full-model Keras serialization (network + fitted adapter). `.weights.h5` round-tripping
# fails to rebuild the coupling stack's layer paths across processes; `.keras` save_model /
# load_model preserves the whole approximator incl. its adapter (see ENGINEERING_LOG.md §5).
MODEL_PATH = os.path.join(OUTDIR, "flu_sir.keras")

POSITIVE6 = ["R0", "gamma", "sigma_lb", "delay_mean", "S0", "I0"]   # log-transformed
# log1p case-count scale in the pre-2018 data is ~N(3, 2); fixed standardization.
CASE_LOG_MEAN, CASE_LOG_STD = 3.0, 2.0
SUMMARY_DIM = 48
TRAJ_KEY = "log_inf"


def build_adapter():
    import bayesflow as bf
    return (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_time_series(["cases", "mask"])
        .log("cases", p1=True)
        .standardize("cases", mean=CASE_LOG_MEAN, std=CASE_LOG_STD)
        .concatenate(["cases", "mask"], into="summary_variables", axis=-1)   # (B, T, 2)
        .log(POSITIVE6)
        .constrain("rho", lower=0, upper=1)
        # target = [7 globals, T_MAX log-infections]; approximator standardizes internally.
        .concatenate([*sim.PARAM_NAMES, TRAJ_KEY], into="inference_variables")
    )


def build_workflow():
    import bayesflow as bf
    summary_network = bf.networks.TimeSeriesNetwork(
        summary_dim=SUMMARY_DIM, recurrent_dim=64, filters=32, bidirectional=True, dropout=0.05,
    )
    inference_network = bf.networks.CouplingFlow()
    return bf.BasicWorkflow(
        simulator=sim.make_simulator(),
        adapter=build_adapter(),
        summary_network=summary_network,
        inference_network=inference_network,
    )


def save_weights(workflow, path: str = MODEL_PATH) -> None:
    import keras
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keras.saving.save_model(workflow.approximator, path)
    print(f"  saved model -> {path}")


def load_trained(path: str = MODEL_PATH):
    """Load the trained approximator (network + fitted adapter) from ``.keras``. The returned
    object exposes ``.sample(conditions=<raw dict>, num_samples=...)`` — the same interface as
    the workflow — and applies its serialized adapter internally, so downstream scripts pass
    raw ``{"cases", "mask"}`` conditions exactly as at training."""
    import bayesflow  # noqa: F401 — registers the serializable approximator/adapter/network classes
    import keras
    return keras.saving.load_model(path)


def make_condition(cases_series: np.ndarray, L: int, t_max: int = sim.T_MAX) -> dict:
    """Build the estimator's condition from an observed weekly case series and cutoff ``L``:
    a (1, t_max, 1) padded/censored case array + observed-week mask, exactly as at training."""
    full = np.zeros((1, t_max), dtype="float32")
    full[0, :L] = np.asarray(cases_series, dtype="float32")[:L]
    cases, mask = sim.observed_and_mask(full, L, t_max)
    return {"cases": cases, "mask": mask}


def sample_posterior(workflow, cases_series: np.ndarray, L: int,
                     num_samples: int = 1000) -> dict:
    """Joint posterior draws for one observed series truncated at cutoff ``L``. Returns a
    dict with each global as a (num_samples,) natural-scale array and ``log_inf`` as a
    (num_samples, T_MAX) array of posterior log(infections+1) trajectories."""
    cond = make_condition(cases_series, L)
    draws = workflow.sample(conditions=cond, num_samples=num_samples)
    out = {}
    for k, v in draws.items():
        a = np.asarray(v)
        out[k] = a.reshape(num_samples, -1) if k == TRAJ_KEY else a.reshape(-1)
    return out


def infections_from_draws(post: dict) -> np.ndarray:
    """Posterior weekly true new-infection trajectories (num_samples, T_MAX) on the natural
    (count) scale, from the log_inf draws. Clipped at 0 (log_inf draws can dip slightly
    below 0, i.e. expm1 < 0, for near-zero weeks)."""
    return np.maximum(np.expm1(post[TRAJ_KEY]), 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-batches", type=int, default=250)
    ap.add_argument("--smoke", action="store_true", help="tiny pilot run")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.num_batches, args.batch_size = 3, 20, 64

    workflow = build_workflow()
    workflow.fit_online(epochs=args.epochs, batch_size=args.batch_size,
                        num_batches_per_epoch=args.num_batches)

    # Posterior-sampling smoke test: simulate one full season, observe to L=12, infer.
    rng = np.random.default_rng(0)
    truth = sim.sample_prior(1, rng)
    full = sim.simulate(truth, n_weeks=sim.T_MAX, rng=rng, return_latent=True)
    L = 12
    post = sample_posterior(workflow, full["cases"][0], L, num_samples=500)
    print(f"\nsmoke posterior on a full-season sim, observed to L={L} "
          f"(cases peak={full['cases'][0].max()}, total={full['cases'][0].sum()}):")
    for k in sim.PARAM_NAMES:
        print(f"  {k:11s} true={truth[k][0]:.4g}   post mean={post[k].mean():.4g}  "
              f"[{np.percentile(post[k],5):.4g}, {np.percentile(post[k],95):.4g}]")
    inf = infections_from_draws(post)                       # (500, T_MAX)
    true_inf = full["weekly_inf"][0]
    band = np.percentile(inf, [5, 95], axis=0)
    covered = np.mean((true_inf >= band[0]) & (true_inf <= band[1]))
    print(f"  latent-infection trajectory: {covered*100:.0f}% of weeks inside the 90% band "
          f"(in-sample weeks 1-{L} should be tight; forecast weeks {L+1}-{sim.T_MAX} wide)")

    if not args.no_save and not args.smoke:
        save_weights(workflow)


if __name__ == "__main__":
    main()
