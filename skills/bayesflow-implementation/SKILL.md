---
name: bayesflow-implementation
description: Practical, validated BayesFlow 2 implementation tips for building amortised SBI workflows — adapters, summary networks, constraining parameters, and how to verify inference. Use when scaffolding or debugging a BayesFlow posterior approximator.
---

# BayesFlow 2 implementation notes

Validated, hard-won guidance for implementing amortised simulation-based inference
with BayesFlow 2 in a user's project. Each item states the fix and *why*. The
worked, tested reference is `examples/toy-normal/` in the guidance kit (infer the
mean and standard deviation of a Normal from a variable number of observations).

These are **engineering** tips. They never override the **scientific** decisions —
priors, model structure, data handling — which require the human's approval.

## Constrain positive / bounded parameters

The inference network (a normalising flow) models parameters in unconstrained ℝ.
A positive parameter such as a scale / standard deviation must be constrained, or
the posterior is biased near the boundary:

    adapter.constrain("sigma", lower=0)   # before concatenating into inference_variables

Without this, `sigma` in the toy example had good *recovery* but a *biased* SBC
(rank mean ≈ 0.48 with a visible skew); adding the constraint centred it (≈ 0.50,
flat histogram). Note: standardising inference variables (which `BasicWorkflow`
does by default) is only an affine rescale and does **not** fix a hard boundary —
you need the constraint. Use `lower=`/`upper=` for other bounded parameters
(e.g. probabilities in [0, 1]).

The constraint must match the parameter's **support**, which is a modelling choice
fixed when the prior is chosen — keep it consistent with what the `prior-elicitation`
skill recorded (a positive scale gets a positive-support prior *and* `lower=0`).

## Exchangeable / variable-length observations → as_set + DeepSet

When each dataset is an unordered set of i.i.d. observations (optionally with a
variable count), treat it as a set and summarise it with a permutation-invariant
network:

    adapter.as_set("x").rename("x", "summary_variables")
    summary_network = bf.networks.DeepSet(summary_dim=16, depth=1)

DeepSet maps any number of observations to a fixed-size embedding, so the number
of observations N can differ between datasets.

## Ordered time series → TimeSeriesNetwork / TimeSeriesTransformer

When each dataset is an *ordered* sequence (not an exchangeable set), summarise it
with a sequence network, which reduces any length-T series to a fixed embedding:

    adapter.as_time_series("y").rename("y", "summary_variables")
    summary_network = bf.networks.TimeSeriesNetwork(summary_dim=48)   # LSTNet: GRU + skip-convs
    # or bf.networks.TimeSeriesTransformer(summary_dim=48, time_axis=1)  # attention, accepts attention_mask

**Variable observed length / forecasting-from-partial-data.** There is no automatic
ragged batching. Pad every series to `T_max` and add a **mask channel** that is 1 on
observed weeks and 0 on padding; concatenate `[value, mask]` into a 2-channel
`summary_variables` and randomise the observed length per batch at training time. One
estimator then serves every observed length (e.g. every monthly forecast cutoff). The
flu example does exactly this (`examples/flu-sir/train.py`, `simulator.observed_and_mask`).

## Latent states / hierarchical models → infer them, don't bolt on another sampler

The single most important scope rule for these models: **latent states and per-unit
parameters are inference targets of the *same* amortised posterior — not something a
separate MCMC / particle filter / custom sampler reconstructs afterwards.** Nesting a
hand-built sampler inside the SBI defeats the purpose of amortisation and is a
scientific-method change that needs explicit human sign-off (see `workflow-orchestration`
stage 4). Two supported patterns, both pure BayesFlow:

- **Latent trajectory as a target.** Concatenate the latent path (e.g. weekly
  log-infections `i₁..i_T`) *with* the globals into `inference_variables`, conditioned on
  the observed series. A `CouplingFlow` (or `FlowMatching` / `DiffusionModel` for a
  higher-dim target) then returns joint posterior draws of `[θ, latent path]`; forecasts
  are the reporting/observation pushforward of the posterior *future* latent states — a
  prediction step, still no second inference method. Validate with **per-step SBC** on the
  trajectory, not just the globals. Worked in `examples/flu-sir/`.
  *If the full-trajectory target won't calibrate* (per-step SBC fails on the far-ahead
  steps because the target is too high-dimensional), fall back to inferring only a
  **fixed-dim boundary state** `[θ, state at the last observed step]` and forecast by
  **forward-simulating the model** from those SBI-inferred boundary draws. That is still
  pure SBI — forward simulation from posterior draws is prediction, not a second inference
  method — it just trades the in-sample latent reconstruction for a lower-dimensional,
  more robust target.
- **Grouped / hierarchical parameters.** `bf.simulators.HierarchicalSimulator([global_sim,
  local_sim])` draws exchangeable global→local levels (`.sample((D, G))` → globals `(D,·)`,
  locals `(D, G, ·)`); the inner sim receives the outer draws by keyword. Use it to pool
  across exchangeable units (states, sites) sharing hyperpriors.

**Composition (`compositional_sample`) is for i.i.d. units only.** `DiffusionModel`'s
`compositional_score` + `CompositionalApproximator.compositional_sample` combine evidence
across **exchangeable** groups (score `(1−n)(1−t)∇log p(θ) + Σᵢ s(θ,t,yᵢ)`). That sum is
correct for conditionally-i.i.d. datasets but **wrong for a Markov temporal chain** (it has
no transition term) — don't reach for it to "stitch" a time series. There is **no
Simformer / arbitrary-subset conditioning** in BayesFlow 2.0.12 (`bf.experimental` is only
`FreeFormFlow`), so "condition on the past, sample the future in one masked network" is not
available; use the latent-trajectory-target pattern above instead. See the BayesFlow
Compositional Diffusion example, and `examples/flu-sir/` (with `ENGINEERING_LOG.md` §3) as
the worked case.

## Adapter hygiene

- Call `.to_array()` before `.convert_dtype(...)`. Raw simulator outputs may
  include Python scalars (e.g. an integer sample size `N`); `convert_dtype` calls
  `.astype` and crashes on a bare `int`.
- `.drop(...)` any simulator variables you do not use as conditions (e.g. a meta
  variable `N` that only sizes each simulation), so they do not flow into
  transforms or the network.

## Variable set sizes and JIT recompilation (JAX)

With `bf.simulators.make_simulator(..., meta_fn=...)`, the meta context (e.g. `N`)
is drawn once per batch: constant within a batch, varying across batches. That is
what makes variable-length sets work. But JAX recompiles once per distinct set
size, so the first epoch is slow and later epochs are fast (sizes get cached). If
there are many possible sizes, expect a longer warm-up — or fix / pad the size.

## Verify inference: recovery AND calibration

Good parameter recovery is necessary but not sufficient — a posterior can track
the truth on average while being mis-calibrated. Always also run simulation-based
calibration (SBC) and check the rank histograms are roughly uniform. In the toy
example, `sigma` had strong recovery yet biased SBC until it was constrained.

**Use BayesFlow's own diagnostics — do not hand-roll these metrics.** The
`bayesflow.diagnostics` module takes `estimates` (a dict of posterior draws, shape
`(datasets, samples, dim)`) and `targets` (a dict of true values). `estimates` is what
`workflow.sample(...)` returns; you supply `targets` (the known true parameters) yourself,
built alongside the draws (see `examples/toy-normal/run_validation.py`). It provides:

    import bayesflow.diagnostics as bfd
    bfd.metrics.root_mean_squared_error(estimates, targets)   # recovery
    bfd.metrics.correlation(estimates, targets)
    bfd.metrics.calibration_error(estimates, targets)         # SBC (ECDF-based)
    bfd.metrics.posterior_z_score(estimates, targets)         # sensitivity
    bfd.metrics.posterior_contraction(estimates, targets)
    bfd.recovery(estimates, targets)                          # plots (matplotlib Figures)
    bfd.calibration_ecdf(estimates, targets)
    bfd.z_score_contraction(estimates, targets)

Judge recovery **relative to the posterior's own uncertainty**, not as point
accuracy: correlation/RMSE of the posterior *mean* are width-blind and look "broken"
whenever the data are uninformative, even when inference is perfect. That is why you
also read **posterior contraction** (1 − Var_post/Var_prior) and the **posterior
z-score** ((mean − truth)/sd_post): low contraction with small |z| **and** low
calibration error is a genuinely poorly-identified parameter (a correct, honest-wide
posterior), not an engine bug, so don't retrain to "fix" it. `bfd.z_score_contraction`
plots exactly this. See the `workflow-orchestration` skill (stage 6) for the
sensitivity-plot quadrants and references.

The one thing BayesFlow cannot give you is an **exact reference posterior**. Where
the problem is low-dimensional, cross-check the amortised posterior against a grid or
analytic posterior — the strongest possible check. Build such references from
validated libraries (`scipy.stats`) rather than hand-coding densities, which is
error-prone (easy to drop a normalising constant). The toy example's `diagnostics.py`
is a worked instance.

BayesFlow also has no **posterior predictive check** (its diagnostics are all
parameter/summary-space). For stage 9, use **ArviZ** (`plot_ppc_dist`,
`plot_ppc_tstat`, `plot_ppc_pit`) with replicates re-simulated from the model's own
likelihood — see the `posterior-predictive-check` skill and the worked
`examples/toy-normal/posterior_predictive.py`.

## Backend

BayesFlow 2 runs on Keras 3 and needs a backend. Select it before importing
anything that pulls in Keras:

    KERAS_BACKEND=jax python your_script.py     # or tensorflow / torch

A backend needs Python 3.11 or 3.12 (as of writing, not 3.14).
