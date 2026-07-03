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

## Exchangeable / variable-length observations → as_set + DeepSet

When each dataset is an unordered set of i.i.d. observations (optionally with a
variable count), treat it as a set and summarise it with a permutation-invariant
network:

    adapter.as_set("x").rename("x", "summary_variables")
    summary_network = bf.networks.DeepSet(summary_dim=16, depth=1)

DeepSet maps any number of observations to a fixed-size embedding, so the number
of observations N can differ between datasets.

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

Judge recovery **relative to the posterior's own uncertainty**, not as point
accuracy: correlation/RMSE of the posterior *mean* are width-blind and look "broken"
whenever the data are uninformative, even when inference is perfect. Report
**posterior contraction** (1 − Var_post/Var_prior) and the **posterior z-score**
((mean − truth)/sd_post) alongside them — low contraction with small |z| **and**
uniform SBC is a genuinely poorly-identified parameter (a correct, honest-wide
posterior), not an engine bug, so don't retrain to "fix" it. See the
`workflow-orchestration` skill (stage 6) for the sensitivity-plot quadrants and
references.

Where the problem is low-dimensional, cross-check the amortised posterior against
an **exact reference** (a grid or analytic posterior) — the strongest possible
check. Build such references from validated libraries (`scipy.stats`) rather than
hand-coding densities, which is error-prone (easy to drop a normalising constant).

## Backend

BayesFlow 2 runs on Keras 3 and needs a backend. Select it before importing
anything that pulls in Keras:

    KERAS_BACKEND=jax python your_script.py     # or tensorflow / torch

A backend needs Python 3.11 or 3.12 (as of writing, not 3.14).
