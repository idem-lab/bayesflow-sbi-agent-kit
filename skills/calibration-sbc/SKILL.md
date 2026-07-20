---
name: calibration-sbc
description: Runs the stage-7 calibration check with simulation-based calibration (SBC) — for many simulated datasets, are the ranks of the true values within their posterior draws uniform? Catches over/under-confidence and bias that parameter recovery misses, and is width-aware, so it certifies that an honestly-wide posterior is nonetheless calibrated. Uses the ECDF-based SBC variant via bayesflow.diagnostics. Use at stage 7, on simulated data only, read together with recovery (stage 6). On fail, route to retraining (an engineering fix); a poorly-identified parameter with uniform ranks is NOT a fail.
---

# Calibration — simulation-based calibration (SBC)

This is stage 7 of the `workflow-orchestration` map, the second check in Act II's
"does the method work?" block, and the partner to parameter recovery (stage 6). It
runs **entirely on simulated data**. Its question is sharper than recovery's: not
"is the posterior accurate on average?" but **"is the posterior *calibrated* — does
its stated uncertainty mean what it claims?"**

This is the check that catches what recovery misses. In the toy example, `sigma` had
strong recovery yet **biased** SBC until it was constrained — inspection would never
have caught it; running SBC did.

## What SBC does, and why it is width-aware

The idea (Talts et al. 2018): if you draw a true parameter from the prior, simulate a
dataset, and compute the **rank** of that true value among its own posterior draws,
then — *for a correctly calibrated posterior* — those ranks are **uniform** across
many simulated datasets. Deviations from uniformity have named shapes:

- **∩ (hump in the middle)** → posteriors too **wide** (under-confident).
- **∪ (piled at the ends)** → posteriors too **narrow** (over-confident).
- **slope / ramp** → **bias** (posteriors systematically off to one side).

The crucial property is that **SBC is width-aware**, and this is what makes it the
partner recovery needs. Uniform ranks certify that a posterior is honest *regardless
of how wide it is* — a genuinely wide posterior over a poorly-identified parameter
still produces uniform ranks. Non-uniform ranks flag a problem *regardless of width*.
So SBC is precisely the tool that separates:

- a genuinely **poorly-identified** parameter (recovery: low contraction, small |z|;
  SBC: **uniform**) — an honest, correct, wide posterior; from
- a **broken engine** (SBC: **non-uniform**) — biased or mis-dispersed, whatever its
  width.

That is why stages 6 and 7 must be read as one picture, never separately.

## Read recovery (stage 6) and SBC (stage 7) together

This combined table is the verdict step for both stages. Bring the stage-6 contraction
and |z| together with the stage-7 rank shape:

| Contraction (s6) | \|z\| (s6) | SBC ranks (s7) | Interpretation | Action |
|---|---|---|---|---|
| high | small | uniform | **ideal** — informative & calibrated | proceed to stage 8 |
| low | small | uniform | genuinely **poorly identified** — honest wide posterior | *scientific finding*; **do not retrain** |
| any | large | slope / ramp | **biased** engine | engineering → **stage 4/5** |
| high | small | ∪ | **over-confident** (too narrow) | more training / larger flow → **stage 5** |
| high | small | ∩ | **under-confident** (too wide) | usually more training → **stage 5** |
| low | large | (any non-uniform) | **prior–likelihood conflict** | *scientific* → **stage 2** |

The single most important row is the second: **uniform SBC with poor point-recovery is
NOT a failure** — it is the poorly-identified case, a scientific finding, and
retraining cannot help.

## Mechanics (BayesFlow)

Use BayesFlow's diagnostics — **do not hand-roll SBC.** The modern variant is
**ECDF-based** (Säilynoja, Bürkner & Vehtari 2022): instead of binning ranks into a
histogram (which is sensitive to bin count), it plots the empirical CDF of the ranks
with **simultaneous confidence bands**, so "inside the band = calibrated" is an
unambiguous read. From the same `(estimates, targets)` format as recovery:

    import bayesflow.diagnostics as bfd
    bfd.metrics.calibration_error(estimates, targets)   # scalar ECDF-based calibration error
    bfd.calibration_ecdf(estimates, targets)            # the ECDF plot with confidence bands

The worked, tested reference is `examples/toy-normal/run_validation.py`. See
`bayesflow-implementation` for shapes and the constraint mechanics.

**Practical notes:**

- **Enough datasets.** SBC needs many simulated datasets to resolve non-uniformity —
  hundreds at minimum, more for a confident read. Too few and everything looks
  "uniform" because the bands are wide.
- **Enough posterior draws per dataset.** The rank resolution depends on the number of
  posterior samples; use a healthy number (the toy uses hundreds).
- **Cross-check against an exact reference where you can.** For low-dimensional
  problems, compare against a grid or analytic posterior built from `scipy.stats`
  (never hand-coded densities) — the strongest check, and the payoff of a toy model.

## The commonest concrete fix: constrain bounded parameters

The single most frequent SBC failure in practice is a **bounded parameter fit in
unconstrained space**. The inference network (a normalising flow) works in
unconstrained ℝ; a positive parameter such as a scale / SD is then biased near its
boundary, which shows up as biased ranks even when recovery looks fine:

    adapter.constrain("sigma", lower=0)     # before concatenating into inference_variables

In the toy example this moved `sigma`'s rank mean from ≈ 0.48 with a visible skew to
≈ 0.50 with a flat histogram. Standardising the inference variables (which
`BasicWorkflow` does by default) is only an affine rescale and does **not** fix a hard
boundary — you need the constraint. The constraint must match the parameter's
*support*, which is a modelling choice fixed when the prior was chosen (see
`prior-elicitation` and `bayesflow-implementation`); keep them consistent.

## On fail — route by the shape

- **Biased ranks (slope)** → back to **stage 4** (add a constraint, fix the adapter or
  summary network) or **stage 5** (train more). Start with the constraint check above.
- **Over-/under-dispersion (∪ / ∩)** → usually more training or a larger flow →
  **stage 5**.
- **Uniform ranks with *poor* stage-6 point-recovery** → **not a fail.** This is the
  poorly-identified case (see the table): a scientific finding, surface it, do not
  retrain.
- **Non-uniform *and* stage-6 shows low contraction with large |z|** → this is a
  *scientific* signal (prior–likelihood conflict), not just an engine fault → surface
  to the human (**stage 2**); do not silently move the prior.

All the retraining routes are *engineering* and yours to do; the prior-conflict and
poorly-identified routes are *science* and the human's to decide — the same owner
split that governs the whole workflow.

## Using this skill

- **Entering from stage 6:** you already have the contraction and z-score per
  parameter. Run SBC on the same held-out simulated data and combine via the table
  above.
- **On a clean pass:** both parameters informative-and-calibrated (or honestly wide
  and calibrated) → proceed to **stage 8** (real-data inference + reliability/OOD
  check), the first stage that involves the real data.
- **Engineering mechanics** are in `bayesflow-implementation`; the worked reference is
  `examples/toy-normal/`.

## References

- Talts, Betancourt, Simpson, Vehtari & Gelman (2018), *Validating Bayesian Inference
  Algorithms with Simulation-Based Calibration* — the rank-uniformity check and the
  reason it is width-aware. arXiv:1804.06788 — https://arxiv.org/abs/1804.06788
- Säilynoja, Bürkner & Vehtari (2022), *Graphical Test for Discrete Uniformity and its
  Applications in Goodness-of-Fit Evaluation and Multiple Sample Comparison* — the
  ECDF-based SBC variant with simultaneous confidence bands used by
  `bayesflow.diagnostics.calibration_ecdf`. arXiv:2103.10522 —
  https://arxiv.org/abs/2103.10522
- Schad, Betancourt & Vasishth (2021), *Toward a principled Bayesian workflow in
  cognitive science* — computational faithfulness (SBC) as one of the four questions,
  and the sensitivity plot read alongside it (stage 6). arXiv:1904.12765 —
  https://arxiv.org/abs/1904.12765
