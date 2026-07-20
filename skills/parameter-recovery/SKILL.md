---
name: parameter-recovery
description: Runs the stage-6 parameter-recovery check — on held-out simulated datasets, does the posterior recover the true parameters relative to its own uncertainty (never as point accuracy)? Reports posterior contraction and posterior z-score per parameter, reads the z-score-vs-contraction sensitivity plot, and separates a genuinely poorly-identified parameter (a scientific finding) from a biased inference engine (an engineering fix). Use at stage 6, on simulated data only, and read together with SBC (stage 7). On fail, decide which failure it is before acting.
---

# Parameter recovery

This is stage 6 of the `workflow-orchestration` map, the first check in Act II's
"does the method work?" block. It runs **entirely on simulated data** — held-out
draws from the prior predictive, never the real observations — and asks: given data
the model itself generated from known parameters, does the trained posterior recover
those parameters?

The whole point of Act II is to earn trust in the method *before* pointing it at real
data. Recovery is the first of two paired checks (recovery + calibration/SBC); read
them **together** (see *Read recovery and SBC together* below), never recovery alone.

## The one idea that makes recovery honest: judge it relative to uncertainty

**Recovery is not point accuracy.** The instinct is to correlate the posterior
*mean* against the truth and compute RMSE, and call high correlation "good recovery."
That instinct is wrong, and acting on it is the most common way to misread this
stage. Correlation and RMSE of the posterior mean are **width-blind**: they look only
at the point estimate and ignore the posterior's own stated uncertainty.

Here is the failure of that view. When the data are genuinely **uninformative** about
a parameter, the posterior correctly collapses back onto the *prior*: the mean barely
tracks the truth, the correlation looks terrible, the RMSE looks large — and yet the
inference is *flawless*. A wide-but-honest posterior that says "I don't know much
about this parameter" is a **correct result**, not a failure. Retraining to "fix" it
is impossible: the data simply do not carry the information.

This is exactly why point-recovery alone **cannot** tell a poorly-identified
parameter apart from a biased inference engine — **both show low correlation.** To
separate them you must look at the error *relative to the posterior's own width*.

## What to report — per parameter

Report these two, per parameter, not the point metrics alone:

- **Posterior contraction** = 1 − Var_post / Var_prior.
  ≈ 0 means the data added almost nothing (posterior ≈ prior — the parameter is
  poorly identified); ≈ 1 means the parameter is pinned down by the data.
- **Posterior z-score** = (posterior mean − true value) / posterior sd.
  A *standardised* error: ≈ N(0, 1) if the posterior is well-behaved. Small |z| means
  the posterior is consistent with the truth **given its own width**; large |z| means
  it is biased or too narrow.

Then plot **z-score against contraction** — the "sensitivity" plot. Its four
quadrants are named and tell you *which regime each parameter is in*:

| | low contraction | high contraction |
|---|---|---|
| **small \|z\|** | parameter **poorly identified** (benign) | **ideal** |
| **large \|z\|** | **prior–likelihood conflict** | **overfitting to noise** |

Report the point metrics (correlation, NRMSE) too — they are not useless — but read
them *through* the contraction/z-score lens, never as the verdict on their own.

## Mechanics (BayesFlow)

Use BayesFlow's own diagnostics — **do not hand-roll these metrics.** Sample
posteriors over many fresh simulated test datasets into the `(estimates, targets)`
format (`estimates[name]`: shape `(datasets, samples, 1)`; `targets[name]`: shape
`(datasets, 1)`), then:

    import bayesflow.diagnostics as bfd
    bfd.metrics.root_mean_squared_error(estimates, targets)   # point recovery (width-blind)
    bfd.metrics.correlation(estimates, targets)               # point recovery (width-blind)
    bfd.metrics.posterior_contraction(estimates, targets)     # informativeness
    bfd.metrics.posterior_z_score(estimates, targets)         # standardised error
    bfd.recovery(estimates, targets)                          # recovery plot
    bfd.z_score_contraction(estimates, targets)               # the sensitivity plot

The worked, tested reference is `examples/toy-normal/run_validation.py` (its
`collect()` builds exactly this format, bucketing test datasets by set size so JAX
compiles at most once per size). See `bayesflow-implementation` for the shapes and
engineering mechanics.

**Where the problem is low-dimensional, also cross-check against an exact reference
posterior** — a grid or analytic posterior built from validated libraries
(`scipy.stats`), not hand-coded densities (easy to drop a normalising constant). This
is the strongest possible recovery check and the payoff of having a toy: the toy's
`diagnostics.py` computes the exact grid posterior, and `run_validation.py` confirms
the BayesFlow posterior means agree with it. Real models rarely afford this, which is
why the contraction/z-score reading above carries the weight there.

## Read recovery and SBC together

Recovery is **necessary but not sufficient** — it cannot certify calibration on its
own, and a posterior can track the truth on average while being mis-calibrated. Always
continue to **SBC (stage 7)** and read the two as one picture. In particular, *small
|z| with low contraction* means "poorly identified" only if **SBC is also uniform**;
if SBC is non-uniform, the same low contraction is hiding a real bias. The combined
decision table lives in the `calibration-sbc` skill — use it to reach the verdict.

## On fail — decide *which* failure it is before acting

The routing depends entirely on the diagnosis; do not reflexively retrain:

- **Poor point-recovery *but* small |z| *and* (stage 7) uniform SBC** → the inference
  is correct and the parameter is genuinely **poorly identified** by the data. This is
  a *scientific / identifiability finding*, **not** an engine bug: surface it to the
  human, and **do not retrain to "improve" it** — you cannot, the data are
  uninformative. (Forward-linked from `prior-elicitation`: category-(c) parameters
  often land here, as flagged at stage 2.)
- **Large |z| and/or non-uniform SBC** → a real inference problem (bias or wrong
  width) → back to **stage 4/5**: constrain a parameter (`adapter.constrain(...)`),
  train longer, use a larger flow, or improve the summary network. This is
  *engineering*, yours to fix.
- **Low contraction *with* large |z|** → **prior–likelihood conflict**: the prior and
  the data disagree. That is a *scientific* signal → diagnose and surface to the human
  (stage 2). **Do not silently move the prior** to resolve it — which prior, and
  whether, is the human's call.

The width-aware reading is the throughline of the whole workflow's owner split: a
poorly-identified parameter is *science* (a finding, the human's to interpret); a
biased engine is *engineering* (yours to fix). Getting the two confused — retraining a
finding, or reporting a bug as a finding — is the specific error this stage exists to
prevent.

## Using this skill

- **Entering from stage 5:** you have a trained pilot. Collect posteriors over held-out
  simulated data and compute the metrics above.
- **Always continue to `calibration-sbc` (stage 7)** and read them together before any
  verdict or loop-back.
- **Engineering mechanics** (metric shapes, `constrain`, summary networks) are in
  `bayesflow-implementation`; the worked reference is `examples/toy-normal/`.

## References

- Gelman et al. (2020), *Bayesian Workflow* — stresses that fake-data recovery means
  recovering parameters *to within their uncertainty*, not to a point. arXiv:2011.01808
  — https://arxiv.org/abs/2011.01808
- Schad, Betancourt & Vasishth (2021), *Toward a principled Bayesian workflow in
  cognitive science* — the source of the **posterior z-score vs. posterior
  contraction** sensitivity plot and its quadrant interpretation used here.
  arXiv:1904.12765 — https://arxiv.org/abs/1904.12765 (sensitivity-plot quadrants
  figure: https://www.researchgate.net/figure/Posterior-z-scores-as-a-function-of-posterior-contraction-Arrows-show-four-possible_fig1_342288675)
