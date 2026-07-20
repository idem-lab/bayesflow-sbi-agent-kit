---
name: posterior-predictive-check
description: Runs the posterior predictive check — does the fitted model, conditioned on the real data, reproduce the features of that data? Draw parameters from the real-data posterior, simulate replicated datasets, and compare them to the observations on the scales the human reasons about, using targeted discrepancy measures (especially features the model was NOT directly fit on). A systematic mismatch is a scientific signal that the model is missing something. Use in Act III, AFTER the reliability/OOD check certifies the posterior is trustworthy. It is a GATE; on fail, route to stage 2 or the model — the human decides the fix.
---

# Posterior predictive check

This is the model-adequacy check of Act III: having fitted the model to the real data,
does the fitted model **reproduce the real data**? Draw parameters from the real-data
posterior, simulate new ("replicated") datasets from them, and compare those
replicates to the actual observations. A systematic, repeatable mismatch means the
**model is missing something real**.

## Precondition: the posterior must already be trustworthy

Run this **only after** the reliability / out-of-distribution check has passed (see
`real-data-reliability`). The posterior predictive check conditions on the real-data
posterior — if that posterior is untrustworthy because the real data is
out-of-distribution, the whole check is uninterpretable: you would be asking whether a
model reproduces the data using parameter draws the network had no business producing.
Reliability first, always, whatever the stage numbering around these two.

## Prior predictive vs. posterior predictive — don't confuse them

Both push parameters through the simulator and compare to data, but they ask opposite
questions at opposite ends of the workflow:

| | draws from | when | tests |
|---|---|---|---|
| **Prior** predictive (stage 3) | the **prior** | before training | are the *beliefs* plausible, and do they cover the data? |
| **Posterior** predictive (here) | the **real-data posterior** | after fitting | does the *fitted model* reproduce the real data? |

Stage 3 tests your assumptions before seeing data; this tests the model *after*
conditioning on data. Reuse the same presentation machinery (the human's conventional
data view, interpretable derived quantities, recognition over recall — see
`prior-predictive-check`), but the object under test is now the model's adequacy, not
the prior's plausibility.

## What to compare — targeted discrepancies, not one number

1. **Overlay replicates on the observations.** Simulate many replicated datasets from
   posterior draws and plot them against the real data in the human's own view. Broad
   graphical agreement is the first, cheapest read.
2. **Choose test quantities that stress what matters — especially features the model
   was *not* directly fit on.** A model fit to the mean will reproduce the mean almost
   by construction; the informative checks are the *other* features: dispersion, tails,
   extremes, number of zeros, autocorrelation, spatial structure, the peak, the timing.
   Pick discrepancy measures `T(y)` that encode the scientifically important features
   and any known failure modes, and compare `T(observed)` against the distribution of
   `T(replicated)`.
3. **Report where the data sits in the replicate distribution** (a posterior
   predictive p-value is one summary of this) — but **prefer graphical and per-feature
   checks to a single p-value.** Posterior predictive p-values are known to be
   conservative and are *not* calibrated frequentist p-values; a value near 0 or 1
   flags a feature the model cannot reproduce, but the number itself should not be
   over-read (Gelman, Meng & Stern 1996; Gelman 2020).
4. **Present on the interpretable scales** the human reasons about (data and derived
   quantities), and ask them to judge adequacy — the same you-show-they-decide split as
   the prior predictive check.

## The SBI-specific payoff: a failure here points at the model

Because inference is amortised, a poor posterior predictive fit could in principle be
either a bad *model* or a bad *inference network*. But by this point you have ruled the
network out: recovery and SBC (stages 6–7) certified the engine on simulated data, and
the reliability check certified the real data is in-distribution. With the engine
exonerated and the data in-distribution, a systematic posterior predictive mismatch is
**model misspecification** — the model is missing a real feature of the process. That
chain is what makes this check conclusive rather than ambiguous, and it is why the Act
II simulated-data checks and the Act III reliability check must come first.

## GATE + on fail

A failing posterior predictive check is a **scientific** signal, and this stage is a
hard approval gate. **Diagnose, do not fix the science:**

- Identify **which features** of the data the model fails to reproduce (which `T(y)`
  are off, and in which direction), and **which components** — a prior (stage 2), the
  likelihood/observation model, or a missing mechanism — are the plausible cause.
- Present that to the human as evidence and options. The remedy — revising priors or
  the model itself — is **theirs to reason about and decide.** Do **not** silently
  retune priors or edit the model/simulator to make the check pass; that is fitting the
  model to your own expectations and it belongs to the human.
- On their decision, route back to **stage 2** (priors) or to the model, then re-run
  the affected downstream stages. Log the loop-back in the ledger.

On a clean pass, the model reproduces the real data on the features that matter →
proceed to **human review** with the fit evidence in hand.

## Note on repo status

There is not yet a worked, tested posterior-predictive report in the kit
(`validation/benchmark-matrix.md` lists it as the toy example's one remaining
unchecked item). Until there is, generate replicates directly from the model's
`likelihood`/simulator at posterior draws (see `examples/toy-normal/simulator.py`), and
keep discrepancy measures simple and validated (`scipy.stats`, not hand-coded).

## Using this skill

- **Entering:** only after `real-data-reliability` has passed. Reuse the human's
  conventional data view and derived quantities captured at intake / stage 3.
- **On fail:** route to `prior-elicitation` (stage 2) or flag the model to the human;
  do not edit priors or mechanism yourself.
- **Engineering mechanics** are in `bayesflow-implementation`; the model/simulator
  reference is `examples/toy-normal/`.

## References

- Gelman et al. (2020), *Bayesian Workflow* — posterior predictive checking and its
  place in the iterative workflow; cautions on posterior predictive p-values.
  arXiv:2011.01808 — https://arxiv.org/abs/2011.01808
- Gelman, Meng & Stern (1996), *Posterior Predictive Assessment of Model Fitness via
  Realized Discrepancies*, Statistica Sinica 6:733–807 — the foundational treatment of
  discrepancy measures `T(y)` and posterior predictive p-values (and why they are
  conservative). https://www.jstor.org/stable/24306036
- Gabry, Simpson, Vehtari, Betancourt & Gelman (2019), *Visualization in Bayesian
  Workflow* — graphical posterior predictive checks on interpretable quantities.
  https://doi.org/10.1111/rssa.12378 — arXiv:1709.01449
- Schad, Betancourt & Vasishth (2021), *Toward a principled Bayesian workflow in
  cognitive science* — posterior prediction as the fourth of the four questions.
  arXiv:1904.12765 — https://arxiv.org/abs/1904.12765
