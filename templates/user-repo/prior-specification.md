# Prior specification

_The record of the priors for this model and the reasoning behind each one. The
assistant fills this in with you during stage 2 (prior review); **you approve every
row**. Nothing here is chosen or changed without your say-so. See the
`prior-elicitation` skill for how this is done._

_The **target summary statistics** you commit to below are the acceptance criteria
for the stage-3 prior predictive check — the numbers we agree simulated data should
match, written down **before** we look at the pushforward, so the check stays
honest._

---

## Parameter triage

Sort each of your parameters by how much you can reason about it directly. The
easiest way to tell which category a parameter is in is to ask **what kind of
statement you could make about it** (see the `prior-elicitation` skill, step 0):

- **(a) Direct & informative** — you can point to a measurement or study.
  _Examples:_ eggs laid per mosquito batch; average leaf weight; a pathogen's
  incubation period. _You can supply:_ an estimate + range/interval from a study, or
  several study estimates to combine.
- **(b) Rough idea, real uncertainty** — a real quantity, but hard to measure directly.
  _Examples:_ a mosquito's average daily dispersal distance; new leaves per tree per
  year; an intervention's effect size. _You can supply:_ a plausible range you'd be
  surprised to fall outside, a probability it's positive or exceeds some value, or a
  most-likely value with a soft spread.
- **(c) Can't reason directly** — no standalone value; unitless, a nuisance, or only
  meaningful jointly. _Examples:_ a negative-binomial overdispersion parameter; a
  Gaussian-process lengthscale; a transmission/contact rate. _The give-away:_ you
  can't make an (a) or (b) statement — so we reason on the scale of the data instead.

Categories can change as understanding improves — and a parameter can move buckets
if we **reparameterise** it (e.g. centring predictors turns an un-interpretable
intercept into "the mean outcome" you can reason about). Note any moves in the
iteration log.

---

## Priors

| Parameter | Meaning & units | Support | Category (a/b/c) | Elicited plausible range | Chosen distribution + hyperparameters | Evidence / reference (esp. category a) | Approved by · date |
|-----------|-----------------|---------|------------------|--------------------------|----------------------------------------|-----------------------------------------|--------------------|
|           |                 |         |                  |                          |                                        |                                         |                    |

---

## Target summary statistics (stage-3 acceptance criteria)

Concrete, quantitative statements about what simulated data from this model *should*
look like — with ranges, not just point values. These are committed to before the
pushforward and carried into the prior predictive check.

| Summary statistic of the data | Central value | Plausible range (surprised outside) | Notes |
|-------------------------------|---------------|-------------------------------------|-------|
|                               |               |                                     |       |

---

## Method notes

- **How priors were set:** _manual elicitation / algorithmic (elicito) / mixed._
- **Fit checked:** _for each prior fitted from stated numbers — units and the
  meaning of the summary (95% interval? min/max? ±1 SD?) confirmed with you, and the
  fitted distribution verified by simulation to reproduce those numbers._
- **Coverage check:** _does the prior-predictive pushforward cover the plausible
  range of the real data? (guards against out-of-distribution failure at stage 9.)_
- **Parameters flagged as likely poorly identified:** _(forward-links stage 6.)_
