---
name: prior-elicitation
description: Helps a user specify priors for an SBI model — triage each parameter by how much they know, elicit on the interpretable/observable scale, translate stated beliefs into distributions, and commit to quantitative summary-statistic targets that the prior predictive check (stage 3) will test. Use at stage 2 of the workflow, before any training. Priors are the human's scientific decision; you facilitate and translate, you do not decide.
---

# Prior specification / elicitation

This is stage 2 of the `workflow-orchestration` map: turn the human's domain
knowledge into priors, agree them, and hand stage 3 (prior predictive check) a set
of **quantitative acceptance criteria** to test them against. It ends at the
stage-2 **GATE** — the human approves the priors before anything is trained.

**The prior is a scientific choice; you facilitate, you do not decide.** Your job is
to ask good questions, translate the answers into distributions, show the human what
those distributions imply about *data*, and record everything. You never pick a
prior and quietly move on, and you never widen or shift a prior to make a later
check pass without the human's say-so.

## Two ideas that shape everything here

1. **Elicit on the interpretable / observable scale, not the raw parameter scale.**
   Domain modellers have intuitions about *data and observable quantities* ("the
   peak is in the hundreds", "it roughly doubles each week"), not about a rate
   constant living in ℝ. So you elicit beliefs about things the human can actually
   reason about, and translate those into priors over parameters — checking the
   translation by *simulation* (the pushforward). This is the central
   recommendation of the elicitation literature (Mikkola et al. 2024: elicit on
   both the parameter and the observable space).

2. **In amortised SBI, the prior *is* the training distribution.** The network only
   learns to do inference over the region the prior samples. A too-narrow prior
   guarantees the real data will be **out-of-distribution** at stage 8 (a
   confidently wrong answer, with no error message); an absurdly wide/flat one
   wastes network capacity and can hurt calibration. So "does the prior's
   pushforward *cover* the plausible real data?" is a first-class design question
   here, not an afterthought. Flat / improper priors are not an option — the
   training range must be effectively bounded.

## Step 0 — triage the parameters (do this first)

Before eliciting anything, ask the human to sort their parameters by **how much they
can reason about each one directly**. Present the three categories below with the
example parameters and, crucially, the **kind of statement they would supply** for
each — because *the form of information a user can bring is itself the clearest
signal of which bucket a parameter is in*. The human classifies their own
parameters; you present the framework and answer questions, but you **do not assign
the categories for them** — which bucket a parameter falls in depends on what *they*
know, and is their judgement to make.

**(a) Direct & informative** — you can point to a measurement or a study.
- *Example parameters:* the number of eggs a mosquito lays per batch; the average
  weight of a leaf on a tree; a pathogen's mean incubation period; a tabulated
  radioactive decay rate.
- *What the human supplies:* an **estimate plus a range or confidence interval from
  a study** (e.g. "mean ≈ 120 eggs, 95% interval 90–150"); or **several estimates
  from different studies** to combine (average them, and widen for between-study
  heterogeneity); or a value with a known precision from a reference table.
- *What you do:* fit a distribution to those quantiles **transparently** (state and
  confirm units and the summary's meaning, show the code, verify by simulation — see
  below) and **record the citation**. These are the tightest, most defensible priors.

**(b) Rough idea, real uncertainty** — a real quantity, but hard to measure directly.
- *Example parameters:* the average distance a mosquito disperses per day; the number
  of new leaves a tree grows per year; the effect size of a novel intervention; a
  case-reporting probability in a new setting.
- *What the human supplies:* a **plausible range** they'd be surprised to fall
  outside ("1–8 new leaves a year, most likely around 3"); a **probability that it
  exceeds a threshold or is positive** ("I'm ~80% sure daily dispersal is under
  500 m"); or a **most-likely value with a soft spread**.
- *What you do:* turn those into a weakly-informative distribution via the manual
  loop below (quantiles → distribution → pushforward), honestly wide — and translate
  the stated numbers transparently, the same way as for (a) (units, meaning, code,
  simulation check).

**(c) Can't reason directly** — no standalone value or even a confident range;
unitless, a nuisance parameter, or non-identified in isolation.
- *Example parameters:* the overdispersion parameter of a negative-binomial sampling
  distribution; the lengthscale of a Gaussian process; an epidemic transmission /
  contact rate (a composite of contact frequency and per-contact probability); two
  parameters that only ever appear as a product or ratio.
- *What the human supplies:* usually **nothing about the parameter itself** — the
  give-away for this bucket is precisely that they *cannot* fill in the (a) or (b)
  statements above. Instead they reason on the **observable-data scale, jointly**
  with the other parameters (the pushforward), or use the algorithmic route.
- *What you do:* elicit on the observable scale, never per-parameter. If there are
  several such parameters, suggest the algorithmic route (see "When to reach for the
  algorithm").

Why the triage matters — it is not just tidiness:

- **Category (c) is exactly where per-parameter elicitation is impossible.** You
  cannot state a marginal belief about the parameter, *and* you cannot check it in
  isolation because its effect on the data is entangled with the other parameters.
  That is precisely the problem observable-scale / simulation-based elicitation was
  built for, so "several category-(c) parameters → joint algorithmic elicitation"
  is the principled move, not a shortcut.
- **The categories forward-link to identifiability (stage 6).** Category-(c)
  parameters are usually the poorly-identified ones — if recovery there is weak with
  a small posterior z-score and uniform SBC, that is a *finding*, not a bug (see
  stage 6). A category-(a) strong prior that later fights the data is the stage-6
  *prior–likelihood conflict* quadrant. Flag both downstream.

The buckets are **not permanent.** A parameter the human thought was (b) can become
(a) once they recall a reference, or reveal itself as effectively (c) when the
pushforward shows their stated belief is internally inconsistent. Re-triage as
understanding improves, and note the move in the ledger.

**A parameter's bucket also depends on its *parameterisation*, not just on
knowledge — and you can change the parameterisation.** The commonest case is a raw
intercept: "the outcome when every predictor is zero" is often an un-interpretable
extrapolation to a nonsensical point (category (c)), but **centre the predictors**
and it becomes "the mean outcome", which the human can usually reason about
(category (a) or (b)). Offering such interpretability-improving transforms —
centring, standardising, reparameterising to a quantity with real-world meaning — is
an *engineering* step you can take (see `bayesflow-implementation` on keeping
parameters scale-free); it is **not** a scientific classification, and the human
still owns which bucket the result lands in and what its prior is. Re-triage after
any reparameterisation.

## The manual elicitation loop (per parameter, or jointly where coupled)

1. **Frame it.** State the parameter in the human's terms: what it means, its units,
   and its support (positive? bounded? a probability? ordered relative to another?).
2. **Elicit plausible values.** Ask for a central guess and a range they would be
   *surprised* to fall outside — a low and a high plausible value. (Quantile /
   "roulette" style elicitation; Garthwaite, Kadane & O'Hagan 2005; the SHELF
   tradition.) For category (a), also capture the **reference**.
3. **Elicit target summary statistics of the *simulated data*. (Required — do not
   skip.)** Push the human to commit to **concrete numbers with ranges** about what
   simulated data from this model should look like — e.g. "the peak weekly count is
   around 200, and I'd be surprised below 50 or above 800", "about 10–30% of runs
   go extinct". Vague statements ("it should look plausible") are not enough. These
   numbers are the **acceptance criteria** you carry into stage 3, committed to
   *before* seeing the pushforward — which is what keeps the check honest.
4. **Choose a distribution that respects the support** (see the recipes below) and
   set its hyperparameters to match the elicited quantiles from step 2. This
   translation is itself a small analysis — do it transparently (state and confirm
   units and the meaning of each number, show the fitting code, and verify by
   simulation). See *"Translating stated numbers into a distribution is itself an
   analysis"* below.
5. **Pushforward check.** Draw from the candidate prior, run the simulator, and
   compare the implied summaries against the numbers from step 3. Pass → record and
   move on. Fail → revise the prior with the human, or offer the algorithmic route.
6. **Record.** Write the prior, its rationale, any evidence, the category, and the
   target summaries into `prior-specification.md`, and log the approval in the status
   ledger's decisions log. The target summaries also flow to stage 3.

Even a well-evidenced category-(a) prior gets the pushforward/coverage check: a
sound *marginal* prior can still combine with the others to imply implausible data,
or leave the plausible real-data range uncovered (the SBI OOD concern above).

## Translating stated numbers into a distribution is itself an analysis

Fitting a prior to a user's stated estimate, range, or probability (step 4 above) is
a small statistical analysis in its own right — and one easily tripped up by a
misread unit or an ambiguous summary. **Do it transparently, never silently.** For
each such parameter:

1. **State the assumptions and have the user confirm them** — this is where the
   errors hide:
   - **Units and scale.** Confirm the stated numbers are in the *same units* as the
     parameter in the simulator — a mm-vs-m or per-day-vs-per-week slip silently
     ruins the prior — and whether they are on the natural or a transformed (e.g.
     log) scale.
   - **Meaning of the summary.** Pin down what a stated interval *is*: a 95%
     interval, a "surprised to fall outside" range, a hard min/max, or ±1 SD — each
     implies different hyperparameters. Likewise what a "probability positive" or
     "probability above X" refers to.
   - **Distributional family.** Name the family you propose and *why* (support, skew,
     tail weight), and be explicit that it is an assumption, not a fact.
   - **Parameter vs observable.** Confirm the numbers describe the *parameter*
     itself. If they actually describe an observable quantity, this is category-(c)
     pushforward elicitation, not direct fitting.
2. **Propose the method and write readable code.** Show the working — e.g. solving
   for the hyperparameters that match the stated quantiles (moment-matching, or
   numerically matching two quantiles) — as inspectable code, not a black box.
3. **Verify by simulation.** Draw a large sample from the fitted distribution and
   check its summaries reproduce what the user stated: the implied mean / quantiles /
   probability-positive should match the elicited numbers within tolerance. Provide
   this simulation code. If they do not match, the fit or an assumption is wrong —
   fix it before moving on. (This is a self-consistency check *against the stated
   numbers*; it is distinct from — and precedes — the pushforward check, which tests
   the prior on the *data* scale.)
4. **The user checks the assumptions and the working.** This is a checkpoint, not a
   courtesy: they confirm the units, the summary's meaning, and the chosen family
   before the prior is accepted.

## Present the choice up front

At the start of stage 2, give the human a short menu:

> "You can (a) specify priors directly and we'll validate them together against
> summary statistics you commit to, or (b) I can walk you through an algorithm that
> *derives* priors from your stated beliefs about the data. Most people start with
> (a); we can switch to (b) anytime, and I'll suggest it if (a) doesn't validate or
> if several parameters turn out to be ones you can't reason about directly."

## Tricky-case recipes

| Situation | Recipe |
|---|---|
| **No intuition for the raw parameter** (rate constants, coupling terms) | Elicit on the observable scale; translate by pushforward. Category (c); consider the algorithm. |
| **Positive / lower-bounded** (rates, scales, SDs) | HalfNormal, lognormal, or Gamma; keep prior mass off the boundary. Pair with `adapter.constrain(name, lower=0)` — see `bayesflow-implementation`. |
| **Probability / proportion in [0,1]** | Beta (elicit on the 0–1 scale), or logit-normal. |
| **Must sum to 1** (compositions) | Dirichlet; elicit jointly, never independently. |
| **Ordered** (θ₁ < θ₂ < …) | Ordered / positive-increment priors; elicit the ordering as a constraint. |
| **Scale / unit sensitivity** | Standardise inputs and keep priors scale-free, so the same prior is not accidentally tight or loose because of a unit choice (Stan Prior Choice Recommendations). |
| **Variance / hierarchical SD** | half-Normal or half-Student-t on the SD — *not* inverse-gamma; note weak identifiability when there are few groups (Gelman 2006). |
| **Turning a published estimate into a prior** | Convert a reported mean±SE or CI into matching moments/quantiles; widen for between-study heterogeneity. Category (a) — record the citation. |
| **Genuinely weak knowledge** | A weakly-informative *bounded* prior — never flat. Flag that this parameter will likely be poorly identified (forward-links the stage-6 sensitivity finding). |
| **Robustness wanted** | Heavier tails (half-Cauchy / half-Student-t for scales) so a single surprising value is not ruled out a priori. |

## When to reach for the algorithm (secondary route)

The BayesFlow ecosystem has tools that *derive* priors by optimisation:
`elicito` and simulation-based prior elicitation (Bockting, Radev & Bürkner). They
simulate summaries from the generative model, compare them to the human's
**elicited target summaries**, and adjust the prior hyperparameters to minimise the
discrepancy. They fit the prior to *stated beliefs*, **never to the real data** — so
there is no data-contamination concern; it is a mechanised version of the manual
pushforward loop above.

Keep it **secondary**: a user who can state sensible priors should not be pushed
into heavier machinery. Suggest it when:

- the human's hand-specified priors repeatedly fail the pushforward check, or
- there are several category-(c) parameters (which cannot be checked in isolation),
  so a *joint* optimisation against observable-scale beliefs is the natural fit, or
- the human simply prefers to start there.

Either way it is opt-in and human-approved, and the pushforward/coverage check
still applies to whatever prior it produces.

## The GATE

Stage 2 ends only when the human has **approved the priors**. Present: each prior
with its rationale and evidence, the committed target summary statistics, and any
parameters you have flagged as likely poorly identified. Then stop and wait. Record
the approval in the ledger; the target summaries go to stage 3.

## References

- Mikkola, Martin, Chandramouli, Hartmann, Abril, Winter, Bürkner, et al. (2024),
  *Prior Knowledge Elicitation: The Past, Present, and Future*, Bayesian Analysis
  19(4):1129–1161 — the comprehensive review; argues elicitation should cover both
  parameter and observable space and integrate into the workflow. arXiv:2112.01380 —
  https://arxiv.org/abs/2112.01380
- Bockting & Bürkner, *elicito: A Python Package for Expert Prior Elicitation*, and
  Bockting, Radev & Bürkner (2024), *Simulation-based prior knowledge elicitation
  for parametric Bayesian models*, Scientific Reports — the algorithmic route above;
  fit priors to elicited beliefs about observable quantities. arXiv:2506.16830 —
  https://arxiv.org/abs/2506.16830 ; https://elicito.readthedocs.io/
- Gelman, Simpson & Betancourt (2017), *The Prior Can Often Only Be Understood in
  the Context of the Likelihood* — why priors are never judged in isolation, which
  is why stage 2 → stage 3 is a tight loop. https://doi.org/10.3390/e19100555
- Stan Development Team, *Prior Choice Recommendations* (wiki) — weakly-informative
  defaults, keeping parameters scale-free, and variance-parameter guidance.
  https://github.com/stan-dev/stan/wiki/prior-choice-recommendations
- Gelman (2006), *Prior distributions for variance parameters in hierarchical
  models*, Bayesian Analysis 1(3):515–534 — half-t (not inverse-gamma) on SDs.
  https://doi.org/10.1214/06-BA117A
- Garthwaite, Kadane & O'Hagan (2005), *Statistical Methods for Eliciting
  Probability Distributions*, JASA 100(470):680–701 — the classic on quantile-based
  expert elicitation (the "roulette"/quantile methods in the manual loop).
  https://doi.org/10.1198/016214505000000105
