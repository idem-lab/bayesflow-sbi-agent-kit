---
name: prior-predictive-check
description: Runs the stage-3 prior predictive check — draw parameters from the joint prior, push them through the simulator, and confirm the model produces plausible data AND covers the plausible real data (the SBI out-of-distribution guard) before any training. Tests the quantitative targets committed at stage 2 (confirmatory) and surfaces implausibility no one pre-registered (exploratory), presented in the user's own terms across data, interpretable latent/derived quantities, and parameters. Use at stage 3. On fail, diagnose and route back to stage 2 — do not fix the science.
---

# Prior predictive check

This is stage 3 of the `workflow-orchestration` map, and the last stage of Act I
(set up the model & your beliefs). It is the **cheapest check in the whole workflow
and one of the highest-value**: no network, no training, just the priors and the
simulator. It catches prior and model mistakes that are otherwise invisible until
you have spent real compute on training — or, worse, until you are staring at a
confidently wrong posterior on real data.

It runs on the output of stage 2 (`prior-elicitation`): the agreed joint prior, the
simulator, and — where they exist — the **quantitative target summary statistics the
human committed to** in `prior-specification.md`.

## What this check is for — two questions, not one

A prior predictive check in a *standard* Bayesian workflow asks one thing:
**are the priors + model producing plausible data?** In amortised SBI it must ask a
second, equally important thing, because **the prior *is* the training distribution**
(see `prior-elicitation`): the network only ever learns to do inference over the
region the prior samples.

1. **Plausibility.** Does the model produce output that looks like what the human
   expects — right order of magnitude, right support, no absurdities?
   *Too-wide / wrong-scale / mis-specified priors show up here.*
2. **Coverage.** Does the prior predictive distribution *span* the plausible real
   data — does the real data sit comfortably inside the cloud of simulated data?
   *Too-narrow priors show up here, and this is the same failure that becomes an
   out-of-distribution (OOD) disaster at stage 8 — a confidently wrong posterior
   with no error message. Catching it now is far cheaper than catching it there.*

These pull in opposite directions (widen for coverage, tighten for plausibility), so
report them **separately**. A prior can pass one and fail the other.

There is also a third, purely mechanical read-out worth surfacing:

3. **Simulator health.** Did the simulator run cleanly across the whole prior — no
   `NaN`/`inf`, no crashes, no non-termination, no degenerate all-identical output?
   Unlike (1) and (2), a failure here is usually an **engineering bug you may fix**
   (with the human's OK), not a scientific decision (see *On fail* below).

## Who judges: you show, the human decides

The check has two halves, and conflating them is how it goes wrong:

- **Confirmatory** — test the targets the human committed to at stage 2. Bounded and
  honest, but only as good as the targets, which may be sparse, missing, or
  insufficient to catch every absurdity.
- **Exploratory** — surface implausibility *no one pre-registered*. Necessary,
  because two committed summaries can both pass while the data is absurd in a
  dimension nobody named. But this is where the risk lives: either **you guess** at
  what is plausible, or you **bury the human** in plots.

The rule that resolves both risks: **you never decide plausibility — you decide
*presentation*; the human renders the verdict.** Any design where the agent judges
"this looks implausible" *is* the guessing to avoid. Say so plainly to the human —
you are not the arbiter of plausibility — so they engage rather than rubber-stamp.

This works even for a human who **could not state their expectations in advance**
(the common case for vague or purely study-based priors), because of one asymmetry:

> **Recognition is easier than recall.** A modeller who cannot pre-state "peak weekly
> counts should be 50–800" will still look at ten simulated curves and say instantly
> "that one peaks at 40 million people — impossible." Pose the check as recognition
> (*"do any of these look wrong to you?"*), not recall (*"what did you expect?"*).

What you do **without judgment**: the objective screens (support / finiteness /
magnitude — that is *validity*, not plausibility); the confirmatory test of whatever
targets exist; and the *presentation*. What you hand over: the verdict.

## What the human can reason about — three interpretable scales

Elicitation stresses the **observable** scale. But a user can usually reason about a
broader set of quantities than the raw data, and a narrower set than "all
parameters." Find out which of these three scales *this* user has intuitions on, and
show the check on those scales — not on whatever is mathematically convenient:

1. **Observable data** — the raw thing they measure, which may be very noisy.
2. **Interpretable latent / derived quantities** — functions of the parameters (and
   latent states) that carry scientific meaning even though they are *neither the
   data nor the priors*: the **noise-free predicted case counts** behind noisy
   reports, R₀, a doubling time, peak prevalence, an equilibrium density, a latent
   growth rate. **In a model with very noisy observations these are often what the
   user has the *strongest* intuition about — stronger than the raw data — and the
   noise-free signal exposes a bad prior that the observation noise would otherwise
   mask.** Ask the human explicitly which non-data, non-prior model quantities they
   would want to check; compute and show those alongside the data. They may even have
   committable targets for them — promote those to acceptance criteria.
3. **Parameters** — sometimes directly interpretable (a category-(a) rate), often
   not. Marginal parameter draws are mostly a *sanity check* that the priors sampled
   as intended, not a plausibility view (the human reasons about consequences, not
   rate constants).

## What to show — the presentation ladder

This is the direct answer to "marginals, joint, or summaries?": it is layered,
**data-and-derived-space first**, cheapest-burden first. Most failures die in the
first rungs, so you rarely need the last ones.

1. **Automated screens** (you alone, no human time) — support / finiteness /
   magnitude. Kills gross bugs before the human sees anything.
2. **~10 example datasets, rendered exactly as the human views their real data**
   (time series, count histogram, map — whatever their data *is*; ask their
   conventional view at intake). The single highest-value, lowest-burden view, and
   recognition-based. This is the primary object the human reacts to.
3. **The nominated interpretable latent / derived quantities** (scale 2 above),
   shown the same recognition way — often *more* diagnostic than the noisy data.
4. **A curated few data-space summary distributions** — the committed targets plus
   generic, data-type-appropriate ones (peak, range, extinction fraction…), with the
   committed ranges marked. *Few*; offer the fuller battery on request, do not dump.
5. **Parameter marginals** — the cheap sanity check that priors sampled as intended.
6. **Parameter joint plots** — reserved for *diagnosis*: when something looks wrong,
   "the absurd datasets all come from the high-σ / low-μ corner" is how you localise
   it, and it is where an *unintended induced correlation* between parameters shows
   up. Not a front-line user-facing view.

For an exploratory check with thin targets it is **datasets and derived quantities
first, numbers second** — never let a green tick on two summaries stand in for the
human actually looking at simulated output.

## How stage 3 differs from the stage-2 pushforward (not redundant)

The `prior-elicitation` loop already does a *pushforward* per parameter (its step 5).
That is a **design-time** tool: run repeatedly, on partial or per-parameter priors,
*while choosing* each prior. Stage 3 is different in three ways, and running it is not
busywork:

- It is run **once, on the final, whole joint prior and the full simulator**, after
  every parameter is agreed.
- It tests the **committed** targets as **pre-registered** criteria — the numbers
  were written down *before* this pushforward, which keeps the check honest (no moving
  the goalposts to match what you happen to see).
- It is **joint**. Sound marginal priors can still combine to imply implausible data:
  `mu` and `sigma` can each be reasonable while their combination puts mass on data no
  one believes. Stage 3 is where that surfaces.

## Running the check

1. **Draw from the *joint* prior.** Sample `M` parameter sets from the full prior —
   all parameters together, never marginally. Use enough draws to estimate the tail
   quantiles you committed to: for a "surprised below the 5th percentile" bound,
   hundreds are too few — use several thousand.
2. **Push through the simulator, retaining the interpretable intermediates.** Run
   each parameter set through the *actual* simulator (including its meta/context
   draws, e.g. a variable sample size `N`), and **keep the latent / derived
   quantities the human nominated** (scale 2 above), not only the final noisy data.
   In BayesFlow the simulator samples the prior predictive directly — no network
   needed:

       sims = simulator.sample(4000)        # dict of arrays: parameters + data

   The pure-NumPy `prior`/`meta`/`likelihood` functions in
   `examples/toy-normal/simulator.py` are the worked reference.
3. **Run the automated screens yourself** (rung 1): support, finiteness, magnitude.
   Resolve or flag anything here before spending the human's attention.
4. **Present via the ladder, in the human's own terms** (rungs 2–4), datasets and
   derived quantities first. Pose recognition questions, not recall questions.
5. **Confirmatory test of any committed targets.** Reuse the *exact* summary
   definitions behind the stage-2 targets — recomputing a subtly different statistic
   invalidates the pre-registration — and check each simulated distribution's central
   value and spread against the committed central value and range.
6. **Assess coverage explicitly.** Does the simulated cloud *contain* the plausible
   real data? If the real observations (or the human's description of them) sit at or
   beyond its edge, that is a coverage failure now and an OOD failure at stage 8.
   Record the answer in the coverage-check row of `prior-specification.md`.
7. **Record and route.** Update `sbi-workflow-status.md` (stage transition, any
   loop-back) and the coverage note in `prior-specification.md`. On pass, proceed to
   stage 4; on fail, follow *On fail*.

### When the targets are thin or missing

The common real case — priors given directly, with few or no observable-scale
targets. Two honest paths; do **not** dress post-hoc judgement up as pre-registration:

- **Recover pre-registration cheaply.** Show example datasets and derived quantities
  (rungs 2–3) — which convey the *character* of the output without committing anyone
  to a number — then ask the human to commit to one or two targets, *then* reveal the
  summary pushforward (rung 4). A few minutes buys back the honesty of
  pre-registration.
- **If they genuinely cannot pre-state anything**, run it exploratory and **label it
  as such in the ledger** (judged post-hoc, not pre-registered), and **promote any
  newly-named implausibility to a recorded target** so the criterion is captured going
  forward. An insufficient set gets *tightened* by the act of looking.

### Plausibility & health checklist

Run through these against the simulated output — each maps to a common, concrete bug:

| Check | A failure looks like | Usual cause |
|---|---|---|
| **Support respected** | negative counts, rates > 1, probabilities outside [0,1], SDs ≤ 0 | an unconstrained parameter, or a wrong-support prior |
| **Order of magnitude** | data (or a derived quantity) 100× or 0.01× what the human expects | a unit slip, or a prior on the wrong scale |
| **No blow-ups** | `inf`, `NaN`, or astronomically large values | heavy-tailed prior with no bound; unstable dynamics for some prior draws |
| **No degeneracy** | every dataset identical, or all-zero / all-constant | a mis-wired simulator, or a prior collapsed to a point |
| **Runs cleanly** | crashes / hangs for some prior draws | simulator not robust across the full prior range (an engineering bug) |
| **Coverage** | real data sits outside the simulated cloud | prior(s) too narrow → will be OOD at stage 8 |

## On fail: diagnose, don't fix the science

A failed prior predictive check sends you **back to stage 2** (and, only with the
human's explicit approval, to the model/simulator itself). Your job is to **diagnose
and present options, not to change the priors or the mechanism** — whichever scale
the failure surfaced on (noisy data, a derived quantity, or a summary), route it the
same way. State which read-out failed and localise the likely cause with evidence:

- **Implausible output (plausibility fail)** → point to the likely culprit: a prior
  that is too wide or on the wrong scale, an unconstrained / wrong-support parameter,
  a mis-parameterised or wrong distribution. Show the offending datasets or derived
  quantities. Which prior to change, and how, is the human's scientific call. Route to
  stage 2.
- **Insufficient coverage (coverage fail)** → the training distribution will not cover
  the real data. Show *which* summaries of the real data fall at/outside the simulated
  cloud's edge and which prior(s) would need to widen. Widening a prior is a
  **scientific** change the human decides and approves; do not silently widen it.
  Route to stage 2, and flag that this is the stage-9 OOD failure caught early.
- **Simulator health fail (a bug)** → the one exception where the fix is usually
  **engineering, not science**. A `NaN` from an unguarded `log(0)`, a crash on an
  edge-case sample size, an indexing error — you *may* fix these, but still **surface
  the bug and get the human's OK before editing the simulator**, because the line
  between "implementation bug" and "the mechanism is wrong" is not always clear, and
  any change to simulator *behaviour* is the human's to approve. Never quietly patch
  simulator logic to make a plausibility check pass.

In all cases: diagnose the *likely* cause, present it as evidence and options, and
**wait**. The governing principle is the workflow's core split — *the agent owns the
engineering, the human owns the science*. A wrong prior or mechanism is science; a
broken line of code is engineering.

## No separate gate — but the loop-back is real

Stage 3 has no human-approval gate of its own (the priors were approved at the
stage-2 gate). But a failure routes back *through* that gate: any prior change the
human decides on must be re-approved at stage 2 before you re-run stage 3. Log the
loop-back in the iteration ledger as one plain sentence, e.g. *"prior predictive:
noise-free peak incidence 5× too high → you narrowed the `beta` prior → re-checked →
passed"*.

On a clean pass, Act I is complete: priors that are both plausible and cover the real
data, checked against pre-registered criteria where they exist and against the
human's recognition where they do not. Proceed to stage 4 (workflow design) — the
first stage that builds and trains a network.

## Using this skill

- **Entering from stage 2:** read the committed targets and coverage note from
  `prior-specification.md`; reuse the stage-2 summary definitions verbatim. Confirm
  the human's conventional view of their data and which interpretable latent/derived
  quantities they can reason about (ideally already noted at intake, stage 1).
- **On fail:** route back to `prior-elicitation` (stage 2) with a specific diagnosis;
  do not edit priors or mechanism yourself.
- **Engineering mechanics** (drawing from the simulator, adapters) live in
  `bayesflow-implementation`; the worked reference is `examples/toy-normal/`.

## References

- Gabry, Simpson, Vehtari, Betancourt & Gelman (2019), *Visualization in Bayesian
  Workflow*, JRSS-A 182(2):389–402 — the canonical treatment of prior predictive
  checks and how to *visualise* them, including checks on interpretable derived
  quantities, not just raw data. https://doi.org/10.1111/rssa.12378 — arXiv:1709.01449
- Gelman et al. (2020), *Bayesian Workflow* — prior predictive checking as a first,
  cheap line of defence in the iterative workflow. arXiv:2011.01808 —
  https://arxiv.org/abs/2011.01808
- Schad, Betancourt & Vasishth (2021), *Toward a principled Bayesian workflow in
  cognitive science* — the "four questions" framing whose **first** question is the
  prior predictive check. arXiv:1904.12765 — https://arxiv.org/abs/1904.12765
- Gelman, Simpson & Betancourt (2017), *The Prior Can Often Only Be Understood in the
  Context of the Likelihood* — why a marginally-reasonable prior must still be checked
  jointly, through the model, on the data scale. https://doi.org/10.3390/e19100555
- *Amortized Bayesian Workflow* (2024) — the coverage/OOD linkage: the prior is the
  training distribution, so prior-predictive coverage now is what prevents
  out-of-distribution failure at inference time (stage 8). arXiv:2409.04332 —
  https://arxiv.org/abs/2409.04332
