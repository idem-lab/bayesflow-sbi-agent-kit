---
name: workflow-orchestration
description: Orchestrates a principled Bayesian simulation-based inference workflow with BayesFlow 2, stage by stage, with human-approval gates and an explicit iteration loop. Use this as the top-level guide when a user starts (or resumes) an SBI project — it routes to per-stage skills and tells you when a failed check sends you back rather than forward.
---

# Bayesian SBI workflow — orchestration

This is the **map** for a principled amortised Bayesian workflow with BayesFlow 2.
Follow it stage by stage. It routes to per-stage skills where they exist and gives
you enough inline guidance to run each stage where they do not yet.

The workflow is grounded in the published Bayesian-workflow literature (see
**References**). Two ideas from that literature shape everything below:

1. **It is a loop, not a line.** A failed check does not mean "note it and move
   on" — it sends you *back* to an earlier stage. Marching through the stages once
   and declaring success is the most common way to get a confidently wrong answer.
2. **You validate the method on simulations before you trust it on real data.**
   Recovery, calibration, and predictive checks on simulated data come *first*;
   only then do you run the real observations through the trained network.

## The human is the scientist

You (the agent) own the *engineering*: adapters, networks, training, diagnostics,
plots. The human owns the *science*: the model/simulator, the priors, what counts
as a good enough fit, and what the results mean. Never change a prior, a simulator
mechanism, or a modelling assumption without the human's explicit approval.

When a check fails, your role is to **diagnose, not to fix the science**. You can
and should localise the likely cause — which parameter, prior, or distribution is
implicated, which part of the simulator, and any implementation bugs you spot (e.g.
an unconstrained parameter, a wrong or mis-parameterised distribution, an indexing
error). Report that as evidence and options. But *whether* the model is actually
wrong, and *how* to change it, is a scientific judgement that belongs to the human:
they are the one who should reason about why the model might be misspecified and
decide the fix. Do not edit the model or simulator yourself to make a check pass.

**The inference method is not one of the engineering knobs you own.** This kit's
method *is* amortised SBI in BayesFlow — that is the whole point. "Owning the
engineering" means adapters, networks, training, and diagnostics *within* that method;
it does **not** include swapping in a different inference engine (MCMC, a particle
filter, ABC, a bespoke sampler) or nesting one inside the SBI. Latent states,
hierarchical structure, and time series all stay inside BayesFlow (see the
`bayesflow-implementation` skill for how). If you become convinced the problem genuinely
needs a different or hybrid method, that is a **scientific/scope decision**: stop,
present the case and the trade-offs, and get explicit human sign-off — do not just build
it.

The **approval gates** below are hard stops: present findings, ask, and wait.

## Stages at a glance

```
  1. Project intake
  2. Prior review .............[GATE: human approves priors]
  3. Prior predictive check ...──fail──> back to 2 (or, w/ approval, the model)
  4. Workflow design
  5. Pilot training
  6. Parameter recovery .......──fail──> back to 4/5 (design/train) or, w/ approval, 2
  7. Calibration (SBC) ........──fail──> back to 4/5 (design/train)
  8. Real-data inference +
     reliability (OOD) check ..──fail──> back to 2 (widen priors/sim range) [GATE]
  9. Posterior predictive .....──fail──> back to 2 or the model [GATE]
 10. Human review ............ [GATE: human interprets & signs off]
```

Stages 4–7 (Act II) run entirely on *simulated* data — you prove the method works
before the real data is ever used. Stage 8 is the first time the real observations
drive inference, and it must certify the amortised posterior is trustworthy (in-
distribution) *before* stage 9 (posterior predictive) checks the fitted model against
those same observations.

## Orient the user, and keep a status ledger

The workflow is a loop, and users — especially domain modellers who are not
statisticians — get lost in the iterations. Two habits prevent that:

**1. Give the map up front (once, at intake).** Before writing any code, explain the
whole journey in plain language. Group the ten stages into three acts so it is not
overwhelming:

- **Act I — set up the model & your beliefs** (stages 1–3: intake, priors, prior
  predictive)
- **Act II — prove the method works on *simulated* data** (stages 4–7: design,
  train, recovery, calibration)
- **Act III — use it on your *real* data & review** (stages 8–10: reliability check,
  posterior predictive, human review)

Say explicitly that **it is a loop** — "we will sometimes go back a step, and that
is the workflow working, not failing" — and state the ground rules: you do the
engineering, they make the scientific calls, and you will stop and ask at each gate.

**2. Keep a living status ledger — a file, not just chat.** At intake, create
`sbi-workflow-status.md` in the user's repo from the template in
`templates/user-repo/sbi-workflow-status.md`. It is the durable source of truth for
"where are we": a *you-are-here* block, the stage checklist, a **decisions log**
(every scientific decision the human approved), and an **iteration log** (each
loop-back written as one plain sentence, e.g. "SBC failed for σ → you approved a
positivity constraint → retrained → passed"). Update it at **every stage transition,
every gate, and every loop-back**. Because it lives in a file, it survives new
sessions and context compaction, the human can read it themselves, and it doubles as
an audit trail that nothing scientific changed without their say-so.

**Keeping the user oriented as you go:**

- Begin each substantive turn with a **one-line status banner**, e.g.
  `📍 Stage 7/10 (Calibration) · iteration 2 · waiting on: your call on the σ prior`.
- At **every stage transition and gate**, give a short re-orientation: what we just
  finished, what is next, and what (if anything) you need from the human — then
  update the ledger.

---

## 1. Project intake

**Goal:** understand the science before writing any code. Elicit, in the human's
words: what process generates the data, what parameters they want to learn, what
the observed data look like (shape, size, how many datasets), and what decision the
inference will inform.

Also capture two things now that stage 3 will need: **how the human conventionally
looks at this kind of data** (the plot or summary they would use — reuse *their* view
for the prior predictive check rather than inventing one), and **which quantities
they can actually reason about** — the observable data, interpretable latent/derived
quantities (e.g. noise-free predicted cases, R₀, a doubling time), or the parameters
themselves. Users of very noisy-observation models often have their strongest
intuition on the derived quantities, not the raw data.

Write this down and read it back to the human. Do not infer a model mechanism they
did not state. Flag intractable-likelihood situations early — that is exactly where
amortised SBI earns its keep, and where MCMC alternatives may not be available.

**Note on when amortisation is worth it:** amortised inference pays off when you
will fit **many datasets** (train once, reuse). For a single one-off fit, the
upfront training cost may not be justified — say so, and let the human decide.

> **Future direction (out of scope for now): a "suggest an inference method" step.**
> This kit assumes amortised SBI is the right tool and stays inside it. A planned
> addition is an explicit step that, at this stage, weighs SBI against alternatives
> (MCMC/HMC, sequential/neural-sequential SBI, particle MCMC, ABC, variational
> inference) — including for the model *extensions* the human is likely to want. Some
> extensions break specific methods: e.g. a particle filter's likelihood factorisation
> assumes a Markov latent chain and breaks for a non-Markovian agent-based model or a
> joint metapopulation coupling all states at once, whereas amortised SBI over the joint
> trajectory does not. Such a step should (a) give each alternative's advantages/limits
> vs SBI, for the current model and its likely extensions; (b) recommend **robust,
> validated packages** (e.g. Stan/PyMC/NumPyro, `sbi`, `particles`, `pyABC`), never
> bespoke sampler code; and (c) explain how choosing it changes the rest of the
> workflow. Until that step exists, treat a method change as a human-approved scope
> decision (see "The human is the scientist"). Do not build a substitute method
> yourself.

## 2. Prior review

**Goal:** agree priors that encode the human's domain knowledge. Route to the
`prior-elicitation` skill — it drives this stage. Priors are a *scientific* choice.

Key moves that skill runs (do them here even without it):

- **Triage the parameters** by how much the human can reason about each directly:
  (a) direct & informative (evidence exists) → fit to their stated value + record
  the reference; (b) rough idea with real uncertainty → quantile elicitation +
  pushforward; (c) can't reason directly → elicit jointly on the observable scale,
  and if there are several, consider the algorithmic route (`elicito`).
- **Elicit on the observable scale, not the raw parameter scale**, and translate to
  distributions by *simulation* (the pushforward).
- **Commit to quantitative target summary statistics of the simulated data** — the
  acceptance criteria stage 3 will test, written down *before* the pushforward.
- Remember the SBI-specific point: **the prior is the training distribution**, so
  its pushforward must *cover* the plausible real data or stage 8 will be OOD. Flat
  priors are not an option.

Record everything in `prior-specification.md` (template in
`templates/user-repo/`); the committed target summaries feed straight into stage 3.

**GATE:** the human must approve the priors before you proceed.

## 3. Prior predictive check

**Goal:** confirm the priors + simulator produce *plausible data*, and that the
prior predictive *covers* the plausible real data, before any training. Route to the
`prior-predictive-check` skill — it drives this stage. This is the cheapest check in
the whole workflow and catches prior/model mistakes otherwise invisible until much
later.

Key moves that skill runs (do them here even without it):

- **Draw from the *joint* prior**, push through the simulator, and compute the
  **quantitative target summary statistics the human committed to at stage 2** (in
  `prior-specification.md`) — test against those pre-registered numbers, not an
  eyeball.
- Read **two** questions separately: *plausibility* (right magnitude/support, no
  absurdities) and *coverage* (does the simulated-data cloud span the real data?).
  Coverage matters uniquely in SBI: the prior is the training distribution, so a
  too-narrow prior here is a stage-8 out-of-distribution failure caught early.
- **You show, the human judges.** You never decide plausibility — you choose the
  *presentation* and the human renders the verdict. Present in *their* terms, on the
  scales they can reason about (data, interpretable latent/derived quantities, or
  parameters), and pose it as recognition ("does any of this look wrong?") not recall
  ("what did you expect?") — so a user who could not pre-state expectations can still
  do this stage. Where committed targets are thin or missing, test what exists and
  surface the rest exploratorily; do not dress post-hoc judgement up as pre-registered.
- Distinguish a **simulator bug** (engineering — you may fix it, with the human's OK)
  from a **wrong prior or mechanism** (science — the human decides).

**On fail:** implausible data or insufficient coverage → back to **stage 2**.
Diagnose, do not fix the science: point to the likely cause and show the evidence;
present options and wait. Do not edit the priors, model, or simulator to force a pass.

## 4. Workflow design

**Goal:** build the BayesFlow objects — simulator, adapter, summary network,
inference network. This is *engineering*; use the `bayesflow-implementation` skill
for the concrete tips, and `examples/toy-normal/` as a worked reference. Key points
that will bite you if skipped:

- **Constrain positive / bounded parameters** (`adapter.constrain("sigma", lower=0)`),
  or calibration will be biased at the boundary.
- **Exchangeable / variable-length observations** → `as_set` + a `DeepSet` summary
  network. **Ordered time series** → `as_time_series` + `TimeSeriesNetwork` /
  `TimeSeriesTransformer`; pad + mask for variable observed length.
- **Adapter hygiene** — `to_array` before `convert_dtype`; `drop` unused meta
  variables.
- **Latent states, hierarchical structure, and time series stay inside BayesFlow.**
  Infer latent trajectories / per-unit parameters as *targets* of the same amortised
  posterior (concatenate them into `inference_variables`); use `HierarchicalSimulator`
  for grouped parameters and `compositional_sample` only for exchangeable pooling. Do
  **not** substitute or nest a hand-built MCMC / particle filter / custom sampler as the
  inference engine — that is a scientific/scope change needing human sign-off, not a
  design detail (see "The human is the scientist" and the `bayesflow-implementation`
  skill's latent-states section; worked in `examples/flu-sir/`).

## 5. Pilot training

**Goal:** a short run to prove the pipeline trains and samples end to end before
spending on a full run. Watch the loss decrease; sample once and confirm shapes.
Set the Keras backend (`KERAS_BACKEND=jax`). Expect a slow first epoch with
variable set sizes (JAX recompiles per distinct size).

Only scale up epochs / network size once the pilot runs clean.

## 6. Parameter recovery

**Goal:** on held-out *simulated* datasets, does the posterior recover the true
parameters — **judged relative to its own uncertainty**, never as point accuracy?
Route to the `parameter-recovery` skill — it drives this stage. Runs on simulated
data only; read together with SBC (stage 7).

Key moves that skill runs (do them here even without it):

- **Do not read recovery as point accuracy.** Correlation / RMSE of the posterior
  *mean* are **width-blind**: an uninformative parameter's posterior correctly
  collapses to the prior, so those metrics look "broken" while inference is flawless.
  Both a poorly-identified parameter and a biased engine show low correlation — point
  metrics alone cannot tell them apart.
- Report, per parameter, **posterior contraction** (1 − Var_post/Var_prior) and
  **posterior z-score** ((mean − truth)/sd), and read the **z-score-vs-contraction
  sensitivity plot** (quadrants: small |z| + low contraction = poorly identified but
  benign; small |z| + high contraction = ideal; large |z| = biased / conflict).
- **Necessary but not sufficient:** always continue to SBC (stage 7) and reach the
  verdict from the two together.

**On fail — decide *which* failure it is first:** poorly identified (small |z|,
uniform SBC) is a *scientific finding* → surface, **do not retrain**; large |z| /
non-uniform SBC is an *engineering* problem → **stage 4/5**; low contraction with
large |z| is *prior–likelihood conflict* → *scientific*, **stage 2**.

## 7. Calibration — simulation-based calibration (SBC)

**Goal:** are the posteriors *calibrated*, not just accurate on average? Route to the
`calibration-sbc` skill — it drives this stage. Runs on simulated data only; read
together with recovery (stage 6).

Key moves that skill runs (do them here even without it):

- Run SBC: for many simulated datasets, the **rank** of each true value within its
  posterior draws should be ~**uniform**. ∩/∪ = over/under-confidence; a slope = bias.
- SBC is **width-aware** — this is what catches problems recovery misses (in the toy,
  `sigma` had strong recovery yet biased SBC until constrained) and what separates a
  genuinely poorly-identified parameter (uniform ranks) from a biased engine
  (non-uniform), so **read it with the stage-6 sensitivity plot** via the combined
  decision table in that skill.
- The commonest concrete fix is **constraining a bounded parameter**
  (`adapter.constrain("sigma", lower=0)`). Where low-dimensional, also cross-check an
  exact grid/analytic reference (`scipy.stats`, not hand-coded).

**On fail:** biased ranks → **stage 4** (add a constraint) or **5** (train more);
over/under-dispersion → usually more training. **Uniform ranks with poor stage-6
point-recovery is *not* a fail** — it is the poorly-identified case, a scientific
finding, not an engine bug.

## 8. Real-data inference + reliability (OOD) check

**Goal:** run the actual observed data through the trained network **and check the
amortised posterior is trustworthy for this data**. Route to the
`real-data-reliability` skill — it drives this stage. An amortised posterior is only
reliable on data resembling the training simulations; a real dataset unlike anything
seen in training is confidently wrong with no error message. This is the first stage
that uses the real observations, and **it gates stage 9 (posterior predictive) and the
human review** — do both only once it passes.

Key moves that skill runs (do them here even without it):

- Produce the posterior via `workflow.sample(...)`, but **do not interpret it until
  the OOD check passes.**
- **OOD check:** verify the real data is *typical* under the training/prior-predictive
  distribution — compute the summary embedding (or simple summaries) for the real data
  and a large prior-predictive sample and check the real data sits **inside that
  cloud** (Mahalanobis distance / low-dim projections / MMD). This is the stage-3
  coverage question, now tested against the actual data.
- Also check the posterior lands in a well-sampled region of parameter space, not out
  on a tail the flow rarely saw.

**On fail (real data is OOD):** the training distribution did not cover the real data
→ priors/simulator ranges too narrow, or the model is misspecified → back to
**stage 2**. **GATE:** widening a prior or simulator range is the human's scientific
call; then you retrain. Importance-sampling reweighting rescues *mild* cases. A
likelihood-based MCMC fallback can be accurate *when a likelihood is available* — but
it is a **last-resort, human-approved escalation, not a default**: for the
intractable-likelihood models this kit targets it usually is not available (which is why
detecting the problem is the whole defence), and reaching for it is a method/scope change
(see "The human is the scientist"), not something you switch to on your own.

## 9. Posterior predictive check

**Goal:** does the fitted model, conditioned on the **real** data, reproduce the
features of that data? Route to the `posterior-predictive-check` skill — it drives
this stage. **Run it only after the reliability/OOD check (stage 8) has passed** — a
posterior predictive check on an untrustworthy (OOD) posterior is uninterpretable.

Key moves that skill runs (do them here even without it):

- Draw from the **real-data posterior**, simulate replicated datasets, and compare
  to the observations on the scales the human reasons about (their data view + derived
  quantities) — using **targeted discrepancy measures, especially features the model
  was *not* directly fit on** (dispersion, tails, extremes, zeros, autocorrelation).
- Prefer graphical / per-feature checks to a single posterior predictive p-value
  (which is conservative and not a calibrated p-value).
- **SBI payoff:** with the engine already exonerated (recovery + SBC passed) and the
  data in-distribution (stage 8), a systematic mismatch is **model misspecification**,
  not an inference-network fault.

**GATE + on fail:** a failing check is a *scientific* signal (the model is missing
something). Diagnose which features fail and which component (priors, likelihood,
missing mechanism) is the cause; present options. The remedy — revising priors
(**stage 2**) or the model — is the human's to decide. Do not silently retune.

## 10. Human review

**GATE:** the human interprets the posterior and decides whether it is fit for its
purpose. Present: the posteriors with uncertainty, every diagnostic (recovery, SBC,
posterior predictive, OOD result), and any caveats (e.g. residual mild
overconfidence, parameters that were weakly identified). Do not overstate. The human
signs off — or sends you back into the loop.

---

## Using this skill

- **Starting fresh:** give the up-front map, create the status ledger, then go top
  to bottom, honouring the gates.
- **Resuming:** read `sbi-workflow-status.md` to see where you are (stage, iteration,
  pending decisions) and continue; if a check has since failed, follow its
  back-arrow.
- **Route out** to per-stage skills (`prior-elicitation`, `prior-predictive-check`,
  `parameter-recovery`, `calibration-sbc`, `real-data-reliability`,
  `posterior-predictive-check`) and to `bayesflow-implementation` for the engineering
  mechanics. `examples/toy-normal/` is the worked reference for the engineering path
  this workflow was validated against — design → pilot training → recovery → SBC →
  reliability/OOD → posterior predictive (stages 4–9). Intake, elicitation and the
  human-judgement parts of stages 1–3 are not scripted (they are conversational).

## References

The workflow synthesises the following published sources; consult them for depth:

- Gelman et al. (2020), *Bayesian Workflow* — the canonical, iterative, non-linear
  workflow; stresses that fake-data recovery means recovering parameters *to within
  their uncertainty*, not to a point (stage 6). arXiv:2011.01808 —
  https://arxiv.org/abs/2011.01808
- Schad, Betancourt & Vasishth (2021), *Toward a principled Bayesian workflow in
  cognitive science* — the "four questions" framing (prior predictive,
  computational faithfulness, model sensitivity, posterior predictive) that our
  stages mirror, and the source of the **posterior z-score vs. posterior
  contraction** sensitivity plot and its quadrant interpretation used in stage 6.
  arXiv:1904.12765 — https://arxiv.org/abs/1904.12765
  (sensitivity-plot quadrants figure: https://www.researchgate.net/figure/Posterior-z-scores-as-a-function-of-posterior-contraction-Arrows-show-four-possible_fig1_342288675)
- Talts, Betancourt, Simpson, Vehtari & Gelman (2018), *Validating Bayesian
  Inference Algorithms with Simulation-Based Calibration* — the SBC rank-uniformity
  check used in stage 7, and the reason it is width-aware. arXiv:1804.06788 —
  https://arxiv.org/abs/1804.06788
- *Amortized Bayesian Workflow* (2024) — the amortised/SBI-specific additions:
  out-of-distribution detection at inference time, importance-sampling correction,
  and MCMC fallback (stage 8). arXiv:2409.04332 — https://arxiv.org/abs/2409.04332
