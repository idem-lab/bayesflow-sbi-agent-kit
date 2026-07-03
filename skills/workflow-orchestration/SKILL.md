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
  8. Posterior predictive .....──fail──> back to 2 or the model [GATE]
  9. Real-data inference +
     reliability (OOD) check ..──fail──> back to 2 (widen priors/sim range) [GATE]
 10. Human review ............ [GATE: human interprets & signs off]
```

Stages 6–8 are the "does the method work?" block, run entirely on *simulated* data.
Stage 9 is the first time real observations drive inference.

## Orient the user, and keep a status ledger

The workflow is a loop, and users — especially domain modellers who are not
statisticians — get lost in the iterations. Two habits prevent that:

**1. Give the map up front (once, at intake).** Before writing any code, explain the
whole journey in plain language. Group the ten stages into three acts so it is not
overwhelming:

- **Act I — set up the model & your beliefs** (stages 1–3: intake, priors, prior
  predictive)
- **Act II — prove the method works on *simulated* data** (stages 4–8: design,
  train, recovery, calibration, posterior predictive)
- **Act III — use it on your *real* data & review** (stages 9–10: reliability check,
  human review)

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

Write this down and read it back to the human. Do not infer a model mechanism they
did not state. Flag intractable-likelihood situations early — that is exactly where
amortised SBI earns its keep, and where MCMC alternatives may not be available.

**Note on when amortisation is worth it:** amortised inference pays off when you
will fit **many datasets** (train once, reuse). For a single one-off fit, the
upfront training cost may not be justified — say so, and let the human decide.

## 2. Prior review

**Goal:** agree priors that encode the human's domain knowledge. Route to the
`prior-elicitation` skill if present. Priors are a *scientific* choice.

**GATE:** the human must approve the priors before you proceed.

## 3. Prior predictive check

**Goal:** confirm the priors + simulator produce *plausible data* before any
training. Draw parameters from the prior, push them through the simulator, and
show the human summaries/plots of the simulated data.

**What you are looking for:** simulated datasets that look like the kind of data
the human expects — right order of magnitude, right support, no absurdities
(e.g. negative counts, impossible rates). This is the cheapest check in the whole
workflow and catches prior/model mistakes that are otherwise invisible until much
later.

**On fail:** implausible simulated data → back to **stage 2**. Diagnose, do not fix
the science: point to the likely cause (a prior that is too wide or on the wrong
scale, an unconstrained parameter, a wrong distribution, or a bug in the simulator)
and show the human the evidence. Whether the model is wrong and how to change it is
their scientific decision — present options; do not edit the priors, model, or
simulator yourself.

## 4. Workflow design

**Goal:** build the BayesFlow objects — simulator, adapter, summary network,
inference network. This is *engineering*; use the `bayesflow-implementation` skill
for the concrete tips, and `examples/toy-normal/` as a worked reference. Key points
that will bite you if skipped:

- **Constrain positive / bounded parameters** (`adapter.constrain("sigma", lower=0)`),
  or calibration will be biased at the boundary.
- **Exchangeable / variable-length observations** → `as_set` + a `DeepSet` summary
  network.
- **Adapter hygiene** — `to_array` before `convert_dtype`; `drop` unused meta
  variables.

## 5. Pilot training

**Goal:** a short run to prove the pipeline trains and samples end to end before
spending on a full run. Watch the loss decrease; sample once and confirm shapes.
Set the Keras backend (`KERAS_BACKEND=jax`). Expect a slow first epoch with
variable set sizes (JAX recompiles per distinct size).

Only scale up epochs / network size once the pilot runs clean.

## 6. Parameter recovery

**Goal:** on held-out *simulated* datasets, does the posterior recover the true
parameters — **judged relative to its own uncertainty**, never in absolute terms?

**Do not read recovery as point accuracy.** Correlation and RMSE of the posterior
*mean* vs. truth are **width-blind**. When the data are genuinely uninformative
about a parameter, the posterior correctly collapses back onto the prior, the mean
barely tracks the truth, and these metrics look "broken" even though the inference
is flawless — a wide-but-honest posterior is a *correct* result, not a failure.
This is exactly why point-recovery alone **cannot** tell a poorly-identified
parameter apart from a biased inference engine: both show low correlation. To
separate them, report, per parameter:

- **posterior contraction** = 1 − Var_post / Var_prior. ≈ 0 means the data added
  almost nothing (posterior ≈ prior); ≈ 1 means the parameter is pinned down.
- **posterior z-score** = (posterior mean − true value) / posterior sd — a
  *standardised* error (≈ N(0, 1) if calibrated). Small |z| means the posterior is
  consistent with the truth *given its own width*.

Plot z-score against contraction (the "sensitivity" plot). Its four quadrants are
named and tell you *which* regime you are in:

| | low contraction | high contraction |
|---|---|---|
| **small \|z\|** | parameter **poorly identified** (benign) | ideal |
| **large \|z\|** | prior–likelihood conflict | overfitting to noise |

**Necessary but not sufficient** — recovery still cannot certify calibration on its
own; always continue to SBC (stage 7) and read the two **together**.

**On fail — decide *which* failure it is before acting:**

- Poor point-recovery **but** small |z| **and** (stage 7) **uniform SBC** → the
  inference is correct and the parameter is genuinely **poorly identified** by the
  data. This is a *scientific / identifiability finding*, not an engine bug: surface
  it to the human; **do not** retrain to try to "improve" it (you can't — the data
  are uninformative).
- **Large |z| and/or non-uniform SBC** → a real inference problem (bias / wrong
  width) → back to **stage 4/5** (constrain a parameter, train longer, larger flow,
  better summary network).
- Low contraction **with large |z|** → **prior–likelihood conflict**: the prior and
  the data disagree. That is a *scientific* signal → diagnose and surface to the
  human (stage 2); do not silently move the prior.

The z-score / contraction diagnostic and its quadrant interpretation are from Schad,
Betancourt & Vasishth (2021); recovery-relative-to-uncertainty is stressed by Gelman
et al. (2020). See **References**.

## 7. Calibration — simulation-based calibration (SBC)

**Goal:** are the posteriors *calibrated*, not just accurate on average? Run SBC:
for many simulated datasets, compute the rank of each true value within its
posterior draws; the ranks should be ~uniform. A ∩ or ∪ shape means over/under-
confidence; a slope means bias.

This is the check that catches problems recovery misses — in the toy example
`sigma` had strong recovery yet biased SBC until it was constrained. Crucially, SBC
is **width-aware**: uniform ranks certify that a *wide* posterior is nonetheless
honest, while non-uniform ranks flag bias regardless of width. That is precisely
what separates a genuinely poorly-identified parameter (stage 6, small |z|) from a
broken engine (biased) — so always read SBC together with the stage-6 sensitivity
plot. Where the problem is low-dimensional, also **cross-check against an exact
reference** (grid or analytic posterior) built from validated libraries
(`scipy.stats`), not hand-coded densities.

**On fail:** biased ranks → back to **stage 4** (e.g. add a constraint) or **5**
(train more). Overdispersion/underdispersion → usually more training / larger flow.
Note: uniform ranks with *poor* stage-6 point-recovery is **not** a fail — it is the
poorly-identified case above, and a scientific finding, not an engine bug.

## 8. Posterior predictive check

**Goal:** does the fitted model reproduce features of the **real** observed data?
Draw parameters from the posterior given the real data, simulate new datasets from
them, and compare to the observations (summaries, overlays). Systematic mismatch
means the *model* is missing something real.

**GATE + on fail:** a failing posterior predictive check is a *scientific* signal —
the model is likely missing something real. Present it to the human and diagnose:
which features of the data the model fails to reproduce, and which components
(priors, likelihood, missing mechanism) are the plausible cause. The remedy —
revising priors (**stage 2**) or the model itself — is the human's to reason about
and decide. Do not silently retune priors or the model to make the check pass.

## 9. Real-data inference + reliability (OOD) check

**Goal:** run the actual observed data through the trained network **and check the
amortised posterior is trustworthy for this data**. An amortised posterior is only
reliable on data that resembles the training simulations; a real dataset unlike
anything seen in training gives a confidently wrong answer with no error message.

**What to do:**

- Produce the posterior for the real observation(s) via `workflow.sample(...)`.
- **Out-of-distribution check:** verify the real dataset is *typical* under the
  training/prior-predictive distribution. A practical approach: compute the summary
  network's embedding (or simple data summaries) for the real data and for a large
  prior-predictive sample, and check the real data sits inside that cloud (e.g.
  Mahalanobis distance vs. the simulated summaries, or eyeball low-dim projections).
- If the real data is OOD, the posterior is **not** to be trusted as-is.

**On fail (real data is OOD):** this means the training distribution did not cover
the real data — the priors or simulated data ranges are likely too narrow, or the
model is misspecified. Route back to **stage 2**. Diagnose which is more likely
(e.g. which summaries of the real data fall outside the simulated cloud, and which
priors/mechanisms would need to move to cover them), and present it. Widening a
prior or a simulator range is a *scientific* change the human must decide and
approve — **GATE** — not something you do on your own to force the data in-
distribution. Once they decide, you retrain. Optionally,
importance-sampling reweighting of the amortised draws can rescue mild cases, and
a likelihood-based fallback (e.g. MCMC seeded from the amortised draws) is the
gold-standard escalation *when a likelihood is available* — but for many
intractable-likelihood SBI problems it is not, which is why the OOD check matters
so much here.

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
- **Route out** to per-stage skills (`prior-elicitation`, prior-predictive,
  calibration, …) as they become available, and to `bayesflow-implementation` for
  the engineering mechanics. `examples/toy-normal/` is the end-to-end worked
  reference that this workflow was validated against.

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
  and MCMC fallback (stage 9). arXiv:2409.04332 — https://arxiv.org/abs/2409.04332
