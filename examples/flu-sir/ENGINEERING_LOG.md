# Engineering log — influenza SIR forecasting example

A running record of the design decisions, experiments, and dead-ends behind this
example. The point of the log is not tidiness: the repo's principle is
*verify by running, not by reasoning*, so this file records **what was actually run
and what it showed**, including things that did not work. It is the raw material for
future "how to engineer an SBI time-series model" and "how to scale training to GPUs"
skills.

Roles throughout (the repo's throughline): **the agent owns the engineering**
(simulator vectorisation, network, training, diagnostics); **the human owns the
science** (model mechanism, priors, what the results mean). This example was built
autonomously as a dogfooding exercise, so the agent also stood in for the human's
scientific calls — every such call is flagged **[SCIENCE — would normally be a gate]**
so a reviewer can find and challenge them.

---

## 0. Task and framing

Forecast weekly influenza cases in US states, one amortised neural estimator trained
once on a mechanistic simulator and reused across all states and all forecast dates.
The scenario is an analyst in **late 2018**, between seasons, preparing a model for the
coming 2018/2019 season. Priors and all method-validation checks may use **only**
seasons *before* 2018/2019; the final fit is the 2018/2019 season.

Target model: **discrete-time, discrete-state stochastic SIR** (chain-binomial;
demographic stochasticity → integer, non-repeating trajectories), single homogeneous
population, **time-varying transmission** `β_t` (autocorrelated), static recovery rate
`γ`, **imperfect + lagged case reporting** (a reported fraction `ρ` and a random
reporting delay). Some parameters are expected to be weakly identified.

---

## 1. Environment (measured, not assumed)

- CPU-only. `jax.devices()` → `[CpuDevice(id=0)]`. **No GPU.** 16 cores, 30 GB RAM.
  This is the single biggest driver of the engineering: the simulator must be fully
  vectorised (batched tensor ops, no per-draw Python loops), the network must stay
  small, and online training must keep the simulator cost per batch low. It is also
  what makes the "dispatch to GPU/runpod" notes (§ end) real rather than hypothetical.
- Stack: Python 3.12, BayesFlow 2.0.12, Keras 3, JAX backend, `KERAS_BACKEND=jax`.
  Same `uv`-managed `.venv` as the toy example.

## 2. Data acquisition (de-risked first)

The whole final stage depends on getting the real data, so this was settled before any
modelling.

- `cdcfluview` was **archived from CRAN** ("not available for this version of R", R
  4.6.1). Installed `cdcfluview_0.9.4` from the CRAN *Archive* + `MMWRweek`; all other
  deps (`sf`, `httr`, `dplyr`, …) already present, GDAL/GEOS/PROJ on the system.
- `cdcfluview::who_nrevss("state")` returned 3 tables. Saved **raw CSVs** under
  `data/` so nothing downstream ever depends on the live CDC endpoint again
  (`who_nrevss_state_*.csv`). The endpoint had already flaked once (503) — caching the
  raw pull is the reproducibility insurance.
- **Signal chosen:** `icl_nrevss_clinical_labs` — weekly `total_a + total_b` (positive
  influenza specimens) per state. It is a clean weekly integer count, and it covers
  2015/16 onward on a single consistent measurement basis, i.e. the 2018/19 target plus
  three usable pre-target seasons (2015/16, 2016/17, 2017/18). The older
  `combined_prior_to_2015_16` table uses a different basis and is not mixed in.
  54 "regions" (50 states + DC + territories + NYC); some sparse early weeks are `NaN`.

### What the pre-target seasons show (drives prior elicitation, stage 2)

Measured from 2015/16–2017/18 (national = sum over states), season indexed from MMWR
week 40:

- **Timing.** Epidemics ramp from ~MMWR wk 47–49 (early–mid Dec) and peak MMWR wk 5–10
  (early Feb–mid Mar). Season activity spans roughly Dec–Apr. → the epidemic "months".
- **Magnitude.** Per-state *peak weekly positives* vary enormously: 10th/50th/90th
  percentile ≈ 40 / 200–370 / 800–1400, max ≈ 2700 (2017/18, the big season). Small
  states peak in the tens; large states in the low thousands. **One estimator must span
  ~2 orders of magnitude of case scale across states** → the effective susceptible pool
  `S0` is inferred per-fit with a broad prior (rather than feeding population as a known
  covariate), and it is deliberately confounded with `ρ` (the identifiability lesson).

### 2018/2019 target window (definition, reproducible)

The national 2018/19 curve rises from ~1090/wk (MMWR 48, early Dec) to a plateau peak
~13 900/wk (MMWR 8, late Feb), and shows two real-world wrinkles the model/PPC will
have to cope with: the **early-January holiday reporting dip** and the **missing MMWR
week 53** (2018 has no wk 53). Using `wk_date` (real Saturdays) as the index sidesteps
the week-numbering gap.

- **Epidemic week 1 ≡ first weekly date on/after 2018-10-01**, then **39 consecutive
  weeks** — the **full CDC forecasting season** (~1 Oct → ~30 Jun; MMWR wk 40 → 26). All
  51 states have exactly 39 weekly reports in this window in every season. This captures
  the *whole* epidemic — slow autumn onset, midwinter growth-through-peak, spring tail —
  for every state on one aligned index. **[SCIENCE — would normally be a gate]** the
  choice of onset/window is a modelling decision; a reviewer may prefer a data-driven
  per-state onset. *(An earlier build used a 16-week Dec-start window; the maintainer
  asked for the full Oct–Jun season the CDC actually forecasts — see Iteration 7.)*
- **Forecast protocol:** an analyst re-forecasts once a "month" (4-week block), so the
  observed length at forecast time is a cutoff `L ∈ {4, 8, …, 36}`; for each, forecast the
  next 4 weeks. The one amortised estimator is trained over a random `L` so it serves every
  cutoff across the whole season.

---

## 3. Design decision: how the latent states are inferred (the crux)

The brief requires (a) one estimator, trained once, reused across **variable-length**
observation windows, producing **full posterior samples**; and (b) forecasts that
**start from posterior samples of the latent epidemic state** at the forecast date, so
in-sample uncertainty is low and forecast uncertainty grows. Crucially, the point of this
example is to **exercise pure amortised SBI in BayesFlow for latent-state / hierarchical
inference** — the inference engine must stay inside BayesFlow.

**Current design — joint SBI over `[globals θ + the full latent infection trajectory]`.**
The latent epidemic states are **inference targets of the same amortised posterior**, not
something a separate sampler reconstructs. Each simulation runs the full `T_MAX = 39`-week
season and the inference target is `[θ (7) + log(weekly infections + 1) for all 39 weeks]`
(46-dim). The *condition* is the observed weekly case series truncated at a random monthly
cutoff `L`, padded to 39 weeks with a **mask channel**, so one estimator amortises over
every forecast date. Network: a 2-channel `TimeSeriesNetwork` summary (`[cases, mask]`) →
`CouplingFlow`.

Everything the brief asks for falls out of that **one** posterior, with no second
inference method:
- **in-sample fit** (weeks ≤ L) = the reporting-model pushforward of the posterior
  infections — data-pinned, tight;
- **forecast** (weeks > L) = the reporting-model pushforward of the posterior *future*
  infections, which the flow learned to extrapolate dynamically-consistently from the
  simulator — uncertainty grows with horizon.

This also answers the brief's "one simulation storing the truth, or a separate forward
step?" question: **one simulation of the whole season; the future weeks are part of the
inference target**, and the forecast is a deterministic reporting pushforward of them —
no separate forward-simulation and no filter. Real-time censoring is automatic: truncating
a full-season report at column L equals re-censoring the reporting process at cutoff L
(a source-week-`w` report at delay `k` lands in column `w+k`, kept iff `w+k < L`).

Why infer *infections* rather than S/I/β directly: infections are the minimal sufficient
latent path (S/I follow by cumulative bookkeeping given S0; cases follow via ρ, delay ∈ θ),
lower-dimensional, and directly what the forecast needs. Validated by **per-week SBC on the
trajectory** (Iteration 7), not just the globals.

### Earlier approach and why it was replaced (honest record)

The first build **factorised** the posterior as `p(θ|y)·p(states|θ,y)`: an amortised flow
for the globals plus a **bootstrap particle filter** (`particle_filter.py`) for the latent
states and forecasts. It worked and calibrated well, but the maintainer correctly flagged
that **nesting a particle filter inside the SBI defeats the purpose of the toolkit** — the
example exists to exercise *pure* amortised SBI for hierarchical/latent-state models, and a
hand-built sampler is a change of inference *method* (a scientific/scope decision), not an
engineering detail. Research against the BayesFlow 2.0.12 source confirmed the pure route is
available and correct (latent trajectory as a target; `HierarchicalSimulator` for grouped
params; composition is exchangeable-only, not for Markov chains; no Simformer/arbitrary-
subset conditioning — these conclusions are written up in the `bayesflow-implementation`
skill's latent-states section). The particle filter was removed and the model reimplemented
as above. **Guardrail added** so future workflow agents don't repeat the mistake:
`skills/bayesflow-implementation` (latent-states section) and `skills/workflow-orchestration`
(stage 4 "stays inside BayesFlow"; the inference method is not an engineering knob).

---

## 4. Iteration log

> **Iterations 1–6 below describe the superseded first build** — a 16-week Dec-start window
> with the amortised-globals + **particle-filter** factorisation of §3. They are retained
> verbatim as an honest record (the identifiability lesson, the stage-8 shape-mismatch
> diagnosis, and the GPU-scaling notes all still apply). **Iteration 7 is the current build**:
> the full Oct–Jun season and the **pure-SBI trajectory** design. Where numbers differ, the
> README status block and Iteration 7 are authoritative.

### Iteration 1 — simulator + prior predictive (stage 3)

- **v1 priors** (γ ≈ 3-day clinical infectious period, R0 ≈ 1.35): prior-predictive
  epidemics grew far too fast — doubling < 1 week (real ≈ 2.5 wk), peak-week median 6
  (real 10), only 32% peaking in weeks 9–16 (real ~75%). The loose "support overlaps"
  coverage band said PASS, but the timing/growth *centre* was wrong. **Verified by
  running**, not reasoned. Also tightened the check: shape/timing summaries must be
  *centred* in the real IQR, not merely span it.
- **Science call [would normally be a gate]:** lengthened the SIR **generation interval**
  to ~6 days (γ ~ LogNormal(log 1/6, 0.35)). Rationale: fitting a single-compartment SIR
  to smoothed surveillance data needs an effective generation interval longer than the
  clinical infectious period (heterogeneity + depletion inflate it) to reproduce the
  observed slow growth at flu-like R0. → peak-week median 11, growth median 0.30, 72%
  peaking weeks 9–16. Timing/growth now centred.
- **Coverage call:** widened `S0` (median 2e5→3e5, σ 1.2→1.35) so the prior-predictive
  peak 95th (2881) reaches the biggest states (2018/19 max peak ~2450) — protects against
  a stage-8 OOD failure on large states. Final stage-3: coverage PASS, plausibility good
  (all-zero 2.5%, no explosions).

### Iteration 2 — estimator build + pilot (stages 4–5)

- **Architecture.** Summary net `TimeSeriesNetwork` (LSTNet: bidir GRU recurrent_dim 64 +
  skip-conv), summary_dim 24; inference net `CouplingFlow` (full posterior samples).
  Adapter: log1p+standardize the case series for the GRU; **`constrain(lower=0)` is
  identity/softplus in this BayesFlow build — verified by running** — so it does NOT
  compress `S0`'s 1e4–1e6 scale; used `.log()` on the 6 positive params and
  `constrain(rho, 0, 1)` (proper logit) instead, then BasicWorkflow affine-standardizes.
- **Bug caught by running the pilot:** `report()` crashed on 4-week windows — a reporting
  delay lag `k ≥ W` produced a negative array slice. Fixed to skip placement (full
  censoring) for `k ≥ W`. Unit-tested across W ∈ {4,5,8,16}.
- **Pilot (3 epochs × 20 batches):** trains, samples; on a held-out 12-week sim all 7
  true params fell in their 90% CIs. First epoch slow (JAX recompiles once per distinct
  window length T ∈ 4..16), then ~0.1–0.3 s/step. → full training run launched in the
  background; weights saved to `outputs/flu_sir.weights.h5` for reuse by all later stages.
- **Full run:** 60 epochs × 250 batches (batch 128), ~20 min on CPU, loss 9.9 → 7.0.
- **Load bug caught by running:** `load_trained` failed — `sample()` calls the adapter with
  `strict=False`, but `concatenate` needs a prior `strict=True` pass (normally done by
  `fit`). Fixed by running a tiny 1-batch `fit_online` to build layers before
  `load_weights`.

### Iteration 3 — state inference: bootstrap particle filter

- `particle_filter.py` (backend-free, NumPy+SciPy). Augmented per-particle state
  (S, I, log β, reporting-delay pipeline); Poisson observation weight around the
  ρ·infections·geometric-delay convolution (SciPy `poisson`, not hand-rolled); resample
  within each θ draw. **Two shape/logic bugs caught by running** (negative-slice censoring
  in `report`; a (horizon,P) vs (P,horizon) transpose). Standalone check with known θ on a
  simulated epidemic (peak 389): in-sample fit hugs the data (avg 90% width ~23) while the
  4-week forecast widths grow 132→305→553→744 and the truth stays in the bands — exactly
  the required low-in-sample / growing-forecast behaviour.

### Iteration 4 — recovery + SBC (stages 6-7)

- All diagnostics from `bayesflow.diagnostics`, computed on the inference scale (log /
  logit). 500 test datasets × 300 draws, at T = 6/10/14 to check amortisation.
- **Result = the "poorly-identified but calibrated" branch, on a realistic model.**
  SBC calibration error small for all (0.005–0.023) and near-zero bias → posteriors
  **calibrated & unbiased**. Contraction is low for `sigma_lb`/`delay_mean` (~0),
  modest for `rho`/`gamma`/`S0`, best for `I0`; and **grows with window length**
  (`S0` 0.00→0.55, `R0` 0.32→0.46 from T=6→14). Per the combined stage-6/7 table this is
  *genuinely poorly identified* (low contraction + small |z| + uniform SBC), **not** an
  engine fault → **do not retrain**. Confirms the stage-2 identifiability flags. Because
  SBC passes, a later posterior-predictive failure implicates the *model*, not the network.
- Note for the writeup: weak identification of individual θ does **not** doom forecasting —
  the PF conditions the latent *state* on data, and short-horizon forecasts depend on the
  current growth/state, which is far better constrained than R0/γ separately.

### Iteration 5 — reliability / OOD (stage 8): a real loop-back

The single most important stage, and the workflow earned its keep here.

- **v2 estimator flagged the real 2018/19 data as OOD**: summary-space MMD p ≈ 0.000 at
  T = 4/8/12 (real MMD 0.47–0.58 vs null 95th ≈ 0.16), and 27–41% of states flagged
  per-state (vs ~5% by chance) — the biggest states (Florida, California, Texas, …).
- **Diagnosis (verified by running, not assumed).** Ruled out overdispersion: real weekly
  counts are only 1.18× jaggier than simulated and *less* over-dispersed (index of
  dispersion 56 vs 87) — adding observation noise would make it worse. The driver is a
  **shape-distribution mismatch**: stage 3 confirmed the *marginal* summary stats (peak,
  timing, growth, total) are covered, but the summary-network MMD is sensitive to the
  *joint fine shape* (plateaus, the holiday dip, secondary waves) that a smooth single-wave
  SIR under-produces.
- **Science call [would normally be a gate] → loop back to stage 2.** The defensible
  structural lever for shape diversity is the transmission volatility: widened
  `sigma_lb` HalfNormal(0.05 → 0.10). Confirmed on the pushforward *before* retraining
  that this moves simulated roughness toward real (0.270 → 0.288 vs 0.318) and keeps
  scale/timing coverage. **Retrained (v3)** and re-ran stage 8. Result recorded in the
  README status block. Honest expectation logged up front: widening volatility should
  *mitigate* but not fully erase a structural shape gap — a simple SIR cannot reproduce
  every real-flu feature, and the value of the stage-8 check is that it forces this to be
  an explicit, recorded scientific decision instead of silently shipping overconfident
  forecasts. Residual-OOD states are carried as an explicit caveat into stage 10, and the
  PF's data-conditioning keeps short-horizon forecasts usable even where the *global* shape
  is atypical.

### Iteration 6 — posterior predictive (stage 9) + forecasts (stage 10)

- **Stage 9 in-sample PPC largely PASSES**, and this taught a real lesson. Targeting
  epidemic-shape statistics (roughness, peak, total, lag-1 autocorrelation) via ArviZ on
  the PF's in-sample posterior-predictive replicates: aggregate roughness p-value median
  **0.64**, only **5% of states** flagged (= chance), featured-state p-values moderate.
  This does **not** contradict stage 8. Stage 8 measured the *prior*-predictive's joint
  shape distribution (OOD); stage 9 measures the *conditioned* fit — and the flexible
  time-varying β lets the fitted model reproduce each state's observed shape. **Lesson:
  OOD-of-the-prior-predictive ≠ poor-fit.** (I had initially hypothesised the model would
  read "too smooth"; running it showed the opposite — replicate roughness slightly exceeds
  observed, consistent with sim dispersion 87 > real 56. Recorded the correction rather
  than the guess.)
- **Stage 10 forecasts.** Per state, per month k∈{1,2,3}: apply the estimator to weeks
  1..4k, particle-filter the states, forward-simulate 4 weeks. **Out-of-sample 90%
  forecast-interval coverage = 89%** across 153 fits (per horizon +1/+2/+3/+4 wk =
  81/86/93/95%). Near-perfect calibration — and it holds even for the stage-8-flagged
  states, because the PF conditions on real data, so short-horizon skill survives a
  global-shape prior mismatch. The +1-week slight under-coverage (81%) is where the
  reporting-delay right-censoring bites hardest — a genuine real-time nowcasting effect the
  model represents but cannot fully pin.

### Iteration 7 — redesign: full Oct–Jun season + pure-SBI trajectory (maintainer loop-back)

The maintainer flagged two things about the first build (Iterations 1–6): it modelled only a
16-week slice, not the full CDC forecasting season; and it used a **hybrid SBI + particle
filter**, which undermines the toolkit's purpose (exercising *pure* amortised SBI for
latent-state / hierarchical models). Both are addressed here; see §3 for the design and the
`bayesflow-implementation` skill (latent-states section) for the BayesFlow research behind it.

- **Full season.** Window widened to **39 weeks (~1 Oct → ~30 Jun)**; all 51 states have
  exactly 39 weekly reports every season. Forecasts issued at monthly cutoffs
  `L ∈ {4,…,36}` across the whole season.
- **Prior re-tune at full length (stage 3, [SCIENCE — would normally be a gate]).** From an
  October start the old priors peaked far too early — median peak week ~11, many peaking in
  weeks 3–5 — because the epidemic took off immediately from a median-50 seed. **Verified by
  running**, then fixed by lowering/tightening the growth and shrinking the autumn seed:
  `R0` LogNormal(log 1.26, 0.08) (growth `≈7·γ(R0−1) ≈ 0.25/wk`, matching the observed
  build-up), `I0` LogNormal(log 9, 0.75), `sigma_lb` HalfNormal(0.08). Stage-3 coverage then
  **PASSES**, conditional on takeoff (peak ≥ 10, ~77% take off): all four summaries centred
  (peak-week median 19 ∈ real IQR 17–22; log-growth median ~0.19; peak/total medians
  ~200/~1400 vs real ~198/~1700). The prior-predictive timing is deliberately a touch more
  diffuse than reality — safe over-coverage for a training prior.
- **Pure-SBI trajectory estimator (stages 4–5).** Target = `[θ (7) + log(weekly infections+1)
  ×39]` (46-dim); condition = 2-channel `[log1p-standardized cases, observed-week mask]`
  padded to 39 weeks, random cutoff `L` per batch. `TimeSeriesNetwork` (bidir GRU, summary_dim
  48) → `CouplingFlow`. 80 epochs × 250 batches (batch 128) on CPU, loss 65 → ~ −70 (negative
  NLL = density > 1, healthy). `particle_filter.py` **deleted**.
- **Serialization bug caught by running.** The old `.weights.h5` round-trip could not rebuild
  the coupling stack's layer paths across processes ("ActNorm expected 2 variables, received
  0"). Switched to full-model `keras.saving.save_model`/`load_model` (`.keras`), which
  preserves the whole approximator **including its fitted adapter** — the loaded object samples
  straight from raw `{"cases","mask"}` conditions. Second gotcha: `load_model` must run **after
  `import bayesflow`** so the `@register_keras_serializable` classes exist (else "Could not
  locate class ContinuousApproximator"). Both now handled in `train.load_trained`.
- **Recovery + SBC (stages 6–7).** Globals stay **poorly-identified but calibrated** (calib err
  0.006–0.042, mean z ≈ 0; contraction ≈ 0 for `R0`/`S0`/`delay_mean`) — the same honest
  confounding lesson, now under pure SBI. **The key new check — per-week SBC on the latent
  infection trajectory — passes**: worst-week calibration error **0.049** (all ≲ 0.05), with
  contraction correctly high in-sample (0.81 for weeks ≤ L) and lower for the forecast weeks
  (0.53), i.e. tight-where-pinned, wide-where-extrapolated. This is the headline result: the
  joint `[θ, trajectory]` posterior is calibrated at every week and every cutoff, with **no
  particle filter** — and the trajectory is *better* identified than the individual globals,
  because the cases directly constrain (reporting-scaled) infections while the split into
  ρ/S0/R0 is confounded.
- **Reliability / OOD (stage 8).** Real 2018/19 data still flagged **OOD** (aggregate MMD
  p ≈ 0.000 at cutoffs L = 4/8/12/20; per-state 18/15/5/10 of 51 flagged), consistently the
  large southern states (Florida, Texas, Georgia, Alabama). Same structural finding as before —
  a smooth single-wave SIR cannot match every real-flu shape — reproduced under the new model
  and carried as an explicit caveat. Most states are in-distribution at mid-season (only 5/51
  flag at L = 12).
- **Posterior predictive (stage 9).** In-sample PPC on 20 observed weeks: median roughness
  p-value **0.206**, **20%** of 44 states significantly jaggier than the model (featured:
  Illinois 0.22, Vermont 0.077, California 0.044). A mild but systematic **"model slightly too
  smooth"** misfit — the inferred infection trajectory + reporting noise under-produces the
  full-season week-to-week jaggedness, coherently reinforcing the stage-8 shape finding. (The
  first build's 16-week growth-phase window looked smoother/cleaner here; the full season with
  its noisy spring tail exposes more.)
- **Forecasts (stage 10).** Per state, per monthly cutoff, in-sample fit and forecast bands are
  the reporting pushforward of the one posterior trajectory — **no forward-simulation step,
  no filter**. **Out-of-sample 90% forecast-interval coverage = 93%** across **459 fits**
  (51 states × 9 cutoffs); per horizon +1/+2/+3/+4 wk = 97/93/91/91%. Well calibrated, in fact
  slightly conservative, and — unlike the PF build — **no short-horizon under-coverage** (the
  trajectory posterior already carries the reporting-censoring uncertainty at the cutoff). Holds
  even for stage-8-flagged states, since the estimator conditions on their real observed weeks.

---

## 5. Compute reality on this machine, and how it would scale on GPUs / runpod

Everything above ran **CPU-only** (16 cores, no GPU). What that cost and where it hurt,
then how a GPU fleet (e.g. runpod) would change the workflow — the raw material for a
future "scaling SBI training" skill.

**Where the time actually went (CPU).**
- *Training* was ~15 min for the full pure-SBI run (80 epochs × 250 batches, batch 128),
  ~35 ms/step. Note a **side benefit of the fixed-length padded design**: because every
  series is padded to `T_MAX = 39` (the observed length varies only via the mask channel,
  not the tensor shape), JAX compiles the step graph **once** — no per-window-length
  recompilation tax, unlike the first build's random `T ∈ 4..16` (~13 compiles/epoch-1).
  (The first build's ~20 min for 60 epochs was dominated by that recompilation + the online
  simulator.)
- *Reliability* (stage 8) was the slowest diagnostic (~8–10 min) because the simulator-
  calibrated null needs hundreds of fresh `summarize` passes; *recovery/SBC* and the
  *all-states forecast* (459 amortised fits × 400 draws) were a few minutes each.
- **Applying the trained estimator is cheap and pure amortised inference** — each
  `(state, cutoff)` fit is one batched `sample` call (~0.3–0.5 s) plus a NumPy reporting
  pushforward; no per-fit training or filtering. (The old build's particle filter is gone —
  see Iteration 7.)

**What would change with GPU / runpod, in priority order.**
1. **Amortised training is the one job worth a GPU.** The bottleneck here was the *online
   simulator feeding the GPU-less trainer*; on a GPU the network step is ~free and the
   simulator becomes the limit. Two fixes: (a) **vectorise the simulator on-device** — the
   chain-binomial loop is `jax.lax.scan`-friendly, moving simulation onto the same GPU and
   removing the host→device transfer; (b) **offline/streamed datasets** — pre-simulate a
   large bank on CPU workers while the GPU trains (BayesFlow supports offline datasets).
   Expect the ~20 min run to drop to low single-digit minutes.
2. **Kill the recompilation tax properly.** Rather than a random T per batch, **bucket by T
   or pad+mask to a fixed length** so JAX compiles once. On CPU the cache hid this after
   epoch 1; at fleet scale (many short jobs) it matters, because each fresh process re-pays
   epoch-1 compiles. A padded fixed-length summary net trades a little wasted compute for
   one compilation.
3. **Experiment throughput, not single-run latency, is the real win.** The expensive part
   of *building* this example was the **iteration loop** (prior→train→recover→SBC→OOD, then
   loop back and retrain). On runpod that maps to **one GPU pod per hyperparameter/prior
   variant, run in parallel**: e.g. the v2→v3 `sigma_lb` change would have been one of a
   sweep of {0.05, 0.10, 0.15, +obs-overdispersion} variants trained concurrently and
   compared on the stage-8 MMD, instead of a sequential guess-and-retrain. Dispatch = a
   small script that launches N pods from one container image (simulator + train.py +
   the diagnostic suite), each writing weights + a metrics JSON to a shared volume; a
   local reducer picks the winner. The MCP `runpod` tools (`create-pod`, `create-endpoint`,
   network volumes) are the mechanism.
4. **What does *not* need a GPU:** the data pipeline, the prior-predictive check, the
   reporting pushforward, and applying the trained estimator to real states (amortised
   inference is cheap). Keep those on CPU; only the training + the OOD-null simulation are
   worth accelerating, and the OOD null is embarrassingly parallel across CPU workers.

**One concrete gotcha for a cloud skill:** set `KERAS_BACKEND` *before* any Keras import,
pin the backend wheel to the pod's CUDA, and cache the JAX compilation dir on the shared
volume so restarted pods don't re-pay epoch-1 compiles. The archived-CRAN `cdcfluview`
install is also a reproducibility trap — bake the raw data CSV into the image, never fetch
the live CDC endpoint from a headless pod.
