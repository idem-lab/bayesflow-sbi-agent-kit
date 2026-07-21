# Example: influenza forecasting with a stochastic SIR + amortised SBI

**This is the first *realistic* worked example** — a full run of the 10-stage Bayesian SBI
workflow on real data, end to end, with every stage's artefact present. Unlike the
toy-normal smoke test, the science here is non-trivial and the outcome is honestly mixed:
the workflow **catches a real model limitation** (stage 8) and forces it to be an explicit,
recorded decision rather than a silently overconfident forecast. Read
[`ENGINEERING_LOG.md`](ENGINEERING_LOG.md) for the blow-by-blow (every dead-end, every
"caught by running", every science call).

The task: forecast weekly influenza cases (positive specimens) in US states across the
**full 2018/2019 CDC season** (~1 Oct → ~30 Jun, 39 weeks), as a late-2018 analyst who may
use only earlier seasons for priors and checks. **One** amortised estimator is trained once
on the simulator and reused for every state and every monthly forecast date.

## The model

A discrete-time, discrete-state **stochastic SIR** in a single homogeneous population,
with time-varying transmission and imperfect, lagged reporting:

```
log β_t ~ mean-reverting AR(1)/OU around log(R0·γ)   # autocorrelated transmission
new infections_t ~ Binomial(S_t, 1 - exp(-β_t · I_t/N))   # daily chain-binomial (demographic stochasticity)
recoveries_t     ~ Binomial(I_t, 1 - exp(-γ))
reported cases_w ~ thin weekly infections by ρ, then a geometric reporting delay   # under-reporting + lag + right-censoring
```

Seven inferred **global** parameters — `R0, γ, σ_logβ, ρ, delay_mean, S0, I0` — several
**deliberately weakly identified** (`ρ`↔`S0` scale confound; `σ_logβ`, `delay_mean`
nuisances) — plus the **latent weekly-infection trajectory** as a joint inference target.
Priors are literature-anchored and validated/adjusted by the stage-3 prior-predictive check;
see [`prior-specification.md`](prior-specification.md).

## The key design decision: latent states are inference targets (pure SBI)

The latent epidemic states are inferred by the **same amortised posterior** as the global
parameters — no particle filter, no MCMC. The joint inference target is

    [ θ (7 globals) , log(weekly infections + 1) for all 39 weeks ]   # 46-dim

conditioned on the observed weekly case series truncated at a monthly cutoff `L`, padded to
39 weeks with a **mask channel** (so one estimator serves every forecast date). A 2-channel
`TimeSeriesNetwork` (LSTNet) summary → `CouplingFlow` gives full posterior samples of the
whole trajectory. Everything else is a deterministic pushforward of that one posterior:

- **in-sample fit** (weeks ≤ L) = reporting model applied to the posterior infections —
  data-pinned, tight;
- **forecast** (weeks > L) = reporting model applied to the posterior *future* infections,
  which the flow learned to extrapolate — uncertainty grows with horizon.

This answers the brief's "one simulation, or a separate forward step?" — **one simulation of
the whole season; the future weeks are part of the inference target.** An earlier build
used a hybrid amortised-globals + particle-filter factorisation; it was replaced with pure
SBI because the point of this example is to exercise BayesFlow's own machinery for
latent-state / hierarchical inference. See `ENGINEERING_LOG.md` §3 and `SBI_REDESIGN_PLAN.md`.

## Files

| File | Needs | Purpose |
|---|---|---|
| `data_flu.py` | pandas | Load the cached CDC FluView clinical-labs CSV; align each (state, season) to the 39-week full-season window; expose the 2018/19 target, the monthly cutoffs, and the pre-2018 library. |
| `simulator.py` | NumPy | The model: vectorised `sample_prior`, `simulate` (chain-binomial SIR + reporting, exposes the latent infection trajectory), `observed_and_mask`, and a lazy BayesFlow `make_simulator`. Backend-free. |
| `prior_predictive.py` | NumPy | **Stage 3** — prior-predictive plausibility + coverage vs the real pre-2018 curves. |
| `train.py` | BayesFlow + backend | **Stages 4-5** — build/train/persist the amortised trajectory estimator; `--smoke` pilot. |
| `run_validation.py` | BayesFlow + backend | **Stages 6-7** — recovery + SBC via `bayesflow.diagnostics` on the globals **and** the latent trajectory (per-week SBC), across cutoffs. |
| `reliability.py` | BayesFlow + backend | **Stage 8** — reliability/OOD on the real data: summary-space MMD (simulator-calibrated null) + per-state flagging. |
| `posterior_predictive.py` | BayesFlow + backend + ArviZ | **Stage 9** — posterior-predictive check targeting epidemic *shape* features; ArviZ. |
| `forecast.py` | BayesFlow + backend | The final analysis: per-state monthly forecasts + figures + out-of-sample calibration. |
| `prior-specification.md` | — | Stage-2 priors, triage, and committed acceptance targets. |
| `ENGINEERING_LOG.md` | — | The process: decisions, experiments, bugs-caught-by-running, GPU/runpod notes. |
| `data/` | — | Cached raw CDC pull (so nothing depends on the live endpoint). |

The pure-NumPy split mirrors `toy-normal/`: model, data loader, and the stage-3 check need
no backend; only training/inference-network stages do. Diagnostics come from
`bayesflow.diagnostics` and ArviZ — none hand-rolled. The latent states are inferred by
BayesFlow itself, not a separate sampler.

## What this covers in the workflow

The full loop on real data: prior elicitation with committed targets (stage 2), a
prior-predictive check that **caught an epidemic-timing misspecification and drove a prior
change** (stage 3), estimator build + pilot (stages 4-5), recovery + SBC showing the
**"poorly-identified globals but calibrated"** regime *and* a **calibrated latent
trajectory** (per-week SBC, stages 6-7), a reliability check that **flags the real data as
OOD** (stage 8), a posterior-predictive check tying a residual shape misfit back to stage 8
(stage 9), and the final per-state monthly forecasts across the whole season with an
out-of-sample calibration summary (stage 10). The loop-backs — including the maintainer
loop-back that replaced a hybrid particle-filter build with pure SBI — are the point.

## Environment

Same `uv`-managed venv as the repo. Getting the data needs R with `cdcfluview`
(archived from CRAN — install `cdcfluview_0.9.4` from the CRAN Archive + `MMWRweek`); the
raw pull is already cached under `data/`, so re-running R is optional.

```bash
uv pip install --python .venv/bin/python -r examples/flu-sir/requirements.txt
```

## Running it

```bash
# backend-free
.venv/bin/python tests/test_flu_sir.py
.venv/bin/python examples/flu-sir/data_flu.py
.venv/bin/python examples/flu-sir/simulator.py
.venv/bin/python examples/flu-sir/prior_predictive.py
# backend (KERAS_BACKEND=jax)
KERAS_BACKEND=jax .venv/bin/python examples/flu-sir/train.py --smoke     # pilot
KERAS_BACKEND=jax .venv/bin/python examples/flu-sir/train.py             # full train + save
KERAS_BACKEND=jax .venv/bin/python examples/flu-sir/run_validation.py    # recovery + SBC
KERAS_BACKEND=jax .venv/bin/python examples/flu-sir/reliability.py       # stage-8 OOD check
KERAS_BACKEND=jax .venv/bin/python examples/flu-sir/posterior_predictive.py
KERAS_BACKEND=jax .venv/bin/python examples/flu-sir/forecast.py [--all]  # forecasts + figures
```

## Status

Verified end-to-end on Python 3.12 (BayesFlow 2.0.12, Keras 3, JAX backend, **CPU-only**).
Representative numbers (run-to-run variation expected); see `ENGINEERING_LOG.md` for context.

- **Data** — `cdcfluview::who_nrevss("state")`, clinical-labs positives, cached under
  `data/`. 51 states+DC; 138 pre-2018 epidemic windows; 2018/19 target = **39 weeks from 1
  Oct** (the full CDC season); monthly forecast cutoffs `L ∈ {4,…,36}`.
- **Stage 3 (prior predictive)** — from an October start the old priors peaked far too
  early; re-tuning growth and shrinking the autumn seed (`R0`~1.26/0.08, `I0`~9/0.75,
  `σ_logβ`~0.08) makes the pushforward **cover and centre** the real full-season curves,
  conditional on takeoff — peak-week median 19 (real IQR 17-24), log-growth ~0.19, peak/total
  medians ~200/~1400 (real ~198/~1700). 8/8 backend-free tests pass.
- **Stages 6-7 (recovery + SBC)** — via `bayesflow.diagnostics`. The **globals** are
  *poorly-identified but calibrated* (calibration error 0.006-0.042, near-zero bias;
  contraction ≈0 for `R0`/`S0`/`delay_mean`) — the `ρ`↔`S0` confound made explicit. The
  **latent infection trajectory is SBC-calibrated per week** (worst-week calibration error
  0.049), with contraction correctly **high in-sample** (0.81 for weeks ≤ L) and **lower for
  forecast weeks** (0.53): tight where data pins it, wide where it extrapolates — the brief's
  requirement, delivered by pure SBI.
- **Stage 8 (reliability / OOD)** — the real 2018/19 data is flagged **OOD** (summary-space
  MMD p≈0 at cutoffs L=4/8/12/20; per-state 18/15/5/10 of 51 flagged). A *shape-distribution*
  mismatch: a smooth single-wave SIR structurally cannot match every real-flu shape (holiday
  dip, plateaus, secondary waves). Most states are in-distribution at mid-season (5/51 flag at
  L=12); the consistently-flagged large southern states (Florida, Texas, Georgia, Alabama) are
  carried as an explicit caveat.
- **Stage 9 (posterior predictive)** — epidemic shape features (roughness, peak, total,
  autocorrelation) via ArviZ on the in-sample trajectory pushforward. Median roughness
  p-value **0.206**, **20% of states** significantly jaggier than the model — a mild but
  systematic **"model slightly too smooth"** misfit that coherently **reinforces the stage-8
  finding** (the single-wave SIR under-produces full-season week-to-week jaggedness).
- **Stage 10 (forecasts)** — per-state monthly forecasts across the whole season (observe to
  cutoff `L`, forecast the next 4 weeks), bands derived from the one posterior trajectory.
  **Out-of-sample 90% forecast-interval coverage = 93%** across **459 fits** (51 states × 9
  cutoffs) — well calibrated, slightly conservative (per horizon: +1wk 97%, +2wk 93%, +3wk
  91%, +4wk 91%; **no short-horizon under-coverage**). Figures in `outputs/forecast_*.png`.
  In-sample uncertainty is low and forecast uncertainty grows with horizon. Forecasts stay
  calibrated **even for OOD states** — the estimator conditions on their real observed weeks.

**Honest bottom line.** For the in-distribution states (most of them — only ~10% flag at
mid-season) the pipeline is trustworthy and calibrated, with a per-week-calibrated latent
trajectory and 93% out-of-sample forecast coverage; for the flagged large/atypical states
the amortised posterior is at the edge of its validated regime and forecasts there carry a
structural-misspecification caveat. That nuance is the workflow working as intended — the
checks localised *where* to trust the model — and it is achieved entirely within BayesFlow's
amortised SBI, no bolted-on sampler.
