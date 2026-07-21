# Benchmark Matrix

Tracks example models and their validation coverage.

Early-stage scope: only the **toy reference model** is active. The realistic
models below are planned and intentionally deferred until the toy model and the
core workflow are proven end-to-end. Do not add them without maintainer approval
(see root `AGENTS.md` — adding a scientific example requires human sign-off).

## Active benchmark

### Toy reference model — Normal(mu, sigma), variable N

Implemented in `examples/toy-normal/` (infer mean & sd of a Normal from 5-20
i.i.d. observations; DeepSet summary over the exchangeable set).

Purpose:

- Fast CI smoke test
- Known or easily checked behaviour (backend-free grid reference posterior)
- Validates basic workflow structure

Required checks:

- [x] Prior predictive simulation — `simulator.py`, verified (tests pass)
- [x] Training smoke test — `train.py --smoke` runs (BayesFlow 2.0.12, jax backend)
- [x] Posterior sampling smoke test — `train.py` samples the posterior
- [x] Parameter recovery — `run_validation.py` (40 epochs), via `bayesflow.diagnostics`: mu corr ~0.92, sigma corr ~0.87 (representative); means track the exact grid reference
- [x] Calibration (SBC) — ECDF-based calibration error small for both (mu ~0.04, sigma ~0.07); `.constrain("sigma", lower=0)` flattens sigma's rank histogram (fixes the gross boundary bias), leaving only a mild residual positive bias (sigma mean z ≈ +0.15); mu unbiased (mean z ≈ −0.03)
- [x] Sensitivity — posterior z-score vs. contraction (`bayesflow.diagnostics.z_score_contraction`); both parameters well-identified (contraction ~0.9) under these priors
- [x] Reliability / out-of-distribution check — `reliability.py`: MMD in summary space (BayesFlow `maximum_mean_discrepancy` + `approximator.summarize`) vs. a prior-predictive reference, with a **simulator-calibrated** null (fresh in-distribution draws) + `mmd_hypothesis_test` plot. In-distribution sample is typical (MMD ~0.05, p ~0.6); an OOD sample (mean far outside the prior) is flagged decisively (MMD ~2.6, p = 0), a ~50× separation
- [x] Posterior predictive report — `posterior_predictive.py`, all diagnostics via **ArviZ** (`plot_ppc_dist` / `plot_ppc_tstat` / `plot_ppc_pit`); replicates re-simulated from the model's own `likelihood`. Well-specified passes (p-values all moderate); `--misspecify` (skewed observation) is caught by the skewness statistic (p ~0.009) while the fitted moments stay quiet — confirming the check has teeth

### Realistic benchmark — Influenza SIR forecasting (real CDC data)

Implemented in `examples/flu-sir/`. Discrete-time stochastic (chain-binomial) SIR with
time-varying transmission and imperfect/lagged reporting; one amortised estimator
(`TimeSeriesNetwork` summary + `CouplingFlow`) reused across all US states and forecast
dates. **Latent epidemic states are inferred as targets of the same amortised posterior**
(joint `[θ + weekly-infection trajectory]`) — pure SBI, no particle filter. Final fit on the
full 2018/2019 CDC season (39 weeks, CDC FluView `who_nrevss("state")`), priors/checks from
pre-2018 seasons only.

Purpose: exercise the **full workflow loop on real data**, including the loop-backs the toy
cannot show (a stage-3 prior revision and a stage-8 OOD finding), and BayesFlow's machinery
for **hierarchical / latent-state / time-series** amortised SBI.

Required checks:

- [x] Prior predictive (stage 3) — `prior_predictive.py`: covers/centres the real full-season
  curves after a growth + autumn-seed re-tune (old priors peaked far too early from an October
  start). Backend-free; 8/8 tests pass.
- [x] Training + pilot (stages 4-5) — `train.py`; full model (`.keras`) persisted for reuse.
- [x] Recovery + SBC (stages 6-7) — `run_validation.py` via `bayesflow.diagnostics`: globals
  **calibrated but poorly-identified** (calib err 0.006-0.042), and the **latent trajectory is
  SBC-calibrated per week** (worst 0.049), tight in-sample / wide in forecast.
- [x] Reliability / OOD (stage 8) — `reliability.py`: real data flagged OOD (shape-distribution
  mismatch), per-state 18/15/5/10 of 51 at cutoffs L=4/8/12/20; structural single-wave gap
  carried as a caveat.
- [x] Posterior predictive (stage 9) — `posterior_predictive.py` via ArviZ: median roughness
  p 0.206, 20% of states jaggier than the model — a mild "too smooth" misfit reinforcing stage 8.
- [x] Forecasts (stage 10) — `forecast.py`: **93% out-of-sample 90%-interval coverage** across
  459 state×cutoff fits; low in-sample, growing forecast uncertainty, no short-horizon under-coverage.

## Planned benchmarks (deferred)

Kept as a roadmap only — not yet in scope.

- **Spatial disease transmission** — spatial outputs, aggregation, local transmission structure.
- **Evolutionary process** — non-epidemic mechanistic simulation (tree / sequence / allele-frequency outputs).
- **Spatio-temporal invasion** — high-dimensional spatio-temporal outputs, propagating-front diagnostics, profiling.

## Status table

| Benchmark | Status | Owner | Last checked | Notes |
|---|---|---:|---:|---|
| Toy reference model | Inference verified | TBD | TBD | 8/8 tests pass; recovers params (mu corr ~0.92, sigma ~0.87), matches exact grid posterior, mu & sigma calibrated via `bayesflow.diagnostics` (sigma needs constrain lower=0); reliability/OOD check via summary-space MMD with a simulator-calibrated null (in-dist typical p~0.6, OOD flagged ~50x); posterior predictive report via ArviZ (passes; skew misfit caught) |
| Influenza SIR forecasting | Full workflow verified | TBD | 2018-07 sim date | `examples/flu-sir/`; real CDC 2018/19 full season (39 wk); one amortised **pure-SBI** estimator (globals + latent infection trajectory as joint target, no particle filter) across 51 states; globals calibrated-but-poorly-identified, latent trajectory SBC-calibrated per week; stage-3 + stage-8 findings; 93% out-of-sample forecast coverage; residual OOD on large states carried as caveat |
| Spatial disease transmission | Deferred | TBD | — | Planned; requires human validation |
| Evolutionary process | Deferred | TBD | — | Planned; requires human validation |
| Spatio-temporal invasion | Deferred | TBD | — | Planned; likely expensive |
