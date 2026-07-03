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
- [x] Parameter recovery — `run_validation.py` (40 epochs): mu corr 0.96, sigma corr 0.94; means track the exact grid reference
- [x] Calibration (SBC) — mu and sigma ranks ~uniform (sigma unbiased after `.constrain("sigma", lower=0)`); mild parameter-agnostic underdispersion remains
- [ ] Posterior predictive report — not yet implemented (recovery/SBC/grid-crosscheck done instead)

## Planned benchmarks (deferred)

Kept as a roadmap only — not yet in scope.

- **Infectious disease time-series** — partially observed temporal dynamics, observation error, identifiability.
- **Spatial disease transmission** — spatial outputs, aggregation, local transmission structure.
- **Evolutionary process** — non-epidemic mechanistic simulation (tree / sequence / allele-frequency outputs).
- **Spatio-temporal invasion** — high-dimensional spatio-temporal outputs, propagating-front diagnostics, profiling.

## Status table

| Benchmark | Status | Owner | Last checked | Notes |
|---|---|---:|---:|---|
| Toy reference model | Inference verified | TBD | TBD | 7/7 tests pass; recovers params (mu corr 0.96, sigma 0.94), matches exact grid posterior, mu & sigma SBC unbiased (sigma needs constrain lower=0) |
| Infectious disease time series | Deferred | TBD | — | Planned; first realistic example |
| Spatial disease transmission | Deferred | TBD | — | Planned; requires human validation |
| Evolutionary process | Deferred | TBD | — | Planned; requires human validation |
| Spatio-temporal invasion | Deferred | TBD | — | Planned; likely expensive |
