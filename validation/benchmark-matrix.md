# Benchmark Matrix

Tracks example models and their validation coverage.

Early-stage scope: only the **toy reference model** is active. The realistic
models below are planned and intentionally deferred until the toy model and the
core workflow are proven end-to-end. Do not add them without maintainer approval
(see root `AGENTS.md` — adding a scientific example requires human sign-off).

## Active benchmark

### Toy reference model

Purpose:

- Fast CI smoke test
- Known or easily checked behaviour
- Validates basic workflow structure

Required checks:

- [ ] Prior predictive simulation
- [ ] Training smoke test
- [ ] Posterior sampling smoke test
- [ ] Parameter recovery structure
- [ ] Posterior predictive report

## Planned benchmarks (deferred)

Kept as a roadmap only — not yet in scope.

- **Infectious disease time-series** — partially observed temporal dynamics, observation error, identifiability.
- **Spatial disease transmission** — spatial outputs, aggregation, local transmission structure.
- **Evolutionary process** — non-epidemic mechanistic simulation (tree / sequence / allele-frequency outputs).
- **Spatio-temporal invasion** — high-dimensional spatio-temporal outputs, propagating-front diagnostics, profiling.

## Status table

| Benchmark | Status | Owner | Last checked | Notes |
|---|---|---:|---:|---|
| Toy reference model | Not started | TBD | TBD | Fast CI target; the only active benchmark |
| Infectious disease time series | Deferred | TBD | — | Planned; first realistic example |
| Spatial disease transmission | Deferred | TBD | — | Planned; requires human validation |
| Evolutionary process | Deferred | TBD | — | Planned; requires human validation |
| Spatio-temporal invasion | Deferred | TBD | — | Planned; likely expensive |
