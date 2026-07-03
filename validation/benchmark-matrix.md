
# Benchmark Matrix

Use this file to track example models and validation coverage.

## Benchmark categories

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

---

### Infectious disease time-series model

Purpose:

- Tests partially observed temporal dynamics
- Tests observation error
- Tests identifiability issues

Required checks:

- [ ] Prior predictive checks
- [ ] Time-series summary/embedding design
- [ ] Parameter recovery
- [ ] Simulation-based calibration
- [ ] Posterior predictive checks
- [ ] Human validation of model interpretation

---

### Spatial disease transmission model

Purpose:

- Tests spatial outputs
- Tests spatial aggregation
- Tests local transmission structure

Required checks:

- [ ] Spatial output adapter
- [ ] Prior predictive spatial maps
- [ ] Parameter recovery
- [ ] Posterior predictive spatial diagnostics
- [ ] Human validation of spatial interpretation

---

### Evolutionary process model

Purpose:

- Tests non-epidemic mechanistic simulation
- Tests tree, sequence, allele-frequency, or summary-statistic outputs

Required checks:

- [ ] Simulator interface audit
- [ ] Domain-approved priors
- [ ] Prior predictive checks
- [ ] Parameter recovery
- [ ] Calibration diagnostics
- [ ] Human validation of interpretation

---

### Spatio-temporal invasion model

Purpose:

- Tests high-dimensional spatio-temporal outputs
- Tests propagating-front diagnostics
- Tests computational experiment design

Required checks:

- [ ] Spatio-temporal adapter
- [ ] Summary-network comparison
- [ ] Prior predictive checks
- [ ] Parameter recovery
- [ ] Simulation-based calibration
- [ ] Posterior predictive checks
- [ ] Runtime and memory profiling

## Benchmark status table

| Benchmark | Status | Owner | Last checked | Notes |
|---|---|---:|---:|---|
| Toy reference model | Not started | TBD | TBD | Fast CI target |
| Infectious disease time series | Not started | TBD | TBD | First realistic example |
| Spatial disease transmission | Not started | TBD | TBD | Requires human validation |
| Evolutionary process | Not started | TBD | TBD | Requires human validation |
| Spatio-temporal invasion | Not started | TBD | TBD | Likely expensive |

