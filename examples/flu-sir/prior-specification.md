# Prior specification — influenza SIR forecasting

_Stage 2 deliverable. In a real engagement **the human approves every row**; this
example was built autonomously, so rows are marked **[agent-as-analyst]** and every
scientific call is a place a reviewer should push back. Targets are committed **before**
the pushforward (stage 3) so the check stays honest._

The observed data are weekly **positive influenza specimens** per US state
(CDC FluView clinical labs). The analyst is situated in **late 2018** and may use only
seasons **before** 2018/19 (2015/16–2017/18) — the source of both the priors' context
and the committed targets below.

---

## Parameter triage

| Parameter | Category | Why |
|---|---|---|
| `R0` | (a) direct | Seasonal-influenza R0 is well studied (meta-analysis below). Caveat: in a single homogeneous SIR fit to smoothed surveillance data, R0 is entangled with the effective generation interval (next row). |
| `gamma` (1/γ = infectious/generation period) | (b) rough idea | Clinical infectious period is ~2–4 d, but the **single-compartment SIR generation interval that reproduces observed epidemic growth is longer** (see iteration log). Elicited as a period, reasoned partly on the data (growth rate). |
| `sigma_lb` (log-β volatility) | (c) can't reason directly | A process-noise scale with no standalone meaning; reasoned only through how wiggly β(t) — hence the case curve — should be. |
| `rho` (reported fraction) | (c) jointly | Only a small fraction of infections become clinical-lab positives, but the fraction is not directly knowable and is **confounded with `S0`** through the case scale ρ·(attack rate)·S0. Reasoned on the data scale. |
| `delay_mean` (reporting delay, weeks) | (b) rough idea | Infection→symptoms→test→report is days-to-a-couple-of-weeks. |
| `S0` (initial susceptible / effective pool) | (c) jointly | Not the census population — the effective transmission pool sampled by clinical labs. Spans ~2 orders of magnitude across states; identified jointly with `rho`. |
| `I0` (initial true infections at ~1 Oct) | (c) jointly | A small autumn **seed** at the season start; a later, larger seed peaks earlier, so `I0` also controls epidemic *timing* over the full season. |

Four of seven parameters are (c) or weakly identified — **this is deliberate**, so the
example exercises the poorly-identified branch of recovery/SBC (stage 6/7).

---

## Priors

LogNormal(μ, σ) below is written so `exp(μ)` is the **median**. All parameters are
positive; `rho ∈ (0,1)`. Constraints are enforced in the adapter (`train.py`).

| Parameter | Support | Elicited plausible range | Chosen prior | Evidence | Approved |
|---|---|---|---|---|---|
| `R0` | >0 | ~1.1–1.5 | LogNormal(log 1.26, 0.08) **[v4]** | Biggerstaff et al. 2014 (BMC Infect Dis) seasonal-flu R median ≈1.28, IQR ≈1.19–1.37. Tightened at v4 so the infection growth rate `≈7·γ(R0−1)` (~0.25/wk) matches the observed full-season build-up, not a too-fast one | [agent-as-analyst] |
| `gamma` | >0 | period 3–9 d (γ 0.11–0.33 /d) | LogNormal(log(1/6), 0.35) **[v2]** | v1 used ~3 d (clinical); widened/lengthened after the stage-3 growth-timing failure | [agent-as-analyst] |
| `sigma_lb` | ≥0 | daily log-β sd ~0.02–0.20 | HalfNormal(0.08) **[v4]** | process-noise scale giving β(t) enough variation for plateaus/dips/secondary waves without dominating timing | [agent-as-analyst] |
| `rho` | (0,1) | ~0.002–0.05 | LogNormal(log 0.01, 0.9) | order-of-magnitude: clinical-lab positives are a small fraction of infections | [agent-as-analyst] |
| `delay_mean` | ≥0 | ~0.2–2 wk | LogNormal(log 0.7, 0.5) | specimen collection + reporting lag | [agent-as-analyst] |
| `S0` | >0 | ~3e4–3e6 | LogNormal(log 3e5, 1.35) | set so ρ·attack·S0 spans the observed case scale across states | [agent-as-analyst] |
| `I0` | >0 | ~2–40 | LogNormal(log 9, 0.75) **[v4]** | small **1-Oct seed**: a tiny autumn seed makes the epidemic build slowly and peak in Dec–Feb (weeks 12–24), matching real full-season timing; the old larger `I0` (median 50) peaked far too early once the window started in October | [agent-as-analyst] |

Fixed structural constants (not inferred, design choices): `STEPS_PER_WEEK=7`,
`KAPPA=0.03`/day (log-β mean-reversion → ~0.8 weekly persistence), `DELAY_KMAX=6` weeks,
`T_MAX=39` weeks (the full CDC season, ~1 Oct → ~30 Jun).

---

## Target summary statistics (stage-3 acceptance criteria)

Measured on the **138 real pre-2018 (state, season) epidemic windows** (39 weeks each,
the full CDC season). Simulated prior-predictive data should **cover** these (compared
*conditional on takeoff*, peak ≥ 10, since a fraction of simulated and real state-seasons
fizzle) — the check is coverage, not a point match. Committed before the pushforward.

| Summary of the weekly case curve | Central (median) | Plausible range (5–95%) | Notes |
|---|---|---|---|
| Peak weekly cases | 198 | 22 – 1300 | magnitude; must span ~2 orders across states |
| Week of peak (0-indexed, of 39) | 19 | 12 – 24 | **timing — real curves peak in Dec–Mar (weeks 12–24)** |
| Total cases over the 39-week season | 1700 | 131 – 11741 | overall scale |
| Log growth rate, first half (per week) | 0.23 | 0.04 – 0.35 | **doubling ≈ 3 weeks** |

---

## Method notes

- **How priors were set:** manual elicitation, literature-anchored, then adjusted at
  stage 3 — the `gamma`/generation-interval widening (v2), and the v4 re-tune of `R0`,
  `sigma_lb` and especially `I0` when the window was widened to the full Oct–Jun season
  (the old priors peaked far too early from an October start). Logged in ENGINEERING_LOG.md.
- **Fit checked:** LogNormal medians/ranges verified by `sample_prior` self-check.
- **Coverage check (v4, full season):** all four targets covered and centred conditional
  on takeoff (peak_week median 19 ∈ real IQR 17–22; log_growth median ~0.19; peak/total
  medians ~200/~1400 vs real ~198/~1700). The prior-predictive timing is a little more
  *diffuse* than reality (safe for a training prior — over-coverage beats OOD). See
  iteration log.
- **Parameters flagged as likely poorly identified (forward-link stage 6):**
  `rho`, `S0` (confounded scale), `sigma_lb`, `delay_mean`.
