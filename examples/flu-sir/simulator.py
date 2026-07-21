"""Discrete-time, discrete-state stochastic SIR simulator with imperfect, lagged reporting.

Mechanism (single homogeneous population; see ENGINEERING_LOG.md §3-4):

- **Daily chain-binomial SIR.** Time advances in daily steps (``STEPS_PER_WEEK`` per
  week). Each day, new infections ~ Binomial(S, 1-exp(-β_t · I/N)) and recoveries ~
  Binomial(I, 1-exp(-γ)). The binomial draws are the *demographic stochasticity*: the
  same parameters give different integer trajectories every run, and the discrete-time
  chain-binomial converges to the continuous-time SIR ODE as the step shrinks.
- **Time-varying transmission.** log β_t follows a mean-reverting (Ornstein-Uhlenbeck /
  AR(1)) process around log(R0·γ) with daily innovation sd ``σ_logβ`` and a fixed daily
  reversion ``KAPPA``. This is the "autocorrelated temporal process" for β.
- **Imperfect, lagged reporting.** Weekly true new infections are thinned by a reported
  fraction ``ρ`` (Binomial), then each reported case is delayed by a geometric number of
  weeks with mean ``delay_mean``. Reports landing past the observation window are
  right-censored (realistic real-time under-counting of the most recent weeks).

Parameters inferred (θ, 7 of them). Several are deliberately only weakly identified —
notably ``ρ`` and ``S0`` are confounded through the observed case *scale* (ρ·attack·S0),
and ``σ_logβ`` / ``delay_mean`` are informed only coarsely by weekly counts. That is the
identifiability lesson of this example, surfaced later by recovery + SBC.

    R0         basic reproduction number         LogNormal   (flu ~1.2-1.4)
    gamma      daily recovery rate  (1/gamma d)   LogNormal   (infectious period ~2-4 d)
    sigma_lb   daily log-beta innovation sd       HalfNormal  (transmission volatility)
    rho        reported fraction of infections    LogNormal   (small; clinical-lab positives)
    delay_mean mean reporting delay (weeks)        LogNormal
    S0         initial susceptibles (eff. pool)   LogNormal   (wide; spans state sizes)
    I0         initial infected at window start    LogNormal

This module is pure NumPy and backend-free (imported by tests, checks, and the
prior-predictive without a deep-learning backend). The only BayesFlow dependency is
confined to ``make_simulator`` / the ``SIRSimulator`` class, imported lazily.

Prior hyperparameters are literature-anchored *starting points* (citations in
prior-specification.md); the prior-predictive check (stage 3) is what validates them
against real pre-2018 epidemic curves.

**Pure-SBI trajectory design (see ENGINEERING_LOG.md §3).** The latent epidemic states are
inference *targets*, not something a separate sampler reconstructs. Each simulation runs
the full ``T_MAX``-week season and exposes the **weekly true new-infection trajectory**
``log(weekly_inf + 1)`` (T_MAX values) alongside the 7 globals — the joint inference target.
The *condition* is the observed weekly case series truncated at a random monthly cutoff
``L`` (padded to T_MAX with a mask channel), so one amortised estimator serves every
forecast date. In-sample fit (weeks ≤ L) and forecast (weeks > L) both fall out of the one
posterior over the trajectory via the reporting model — no particle filter, no MCMC.

Real-time censoring is automatic: because a source-week-``w`` report at delay ``k`` lands in
column ``w+k`` (kept iff ``w+k < W``), truncating a full-season report at column ``L`` equals
re-censoring the reporting process at cutoff ``L`` — i.e. exactly what an analyst sees at
week ``L``, with the most recent weeks still under-reported.
"""

from __future__ import annotations

import numpy as np

# --- Fixed structural constants (design choices, not inferred) ---------------
STEPS_PER_WEEK = 7          # daily time step
KAPPA = 0.03               # daily OU reversion of log-beta (weekly persistence ~0.8)
DELAY_KMAX = 6             # truncate the reporting-delay tail at 6 weeks
T_MAX = 39                 # full CDC season length in weeks (~1 Oct → ~30 Jun)
PARAM_NAMES = ["R0", "gamma", "sigma_lb", "rho", "delay_mean", "S0", "I0"]

# --- Prior hyperparameters (log-scale mu/sigma unless noted) -----------------
# LogNormal(mu, sigma) is parameterised so exp(mu) is the median.
PRIORS = {
    "R0":         dict(kind="lognormal", mu=np.log(1.26), sigma=0.08),    # median 1.26, tight: growth ~7·γ(R0−1) ≈ 0.25/wk (v4, full-season)
    "gamma":      dict(kind="lognormal", mu=np.log(1 / 6.0), sigma=0.35),  # ~6-day SIR generation interval (see log)
    "sigma_lb":   dict(kind="halfnormal", scale=0.08),                    # daily log-beta volatility (transmission shape flexibility)
    "rho":        dict(kind="lognormal", mu=np.log(0.01), sigma=0.9),     # median 1%, ~[0.002,0.05]
    "delay_mean": dict(kind="lognormal", mu=np.log(0.7), sigma=0.5),      # median 0.7 wk
    "S0":         dict(kind="lognormal", mu=np.log(3e5), sigma=1.35),     # median 3e5, ~[3e4,3e6]; wide to cover largest states
    "I0":         dict(kind="lognormal", mu=np.log(9.0), sigma=0.75),     # median 9 seeds at 1 Oct → slow autumn onset, Dec–Feb peak (v4)
}


def sample_prior(batch: int, rng: np.random.Generator) -> dict:
    """Draw ``batch`` parameter sets from the joint prior. Returns a dict of (batch,)
    float arrays on the natural (constrained) scale."""
    out = {}
    for name, spec in PRIORS.items():
        if spec["kind"] == "lognormal":
            out[name] = np.exp(rng.normal(spec["mu"], spec["sigma"], size=batch))
        elif spec["kind"] == "halfnormal":
            out[name] = np.abs(rng.normal(0.0, spec["scale"], size=batch))
        else:  # pragma: no cover
            raise ValueError(spec["kind"])
    return {k: v.astype("float64") for k, v in out.items()}


def simulate(params: dict, n_weeks: int = T_MAX, rng: np.random.Generator = None,
             steps_per_week: int = STEPS_PER_WEEK,
             return_latent: bool = False) -> dict:
    """Vectorised stochastic SIR + reporting for a batch of parameter sets.

    ``params``: dict of (B,) arrays (natural scale). Returns ``{"cases": (B, n_weeks)}``
    of integer weekly reported cases; with ``return_latent`` also returns the weekly true
    new-infections trajectory (the SBI target) plus the daily latent S/I and beta paths.
    """
    if rng is None:
        rng = np.random.default_rng()
    B = len(params["R0"])
    D = n_weeks * steps_per_week
    R0 = params["R0"]; gamma = params["gamma"]; sigma_lb = params["sigma_lb"]
    rho = params["rho"]; delay_mean = params["delay_mean"]
    S0 = np.rint(params["S0"]).astype(np.int64)
    I0 = np.rint(params["I0"]).astype(np.int64)

    N = np.maximum(S0 + I0, 1)                      # population (S0 + I0), R starts at 0
    S = S0.copy(); I = np.minimum(I0, N).copy()
    log_beta_mean = np.log(np.maximum(R0 * gamma, 1e-6))
    log_beta = log_beta_mean.copy()                 # start at the mean level
    p_rec = 1.0 - np.exp(-gamma)                    # per-day recovery prob

    weekly_inf = np.zeros((B, n_weeks), dtype=np.int64)
    lat_S = np.empty((B, D + 1)) if return_latent else None
    lat_I = np.empty((B, D + 1)) if return_latent else None
    lat_beta = np.empty((B, D)) if return_latent else None
    if return_latent:
        lat_S[:, 0] = S; lat_I[:, 0] = I

    for d in range(D):
        # OU update of log-beta, then the day's transmission rate
        eps = rng.standard_normal(B)
        log_beta = log_beta + KAPPA * (log_beta_mean - log_beta) + sigma_lb * eps
        beta = np.exp(log_beta)
        p_inf = 1.0 - np.exp(-beta * I / N)         # force of infection (per susceptible)
        new_inf = rng.binomial(np.maximum(S, 0), np.clip(p_inf, 0, 1))
        new_rec = rng.binomial(np.maximum(I, 0), np.clip(p_rec, 0, 1))
        S = S - new_inf
        I = I + new_inf - new_rec
        weekly_inf[:, d // steps_per_week] += new_inf
        if return_latent:
            lat_S[:, d + 1] = S; lat_I[:, d + 1] = I; lat_beta[:, d] = beta

    cases = report(weekly_inf, rho, delay_mean, rng)

    out = {"cases": cases}
    if return_latent:
        out.update(weekly_inf=weekly_inf, lat_S=lat_S, lat_I=lat_I, lat_beta=lat_beta, N=N)
    return out


def report(weekly_inf: np.ndarray, rho: np.ndarray, delay_mean: np.ndarray,
           rng: np.random.Generator, kmax: int = DELAY_KMAX) -> np.ndarray:
    """Thin weekly true infections by ``rho`` (Binomial) and spread each reported case
    over future weeks by a geometric delay with mean ``delay_mean`` weeks. Reports past
    the window are censored. Fully vectorised over (batch, week); loop over delay lag."""
    B, W = weekly_inf.shape
    rho = np.clip(np.asarray(rho, float), 0.0, 1.0)[:, None]
    reported_total = rng.binomial(weekly_inf, rho)            # (B, W) who is ever reported
    q = (1.0 / (1.0 + np.asarray(delay_mean, float)))[:, None]  # geometric hazard per week
    q = np.clip(q, 1e-3, 1.0)
    remaining = reported_total.copy()
    cases = np.zeros((B, W), dtype=np.int64)
    for k in range(kmax + 1):
        emit = rng.binomial(remaining, q if k < kmax else np.ones_like(q))  # emitted at lag k
        if k < W:
            cases[:, k:] += emit[:, :W - k]      # source week w placed at w+k (w+k<W); rest censored
        # k >= W: every emission falls past the window end -> fully censored (dropped)
        remaining = remaining - emit
    return cases


# --- Observed-series masking (shared by training + inference) -----------------
# Monthly forecast cutoffs: an analyst re-forecasts every 4-week block, so the observed
# length at forecast time is L in {4, 8, ..., T_MAX-3}. One random L per training batch.
CUTOFFS = list(range(4, T_MAX, 4))     # [4, 8, 12, ..., 36]


def observed_and_mask(cases_full: np.ndarray, L: int, t_max: int = T_MAX):
    """Given a full-season reported-case array ``(B, T_MAX)`` and a cutoff ``L``, return
    ``(cases_obs, mask)`` each ``(B, t_max, 1)``: the observed weeks 1..L (real-time
    censored — see module docstring), zeros beyond, and a 0/1 observed-week mask. This is
    the estimator's *condition*; the same function builds it from real data at inference."""
    B = cases_full.shape[0]
    cases_obs = np.zeros((B, t_max), dtype="float32")
    cases_obs[:, :L] = cases_full[:, :L]
    mask = np.zeros((B, t_max), dtype="float32")
    mask[:, :L] = 1.0
    return cases_obs[..., None], mask[..., None]


# --- BayesFlow wrapper (lazy import) -----------------------------------------
def make_simulator(cutoffs: list[int] = CUTOFFS, t_max: int = T_MAX):
    """A BayesFlow simulator whose ``.sample(batch_shape)`` draws priors, runs the full
    ``t_max``-week season, and returns:

    - the 7 globals, each ``(B, 1)``;
    - ``log_inf`` ``(B, t_max)`` — ``log(weekly true new-infections + 1)``, the latent
      trajectory that (with the globals) forms the joint inference target;
    - ``cases`` ``(B, t_max, 1)`` and ``mask`` ``(B, t_max, 1)`` — the observed weekly case
      series truncated at one random monthly cutoff ``L`` (same ``L`` for the whole batch,
      so length varies across batches → one estimator amortises over every forecast date).
    """
    import bayesflow as bf

    class SIRSimulator(bf.simulators.Simulator):
        def sample(self, batch_shape, **kwargs):
            B = int(np.prod(batch_shape))
            rng = np.random.default_rng()
            L = int(rng.choice(cutoffs))
            params = sample_prior(B, rng)
            sim = simulate(params, t_max, rng, return_latent=True)
            log_inf = np.log(sim["weekly_inf"].astype("float32") + 1.0)   # (B, t_max)
            cases_obs, mask = observed_and_mask(sim["cases"].astype("float32"), L, t_max)
            out = {k: v.astype("float32").reshape(B, 1) for k, v in params.items()}
            out["log_inf"] = log_inf.reshape(B, t_max)
            out["cases"] = cases_obs                                       # (B, t_max, 1)
            out["mask"] = mask                                             # (B, t_max, 1)
            return out

    return SIRSimulator()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = sample_prior(6, rng)
    print("prior draws (natural scale):")
    for i in range(6):
        print("  " + "  ".join(f"{n}={p[n][i]:.4g}" for n in PARAM_NAMES))
    out = simulate(p, n_weeks=T_MAX, rng=rng, return_latent=True)
    print(f"\nweekly reported cases (6 draws x {T_MAX} weeks):")
    for i in range(6):
        print(f"  draw {i}: peak={out['cases'][i].max():5d}  total={out['cases'][i].sum():6d}  "
              f"peak_wk={int(out['cases'][i].argmax()):2d}  inf_total={int(out['weekly_inf'][i].sum())}")
