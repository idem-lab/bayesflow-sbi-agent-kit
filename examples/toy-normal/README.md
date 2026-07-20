# Toy example: mean & standard deviation of a Normal

**This is a workflow test, not scientific evidence.** It is the fast reference
model used to exercise the end-to-end BayesFlow workflow and to serve as a
CI smoke test. See `validation/benchmark-matrix.md`.

## The model

Infer the mean and standard deviation of a Normal distribution from a *variable*
number of i.i.d. observations:

```
mu    ~ Normal(0, 1)        # location
sigma ~ HalfNormal(1)       # scale, > 0
N     ~ randint(5, 20)      # observations per dataset (variable)
x_i   ~ Normal(mu, sigma)   # i = 1..N, i.i.d.
```

The priors were chosen and human-approved for this toy; they are weakly
informative and keep the data on a clean scale so recovery and calibration are
easy to read. They are *not* a template for any real analysis.

Because the observations are exchangeable and variable-length, the inference
network sees them through a permutation-invariant **DeepSet** summary network
(via the adapter's `.as_set("x")`), which maps any N observations to a
fixed-size embedding. This is the core BayesFlow capability the toy exercises.

## Files

| File | Needs | Purpose |
|---|---|---|
| `simulator.py` | NumPy | The model: `prior`, `meta` (variable N), `likelihood`, and a lazy `make_simulator()`. |
| `diagnostics.py` | NumPy + SciPy | **Reference posterior** on a (mu, sigma) grid — the bespoke, model-specific ground truth BayesFlow cannot provide. |
| `train.py` | BayesFlow + backend | BayesFlow 2 adapter + DeepSet + CouplingFlow; trains and samples. |
| `run_validation.py` | BayesFlow + backend | Fuller training run + inference checks: recovery, calibration and sensitivity via `bayesflow.diagnostics`, plus the grid cross-check. |
| `posterior_predictive.py` | BayesFlow + backend + ArviZ | Posterior predictive check (stage 9): re-simulates replicated datasets from the model's own `likelihood` at posterior draws and compares them to an observation with **ArviZ** (`plot_ppc_dist`, `plot_ppc_tstat`, `plot_ppc_pit`). `--misspecify` shows the check catching a skewed misfit. |
| `requirements.txt` | — | Pinned dependencies and backend selection. |

The split is deliberate. The model is pure NumPy; the grid reference adds only SciPy
(whose `scipy.stats` supplies validated densities rather than hand-coded ones); only
`train.py` / `run_validation.py` / `posterior_predictive.py` need a deep-learning
backend. Recovery, calibration (SBC) and the z-score/contraction sensitivity diagnostic
are **not** hand-rolled — they come straight from `bayesflow.diagnostics`; the
posterior-predictive plots and Bayesian p-values come straight from **ArviZ**. The only
model-specific thing we compute ourselves is the exact grid reference posterior, which no
library can provide.

## What this covers in the workflow

Prior-predictive simulation (`simulator.py`), a training smoke test and
posterior-sampling smoke test (`train.py`), and the structure for parameter
recovery and posterior-predictive checks against the grid reference
(`diagnostics.py`). The grid reference is the payoff of using a toy: it gives a
ground-truth posterior to validate the amortised BayesFlow posterior against —
a check that is impossible for real models.

## Environment setup

BayesFlow 2 needs Python 3.11/3.12 and a Keras 3 backend. The reproducible route
(independent of any system/Homebrew Python) uses [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv                                    # isolated interpreter
uv pip install --python .venv/bin/python -r examples/toy-normal/requirements.txt
```

The simulator, diagnostics, and tests need only NumPy + SciPy; `train.py` also
needs the backend.

## Running it

NumPy/SciPy parts:

```bash
.venv/bin/python tests/test_toy_normal.py          # or: pytest tests/test_toy_normal.py
.venv/bin/python examples/toy-normal/simulator.py  # prior-predictive peek
.venv/bin/python examples/toy-normal/diagnostics.py
```

Training (needs the backend):

```bash
KERAS_BACKEND=jax .venv/bin/python examples/toy-normal/train.py --smoke   # tiny run
KERAS_BACKEND=jax .venv/bin/python examples/toy-normal/train.py           # fuller run
KERAS_BACKEND=jax .venv/bin/python examples/toy-normal/run_validation.py  # recovery/SBC/sensitivity
KERAS_BACKEND=jax .venv/bin/python examples/toy-normal/posterior_predictive.py             # posterior predictive check
KERAS_BACKEND=jax .venv/bin/python examples/toy-normal/posterior_predictive.py --misspecify # ...catching a misfit
```

## Status

Verified end-to-end on Python 3.12 (BayesFlow 2.0.12, Keras 3.15, JAX backend):

- **Simulator + grid reference** — all 8 tests pass, including a check that the
  prior integrates to 1 and the posterior-predictive p-value logic (well-specified vs.
  a skewed misfit). Densities come from `scipy.stats` (validated), not hand-coded.
- **`train.py --smoke`** — trains, summarises, and samples the posterior.
- **`run_validation.py` (40 epochs, 300 test datasets)** — the inference works.
  All metrics below come from `bayesflow.diagnostics`; a representative run (numbers
  vary run to run):
  - Recovery: μ correlation **0.92** (NRMSE 0.26), σ correlation **0.87** (NRMSE 0.32).
  - Calibration (ECDF-based SBC): calibration error is small for both — μ **0.04**,
    σ **0.07**. Mean posterior z-score ≈ **−0.03** for μ (unbiased) and **+0.15**
    for σ (a mild residual positive bias). Posterior contraction ≈ 0.9 for both, so
    both are well-identified under these priors.
  - Grid cross-check: BayesFlow posterior means agree with the exact grid-reference
    posterior, near-perfectly for μ and well for σ (largest gaps at small N).
- **`posterior_predictive.py` (30 epochs, N=20 observation)** — the posterior
  predictive check (stage 9), all diagnostics via **ArviZ**. On the correctly
  specified model it passes: Bayesian p-values are all moderate (representative run —
  mean 0.42, std 0.61, min 0.32, max 0.40, iqr 0.87, skew 0.23), and the observed
  density sits inside the replicated cloud. With `--misspecify` (a skewed observation
  matched in mean and sd), the fitted moments still look fine (mean 0.64, std 0.54) and
  the generic tail checks stay quiet (min 0.22, max 0.36, iqr 0.68), but **skewness
  flags the misfit (p ≈ 0.009)** — the observed skewness lands far outside the
  replicated cloud. Plots (`plot_ppc_dist`, `plot_ppc_tstat`, `plot_ppc_pit`) are
  written to `outputs/` (misspecified ones prefixed `misspec_`).

The σ calibration depends on `.constrain("sigma", lower=0)` in the adapter
(`train.py`): σ is positive, but the flow works in unconstrained ℝ, so without the
constraint σ posteriors are biased near the σ=0 boundary. See the comment there.

The smoke run only checks the pipeline runs; `run_validation.py` is the real
inference check and writes recovery, calibration-ECDF and z-score/contraction plots
to `outputs/`.
