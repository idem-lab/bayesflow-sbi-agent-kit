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
| `diagnostics.py` | NumPy + SciPy | **Reference posterior** on a (mu, sigma) grid, plus recovery / SBC helpers. |
| `train.py` | BayesFlow + backend | BayesFlow 2 adapter + DeepSet + CouplingFlow; trains and samples. |
| `requirements.txt` | — | Pinned dependencies and backend selection. |

The split is deliberate. The model is pure NumPy; the diagnostics add only SciPy
(whose `scipy.stats` supplies validated densities rather than hand-coded ones);
only `train.py` needs a deep-learning backend.

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
```

## Status

Verified end-to-end on Python 3.12 (BayesFlow 2.0.12, Keras 3.15, JAX backend):

- **Simulator + diagnostics** — all 7 tests pass, including a check that the
  prior integrates to 1. Densities come from `scipy.stats` (validated), not
  hand-coded.
- **`train.py --smoke`** — trains, summarises, and samples the posterior.
- **`run_validation.py` (40 epochs, 300 test datasets)** — the inference works:
  - Recovery: μ correlation **0.96** (RMSE 0.25), σ correlation **0.94** (RMSE 0.19).
  - Grid cross-check: BayesFlow posterior means agree with the exact grid-reference
    posterior, near-perfectly for μ and well for σ (largest gaps at small N).
  - SBC: rank statistics are ~uniform for **both** μ (mean 0.52) and σ (mean 0.50)
    — σ is unbiased. A mild, parameter-agnostic underdispersion remains (rank std
    ~0.27 vs ideal 0.29, i.e. posteriors slightly overconfident); more epochs or a
    larger flow would tighten it.

The σ calibration depends on `.constrain("sigma", lower=0)` in the adapter
(`train.py`): σ is positive, but the flow works in unconstrained ℝ, so without the
constraint σ posteriors are biased near the σ=0 boundary. See the comment there.

The smoke run only checks the pipeline runs; `run_validation.py` is the real
inference check and writes recovery / SBC plots to `outputs/`.
