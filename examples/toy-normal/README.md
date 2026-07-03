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

## Running it

Backend-free parts (run anywhere with NumPy):

```bash
python3 tests/test_toy_normal.py          # or: pytest tests/test_toy_normal.py
python3 examples/toy-normal/simulator.py  # prior-predictive peek
python3 examples/toy-normal/diagnostics.py
```

Full training (needs an isolated env with a backend — see `requirements.txt`):

```bash
KERAS_BACKEND=jax python examples/toy-normal/train.py --smoke   # tiny CI run
KERAS_BACKEND=jax python examples/toy-normal/train.py           # fuller run
```

## Status / caveats

- **Simulator** — verified: the 3 NumPy-only tests pass here.
- **Diagnostics** — all probability densities come from `scipy.stats`
  (validated), not hand-coded. The 4 diagnostics tests (including a check that
  the prior integrates to 1) require SciPy; they were **not run in the
  development environment** because it has no working `pip`, but they run
  wherever SciPy is installed.
- **`train.py`** — written against the BayesFlow 2.x API but **not yet executed
  end-to-end**; it needs a backend on Python 3.11/3.12.

Running the diagnostics and training in a proper environment and recording
results is the next step to tick the remaining boxes in the benchmark matrix.
