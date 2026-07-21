"""Stage 8 — real-data inference + reliability / out-of-distribution check.

First Act III stage: the real 2018/2019 state case series finally enter, and before any
posterior is trusted we verify the real data is *typical* under the training / prior-
predictive distribution. An amortised posterior is only reliable on data resembling the
training simulations; a real series unlike anything seen in training is confidently wrong
with no error (Schmitt et al. 2023; Hermans et al. 2021). This gates stage 9.

Mirrors the toy's approach (embeddings + MMD + a **simulator-calibrated** null), adapted
to the flu model and made per-state actionable:

- **Aggregate MMD** — embed the 51 real state windows and a large prior-predictive
  reference with the trained summary network (``approximator.summarize``), compare in
  that summary space by MMD (``bayesflow.metrics.functional.maximum_mean_discrepancy``),
  and calibrate against a null built from *fresh in-distribution* samples of the same size
  (a proper Monte Carlo null, not a resample of the reference).
- **Per-state flag** — whiten the embedding by the reference (PCA), and flag any state
  whose whitened distance-to-cloud exceeds the 95th percentile of the same distance for
  fresh in-distribution draws. This says *which* states, if any, the amortised posterior
  should not be trusted on (route those back to stage 2 — a scientific widening decision).

Checked at the forecasting window lengths (T = 4, 8, 12).

    KERAS_BACKEND=jax .venv/bin/python examples/flu-sir/reliability.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from keras.ops import convert_to_numpy, convert_to_tensor

import bayesflow.diagnostics as bfd
from bayesflow.metrics.functional import maximum_mean_discrepancy

import simulator as sim
from data_flu import list_states, target_series
from train import load_trained

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")


def summaries(approx, cases_full: np.ndarray, L: int) -> np.ndarray:
    """Embed full-season (B, T_MAX) case arrays observed to cutoff ``L`` (padded + masked,
    exactly as at training) with the trained summary network → (B, summary_dim)."""
    cases, mask = sim.observed_and_mask(np.asarray(cases_full, dtype="float32"), L)
    return convert_to_numpy(approx.summarize({"cases": cases, "mask": mask}))


def sample_ref(n: int, rng: np.random.Generator) -> np.ndarray:
    """Fresh prior-predictive full-season case arrays (B, T_MAX)."""
    return sim.simulate(sim.sample_prior(n, rng), sim.T_MAX, rng)["cases"]


def mmd(a: np.ndarray, b: np.ndarray) -> float:
    return float(convert_to_numpy(maximum_mean_discrepancy(
        convert_to_tensor(a, dtype="float32"), convert_to_tensor(b, dtype="float32"))))


def real_windows() -> tuple[list, np.ndarray]:
    """The real 2018/19 full-season (T_MAX-week) case arrays, one per state."""
    states = list_states()
    rows, names = [], []
    for st in states:
        _, cases = target_series(st)
        if len(cases) >= sim.T_MAX:
            rows.append(cases[:sim.T_MAX]); names.append(st)
    return names, np.array(rows)


def whitening(ref_emb: np.ndarray, k: int = 10):
    """PCA whitening fit on the reference embeddings; returns a transform to whitened space."""
    mu = ref_emb.mean(0)
    U, S, Vt = np.linalg.svd(ref_emb - mu, full_matrices=False)
    k = min(k, (S > 1e-8).sum())
    comp = Vt[:k]; scale = S[:k] / np.sqrt(len(ref_emb) - 1)
    return lambda X: ((X - mu) @ comp.T) / np.maximum(scale, 1e-8)


def check_T(approx, L: int, n_ref: int, num_null: int, rng: np.random.Generator,
            real_full: np.ndarray, names: list) -> dict:
    ref = summaries(approx, sample_ref(n_ref, rng), L)
    real = summaries(approx, real_full, L)

    # aggregate MMD with a simulator-calibrated null (fresh in-dist samples of size n_real)
    n_real = len(real)
    d_real = mmd(real, ref)
    null = np.array([mmd(summaries(approx, sample_ref(n_real, rng), L), ref) for _ in range(num_null)])
    p_agg = float(np.mean(null >= d_real))

    # per-state whitened distance-to-cloud; null = fresh in-dist single-sample distances
    W = whitening(ref)
    real_d2 = (W(real) ** 2).sum(1)
    null_emb = summaries(approx, sample_ref(2000, rng), L)
    null_d2 = (W(null_emb) ** 2).sum(1)
    thresh = np.percentile(null_d2, 95)
    flagged = [names[i] for i in np.argsort(-real_d2) if real_d2[i] > thresh]
    return dict(T=L, d_real=d_real, null=null, p_agg=p_agg, names=names,
                real_d2=real_d2, thresh=thresh, flagged=flagged, n_real=n_real)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ref", type=int, default=1000)
    ap.add_argument("--num-null", type=int, default=150)
    args = ap.parse_args()

    approx = load_trained()
    rng = np.random.default_rng(0)
    os.makedirs(OUTDIR, exist_ok=True)
    names, real_full = real_windows()

    print("=== Stage 8: reliability / OOD check on real 2018/19 data ===")
    print("(aggregate: p = P(null MMD >= real MMD); small p => real data atypical => OOD)\n")
    for T in (4, 8, 12, 20):
        r = check_T(approx, T, args.n_ref, args.num_null, rng, real_full, names)
        verdict = "OOD — do NOT trust (route to stage 2)" if r["p_agg"] < 0.05 else "in-distribution — trustworthy"
        pct_flagged = 100 * len(r["flagged"]) / r["n_real"]
        print(f"T={T:2d}: real MMD={r['d_real']:.4f}  null95={np.percentile(r['null'],95):.4f}  "
              f"p={r['p_agg']:.3f}  -> {verdict}")
        print(f"       per-state: {len(r['flagged'])}/{r['n_real']} flagged (>{pct_flagged:.0f}% would be ~5% by chance)"
              + (f"; e.g. {', '.join(r['flagged'][:5])}" if r["flagged"] else ""))
        try:
            import matplotlib; matplotlib.use("Agg")
            fig = bfd.mmd_hypothesis_test(mmd_null=r["null"], mmd_observed=r["d_real"], alpha_level=0.05)
            p = os.path.join(OUTDIR, f"reliability_mmd_T{T}.png"); fig.savefig(p, dpi=110)
        except Exception as e:  # pragma: no cover
            print(f"       (plot skipped: {e})")
    print(f"\nplots -> {OUTDIR}/reliability_mmd_T*.png")


if __name__ == "__main__":
    main()
