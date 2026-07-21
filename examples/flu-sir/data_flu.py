"""Real influenza data: load, clean, and align to a common epidemic-week index.

Source: ``cdcfluview::who_nrevss("state")`` (CDC FluView, clinical-labs table),
pulled once and cached as CSV under ``data/`` so nothing here depends on the live
endpoint. The observed "cases" signal is weekly **positive influenza specimens**
(``total_a + total_b``) per state.

This module is pure pandas/NumPy and backend-free: it is imported by the checks,
the reliability reference, and the final forecasting run, none of which should need
a deep-learning backend just to read data.

Design choices (see ENGINEERING_LOG.md §2):

- **Season** = the surveillance year starting at MMWR week 40. Season ``Y`` covers
  autumn ``Y`` through spring ``Y+1``.
- **Epidemic window** = the **full CDC forecasting season**: 39 consecutive weekly dates
  starting at the first reporting date on/after 1 October of the season year (≈ MMWR week
  40 → week 26, ~1 Oct → ~30 Jun). Indexing by real ``wk_date`` (Saturdays) avoids the
  missing-MMWR-week-53 gap (2018 has no week 53). This captures the whole epidemic — slow
  autumn onset, midwinter growth-through-peak, spring tail — for every state on one aligned
  index. (All 51 states have exactly 39 weekly reports in this window in every season.)
- **Monthly forecast cutoffs**: an analyst re-forecasts once a "month" (4-week block), so
  the observed length at forecast time is ``L ∈ {4, 8, …, 36}`` weeks; the amortised
  estimator is trained over a random ``L`` so one network serves every cutoff.
- The **target** season is 2018/2019. Everything a late-2018 analyst is allowed to use
  for priors/checks comes from seasons **< 2018** (2015/16, 2016/17, 2017/18 in this
  table).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLINICAL_CSV = os.path.join(DATA_DIR, "who_nrevss_state_icl_nrevss_clinical_labs.csv")

TARGET_SEASON = 2018            # 2018/2019
PRETARGET_SEASONS = (2015, 2016, 2017)
WINDOW_WEEKS = 39               # full CDC season length (weeks): ~1 Oct → ~30 Jun
WINDOW_START_MONTH_DAY = (10, 1)  # first reporting date on/after 1 Oct of the season year
MONTHLY_STEP = 4                # a "month" is a 4-week block; forecasts issued at multiples

# Non-state regions in the "States" region_type to exclude from the state analysis.
NON_STATES = {"New York City", "Puerto Rico", "Virgin Islands", "Commonwealth of the Northern Mariana Islands"}


def load_clinical(path: str = CLINICAL_CSV) -> pd.DataFrame:
    """Load the clinical-labs table with a ``pos`` (positive specimens) column and a
    parsed ``wk_date``. Missing weekly counts are left as NaN (not zero-filled)."""
    df = pd.read_csv(path)
    df["wk_date"] = pd.to_datetime(df["wk_date"])
    df["pos"] = df["total_a"].fillna(0) + df["total_b"].fillna(0)
    # season = surveillance year starting at MMWR week 40
    df["season"] = np.where(df["week"] >= 40, df["year"], df["year"] - 1)
    return df


def list_states(df: pd.DataFrame | None = None) -> list[str]:
    """Sorted list of the 50 states + DC present in the table (territories/NYC dropped)."""
    if df is None:
        df = load_clinical()
    regions = sorted(set(df["region"].unique()) - NON_STATES)
    return regions


def epidemic_window(df: pd.DataFrame, state: str, season: int,
                    n_weeks: int = WINDOW_WEEKS) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dates, cases)`` for one (state, season) epidemic window.

    ``cases`` is a length-``n_weeks`` int array of weekly positive specimens; weeks with
    no report are 0. If the state has no data in the window, returns empty arrays.
    """
    start = pd.Timestamp(year=season, month=WINDOW_START_MONTH_DAY[0], day=WINDOW_START_MONTH_DAY[1])
    sub = df[(df["region"] == state) & (df["wk_date"] >= start)].sort_values("wk_date")
    sub = sub.head(n_weeks)
    if len(sub) == 0:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype=int)
    dates = sub["wk_date"].to_numpy()
    cases = np.rint(sub["pos"].to_numpy()).astype(int)
    return dates, cases


def target_series(state: str, n_weeks: int = WINDOW_WEEKS,
                  df: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The 2018/2019 epidemic window for one state: ``(dates, cases)``."""
    if df is None:
        df = load_clinical()
    return epidemic_window(df, state, TARGET_SEASON, n_weeks)


def pretarget_windows(n_weeks: int = WINDOW_WEEKS, min_peak: int = 10,
                      df: pd.DataFrame | None = None) -> list[np.ndarray]:
    """Every (state, season) epidemic window from seasons < 2018 (the data a late-2018
    analyst may use). Returns a list of length-``n_weeks`` int case arrays.

    Windows with a peak below ``min_peak`` (essentially no epidemic recorded) are dropped
    so the "real epidemic curves" library used for coverage/reliability isn't diluted by
    near-empty series. Windows shorter than ``n_weeks`` (truncated records) are dropped.
    """
    if df is None:
        df = load_clinical()
    out = []
    for state in list_states(df):
        for season in PRETARGET_SEASONS:
            _, cases = epidemic_window(df, state, season, n_weeks)
            if len(cases) == n_weeks and cases.max() >= min_peak:
                out.append(cases)
    return out


def monthly_cutoffs(n_weeks: int = WINDOW_WEEKS, step: int = MONTHLY_STEP) -> list[int]:
    """Observed-length cutoffs ``L`` at which a monthly forecast is issued: 4, 8, …, up to
    the last full block strictly inside the season (so there is always ≥1 week to forecast).
    With the 39-week season this is ``[4, 8, 12, …, 36]``."""
    return list(range(step, n_weeks, step))


def _summary(cases: np.ndarray) -> dict:
    """A few interpretable summaries of one weekly case curve (used by the checks)."""
    t = np.arange(len(cases))
    total = cases.sum()
    peak = cases.max()
    peak_week = int(np.argmax(cases))
    # growth rate over the first half (log-linear slope of cases+1)
    half = max(3, len(cases) // 2)
    y = np.log(cases[:half] + 1.0)
    slope = np.polyfit(t[:half], y, 1)[0] if half >= 2 else 0.0
    return dict(total=int(total), peak=int(peak), peak_week=peak_week, log_growth=float(slope))


if __name__ == "__main__":
    df = load_clinical()
    states = list_states(df)
    print(f"clinical-labs table: {df.shape[0]} rows, {len(states)} states+DC")
    print(f"target season {TARGET_SEASON}/{TARGET_SEASON+1}, window = {WINDOW_WEEKS} weeks from "
          f"{WINDOW_START_MONTH_DAY[0]}/{WINDOW_START_MONTH_DAY[1]}")
    print(f"monthly forecast cutoffs L = {monthly_cutoffs()}\n")

    # Target: a few featured states
    for st in ["Illinois", "California", "Vermont"]:
        dates, cases = target_series(st, df=df)
        if len(cases):
            print(f"{st:12s} 2018/19: weeks={len(cases)}  cases={cases.tolist()}")
            print(f"{'':12s}         {_summary(cases)}")

    lib = pretarget_windows(df=df)
    peaks = np.array([c.max() for c in lib])
    totals = np.array([c.sum() for c in lib])
    print(f"\npre-target library (seasons {PRETARGET_SEASONS}): {len(lib)} epidemic windows")
    print(f"  peak weekly positives  q10/50/90 = "
          f"{np.percentile(peaks,10):.0f}/{np.percentile(peaks,50):.0f}/{np.percentile(peaks,90):.0f}")
    print(f"  window total positives q10/50/90 = "
          f"{np.percentile(totals,10):.0f}/{np.percentile(totals,50):.0f}/{np.percentile(totals,90):.0f}")
