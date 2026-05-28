"""Should we cut MORE years? Test aggressive cutoffs with bootstrap CIs.

The 2012+ window was the robustness sweet spot. The question: does cutting
further (2018, 2019, 2020, 2021) keep improving confidence, or is the
apparent within-5% bump just noise bought by sacrificing robustness?

Method: fix the test set to the most recent seasons (2023-2025 — the best
proxy for "next prediction"). For each training-start cutoff, train on
[cutoff, test_year) and predict each test year. Pool the errors, then:
  - within-5% / within-10% / median
  - BOOTSTRAP 95% CI on within-5% (resample pooled errors 2000x) so we can
    see whether differences between cutoffs are statistically real
  - training sample size

If within-5% CIs overlap heavily while within-10% falls and sample size
shrinks, "cut more" is chasing noise at the cost of robustness.

Usage:
    python -u scripts/experiment_recency_aggressive.py
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

from train_ml_model_v2 import (
    build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS,
)
from train_ml_model_v3 import make_X_pruned

REG_HP = dict(max_iter=800, max_depth=5, learning_rate=0.02,
              min_samples_leaf=25, l2_regularization=0.1)

CUTOFFS = [2012, 2014, 2016, 2018, 2019, 2020, 2021]
TEST_YEARS = [2023, 2024, 2025]          # fixed recent test set for all cutoffs
RNG = np.random.RandomState(7)


def collect_errors(df, X, sy, cutoff):
    """Pooled |error| as % of cap across TEST_YEARS for one cutoff."""
    errs, train_ns = [], []
    for ty in TEST_YEARS:
        trm = (sy >= cutoff) & (sy < ty)
        tem = (sy == ty)
        if trm.sum() < 100 or tem.sum() < 10:
            continue
        reg = HistGradientBoostingRegressor(random_state=42, **REG_HP).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        cap = df.loc[tem, "cap_curr"].values
        pred = np.clip(reg.predict(X[tem]), 0.001, 0.45) * cap
        actual = df.loc[tem, "salary_curr"].values
        errs.extend((np.abs(actual - pred) / cap * 100).tolist())
        train_ns.append(int(trm.sum()))
    return np.array(errs), int(np.mean(train_ns)) if train_ns else 0


def boot_ci(errs, thresh=5.0, n_boot=2000):
    hits = (errs <= thresh).astype(float)
    n = len(hits)
    samples = [hits[RNG.randint(0, n, n)].mean() * 100 for _ in range(n_boot)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main():
    print("Building Barrett rows...", flush=True)
    t0 = time.time()
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    X = make_X_pruned(df)
    sy = df["start_year"].values
    print(f"  {len(df)} contracts in {time.time()-t0:.1f}s", flush=True)

    print(f"\n  Fixed test set: seasons {TEST_YEARS}", flush=True)
    print("\n" + "=" * 86, flush=True)
    print(f"  {'window':<9} {'train n':>8} {'within-5%':>11} {'95% CI':>16} "
          f"{'within-10%':>12} {'median':>9}", flush=True)
    print("  " + "-" * 84, flush=True)

    rows = []
    for c in CUTOFFS:
        errs, train_n = collect_errors(df, X, sy, c)
        if len(errs) == 0:
            continue
        w5 = float(np.mean(errs <= 5) * 100)
        w10 = float(np.mean(errs <= 10) * 100)
        med = float(np.median(errs))
        lo, hi = boot_ci(errs, 5.0)
        rows.append((c, train_n, w5, lo, hi, w10, med, len(errs)))
        print(f"  {c}+{'':<6} {train_n:>8,} {w5:>10.1f}% "
              f"[{lo:>4.1f}, {hi:>4.1f}] {w10:>11.1f}% {med:>8.2f}%", flush=True)

    print("  " + "-" * 84, flush=True)
    n_test = rows[0][7] if rows else 0
    print(f"  (test set = {n_test} contracts; CI = bootstrap 95% on within-5%)", flush=True)

    # Interpretation.
    base = next(r for r in rows if r[0] == 2012)
    print("\n" + "=" * 86, flush=True)
    print("READ", flush=True)
    print("=" * 86, flush=True)
    print(f"  2012+ baseline: within-5% {base[2]:.1f}% (CI {base[3]:.1f}-{base[4]:.1f}), "
          f"within-10% {base[5]:.1f}%", flush=True)
    # Does any aggressive cutoff beat 2012+ outside its CI?
    sig = [r for r in rows if r[0] > 2012 and r[2] > base[4]]
    if sig:
        print(f"  Cutoffs beating 2012+ CI on within-5%: {[r[0] for r in sig]}", flush=True)
    else:
        print("  NO cutoff beats the 2012+ within-5% confidence interval —", flush=True)
        print("  every 'gain' from cutting more is within statistical noise.", flush=True)
    # Robustness trend.
    aggressive = [r for r in rows if r[0] >= 2019]
    if aggressive:
        worst10 = min(aggressive, key=lambda r: r[5])
        print(f"  Within-10% (robustness) at aggressive cutoffs falls to "
              f"{worst10[5]:.1f}% ({worst10[0]}+), vs 2012+'s {base[5]:.1f}%.", flush=True)
        print(f"  Training data shrinks from {base[1]:,} (2012+) to "
              f"{aggressive[-1][1]:,} ({aggressive[-1][0]}+).", flush=True)


if __name__ == "__main__":
    main()
