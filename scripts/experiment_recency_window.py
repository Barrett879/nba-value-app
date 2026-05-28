"""How far back should training data go to best predict CURRENT contracts?

The goal is predicting this season's deals. Old contracts come from a
different financial era (pre-2016 low cap, the 2016 spike, the 2017 supermax
rules, the 2023 second apron) — they may be noise, not signal, for a 2026
prediction. But cutting data raises variance. This finds the sweet spot
empirically.

Method: for each candidate training-start cutoff, predict each RECENT test
season (2021-2025) using a model trained only on [cutoff, test_year-1], and
measure accuracy. The cutoff that best predicts recent seasons wins.

Barrett feature pipeline (the shipped model). Within-5%/10% of cap.

Usage:
    python -u scripts/experiment_recency_window.py
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

START_CUTOFFS = [1999, 2008, 2012, 2014, 2016, 2018]
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def w5(actual, pred, cap):
    e = np.abs(actual - pred) / cap * 100
    return float(np.mean(e <= 5) * 100), float(np.mean(e <= 10) * 100)


def main():
    print("Building Barrett rows...", flush=True)
    t0 = time.time()
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    X = make_X_pruned(df)
    sy = df["start_year"].values
    print(f"  {len(df)} contracts in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("RECENCY-WINDOW SEARCH — predict recent seasons (2021-2025)", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'train start':<12} {'train n*':>9} {'within-5%':>11} {'within-10%':>12}", flush=True)
    print("  " + "-" * 48, flush=True)

    results = []
    for cutoff in START_CUTOFFS:
        w5s, w10s, ns, train_ns = [], [], [], []
        for ty in TEST_YEARS:
            trm = (sy >= cutoff) & (sy < ty)
            tem = (sy == ty)
            if trm.sum() < 150 or tem.sum() < 10:
                continue
            reg = HistGradientBoostingRegressor(random_state=42, **REG_HP).fit(
                X[trm], df.loc[trm, "salary_curr_pct"].values)
            cap = df.loc[tem, "cap_curr"].values
            pred = np.clip(reg.predict(X[tem]), 0.001, 0.45) * cap
            actual = df.loc[tem, "salary_curr"].values
            a, b = w5(actual, pred, cap)
            w5s.append(a); w10s.append(b); ns.append(int(tem.sum()))
            train_ns.append(int(trm.sum()))
        ns = np.array(ns)
        agg5 = float(np.average(w5s, weights=ns))
        agg10 = float(np.average(w10s, weights=ns))
        avg_train = int(np.mean(train_ns))
        results.append((cutoff, avg_train, agg5, agg10))
        print(f"  {cutoff}+{'':<7} {avg_train:>9,} {agg5:>10.2f}% {agg10:>11.2f}%", flush=True)

    print("  " + "-" * 48, flush=True)
    best = max(results, key=lambda r: r[2])
    print(f"\n  Best window: {best[0]}+  ({best[2]:.2f}% within 5%, {best[3]:.2f}% within 10%)", flush=True)
    print(f"  (avg ~{best[1]:,} training contracts per recent-season prediction)", flush=True)

    # Per-year detail for the best cutoff vs the all-data 1999 cutoff.
    print("\n  Per-year detail (best window vs all-data 1999+):", flush=True)
    print(f"  {'test yr':<9} {'best w5':>9} {'1999+ w5':>10}", flush=True)
    for ty in TEST_YEARS:
        row = {}
        for cutoff in [best[0], 1999]:
            trm = (sy >= cutoff) & (sy < ty); tem = (sy == ty)
            if trm.sum() < 150 or tem.sum() < 10:
                row[cutoff] = None; continue
            reg = HistGradientBoostingRegressor(random_state=42, **REG_HP).fit(
                X[trm], df.loc[trm, "salary_curr_pct"].values)
            cap = df.loc[tem, "cap_curr"].values
            pred = np.clip(reg.predict(X[tem]), 0.001, 0.45) * cap
            a, _ = w5(df.loc[tem, "salary_curr"].values, pred, cap)
            row[cutoff] = a
        b = f"{row[best[0]]:.1f}%" if row[best[0]] is not None else "—"
        a = f"{row[1999]:.1f}%" if row[1999] is not None else "—"
        print(f"  {ty:<9} {b:>9} {a:>10}", flush=True)


if __name__ == "__main__":
    main()
