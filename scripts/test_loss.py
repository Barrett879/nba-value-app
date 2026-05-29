"""Model-level lever: does ABSOLUTE-ERROR loss beat the default SQUARED-ERROR?

We grade on "within X% of cap" — a band hit-rate, essentially a median-accuracy
metric. Squared-error loss targets the MEAN and is dragged by the unpredictable
breakout tail (huge undershoots); absolute-error targets the MEDIAN and is
robust to those outliers, which should fit the bulk tighter. Principled match
to the metric — worth a real forward test.

Expanding-window temporal CV 2021-2025, fresh models per fold, eval gradeable.
Loads the cached pool.

Usage:
    python -u scripts/test_loss.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from build_production_histgbm import (
    HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask, apply_cba_postprocess,
)

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def main():
    df = pd.read_pickle("/tmp/pool_df.pkl").reset_index(drop=True)
    X = np.load("/tmp/pool_X.npy")
    sy = df["start_year"].values
    grade = gradeable_mask(df).values

    variants = {
        "squared (current)": dict(HISTGBM_PARAMS),
        "absolute_error":    {**HISTGBM_PARAMS, "loss": "absolute_error"},
    }

    folds = []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        sub = df[tem]
        folds.append((ty, trm, tem, sub,
                      sub["salary_curr"].values / sub["cap_curr"].values,
                      df.loc[trm, "salary_curr_pct"].values))

    print("=" * 72)
    print("FORWARD CV — loss function (within-5% / within-10%, per year)")
    print("=" * 72)
    results = {}
    for name, params in variants.items():
        w5s, w10s, ns = [], [], []
        for ty, trm, tem, sub, a, y in folds:
            m = HistGradientBoostingRegressor(**params).fit(X[trm], y)
            p = apply_cba_postprocess(m.predict(X[tem]), sub)
            w5s.append(np.mean(np.abs(p - a) * 100 <= 5) * 100)
            w10s.append(np.mean(np.abs(p - a) * 100 <= 10) * 100)
            ns.append(len(sub))
        ns = np.array(ns)
        results[name] = (np.array(w5s), np.array(w10s), ns)
        print(f"\n  {name}")
        for (ty, *_), w5, w10 in zip(folds, w5s, w10s):
            print(f"    {ty}:  w5 {w5:5.1f}%   w10 {w10:5.1f}%")
        print(f"    weighted:  w5 {np.average(w5s, weights=ns):5.2f}%   "
              f"w10 {np.average(w10s, weights=ns):5.2f}%")

    b5, b10, ns = results["squared (current)"]
    a5, a10, _ = results["absolute_error"]
    d5 = np.average(a5 - b5, weights=ns); d10 = np.average(a10 - b10, weights=ns)
    up = int(((a5 - b5) > 0.01).sum()); dn = int(((a5 - b5) < -0.01).sum())
    print("\n" + "=" * 72)
    print(f"  absolute_error vs squared:  Δw5 {d5:+.2f}pp   Δw10 {d10:+.2f}pp   "
          f"({up} up / {dn} down)")
    print(f"  VERDICT: {'HELPS — switch loss.' if d5 > 0.1 and up >= dn else 'WASH/HURT — keep squared.' if d5 <= 0.1 else 'MIXED.'}")


if __name__ == "__main__":
    main()
