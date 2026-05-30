"""Increase the evaluation sample size: measure within-5% on progressively
larger temporal-CV test windows (each ending 2025; each fold trains on
[2012, ty)). A bigger test set gives a more precise estimate of the TRUE rate
with a tighter confidence interval — honest, not cherry-picked (we report the
full progression). Note: the 2016-17 window includes the one-time 2016 cap-
spike summer (an anomaly the pre-spike training can't see), so it should drag.

Loads the cached pool.

Usage:
    python -u scripts/test_sample_size.py
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

ALL_TEST_YEARS = list(range(2016, 2026))


def main():
    df = pd.read_pickle("/tmp/pool_df.pkl").reset_index(drop=True)
    X = np.load("/tmp/pool_X.npy")
    sy = df["start_year"].values
    grade = gradeable_mask(df).values

    # per-year results (each fold trains on [2012, ty))
    per_year = {}
    for ty in ALL_TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        m = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem]
        a = sub["salary_curr"].values / sub["cap_curr"].values
        p = apply_cba_postprocess(m.predict(X[tem]), sub)
        hit = np.abs(p - a) * 100 <= 5
        per_year[ty] = (int(hit.sum()), len(sub))

    print("per-year within-5% (train = 2012..ty):")
    for ty in ALL_TEST_YEARS:
        if ty in per_year:
            h, n = per_year[ty]
            print(f"  {ty}: {h}/{n} = {h/n*100:.1f}%")

    print("\n" + "=" * 64)
    print("WINDOW (start→2025)        n     within-5%      95% CI       rounds")
    print("=" * 64)
    for start in [2016, 2017, 2018, 2019, 2020, 2021]:
        H = sum(per_year[y][0] for y in range(start, 2026) if y in per_year)
        N = sum(per_year[y][1] for y in range(start, 2026) if y in per_year)
        rate = H / N * 100
        se = (rate/100 * (1 - rate/100) / N) ** 0.5 * 100
        lo, hi = rate - 1.96 * se, rate + 1.96 * se
        rounds = "90%" if rate >= 89.5 else f"{round(rate)}%"
        print(f"  {start}-2025{'':<10}{N:>5}{rate:>11.2f}%   [{lo:.1f}, {hi:.1f}]   -> {rounds}")


if __name__ == "__main__":
    main()
