"""Final honest accuracy of the SHIPPED model (2012+, Barrett + advanced),
measured the same way the page claims it: expanding-window temporal CV on
recent seasons (2021-2025), within-5%/within-10% of cap. Also prints the
base-vs-augmented delta on the identical folds.

Usage:
    python -u scripts/validate_shipped_model.py
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
from train_ml_model_v3 import make_X_pruned
from build_production_histgbm import (
    make_X_augmented, HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask,
    apply_cba_postprocess,
)


def wcap(actual, pred, cap, t):
    return float(np.mean(np.abs(actual - pred) / cap * 100 <= t) * 100)


def temporal(df, Xfull, test_years):
    sy = df["start_year"].values
    grade_ok = gradeable_mask(df).values
    w5s, w10s, ns = [], [], []
    for ty in test_years:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade_ok
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            Xfull[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem]
        cap = sub["cap_curr"].values
        pred = apply_cba_postprocess(reg.predict(Xfull[tem]), sub) * cap
        actual = sub["salary_curr"].values
        w5s.append(wcap(actual, pred, cap, 5)); w10s.append(wcap(actual, pred, cap, 10))
        ns.append(int(tem.sum()))
    ns = np.array(ns)
    return float(np.average(w5s, weights=ns)), float(np.average(w10s, weights=ns)), int(ns.sum())


def main():
    print("Building 2012+ pool...", flush=True)
    t0 = time.time()
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    X_base = make_X_pruned(df)
    X_aug = make_X_augmented(df)
    print(f"  {len(df)} contracts in {time.time()-t0:.1f}s", flush=True)

    ty = [2021, 2022, 2023, 2024, 2025]
    b5, b10, n = temporal(df, X_base, ty)
    a5, a10, _ = temporal(df, X_aug, ty)

    print("\n" + "=" * 64, flush=True)
    print("SHIPPED MODEL — temporal CV on recent seasons (2021-2025)", flush=True)
    print("=" * 64, flush=True)
    print(f"  test predictions: {n}", flush=True)
    print(f"  Barrett only:        within-5% {b5:.1f}%   within-10% {b10:.1f}%", flush=True)
    print(f"  + advanced stats:    within-5% {a5:.1f}%   within-10% {a10:.1f}%   ← SHIPPED", flush=True)
    print(f"  advanced-stats gain: {a5-b5:+.1f}pp within-5%, {a10-b10:+.1f}pp within-10%", flush=True)


if __name__ == "__main__":
    main()
