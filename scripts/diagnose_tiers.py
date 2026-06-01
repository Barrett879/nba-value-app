"""Honest tier-by-tier diagnostic for the SHIPPED contract model.

Reuses the exact production pipeline (make_X_augmented = 28 features,
apply_cba_postprocess, gradeable_mask) and the same temporal CV as
validate_shipped_model.py (train pre-year, predict that unseen year, for
2021-2025), but collects every out-of-sample prediction so we can break the
error down by REAL-DOLLAR salary tier and report signed bias.

This is the baseline for the model-improvement work — it tells us where the
model is wrong (esp. the mid-tier over-projection) and by how much.

Usage:  python -u scripts/diagnose_tiers.py
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
from train_ml_model_v3 import make_X_pruned  # noqa: F401 (kept for parity)
from build_production_histgbm import (
    make_X_augmented, apply_cba_postprocess, gradeable_mask,
    HISTGBM_PARAMS, TRAINING_START_YEAR,
)

# Current cap so "% of cap" errors translate to today's dollars.
CURRENT_CAP_M = 165.0
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def collect_oos(df, X):
    """Walk-forward: for each test year, train on all prior seasons and predict
    that year's gradeable contracts. Returns arrays of actual/pred/cap (dollars)
    for every out-of-sample prediction."""
    sy = df["start_year"].values
    grade_ok = gradeable_mask(df).values
    acts, preds, caps = [], [], []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade_ok
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem]
        cap = sub["cap_curr"].values
        pred = apply_cba_postprocess(reg.predict(X[tem]), sub) * cap
        acts.append(sub["salary_curr"].values)
        preds.append(pred)
        caps.append(cap)
    return (np.concatenate(acts), np.concatenate(preds), np.concatenate(caps))


def report(actual, pred, cap):
    n = len(actual)
    err_pct_cap = np.abs(actual - pred) / cap * 100
    signed_cap = (pred - actual) / cap * 100          # + = model OVER-projects
    # translate to today's dollars (M)
    actual_today_M = actual / cap * CURRENT_CAP_M
    pred_today_M   = pred / cap * CURRENT_CAP_M
    err_today_M    = np.abs(actual_today_M - pred_today_M)
    signed_today_M = pred_today_M - actual_today_M

    print("\n" + "=" * 72, flush=True)
    print(f"OUT-OF-SAMPLE (2021-2025 walk-forward), n={n}", flush=True)
    print("=" * 72, flush=True)
    print(f"  Within 5% of cap   : {np.mean(err_pct_cap <= 5) * 100:.1f}%", flush=True)
    print(f"  Within 10% of cap  : {np.mean(err_pct_cap <= 10) * 100:.1f}%", flush=True)
    print(f"  Median |err|       : {np.median(err_pct_cap):.2f}% of cap "
          f"(${np.median(err_today_M):.2f}M today)", flush=True)
    print(f"  Median signed bias : {np.median(signed_cap):+.2f}% of cap "
          f"(${np.median(signed_today_M):+.2f}M)   [+ = over-projects]", flush=True)

    print("\nTier breakdown (by ACTUAL salary, today's $):", flush=True)
    print(f"  {'tier':<11} {'n':>4}  {'med|err|':>9}  {'med bias':>9}  "
          f"{'±$2M':>5} {'±$3M':>5} {'±$5M':>5}", flush=True)
    tiers = [
        ("Max/super", actual_today_M >= 40),
        ("Big star",  (actual_today_M >= 25) & (actual_today_M < 40)),
        ("Mid-tier",  (actual_today_M >= 15) & (actual_today_M < 25)),
        ("Rotation",  (actual_today_M >= 7)  & (actual_today_M < 15)),
        ("Min-ish",   actual_today_M < 7),
    ]
    for name, m in tiers:
        k = int(m.sum())
        if k == 0:
            continue
        e = err_today_M[m]; b = signed_today_M[m]
        print(f"  {name:<11} {k:>4}  ${np.median(e):>7.2f}M  ${np.median(b):>+7.2f}M  "
              f"{np.mean(e<=2)*100:>4.0f}% {np.mean(e<=3)*100:>4.0f}% "
              f"{np.mean(e<=5)*100:>4.0f}%", flush=True)


def main():
    print("Building 2012+ pool...", flush=True)
    t0 = time.time()
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    X = make_X_augmented(df)
    print(f"  {len(df)} contracts in {time.time()-t0:.1f}s", flush=True)
    actual, pred, cap = collect_oos(df, X)
    report(actual, pred, cap)


if __name__ == "__main__":
    main()
