"""Validate the HistGBM v2 model on canonical out-of-sample pool.

Uses the HOLDOUT model (trained on 1999-2014) to honestly score on 2015+
contracts using the same pool definition as scripts/analyze_accuracy.py.

Comparison vs canonical baseline:
    HistGBM:  expected ~80.76% within 5% of cap on 2015+ test
    Baseline: 79.0% on same test set

Usage:
    python -u scripts/validate_histgbm_canonical.py
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from utils import (
    SEASONS, SALARY_CAP_M, fetch_league_stats,
)

sys.path.insert(0, str(Path(__file__).parent))
from train_ml_model_v2 import (
    build_career_indexes, build_rows, fetch_all_nba_selections, TRAIN_PAIRS, TEST_PAIRS, PAIRS,
    predict_canonical_baseline, CURRENT_CAP_M,
)
from train_ml_model_v3 import make_X_pruned


MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "contract_histgbm_v2_holdout.joblib"
SPLIT_YEAR = 2015


def score_full(df, pred):
    actual = df["salary_curr"].values
    cap = df["cap_curr"].values
    err_pct_cap = np.abs(actual - pred) / cap * 100
    signed = (actual - pred) / cap * 100
    err_today_M = err_pct_cap / 100 * CURRENT_CAP_M
    return {
        "n":             len(df),
        "median_err":    float(np.median(err_pct_cap)),
        "within_5":      float(np.mean(err_pct_cap <= 5.0) * 100),
        "within_10":     float(np.mean(err_pct_cap <= 10.0) * 100),
        "median_err_M":  float(np.median(err_today_M)),
        "median_bias":   float(np.median(signed)),
        "err_today_M":   err_today_M,
        "actual_today_M": actual / cap * CURRENT_CAP_M,
    }


def report(label, s):
    print(f"\n{label}", flush=True)
    print(f"  n             = {s['n']:,}", flush=True)
    print(f"  Median |err|  = {s['median_err']:.2f}% of cap  (${s['median_err_M']:.2f}M)", flush=True)
    print(f"  Within 5%     = {s['within_5']:.2f}%", flush=True)
    print(f"  Within 10%    = {s['within_10']:.2f}%", flush=True)
    print(f"  Median bias   = {s['median_bias']:+.2f}% of cap", flush=True)


def report_tiers(label, s):
    print(f"\nTier breakdown — {label}", flush=True)
    actual = s["actual_today_M"]
    err    = s["err_today_M"]
    tiers = [
        ("Max/super",  actual >= 40),
        ("Big stars",  (actual >= 25) & (actual < 40)),
        ("Mid-tier",   (actual >= 15) & (actual < 25)),
        ("Rotation",   (actual >=  7) & (actual < 15)),
        ("Min-ish",    actual <  7),
    ]
    for name, mask in tiers:
        n = int(mask.sum())
        if n == 0: continue
        sub_err = err[mask]
        w3 = float(np.mean(sub_err <= 3) * 100)
        w5 = float(np.mean(sub_err <= 5) * 100)
        med = float(np.median(sub_err))
        print(f"  {name:<12} n={n:>4}   median ${med:>5.2f}M   "
              f"±$3M {w3:>4.0f}%   ±$5M {w5:>4.0f}%", flush=True)


def main():
    print(f"Loading model from {MODEL_PATH}", flush=True)
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    print(f"  Loaded {artifact['version']} ({artifact['model_class']}, "
          f"n_train={artifact['n_train_rows']})", flush=True)

    print("\nLoading data...", flush=True)
    t0 = time.time()
    careers_rs = build_career_indexes(playoffs=False)
    print(f"  RS careers: {len(careers_rs)} in {time.time()-t0:.1f}s", flush=True)
    careers_po = {}
    all_nba = fetch_all_nba_selections()
    print(f"  All-NBA: {len(all_nba)} players", flush=True)

    # Use TEST_PAIRS (2015+ — out of sample for the holdout model).
    print(f"\nBuilding rows for {len(TEST_PAIRS)} test pairs (2015+)...", flush=True)
    t0 = time.time()
    test_df = build_rows(TEST_PAIRS, careers_rs, careers_po, all_nba)
    print(f"  {len(test_df)} contracts in {time.time()-t0:.1f}s", flush=True)

    cap_test = test_df["cap_curr"].values
    X = make_X_pruned(test_df)
    pred_pct = np.clip(model.predict(X), 0.001, 0.45)
    pred = pred_pct * cap_test

    pred_baseline = predict_canonical_baseline(test_df)

    s_h = score_full(test_df, pred)
    s_b = score_full(test_df, pred_baseline)

    print("\n" + "=" * 78, flush=True)
    print("OUT-OF-SAMPLE CANONICAL VALIDATION  (2015+ pool, same as analyze_accuracy.py)", flush=True)
    print("=" * 78, flush=True)
    report("CANONICAL BASELINE", s_b)
    report("HISTGBM v2 (new model)", s_h)

    delta_w5 = s_h["within_5"] - s_b["within_5"]
    delta_w10 = s_h["within_10"] - s_b["within_10"]
    print(f"\n  Δ Within 5%   = {delta_w5:+.2f}pp", flush=True)
    print(f"  Δ Within 10%  = {delta_w10:+.2f}pp", flush=True)

    # Modern era subset (last 10 seasons).
    modern_years = sorted({int(s.split("-")[0]) for s in SEASONS[:10]})
    modern = test_df[test_df["start_year"].isin(modern_years)]
    if len(modern):
        Xm = make_X_pruned(modern)
        capm = modern["cap_curr"].values
        pred_m = np.clip(model.predict(Xm), 0.001, 0.45) * capm
        pred_bm = predict_canonical_baseline(modern)
        smh = score_full(modern, pred_m)
        smb = score_full(modern, pred_bm)
        print("\n--- MODERN ERA (last 10 seasons) ---", flush=True)
        report("Canonical baseline (modern era)", smb)
        report("HistGBM v2 (modern era)", smh)
        print(f"\n  Δ Within 5% (modern)   = {smh['within_5'] - smb['within_5']:+.2f}pp", flush=True)
        print(f"  Δ Within 10% (modern)  = {smh['within_10'] - smb['within_10']:+.2f}pp", flush=True)

    report_tiers("Canonical baseline (out-of-sample)", s_b)
    report_tiers("HistGBM v2 (out-of-sample)",         s_h)


if __name__ == "__main__":
    main()
