"""Build the production HistGradientBoostingRegressor — v2 winner.

Trained on ALL 1999+ pairs with v2 feature set (All-NBA + breakout).
Hyperparameters: best from train_ml_model_v3.py sweep
  max_iter=800, max_depth=4, learning_rate=0.02,
  min_samples_leaf=25, l2_regularization=0.1

Out-of-sample (1999-2014 train, 2015+ test):
  Within 5% of cap:  80.76%  (+1.77pp vs canonical baseline 79.0%)
  Within 10% of cap: 95.00%  (+3.86pp)
  Median |err|:      $2.97M
  Big Star median miss: $12.08M (vs baseline $18.78M = -36%)

Usage:
    python -u scripts/build_production_histgbm.py
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

from sklearn.ensemble import HistGradientBoostingRegressor

# Reuse data assembly from v3 (which reuses v2's).
sys.path.insert(0, str(Path(__file__).parent))
from train_ml_model_v2 import (
    build_career_indexes, build_rows, fetch_all_nba_selections,
    TRAIN_PAIRS, TEST_PAIRS, PAIRS, FEATURE_COLS, SEASONS, SALARY_CAP_M,
)
from train_ml_model_v3 import PRUNED_FEATURES, make_X_pruned


# Best hyperparameters — the exact config validated by cross-validation in
# scripts/validate_barrett_cv.py (83% within 5% / 95% within 10% by
# expanding-window temporal CV; single regressor beat two-stage). Shipping
# the precise config that was measured so the page claim is airtight.
HISTGBM_PARAMS = dict(
    max_iter=800,
    max_depth=5,
    learning_rate=0.02,
    min_samples_leaf=25,
    l2_regularization=0.1,
    random_state=42,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH         = MODELS_DIR / "contract_histgbm_v2.joblib"           # all 1999+ data
MODEL_HOLDOUT_PATH = MODELS_DIR / "contract_histgbm_v2_holdout.joblib"   # pre-2015 only

HOLDOUT_SPLIT_YEAR = 2015


def main() -> None:
    print(f"Building production HistGBM (v2 winner)", flush=True)
    print(f"Hyperparameters: {HISTGBM_PARAMS}", flush=True)

    print("\nLoading data...", flush=True)
    t0 = time.time()
    careers_rs = build_career_indexes(playoffs=False)
    print(f"  RS careers: {len(careers_rs)} in {time.time()-t0:.1f}s", flush=True)
    careers_po = {}
    all_nba_lookup = fetch_all_nba_selections()
    print(f"  All-NBA: {len(all_nba_lookup)} players", flush=True)

    print("\nBuilding training rows for ALL 1999+ pairs...", flush=True)
    t0 = time.time()
    train_df = build_rows(PAIRS, careers_rs, careers_po, all_nba_lookup)
    print(f"  {len(train_df)} rows in {time.time()-t0:.1f}s.", flush=True)
    if train_df.empty:
        print("ERROR: no training data."); return

    # ── Full-data model (for production) ─────────────────────────────────────
    print("\nFitting HistGBM (full data)...", flush=True)
    t0 = time.time()
    y = train_df["salary_curr_pct"].values
    X = make_X_pruned(train_df)
    model_full = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X, y)
    print(f"  Fit in {time.time()-t0:.1f}s.", flush=True)

    MODELS_DIR.mkdir(exist_ok=True)
    artifact = {
        "model":          model_full,
        "feature_cols":   PRUNED_FEATURES,
        "params":         HISTGBM_PARAMS,
        "n_train_rows":   len(train_df),
        "trained_on":     f"{PAIRS[0][1]} → {PAIRS[-1][1]}",
        "model_class":    "HistGradientBoostingRegressor",
        "version":        "v2",
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"  Saved full-data model to {MODEL_PATH}  ({MODEL_PATH.stat().st_size / 1024:.1f} KB)", flush=True)

    # ── Holdout model (for honest validation) ────────────────────────────────
    print(f"\nFitting HistGBM HOLDOUT (pre-{HOLDOUT_SPLIT_YEAR} only)...", flush=True)
    holdout_train = train_df[train_df["start_year"] < HOLDOUT_SPLIT_YEAR]
    print(f"  Train rows (holdout): {len(holdout_train)}", flush=True)
    y_h = holdout_train["salary_curr_pct"].values
    X_h = make_X_pruned(holdout_train)
    model_holdout = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X_h, y_h)

    artifact_holdout = {
        "model":          model_holdout,
        "feature_cols":   PRUNED_FEATURES,
        "params":         HISTGBM_PARAMS,
        "n_train_rows":   len(holdout_train),
        "trained_on":     f"1999-2000 → 2014-15",
        "model_class":    "HistGradientBoostingRegressor",
        "version":        "v2-holdout",
    }
    joblib.dump(artifact_holdout, MODEL_HOLDOUT_PATH)
    print(f"  Saved holdout model to {MODEL_HOLDOUT_PATH}  ({MODEL_HOLDOUT_PATH.stat().st_size / 1024:.1f} KB)", flush=True)

    print(f"\nExpected out-of-sample accuracy: 80.76% within 5% of cap", flush=True)
    print(f"Big Star median miss: $12M (vs $18.78M baseline)", flush=True)


if __name__ == "__main__":
    main()
