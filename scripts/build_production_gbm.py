"""Train the production GBM contract-prediction model on ALL 1999+ data
and save to cache/. Uses the best hyperparameters from the sweep in
train_ml_model.py.

Out-of-sample accuracy (estimated from 1999-2014 train / 2015+ test split):
  Within 5% of cap: 80.1%  (baseline 79.0%, +1.14pp)

Run this script once when retraining is needed (e.g., after adding new
season data). The live Contract Predictor loads the saved model at startup.

Usage:
    python scripts/build_production_gbm.py
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

from sklearn.ensemble import GradientBoostingRegressor

# Reuse the data assembly + feature engineering from train_ml_model.py.
sys.path.insert(0, str(Path(__file__).parent))
from train_ml_model import (
    build_career_indexes, build_rows, make_X, FEATURE_COLS,
    PAIRS, SEASONS, SALARY_CAP_M,
)


# Best hyperparameters from sweep in train_ml_model.py.
GBM_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    min_samples_leaf=15,
    subsample=0.8,
    random_state=42,
)

# Save destinations.
CACHE_DIR = Path(__file__).parent.parent / "cache"
MODEL_PATH         = CACHE_DIR / "contract_gbm_v1.joblib"           # full-data model (for production)
MODEL_HOLDOUT_PATH = CACHE_DIR / "contract_gbm_v1_holdout.joblib"   # 1999-2014 train (for validation)

HOLDOUT_SPLIT_YEAR = 2015


def main() -> None:
    print(f"Building production GBM model")
    print(f"Training on ALL 1999+ pairs ({len(PAIRS)} pairs)")
    print(f"Hyperparameters: {GBM_PARAMS}")

    print("\nBuilding career indexes...")
    t0 = time.time()
    careers = build_career_indexes()
    print(f"  {len(careers)} careers in {time.time()-t0:.1f}s.")

    print("\nBuilding training rows (1999+ all pairs)...")
    t0 = time.time()
    train_df = build_rows(PAIRS, careers)
    print(f"  {len(train_df)} rows in {time.time()-t0:.1f}s.")
    if train_df.empty:
        print("ERROR: no training data."); return

    print("\nFitting Gradient Boosting...")
    t0 = time.time()
    y = train_df["salary_curr_pct"].values
    X = make_X(train_df)
    gb = GradientBoostingRegressor(**GBM_PARAMS).fit(X, y)
    print(f"  Fit in {time.time()-t0:.1f}s.")

    # Feature importances.
    print("\nFEATURE IMPORTANCES:")
    fnames = list(FEATURE_COLS) + ["age²", "barrett²", "log(rank+1)",
                                     "tier_30pct", "tier_35pct",
                                     "is_Guard", "is_Forward"]
    importances = sorted(zip(fnames, gb.feature_importances_),
                         key=lambda kv: -kv[1])
    for name, imp in importances:
        print(f"  {name:<22} {imp:.3f}")

    # Save full-data model + feature column list as a single artifact.
    CACHE_DIR.mkdir(exist_ok=True)
    artifact = {
        "model":         gb,
        "feature_cols":  FEATURE_COLS,
        "params":        GBM_PARAMS,
        "n_train_rows":  len(train_df),
        "trained_on":    f"{PAIRS[0][1]} → {PAIRS[-1][1]}",
        "version":       "v1",
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved full-data model to {MODEL_PATH}  ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")

    # Holdout model — for honest validation. Train only on pre-2015 pairs.
    print(f"\nTraining HOLDOUT model (pre-{HOLDOUT_SPLIT_YEAR} only, for validation)...")
    holdout_train = train_df[train_df["start_year"] < HOLDOUT_SPLIT_YEAR]
    print(f"  Train rows (holdout): {len(holdout_train)}")
    y_h = holdout_train["salary_curr_pct"].values
    X_h = make_X(holdout_train)
    gb_h = GradientBoostingRegressor(**GBM_PARAMS).fit(X_h, y_h)
    artifact_holdout = {
        "model":         gb_h,
        "feature_cols":  FEATURE_COLS,
        "params":        GBM_PARAMS,
        "n_train_rows":  len(holdout_train),
        "trained_on":    f"1999-2000 → {HOLDOUT_SPLIT_YEAR-1}-{HOLDOUT_SPLIT_YEAR-2000:02d}",
        "version":       "v1-holdout",
    }
    joblib.dump(artifact_holdout, MODEL_HOLDOUT_PATH)
    print(f"  Saved holdout model to {MODEL_HOLDOUT_PATH}  ({MODEL_HOLDOUT_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"\nExpected out-of-sample accuracy: 80.1% within 5% of cap")


if __name__ == "__main__":
    main()
