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
from train_raw_model import combined_season
from utils import normalize as _norm

# Advanced stats appended to the Barrett feature set — confirmed a real gain
# by paired repeated CV (+1.12pp within-5%, t=3.9). Order is fixed and MUST
# match pages/Contract_Predictor._histgbm_feature_vector.
ADV_COLS = ["USG_PCT", "PIE", "NET_RATING", "TS_PCT", "AST_PCT", "REB_PCT"]


def _advanced_features(df: pd.DataFrame) -> np.ndarray:
    """Advanced-stat columns (ADV_COLS order) aligned to df rows by
    (normalized player name, prev season)."""
    lookup = {}
    for prev in df["prev"].unique():
        cs = combined_season(prev)
        if cs.empty:
            continue
        for _, r in cs.iterrows():
            lookup[(_norm(str(r.get("PLAYER_NAME", ""))), prev)] = {
                c: float(r.get(c, 0) or 0) for c in ADV_COLS
            }
    rows = [lookup.get((_norm(str(p)), pv), {c: 0.0 for c in ADV_COLS})
            for p, pv in zip(df["player"], df["prev"])]
    return pd.DataFrame(rows)[ADV_COLS].fillna(0).values


def make_X_augmented(df: pd.DataFrame) -> np.ndarray:
    """Production feature matrix: Barrett pruned features + advanced stats."""
    return np.hstack([make_X_pruned(df), _advanced_features(df)])


# ── Data-quality + non-contract filters ─────────────────────────────────────
# Verified salary-data errors the objective rule below can't catch — each
# checked one-by-one against the player's real contract. Bad LABELS (the
# recorded salary doesn't reflect reality), so excluded from BOTH training and
# grading: they'd teach the model noise and aren't fair to grade against.
KNOWN_BAD_LABELS = {
    # (player, season): why it's wrong
    ("Russell Westbrook", "2022-23"): "recorded $0.5M; actual ~$47M option (prorated min after Feb buyout)",
    ("Myles Turner",      "2022-23"): "recorded $35.1M; actual ~$18M (career-high salary ~$21M)",
    ("Andrew Wiggins",    "2023-24"): "flagged as a new deal but mid-contract — no signing that year",
}
_KNOWN_BAD_KEYS = {(_norm(p), s) for (p, s) in KNOWN_BAD_LABELS}


def _is_known_bad(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [(_norm(str(p)), s) in _KNOWN_BAD_KEYS for p, s in zip(df["player"], df["curr"])],
        index=df.index)


def _is_buyout_artifact(df: pd.DataFrame) -> pd.Series:
    """Mid-season buyout/waiver: a player on a real contract (>10% of cap)
    whose recorded current salary collapses to near-zero (<2% of cap). That's
    a prorated partial figure after a trade+buyout, not a freely negotiated
    new contract — and it misrepresents what they were actually paid that
    season. (Westbrook $44M→$0.5M, Dinwiddie $19.5M→$1.6M, Oladipo, Oubre.)
    A genuinely cheap player re-signing cheap (low prior) is NOT caught."""
    return (df["salary_prev_pct"] > 0.10) & (df["salary_curr_pct"] < 0.02)


def _is_bad_data(df: pd.DataFrame) -> pd.Series:
    """Corrupted/misleading salary labels — excluded from training AND grading."""
    return _is_known_bad(df) | _is_buyout_artifact(df)


def _is_rookie_stepup(df: pd.DataFrame) -> pd.Series:
    """A rookie-scale step-up is NOT a new contract — it's the CBA-mandated
    next-year salary of the player's EXISTING rookie deal (no signing
    happened). Our '≥25% YoY raise = new contract' detector wrongly flags
    these, so we drop them. Identified tightly so we don't catch young
    players signing real second contracts: a modest raise (< 1.6x) that keeps
    them at a low salary (< 10% of cap), off a low base (< 8% of cap),
    age ≤ 24. (e.g. Luka 21-22: $8M→$10M, 1.3x, 9% of cap.) A real breakout
    extension jumps well past 10% of cap, so it stays in the pool."""
    age = df["age"].fillna(99)
    ratio = df["salary_curr"] / df["salary_prev"].clip(lower=1)
    return ((age <= 24)
            & (df["salary_prev_pct"] < 0.08)
            & (df["salary_curr_pct"] < 0.10)
            & (ratio < 1.6))


def gradeable_mask(df: pd.DataFrame) -> pd.Series:
    """Rows to GRADE on: every REAL, correctly-labeled new contract —
    minimums and market deals all count. Exclusions are only:
      - rookie-scale step-ups (not new signings — see _is_rookie_stepup)
      - corrupted salary labels / mid-season buyout artifacts (see _is_bad_data)"""
    return ~_is_rookie_stepup(df) & ~_is_bad_data(df)


# Best hyperparameters — the exact config validated by cross-validation in
# scripts/validate_barrett_cv.py (single regressor beat two-stage). Shipping
# the precise config that was measured so the page claim is airtight.
HISTGBM_PARAMS = dict(
    max_iter=800,
    max_depth=5,
    learning_rate=0.02,
    min_samples_leaf=25,
    l2_regularization=0.1,
    random_state=42,
)

# Recency window — the model predicts CURRENT-season contracts, so it trains
# only on the modern CBA era (2012-13 onward). scripts/experiment_recency_
# window.py showed trimming the pre-2012 low-cap regime improves recent-season
# accuracy (86.8% within 5% / 97.1% within 10% on 2021-2025) while keeping the
# best tail-robustness of the recent windows. Older deals are a different
# financial era — noise, not signal, for a 2026 prediction.
TRAINING_START_YEAR = 2012

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH         = MODELS_DIR / "contract_histgbm_v2.joblib"           # 2012+ full data
MODEL_HOLDOUT_PATH = MODELS_DIR / "contract_histgbm_v2_holdout.joblib"   # 2012-2021 (recent holdout)

HOLDOUT_SPLIT_YEAR = 2022  # holdout model tests on recent 2022-2025


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

    print(f"\nBuilding training rows ({TRAINING_START_YEAR}+ modern CBA era)...", flush=True)
    t0 = time.time()
    train_df = build_rows(PAIRS, careers_rs, careers_po, all_nba_lookup)
    before = len(train_df)
    train_df = train_df[train_df["start_year"] >= TRAINING_START_YEAR].reset_index(drop=True)
    # Drop corrupted salary labels / buyout artifacts (bad data — would teach
    # the model noise). Rookie step-ups stay in training (accurate low-end
    # labels); they're only excluded at grading time.
    n_pre = len(train_df)
    train_df = train_df[~_is_bad_data(train_df)].reset_index(drop=True)
    print(f"  {len(train_df)} rows ({before} before {TRAINING_START_YEAR}+ trim, "
          f"{n_pre - len(train_df)} bad-data rows dropped) in {time.time()-t0:.1f}s.", flush=True)
    if train_df.empty:
        print("ERROR: no training data."); return

    # ── Full-data model (for production) ─────────────────────────────────────
    print("\nFitting HistGBM (full data)...", flush=True)
    t0 = time.time()
    y = train_df["salary_curr_pct"].values
    X = make_X_augmented(train_df)
    model_full = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X, y)
    print(f"  Fit in {time.time()-t0:.1f}s. ({X.shape[1]} features incl advanced)", flush=True)

    MODELS_DIR.mkdir(exist_ok=True)
    artifact = {
        "model":          model_full,
        "feature_cols":   PRUNED_FEATURES,
        "adv_cols":       ADV_COLS,
        "params":         HISTGBM_PARAMS,
        "n_train_rows":   len(train_df),
        "trained_on":     f"{TRAINING_START_YEAR}-13 → {PAIRS[0][1]}",
        "model_class":    "HistGradientBoostingRegressor",
        "version":        "v4-advanced",
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"  Saved full-data model to {MODEL_PATH}  ({MODEL_PATH.stat().st_size / 1024:.1f} KB)", flush=True)

    # ── Holdout model (for honest validation: train 2012-2021, test 2022+) ───
    print(f"\nFitting HistGBM HOLDOUT ({TRAINING_START_YEAR}-{HOLDOUT_SPLIT_YEAR-1})...", flush=True)
    holdout_train = train_df[train_df["start_year"] < HOLDOUT_SPLIT_YEAR]
    print(f"  Train rows (holdout): {len(holdout_train)}", flush=True)
    y_h = holdout_train["salary_curr_pct"].values
    X_h = make_X_augmented(holdout_train)
    model_holdout = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X_h, y_h)

    artifact_holdout = {
        "model":          model_holdout,
        "feature_cols":   PRUNED_FEATURES,
        "adv_cols":       ADV_COLS,
        "params":         HISTGBM_PARAMS,
        "n_train_rows":   len(holdout_train),
        "trained_on":     f"{TRAINING_START_YEAR}-13 → {HOLDOUT_SPLIT_YEAR-1}-{HOLDOUT_SPLIT_YEAR-2000:02d}",
        "model_class":    "HistGradientBoostingRegressor",
        "version":        "v4-holdout",
    }
    joblib.dump(artifact_holdout, MODEL_HOLDOUT_PATH)
    print(f"  Saved holdout model to {MODEL_HOLDOUT_PATH}  ({MODEL_HOLDOUT_PATH.stat().st_size / 1024:.1f} KB)", flush=True)

    print(f"\nExpected accuracy (temporal CV, recent seasons, all real contracts):", flush=True)
    print(f"  ~86% within 5% of cap, ~97% within 10%", flush=True)


if __name__ == "__main__":
    main()
