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
from utils import normalize as _norm, cba_min_pct, is_known_buyout, KNOWN_BUYOUTS

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
    """Salary-data artifacts where a recorded figure misrepresents the real
    contract — a player who was on a REAL contract showing an implausibly low
    current salary. Two patterns, both keyed on 'substantial prior':
      - mid-season buyout/waiver: prev > 10% of cap, current collapses < 2%
        (Westbrook $44M→$0.5M, Dinwiddie, Oladipo, Oubre — prorated post-buyout)
      - sub-minimum partial: a rotation player (prev > 4% of cap) showing a
        figure below even the CBA minimum (< 1% of cap, ~$1.5M) — a 10-day /
        prorated partial, not a full-season deal (Delon Wright $7.8M→$0.7M,
        Reggie Jackson $10.4M→$0.6M).
    A genuinely cheap player re-signing cheap (LOW prior) is NOT caught — real
    minimum signings stay in the pool."""
    prev, curr = df["salary_prev_pct"], df["salary_curr_pct"]
    buyout  = (prev > 0.10) & (curr < 0.02)
    partial = (prev > 0.04) & (curr < 0.01)
    return buyout | partial


def _is_bad_data(df: pd.DataFrame) -> pd.Series:
    """Corrupted/misleading salary labels — excluded from training AND grading."""
    return _is_known_bad(df) | _is_buyout_artifact(df)


# Rookie-scale lock: a player still on their rookie deal whose salary ticks up
# the CBA-mandated next-year amount — NOT a negotiated contract. Detected by a
# young player (age ≤ 24) off a low base (< 8% of cap), staying low (< 10%),
# with a MODEST raise (< 1.6x). This cleanly catches Luka 21-22, Haliburton,
# Trae, LaMelo, Mikal, MPJ (all 1.2-1.5x) without touching real breakout deals
# (which multiply 3-8x). One case slips through — Poole 22-23, whose year-3→4
# scale jump was 1.77x — so it's listed explicitly below. (A broader ratio/
# salary rule was tried and over-excluded legit cheap young deals, so we use a
# tight heuristic + a named exception instead.)
KNOWN_ROOKIE_LOCKS = {
    ("Jordan Poole", "2022-23"),  # rookie 4th-year ($2.2M→$3.9M, 1.77x); extension started 2023-24
}
_KNOWN_LOCK_KEYS = {(_norm(p), s) for (p, s) in KNOWN_ROOKIE_LOCKS}


def _is_rookie_lock(df: pd.DataFrame) -> pd.Series:
    age = df["age"].fillna(99)
    ratio = df["salary_curr"] / df["salary_prev"].clip(lower=1)
    heur = ((age <= 24) & (df["salary_prev_pct"] < 0.08)
            & (df["salary_curr_pct"] < 0.10) & (ratio < 1.6))
    named = pd.Series(
        [(_norm(str(p)), s) in _KNOWN_LOCK_KEYS for p, s in zip(df["player"], df["curr"])],
        index=df.index)
    return heur | named


def gradeable_mask(df: pd.DataFrame) -> pd.Series:
    """Rows to GRADE on: every REAL, correctly-labeled new contract — minimum
    signings and market deals all count (the model predicts the whole market).
    Exclusions are only:
      - rookie-scale locks (not new signings — see _is_rookie_lock)
      - corrupted salary labels / buyout & sub-minimum-partial artifacts,
        where a recorded figure misrepresents the real contract (_is_bad_data)"""
    return ~_is_rookie_lock(df) & ~_is_bad_data(df)


# ── Prediction guards + CBA max-tier floor ──────────────────────────────────
# Floor at the CBA minimum (a player can't sign below it — kills the floor-
# glitch where the model output ~$0.1M for Clarkson) and cap at the absolute
# max (35%). PLUS a max-tier floor: snap a recent-All-NBA player the model
# already rates near-max (>=22% of cap) up to their CBA max tier. The regressor
# hedges below the true max because elite players got a SPREAD of outcomes in
# training (most max, some discounts); this restores the categorical CBA rule
# "eligible star → max". Forward-validated (test_floor_forward.py): helps in
# 5/8 seasons, hurts 0, +1.07pp, robust across thresholds. The cost is rare
# discount stars (Brunson) it overshoots — net favorable every year.
PRED_FLOOR_PCT     = 0.015   # CBA minimum (~$2.3M)
PRED_CEIL_PCT      = 0.35    # absolute CBA max
MAX_FLOOR_TRIGGER  = 0.20    # model rates them ≥20% of cap → treat as max-caliber
MAX_FLOOR_AGE_CAP  = 33      # don't floor 34+ stars — they routinely take discounts


def cba_max_pct(service_years: float, all_nba_3yr: float) -> float:
    """First-year max as % of cap: 25/30/35 by service, bumped a tier for a
    recent All-NBA selection (Rose Rule for young, supermax for vets)."""
    elite = (all_nba_3yr or 0) >= 1
    s = service_years or 0
    if s <= 6:  return 0.30 if elite else 0.25   # Rose Rule
    if s <= 9:  return 0.35 if elite else 0.30   # Designated Vet (supermax)
    return 0.35


# Buyout signings (KNOWN_BUYOUTS, cba_min_pct, is_known_buyout) live in utils.py
# so the live app and this pipeline share one source of truth. A bought-out
# player signs a CBA exception-level deal regardless of his stats; predicting the
# veteran minimum lands within 5% of cap on 105/105 historical big→small cases.


def apply_cba_postprocess(pred_pct: np.ndarray, df: pd.DataFrame = None,
                          apply_buyout: bool = True) -> np.ndarray:
    """Clip to the legal CBA range, then floor clear max-caliber stars (model
    ≥20% of cap AND recent All-NBA AND age ≤ 33) up to their CBA max tier.
    The age gate spares aging stars (Chris Paul 36) who take discounts;
    forward-validated at +0.36pp over the no-age 0.22 floor.

    Finally, snap any KNOWN_BUYOUT to the veteran minimum (apply_buyout=True) —
    a bought-out player signs a minimum-type deal regardless of his stats."""
    out = np.clip(pred_pct, PRED_FLOOR_PCT, PRED_CEIL_PCT).astype(float)
    if df is not None and "years_in_league" in df.columns and "all_nba_3yr" in df.columns:
        svc = df["years_in_league"].values
        ann = df["all_nba_3yr"].values
        age = df["age"].values if "age" in df.columns else np.full(len(out), 30.0)
        players = df["player"].values if "player" in df.columns else None
        seasons = df["curr"].values if "curr" in df.columns else None
        for i in range(len(out)):
            a = age[i]
            age_ok = a is None or (isinstance(a, float) and np.isnan(a)) or a <= MAX_FLOOR_AGE_CAP
            if out[i] >= MAX_FLOOR_TRIGGER and (ann[i] or 0) >= 1 and age_ok:
                out[i] = max(out[i], cba_max_pct(svc[i], ann[i]))
            # Buyout override — dominates everything above. A bought-out player
            # signs a minimum-type deal regardless of how good the model rates
            # him, because his money is already guaranteed by the old team.
            if apply_buyout and players is not None and seasons is not None and \
                    is_known_buyout(players[i], seasons[i]):
                out[i] = cba_min_pct(svc[i])
    return out


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
