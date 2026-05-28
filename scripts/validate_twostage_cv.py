"""Honest cross-validation of the two-stage architecture.

The temporal holdout result (82.9%) was reached by tuning many configs on
the SAME 2015+ test set — risking test-set overfitting via researcher
degrees of freedom. This script re-estimates the two-stage soft-blend
approach with FIXED hyperparameters (no per-fold tuning) two ways:

  1. 5-fold random CV over the full 1999+ pool  → mean ± std within-5%
  2. Expanding-window temporal CV (train ≤ year Y, test year Y+1) → the
     honest "predict the future" estimate, fold by fold.

If both land near 82-83%, the gain is real, not a tuned fluke.

Pure raw features, no Barrett. Fixed config chosen from v2 reasoning:
  regressor: HistGBM 800/5/0.02/25/0.1
  classifier: HistGBM balanced 800/5/0.02/25/0.1
  combine: probability-weighted soft blend, snap MIN→min scale, MAX→CBA max

Usage:
    python -u scripts/validate_twostage_cv.py
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

from sklearn.ensemble import (
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from sklearn.model_selection import KFold

from train_raw_model import (
    build_career_index, build_rows, fetch_all_nba_selections,
    PAIRS, make_X, CURRENT_CAP_M,
)
from train_raw_model_v2 import cba_min_pct
from train_twostage_model import regime_label, cba_max_pct

REG_HP = dict(max_iter=800, max_depth=5, learning_rate=0.02,
              min_samples_leaf=25, l2_regularization=0.1)
CLF_HP = dict(max_iter=800, max_depth=5, learning_rate=0.02,
              min_samples_leaf=25, l2_regularization=0.1)


def fit_predict_single(tr_df, te_df, X_tr, X_te):
    """Single-regressor baseline on the same folds (matched comparison)."""
    y = tr_df["salary_curr_pct"].values
    cap_te = te_df["cap_curr"].values
    reg = HistGradientBoostingRegressor(random_state=42, **REG_HP).fit(X_tr, y)
    return np.clip(reg.predict(X_te), 0.001, 0.45) * cap_te


def fit_predict_twostage(tr_df, te_df, X_tr, X_te):
    """Train two-stage on tr, predict % of cap on te. Returns predicted $."""
    y = tr_df["salary_curr_pct"].values
    reg_lab = tr_df["regime"].values
    cap_te = te_df["cap_curr"].values

    reg = HistGradientBoostingRegressor(random_state=42, **REG_HP).fit(X_tr, y)
    mid_pred = np.clip(reg.predict(X_te), 0.001, 0.45) * cap_te

    # Classifier needs all 3 classes present in train.
    if len(np.unique(reg_lab)) < 3:
        return mid_pred
    clf = HistGradientBoostingClassifier(
        random_state=42, class_weight="balanced", **CLF_HP).fit(X_tr, reg_lab)
    proba = clf.predict_proba(X_te)
    # Map class index → column (classes_ may not be [0,1,2] order).
    col = {c: i for i, c in enumerate(clf.classes_)}
    p_min = proba[:, col[0]] if 0 in col else np.zeros(len(te_df))
    p_mid = proba[:, col[1]] if 1 in col else np.zeros(len(te_df))
    p_max = proba[:, col[2]] if 2 in col else np.zeros(len(te_df))

    svc = te_df["service_years"].values
    ann = te_df["all_nba_3yr"].values
    min_val = np.array([cba_min_pct(s) for s in svc]) * cap_te
    max_val = np.array([cba_max_pct(s, a) for s, a in zip(svc, ann)]) * cap_te
    return p_min * min_val + p_max * max_val + p_mid * mid_pred


def w5(te_df, pred):
    e = np.abs(te_df["salary_curr"].values - pred) / te_df["cap_curr"].values * 100
    return float(np.mean(e <= 5) * 100), float(np.mean(e <= 10) * 100)


def main():
    print("Building full 1999+ pool...", flush=True)
    t0 = time.time()
    careers = build_career_index()
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, all_nba).reset_index(drop=True)
    df["regime"] = df["salary_curr_pct"].apply(regime_label)
    print(f"  {len(df)} contracts in {time.time()-t0:.1f}s", flush=True)
    X = make_X(df)

    # ── 1. 5-fold random CV ──────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("1. FIVE-FOLD RANDOM CV (fixed hyperparameters)", flush=True)
    print("=" * 70, flush=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    w5s, w10s, w5_single = [], [], []
    for i, (tri, tei) in enumerate(kf.split(X), 1):
        tr_df, te_df = df.iloc[tri], df.iloc[tei]
        pred = fit_predict_twostage(tr_df, te_df, X[tri], X[tei])
        pred_s = fit_predict_single(tr_df, te_df, X[tri], X[tei])
        a, b = w5(te_df, pred)
        sa, _ = w5(te_df, pred_s)
        w5s.append(a); w10s.append(b); w5_single.append(sa)
        print(f"  fold {i}: two-stage {a:.2f}%   single {sa:.2f}%   (Δ {a-sa:+.2f})", flush=True)
    print(f"  MEAN two-stage within-5%: {np.mean(w5s):.2f}% ± {np.std(w5s):.2f}", flush=True)
    print(f"  MEAN single    within-5%: {np.mean(w5_single):.2f}% ± {np.std(w5_single):.2f}", flush=True)
    print(f"  MEAN two-stage within-10%: {np.mean(w10s):.2f}% ± {np.std(w10s):.2f}", flush=True)
    print(f"  → two-stage adds {np.mean(w5s)-np.mean(w5_single):+.2f}pp on matched folds", flush=True)

    # ── 2. Expanding-window temporal CV ──────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("2. EXPANDING-WINDOW TEMPORAL CV (predict each future year)", flush=True)
    print("=" * 70, flush=True)
    years = sorted(df["start_year"].unique())
    test_years = [y for y in years if y >= 2010]  # need enough history first
    tw5, tw10, tn, ts5 = [], [], [], []
    for ty in test_years:
        tr_mask = df["start_year"] < ty
        te_mask = df["start_year"] == ty
        if tr_mask.sum() < 200 or te_mask.sum() < 10:
            continue
        tr_df, te_df = df[tr_mask], df[te_mask]
        pred = fit_predict_twostage(tr_df, te_df, X[tr_mask.values], X[te_mask.values])
        pred_s = fit_predict_single(tr_df, te_df, X[tr_mask.values], X[te_mask.values])
        a, b = w5(te_df, pred)
        sa, _ = w5(te_df, pred_s)
        tw5.append(a); tw10.append(b); tn.append(len(te_df)); ts5.append(sa)
        print(f"  test {ty}: n={len(te_df):>3}  two-stage {a:.1f}%   single {sa:.1f}%   (Δ {a-sa:+.1f})", flush=True)
    tn = np.array(tn)
    wmean5 = float(np.average(tw5, weights=tn))
    wmean10 = float(np.average(tw10, weights=tn))
    wsingle5 = float(np.average(ts5, weights=tn))
    print(f"  WEIGHTED two-stage within-5%:  {wmean5:.2f}%", flush=True)
    print(f"  WEIGHTED single    within-5%:  {wsingle5:.2f}%", flush=True)
    print(f"  WEIGHTED two-stage within-10%: {wmean10:.2f}%", flush=True)
    print(f"  → two-stage adds {wmean5-wsingle5:+.2f}pp (temporal, matched)", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("HONEST VERDICT", flush=True)
    print("=" * 70, flush=True)
    print(f"  Tuned single-split (test-set-optimistic):   82.91%", flush=True)
    print(f"  5-fold random CV:                           {np.mean(w5s):.2f}% ± {np.std(w5s):.2f}", flush=True)
    print(f"  Expanding-window temporal CV (weighted):    {wmean5:.2f}%", flush=True)
    print(f"  Single-regressor plateau (for reference):   ~80.8%", flush=True)


if __name__ == "__main__":
    main()
