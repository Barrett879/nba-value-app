"""Honest cross-validation of the BARRETT model (the production architecture).

The live Contract Predictor uses a HistGBM on Barrett-derived features. Its
page claims "81% within 5%" — but that came from one hard temporal split
(2015+, which includes the 2016 cap-spike anomaly). This script measures the
Barrett model fairly:

  1. 5-fold random CV
  2. Expanding-window temporal CV (train ≤ year Y, predict year Y+1)

and tests single-regressor vs two-stage on the Barrett features, so we know
which Barrett model is genuinely best and what its honest accuracy is.

Usage:
    python -u scripts/validate_barrett_cv.py
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

from train_ml_model_v2 import (
    build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS,
)
from train_ml_model_v3 import make_X_pruned
from train_raw_model_v2 import cba_min_pct
from train_twostage_model import regime_label, cba_max_pct

CURRENT_CAP_M = 154.6
REG_HP = dict(max_iter=800, max_depth=5, learning_rate=0.02,
              min_samples_leaf=25, l2_regularization=0.1)
CLF_HP = dict(max_iter=800, max_depth=5, learning_rate=0.02,
              min_samples_leaf=25, l2_regularization=0.1)


def w5(te_df, pred):
    e = np.abs(te_df["salary_curr"].values - pred) / te_df["cap_curr"].values * 100
    return float(np.mean(e <= 5) * 100), float(np.mean(e <= 10) * 100)


def fit_single(tr_df, te_df, Xtr, Xte):
    y = tr_df["salary_curr_pct"].values
    cap = te_df["cap_curr"].values
    reg = HistGradientBoostingRegressor(random_state=42, **REG_HP).fit(Xtr, y)
    return np.clip(reg.predict(Xte), 0.001, 0.45) * cap


def fit_twostage(tr_df, te_df, Xtr, Xte):
    y = tr_df["salary_curr_pct"].values
    lab = tr_df["regime"].values
    cap = te_df["cap_curr"].values
    reg = HistGradientBoostingRegressor(random_state=42, **REG_HP).fit(Xtr, y)
    mid = np.clip(reg.predict(Xte), 0.001, 0.45) * cap
    if len(np.unique(lab)) < 3:
        return mid
    clf = HistGradientBoostingClassifier(random_state=42, class_weight="balanced", **CLF_HP).fit(Xtr, lab)
    proba = clf.predict_proba(Xte)
    col = {c: i for i, c in enumerate(clf.classes_)}
    p_min = proba[:, col[0]] if 0 in col else 0
    p_mid = proba[:, col[1]] if 1 in col else 0
    p_max = proba[:, col[2]] if 2 in col else 0
    svc = te_df["years_in_league"].values
    ann = te_df["all_nba_3yr"].values
    min_val = np.array([cba_min_pct(s) for s in svc]) * cap
    max_val = np.array([cba_max_pct(s, a) for s, a in zip(svc, ann)]) * cap
    return p_min * min_val + p_max * max_val + p_mid * mid


def main():
    print("Building Barrett rows (production feature pipeline)...", flush=True)
    t0 = time.time()
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    df["regime"] = df["salary_curr_pct"].apply(regime_label)
    print(f"  {len(df)} contracts in {time.time()-t0:.1f}s", flush=True)
    X = make_X_pruned(df)

    # ── 5-fold random CV ─────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("1. FIVE-FOLD RANDOM CV  (Barrett model)", flush=True)
    print("=" * 70, flush=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    s5, s5_2, w10s = [], [], []
    for i, (tri, tei) in enumerate(kf.split(X), 1):
        tr, te = df.iloc[tri], df.iloc[tei]
        ps = fit_single(tr, te, X[tri], X[tei])
        p2 = fit_twostage(tr, te, X[tri], X[tei])
        a, _ = w5(te, ps); b, b10 = w5(te, p2)
        s5.append(a); s5_2.append(b); w10s.append(b10)
        print(f"  fold {i}: single {a:.2f}%   two-stage {b:.2f}%", flush=True)
    print(f"  MEAN single:    {np.mean(s5):.2f}% ± {np.std(s5):.2f}", flush=True)
    print(f"  MEAN two-stage: {np.mean(s5_2):.2f}% ± {np.std(s5_2):.2f}", flush=True)

    # ── Expanding-window temporal CV ─────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("2. EXPANDING-WINDOW TEMPORAL CV  (Barrett model)", flush=True)
    print("=" * 70, flush=True)
    years = sorted(df["start_year"].unique())
    tw5s, tw5_2, tn, tw10 = [], [], [], []
    for ty in [y for y in years if y >= 2010]:
        trm = df["start_year"] < ty
        tem = df["start_year"] == ty
        if trm.sum() < 200 or tem.sum() < 10:
            continue
        tr, te = df[trm], df[tem]
        ps = fit_single(tr, te, X[trm.values], X[tem.values])
        p2 = fit_twostage(tr, te, X[trm.values], X[tem.values])
        a, _ = w5(te, ps); b, b10 = w5(te, p2)
        tw5s.append(a); tw5_2.append(b); tn.append(len(te)); tw10.append(b10)
        print(f"  test {ty}: n={len(te):>3}  single {a:.1f}%   two-stage {b:.1f}%", flush=True)
    tn = np.array(tn)
    ws = float(np.average(tw5s, weights=tn))
    w2 = float(np.average(tw5_2, weights=tn))
    w10 = float(np.average(tw10, weights=tn))
    print(f"  WEIGHTED single:    {ws:.2f}%", flush=True)
    print(f"  WEIGHTED two-stage: {w2:.2f}%  (within-10% {w10:.2f}%)", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("VERDICT — best BARRETT model", flush=True)
    print("=" * 70, flush=True)
    print(f"  Page currently claims:        ~81% (hard 2015+ split)", flush=True)
    print(f"  Random CV  — single:          {np.mean(s5):.2f}%   two-stage: {np.mean(s5_2):.2f}%", flush=True)
    print(f"  Temporal CV — single:         {ws:.2f}%   two-stage: {w2:.2f}%", flush=True)
    better = "two-stage" if w2 > ws + 0.5 and np.mean(s5_2) >= np.mean(s5) else "single regressor"
    print(f"  → Best defensible Barrett model: {better}", flush=True)
    print(f"  → Honest temporal-CV accuracy:   {max(ws, w2):.0f}% within 5% of cap, {w10:.0f}% within 10%", flush=True)


if __name__ == "__main__":
    main()
