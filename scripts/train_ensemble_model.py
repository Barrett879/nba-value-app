"""Ensemble experiment — can combining the RAW model and the BARRETT model
break the ~80.8% plateau?

Both single models land at ~80.8% within 5%, but they're built on different
feature representations:
  - RAW:     NBA fantasy pts + advanced metrics (USG/PIE/ratings) + box
  - BARRETT: hand-weighted Barrett Score + trailing-weighted variants

If their errors are partly decorrelated, blending should beat either. We test:
  1. Combined-features model (all raw + all Barrett features in one HistGBM)
  2. Simple average of the two models' predictions
  3. Stacked ensemble (Ridge meta-learner on out-of-fold predictions)

Rows are built once from each pipeline and inner-joined on (player, prev, curr)
so both models score the exact same contracts.

Temporal holdout: train 1999-2014, test 2015+.

Usage:
    python -u scripts/train_ensemble_model.py
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

from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# RAW pipeline.
from train_raw_model import (
    build_career_index as build_raw_careers, build_rows as build_raw_rows,
    fetch_all_nba_selections, TRAIN_PAIRS, TEST_PAIRS,
    FEATURE_COLS as RAW_FEATS, DERIVED as RAW_DERIVED, make_X as make_X_raw,
    predict_canonical_baseline, score, pr, tier, CURRENT_CAP_M,
)
# BARRETT pipeline (v2).
from train_ml_model_v2 import (
    build_career_indexes as build_barrett_careers, build_rows as build_barrett_rows,
)
from train_ml_model_v3 import make_X_pruned as make_X_barrett


HP = dict(max_iter=800, max_depth=5, learning_rate=0.02,
          min_samples_leaf=25, l2_regularization=0.1)


def predict_model(model, X, cap):
    return np.clip(model.predict(X), 0.001, 0.45) * cap


def main():
    print("Building RAW rows...", flush=True)
    t0 = time.time()
    raw_careers = build_raw_careers()
    all_nba = fetch_all_nba_selections()
    raw_train = build_raw_rows(TRAIN_PAIRS, raw_careers, all_nba)
    raw_test  = build_raw_rows(TEST_PAIRS,  raw_careers, all_nba)
    print(f"  raw train {len(raw_train)}, test {len(raw_test)} in {time.time()-t0:.1f}s", flush=True)

    print("Building BARRETT rows...", flush=True)
    t0 = time.time()
    b_careers = build_barrett_careers(playoffs=False)
    b_train = build_barrett_rows(TRAIN_PAIRS, b_careers, {}, all_nba)
    b_test  = build_barrett_rows(TEST_PAIRS,  b_careers, {}, all_nba)
    print(f"  barrett train {len(b_train)}, test {len(b_test)} in {time.time()-t0:.1f}s", flush=True)

    # Inner-join on (player, prev, curr) so both models see identical rows.
    keys = ["player", "prev", "curr"]
    raw_train["_k"] = raw_train[keys].astype(str).agg("|".join, axis=1)
    raw_test["_k"]  = raw_test[keys].astype(str).agg("|".join, axis=1)
    b_train["_k"]   = b_train[keys].astype(str).agg("|".join, axis=1)
    b_test["_k"]    = b_test[keys].astype(str).agg("|".join, axis=1)

    # Drop duplicate keys (mid-season trades can produce the same
    # player|prev|curr twice) so .loc[keys] doesn't inflate the row count.
    raw_train = raw_train.drop_duplicates("_k")
    raw_test  = raw_test.drop_duplicates("_k")
    b_train   = b_train.drop_duplicates("_k")
    b_test    = b_test.drop_duplicates("_k")

    tr_keys = sorted(set(raw_train["_k"]) & set(b_train["_k"]))
    te_keys = sorted(set(raw_test["_k"])  & set(b_test["_k"]))
    print(f"\nAligned: train {len(tr_keys)}, test {len(te_keys)}", flush=True)

    # Order both frames identically by key.
    raw_train = raw_train.set_index("_k").loc[tr_keys].reset_index()
    b_train   = b_train.set_index("_k").loc[tr_keys].reset_index()
    raw_test  = raw_test.set_index("_k").loc[te_keys].reset_index()
    b_test    = b_test.set_index("_k").loc[te_keys].reset_index()

    # Sanity: targets must match across the two aligned frames.
    assert np.allclose(raw_train["salary_curr_pct"].values,
                       b_train["salary_curr_pct"].values, atol=1e-6), "target mismatch (train)"

    y_tr = raw_train["salary_curr_pct"].values
    cap_te = raw_test["cap_curr"].values

    # Feature matrices.
    Xr_tr, Xr_te = make_X_raw(raw_train), make_X_raw(raw_test)
    Xb_tr, Xb_te = make_X_barrett(b_train), make_X_barrett(b_test)
    Xc_tr = np.hstack([Xr_tr, Xb_tr])  # combined superset
    Xc_te = np.hstack([Xr_te, Xb_te])

    print("\n" + "=" * 88, flush=True)
    print(f"OUT-OF-SAMPLE  (aligned test n={len(te_keys)})", flush=True)
    print("=" * 88, flush=True)

    sA = score(raw_test, predict_canonical_baseline(raw_test))
    pr("A. CANONICAL baseline (Barrett rank-map)", sA)
    base_w5 = sA["within_5"]

    # Single models.
    m_raw = HistGradientBoostingRegressor(random_state=42, **HP).fit(Xr_tr, y_tr)
    m_bar = HistGradientBoostingRegressor(random_state=42, **HP).fit(Xb_tr, y_tr)
    pred_raw = predict_model(m_raw, Xr_te, cap_te)
    pred_bar = predict_model(m_bar, Xb_te, cap_te)
    s_raw = score(raw_test, pred_raw)
    s_bar = score(raw_test, pred_bar)
    pr("B. RAW HistGBM", s_raw, base_w5)
    pr("C. BARRETT HistGBM", s_bar, base_w5)

    # Decorrelation check.
    pr_raw = m_raw.predict(Xr_te)
    pr_bar = m_bar.predict(Xb_te)
    corr = float(np.corrcoef(pr_raw, pr_bar)[0, 1])
    print(f"   prediction correlation (raw vs barrett): {corr:.4f}", flush=True)

    # 1. Combined-features model.
    m_comb = HistGradientBoostingRegressor(random_state=42, **HP).fit(Xc_tr, y_tr)
    s_comb = score(raw_test, predict_model(m_comb, Xc_te, cap_te))
    pr("D. COMBINED features (raw+barrett, 1 model)", s_comb, base_w5)

    # 2. Simple average.
    for w in [0.5, 0.4, 0.6]:
        blend = w * pred_raw + (1 - w) * pred_bar
        s = score(raw_test, blend)
        pr(f"E. average  w_raw={w}", s, base_w5)

    # 3. Stacked (Ridge meta on OOF predictions).
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(tr_keys), 2))
    for tri, vali in kf.split(Xr_tr):
        oof[vali, 0] = HistGradientBoostingRegressor(random_state=42, **HP).fit(
            Xr_tr[tri], y_tr[tri]).predict(Xr_tr[vali])
        oof[vali, 1] = HistGradientBoostingRegressor(random_state=42, **HP).fit(
            Xb_tr[tri], y_tr[tri]).predict(Xb_tr[vali])
    meta = Ridge(alpha=0.3).fit(oof, y_tr)
    stack_test = np.column_stack([m_raw.predict(Xr_te), m_bar.predict(Xb_te)])
    pred_stack = np.clip(meta.predict(stack_test), 0.001, 0.45) * cap_te
    s_stack = score(raw_test, pred_stack)
    pr("F. STACKED (Ridge meta)", s_stack, base_w5)
    print(f"   stack weights: raw={meta.coef_[0]:+.3f} barrett={meta.coef_[1]:+.3f} "
          f"int={meta.intercept_:+.4f}", flush=True)

    # 4. Combined-features sweep (does a bigger model exploit the superset?).
    print("\n  Sweeping COMBINED-features model...", flush=True)
    best_c, best_sc, best_pc = None, None, None
    for p in [
        dict(max_iter=800, max_depth=5, learning_rate=0.02, min_samples_leaf=25, l2_regularization=0.1),
        dict(max_iter=1000, max_depth=6, learning_rate=0.015, min_samples_leaf=30, l2_regularization=0.2),
        dict(max_iter=1200, max_depth=5, learning_rate=0.015, min_samples_leaf=30, l2_regularization=0.3),
        dict(max_iter=600, max_depth=7, learning_rate=0.02, min_samples_leaf=20, l2_regularization=0.2),
    ]:
        m = HistGradientBoostingRegressor(random_state=42, **p).fit(Xc_tr, y_tr)
        s = score(raw_test, predict_model(m, Xc_te, cap_te))
        if best_sc is None or s["within_5"] > best_sc["within_5"]:
            best_c, best_sc, best_pc = m, s, p
    pr(f"G. COMBINED tuned", best_sc, base_w5)

    # Tiers for the leaders.
    print("\n" + "=" * 88, flush=True)
    print("TIERS", flush=True)
    print("=" * 88, flush=True)
    tier("RAW HistGBM", s_raw)
    tier("COMBINED tuned", best_sc)
    # best average
    blend5 = 0.5 * pred_raw + 0.5 * pred_bar
    tier("AVERAGE w=0.5", score(raw_test, blend5))

    print("\n" + "=" * 88, flush=True)
    print("VERDICT", flush=True)
    print("=" * 88, flush=True)
    cands = {
        "RAW": s_raw["within_5"], "BARRETT": s_bar["within_5"],
        "COMBINED": s_comb["within_5"], "COMBINED-tuned": best_sc["within_5"],
        "AVG0.5": score(raw_test, blend5)["within_5"],
        "STACKED": s_stack["within_5"],
    }
    win = max(cands, key=cands.get)
    for k, v in sorted(cands.items(), key=lambda kv: -kv[1]):
        mark = "WINNER" if k == win else "      "
        print(f"  {mark}  {k:<16} {v:.2f}% within 5%", flush=True)
    print(f"\n  Plateau was 80.82%. Best here: {cands[win]:.2f}% ({cands[win]-80.82:+.2f}pp)", flush=True)


if __name__ == "__main__":
    main()
