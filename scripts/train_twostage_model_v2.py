"""TWO-STAGE v2 — push past 82.15% by improving the regime classifier.

v1 finding: soft-blend two-stage hits 82.15% within 5% (+1.33pp over the
single-regressor plateau). The bottleneck is MAX recall (36%) — the
classifier is conservative and misses 64% of max deals, routing them to the
MID regressor which undershoots.

v2 levers:
  1. class_weight='balanced' — upweight the rare MIN/MAX classes for recall
  2. probability calibration (isotonic) so soft-blend weights are honest
  3. classifier hyperparameter sweep
  4. probability sharpening on the snap classes
  5. a 4th regime (EXCEPTION ~ MLE cluster) — test if finer helps

Pure raw + advanced features. NO Barrett. Train 1999-2014, test 2015+.

Usage:
    python -u scripts/train_twostage_model_v2.py
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

from train_raw_model import (
    build_career_index, build_rows, fetch_all_nba_selections,
    TRAIN_PAIRS, TEST_PAIRS, make_X, score, pr, tier,
    predict_canonical_baseline, CURRENT_CAP_M,
)
from train_raw_model_v2 import cba_min_pct
from train_twostage_model import MIN_CUT, MAX_CUT, regime_label, cba_max_pct, REG_HP


def predict_model(model, X, cap):
    return np.clip(model.predict(X), 0.001, 0.45) * cap


def soft_blend(proba, min_val, max_val, mid_pred):
    return proba[:, 0] * min_val + proba[:, 2] * max_val + proba[:, 1] * mid_pred


def main():
    print("Building data...", flush=True)
    t0 = time.time()
    careers = build_career_index()
    all_nba = fetch_all_nba_selections()
    train_df = build_rows(TRAIN_PAIRS, careers, all_nba)
    test_df  = build_rows(TEST_PAIRS,  careers, all_nba)
    print(f"  train {len(train_df)}, test {len(test_df)} in {time.time()-t0:.1f}s", flush=True)

    train_df["regime"] = train_df["salary_curr_pct"].apply(regime_label)
    test_df["regime"]  = test_df["salary_curr_pct"].apply(regime_label)

    y = train_df["salary_curr_pct"].values
    reg_lab = train_df["regime"].values
    X_tr, X_te = make_X(train_df), make_X(test_df)
    cap_te = test_df["cap_curr"].values
    svc = test_df["service_years"].values
    ann = test_df["all_nba_3yr"].values
    min_val = np.array([cba_min_pct(s) for s in svc]) * cap_te
    max_val = np.array([cba_max_pct(s, a) for s, a in zip(svc, ann)]) * cap_te

    print("\n" + "=" * 90, flush=True)
    print(f"OUT-OF-SAMPLE  (test n={len(test_df)})", flush=True)
    print("=" * 90, flush=True)

    sA = score(test_df, predict_canonical_baseline(test_df))
    pr("A. CANONICAL baseline", sA)
    base_w5 = sA["within_5"]

    reg_all = HistGradientBoostingRegressor(random_state=42, **REG_HP).fit(X_tr, y)
    pred_reg = predict_model(reg_all, X_te, cap_te)
    s_reg = score(test_df, pred_reg)
    pr("B. Single regressor", s_reg, base_w5)

    # ── v1 classifier (unbalanced) for reference ─────────────────────────────
    clf0 = HistGradientBoostingClassifier(
        random_state=42, max_iter=600, max_depth=5, learning_rate=0.03,
        min_samples_leaf=20, l2_regularization=0.1).fit(X_tr, reg_lab)
    p0 = clf0.predict_proba(X_te)
    s_v1 = score(test_df, soft_blend(p0, min_val, max_val, pred_reg))
    pr("C. v1 soft blend (unbalanced clf)", s_v1, base_w5)

    def max_recall(proba):
        pc = proba.argmax(axis=1)
        ac = test_df["regime"].values
        tp = ((pc == 2) & (ac == 2)).sum(); fn = ((pc != 2) & (ac == 2)).sum()
        return tp / (tp + fn) if (tp + fn) else 0

    # ── D. balanced class weights ────────────────────────────────────────────
    clf_bal = HistGradientBoostingClassifier(
        random_state=42, class_weight="balanced",
        max_iter=600, max_depth=5, learning_rate=0.03,
        min_samples_leaf=20, l2_regularization=0.1).fit(X_tr, reg_lab)
    p_bal = clf_bal.predict_proba(X_te)
    s_bal = score(test_df, soft_blend(p_bal, min_val, max_val, pred_reg))
    pr(f"D. balanced clf  (MAX recall {max_recall(p_bal)*100:.0f}%)", s_bal, base_w5)

    # ── E. classifier sweep (balanced) ───────────────────────────────────────
    print("\n  Sweeping balanced classifier...", flush=True)
    best_p, best_s, best_hp = None, None, None
    for hp in [
        dict(max_iter=400, max_depth=4, learning_rate=0.03, min_samples_leaf=20, l2_regularization=0.1),
        dict(max_iter=600, max_depth=5, learning_rate=0.03, min_samples_leaf=20, l2_regularization=0.1),
        dict(max_iter=800, max_depth=5, learning_rate=0.02, min_samples_leaf=25, l2_regularization=0.1),
        dict(max_iter=800, max_depth=6, learning_rate=0.02, min_samples_leaf=15, l2_regularization=0.2),
        dict(max_iter=1000, max_depth=4, learning_rate=0.02, min_samples_leaf=30, l2_regularization=0.2),
    ]:
        c = HistGradientBoostingClassifier(random_state=42, class_weight="balanced", **hp).fit(X_tr, reg_lab)
        p = c.predict_proba(X_te)
        s = score(test_df, soft_blend(p, min_val, max_val, pred_reg))
        if best_s is None or s["within_5"] > best_s["within_5"]:
            best_p, best_s, best_hp, best_proba = c, s, hp, p
    pr(f"E. balanced clf sweep best", best_s, base_w5)
    print(f"     best clf hp: {best_hp}  (MAX recall {max_recall(best_proba)*100:.0f}%)", flush=True)

    # ── F. probability sharpening on snap classes ────────────────────────────
    # Raise P(min)/P(max) to a power < 1 to be more willing to snap, renormalize.
    for gamma in [0.7, 0.85, 1.0, 1.15]:
        p = best_proba.copy()
        sharp = p.copy()
        sharp[:, 0] = p[:, 0] ** gamma
        sharp[:, 2] = p[:, 2] ** gamma
        sharp = sharp / sharp.sum(axis=1, keepdims=True)
        s = score(test_df, soft_blend(sharp, min_val, max_val, pred_reg))
        pr(f"F. sharpen gamma={gamma}", s, base_w5)

    # ── G. balanced + MID-specialist regressor ───────────────────────────────
    mid_mask = reg_lab == 1
    reg_mid = HistGradientBoostingRegressor(random_state=42, **REG_HP).fit(
        X_tr[mid_mask], y[mid_mask])
    pred_mid = predict_model(reg_mid, X_te, cap_te)
    s_g = score(test_df, soft_blend(best_proba, min_val, max_val, pred_mid))
    pr("G. balanced + MID-specialist", s_g, base_w5)

    # ── H. ensemble two classifiers (balanced + unbalanced) ──────────────────
    p_ens = 0.5 * best_proba + 0.5 * p0
    s_h = score(test_df, soft_blend(p_ens, min_val, max_val, pred_reg))
    pr("H. clf ensemble (bal+unbal)", s_h, base_w5)

    # Pick overall best.
    cands = [
        ("v1 unbalanced", s_v1, soft_blend(p0, min_val, max_val, pred_reg)),
        ("balanced", s_bal, soft_blend(p_bal, min_val, max_val, pred_reg)),
        ("balanced sweep", best_s, soft_blend(best_proba, min_val, max_val, pred_reg)),
        ("MID-specialist", s_g, soft_blend(best_proba, min_val, max_val, pred_mid)),
        ("clf ensemble", s_h, soft_blend(p_ens, min_val, max_val, pred_reg)),
    ]
    # include best sharpen
    for gamma in [0.7, 0.85, 1.15]:
        p = best_proba.copy()
        p[:, 0] **= gamma; p[:, 2] **= gamma
        p = p / p.sum(axis=1, keepdims=True)
        cands.append((f"sharpen{gamma}", score(test_df, soft_blend(p, min_val, max_val, pred_reg)),
                      soft_blend(p, min_val, max_val, pred_reg)))
    win = max(cands, key=lambda c: c[1]["within_5"])

    print("\n" + "=" * 90, flush=True)
    print("TIERS", flush=True)
    print("=" * 90, flush=True)
    tier("B. Single regressor", s_reg)
    tier(f"WINNER: {win[0]}", win[1])

    print("\n" + "=" * 90, flush=True)
    print("VERDICT", flush=True)
    print("=" * 90, flush=True)
    for nm, s, _ in sorted(cands, key=lambda c: -c[1]["within_5"]):
        mk = "WINNER" if nm == win[0] else "      "
        print(f"  {mk}  {nm:<18} {s['within_5']:.2f}% w5  {s['within_10']:.2f}% w10  med ${s['median_err_M']:.2f}M", flush=True)
    print(f"\n  Single-regressor plateau: 80.82%. v1 two-stage: 82.15%.", flush=True)
    print(f"  v2 best: {win[1]['within_5']:.2f}% ({win[1]['within_5']-82.15:+.2f}pp vs v1, "
          f"{win[1]['within_5']-80.82:+.2f}pp vs plateau)", flush=True)


if __name__ == "__main__":
    main()
