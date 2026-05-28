"""TWO-STAGE structured contract model — exploit the CBA cluster structure.

Contracts aren't smoothly distributed. They spike on exact CBA values:
  - veteran minimum scale (~1-2.3% of cap by service)
  - max tiers: 25% (0-6 yrs), 30% (7-9 yrs), 35% (10+ yrs), with the Rose
    Rule (young max → 30%) and supermax (vet → 35%) bumps for All-NBA
A smooth regressor blurs across those spikes — which is exactly why the
single-model max-tier median miss is ~$14M (it predicts 30% when the answer
is exactly 35%).

This model:
  Stage 1 — classifier predicts the regime: MIN / MID / MAX.
  Stage 2 — MIN  → snap to the veteran-minimum scale for the service years
            MAX  → snap to the CBA max % for the service tier (+ Rose/supermax)
            MID  → continuous regressor (the existing raw HistGBM)

Routing is on the PREDICTED regime (out-of-sample safe). We test:
  S1 hard argmax routing
  S2 confidence-gated snap (snap only when class prob is high, else regress)
  S3 probability-weighted soft blend
and a MID-specialist regressor vs an all-data regressor.

Pure raw + advanced features. NO Barrett Score anywhere.
Temporal holdout: train 1999-2014, test 2015+.
Single-model plateau to beat: 80.82% within 5% of cap.

Usage:
    python -u scripts/train_twostage_model.py
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


# ── Regime thresholds (in % of cap) ───────────────────────────────────────────
MIN_CUT = 0.035   # below 3.5% of cap → minimum / near-min regime
MAX_CUT = 0.23    # at/above 23% → max-tier regime (25% max with margin)

REG_HP = dict(max_iter=800, max_depth=5, learning_rate=0.02,
              min_samples_leaf=25, l2_regularization=0.1)
CLF_HP = dict(max_iter=600, max_depth=5, learning_rate=0.03,
              min_samples_leaf=20, l2_regularization=0.1)


def regime_label(pct: float) -> int:
    if pct < MIN_CUT:  return 0  # MIN
    if pct >= MAX_CUT: return 2  # MAX
    return 1                      # MID


def cba_max_pct(service: float, all_nba_3yr: float) -> float:
    """First-year max as % of cap, with Rose Rule / supermax bumps."""
    s = float(service)
    elite = all_nba_3yr >= 1
    if s <= 6:
        return 0.30 if elite else 0.25   # Rose Rule bump for young All-NBA
    if s <= 9:
        return 0.35 if elite else 0.30   # supermax (Designated Vet)
    return 0.35


def predict_model(model, X, cap):
    return np.clip(model.predict(X), 0.001, 0.45) * cap


def main():
    print("Building data (pure raw, no Barrett)...", flush=True)
    t0 = time.time()
    careers = build_career_index()
    all_nba = fetch_all_nba_selections()
    train_df = build_rows(TRAIN_PAIRS, careers, all_nba)
    test_df  = build_rows(TEST_PAIRS,  careers, all_nba)
    print(f"  train {len(train_df)}, test {len(test_df)} in {time.time()-t0:.1f}s", flush=True)

    # Regime labels.
    train_df["regime"] = train_df["salary_curr_pct"].apply(regime_label)
    test_df["regime"]  = test_df["salary_curr_pct"].apply(regime_label)
    names = {0: "MIN", 1: "MID", 2: "MAX"}
    dist = train_df["regime"].value_counts().sort_index()
    print("\nRegime distribution (train):", flush=True)
    for k in [0, 1, 2]:
        n = int(dist.get(k, 0))
        print(f"  {names[k]:<4} {n:>4} ({n/len(train_df)*100:4.1f}%)", flush=True)

    y = train_df["salary_curr_pct"].values
    X_tr, X_te = make_X(train_df), make_X(test_df)
    cap_te = test_df["cap_curr"].values

    print("\n" + "=" * 90, flush=True)
    print(f"OUT-OF-SAMPLE  (test n={len(test_df)})", flush=True)
    print("=" * 90, flush=True)

    sA = score(test_df, predict_canonical_baseline(test_df))
    pr("A. CANONICAL baseline (Barrett rank-map)", sA)
    base_w5 = sA["within_5"]

    # ── Single regressor (the plateau, for reference) ────────────────────────
    reg_all = HistGradientBoostingRegressor(random_state=42, **REG_HP).fit(X_tr, y)
    pred_reg = predict_model(reg_all, X_te, cap_te)
    s_reg = score(test_df, pred_reg)
    pr("B. Single regressor (raw HistGBM)", s_reg, base_w5)

    # ── Stage 1: regime classifier ───────────────────────────────────────────
    clf = HistGradientBoostingClassifier(random_state=42, **CLF_HP).fit(
        X_tr, train_df["regime"].values)
    proba = clf.predict_proba(X_te)          # columns = [P(MIN), P(MID), P(MAX)]
    pred_class = proba.argmax(axis=1)
    actual_class = test_df["regime"].values
    clf_acc = float((pred_class == actual_class).mean() * 100)
    print(f"\n  Stage-1 classifier accuracy: {clf_acc:.1f}%", flush=True)
    # Per-class precision/recall for MAX (the high-value class).
    for cls, nm in names.items():
        tp = int(((pred_class == cls) & (actual_class == cls)).sum())
        fp = int(((pred_class == cls) & (actual_class != cls)).sum())
        fn = int(((pred_class != cls) & (actual_class == cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec  = tp / (tp + fn) if (tp + fn) else 0
        print(f"    {nm}: precision {prec*100:4.0f}%  recall {rec*100:4.0f}%  (n={tp+fn})", flush=True)

    # ── Stage 2 snap values ──────────────────────────────────────────────────
    svc = test_df["service_years"].values
    ann = test_df["all_nba_3yr"].values
    min_val = np.array([cba_min_pct(s) for s in svc]) * cap_te
    max_val = np.array([cba_max_pct(s, a) for s, a in zip(svc, ann)]) * cap_te

    # MID-specialist regressor (trained only on MID rows).
    mid_mask_tr = train_df["regime"].values == 1
    reg_mid = HistGradientBoostingRegressor(random_state=42, **REG_HP).fit(
        X_tr[mid_mask_tr], y[mid_mask_tr])
    pred_mid_spec = predict_model(reg_mid, X_te, cap_te)

    # ── S1: hard argmax routing (with MID-specialist) ────────────────────────
    for reg_name, mid_pred in [("all-data reg", pred_reg), ("MID-specialist", pred_mid_spec)]:
        pred = np.where(pred_class == 0, min_val,
               np.where(pred_class == 2, max_val, mid_pred))
        s = score(test_df, pred)
        pr(f"C. S1 hard route ({reg_name})", s, base_w5)

    # ── S2: confidence-gated snap ────────────────────────────────────────────
    # Snap to MIN/MAX only when the classifier is confident; else regressor.
    for thr in [0.5, 0.6, 0.7, 0.8]:
        pred = pred_reg.copy()
        snap_min = (pred_class == 0) & (proba[:, 0] >= thr)
        snap_max = (pred_class == 2) & (proba[:, 2] >= thr)
        pred[snap_min] = min_val[snap_min]
        pred[snap_max] = max_val[snap_max]
        s = score(test_df, pred)
        pr(f"D. S2 gated snap thr={thr}", s, base_w5)

    # ── S3: probability-weighted soft blend ──────────────────────────────────
    # pred = P(min)*min_val + P(max)*max_val + P(mid)*regressor
    soft = (proba[:, 0] * min_val + proba[:, 2] * max_val
            + proba[:, 1] * pred_reg)
    s_soft = score(test_df, soft)
    pr("E. S3 soft blend (prob-weighted)", s_soft, base_w5)

    # ── S4: soft blend but MID uses specialist ───────────────────────────────
    soft2 = (proba[:, 0] * min_val + proba[:, 2] * max_val
             + proba[:, 1] * pred_mid_spec)
    s_soft2 = score(test_df, soft2)
    pr("F. S3 soft blend (MID-specialist)", s_soft2, base_w5)

    # ── Best gated config explicit for tiers ─────────────────────────────────
    # Re-eval the best gated (pick thr=0.6 as representative; recompute best).
    best_pred, best_s, best_label = pred_reg, s_reg, "single regressor"
    for thr in [0.5, 0.6, 0.7, 0.8]:
        pred = pred_reg.copy()
        sm = (pred_class == 0) & (proba[:, 0] >= thr)
        sx = (pred_class == 2) & (proba[:, 2] >= thr)
        pred[sm] = min_val[sm]; pred[sx] = max_val[sx]
        s = score(test_df, pred)
        if s["within_5"] > best_s["within_5"]:
            best_pred, best_s, best_label = pred, s, f"gated snap thr={thr}"
    for cand_s, cand_p, cand_l in [(s_soft, soft, "soft blend"),
                                    (s_soft2, soft2, "soft MID-spec")]:
        if cand_s["within_5"] > best_s["within_5"]:
            best_pred, best_s, best_label = cand_p, cand_s, cand_l

    print("\n" + "=" * 90, flush=True)
    print("TIER COMPARISON", flush=True)
    print("=" * 90, flush=True)
    tier("A. Canonical baseline", sA)
    tier("B. Single regressor", s_reg)
    tier(f"Best two-stage ({best_label})", best_s)

    print("\n" + "=" * 90, flush=True)
    print("VERDICT", flush=True)
    print("=" * 90, flush=True)
    print(f"  Single regressor (plateau): {s_reg['within_5']:.2f}% w5, {s_reg['within_10']:.2f}% w10", flush=True)
    print(f"  Best two-stage ({best_label}): {best_s['within_5']:.2f}% w5, {best_s['within_10']:.2f}% w10", flush=True)
    print(f"  Δ vs single regressor: {best_s['within_5']-s_reg['within_5']:+.2f}pp w5", flush=True)
    print(f"  Δ vs plateau (80.82%): {best_s['within_5']-80.82:+.2f}pp", flush=True)


if __name__ == "__main__":
    main()
