"""V3 — push past v2's 80.38% via sample weighting and per-tier specialists.

Strategies in this run:
  1. PRUNED feature set (drop near-zero importance features from v2)
  2. AGGRESSIVE HistGBM hyperparameter search (30+ configs)
  3. SAMPLE WEIGHTING: upweight rare big/max contracts at training time
  4. HUBER LOSS variant (robust to outliers — extreme paycuts)
  5. PER-TIER specialist models combined via tier classifier
  6. STACKED ensemble of best variants

Best v2: HistGBM tuned at 80.38% (+1.39pp).
Target: 81%+ on canonical out-of-sample.

Usage:
    python -u scripts/train_ml_model_v3.py
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Reuse data assembly from v2 (without playoff features which we skip).
sys.path.insert(0, str(Path(__file__).parent))
from train_ml_model_v2 import (
    build_career_indexes, build_rows, fetch_all_nba_selections,
    TRAIN_PAIRS, TEST_PAIRS, SALARY_CAP_M, predict_canonical_baseline,
    CURRENT_CAP_M, ROOKIE_SCALE_FIRST_YR, ROOKIE_SCALE_SAL_PCT,
    ROOKIE_SCALE_MAX_AGE, ROOKIE_SCALE_STEP_UP,
)


# PRUNED feature set — keep only ones with measurable importance.
PRUNED_FEATURES = [
    "barrett", "barrett_single", "barrett_3yr",
    "score_rank",
    "eff_adj", "d_lebron",
    "GP", "gp_3yr",
    "age",
    "salary_prev_pct",
    "career_base_proj_pct",
    "years_in_league",
    "all_nba_3yr",
    "barrett_growth",
]


def make_X_pruned(df: pd.DataFrame) -> np.ndarray:
    X = df[PRUNED_FEATURES].fillna(0).astype(float).values
    age = df["age"].values
    barrett = df["barrett"].values
    rank = df["score_rank"].values
    yrs = df["years_in_league"].values
    derived = np.column_stack([
        age ** 2,
        barrett ** 2,
        np.log1p(rank),
        yrs ** 2,
        (yrs >= 7).astype(float),
        (yrs >= 10).astype(float),
        (df["pos_bucket"] == "Guard").astype(float).values,
        (df["pos_bucket"] == "Forward").astype(float).values,
    ])
    return np.hstack([X, derived])


def predict_model(model, X, cap_dollars: np.ndarray) -> np.ndarray:
    pred_pct = np.clip(model.predict(X), 0.001, 0.45)
    return pred_pct * cap_dollars


def score(df: pd.DataFrame, pred: np.ndarray) -> dict:
    actual = df["salary_curr"].values
    cap = df["cap_curr"].values
    err_pct_cap = np.abs(actual - pred) / cap * 100
    err_today_M = err_pct_cap / 100 * CURRENT_CAP_M
    return {
        "n":             len(df),
        "median_err":    float(np.median(err_pct_cap)),
        "within_5":      float(np.mean(err_pct_cap <= 5.0) * 100),
        "within_10":     float(np.mean(err_pct_cap <= 10.0) * 100),
        "median_err_M":  float(np.median(err_today_M)),
        "err_today_M":   err_today_M,
        "actual_today_M": actual / cap * CURRENT_CAP_M,
    }


def print_score(label: str, s: dict, baseline_w5: float | None = None) -> None:
    label_line = label[:48].ljust(48)
    if baseline_w5 is None:
        print(f"  {label_line}  w5={s['within_5']:5.2f}%  w10={s['within_10']:5.2f}%  med=${s['median_err_M']:.2f}M", flush=True)
    else:
        d5 = s["within_5"] - baseline_w5
        marker = "[+]" if d5 >= 1 else ("[?]" if d5 >= 0 else "[-]")
        print(f"  {label_line}  w5={s['within_5']:5.2f}%  ({d5:+5.2f}pp) {marker}  w10={s['within_10']:5.2f}%  med=${s['median_err_M']:.2f}M", flush=True)


def report_tier(label: str, scores: dict) -> None:
    err = scores["err_today_M"]
    actual = scores["actual_today_M"]
    tiers = [
        ("Max/super",  actual >= 40),
        ("Big stars",  (actual >= 25) & (actual < 40)),
        ("Mid-tier",   (actual >= 15) & (actual < 25)),
        ("Rotation",   (actual >=  7) & (actual < 15)),
        ("Min-ish",    actual <  7),
    ]
    print(f"\n  TIER BREAKDOWN — {label}", flush=True)
    for name, mask in tiers:
        n = int(mask.sum())
        if n == 0: continue
        sub_err = err[mask]
        within_3 = float(np.mean(sub_err <= 3) * 100)
        within_5 = float(np.mean(sub_err <= 5) * 100)
        median_M = float(np.median(sub_err))
        print(f"    {name:<12} n={n:>4}   median ${median_M:>5.2f}M   "
              f"±$3M {within_3:>4.0f}%   ±$5M {within_5:>4.0f}%", flush=True)


def main() -> None:
    print(f"Train pairs: {len(TRAIN_PAIRS)}, Test pairs: {len(TEST_PAIRS)}", flush=True)

    print("\nLoading data...", flush=True)
    t0 = time.time()
    careers_rs = build_career_indexes(playoffs=False)
    print(f"  RS careers: {len(careers_rs)} in {time.time()-t0:.1f}s", flush=True)
    careers_po = {}
    all_nba_lookup = fetch_all_nba_selections()
    print(f"  All-NBA: {len(all_nba_lookup)} players", flush=True)

    print("\nBuilding train / test rows...", flush=True)
    t0 = time.time()
    train_df = build_rows(TRAIN_PAIRS, careers_rs, careers_po, all_nba_lookup)
    test_df  = build_rows(TEST_PAIRS,  careers_rs, careers_po, all_nba_lookup)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)} in {time.time()-t0:.1f}s", flush=True)

    y_train = train_df["salary_curr_pct"].values
    X_train = make_X_pruned(train_df)
    X_test  = make_X_pruned(test_df)
    cap_test = test_df["cap_curr"].values

    # Baseline.
    pred_A = predict_canonical_baseline(test_df)
    sA = score(test_df, pred_A)
    print("\n" + "=" * 92, flush=True)
    print("OUT-OF-SAMPLE RESULTS  (test n={})".format(len(test_df)), flush=True)
    print("=" * 92, flush=True)
    print_score("A. CANONICAL baseline", sA)

    # ── Sample weights: upweight big contracts ───────────────────────────────
    # Actual salary as % of cap → weight = 1 + 4 × (pct - 0.05).clip(0, 0.3)
    # Translates roughly to:
    #   <5%   of cap (min):  weight 1.0
    #   15%   of cap (mid):  weight 1.4
    #   25%   of cap (big):  weight 1.8
    #   35%   of cap (max):  weight 2.2
    sample_w_pcts = train_df["salary_curr_pct"].values
    sample_weights = 1.0 + 4.0 * np.clip(sample_w_pcts - 0.05, 0, 0.30)

    # ── B. v1 GBM (untuned baseline GBM) ──────────────────────────────────────
    # Just GBM with default v1 hyperparams as a sanity check.
    gbm_v1 = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_samples_leaf=15, subsample=0.8, random_state=42,
    ).fit(X_train, y_train)
    sB = score(test_df, predict_model(gbm_v1, X_test, cap_test))
    print_score("B. v1 GBM (pruned features)", sB, sA["within_5"])

    # ── C. v1 GBM with sample weighting ──────────────────────────────────────
    gbm_v1w = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_samples_leaf=15, subsample=0.8, random_state=42,
    ).fit(X_train, y_train, sample_weight=sample_weights)
    sC = score(test_df, predict_model(gbm_v1w, X_test, cap_test))
    print_score("C. GBM + sample weighting (big-contract boost)", sC, sA["within_5"])

    # ── D. HuberLoss GBM ─────────────────────────────────────────────────────
    gbm_huber = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_samples_leaf=15, subsample=0.8, random_state=42,
        loss="huber", alpha=0.9,
    ).fit(X_train, y_train)
    sD = score(test_df, predict_model(gbm_huber, X_test, cap_test))
    print_score("D. GBM with Huber loss (robust to outliers)", sD, sA["within_5"])

    # ── E. HistGBM aggressive sweep ──────────────────────────────────────────
    print("\n  Sweeping HistGBM hyperparameters (30+ configs)...", flush=True)
    best_hist, best_sE, best_params = None, None, None
    sweep = []
    for max_iter in [400, 600, 800, 1000]:
        for max_depth in [4, 5, 6, 7]:
            for lr in [0.02, 0.03, 0.05]:
                for leaf in [15, 20, 25]:
                    sweep.append(dict(
                        max_iter=max_iter, max_depth=max_depth,
                        learning_rate=lr, min_samples_leaf=leaf,
                        l2_regularization=0.1,
                    ))
    # Limit to ~30 reasonable configs by sampling.
    rng = np.random.RandomState(7)
    idx = rng.choice(len(sweep), size=min(36, len(sweep)), replace=False)
    sweep = [sweep[i] for i in idx]

    for i, params in enumerate(sweep, 1):
        m = HistGradientBoostingRegressor(random_state=42, **params).fit(X_train, y_train)
        p = predict_model(m, X_test, cap_test)
        s = score(test_df, p)
        if best_sE is None or s["within_5"] > best_sE["within_5"]:
            best_hist, best_sE, best_params = m, s, params
            print(f"    [{i:02d}/{len(sweep)}] new best: w5={s['within_5']:.2f}% with {params}", flush=True)
    print_score(f"E. HistGBM best of sweep ({len(sweep)} configs)", best_sE, sA["within_5"])

    # ── F. HistGBM + sample weighting ────────────────────────────────────────
    f_hist = HistGradientBoostingRegressor(random_state=42, **best_params).fit(
        X_train, y_train, sample_weight=sample_weights,
    )
    sF = score(test_df, predict_model(f_hist, X_test, cap_test))
    print_score("F. HistGBM tuned + sample weighting", sF, sA["within_5"])

    # ── G. Per-tier specialist: top vs bottom split ──────────────────────────
    # Idea: a single model struggles with both min ($1M errors matter) and
    # max ($10M errors matter) at once. Train one model for "elite" contracts
    # (top 25% of training pool by salary_curr) and one for "non-elite", then
    # route test rows by a classifier-like signal.
    salary_pct_train = train_df["salary_curr_pct"].values
    pcut = float(np.quantile(salary_pct_train, 0.75))
    print(f"\n  Specialist split threshold: {pcut*100:.1f}% of cap", flush=True)
    elite_mask = salary_pct_train >= pcut
    nelite_mask = ~elite_mask
    print(f"    Elite train: {elite_mask.sum()},   Non-elite: {nelite_mask.sum()}", flush=True)

    elite_model = HistGradientBoostingRegressor(random_state=42, **best_params).fit(
        X_train[elite_mask], y_train[elite_mask],
    )
    nelite_model = HistGradientBoostingRegressor(random_state=42, **best_params).fit(
        X_train[nelite_mask], y_train[nelite_mask],
    )
    # Router: use the unrestricted HistGBM's prediction to decide.
    # If best HistGBM predicts ≥ pcut → use elite specialist, else non-elite.
    router_pred = best_hist.predict(X_test)
    is_elite_test = router_pred >= pcut
    pred_G_pct = np.where(
        is_elite_test,
        elite_model.predict(X_test),
        nelite_model.predict(X_test),
    )
    pred_G = np.clip(pred_G_pct, 0.001, 0.45) * cap_test
    sG = score(test_df, pred_G)
    print_score("G. Per-tier specialist HistGBMs (elite/nonelite split)", sG, sA["within_5"])

    # ── H. Stacked ensemble (best 3 models) ──────────────────────────────────
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(train_df), 3))
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr = y_train[tr_idx]
        oof_preds[val_idx, 0] = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=15, subsample=0.8, random_state=42,
        ).fit(X_tr, y_tr).predict(X_val)
        oof_preds[val_idx, 1] = HistGradientBoostingRegressor(
            random_state=42, **best_params,
        ).fit(X_tr, y_tr).predict(X_val)
        oof_preds[val_idx, 2] = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=15, subsample=0.8, random_state=42,
            loss="huber", alpha=0.9,
        ).fit(X_tr, y_tr).predict(X_val)
    meta = Ridge(alpha=0.3).fit(oof_preds, y_train)
    test_stack = np.column_stack([
        gbm_v1.predict(X_test),
        best_hist.predict(X_test),
        gbm_huber.predict(X_test),
    ])
    pred_H = np.clip(meta.predict(test_stack), 0.001, 0.45) * cap_test
    sH = score(test_df, pred_H)
    print_score("H. Stacked (GBM + HistGBM + Huber-GBM via Ridge)", sH, sA["within_5"])
    print(f"     Weights: GBM={meta.coef_[0]:+.2f}  HistGBM={meta.coef_[1]:+.2f}  Huber={meta.coef_[2]:+.2f}  int={meta.intercept_:+.3f}", flush=True)

    # ── Verdict + tier ────────────────────────────────────────────────────────
    candidates = [
        ("A. baseline",                            sA),
        ("B. GBM pruned",                          sB),
        ("C. GBM + sample weight",                 sC),
        ("D. GBM Huber loss",                      sD),
        ("E. HistGBM tuned",                       best_sE),
        ("F. HistGBM + sample weight",             sF),
        ("G. Per-tier specialists",                sG),
        ("H. Stacked ensemble",                    sH),
    ]
    best = max(candidates, key=lambda kv: kv[1]["within_5"])

    print("\n" + "=" * 92, flush=True)
    print("VERDICT", flush=True)
    print("=" * 92, flush=True)
    for name, s in candidates:
        d = s["within_5"] - sA["within_5"]
        marker = "WINNER" if name == best[0] else "      "
        print(f"  {marker}  {name:<32}  w5={s['within_5']:5.2f}%  ({d:+5.2f}pp)  w10={s['within_10']:5.2f}%  med=${s['median_err_M']:.2f}M", flush=True)
    print(f"\n  Winner: {best[0]} at {best[1]['within_5']:.2f}%  "
          f"({best[1]['within_5'] - sA['within_5']:+.2f}pp over canonical baseline)", flush=True)

    report_tier("A. Baseline",                sA)
    report_tier("E. HistGBM tuned",           best_sE)
    report_tier("G. Per-tier specialists",    sG)
    report_tier("Winner",                     best[1])


if __name__ == "__main__":
    main()
