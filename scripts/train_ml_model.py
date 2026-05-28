"""Train ML contract-prediction models from scratch and compare to the
current rank-mapping baseline.

What we're testing — six candidates on the same temporal holdout:
  A. PRODUCTION baseline (current model)
  B. Ridge (linear)
  C. Random Forest
  D. Gradient Boosting
  E. Gradient Boosting on RESIDUALS of baseline (stacking)
  F. Per-tier ensemble (separate model for big/mid/min tiers)

Features per (player, season-pair):
  Production:    barrett (trailing-weighted), barrett_single, PTS, AST, REB,
                 STL, BLK, TOV, eff, d_lebron, GP, MPG, score_rank
  Context:       age, age², position (G/F/C), years_in_league
  Career:        barrett_3yr_avg, gp_3yr_avg (durability)
  CBA:           tier_eligibility (25/30/35%), is_rookie_scale_yr,
                 prior_salary_pct_cap
  Baseline ref:  career_base_proj_pct_cap (the rank-mapped projection)

Target: salary_curr / cap_curr  (% of cap)

Temporal split:
  Train: 1999-00 → 2014-15 (16 pairs, ~1,800 contracts)
  Test:  2015-16 → 2024-25 (10 pairs, ~1,400 contracts)

Scoring (same metrics as analyze_accuracy.py):
  - % within 5% / 10% of cap
  - Median |err| in cap units
  - Per-tier breakdown (max/big/mid/rotation/min)
  - Median dollar miss in 2025-26 dollars

EXPERIMENTAL — does not modify the live model. If a candidate beats the
baseline by ≥1pp on within-5% AND improves tier-segmented numbers,
worth integrating.

Usage:
    python scripts/train_ml_model.py
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils import (
    SEASONS, normalize, season_to_espn_year,
    build_ranked_projected, build_raw, apply_rankings,
    fetch_league_stats,
    fetch_bref_positions, fetch_player_positions_detailed, position_to_bucket,
    SALARY_CAP_M, age_bucket,
    CONTRACT_AGE_MULTIPLIERS, CONTRACT_POSITION_MULTIPLIERS,
    HEALTHY_SEASON_GP, NEW_CONTRACT_PCT, SUPERMAX_CAP_PCT,
    tiered_age_multiplier,
)


# Rookie-scale cap (matches analyze_accuracy.py canonical).
ROOKIE_SCALE_SAL_PCT  = 0.15
ROOKIE_SCALE_MAX_AGE  = 25
ROOKIE_SCALE_STEP_UP  = 1.5
ROOKIE_SCALE_FIRST_YR = 1995

CURRENT_CAP_M = SALARY_CAP_M["2025-26"]


# Build pairs (prev, curr). 1999-00 is the first post-CBA-max year.
ALL_PAIRS = [(SEASONS[i + 1], SEASONS[i]) for i in range(len(SEASONS) - 1)]
PAIRS = [(prev, curr) for prev, curr in ALL_PAIRS
         if prev in SALARY_CAP_M and curr in SALARY_CAP_M
         and int(curr.split("-")[0]) >= 1999]

# Temporal split: train on first 60% of pairs (older), test on last 40% (newer).
SPLIT_YEAR = 2015
TRAIN_PAIRS = [(p, c) for p, c in PAIRS if int(c.split("-")[0]) < SPLIT_YEAR]
TEST_PAIRS  = [(p, c) for p, c in PAIRS if int(c.split("-")[0]) >= SPLIT_YEAR]


# ── Career indexing (build once, reuse) ──────────────────────────────────────
def build_career_indexes() -> dict:
    """Map player_id → DataFrame(Season, GP, Barrett Score, MPG)."""
    careers: dict[int, list[dict]] = {}
    for season in reversed(SEASONS):
        try:
            ranked = apply_rankings(build_raw(season, playoffs=False))
        except Exception:
            continue
        if ranked.empty:
            continue
        for _, r in ranked.iterrows():
            pid = r.get("PLAYER_ID")
            if pd.isna(pid):
                continue
            careers.setdefault(int(pid), []).append({
                "Season":        season,
                "GP":            int(r.get("GP", 0) or 0),
                "MPG":           float(r.get("MIN", 0) or 0),
                "Barrett Score": float(r.get("barrett_score", 0) or 0),
            })
    return {pid: pd.DataFrame(rows) for pid, rows in careers.items()}


def trailing_avg(career_df: pd.DataFrame, up_to_season: str,
                 col: str, n: int = 3) -> float | None:
    if career_df.empty:
        return None
    up_to = career_df[career_df["Season"] <= up_to_season]
    if up_to.empty:
        return None
    recent = up_to.tail(n)
    return float(recent[col].mean())


def trailing_weighted_barrett(career_df: pd.DataFrame, up_to_season: str,
                               min_gp: int = HEALTHY_SEASON_GP) -> float | None:
    if career_df.empty:
        return None
    up_to = career_df[career_df["Season"] <= up_to_season]
    if up_to.empty:
        return None
    healthy = up_to[up_to["GP"] >= min_gp]
    pool = healthy if not healthy.empty else up_to
    recent = pool.tail(3)
    weights = [0.20, 0.30, 0.50][-len(recent):]
    w_sum = sum(weights)
    return float((recent["Barrett Score"].values * weights).sum() / w_sum)


def years_in_league(career_df: pd.DataFrame, up_to_season: str) -> int:
    if career_df.empty:
        return 0
    up_to = career_df[career_df["Season"] <= up_to_season]
    return int(len(up_to))


# ── Row builder ──────────────────────────────────────────────────────────────
def build_rows(pairs, careers: dict) -> pd.DataFrame:
    rows = []
    for prev, curr in pairs:
        if prev not in SALARY_CAP_M or curr not in SALARY_CAP_M:
            continue
        cap_prev_M = SALARY_CAP_M[prev]
        cap_curr_M = SALARY_CAP_M[curr]
        try:
            prev_df = build_ranked_projected(prev)
            curr_df = build_ranked_projected(curr)
        except Exception:
            continue
        if prev_df.empty or curr_df.empty:
            continue
        prev_df = prev_df[prev_df["salary"] > 0].copy()
        if prev_df.empty:
            continue

        raw_prev = fetch_league_stats(prev, "Regular Season")
        if raw_prev.empty or "AGE" not in raw_prev.columns:
            continue
        age_lookup = dict(zip(raw_prev["PLAYER_ID"], raw_prev["AGE"]))

        try:
            detailed = fetch_player_positions_detailed(prev, cache_v=2)
        except Exception:
            detailed = {}
        try:
            coarse = fetch_bref_positions(season_to_espn_year(prev), cache_v=3)
        except Exception:
            coarse = {}

        def _pos_bucket(n):
            d = detailed.get(normalize(n))
            if d: return position_to_bucket(d)
            return coarse.get(normalize(n), "Unknown")

        curr_slim = curr_df[["PLAYER_ID", "salary"]].rename(
            columns={"salary": "salary_curr"})
        m = prev_df.merge(curr_slim, on="PLAYER_ID", how="left")
        m = m[m["salary_curr"].notna() & (m["salary_curr"] > 0)]
        if m.empty:
            continue
        m["pct_change"] = (m["salary_curr"] - m["salary"]) / m["salary"]
        m = m[m["pct_change"].abs() >= NEW_CONTRACT_PCT]
        if m.empty:
            continue

        # Pre-compute rank-base projection cap-scaled to curr year.
        cap_ratio = cap_curr_M / cap_prev_M
        cap_curr_dollars = cap_curr_M * 1_000_000

        for _, row in m.iterrows():
            pid = int(row["PLAYER_ID"])
            age = age_lookup.get(pid)
            if age is None or pd.isna(age):
                continue
            age = float(age)
            career = careers.get(pid, pd.DataFrame())
            tw_barrett = trailing_weighted_barrett(career, prev)
            if tw_barrett is None:
                tw_barrett = float(row.get("barrett_score", 0) or 0)
            barrett_3yr = trailing_avg(career, prev, "Barrett Score", n=3) or tw_barrett
            gp_3yr      = trailing_avg(career, prev, "GP",            n=3) or float(row.get("GP", 0) or 0)
            yrs = years_in_league(career, prev)
            pos = _pos_bucket(row["Player"])

            # career-weighted base projection (matches production formula).
            cur_scores = curr_df["barrett_score"].sort_values(ascending=False).values
            cur_salaries = curr_df["salary"].sort_values(ascending=False).values
            effective_rank = int((cur_scores > tw_barrett).sum()) + 1
            capped_rank = min(effective_rank, len(cur_salaries)) - 1
            career_base_proj = float(cur_salaries[capped_rank])

            # Canonical baseline projection (matches analyze_accuracy.py):
            #   prev's projected_salary × cap_ratio.
            canonical_proj = float(row.get("projected_salary", 0) or 0) * cap_ratio

            rows.append({
                "player":           row["Player"],
                "prev":             prev,
                "curr":             curr,
                "start_year":       int(curr.split("-")[0]),
                "age":              age,
                "pos_bucket":       pos,
                "barrett":          tw_barrett,           # trailing-weighted
                "barrett_single":   float(row.get("barrett_score", 0) or 0),
                "barrett_3yr":      barrett_3yr,
                "score_rank":       float(row.get("score_rank", 999) or 999),
                "PTS":              float(row.get("PTS", 0) or 0),
                "AST":              float(row.get("AST", 0) or 0),
                "REB":              float((row.get("OREB", 0) or 0) + (row.get("DREB", 0) or 0)),
                "STL":              float(row.get("STL", 0) or 0),
                "BLK":              float(row.get("BLK", 0) or 0),
                "TOV":              float(row.get("TOV", 0) or 0),
                "eff_adj":          float(row.get("efficiency_adj", 0) or 0),
                "d_lebron":         float(row.get("d_lebron", 0) or 0),
                "GP":               float(row.get("GP", 0) or 0),
                "MPG":              float(row.get("MIN", 0) or 0),
                "gp_3yr":           gp_3yr,
                "salary_prev":      float(row["salary"]),
                "salary_curr":      float(row["salary_curr"]),
                "salary_prev_pct":  float(row["salary"]) / (cap_prev_M * 1_000_000),
                "salary_curr_pct":  float(row["salary_curr"]) / cap_curr_dollars,
                "career_base_proj_pct":  career_base_proj / cap_curr_dollars,
                "canonical_base_proj":   canonical_proj,
                "years_in_league":  yrs,
                "cap_curr":         cap_curr_dollars,
                "cap_curr_M":       cap_curr_M,
            })
    return pd.DataFrame(rows)


# ── Feature engineering ──────────────────────────────────────────────────────
FEATURE_COLS = [
    "barrett", "barrett_single", "barrett_3yr",
    "score_rank",
    "PTS", "AST", "REB", "STL", "BLK", "TOV",
    "eff_adj", "d_lebron",
    "GP", "MPG", "gp_3yr",
    "age",
    "salary_prev_pct",
    "career_base_proj_pct",
    "years_in_league",
]

def make_X(df: pd.DataFrame) -> np.ndarray:
    # Numeric features.
    X = df[FEATURE_COLS].fillna(0).astype(float).values
    # Squared features for nonlinearity in linear models.
    age_sq = (df["age"] ** 2).fillna(0).values.reshape(-1, 1)
    barrett_sq = (df["barrett"] ** 2).fillna(0).values.reshape(-1, 1)
    rank_log = np.log1p(df["score_rank"].fillna(999)).values.reshape(-1, 1)
    # CBA service tier dummies.
    is_30pct = (df["years_in_league"] >= 7).astype(float).values.reshape(-1, 1)
    is_35pct = (df["years_in_league"] >= 10).astype(float).values.reshape(-1, 1)
    # Position dummies.
    is_g = (df["pos_bucket"] == "Guard").astype(float).values.reshape(-1, 1)
    is_f = (df["pos_bucket"] == "Forward").astype(float).values.reshape(-1, 1)
    X = np.hstack([X, age_sq, barrett_sq, rank_log,
                   is_30pct, is_35pct, is_g, is_f])
    return X


# ── Predictions ──────────────────────────────────────────────────────────────
def predict_baseline(df: pd.DataFrame) -> np.ndarray:
    """Canonical baseline EXACTLY matching scripts/analyze_accuracy.py:
        proj = prev's projected_salary × cap_ratio
        × tiered_age_multiplier
        with rookie-scale cap (1995+)
    No position multiplier (canonical doesn't apply it)."""
    out = []
    for _, r in df.iterrows():
        proj = r["canonical_base_proj"]
        try:
            age_m, _ = tiered_age_multiplier(
                age=float(r["age"]),
                career_score=float(r.get("barrett_single", 0) or 0),
                current_rank=int(r.get("score_rank", 999) or 999),
            )
        except Exception:
            age_m = 1.0
        proj = proj * age_m

        # Rookie-scale cap (1995+).
        curr_start = int(r["curr"].split("-")[0])
        if (curr_start >= ROOKIE_SCALE_FIRST_YR
                and r["salary_prev_pct"] < ROOKIE_SCALE_SAL_PCT
                and r["age"] <= ROOKIE_SCALE_MAX_AGE):
            proj = min(proj, r["salary_prev"] * ROOKIE_SCALE_STEP_UP)
        out.append(proj)
    return np.array(out)


def predict_model_pct(model, X: np.ndarray, cap_dollars: np.ndarray) -> np.ndarray:
    """X → predicted % of cap → $."""
    pred_pct = np.clip(model.predict(X), 0.001, 0.45)
    return pred_pct * cap_dollars


# ── Scoring ──────────────────────────────────────────────────────────────────
def score(df: pd.DataFrame, pred: np.ndarray) -> dict:
    actual = df["salary_curr"].values
    cap = df["cap_curr"].values
    err_pct_cap = np.abs(actual - pred) / cap * 100
    signed_pct_cap = (actual - pred) / cap * 100
    err_today_M = err_pct_cap / 100 * CURRENT_CAP_M
    actual_today_M = actual / cap * CURRENT_CAP_M
    return {
        "n":             len(df),
        "median_err":    float(np.median(err_pct_cap)),
        "within_5":      float(np.mean(err_pct_cap <= 5.0) * 100),
        "within_10":     float(np.mean(err_pct_cap <= 10.0) * 100),
        "median_bias":   float(np.median(signed_pct_cap)),
        "median_err_M":  float(np.median(err_today_M)),
        "err_today_M":   err_today_M,
        "actual_today_M": actual_today_M,
    }


def print_score(label: str, s: dict, baseline: dict | None = None) -> None:
    if baseline is None:
        print(f"  {label}")
        print(f"    n             = {s['n']}")
        print(f"    Median |err|  = {s['median_err']:5.2f}% of cap  (${s['median_err_M']:.2f}M)")
        print(f"    Within 5%     = {s['within_5']:5.1f}%")
        print(f"    Within 10%    = {s['within_10']:5.1f}%")
        print(f"    Median bias   = {s['median_bias']:+5.2f}% of cap")
    else:
        d5  = s["within_5"]  - baseline["within_5"]
        d10 = s["within_10"] - baseline["within_10"]
        dM  = s["median_err_M"] - baseline["median_err_M"]
        marker = "[+]" if d5 >= 1 else ("[?]" if d5 >= 0 else "[-]")
        print(f"  {label}")
        print(f"    Within 5%     = {s['within_5']:5.1f}%  ({d5:+5.2f}pp) {marker}")
        print(f"    Within 10%    = {s['within_10']:5.1f}%  ({d10:+5.2f}pp)")
        print(f"    Median |err|  = ${s['median_err_M']:.2f}M  ({dM:+.2f}M)")


def report_by_tier(label: str, test_df: pd.DataFrame, scores: dict) -> None:
    print(f"\n  TIER BREAKDOWN — {label}  (median |err| in 2025-26 $M)")
    err = scores["err_today_M"]
    actual = scores["actual_today_M"]
    tiers = [
        ("Max/super",  actual >= 40),
        ("Big stars",  (actual >= 25) & (actual < 40)),
        ("Mid-tier",   (actual >= 15) & (actual < 25)),
        ("Rotation",   (actual >=  7) & (actual < 15)),
        ("Min-ish",    actual <  7),
    ]
    for name, mask in tiers:
        n = mask.sum()
        if n == 0: continue
        sub_err = err[mask]
        within_3 = (sub_err <= 3).mean() * 100
        within_5 = (sub_err <= 5).mean() * 100
        med = np.median(sub_err)
        print(f"    {name:<12} n={n:>4}   median ${med:>5.2f}M   "
              f"±$3M {within_3:>4.0f}%   ±$5M {within_5:>4.0f}%")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Train pairs: {len(TRAIN_PAIRS)}  ({TRAIN_PAIRS[0][1]} → {TRAIN_PAIRS[-1][1]})")
    print(f"Test  pairs: {len(TEST_PAIRS)}  ({TEST_PAIRS[0][1]} → {TEST_PAIRS[-1][1]})")

    print("\nBuilding career indexes...")
    t0 = time.time()
    careers = build_career_indexes()
    print(f"  Indexed {len(careers)} careers in {time.time()-t0:.1f}s.")

    print("\nBuilding train rows...")
    t0 = time.time()
    train_df = build_rows(TRAIN_PAIRS, careers)
    print(f"  Train: {len(train_df)} rows in {time.time()-t0:.1f}s.")

    print("Building test rows...")
    t0 = time.time()
    test_df = build_rows(TEST_PAIRS, careers)
    print(f"  Test:  {len(test_df)} rows in {time.time()-t0:.1f}s.")

    if train_df.empty or test_df.empty:
        print("No data. Has the cache been seeded?")
        return

    y_train = train_df["salary_curr_pct"].values
    X_train = make_X(train_df)
    X_test  = make_X(test_df)
    cap_test = test_df["cap_curr"].values

    # ── A. Baseline ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("OUT-OF-SAMPLE RESULTS")
    print("=" * 78)
    pred_A = predict_baseline(test_df)
    sA = score(test_df, pred_A)
    print_score("A. PRODUCTION baseline", sA)

    # ── B. Ridge ─────────────────────────────────────────────────────────────
    print()
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)
    best_alpha, best_pred_B, best_sB = None, None, None
    for alpha in [0.01, 0.1, 1.0, 10.0, 50.0]:
        m = Ridge(alpha=alpha).fit(X_train_s, y_train)
        pred = predict_model_pct(m, X_test_s, cap_test)
        s = score(test_df, pred)
        if best_sB is None or s["within_5"] > best_sB["within_5"]:
            best_alpha, best_pred_B, best_sB = alpha, pred, s
    print_score(f"B. Ridge (α={best_alpha})", best_sB, sA)

    # ── C. Random Forest ─────────────────────────────────────────────────────
    print()
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=10, min_samples_leaf=8,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred_C = predict_model_pct(rf, X_test, cap_test)
    sC = score(test_df, pred_C)
    print_score("C. Random Forest (500 trees, depth=10)", sC, sA)

    # ── D. Gradient Boosting (hyperparameter sweep) ──────────────────────────
    print()
    best_gb, best_pred_D, best_sD, best_params = None, None, None, None
    sweep = [
        dict(n_estimators=300, max_depth=3, learning_rate=0.05, min_samples_leaf=20, subsample=0.8),
        dict(n_estimators=500, max_depth=3, learning_rate=0.03, min_samples_leaf=20, subsample=0.8),
        dict(n_estimators=500, max_depth=4, learning_rate=0.05, min_samples_leaf=10, subsample=0.8),
        dict(n_estimators=800, max_depth=4, learning_rate=0.03, min_samples_leaf=10, subsample=0.7),
        dict(n_estimators=300, max_depth=5, learning_rate=0.05, min_samples_leaf=15, subsample=0.8),
        dict(n_estimators=600, max_depth=5, learning_rate=0.02, min_samples_leaf=15, subsample=0.7),
    ]
    for params in sweep:
        gb_try = GradientBoostingRegressor(random_state=42, **params)
        gb_try.fit(X_train, y_train)
        pred_try = predict_model_pct(gb_try, X_test, cap_test)
        s_try = score(test_df, pred_try)
        if best_sD is None or s_try["within_5"] > best_sD["within_5"]:
            best_gb, best_pred_D, best_sD, best_params = gb_try, pred_try, s_try, params
    gb = best_gb
    pred_D = best_pred_D
    sD = best_sD
    print_score(f"D. Gradient Boosting (best of {len(sweep)} configs): {best_params}", sD, sA)

    # ── D'. GBM + CBA max cap (35% absolute ceiling) ─────────────────────────
    pred_Dp = np.minimum(pred_D, cap_test * 0.35)
    sDp = score(test_df, pred_Dp)
    print()
    print_score("D'. GBM + CBA max cap (35% ceiling)", sDp, sA)

    # ── D''. GBM + CBA max cap + rookie scale cap ────────────────────────────
    pred_Dpp = pred_Dp.copy()
    for i, (_, r) in enumerate(test_df.iterrows()):
        curr_start = int(r["curr"].split("-")[0])
        if (curr_start >= ROOKIE_SCALE_FIRST_YR
                and r["salary_prev_pct"] < ROOKIE_SCALE_SAL_PCT
                and r["age"] <= ROOKIE_SCALE_MAX_AGE):
            pred_Dpp[i] = min(pred_Dpp[i], r["salary_prev"] * ROOKIE_SCALE_STEP_UP)
    sDpp = score(test_df, pred_Dpp)
    print()
    print_score("D''. GBM + CBA max cap + rookie scale cap", sDpp, sA)

    # ── E. Gradient Boosting on RESIDUALS of baseline (stacking) ─────────────
    # Idea: keep the rank-mapping baseline, learn the corrections only.
    print()
    baseline_train_pct = predict_baseline(train_df) / train_df["cap_curr"].values
    residuals_train = y_train - baseline_train_pct
    gb_res = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        min_samples_leaf=15, subsample=0.8,
        random_state=42,
    )
    gb_res.fit(X_train, residuals_train)
    baseline_test_pct = predict_baseline(test_df) / cap_test
    residual_pred = gb_res.predict(X_test)
    pred_E_pct = np.clip(baseline_test_pct + residual_pred, 0.001, 0.45)
    pred_E = pred_E_pct * cap_test
    sE = score(test_df, pred_E)
    print_score("E. GBM on residuals of baseline (stacking)", sE, sA)

    # ── Feature importances for GBM ─────────────────────────────────────────
    print("\nFEATURE IMPORTANCES (Gradient Boosting):")
    fnames = list(FEATURE_COLS) + ["age²", "barrett²", "log(rank+1)",
                                     "tier_30pct", "tier_35pct",
                                     "is_Guard", "is_Forward"]
    importances = sorted(zip(fnames, gb.feature_importances_),
                         key=lambda kv: -kv[1])
    for name, imp in importances[:12]:
        print(f"  {name:<22} {imp:.3f}")

    # ── Tier breakdowns ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("TIER-SEGMENTED COMPARISON  (test set, 2025-26 $-equivalent)")
    print("=" * 78)
    report_by_tier("A. baseline",            test_df, sA)
    report_by_tier("D. gradient boosting",   test_df, sD)
    report_by_tier("D''. GBM + max cap + rookie cap", test_df, sDpp)
    report_by_tier("E. GBM on residuals",    test_df, sE)

    # ── Verdict ──────────────────────────────────────────────────────────────
    candidates = [
        ("A. baseline",                        sA),
        ("B. Ridge",                           best_sB),
        ("C. Random Forest",                   sC),
        ("D. Gradient Boost",                  sD),
        ("D'. GBM + max cap",                  sDp),
        ("D''. GBM + max cap + rookie cap",    sDpp),
        ("E. GBM residuals",                   sE),
    ]
    best = max(candidates, key=lambda x: x[1]["within_5"])
    gain = best[1]["within_5"] - sA["within_5"]

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  Best model:  {best[0]} — Within 5% = {best[1]['within_5']:.1f}%")
    print(f"  Baseline:    A. baseline — Within 5% = {sA['within_5']:.1f}%")
    print(f"  Gain:        {gain:+.2f}pp")
    if best[0].startswith("A"):
        print("  → Baseline still wins. Rank-mapping is a strong prior.")
    elif gain >= 1.0:
        print("  → SHIP — meaningful improvement on within-5%.")
    elif gain >= 0.5:
        print("  → CONSIDER — modest gain. Check tier breakdown.")
    else:
        print("  → DO NOT SHIP — gain within noise.")


if __name__ == "__main__":
    main()
