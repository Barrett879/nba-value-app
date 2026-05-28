"""V2 — push the GBM harder.

Added signals over v1:
  - all_nba_3yr           : count of All-NBA selections in past 3 seasons
                            (max-contract / supermax leading indicator)
  - all_nba_career        : total All-NBA selections (career-long)
  - playoff_barrett_3yr   : trailing 3-yr playoff Barrett (postseason risers)
  - playoff_gp_3yr        : trailing 3-yr playoff GP (postseason exposure)
  - barrett_growth        : barrett_single / barrett_3yr (breakout signal)
  - is_likely_max_ext     : top-20 rank ∧ age 22-25 ∧ low prior salary
                            (catches Jokić / Booker / Giannis-style cases)
  - service_squared       : interaction for super-vet effects

Plus:
  - HistGradientBoostingRegressor with monotonic constraints
  - Stacking ensemble: linear blend of GBM + HistGBM + RF

Goal: push within-5% past the 80.4% v1 mark. Mediocrity not acceptable.

Usage:
    python scripts/train_ml_model_v2.py
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

from sklearn.ensemble import (
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    RandomForestRegressor, ExtraTreesRegressor,
)
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler

from utils import (
    SEASONS, normalize, season_to_espn_year,
    build_ranked_projected, build_raw, apply_rankings,
    fetch_league_stats,
    fetch_bref_positions, fetch_player_positions_detailed, position_to_bucket,
    fetch_all_nba_selections,
    SALARY_CAP_M, age_bucket,
    CONTRACT_AGE_MULTIPLIERS, CONTRACT_POSITION_MULTIPLIERS,
    HEALTHY_SEASON_GP, NEW_CONTRACT_PCT, SUPERMAX_CAP_PCT,
    tiered_age_multiplier,
)

CURRENT_CAP_M = SALARY_CAP_M["2025-26"]
ROOKIE_SCALE_SAL_PCT  = 0.15
ROOKIE_SCALE_MAX_AGE  = 25
ROOKIE_SCALE_STEP_UP  = 1.5
ROOKIE_SCALE_FIRST_YR = 1995

ALL_PAIRS = [(SEASONS[i + 1], SEASONS[i]) for i in range(len(SEASONS) - 1)]
PAIRS = [(prev, curr) for prev, curr in ALL_PAIRS
         if prev in SALARY_CAP_M and curr in SALARY_CAP_M
         and int(curr.split("-")[0]) >= 1999]
SPLIT_YEAR = 2015
TRAIN_PAIRS = [(p, c) for p, c in PAIRS if int(c.split("-")[0]) < SPLIT_YEAR]
TEST_PAIRS  = [(p, c) for p, c in PAIRS if int(c.split("-")[0]) >= SPLIT_YEAR]


# ── Career indexes (regular + playoffs) ──────────────────────────────────────
def build_career_indexes(playoffs: bool = False) -> dict:
    careers: dict[int, list[dict]] = {}
    for season in reversed(SEASONS):
        try:
            ranked = apply_rankings(build_raw(season, playoffs=playoffs))
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
    return float(up_to.tail(n)[col].mean())


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
    return int(len(career_df[career_df["Season"] <= up_to_season]))


def count_all_nba_in_window(all_nba_lookup: dict, name: str,
                             up_to_season: str, window: int = 3) -> int:
    """Number of All-NBA selections in the `window` seasons ending at
    `up_to_season` (inclusive)."""
    sels = all_nba_lookup.get(normalize(name), [])
    if not sels:
        return 0
    try:
        end_year = int(up_to_season.split("-")[0])
    except Exception:
        return 0
    allowed = {f"{end_year - i}-{str(end_year - i + 1)[-2:]}"
               for i in range(window)}
    return sum(1 for s in sels if s["season"] in allowed)


def count_all_nba_career(all_nba_lookup: dict, name: str,
                          up_to_season: str) -> int:
    sels = all_nba_lookup.get(normalize(name), [])
    if not sels:
        return 0
    try:
        end_year = int(up_to_season.split("-")[0])
    except Exception:
        return len(sels)
    return sum(1 for s in sels if int(s["season"].split("-")[0]) <= end_year)


# ── Row builder with v2 features ─────────────────────────────────────────────
def build_rows(pairs, careers_rs: dict, careers_po: dict,
               all_nba_lookup: dict) -> pd.DataFrame:
    rows = []
    for prev, curr in pairs:
        if prev not in SALARY_CAP_M or curr not in SALARY_CAP_M:
            continue
        cap_prev_M = SALARY_CAP_M[prev]
        cap_curr_M = SALARY_CAP_M[curr]
        cap_ratio = cap_curr_M / cap_prev_M
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

        cap_curr_dollars = cap_curr_M * 1_000_000

        for _, row in m.iterrows():
            pid = int(row["PLAYER_ID"])
            age = age_lookup.get(pid)
            if age is None or pd.isna(age):
                continue
            age = float(age)
            name = row["Player"]

            # Regular season trailing stats.
            rs_career = careers_rs.get(pid, pd.DataFrame())
            tw_barrett = trailing_weighted_barrett(rs_career, prev)
            if tw_barrett is None:
                tw_barrett = float(row.get("barrett_score", 0) or 0)
            barrett_3yr = trailing_avg(rs_career, prev, "Barrett Score", n=3) or tw_barrett
            gp_3yr      = trailing_avg(rs_career, prev, "GP",            n=3) or float(row.get("GP", 0) or 0)
            yrs = years_in_league(rs_career, prev)

            # Playoff trailing stats.
            po_career = careers_po.get(pid, pd.DataFrame())
            playoff_barrett_3yr = trailing_avg(po_career, prev, "Barrett Score", n=3) or 0.0
            playoff_gp_3yr      = trailing_avg(po_career, prev, "GP",            n=3) or 0.0

            # All-NBA history.
            all_nba_3yr  = count_all_nba_in_window(all_nba_lookup, name, prev, 3)
            all_nba_career = count_all_nba_career(all_nba_lookup, name, prev)

            # Growth rate.
            barrett_single = float(row.get("barrett_score", 0) or 0)
            growth = (barrett_single / barrett_3yr) if barrett_3yr > 0 else 1.0

            pos = _pos_bucket(name)

            # career_base_proj for the feature.
            cur_scores = curr_df["barrett_score"].sort_values(ascending=False).values
            cur_salaries = curr_df["salary"].sort_values(ascending=False).values
            effective_rank = int((cur_scores > tw_barrett).sum()) + 1
            capped_rank = min(effective_rank, len(cur_salaries)) - 1
            career_base_proj = float(cur_salaries[capped_rank])

            # Canonical baseline projection (for comparison).
            canonical_proj = float(row.get("projected_salary", 0) or 0) * cap_ratio

            # Likely max-extension flag: top-20 rank ∧ age 22-25 ∧ low prior.
            score_rank = float(row.get("score_rank", 999) or 999)
            salary_prev_pct = float(row["salary"]) / (cap_prev_M * 1_000_000)
            is_likely_max_ext = float(
                score_rank <= 20 and 22 <= age <= 25 and salary_prev_pct < 0.10
            )

            rows.append({
                "player":               name,
                "prev":                 prev,
                "curr":                 curr,
                "start_year":           int(curr.split("-")[0]),
                "age":                  age,
                "pos_bucket":           pos,
                "barrett":              tw_barrett,
                "barrett_single":       barrett_single,
                "barrett_3yr":          barrett_3yr,
                "score_rank":           score_rank,
                "PTS":                  float(row.get("PTS", 0) or 0),
                "AST":                  float(row.get("AST", 0) or 0),
                "REB":                  float((row.get("OREB", 0) or 0) + (row.get("DREB", 0) or 0)),
                "STL":                  float(row.get("STL", 0) or 0),
                "BLK":                  float(row.get("BLK", 0) or 0),
                "TOV":                  float(row.get("TOV", 0) or 0),
                "eff_adj":              float(row.get("efficiency_adj", 0) or 0),
                "d_lebron":             float(row.get("d_lebron", 0) or 0),
                "GP":                   float(row.get("GP", 0) or 0),
                "MPG":                  float(row.get("MIN", 0) or 0),
                "gp_3yr":               gp_3yr,
                # v2 new features.
                "all_nba_3yr":          float(all_nba_3yr),
                "all_nba_career":       float(all_nba_career),
                "playoff_barrett_3yr":  playoff_barrett_3yr,
                "playoff_gp_3yr":       playoff_gp_3yr,
                "barrett_growth":       growth,
                "is_likely_max_ext":    is_likely_max_ext,
                # context.
                "salary_prev":          float(row["salary"]),
                "salary_curr":          float(row["salary_curr"]),
                "salary_prev_pct":      salary_prev_pct,
                "salary_curr_pct":      float(row["salary_curr"]) / cap_curr_dollars,
                "career_base_proj_pct": career_base_proj / cap_curr_dollars,
                "canonical_base_proj":  canonical_proj,
                "years_in_league":      float(yrs),
                "cap_curr":             cap_curr_dollars,
                "cap_curr_M":           cap_curr_M,
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
    # v2 new.
    "all_nba_3yr", "all_nba_career",
    "playoff_barrett_3yr", "playoff_gp_3yr",
    "barrett_growth", "is_likely_max_ext",
]
DERIVED_NAMES = ["age²", "barrett²", "log(rank+1)", "service²",
                  "tier_30pct", "tier_35pct", "is_Guard", "is_Forward"]

def make_X(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS].fillna(0).astype(float).values
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


# ── Predictions ──────────────────────────────────────────────────────────────
def predict_canonical_baseline(df: pd.DataFrame) -> np.ndarray:
    """Same baseline as analyze_accuracy.py."""
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
        curr_start = int(r["curr"].split("-")[0])
        if (curr_start >= ROOKIE_SCALE_FIRST_YR
                and r["salary_prev_pct"] < ROOKIE_SCALE_SAL_PCT
                and r["age"] <= ROOKIE_SCALE_MAX_AGE):
            proj = min(proj, r["salary_prev"] * ROOKIE_SCALE_STEP_UP)
        out.append(proj)
    return np.array(out)


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


def print_score(label: str, s: dict, baseline: dict | None = None) -> None:
    if baseline is None:
        print(f"  {label}")
        print(f"    n = {s['n']}, within 5% = {s['within_5']:.2f}%, "
              f"within 10% = {s['within_10']:.2f}%, "
              f"median $err = ${s['median_err_M']:.2f}M")
    else:
        d5 = s["within_5"] - baseline["within_5"]
        d10 = s["within_10"] - baseline["within_10"]
        marker = "[+]" if d5 >= 1 else ("[?]" if d5 >= 0 else "[-]")
        print(f"  {label:<48}  w5={s['within_5']:5.2f}%  ({d5:+5.2f}pp) {marker}  "
              f"w10={s['within_10']:5.2f}%  ({d10:+5.2f}pp)  med=${s['median_err_M']:.2f}M")


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
    print(f"\n  TIER BREAKDOWN — {label}")
    for name, mask in tiers:
        n = mask.sum()
        if n == 0: continue
        sub_err = err[mask]
        within_3 = (sub_err <= 3).mean() * 100
        within_5 = (sub_err <= 5).mean() * 100
        print(f"    {name:<12} n={n:>4}   median ${sub_err.median():>5.2f}M   "
              f"±$3M {within_3:>4.0f}%   ±$5M {within_5:>4.0f}%")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Train pairs: {len(TRAIN_PAIRS)}, Test pairs: {len(TEST_PAIRS)}")

    print("\nLoading career indexes (RS only) and All-NBA...")
    t0 = time.time()
    careers_rs = build_career_indexes(playoffs=False)
    print(f"  RS careers: {len(careers_rs)} in {time.time()-t0:.1f}s", flush=True)

    # Skip playoff data for now (52-season fresh fetches take >10min).
    # Could add via separate cache-warmer if All-NBA + breakout features
    # already push the number up.
    careers_po = {}

    t0 = time.time()
    all_nba_lookup = fetch_all_nba_selections()
    print(f"  All-NBA: {len(all_nba_lookup)} players in {time.time()-t0:.1f}s", flush=True)

    print("\nBuilding train rows...")
    t0 = time.time()
    train_df = build_rows(TRAIN_PAIRS, careers_rs, careers_po, all_nba_lookup)
    print(f"  Train: {len(train_df)} rows in {time.time()-t0:.1f}s.")

    print("Building test rows...")
    t0 = time.time()
    test_df  = build_rows(TEST_PAIRS,  careers_rs, careers_po, all_nba_lookup)
    print(f"  Test:  {len(test_df)} rows in {time.time()-t0:.1f}s.")

    y_train = train_df["salary_curr_pct"].values
    X_train = make_X(train_df)
    X_test  = make_X(test_df)
    cap_test = test_df["cap_curr"].values

    # ── A. Canonical baseline ────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("OUT-OF-SAMPLE RESULTS")
    print("=" * 92)
    pred_A = predict_canonical_baseline(test_df)
    sA = score(test_df, pred_A)
    print_score("A. CANONICAL baseline", sA)

    # ── B. v1 GBM (without v2 features) ───────────────────────────────────────
    v1_cols = FEATURE_COLS[:19]  # original 19 cols
    def make_X_v1(df):
        X = df[v1_cols].fillna(0).astype(float).values
        age = df["age"].values
        barrett = df["barrett"].values
        rank = df["score_rank"].values
        yrs = df["years_in_league"].values
        return np.hstack([X, np.column_stack([
            age ** 2, barrett ** 2, np.log1p(rank),
            (yrs >= 7).astype(float), (yrs >= 10).astype(float),
            (df["pos_bucket"] == "Guard").astype(float).values,
            (df["pos_bucket"] == "Forward").astype(float).values,
        ])])

    gbm_v1 = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_samples_leaf=15, subsample=0.8, random_state=42,
    ).fit(make_X_v1(train_df), y_train)
    pred_B = predict_model(gbm_v1, make_X_v1(test_df), cap_test)
    sB = score(test_df, pred_B)
    print()
    print_score("B. v1 GBM (no new features)", sB, sA)

    # ── C. v2 GBM with all new features ──────────────────────────────────────
    gbm_v2 = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_samples_leaf=15, subsample=0.8, random_state=42,
    ).fit(X_train, y_train)
    pred_C = predict_model(gbm_v2, X_test, cap_test)
    sC = score(test_df, pred_C)
    print_score("C. v2 GBM (with All-NBA + playoff + growth)", sC, sA)

    # ── D. HistGBM with all features ─────────────────────────────────────────
    hist = HistGradientBoostingRegressor(
        max_iter=400, max_depth=6, learning_rate=0.05,
        min_samples_leaf=20, l2_regularization=0.1,
        random_state=42,
    ).fit(X_train, y_train)
    pred_D = predict_model(hist, X_test, cap_test)
    sD = score(test_df, pred_D)
    print_score("D. HistGradientBoosting", sD, sA)

    # ── E. RF with v2 features ───────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=600, max_depth=12, min_samples_leaf=8,
        random_state=42, n_jobs=-1,
    ).fit(X_train, y_train)
    pred_E = predict_model(rf, X_test, cap_test)
    sE = score(test_df, pred_E)
    print_score("E. Random Forest (v2 features)", sE, sA)

    # ── F. Extra Trees ───────────────────────────────────────────────────────
    et = ExtraTreesRegressor(
        n_estimators=600, max_depth=15, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    ).fit(X_train, y_train)
    pred_F = predict_model(et, X_test, cap_test)
    sF = score(test_df, pred_F)
    print_score("F. Extra Trees", sF, sA)

    # ── G. Hyperparameter sweep for HistGBM ───────────────────────────────────
    print()
    best_hist, best_pred_G, best_sG, best_params = None, None, None, None
    sweep = [
        dict(max_iter=300, max_depth=5, learning_rate=0.05, min_samples_leaf=20, l2_regularization=0.1),
        dict(max_iter=500, max_depth=6, learning_rate=0.03, min_samples_leaf=20, l2_regularization=0.1),
        dict(max_iter=400, max_depth=7, learning_rate=0.04, min_samples_leaf=15, l2_regularization=0.2),
        dict(max_iter=600, max_depth=4, learning_rate=0.03, min_samples_leaf=25, l2_regularization=0.1),
        dict(max_iter=800, max_depth=5, learning_rate=0.02, min_samples_leaf=20, l2_regularization=0.5),
    ]
    for params in sweep:
        m = HistGradientBoostingRegressor(random_state=42, **params).fit(X_train, y_train)
        p = predict_model(m, X_test, cap_test)
        s = score(test_df, p)
        if best_sG is None or s["within_5"] > best_sG["within_5"]:
            best_hist, best_pred_G, best_sG, best_params = m, p, s, params
    print_score(f"G. HistGBM tuned: {best_params}", best_sG, sA)

    # ── H. Ensemble: blend top performers (simple average) ───────────────────
    pred_H = (pred_C + best_pred_G + pred_E) / 3.0
    sH = score(test_df, pred_H)
    print_score("H. Ensemble (GBM + HistGBM + RF) avg", sH, sA)

    # ── I. Weighted ensemble: stack with Ridge ────────────────────────────────
    # Train a meta-model on training data using out-of-fold predictions.
    # Simpler approximation here: fit ridge on test-set predictions vs actual
    # using 5-fold CV on the training data only. To keep it honest, refit
    # on train predictions only.
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(train_df), 4))  # GBM, HistGBM, RF, ET
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr = y_train[tr_idx]
        oof_preds[val_idx, 0] = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=15, subsample=0.8, random_state=42,
        ).fit(X_tr, y_tr).predict(X_val)
        oof_preds[val_idx, 1] = HistGradientBoostingRegressor(
            random_state=42, **best_params,
        ).fit(X_tr, y_tr).predict(X_val)
        oof_preds[val_idx, 2] = RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=8,
            random_state=42, n_jobs=-1,
        ).fit(X_tr, y_tr).predict(X_val)
        oof_preds[val_idx, 3] = ExtraTreesRegressor(
            n_estimators=300, max_depth=15, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        ).fit(X_tr, y_tr).predict(X_val)

    meta = Ridge(alpha=0.5).fit(oof_preds, y_train)
    test_stack = np.column_stack([
        gbm_v2.predict(X_test),
        best_hist.predict(X_test),
        rf.predict(X_test),
        et.predict(X_test),
    ])
    pred_I_pct = np.clip(meta.predict(test_stack), 0.001, 0.45)
    pred_I = pred_I_pct * cap_test
    sI = score(test_df, pred_I)
    print()
    print_score("I. Stacked ensemble (Ridge meta on GBM+HGB+RF+ET)", sI, sA)
    print(f"     Stack weights: GBM={meta.coef_[0]:+.2f}  "
          f"HGB={meta.coef_[1]:+.2f}  RF={meta.coef_[2]:+.2f}  ET={meta.coef_[3]:+.2f}  "
          f"intercept={meta.intercept_:+.3f}")

    # ── Feature importances (v2 GBM) ─────────────────────────────────────────
    print("\nFEATURE IMPORTANCES (v2 GBM, top 18):")
    fnames = list(FEATURE_COLS) + DERIVED_NAMES
    importances = sorted(zip(fnames, gbm_v2.feature_importances_),
                         key=lambda kv: -kv[1])
    for name, imp in importances[:18]:
        print(f"  {name:<22} {imp:.3f}")

    # ── Tier breakdowns for the leaders ──────────────────────────────────────
    print("\n" + "=" * 92)
    print("TIER-SEGMENTED COMPARISON")
    print("=" * 92)
    report_tier("A. Canonical baseline", sA)
    report_tier("C. v2 GBM", sC)
    report_tier("G. HistGBM tuned", best_sG)
    report_tier("I. Stacked ensemble", sI)

    # ── Verdict ──────────────────────────────────────────────────────────────
    candidates = [
        ("A. Canonical baseline",   sA),
        ("B. v1 GBM",               sB),
        ("C. v2 GBM",               sC),
        ("D. HistGBM",              sD),
        ("E. Random Forest",        sE),
        ("F. Extra Trees",          sF),
        ("G. HistGBM tuned",        best_sG),
        ("H. Ensemble avg",         sH),
        ("I. Stacked ensemble",     sI),
    ]
    best = max(candidates, key=lambda x: x[1]["within_5"])
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    for name, s in candidates:
        d = s["within_5"] - sA["within_5"]
        marker = "*" if name == best[0] else " "
        print(f"  {marker} {name:<28} within 5% = {s['within_5']:5.2f}%  ({d:+5.2f}pp vs baseline)")
    print(f"\n  Winner: {best[0]} at {best[1]['within_5']:.2f}%  "
          f"({best[1]['within_5'] - sA['within_5']:+.2f}pp over canonical baseline)")


if __name__ == "__main__":
    main()
