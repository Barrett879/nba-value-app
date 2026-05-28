"""Validate the production GBM through the canonical analyze_accuracy.py
metrics. This is the apples-to-apples comparison: same pool definition,
same eras, same scoring as the official headline.

Headline metric: within 5% of cap, 1999+ (CBA-max era).

Usage:
    python scripts/validate_gbm_canonical.py
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

from utils import (
    SEASONS, build_ranked_projected, SALARY_CAP_M, fetch_league_stats,
    tiered_age_multiplier, normalize, season_to_espn_year,
    apply_rankings, build_raw,
    fetch_bref_positions, fetch_player_positions_detailed, position_to_bucket,
    HEALTHY_SEASON_GP, NEW_CONTRACT_PCT,
)

# Reuse career-building utilities from train_ml_model.py.
sys.path.insert(0, str(Path(__file__).parent))
from train_ml_model import (
    build_career_indexes, trailing_weighted_barrett, trailing_avg,
    years_in_league, FEATURE_COLS,
)


CACHE_DIR = Path(__file__).parent.parent / "cache"
# Use the HOLDOUT model (trained on 1999-2014 only) for honest out-of-
# sample validation. The full-data model is for production use, not for
# validation (would leak training data).
MODEL_PATH = CACHE_DIR / "contract_gbm_v1_holdout.joblib"
HOLDOUT_SPLIT_YEAR = 2015
CURRENT_CAP_M = SALARY_CAP_M["2025-26"]

ROOKIE_SCALE_SAL_PCT  = 0.15
ROOKIE_SCALE_MAX_AGE  = 25
ROOKIE_SCALE_STEP_UP  = 1.5
ROOKIE_SCALE_FIRST_YR = 1995


def _cap(season: str) -> float:
    return SALARY_CAP_M.get(season, 1.0) * 1_000_000


def gbm_predict_row(model, r: pd.Series, cap_dollars: float) -> float:
    """Predict salary as dollars for a single row using the GBM."""
    barrett = float(r.get("barrett") or 0)
    score_rank = float(r.get("score_rank") or 999)
    age = float(r.get("age") or 25)
    yrs = float(r.get("years_in_league") or 0)
    pos_bucket = r.get("pos_bucket", "Unknown")

    feat_vec = [
        barrett,                                          # barrett (trailing-weighted)
        float(r.get("barrett_single") or 0),              # barrett_single
        float(r.get("barrett_3yr") or barrett),           # barrett_3yr
        score_rank,
        float(r.get("PTS") or 0),
        float(r.get("AST") or 0),
        float(r.get("REB") or 0),
        float(r.get("STL") or 0),
        float(r.get("BLK") or 0),
        float(r.get("TOV") or 0),
        float(r.get("eff_adj") or 0),
        float(r.get("d_lebron") or 0),
        float(r.get("GP") or 0),
        float(r.get("MPG") or 0),
        float(r.get("gp_3yr") or 0),
        age,
        float(r.get("salary_prev_pct") or 0),
        float(r.get("career_base_proj_pct") or 0),
        yrs,
        # Derived:
        age ** 2,
        barrett ** 2,
        float(np.log1p(score_rank)),
        1.0 if yrs >= 7 else 0.0,                         # tier_30pct
        1.0 if yrs >= 10 else 0.0,                        # tier_35pct
        1.0 if pos_bucket == "Guard" else 0.0,
        1.0 if pos_bucket == "Forward" else 0.0,
    ]
    pred_pct = float(np.clip(model.predict(np.array([feat_vec]))[0], 0.001, 0.45))
    return pred_pct * cap_dollars


def analyze_pair_gbm(model, prev_season: str, curr_season: str,
                      careers: dict) -> pd.DataFrame:
    if prev_season not in SALARY_CAP_M or curr_season not in SALARY_CAP_M:
        return pd.DataFrame()
    try:
        prev_df = build_ranked_projected(prev_season)
        curr_df = build_ranked_projected(curr_season)
    except Exception:
        return pd.DataFrame()
    if prev_df.empty or curr_df.empty:
        return pd.DataFrame()
    prev_df = prev_df[prev_df["salary"] > 0].copy()
    if prev_df.empty:
        return pd.DataFrame()

    cap_prev, cap_curr = _cap(prev_season), _cap(curr_season)
    cap_ratio = cap_curr / cap_prev

    raw_prev = fetch_league_stats(prev_season, "Regular Season")
    age_lookup = dict(zip(raw_prev["PLAYER_ID"], raw_prev.get("AGE", [])))

    try:
        detailed = fetch_player_positions_detailed(prev_season, cache_v=2)
    except Exception:
        detailed = {}
    try:
        coarse = fetch_bref_positions(season_to_espn_year(prev_season), cache_v=3)
    except Exception:
        coarse = {}

    def _pos_bucket(n):
        d = detailed.get(normalize(n))
        if d: return position_to_bucket(d)
        return coarse.get(normalize(n), "Unknown")

    curr_slim = curr_df[["PLAYER_ID", "salary"]].rename(
        columns={"salary": "salary_curr"})
    m = prev_df.merge(curr_slim, on="PLAYER_ID", how="left")
    m["salary_curr"] = m["salary_curr"].fillna(0)
    m["pct_change"] = (m["salary_curr"] - m["salary"]) / m["salary"]
    pool = m[
        (m["projected_salary"].fillna(0) > 0)
        & (m["salary_curr"] > 0)
        & (m["pct_change"].abs() >= NEW_CONTRACT_PCT)
    ].copy()
    if pool.empty:
        return pd.DataFrame()

    # Build per-row features for the GBM.
    out_rows = []
    for _, row in pool.iterrows():
        pid = int(row["PLAYER_ID"])
        age = age_lookup.get(pid)
        if age is None or pd.isna(age):
            continue
        age = float(age)
        career = careers.get(pid, pd.DataFrame())
        tw_barrett = trailing_weighted_barrett(career, prev_season)
        if tw_barrett is None:
            tw_barrett = float(row.get("barrett_score") or 0)
        barrett_3yr = trailing_avg(career, prev_season, "Barrett Score", n=3) or tw_barrett
        gp_3yr      = trailing_avg(career, prev_season, "GP",            n=3) or float(row.get("GP") or 0)
        yrs = years_in_league(career, prev_season)
        pos = _pos_bucket(row["Player"])

        # career_base_proj for the GBM feature (curr's salary at this barrett's effective rank).
        cur_scores = curr_df["barrett_score"].sort_values(ascending=False).values
        cur_salaries = curr_df["salary"].sort_values(ascending=False).values
        effective_rank = int((cur_scores > tw_barrett).sum()) + 1
        capped_rank = min(effective_rank, len(cur_salaries)) - 1
        career_base_proj = float(cur_salaries[capped_rank])

        feat_row = pd.Series({
            "barrett":              tw_barrett,
            "barrett_single":       float(row.get("barrett_score") or 0),
            "barrett_3yr":          barrett_3yr,
            "score_rank":           float(row.get("score_rank") or 999),
            "PTS":                  float(row.get("PTS") or 0),
            "AST":                  float(row.get("AST") or 0),
            "REB":                  float((row.get("OREB") or 0) + (row.get("DREB") or 0)),
            "STL":                  float(row.get("STL") or 0),
            "BLK":                  float(row.get("BLK") or 0),
            "TOV":                  float(row.get("TOV") or 0),
            "eff_adj":              float(row.get("efficiency_adj") or 0),
            "d_lebron":             float(row.get("d_lebron") or 0),
            "GP":                   float(row.get("GP") or 0),
            "MPG":                  float(row.get("MIN") or 0),
            "gp_3yr":               gp_3yr,
            "age":                  age,
            "salary_prev_pct":      float(row["salary"]) / cap_prev,
            "career_base_proj_pct": career_base_proj / cap_curr,
            "years_in_league":      yrs,
            "pos_bucket":           pos,
        })
        pred_dollars = gbm_predict_row(model, feat_row, cap_curr)
        err = abs(float(row["salary_curr"]) - pred_dollars) / cap_curr * 100
        signed_err = (float(row["salary_curr"]) - pred_dollars) / cap_curr * 100
        actual_today_M = float(row["salary_curr"]) / cap_curr * CURRENT_CAP_M

        out_rows.append({
            "signed_in":          curr_season,
            "start_year":         int(curr_season.split("-")[0]),
            "abs_err_pct_cap":    err,
            "signed_err_pct_cap": signed_err,
            "abs_err_today_M":    err / 100 * CURRENT_CAP_M,
            "actual_today_M":     actual_today_M,
        })
    return pd.DataFrame(out_rows)


def report_headline(label: str, errs: pd.DataFrame) -> None:
    n = len(errs)
    w5  = (errs["abs_err_pct_cap"] <= 5).mean() * 100
    w10 = (errs["abs_err_pct_cap"] <= 10).mean() * 100
    med = errs["abs_err_pct_cap"].median()
    med_M = errs["abs_err_today_M"].median()
    bias = errs["signed_err_pct_cap"].median()
    print(f"\n{label}")
    print(f"  n             = {n:,}")
    print(f"  Median |err|  = {med:.2f}% of cap  (${med_M:.2f}M)")
    print(f"  Within 5%     = {w5:.1f}%")
    print(f"  Within 10%    = {w10:.1f}%")
    print(f"  Median bias   = {bias:+.2f}% of cap")


def report_tiers(label: str, errs: pd.DataFrame) -> None:
    print(f"\nTier breakdown — {label}")
    actual = errs["actual_today_M"]
    err    = errs["abs_err_today_M"]
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
        med = sub_err.median()
        print(f"  {name:<12} n={n:>4}   median ${med:>5.2f}M   "
              f"±$3M {within_3:>4.0f}%   ±$5M {within_5:>4.0f}%")


def main() -> None:
    print(f"Loading GBM from {MODEL_PATH}...")
    if not MODEL_PATH.exists():
        print(f"ERROR: model file not found at {MODEL_PATH}")
        print("Run scripts/build_production_gbm.py first.")
        return
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    print(f"  Loaded {artifact['version']} (trained on {artifact['n_train_rows']} rows)")

    print("\nBuilding career indexes...")
    t0 = time.time()
    careers = build_career_indexes()
    print(f"  {len(careers)} careers in {time.time()-t0:.1f}s.")

    # Iterate all pairs.
    pairs = [(SEASONS[i + 1], SEASONS[i]) for i in range(len(SEASONS) - 1)]
    print(f"\nIterating {len(pairs)} season pairs...")
    rows = []
    for prev, curr in pairs:
        df = analyze_pair_gbm(model, prev, curr, careers)
        if not df.empty:
            rows.append(df)
    combined = pd.concat(rows, ignore_index=True)
    print(f"Total contracts: {len(combined):,}")
    print(f"(Filtering to out-of-sample only: start_year ≥ {HOLDOUT_SPLIT_YEAR})")

    # Honest validation: only score on OUT-OF-SAMPLE pairs (post-2014).
    # Pre-2015 pairs were in the holdout model's training set.
    oos = combined[combined["start_year"] >= HOLDOUT_SPLIT_YEAR].copy()
    modern_years = sorted({int(s.split("-")[0]) for s in SEASONS[:10]})
    modern = oos[oos["start_year"].isin(modern_years)]

    print("\n" + "=" * 70)
    print("GBM OUT-OF-SAMPLE VALIDATION  (canonical pool, post-training years)")
    print("=" * 70)
    report_headline(f"OUT-OF-SAMPLE ALL ({HOLDOUT_SPLIT_YEAR}+) — headline",  oos)
    report_headline("MODERN ERA (last 10 seasons)",                          modern)

    print()
    report_tiers(f"Out-of-sample ({HOLDOUT_SPLIT_YEAR}+)", oos)


if __name__ == "__main__":
    main()
