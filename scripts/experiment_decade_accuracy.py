"""Decade-stratified accuracy experiment — does the model hold up across
40 years of NBA history, or is it modern-era overfit?

The existing analyze_accuracy.py reports two numbers: all-era (1984-2024)
and modern-era only (2015-2024). This script breaks that down by decade
so we can see whether the rank-mapping base projection generalizes.

EXPERIMENTAL — not used by the live app. The Contract Predictor's
methodology already uses cap-relative error internally; this script
just exposes the per-decade breakdown.

Usage:
    python scripts/experiment_decade_accuracy.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from utils import SEASONS, build_ranked_projected, SALARY_CAP_M, fetch_league_stats


NEW_DEAL_PCT_THRESHOLD = 0.25  # ≥25% YoY = "real new contract"

# Same rookie-ladder filter used in pages/Contract_Predictor.load_historical_signings.
# CBA-mandated year-2/3/4 step-ups on a first-round rookie deal cross the
# ≥25% YoY threshold but aren't market-rate signings — they're contractually
# determined. The model EXPLICITLY excludes these from comp pools and caveats
# them out of predictions, so they shouldn't count against accuracy either.
ROOKIE_LADDER_SAL_PCT_PREV  = 0.15
ROOKIE_LADDER_SAL_PCT_CURR  = 0.18
ROOKIE_LADDER_MAX_AGE       = 25


def _cap(season: str) -> float:
    return SALARY_CAP_M.get(season, 1.0) * 1_000_000


def _decade_label(season: str) -> str:
    """Map '1995-96' → '1990s', '2005-06' → '2000s', etc."""
    year = int(season.split("-")[0])
    return f"{(year // 10) * 10}s"


def analyze_pair(prev_season: str, curr_season: str) -> pd.DataFrame:
    """Returns one row per contract-changed-this-year player with
    columns: signed_in, abs_err_pct_cap, signed_err_pct_cap."""
    if prev_season not in SALARY_CAP_M or curr_season not in SALARY_CAP_M:
        return pd.DataFrame()
    try:
        prev_df = build_ranked_projected(prev_season)
        curr_df = build_ranked_projected(curr_season)
    except Exception:
        return pd.DataFrame()
    if prev_df.empty or curr_df.empty:
        return pd.DataFrame()

    prev_df = prev_df[prev_df["salary"] > 0]
    if prev_df.empty:
        return pd.DataFrame()

    cap_prev = _cap(prev_season)
    cap_curr = _cap(curr_season)
    cap_ratio = cap_curr / cap_prev

    m = (
        prev_df[["PLAYER_ID", "salary", "projected_salary"]]
        .rename(columns={"salary": "salary_prev", "projected_salary": "proj_prev"})
        .merge(
            curr_df[["PLAYER_ID", "salary"]].rename(columns={"salary": "salary_curr"}),
            on="PLAYER_ID", how="left",
        )
    )
    m["salary_curr"] = m["salary_curr"].fillna(0)
    m["pct_change"] = (m["salary_curr"] - m["salary_prev"]) / m["salary_prev"]
    pool = m[
        (m["proj_prev"] > 0)
        & (m["salary_curr"] > 0)
        & (m["pct_change"].abs() >= NEW_DEAL_PCT_THRESHOLD)
    ].copy()
    if pool.empty:
        return pd.DataFrame()

    # CBA rookie-scale CAP on predictions (don't filter — adjust the
    # prediction to what's CBA-possible). For players on rookie scale
    # (salary_prev < 15% cap, age ≤ 25), the next-year salary is
    # CBA-mandated and ≤ ~150% of prior year. The base rank-mapping
    # doesn't know about CBA rules; cap its prediction here to match
    # what the live Contract Predictor does at prediction time.
    try:
        raw_prev_stats = fetch_league_stats(prev_season, "Regular Season")
        age_lookup = dict(zip(raw_prev_stats["PLAYER_ID"], raw_prev_stats.get("AGE", [])))
        pool["age"] = pool["PLAYER_ID"].map(age_lookup)
    except Exception:
        pool["age"] = None
    pool["sal_prev_pct"] = pool["salary_prev"] / cap_prev
    is_rookie_scale = (
        (pool["sal_prev_pct"] < ROOKIE_LADDER_SAL_PCT_PREV)
        & pool["age"].notna()
        & (pool["age"] <= ROOKIE_LADDER_MAX_AGE)
    )
    pool["proj_capadj"] = pool["proj_prev"] * cap_ratio
    # Cap rookie-scale predictions at 150% of prior salary (a generous
    # estimate of the max year-over-year rookie-scale step-up).
    rookie_cap = pool["salary_prev"] * 1.5
    pool.loc[is_rookie_scale, "proj_capadj"] = pool.loc[is_rookie_scale, "proj_capadj"].clip(
        upper=rookie_cap[is_rookie_scale]
    )
    pool["abs_err_pct_cap"] = (pool["salary_curr"] - pool["proj_capadj"]).abs() / cap_curr * 100
    pool["signed_err_pct_cap"] = (pool["salary_curr"] - pool["proj_capadj"]) / cap_curr * 100
    pool["signed_in"] = curr_season
    pool["decade"] = _decade_label(curr_season)
    return pool[["signed_in", "decade", "abs_err_pct_cap", "signed_err_pct_cap"]]


def main() -> None:
    print("Running decade-stratified accuracy validation...")
    print("Iterating season pairs (this takes 2-3 minutes)...\n")

    pairs = [(SEASONS[i + 1], SEASONS[i]) for i in range(len(SEASONS) - 1)]
    all_rows = []
    for prev, curr in pairs:
        df = analyze_pair(prev, curr)
        if not df.empty:
            all_rows.append(df)
    if not all_rows:
        print("No data — something's wrong with the caches.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"Total contracts analyzed: {len(combined):,}\n")

    # Per-decade summary.
    print("=" * 72)
    print("ACCURACY BY DECADE")
    print("=" * 72)
    print(f"{'Decade':<8} {'N':>6} {'Within 5%':>11} {'Within 10%':>12} "
          f"{'Median err':>11} {'Bias':>8}")
    print("-" * 72)

    for decade, group in combined.groupby("decade"):
        n = len(group)
        w5 = (group["abs_err_pct_cap"] < 5).mean() * 100
        w10 = (group["abs_err_pct_cap"] < 10).mean() * 100
        median_err = group["abs_err_pct_cap"].median()
        bias = group["signed_err_pct_cap"].median()
        print(f"{decade:<8} {n:>6,} {w5:>10.1f}% {w10:>11.1f}% "
              f"{median_err:>10.2f}% {bias:>+7.2f}%")

    print("-" * 72)
    n = len(combined)
    w5 = (combined["abs_err_pct_cap"] < 5).mean() * 100
    w10 = (combined["abs_err_pct_cap"] < 10).mean() * 100
    median_err = combined["abs_err_pct_cap"].median()
    bias = combined["signed_err_pct_cap"].median()
    print(f"{'TOTAL':<8} {n:>6,} {w5:>10.1f}% {w10:>11.1f}% "
          f"{median_err:>10.2f}% {bias:>+7.2f}%")

    print()
    print("Interpretation:")
    print("  - 'Within X%' = % of contracts where |predicted - actual| < X% of cap")
    print("  - 'Median err' = median absolute error in % of cap")
    print("  - 'Bias' = median signed error (positive → model overshoots actual)")
    print()
    print("If the model is era-agnostic, the per-decade numbers should be")
    print("similar. Big swings indicate the model fits one era better.")


if __name__ == "__main__":
    main()
