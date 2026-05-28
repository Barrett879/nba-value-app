"""Diagnose where the predictor misses worst. For each decade, show the
biggest |error| contracts so we can spot patterns in the failures.

After D-LEBRON proxy: 1980s/1990s improved a lot, but 2000s/2010s sit
around 72-75%. Where are those misses concentrated?
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from utils import SEASONS, build_ranked_projected, SALARY_CAP_M


NEW_DEAL_PCT_THRESHOLD = 0.25
TOP_K_MISSES = 10


def _cap(season: str) -> float:
    return SALARY_CAP_M.get(season, 1.0) * 1_000_000


def _decade(season: str) -> str:
    return f"{(int(season.split('-')[0]) // 10) * 10}s"


def analyze_pair(prev_season: str, curr_season: str) -> pd.DataFrame:
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
    cap_prev, cap_curr = _cap(prev_season), _cap(curr_season)
    ratio = cap_curr / cap_prev

    m = (
        prev_df[["PLAYER_ID", "Player", "salary", "projected_salary",
                 "barrett_score"]]
        .rename(columns={"salary": "salary_prev",
                         "projected_salary": "proj_prev"})
        .merge(curr_df[["PLAYER_ID", "salary"]]
                 .rename(columns={"salary": "salary_curr"}),
               on="PLAYER_ID", how="left")
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

    pool["proj_capadj"] = pool["proj_prev"] * ratio
    pool["pct_cap_err"] = (pool["salary_curr"] - pool["proj_capadj"]).abs() / cap_curr * 100
    pool["signed_err_pct_cap"] = (pool["salary_curr"] - pool["proj_capadj"]) / cap_curr * 100
    pool["signed_in"] = curr_season
    pool["decade"] = _decade(curr_season)
    pool["actual_M"] = pool["salary_curr"] / 1e6
    pool["pred_M"] = pool["proj_capadj"] / 1e6
    pool["actual_pct_cap"] = pool["salary_curr"] / cap_curr * 100
    pool["pred_pct_cap"] = pool["proj_capadj"] / cap_curr * 100
    return pool[["signed_in", "decade", "Player", "barrett_score",
                 "actual_M", "pred_M",
                 "actual_pct_cap", "pred_pct_cap",
                 "pct_cap_err", "signed_err_pct_cap"]]


def main() -> None:
    print("Loading...")
    pairs = [(SEASONS[i + 1], SEASONS[i]) for i in range(len(SEASONS) - 1)]
    rows = []
    for prev, curr in pairs:
        df = analyze_pair(prev, curr)
        if not df.empty:
            rows.append(df)
    combined = pd.concat(rows, ignore_index=True)
    print(f"Total contracts: {len(combined):,}\n")

    # Per decade: worst K misses (signed_err can be + or -, look at both).
    for decade, g in combined.groupby("decade"):
        n = len(g)
        median_err = g["pct_cap_err"].median()
        bias = g["signed_err_pct_cap"].median()
        print("=" * 78)
        print(f"{decade}  ·  n={n:,}  ·  median err={median_err:.2f}% cap  ·  bias={bias:+.2f}%")
        print("=" * 78)

        # Worst overshoots (model way too high).
        print(f"\n  TOP {min(TOP_K_MISSES, len(g))} OVERSHOOTS (model too high):")
        over = g.nlargest(min(TOP_K_MISSES, len(g)), "signed_err_pct_cap" * -1) \
            if False else g.nsmallest(min(TOP_K_MISSES, len(g)), "signed_err_pct_cap")
        # Above is "actual << pred" → signed_err very negative.
        print(f"  {'Player':<22} {'Season':<8} {'Barrett':>7} "
              f"{'Pred':>8} {'Actual':>8} {'Err':>7}")
        for _, r in over.iterrows():
            print(f"  {r['Player'][:22]:<22} {r['signed_in']:<8} "
                  f"{r['barrett_score']:>7.1f} "
                  f"{r['pred_M']:>7.1f}M {r['actual_M']:>7.1f}M "
                  f"{r['signed_err_pct_cap']:>+6.1f}%")

        # Worst undershoots (model too low).
        print(f"\n  TOP {min(TOP_K_MISSES, len(g))} UNDERSHOOTS (model too low):")
        under = g.nlargest(min(TOP_K_MISSES, len(g)), "signed_err_pct_cap")
        print(f"  {'Player':<22} {'Season':<8} {'Barrett':>7} "
              f"{'Pred':>8} {'Actual':>8} {'Err':>7}")
        for _, r in under.iterrows():
            print(f"  {r['Player'][:22]:<22} {r['signed_in']:<8} "
                  f"{r['barrett_score']:>7.1f} "
                  f"{r['pred_M']:>7.1f}M {r['actual_M']:>7.1f}M "
                  f"{r['signed_err_pct_cap']:>+6.1f}%")
        print()


if __name__ == "__main__":
    main()
