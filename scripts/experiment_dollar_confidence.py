"""Confidence at $0.5M increments — answers "if I predict $X for a player,
how confident am I in landing within $Y of their actual signing?"

Reports cumulative % of historical predictions that landed within each
dollar threshold, expressed in 2025-26 cap-equivalent dollars (so a
$10M error in 1999 is normalized to its 2025-26 purchasing power).

Headline pool: 1999+ CBA-max era (3,134 contracts) — the regime the
model is actually trained to predict.

EXPERIMENTAL — not used by the live app.

Usage:
    python scripts/experiment_dollar_confidence.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from utils import (
    SEASONS, build_ranked_projected, SALARY_CAP_M, fetch_league_stats,
    tiered_age_multiplier,
)


NEW_DEAL_PCT_THRESHOLD = 0.25
ROOKIE_LADDER_SAL_PCT_PREV  = 0.15
ROOKIE_LADDER_MAX_AGE       = 25
ROOKIE_SCALE_FIRST_YEAR     = 1995
CBA_MAX_FIRST_YEAR          = 1999

CURRENT_CAP = SALARY_CAP_M["2025-26"] * 1_000_000  # $154.6M

# Dollar thresholds (in 2025-26 dollars) to report confidence at.
DOLLAR_THRESHOLDS_M = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                       6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]


def _cap(season: str) -> float:
    return SALARY_CAP_M.get(season, 1.0) * 1_000_000


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
    if prev_df.empty:
        return pd.DataFrame()

    cap_prev, cap_curr = _cap(prev_season), _cap(curr_season)
    cap_ratio = cap_curr / cap_prev

    m = (
        prev_df[["PLAYER_ID", "salary", "projected_salary",
                 "barrett_score", "score_rank"]]
        .rename(columns={"salary": "salary_prev", "projected_salary": "proj_prev"})
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

    pool["proj_capadj"] = pool["proj_prev"] * cap_ratio

    # AGE.
    try:
        raw_prev_stats = fetch_league_stats(prev_season, "Regular Season")
        age_lookup = dict(zip(raw_prev_stats["PLAYER_ID"], raw_prev_stats.get("AGE", [])))
        pool["age"] = pool["PLAYER_ID"].map(age_lookup)
    except Exception:
        pool["age"] = None

    # Age multiplier.
    def _age_mult_row(r) -> float:
        a = r.get("age")
        if a is None or pd.isna(a):
            return 1.0
        try:
            mult, _ = tiered_age_multiplier(
                age=float(a),
                career_score=float(r.get("barrett_score", 0)),
                current_rank=int(r.get("score_rank", 0) or 0),
            )
            return float(mult)
        except Exception:
            return 1.0

    pool["age_mult"] = pool.apply(_age_mult_row, axis=1)
    pool["proj_capadj"] = pool["proj_capadj"] * pool["age_mult"]

    # Rookie scale cap.
    curr_start_year = int(curr_season.split("-")[0])
    if curr_start_year >= ROOKIE_SCALE_FIRST_YEAR:
        pool["sal_prev_pct"] = pool["salary_prev"] / cap_prev
        is_rookie_scale = (
            (pool["sal_prev_pct"] < ROOKIE_LADDER_SAL_PCT_PREV)
            & pool["age"].notna()
            & (pool["age"] <= ROOKIE_LADDER_MAX_AGE)
        )
        rookie_cap = pool["salary_prev"] * 1.5
        pool.loc[is_rookie_scale, "proj_capadj"] = pool.loc[is_rookie_scale, "proj_capadj"].clip(
            upper=rookie_cap[is_rookie_scale]
        )

    # Cap-relative error → 2025-26 dollar equivalent.
    pool["abs_err_pct_cap"]    = (pool["salary_curr"] - pool["proj_capadj"]).abs() / cap_curr
    pool["abs_err_today_M"]    = pool["abs_err_pct_cap"] * CURRENT_CAP / 1_000_000
    pool["signed_err_today_M"] = (pool["salary_curr"] - pool["proj_capadj"]) / cap_curr * CURRENT_CAP / 1_000_000
    pool["actual_today_M"]     = pool["salary_curr"] / cap_curr * CURRENT_CAP / 1_000_000
    pool["signed_in"]          = curr_season
    pool["start_year"]         = int(curr_season.split("-")[0])
    return pool[["signed_in", "start_year", "abs_err_today_M",
                 "signed_err_today_M", "actual_today_M"]]


def report(label: str, errs: pd.Series) -> None:
    n = len(errs)
    print(f"\n{label}  (n={n:,})")
    print("-" * 62)
    print(f"  {'Within':<12} {'Confidence':>14} {'Cumulative miss':>18}")
    for thresh in DOLLAR_THRESHOLDS_M:
        pct = (errs <= thresh).mean() * 100
        miss = n - int((errs <= thresh).sum())
        thresh_str = f"±${thresh:.1f}M"
        print(f"  {thresh_str:<12} {pct:>13.1f}% {miss:>14,} contracts")
    print(f"  {'Median |err|':<12} {f'${errs.median():.2f}M':>14}")


def main() -> None:
    print("Computing dollar-level confidence (2025-26 cap-equivalent)...")
    print("Iterating season pairs (2-3 minutes)...\n")
    pairs = [(SEASONS[i + 1], SEASONS[i]) for i in range(len(SEASONS) - 1)]
    rows = []
    for prev, curr in pairs:
        df = analyze_pair(prev, curr)
        if not df.empty:
            rows.append(df)
    if not rows:
        print("No data.")
        return
    combined = pd.concat(rows, ignore_index=True)

    print("=" * 62)
    print("DOLLAR-LEVEL CONFIDENCE  (errors in 2025-26 cap-equivalent $)")
    print("=" * 62)
    print(f"Current cap: ${CURRENT_CAP/1_000_000:.1f}M  ·  median = '50% of contracts")
    print(f"land within this dollar amount of the model's prediction'")

    # 1999+ headline.
    cba = combined[combined["start_year"] >= 1999]
    report("CBA-MAX ERA (1999+) — headline", cba["abs_err_today_M"])

    # Modern era — last 10 seasons.
    modern_years = sorted({int(s.split("-")[0]) for s in SEASONS[:10]})
    modern = combined[combined["start_year"].isin(modern_years)]
    report("MODERN ERA (last 10 seasons)", modern["abs_err_today_M"])

    # All-era.
    report("ALL-ERA (1984+)", combined["abs_err_today_M"])

    # Bias check.
    print("\nBIAS  (positive = model under-predicts, negative = over-predicts):")
    print(f"  CBA-max era median signed err: ${cba['signed_err_today_M'].median():+.2f}M")
    print(f"  Modern era    median signed err: ${modern['signed_err_today_M'].median():+.2f}M")

    # ── Segmented by contract size (1999+ pool) ──────────────────────────────
    print("\n" + "=" * 70)
    print("SEGMENTED BY CONTRACT SIZE  (1999+ pool, 2025-26 equivalent $)")
    print("=" * 70)
    print("These are the questions analysts actually care about: 'how well")
    print("does the model predict THIS tier of contract specifically?'")
    print()

    buckets = [
        ("Max / supermax", "≥ $40M",         cba[cba["actual_today_M"] >= 40]),
        ("Big stars",      "$25-40M",        cba[(cba["actual_today_M"] >= 25) & (cba["actual_today_M"] < 40)]),
        ("Mid-tier",       "$15-25M",        cba[(cba["actual_today_M"] >= 15) & (cba["actual_today_M"] < 25)]),
        ("Rotation",       "$7-15M",         cba[(cba["actual_today_M"] >=  7) & (cba["actual_today_M"] < 15)]),
        ("Min-ish",        "< $7M",          cba[cba["actual_today_M"] <  7]),
    ]

    thresholds = [1, 2, 3, 5, 8, 12]
    header = f"  {'Tier':<18} {'Range':<12} {'N':>5}"
    for t in thresholds:
        header += f"  ±${t}M".rjust(8)
    header += f"  {'Median':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, rng, sub in buckets:
        if sub.empty:
            continue
        n = len(sub)
        line = f"  {label:<18} {rng:<12} {n:>5,}"
        for t in thresholds:
            pct = (sub["abs_err_today_M"] <= t).mean() * 100
            line += f"  {pct:>6.0f}%"
        med = sub["abs_err_today_M"].median()
        line += f"  ${med:>5.1f}M"
        print(line)


if __name__ == "__main__":
    main()
