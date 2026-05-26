"""Cumulative accuracy analysis for the Barrett Score.

Runs the Track Record logic for every viable consecutive-season pair on
disk and prints aggregate directional + dollar accuracy. Designed to run
on Render where the full cache lives (raw_*.parquet, salary scrapes,
D-LEBRON pulls). On a fresh local checkout it'll skip most pairs.

Usage:
    python analyze_accuracy.py            # cumulative summary only
    python analyze_accuracy.py --verbose  # also print per-pair breakdown

Requires the ranked-projected parquet pipeline to be seeded — i.e. the
Rankings page can render those seasons. If not, run seed_cache.py first.
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import pandas as pd

from utils import SEASONS, build_ranked_projected


TOP_N = 20  # Top-20 underpaid/overpaid calls per season


def analyze_pair(prev_season: str, curr_season: str) -> dict | None:
    """Return accuracy metrics for one (prev, curr) season pair.
    Returns None if either season has no data."""
    try:
        prev_df = build_ranked_projected(prev_season)
        curr_df = build_ranked_projected(curr_season)
    except Exception as e:
        return {"prev": prev_season, "curr": curr_season, "error": str(e)}
    if prev_df.empty or curr_df.empty:
        return None

    prev_df = prev_df[prev_df["salary"] > 0].copy()
    if prev_df.empty:
        return None

    prev_slim = prev_df[["PLAYER_ID", "Player", "salary",
                         "projected_salary", "value_diff"]].rename(columns={
        "salary":           "salary_prev",
        "projected_salary": "proj_prev",
        "value_diff":       "value_diff_prev",
    })
    curr_slim = curr_df[["PLAYER_ID", "salary"]].rename(columns={
        "salary": "salary_curr",
    })
    m = prev_slim.merge(curr_slim, on="PLAYER_ID", how="left")
    m["salary_change"] = m["salary_curr"].fillna(0) - m["salary_prev"]
    m["pct_change"]    = (m["salary_change"] / m["salary_prev"]).where(m["salary_prev"] > 0)
    m["still"]         = m["salary_curr"].fillna(0) > 0
    m["abs_err"]       = (m["salary_curr"].fillna(0) - m["proj_prev"]).abs()
    m["signed_err"]    = m["salary_curr"].fillna(0) - m["proj_prev"]

    # ── Directional accuracy ──────────────────────────────────────────────────
    under = m.nsmallest(TOP_N, "value_diff_prev").copy()
    over  = m.nlargest(TOP_N, "value_diff_prev").copy()

    def _under_correct(r):
        if not r["still"]:           return None
        if pd.isna(r["pct_change"]): return None
        if r["pct_change"] >= 0.15:  return 1
        if r["pct_change"] <= -0.10: return 0
        return None  # ↔ Flat — not gradeable

    def _over_correct(r):
        if not r["still"]:           return 1   # left league = correct call
        if pd.isna(r["pct_change"]): return None
        if r["pct_change"] <= -0.10: return 1
        if r["pct_change"] >= 0.15:  return 0
        return None  # ↔ Flat — not gradeable

    under["correct"] = under.apply(_under_correct, axis=1)
    over["correct"]  = over.apply(_over_correct, axis=1)
    under_g = under["correct"].dropna()
    over_g  = over["correct"].dropna()

    # ── Dollar accuracy (filtered to YoY-changed deals only — players on
    # ── locked multi-year contracts are trivially "predicted") ───────────────
    pool = m[
        (m["proj_prev"].fillna(0) > 0)
        & (m["salary_curr"].fillna(0) > 0)
        & (m["pct_change"].abs() >= 0.15)
    ].copy()

    return {
        "prev":           prev_season,
        "curr":           curr_season,
        "n_prev_pool":    len(prev_df),
        "n_matched":      int(m["still"].sum()),
        # Directional
        "under_correct":  int(under_g.sum()),
        "under_graded":   len(under_g),
        "under_acc":      under_g.mean() * 100 if len(under_g) else None,
        "over_correct":   int(over_g.sum()),
        "over_graded":    len(over_g),
        "over_acc":       over_g.mean() * 100 if len(over_g) else None,
        # Dollar
        "n_dollar_pool":  len(pool),
        "median_err_M":   pool["abs_err"].median() / 1_000_000 if len(pool) else None,
        "within_5M_pct":  (pool["abs_err"] <= 5_000_000).mean() * 100 if len(pool) else None,
        "within_10M_pct": (pool["abs_err"] <= 10_000_000).mean() * 100 if len(pool) else None,
        "median_bias_M":  pool["signed_err"].median() / 1_000_000 if len(pool) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-season-pair breakdown.")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limit analysis to N most recent pairs.")
    args = parser.parse_args()

    # Iterate through consecutive season pairs (newest to oldest).
    pairs = []
    for i, prev in enumerate(SEASONS[1:], start=1):
        curr = SEASONS[i - 1]
        pairs.append((prev, curr))
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    print(f"Analyzing {len(pairs)} season pairs...\n")
    rows = []
    for prev, curr in pairs:
        result = analyze_pair(prev, curr)
        if result is None:
            if args.verbose:
                print(f"  skip {prev} → {curr}: no data on disk")
            continue
        if "error" in result:
            if args.verbose:
                print(f"  skip {prev} → {curr}: {result['error']}")
            continue
        rows.append(result)
        if args.verbose:
            ua = f"{result['under_acc']:.0f}%" if result['under_acc'] is not None else "—"
            oa = f"{result['over_acc']:.0f}%"  if result['over_acc']  is not None else "—"
            me = f"${result['median_err_M']:.1f}M" if result['median_err_M'] is not None else "—"
            w5 = f"{result['within_5M_pct']:.0f}%" if result['within_5M_pct'] is not None else "—"
            print(f"  ok   {prev} → {curr}: "
                  f"under={ua} ({result['under_correct']}/{result['under_graded']}), "
                  f"over={oa} ({result['over_correct']}/{result['over_graded']}), "
                  f"$ med err={me}, within $5M={w5} (n={result['n_dollar_pool']})")

    if not rows:
        print("\nNo valid season pairs. Has the cache been seeded? Try `python seed_cache.py`.")
        return

    df = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print(f"CUMULATIVE ACCURACY · {len(df)} season pairs")
    print("=" * 72)

    # ── Directional ──────────────────────────────────────────────────────────
    tuc = df["under_correct"].sum()
    tug = df["under_graded"].sum()
    toc = df["over_correct"].sum()
    tog = df["over_graded"].sum()
    cc  = tuc + toc
    cg  = tug + tog

    print(f"\nDIRECTIONAL ACCURACY (top-20 calls per side per season):")
    if tug:
        print(f"  Underpaid calls:  {tuc:4d}/{tug:4d} = {tuc/tug*100:5.1f}%")
    if tog:
        print(f"  Overpaid calls:   {toc:4d}/{tog:4d} = {toc/tog*100:5.1f}%")
    if cg:
        print(f"  Combined:         {cc:4d}/{cg:4d} = {cc/cg*100:5.1f}%")

    # ── Dollar ───────────────────────────────────────────────────────────────
    total_dollar_n = int(df["n_dollar_pool"].sum())
    if total_dollar_n:
        weighted_w5  = (df["within_5M_pct"]  * df["n_dollar_pool"]).sum() / total_dollar_n
        weighted_w10 = (df["within_10M_pct"] * df["n_dollar_pool"]).sum() / total_dollar_n
        weighted_bias = (df["median_bias_M"] * df["n_dollar_pool"]).sum() / total_dollar_n
        median_of_medians = df["median_err_M"].median()

        print(f"\nDOLLAR-AMOUNT ACCURACY (players with ≥15% YoY salary change):")
        print(f"  Sample size:       {total_dollar_n} predictions across {len(df)} pairs")
        print(f"  Median |error|:    ${median_of_medians:.1f}M  (median of pair-level medians)")
        print(f"  Within $5M:        {weighted_w5:5.1f}%")
        print(f"  Within $10M:       {weighted_w10:5.1f}%")
        print(f"  Median bias:       ${weighted_bias:+5.1f}M  (positive = market pays more,"
              " mostly cap inflation)")

    # ── Per-pair table ───────────────────────────────────────────────────────
    if args.verbose:
        show = df[[
            "prev", "curr",
            "under_acc", "over_acc",
            "n_dollar_pool", "median_err_M", "within_5M_pct", "within_10M_pct",
        ]].copy()
        show.columns = ["Prev", "Curr", "Under%", "Over%", "$ n", "Med |err|", "≤$5M%", "≤$10M%"]
        print("\nPer-pair breakdown:")
        print(show.to_string(index=False, float_format=lambda v: f"{v:5.1f}"))


if __name__ == "__main__":
    main()
