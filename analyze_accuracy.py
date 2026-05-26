"""Rigorous accuracy analysis for the Barrett Score.

The question this answers: when a player signs a NEW contract, how close
was the Barrett Score's projection to what they actually signed for?

Three methodology decisions that matter:

1. Error is measured in % of salary cap, not absolute dollars. $5M means
   wildly different things in 1993 vs. 2025; cap-relative errors are
   era-fair.

2. The dollar pool filters to players with ≥25% YoY salary change. That
   filters out CBA-mandated rookie-scale step-ups, supermax extensions
   that happen automatically, and players still mid-deal. What's left
   is approximately "real new contracts negotiated this offseason."

3. The previous year's projected salary is scaled by cap growth before
   being compared to current actual. Without this, the model gets
   penalized for cap inflation it had no way to predict.

4. "Left league" is non-gradeable, not auto-correct. Retirement and
   injuries are real-world noise the model shouldn't get credit for.

Usage:
    python analyze_accuracy.py            # cumulative summary only
    python analyze_accuracy.py --verbose  # also print per-pair breakdown
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from utils import SEASONS, build_ranked_projected


TOP_N = 20
NEW_DEAL_PCT_THRESHOLD = 0.25  # ≥25% YoY salary change = "new contract" proxy


# NBA salary cap by season (in millions, applied to the "curr" side of a pair).
# Sources: NBA / Spotrac historical cap. Pre-1984 had no cap.
SALARY_CAP_M = {
    "1984-85":   3.6,  "1985-86":   4.2,  "1986-87":   4.9,  "1987-88":   6.2,
    "1988-89":   7.2,  "1989-90":   9.8,  "1990-91":  11.9,  "1991-92":  12.5,
    "1992-93":  14.0,  "1993-94":  15.2,  "1994-95":  15.9,  "1995-96":  23.0,
    "1996-97":  24.4,  "1997-98":  26.9,  "1998-99":  30.0,  "1999-00":  34.0,
    "2000-01":  35.5,  "2001-02":  42.5,  "2002-03":  40.3,  "2003-04":  43.8,
    "2004-05":  43.9,  "2005-06":  49.5,  "2006-07":  53.1,  "2007-08":  55.6,
    "2008-09":  58.7,  "2009-10":  57.7,  "2010-11":  58.0,  "2011-12":  58.0,
    "2012-13":  58.0,  "2013-14":  58.7,  "2014-15":  63.1,  "2015-16":  70.0,
    "2016-17":  94.1,  "2017-18":  99.1,  "2018-19": 101.9,  "2019-20": 109.1,
    "2020-21": 109.1,  "2021-22": 112.4,  "2022-23": 123.7,  "2023-24": 136.0,
    "2024-25": 140.6,  "2025-26": 154.6,
}


def _cap(season: str) -> float:
    """Salary cap in $M for a season. Falls back to 1.0 if pre-cap era so
    division still works (and we'll filter those out anyway)."""
    return SALARY_CAP_M.get(season, 1.0) * 1_000_000


def analyze_pair(prev_season: str, curr_season: str) -> dict | None:
    try:
        prev_df = build_ranked_projected(prev_season)
        curr_df = build_ranked_projected(curr_season)
    except Exception as e:
        return {"prev": prev_season, "curr": curr_season, "error": str(e)}
    if prev_df.empty or curr_df.empty:
        return None
    if prev_season not in SALARY_CAP_M or curr_season not in SALARY_CAP_M:
        return None  # pre-1984 — no cap, skip

    prev_df = prev_df[prev_df["salary"] > 0].copy()
    if prev_df.empty:
        return None

    cap_prev = _cap(prev_season)
    cap_curr = _cap(curr_season)
    cap_ratio = cap_curr / cap_prev

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
    m["still"]       = m["salary_curr"].fillna(0) > 0
    m["pct_change"]  = ((m["salary_curr"].fillna(0) - m["salary_prev"])
                        / m["salary_prev"]).where(m["salary_prev"] > 0)

    # Scale the previous projection by cap growth — gives the model credit
    # for "predicting the rank" rather than penalizing it for cap inflation.
    m["proj_capadj"] = m["proj_prev"] * cap_ratio
    m["abs_err"]     = (m["salary_curr"].fillna(0) - m["proj_capadj"]).abs()
    m["signed_err"]  = m["salary_curr"].fillna(0) - m["proj_capadj"]
    m["pct_cap_err"] = m["abs_err"] / cap_curr * 100  # error as % of curr cap

    # ── Directional accuracy (strict: no auto-credit for retirement) ─────────
    under = m.nsmallest(TOP_N, "value_diff_prev").copy()
    over  = m.nlargest(TOP_N, "value_diff_prev").copy()

    def _under_correct(r):
        if not r["still"]:           return None  # not gradeable (was: 1)
        if pd.isna(r["pct_change"]): return None
        if r["pct_change"] >= 0.15:  return 1
        if r["pct_change"] <= -0.10: return 0
        return None  # flat — not gradeable

    def _over_correct(r):
        if not r["still"]:           return None  # not gradeable (was: 1)
        if pd.isna(r["pct_change"]): return None
        if r["pct_change"] <= -0.10: return 1
        if r["pct_change"] >= 0.15:  return 0
        return None

    under["correct"] = under.apply(_under_correct, axis=1)
    over["correct"]  = over.apply(_over_correct, axis=1)
    under_g = under["correct"].dropna()
    over_g  = over["correct"].dropna()

    # ── Dollar accuracy: real new-contract pool only ─────────────────────────
    # Players who actually signed a new deal between seasons (proxy: ≥25%
    # YoY salary change). Cap-relative error.
    pool = m[
        (m["proj_prev"].fillna(0) > 0)
        & (m["salary_curr"].fillna(0) > 0)
        & (m["pct_change"].abs() >= NEW_DEAL_PCT_THRESHOLD)
    ].copy()

    return {
        "prev":           prev_season,
        "curr":           curr_season,
        "cap_curr_M":     cap_curr / 1_000_000,
        "n_prev_pool":    len(prev_df),
        # Directional
        "under_correct":  int(under_g.sum()) if len(under_g) else 0,
        "under_graded":   len(under_g),
        "under_acc":      under_g.mean() * 100 if len(under_g) else None,
        "over_correct":   int(over_g.sum()) if len(over_g) else 0,
        "over_graded":    len(over_g),
        "over_acc":       over_g.mean() * 100 if len(over_g) else None,
        # Dollar (cap-relative, real new-deal pool)
        "n_dollar_pool":   len(pool),
        "median_err_cap": pool["pct_cap_err"].median() if len(pool) else None,
        "within_5cap":    (pool["pct_cap_err"] <= 5.0).mean()  * 100 if len(pool) else None,
        "within_10cap":   (pool["pct_cap_err"] <= 10.0).mean() * 100 if len(pool) else None,
        "median_err_M":   pool["abs_err"].median() / 1_000_000 if len(pool) else None,
        "median_bias_cap": (pool["signed_err"] / cap_curr * 100).median() if len(pool) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    pairs = [(SEASONS[i], SEASONS[i - 1]) for i in range(1, len(SEASONS))]
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    print(f"Analyzing {len(pairs)} season pairs (post-1984, when cap data exists)...\n")
    rows = []
    for prev, curr in pairs:
        result = analyze_pair(prev, curr)
        if result is None:
            if args.verbose:
                print(f"  skip {prev} → {curr}: no data / pre-cap era")
            continue
        if "error" in result:
            if args.verbose:
                print(f"  skip {prev} → {curr}: {result['error']}")
            continue
        rows.append(result)
        if args.verbose:
            ua = f"{result['under_acc']:.0f}%" if result['under_acc'] is not None else "—"
            oa = f"{result['over_acc']:.0f}%"  if result['over_acc']  is not None else "—"
            me = f"{result['median_err_cap']:.1f}%" if result['median_err_cap'] is not None else "—"
            w5 = f"{result['within_5cap']:.0f}%"   if result['within_5cap'] is not None else "—"
            mm = f"${result['median_err_M']:.1f}M" if result['median_err_M'] is not None else "—"
            print(f"  ok   {prev} → {curr}: "
                  f"under={ua} ({result['under_correct']}/{result['under_graded']}), "
                  f"over={oa} ({result['over_correct']}/{result['over_graded']}), "
                  f"err={me} of cap ({mm}), within 5% cap={w5} (n={result['n_dollar_pool']})")

    if not rows:
        print("\nNo valid season pairs.")
        return

    df = pd.DataFrame(rows)

    # ── Headline ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print(f"BARRETT SCORE — NEW CONTRACT PREDICTION ACCURACY · {len(df)} season pairs")
    print("=" * 76)
    print("Filtered to players with ≥25% YoY salary change (real new deals only).")
    print("Error measured as % of that season's salary cap (era-fair).")
    print("Projections cap-scaled to match current-year dollars.")
    print("'Left league' is non-gradeable, not auto-correct.")
    print()

    # ── Directional ──────────────────────────────────────────────────────────
    tuc = df["under_correct"].sum()
    tug = df["under_graded"].sum()
    toc = df["over_correct"].sum()
    tog = df["over_graded"].sum()
    cc, cg = tuc + toc, tug + tog

    print("DIRECTIONAL ACCURACY (top-20 calls per side per season):")
    if tug:
        print(f"  Underpaid called correctly: {tuc:4d}/{tug:4d} = {tuc/tug*100:5.1f}%")
    if tog:
        print(f"  Overpaid called correctly:  {toc:4d}/{tog:4d} = {toc/tog*100:5.1f}%")
    if cg:
        print(f"  Combined:                   {cc:4d}/{cg:4d} = {cc/cg*100:5.1f}%")

    # ── Dollar (cap-relative) ────────────────────────────────────────────────
    total_n = int(df["n_dollar_pool"].sum())
    if total_n:
        w5  = (df["within_5cap"]  * df["n_dollar_pool"]).sum() / total_n
        w10 = (df["within_10cap"] * df["n_dollar_pool"]).sum() / total_n
        bias = (df["median_bias_cap"] * df["n_dollar_pool"]).sum() / total_n
        med_of_med_cap = df["median_err_cap"].median()
        med_of_med_M   = df["median_err_M"].median()

        print(f"\nDOLLAR-AMOUNT ACCURACY (new contracts, cap-relative error):")
        print(f"  Sample size:                {total_n} new-contract predictions across {len(df)} pairs")
        print(f"  Median |error|:             {med_of_med_cap:5.1f}% of cap  (~${med_of_med_M:.1f}M in current dollars)")
        print(f"  Within 5%  of cap (~$8M):   {w5:5.1f}%")
        print(f"  Within 10% of cap (~$15M):  {w10:5.1f}%")
        print(f"  Median bias:               {bias:+5.1f}% of cap "
              f"({'market pays more' if bias > 0 else 'model overshoots'})")

    # ── Modern subset (last 10 years) for marketing-defensible numbers ──────
    modern = df[df["curr"].isin([s for s in SEASONS[:10]])]
    if len(modern):
        m_n = int(modern["n_dollar_pool"].sum())
        if m_n:
            m_w5  = (modern["within_5cap"]  * modern["n_dollar_pool"]).sum() / m_n
            m_w10 = (modern["within_10cap"] * modern["n_dollar_pool"]).sum() / m_n
            m_med = modern["median_err_cap"].median()
            m_medM = modern["median_err_M"].median()
            m_tuc, m_tug = modern["under_correct"].sum(), modern["under_graded"].sum()
            m_toc, m_tog = modern["over_correct"].sum(),  modern["over_graded"].sum()

            print(f"\nMODERN ERA ONLY (last 10 season pairs — defensible v. cap-inflated past):")
            print(f"  Directional:                under={m_tuc}/{m_tug} ({m_tuc/m_tug*100:.0f}%) "
                  f"· over={m_toc}/{m_tog} ({m_toc/m_tog*100:.0f}%)")
            print(f"  Sample size:                {m_n} new-contract predictions")
            print(f"  Median |error|:             {m_med:5.1f}% of cap  (~${m_medM:.1f}M)")
            print(f"  Within 5%  of cap (~$8M):   {m_w5:5.1f}%")
            print(f"  Within 10% of cap (~$15M):  {m_w10:5.1f}%")

    if args.verbose:
        show = df[[
            "prev", "curr", "cap_curr_M",
            "under_acc", "over_acc",
            "n_dollar_pool", "median_err_cap", "median_err_M",
            "within_5cap", "within_10cap",
        ]].copy()
        show.columns = ["Prev", "Curr", "Cap $M", "Under%", "Over%",
                        "$ n", "Err %cap", "Err $M", "≤5%cap", "≤10%cap"]
        print("\nPer-pair breakdown:")
        print(show.to_string(index=False, float_format=lambda v: f"{v:5.1f}"))


if __name__ == "__main__":
    main()
