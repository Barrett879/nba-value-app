"""Experiment: does adding age curves + positional buckets improve accuracy?

Standalone test — does NOT modify the live model. Fits adjustments on a
training set (2014-15 → 2021-22 season pairs) and grades on a held-out
test set (2022-23 → 2024-25 pairs). Out-of-sample so the numbers are
honest, not overfit.

Approach
--------
The current model: projected_salary = salary of player at same Barrett rank.
This script tries three calibration layers on top:

  1. Age multiplier — fitted as median(actual / baseline_proj) per age
     bucket on the training set, applied to test predictions.
  2. Position multiplier — same idea but per BBRef position bucket.
  3. Combined — both multipliers stacked.

These are simple, interpretable, and they ride on top of the existing
Barrett Score (no changes to weights, no changes to the site).

Usage:
    python test_age_position.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from utils import (
        SEASONS, normalize, season_to_espn_year,
    build_ranked_projected, fetch_league_stats, fetch_bref_positions,
    SALARY_CAP_M,
    age_bucket as _age_bucket,
)


# ── NBA cap by season (matches analyze_accuracy.py) ─────────────────────────
# Train/test split — chronological hold-out.
TRAIN_PAIRS = [
    ("2014-15", "2015-16"), ("2015-16", "2016-17"), ("2016-17", "2017-18"),
    ("2017-18", "2018-19"), ("2018-19", "2019-20"), ("2019-20", "2020-21"),
    ("2020-21", "2021-22"), ("2021-22", "2022-23"),
]
TEST_PAIRS = [
    ("2022-23", "2023-24"), ("2023-24", "2024-25"), ("2024-25", "2025-26"),
]

NEW_DEAL_THRESHOLD = 0.25  # ≥25% YoY salary change = new-contract proxy


def _norm_position(pos: str) -> str:
    """Collapse BBRef position strings (e.g. 'PG-SG', 'SF') to one of
    PG/SG/SF/PF/C. Hyphenated positions take the first."""
    if not pos:
        return "UNK"
    first = pos.split("-")[0].strip().upper()
    if first in {"PG", "SG", "SF", "PF", "C"}:
        return first
    return "UNK"




def build_pair_dataset(prev: str, curr: str) -> pd.DataFrame | None:
    """One row per qualifying player. Combines Barrett projection, age,
    position, and the actual current-season salary outcome."""
    if prev not in SALARY_CAP_M or curr not in SALARY_CAP_M:
        return None
    cap_prev_M = SALARY_CAP_M[prev]
    cap_curr_M = SALARY_CAP_M[curr]
    cap_ratio = cap_curr_M / cap_prev_M

    try:
        prev_df = build_ranked_projected(prev)
        curr_df = build_ranked_projected(curr)
    except Exception:
        return None
    if prev_df.empty or curr_df.empty:
        return None

    # Pull AGE from the per-game NBA Stats parquet for the prev season.
    raw_prev = fetch_league_stats(prev, "Regular Season")
    if raw_prev.empty or "AGE" not in raw_prev.columns:
        return None
    age_lookup = dict(zip(raw_prev["PLAYER_ID"], raw_prev["AGE"]))

    # Pull positions from BBRef cache.
    try:
        pos_lookup_norm = fetch_bref_positions(season_to_espn_year(prev), cache_v=3)
    except Exception:
        pos_lookup_norm = {}

    prev_df = prev_df[prev_df["salary"] > 0].copy()
    curr_slim = curr_df[["PLAYER_ID", "salary"]].rename(columns={"salary": "salary_curr"})
    m = prev_df.merge(curr_slim, on="PLAYER_ID", how="left")

    m["age"] = m["PLAYER_ID"].map(age_lookup)
    m["pos"] = m["Player"].map(lambda n: _norm_position(pos_lookup_norm.get(normalize(n), "")))
    m["age_bucket"] = m["age"].apply(_age_bucket)

    # Cap-scale the prev projection to current dollars (matches analyze_accuracy).
    m["proj_capadj"] = m["projected_salary"] * cap_ratio
    m["still"] = m["salary_curr"].fillna(0) > 0
    m["pct_change"] = ((m["salary_curr"].fillna(0) - m["salary"]) / m["salary"]).where(m["salary"] > 0)
    m["cap_curr"] = cap_curr_M * 1_000_000
    m["cap_curr_M"] = cap_curr_M

    return m[[
        "PLAYER_ID", "Player", "age", "age_bucket", "pos",
        "salary", "projected_salary", "proj_capadj",
        "salary_curr", "still", "pct_change",
        "cap_curr", "cap_curr_M",
    ]]


def _filter_new_deal_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Drop locked multi-year contracts — keep only players with ≥25% YoY
    salary change. Same filter used in analyze_accuracy.py."""
    return df[
        df["still"]
        & (df["proj_capadj"] > 0)
        & (df["pct_change"].abs() >= NEW_DEAL_THRESHOLD)
    ].copy()


def fit_age_multipliers(train: pd.DataFrame) -> dict:
    """Per-age-bucket: median(actual / baseline). The multiplier corrects
    the model's systematic over/under-projection for that age."""
    train = train[train["proj_capadj"] > 0].copy()
    train["ratio"] = train["salary_curr"] / train["proj_capadj"]
    mults = train.groupby("age_bucket")["ratio"].median().to_dict()
    return mults


def fit_position_multipliers(train: pd.DataFrame) -> dict:
    """Same idea but per position bucket."""
    train = train[train["proj_capadj"] > 0].copy()
    train["ratio"] = train["salary_curr"] / train["proj_capadj"]
    mults = train.groupby("pos")["ratio"].median().to_dict()
    return mults


def score(df: pd.DataFrame, proj_col: str = "proj_capadj") -> dict:
    """Compute accuracy stats for one prediction column."""
    df = df[(df[proj_col] > 0) & df["still"]].copy()
    err_pct = (df["salary_curr"] - df[proj_col]).abs() / df["cap_curr"] * 100
    signed_pct = (df["salary_curr"] - df[proj_col]) / df["cap_curr"] * 100
    return {
        "n":              len(df),
        "median_err_cap": float(err_pct.median()) if len(df) else None,
        "mean_err_cap":   float(err_pct.mean())   if len(df) else None,
        "within_5":       float((err_pct <= 5.0).mean() * 100) if len(df) else None,
        "within_10":      float((err_pct <= 10.0).mean() * 100) if len(df) else None,
        "median_bias":    float(signed_pct.median()) if len(df) else None,
    }


def _print_score(label: str, s: dict, indent: str = "  "):
    print(f"{indent}{label}")
    print(f"{indent}  n            = {s['n']}")
    print(f"{indent}  Median |err| = {s['median_err_cap']:5.2f}% of cap")
    print(f"{indent}  Mean   |err| = {s['mean_err_cap']:5.2f}% of cap")
    print(f"{indent}  Within 5%    = {s['within_5']:5.1f}%")
    print(f"{indent}  Within 10%   = {s['within_10']:5.1f}%")
    print(f"{indent}  Median bias  = {s['median_bias']:+5.2f}% of cap")


def main():
    print("Loading data for train + test pairs...")
    train_dfs, test_dfs = [], []
    for prev, curr in TRAIN_PAIRS:
        d = build_pair_dataset(prev, curr)
        if d is None or d.empty:
            print(f"  skip TRAIN {prev} → {curr}")
            continue
        d["pair"] = f"{prev} → {curr}"
        train_dfs.append(d)
        print(f"  load TRAIN {prev} → {curr}: {len(d)} rows")
    for prev, curr in TEST_PAIRS:
        d = build_pair_dataset(prev, curr)
        if d is None or d.empty:
            print(f"  skip TEST  {prev} → {curr}")
            continue
        d["pair"] = f"{prev} → {curr}"
        test_dfs.append(d)
        print(f"  load TEST  {prev} → {curr}: {len(d)} rows")

    if not test_dfs:
        print("\nNo test data available.")
        return

    train_all = pd.concat(train_dfs, ignore_index=True)
    test_all  = pd.concat(test_dfs,  ignore_index=True)
    train_pool = _filter_new_deal_pool(train_all)
    test_pool  = _filter_new_deal_pool(test_all)

    print(f"\nFiltered pools (≥25% YoY change):")
    print(f"  Train: {len(train_pool)} new-contract predictions")
    print(f"  Test:  {len(test_pool)} new-contract predictions")

    # ── Fit ───────────────────────────────────────────────────────────────────
    age_mults = fit_age_multipliers(train_pool)
    pos_mults = fit_position_multipliers(train_pool)

    print(f"\nLearned AGE multipliers (from training set):")
    for b in ["≤22", "23-25", "26-28", "29-31", "32-34", "35+", "UNK"]:
        if b in age_mults:
            print(f"  {b:7s}: ×{age_mults[b]:.3f}")
    print(f"\nLearned POSITION multipliers (from training set):")
    for p in ["PG", "SG", "SF", "PF", "C", "UNK"]:
        if p in pos_mults:
            print(f"  {p:5s}: ×{pos_mults[p]:.3f}")

    # ── Apply to test set ─────────────────────────────────────────────────────
    test_pool = test_pool.copy()
    test_pool["proj_age"]      = test_pool["proj_capadj"] * test_pool["age_bucket"].map(age_mults).fillna(1.0)
    test_pool["proj_pos"]      = test_pool["proj_capadj"] * test_pool["pos"].map(pos_mults).fillna(1.0)
    test_pool["proj_combined"] = (test_pool["proj_capadj"]
                                  * test_pool["age_bucket"].map(age_mults).fillna(1.0)
                                  * test_pool["pos"].map(pos_mults).fillna(1.0))

    print("\n" + "=" * 76)
    print("OUT-OF-SAMPLE RESULTS · test set (3 most recent season pairs)")
    print("=" * 76)

    base = score(test_pool, "proj_capadj")
    age  = score(test_pool, "proj_age")
    pos  = score(test_pool, "proj_pos")
    comb = score(test_pool, "proj_combined")

    _print_score("BASELINE (current model, cap-scaled)", base)
    print()
    _print_score("+ AGE multiplier", age)
    print()
    _print_score("+ POSITION multiplier", pos)
    print()
    _print_score("+ BOTH (age × position)", comb)

    # ── Δ summary ─────────────────────────────────────────────────────────────
    def _delta(s):
        return {
            "median_err_cap": s["median_err_cap"] - base["median_err_cap"],
            "within_5":       s["within_5"]       - base["within_5"],
            "within_10":      s["within_10"]      - base["within_10"],
        }

    print("\n" + "=" * 76)
    print("IMPROVEMENT vs. BASELINE (out-of-sample)")
    print("=" * 76)
    print(f"  {'Variant':<22s} {'Within 5%':>11s} {'Within 10%':>12s} {'Median |err|':>14s}")
    for name, s in [("+ AGE only", age), ("+ POSITION only", pos), ("+ BOTH", comb)]:
        d = _delta(s)
        print(f"  {name:<22s} {d['within_5']:+10.2f}pp {d['within_10']:+11.2f}pp "
              f"{d['median_err_cap']:+13.2f}pp")

    print("\n" + "=" * 76)
    print("INTERPRETATION")
    print("=" * 76)
    print("These are out-of-sample numbers — the multipliers were fit on 2014-22")
    print("data and applied to 2022-25. Whatever improvement you see here is real,")
    print("not memorized. A ≥1pp gain on Within 5% is meaningful; <0.5pp is noise.")


if __name__ == "__main__":
    main()
