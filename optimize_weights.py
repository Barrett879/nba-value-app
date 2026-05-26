"""Optimize the Barrett Score box-score weights for next-contract accuracy.

The current weights (utils.base_score) are basketball-intuitive but never
fit to data:
    PTS=1.0, AST=1.5, OREB=0.5, DREB=0.333, BLK=0.5, STL=0.667,
    TOV=-0.667, PF=-0.333, D-LEBRON=2.0, efficiency_adj=2.0

This script:
1. Loads every season pair with cached data
2. Filters to players with ≥25% YoY salary change (real new contracts)
3. Runs differential evolution to find weights that minimize median
   |error| as % of salary cap across all real new-contract predictions
4. Reports the optimal weights + the resulting accuracy improvement

PTS is pinned to weight 1.0 since the rank mapping is scale-invariant —
only relative weights matter. The optimizer searches the other 9 weights.

Usage:
    python optimize_weights.py              # full search (~5-10 min)
    python optimize_weights.py --quick      # smaller search budget
    python optimize_weights.py --train-only-modern  # fit on last 10 pairs only
"""
import argparse
import sys
import warnings
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from utils import (
    SEASONS, fetch_league_stats, build_raw, apply_rankings,
)


# ── Hand-rolled optimizer (avoids scipy dependency on Render) ────────────────
def random_search_plus_coordinate_descent(
    objective, x0, bounds, n_restarts: int = 12,
    n_random: int = 200, cd_rounds: int = 4, cd_steps: int = 25,
    rng_seed: int = 42, verbose: bool = False,
):
    """Stochastic + coordinate-descent optimizer.

    1. Random-search the bounded box for n_random candidates, keep top K
    2. From each (plus the seed x0), run cd_rounds of coordinate descent
       — for each coord, sweep cd_steps values in its bound and keep the
       best
    3. Return the global best across all restarts
    """
    rng = np.random.default_rng(rng_seed)
    bounds = np.array(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    dim = len(x0)

    # Phase 1: random search
    samples = rng.uniform(lo, hi, size=(n_random, dim))
    samples = np.vstack([samples, x0[None, :]])  # always include the seed
    losses = np.array([objective(x) for x in samples])
    order = np.argsort(losses)
    starts = samples[order[:n_restarts]].copy()
    best_x = samples[order[0]].copy()
    best_loss = float(losses[order[0]])
    if verbose:
        print(f"    Random search: best loss = {best_loss:.4f}")

    # Phase 2: coordinate descent from each start
    for s_idx, start in enumerate(starts):
        x = start.copy()
        cur = float(objective(x))
        for _round in range(cd_rounds):
            improved = False
            # Try coordinates in shuffled order each round.
            for d in rng.permutation(dim):
                grid = np.linspace(lo[d], hi[d], cd_steps)
                best_v, best_c = x[d], cur
                for v in grid:
                    x_try = x.copy()
                    x_try[d] = v
                    c = float(objective(x_try))
                    if c < best_c - 1e-6:
                        best_c, best_v = c, v
                if best_v != x[d]:
                    x[d] = best_v
                    cur = best_c
                    improved = True
            if not improved:
                break
        if cur < best_loss:
            best_loss = cur
            best_x = x.copy()
            if verbose:
                print(f"    Restart {s_idx+1}/{len(starts)}: new best {cur:.4f}")

    class _Result:
        pass
    r = _Result()
    r.x = best_x
    r.fun = best_loss
    return r


NEW_DEAL_PCT_THRESHOLD = 0.25

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


# Current defaults from utils.base_score (PTS pinned to 1.0 since model is
# scale-invariant — only relative weights matter for rank).
CURRENT_WEIGHTS = {
    "AST":     1.5,
    "OREB":    0.5,
    "DREB":    1.0 / 3,
    "BLK":     0.5,
    "STL":     1.0 / 1.5,
    "TOV":    -1.0 / 1.5,
    "PF":     -1.0 / 3,
    "DLEBRON": 2.0,
    "EFFADJ":  2.0,
}
WEIGHT_KEYS = list(CURRENT_WEIGHTS.keys())

# Reasonable bounds: positive stats stay positive, negative stats stay
# negative. Width is generous enough to find improvements but tight enough
# to keep the optimizer focused.
WEIGHT_BOUNDS = {
    "AST":     (0.2, 3.0),
    "OREB":    (0.0, 2.0),
    "DREB":    (0.0, 2.0),
    "BLK":     (0.0, 2.0),
    "STL":     (0.0, 2.0),
    "TOV":    (-2.5, 0.0),
    "PF":     (-2.0, 0.0),
    "DLEBRON": (0.0, 5.0),
    "EFFADJ":  (0.0, 5.0),
}


def load_pair_data(prev_season: str, curr_season: str) -> pd.DataFrame | None:
    """Returns one row per player in prev_season's pool with:
    - raw stats (PTS/AST/OREB/DREB/BLK/STL/TOV/PF)
    - d_lebron, efficiency_adj, avail_mult (from build_raw)
    - salary_prev (player's actual salary in prev_season)
    - salary_curr (player's actual salary in curr_season, NaN if left league)
    - rank_idx (index into sorted-salary array for this season — needed
                to look up projection given any candidate weighting)
    Also returns the sorted salary array as a side payload via the DF's
    .attrs dict for use during optimization.
    """
    if prev_season not in SALARY_CAP_M or curr_season not in SALARY_CAP_M:
        return None
    try:
        raw_stats = fetch_league_stats(prev_season, "Regular Season")
        if raw_stats.empty:
            return None
        ranked = apply_rankings(build_raw(prev_season))
        if ranked.empty:
            return None
        curr_ranked = apply_rankings(build_raw(curr_season))
        if curr_ranked.empty:
            return None
    except Exception:
        return None

    raw_keep = raw_stats[["PLAYER_NAME", "PTS", "AST",
                          "OREB", "DREB", "BLK", "STL", "TOV", "PF"]].copy()
    ranked2 = ranked[[
        "PLAYER_ID", "Player", "salary",
        "avail_mult", "d_lebron", "efficiency_adj",
    ]].copy()
    # Merge raw box-score onto ranked by normalized name (PLAYER_ID isn't
    # on the league_stats parquet for every season).
    ranked2["_key"] = ranked2["Player"].str.lower().str.strip()
    raw_keep["_key"] = raw_keep["PLAYER_NAME"].str.lower().str.strip()
    merged = ranked2.merge(raw_keep, on="_key", how="inner", suffixes=("", "_raw"))
    if merged.empty:
        return None

    curr_keep = curr_ranked[["PLAYER_ID", "salary"]].rename(columns={"salary": "salary_curr"})
    full = merged.merge(curr_keep, on="PLAYER_ID", how="left")
    full["salary_prev"] = full["salary"]
    full = full[full["salary_prev"] > 0].copy()

    # Sorted descending salary array for this season — projection[rank] =
    # salaries_by_rank[rank - 1]. Needed for fast rank → $ lookup during
    # the inner loop.
    salaries_by_rank = np.sort(full["salary_prev"].values)[::-1]
    full.attrs["salaries_by_rank"] = salaries_by_rank
    full.attrs["cap_curr"] = SALARY_CAP_M[curr_season] * 1_000_000
    full.attrs["prev_season"] = prev_season
    full.attrs["curr_season"] = curr_season
    return full


def compute_loss(weights_vec: np.ndarray, pair_data: list[pd.DataFrame],
                 metric: str = "median_pct_cap") -> float:
    """Given a vector of weights (in WEIGHT_KEYS order), compute the
    aggregate prediction loss across all pair_data DataFrames.

    metric: 'median_pct_cap' or 'mean_abs_pct_cap' or 'within_5pct' (negated
    for minimization).
    """
    w = dict(zip(WEIGHT_KEYS, weights_vec))
    all_errors_pct = []

    for df in pair_data:
        # Recompute base_score with candidate weights.
        bs = (
            df["PTS"].values
            + w["AST"]     * df["AST"].values
            + w["OREB"]    * df["OREB"].values
            + w["DREB"]    * df["DREB"].values
            + w["BLK"]     * df["BLK"].values
            + w["STL"]     * df["STL"].values
            + w["TOV"]     * df["TOV"].values
            + w["PF"]      * df["PF"].values
            + w["DLEBRON"] * df["d_lebron"].values
            + w["EFFADJ"]  * df["efficiency_adj"].values
        )
        barrett = bs * df["avail_mult"].values
        # Rank by barrett (highest = 1). argsort gives ascending indices.
        order = np.argsort(-barrett)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))  # 0-indexed

        salaries_by_rank = df.attrs["salaries_by_rank"]
        n = len(salaries_by_rank)
        proj = salaries_by_rank[np.minimum(ranks, n - 1)]

        # Cap-scaled projection to current season dollars.
        cap_prev = SALARY_CAP_M[df.attrs["prev_season"]] * 1_000_000
        cap_curr = df.attrs["cap_curr"]
        proj_scaled = proj * (cap_curr / cap_prev)

        # Filter to new-contract pool only.
        salary_prev = df["salary_prev"].values
        salary_curr = df["salary_curr"].fillna(0).values
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_change = np.where(salary_prev > 0,
                                  (salary_curr - salary_prev) / salary_prev, 0.0)
        mask = (
            (proj > 0) & (salary_curr > 0)
            & (np.abs(pct_change) >= NEW_DEAL_PCT_THRESHOLD)
        )
        if not mask.any():
            continue

        abs_err   = np.abs(salary_curr[mask] - proj_scaled[mask])
        pct_err   = abs_err / cap_curr * 100
        all_errors_pct.append(pct_err)

    if not all_errors_pct:
        return 1e9
    pooled = np.concatenate(all_errors_pct)

    if metric == "median_pct_cap":
        return float(np.median(pooled))
    if metric == "mean_abs_pct_cap":
        return float(np.mean(pooled))
    if metric == "within_5pct":
        # Negated so minimizing loss = maximizing % within 5%.
        return -float((pooled <= 5.0).mean())
    return float(np.median(pooled))


def accuracy_report(weights_vec: np.ndarray, pair_data: list[pd.DataFrame],
                    label: str) -> dict:
    """Print and return the full accuracy report for a given weight vector."""
    w = dict(zip(WEIGHT_KEYS, weights_vec))
    all_errors_pct = []

    for df in pair_data:
        bs = (
            df["PTS"].values
            + w["AST"]     * df["AST"].values
            + w["OREB"]    * df["OREB"].values
            + w["DREB"]    * df["DREB"].values
            + w["BLK"]     * df["BLK"].values
            + w["STL"]     * df["STL"].values
            + w["TOV"]     * df["TOV"].values
            + w["PF"]      * df["PF"].values
            + w["DLEBRON"] * df["d_lebron"].values
            + w["EFFADJ"]  * df["efficiency_adj"].values
        )
        barrett = bs * df["avail_mult"].values
        order = np.argsort(-barrett)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))

        salaries_by_rank = df.attrs["salaries_by_rank"]
        n = len(salaries_by_rank)
        proj = salaries_by_rank[np.minimum(ranks, n - 1)]
        cap_prev = SALARY_CAP_M[df.attrs["prev_season"]] * 1_000_000
        cap_curr = df.attrs["cap_curr"]
        proj_scaled = proj * (cap_curr / cap_prev)

        salary_prev = df["salary_prev"].values
        salary_curr = df["salary_curr"].fillna(0).values
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_change = np.where(salary_prev > 0,
                                  (salary_curr - salary_prev) / salary_prev, 0.0)
        mask = (
            (proj > 0) & (salary_curr > 0)
            & (np.abs(pct_change) >= NEW_DEAL_PCT_THRESHOLD)
        )
        if not mask.any():
            continue
        abs_err = np.abs(salary_curr[mask] - proj_scaled[mask])
        pct_err = abs_err / cap_curr * 100
        all_errors_pct.append(pct_err)

    pooled = np.concatenate(all_errors_pct)
    med = float(np.median(pooled))
    mean = float(np.mean(pooled))
    w5 = float((pooled <= 5.0).mean() * 100)
    w10 = float((pooled <= 10.0).mean() * 100)

    print(f"\n[{label}]")
    print(f"  Sample size:    {len(pooled)}")
    print(f"  Median |error|: {med:.2f}% of cap")
    print(f"  Mean |error|:   {mean:.2f}% of cap")
    print(f"  Within 5%:      {w5:.1f}%")
    print(f"  Within 10%:     {w10:.1f}%")
    return {"median": med, "mean": mean, "within_5": w5, "within_10": w10}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Smaller optimization budget (faster, less accurate).")
    parser.add_argument("--train-only-modern", action="store_true",
                        help="Fit weights using only the last 10 season pairs.")
    parser.add_argument("--metric", default="within_5pct",
                        choices=["median_pct_cap", "mean_abs_pct_cap", "within_5pct"],
                        help="Optimization target. within_5pct is what users feel.")
    args = parser.parse_args()

    # Build pair list (newest → oldest).
    all_pairs = [(SEASONS[i], SEASONS[i - 1]) for i in range(1, len(SEASONS))]
    print(f"Loading data for {len(all_pairs)} season pairs...")
    t0 = time.time()
    pair_data = []
    for prev, curr in all_pairs:
        d = load_pair_data(prev, curr)
        if d is not None and len(d) > 0:
            pair_data.append(d)
    print(f"  Loaded {len(pair_data)} pairs in {time.time() - t0:.1f}s")
    if not pair_data:
        print("No usable data — has the cache been seeded?")
        return

    # Split into train and held-out test (modern era = last 10 pairs).
    # In --train-only-modern mode, fit on those 10 and skip the rest.
    modern_pairs = pair_data[:10]
    older_pairs  = pair_data[10:]
    if args.train_only_modern:
        train_data = modern_pairs
        test_data  = modern_pairs  # in-sample only
        print("Training on modern era (last 10 pairs) only.")
    else:
        train_data = pair_data
        test_data  = pair_data
        print(f"Training on all {len(pair_data)} pairs (in-sample report).")

    # ── Baseline (current weights) ────────────────────────────────────────────
    current_vec = np.array([CURRENT_WEIGHTS[k] for k in WEIGHT_KEYS])
    print("\n" + "=" * 76)
    print("BASELINE — current Barrett Score weights")
    print("=" * 76)
    for k, v in CURRENT_WEIGHTS.items():
        print(f"  {k:10s}: {v:+.3f}")
    baseline = accuracy_report(current_vec, test_data, "CURRENT WEIGHTS")
    if args.train_only_modern:
        accuracy_report(current_vec, modern_pairs, "CURRENT WEIGHTS — modern era")

    # ── Optimization ──────────────────────────────────────────────────────────
    bounds = [WEIGHT_BOUNDS[k] for k in WEIGHT_KEYS]
    obj = lambda w: compute_loss(w, train_data, metric=args.metric)

    print("\n" + "=" * 76)
    print(f"OPTIMIZING — metric: {args.metric}")
    print("=" * 76)
    print("Running random search + coordinate descent...")
    t0 = time.time()
    if args.quick:
        result = random_search_plus_coordinate_descent(
            obj, current_vec, bounds,
            n_restarts=6, n_random=80, cd_rounds=3, cd_steps=15,
            verbose=True,
        )
    else:
        result = random_search_plus_coordinate_descent(
            obj, current_vec, bounds,
            n_restarts=15, n_random=300, cd_rounds=5, cd_steps=25,
            verbose=True,
        )
    print(f"  Done in {time.time() - t0:.1f}s. Final loss: {result.fun:.4f}")

    optimal_vec = result.x

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("OPTIMAL WEIGHTS")
    print("=" * 76)
    print(f"  {'COMPONENT':10s} {'CURRENT':>10s} {'OPTIMAL':>10s} {'Δ':>10s}")
    for i, k in enumerate(WEIGHT_KEYS):
        cur, opt = CURRENT_WEIGHTS[k], optimal_vec[i]
        delta = opt - cur
        print(f"  {k:10s} {cur:+10.3f} {opt:+10.3f} {delta:+10.3f}")

    optimized = accuracy_report(optimal_vec, test_data, "OPTIMIZED WEIGHTS")
    if args.train_only_modern:
        accuracy_report(optimal_vec, modern_pairs, "OPTIMIZED WEIGHTS — modern era")
    elif older_pairs:
        accuracy_report(optimal_vec, modern_pairs, "OPTIMIZED — modern era (last 10)")

    print("\n" + "=" * 76)
    print("IMPROVEMENT")
    print("=" * 76)
    print(f"  Median |error|: {baseline['median']:.2f}% → {optimized['median']:.2f}%  "
          f"({optimized['median'] - baseline['median']:+.2f}pp)")
    print(f"  Within 5%:      {baseline['within_5']:.1f}% → {optimized['within_5']:.1f}%  "
          f"({optimized['within_5'] - baseline['within_5']:+.1f}pp)")
    print(f"  Within 10%:     {baseline['within_10']:.1f}% → {optimized['within_10']:.1f}%  "
          f"({optimized['within_10'] - baseline['within_10']:+.1f}pp)")

    # ── Drop-in code for utils.base_score ─────────────────────────────────────
    print("\nDROP-IN REPLACEMENT for utils.base_score():")
    print("-" * 76)
    print("def base_score(row) -> float:")
    print('    d_lebron = row["d_lebron"] if "d_lebron" in row.index else 0')
    print('    eff_adj  = row.get("efficiency_adj", 0) if hasattr(row, "get") else 0')
    print("    return (")
    print('        row["PTS"]')
    for k in WEIGHT_KEYS:
        v = optimal_vec[WEIGHT_KEYS.index(k)]
        if k in ("AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF"):
            print(f'        + ({v:+.4f}) * row["{k}"]')
        elif k == "DLEBRON":
            print(f'        + ({v:+.4f}) * d_lebron')
        elif k == "EFFADJ":
            print(f'        + ({v:+.4f}) * eff_adj')
    print("    )")
    print("-" * 76)


if __name__ == "__main__":
    main()
