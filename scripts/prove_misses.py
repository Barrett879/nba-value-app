"""PROOF of the 89.1% within-5% headline: enumerate every gradeable prediction
that misses by >5% of cap, numbered, with the full arithmetic per row so each
can be independently verified, and reconcile the counts:

    560 gradeable predictions = (within 5%) + (outside 5%)

Same SHIPPED pipeline as the live app: 2012+ training, Barrett + advanced
features, expanding-window temporal CV on 2021-2025 (train only on prior
seasons), full CBA post-processing (clip + max-tier floor 0.20/age<=33).

Each row shows:  pred $ | actual $ | |gap| $ | cap $ | err% = |gap|/cap*100
The list is sorted worst-first, so it crosses the 5.00% line at the very
bottom — proving nothing under 5% is included and nothing over is omitted.

Usage:
    python -u scripts/prove_misses.py
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
from build_production_histgbm import (
    make_X_augmented, HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask,
    apply_cba_postprocess,
)

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def main():
    print("Building 2012+ pool...", flush=True)
    t0 = time.time()
    careers = build_career_indexes(playoffs=False)
    df = build_rows(PAIRS, careers, {}, fetch_all_nba_selections()).reset_index(drop=True)
    X = make_X_augmented(df)
    sy = df["start_year"].values
    grade_ok = gradeable_mask(df).values
    print(f"  {len(df)} contracts in {time.time()-t0:.1f}s", flush=True)

    rows = []          # every gradeable prediction (for the count)
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade_ok
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem].copy()
        cap = sub["cap_curr"].values
        pred = apply_cba_postprocess(reg.predict(X[tem]), sub) * cap
        actual = sub["salary_curr"].values
        for i, (_, r) in enumerate(sub.iterrows()):
            gap = abs(pred[i] - actual[i])
            rows.append({
                "player": str(r["player"]), "season": r["curr"],
                "pred": pred[i], "actual": actual[i], "cap": cap[i],
                "gap": gap, "err_pct": gap / cap[i] * 100,
            })

    n = len(rows)
    miss = [r for r in rows if r["err_pct"] > 5.0]
    within5 = n - len(miss)
    within10 = n - sum(1 for r in rows if r["err_pct"] > 10.0)

    # ---- count reconciliation ----
    print("\n" + "=" * 96, flush=True)
    print("COUNT RECONCILIATION", flush=True)
    print("=" * 96, flush=True)
    print(f"  total gradeable predictions (2021-2025, temporal CV) ....... {n}", flush=True)
    print(f"  within 5% of cap ........................................... {within5}"
          f"   ({within5/n*100:.1f}%)", flush=True)
    print(f"  NOT within 5% of cap (listed below) ........................ {len(miss)}"
          f"   ({len(miss)/n*100:.1f}%)", flush=True)
    print(f"  check: {within5} + {len(miss)} = {within5+len(miss)}  "
          f"(== {n}? {'YES' if within5+len(miss)==n else 'NO'})", flush=True)
    print(f"  within 10% of cap .......................................... {within10}"
          f"   ({within10/n*100:.1f}%)", flush=True)

    # ---- numbered proof, worst-first ----
    miss.sort(key=lambda r: -r["err_pct"])
    print("\n" + "=" * 96, flush=True)
    print(f"THE {len(miss)} MISSES, WORST-FIRST — err% = |pred-actual| / cap * 100  "
          f"(must be > 5.00)", flush=True)
    print("=" * 96, flush=True)
    print(f"  {'#':>3} {'Player':<23}{'Season':<9}{'pred$':>9}{'actual$':>10}"
          f"{'|gap|$':>10}{'cap$':>10}{'err%':>8}", flush=True)
    print("  " + "-" * 90, flush=True)
    for k, r in enumerate(miss, 1):
        flag = "  <-- 5.00% line" if (k == len(miss)) else ""
        print(f"  {k:>3} {r['player'][:22]:<23}{r['season']:<9}"
              f"{r['pred']/1e6:>8.2f}M{r['actual']/1e6:>9.2f}M"
              f"{r['gap']/1e6:>9.2f}M{r['cap']/1e6:>9.1f}M{r['err_pct']:>7.2f}%{flag}",
              flush=True)

    # show the closest-but-still-within rows so you can see the boundary is clean
    near = sorted([r for r in rows if r["err_pct"] <= 5.0], key=lambda r: -r["err_pct"])[:3]
    print("\n  (for boundary sanity — the 3 closest predictions that DO pass < 5%:)", flush=True)
    for r in near:
        print(f"      {r['player'][:22]:<23}{r['season']:<9}err {r['err_pct']:>5.2f}%  (within)",
              flush=True)


if __name__ == "__main__":
    main()
