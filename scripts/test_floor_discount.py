"""The residual hunt found the max-tier floor OVERSHOOTS the top tier / All-NBA
group by ~2pp every year — it snaps eligible stars to the THEORETICAL CBA max,
but the EMPIRICAL eligible-star salary is a bit lower (some take discounts).

Test snapping the floor to (max_tier - delta) instead of the full max, for
several deltas. Forward CV 2021-2025; report overall within-5% and the All-NBA
subgroup within-5%. Ship the delta that helps overall without wrecking a year.

Loads the cached pool.

Usage:
    python -u scripts/test_floor_discount.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from build_production_histgbm import (
    HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask, cba_max_pct,
    PRED_FLOOR_PCT, PRED_CEIL_PCT, MAX_FLOOR_TRIGGER, MAX_FLOOR_AGE_CAP,
)

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
DELTAS = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.12]   # pp shaved off the snap target


def postprocess(pred_pct, sub, delta):
    out = np.clip(pred_pct, PRED_FLOOR_PCT, PRED_CEIL_PCT).astype(float)
    svc = sub["years_in_league"].values
    ann = sub["all_nba_3yr"].values
    age = sub["age"].values
    for i in range(len(out)):
        a = age[i]
        age_ok = (a is None) or (isinstance(a, float) and np.isnan(a)) or (a <= MAX_FLOOR_AGE_CAP)
        if out[i] >= MAX_FLOOR_TRIGGER and (ann[i] or 0) >= 1 and age_ok:
            target = max(cba_max_pct(svc[i], ann[i]) - delta, MAX_FLOOR_TRIGGER)
            out[i] = max(out[i], target)
    return out


def main():
    df = pd.read_pickle("/tmp/pool_df.pkl").reset_index(drop=True)
    X = np.load("/tmp/pool_X.npy")
    sy = df["start_year"].values
    grade = gradeable_mask(df).values

    folds = []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        m = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem].copy()
        raw = m.predict(X[tem])
        a = sub["salary_curr"].values / sub["cap_curr"].values
        annflag = (sub["all_nba_3yr"].fillna(0) >= 1).values
        folds.append((ty, sub, raw, a, annflag, len(sub)))

    print("=" * 74)
    print("FORWARD CV — shave the max-floor snap target by δ  (within-5%)")
    print("=" * 74)
    print(f"  {'δ (pp)':<8}{'overall wt w5':>15}{'per-year':>26}{'All-NBA w5':>13}")
    for delta in DELTAS:
        per, ns, an_hit, an_tot = [], [], 0, 0
        for ty, sub, raw, a, annflag, n in folds:
            p = postprocess(raw, sub, delta)
            hit = np.abs(p - a) * 100 <= 5
            per.append(np.mean(hit) * 100); ns.append(n)
            an_hit += int(hit[annflag].sum()); an_tot += int(annflag.sum())
        wt = np.average(per, weights=ns)
        per_str = " ".join(f"{v:.0f}" for v in per)
        an = an_hit / an_tot * 100 if an_tot else float("nan")
        mark = "  <-- current" if delta == 0 else ""
        print(f"  {delta*100:<8.0f}{wt:>14.2f}%{per_str:>26}{an:>12.0f}%{mark}")

    print("\n  (All-NBA n across all folds: "
          f"{sum(int((f[4]).sum()) for f in folds)})")


if __name__ == "__main__":
    main()
