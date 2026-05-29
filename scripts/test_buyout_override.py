"""Forward-validate the buyout-minimum override. Loads the cached pool
(/tmp/pool_df.pkl) so it's instant. Expanding-window temporal CV on 2021-2025:
train on prior seasons, predict each year, score within-5%/10% of cap WITH vs
WITHOUT the override (apply_buyout toggled). Since the override only touches
flagged buyouts and snaps them to the minimum (validated within-5% on 105/105
big->small cases), it can only help or be neutral — this measures by how much
and confirms it never hurts a year.

Usage:
    python -u scripts/test_buyout_override.py
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
    HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask, apply_cba_postprocess,
)
from utils import normalize as _norm, is_known_buyout

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def main():
    df = pd.read_pickle("/tmp/pool_df.pkl").reset_index(drop=True)
    X = np.load("/tmp/pool_X.npy")
    sy = df["start_year"].values
    grade_ok = gradeable_mask(df).values

    print("=" * 80)
    print("FORWARD CV — buyout override OFF vs ON  (within-5% / within-10% of cap)")
    print("=" * 80)
    print(f"  {'year':<6}{'n':>5}{'off w5':>9}{'on w5':>8}{'Δ5':>6}"
          f"{'off w10':>10}{'on w10':>8}{'Δ10':>6}   buyouts graded")
    rows = []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade_ok
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem]
        a = sub["salary_curr"].values / sub["cap_curr"].values
        raw = reg.predict(X[tem])
        p_off = apply_cba_postprocess(raw, sub, apply_buyout=False)
        p_on = apply_cba_postprocess(raw, sub, apply_buyout=True)
        w5o = np.mean(np.abs(p_off - a) * 100 <= 5) * 100
        w5n = np.mean(np.abs(p_on - a) * 100 <= 5) * 100
        w10o = np.mean(np.abs(p_off - a) * 100 <= 10) * 100
        w10n = np.mean(np.abs(p_on - a) * 100 <= 10) * 100
        # which buyouts are in this graded year
        bo = [str(p) for p, s in zip(sub["player"], sub["curr"])
              if is_known_buyout(p, s)]
        rows.append((ty, len(sub), w5o, w5n, w10o, w10n))
        print(f"  {ty:<6}{len(sub):>5}{w5o:>8.1f}%{w5n:>7.1f}%{w5n-w5o:>+6.1f}"
              f"{w10o:>9.1f}%{w10n:>7.1f}%{w10n-w10o:>+6.1f}   {', '.join(bo) or '—'}")

    ns = np.array([r[1] for r in rows])
    d5 = np.array([r[3] - r[2] for r in rows])
    d10 = np.array([r[5] - r[4] for r in rows])
    print("  " + "-" * 70)
    print(f"  weighted Δ within-5%: {np.average(d5, weights=ns):+.2f}pp     "
          f"weighted Δ within-10%: {np.average(d10, weights=ns):+.2f}pp")
    w5 = int((d5 > 0.01).sum()); l5 = int((d5 < -0.01).sum())
    print(f"  within-5%: {w5} years up, {l5} down, {len(rows)-w5-l5} flat   "
          f"(override can't hurt non-buyouts, so 0 down expected)")

    # also: aggregate across all graded test rows (the headline-style number)
    allp_off, allp_on, alla = [], [], []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade_ok
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem]
        raw = reg.predict(X[tem])
        allp_off.append(apply_cba_postprocess(raw, sub, apply_buyout=False))
        allp_on.append(apply_cba_postprocess(raw, sub, apply_buyout=True))
        alla.append(sub["salary_curr"].values / sub["cap_curr"].values)
    off = np.concatenate(allp_off); on = np.concatenate(allp_on); act = np.concatenate(alla)
    print("\n  POOLED across all graded test rows (n={}):".format(len(act)))
    print(f"    within-5%:  {np.mean(np.abs(off-act)*100<=5)*100:.1f}%  ->  "
          f"{np.mean(np.abs(on-act)*100<=5)*100:.1f}%")
    print(f"    within-10%: {np.mean(np.abs(off-act)*100<=10)*100:.1f}%  ->  "
          f"{np.mean(np.abs(on-act)*100<=10)*100:.1f}%")


if __name__ == "__main__":
    main()
