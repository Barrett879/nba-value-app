"""Hunt for a CORRECTABLE bias: decompose the shipped model's residuals
(actual - predicted, in pp of cap) by subgroup over the temporal CV. A subgroup
with a consistent NON-ZERO mean residual is a systematic bias we can correct
(like the max-tier floor fixed the star undershoot). A subgroup with mean ~0
but big spread is just variance — not correctable. Also reports per-year sign
of each subgroup's mean residual, so we only act on STABLE biases.

Loads the cached pool.

Usage:
    python -u scripts/diag_residuals.py
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

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def main():
    df = pd.read_pickle("/tmp/pool_df.pkl").reset_index(drop=True)
    X = np.load("/tmp/pool_X.npy")
    sy = df["start_year"].values
    grade = gradeable_mask(df).values

    recs = []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        m = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem].copy()
        pred = apply_cba_postprocess(m.predict(X[tem]), sub)
        actual = sub["salary_curr"].values / sub["cap_curr"].values
        sub["pred_pct"] = pred
        sub["actual_pct"] = actual
        sub["resid_pp"] = (actual - pred) * 100      # + = model UNDERSHOT
        sub["abs_pp"] = np.abs(actual - pred) * 100
        sub["ty"] = ty
        recs.append(sub)
    R = pd.concat(recs, ignore_index=True)
    print(f"n test predictions: {len(R)}   overall mean resid "
          f"{R['resid_pp'].mean():+.2f}pp   within-5% {np.mean(R['abs_pp']<=5)*100:.1f}%")

    def band(col, fn, labels):
        return R[col].map(fn) if callable(fn) else fn

    R["age_band"] = pd.cut(R["age"], [0, 23, 27, 31, 99],
                           labels=["≤23", "24-27", "28-31", "32+"])
    R["svc_band"] = pd.cut(R["years_in_league"], [-1, 4, 9, 99],
                           labels=["rookie-scale(≤4)", "mid(5-9)", "vet(10+)"])
    R["pred_tier"] = pd.cut(R["pred_pct"], [0, 0.05, 0.15, 0.25, 1.0],
                            labels=["<5%", "5-15%", "15-25%", "25%+"])
    R["allnba"] = np.where(R["all_nba_3yr"].fillna(0) >= 1, "All-NBA", "not")

    def show(col):
        print(f"\n  by {col}:")
        print(f"    {'group':<18}{'n':>5}{'mean resid':>12}{'within5%':>10}"
              f"{'per-year mean resid':>30}")
        for g, gdf in R.groupby(col, observed=True):
            if len(gdf) < 8:
                continue
            per = gdf.groupby("ty")["resid_pp"].mean()
            per_str = " ".join(f"{per.get(y, float('nan')):+.0f}" for y in TEST_YEARS)
            mark = "  <-- bias?" if abs(gdf["resid_pp"].mean()) >= 1.5 and len(gdf) >= 20 else ""
            print(f"    {str(g):<18}{len(gdf):>5}{gdf['resid_pp'].mean():>+11.2f}"
                  f"{np.mean(gdf['abs_pp']<=5)*100:>9.0f}%{per_str:>30}{mark}")

    for col in ["pos_bucket", "age_band", "svc_band", "pred_tier", "allnba"]:
        show(col)

    # cross-tab: position x pred_tier (where the center-discount might live)
    print("\n  position × pred_tier (mean resid pp, n):")
    for pos in ["Guard", "Forward", "Center"]:
        row = []
        for tier in ["<5%", "5-15%", "15-25%", "25%+"]:
            cell = R[(R["pos_bucket"] == pos) & (R["pred_tier"] == tier)]
            row.append(f"{tier}:{cell['resid_pp'].mean():+.1f}(n{len(cell)})" if len(cell) else f"{tier}:--")
        print(f"    {pos:<8} " + "  ".join(row))


if __name__ == "__main__":
    main()
