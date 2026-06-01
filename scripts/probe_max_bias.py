"""Probe: is the stable -$5M Max-tier bias the 8%-raise/escalation gap?

If max-tier actual salaries are escalated years of multi-year max deals while
the model predicts a year-1 max %, the residual should correlate with how far
ABOVE the contract-year max the actual salary sits. Print the worst Max-tier
under-projections with their actual %-of-cap vs the CBA max % tiers (25/30/35).
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
    make_X_augmented, apply_cba_postprocess, gradeable_mask, HISTGBM_PARAMS, TRAINING_START_YEAR,
)
CURRENT_CAP_M = 165.0
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]

def main():
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    X = make_X_augmented(df)
    sy = df["start_year"].values; grade_ok = gradeable_mask(df).values
    recs = []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty); tem = (sy == ty) & grade_ok
        if trm.sum() < 100 or tem.sum() < 5: continue
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X[trm], df.loc[trm,"salary_curr_pct"].values)
        sub = df[tem]; cap = sub["cap_curr"].values
        pred = apply_cba_postprocess(reg.predict(X[tem]), sub) * cap
        actual = sub["salary_curr"].values
        for j, idx in enumerate(sub.index):
            a_pct = actual[j]/cap[j]; p_pct = pred[j]/cap[j]
            recs.append((df.at[idx,"player"], df.at[idx,"curr"], a_pct, p_pct,
                         (actual[j]-pred[j])/cap[j]*CURRENT_CAP_M))
    # max-tier actual >= 40M today
    maxes = [r for r in recs if r[2]*CURRENT_CAP_M >= 40]
    print(f"\nMax-tier contracts (actual >= $40M today): n={len(maxes)}", flush=True)
    print(f"  actual %-of-cap: min {min(r[2] for r in maxes)*100:.1f}  "
          f"median {np.median([r[2] for r in maxes])*100:.1f}  "
          f"max {max(r[2] for r in maxes)*100:.1f}", flush=True)
    above35 = [r for r in maxes if r[2] > 0.355]
    print(f"  actual ABOVE the 35% supermax line: {len(above35)}/{len(maxes)} "
          f"({len(above35)/len(maxes)*100:.0f}%)  ← escalated/raised max years", flush=True)
    print("\nWorst 12 max-tier under-projections (model below actual):", flush=True)
    print(f"  {'player':<22} {'season':<8} {'act%':>6} {'pred%':>6} {'miss$M':>7}", flush=True)
    for p,c,a,pr,miss in sorted(maxes, key=lambda r:r[4])[:12]:
        print(f"  {p:<22} {c:<8} {a*100:>5.1f}% {pr*100:>5.1f}% {miss:>+7.1f}", flush=True)

if __name__ == "__main__":
    main()
