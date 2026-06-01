"""R&D: honest prediction intervals via quantile-loss GBM.

The point estimate is at its ceiling, so make the BAND honest instead. Train
HistGBM with quantile loss at the 10th/50th/90th percentiles of salary % of cap.
Validate EMPIRICAL COVERAGE on the 2021-25 walk-forward OOS:
  - does ~80% of actual salaries fall inside [q10, q90]?  (that's the promise)
  - is the band sensibly tier-dependent (tight for role players, wide for stars)?
If coverage is honest, this replaces the heuristic ±X% band on the page.

Usage:  python -u scripts/experiment_quantile.py
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
    make_X_augmented, gradeable_mask, HISTGBM_PARAMS, TRAINING_START_YEAR,
    PRED_FLOOR_PCT, PRED_CEIL_PCT,
)
CURRENT_CAP_M = 165.0
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def qparams(q):
    p = dict(HISTGBM_PARAMS)
    p["loss"] = "quantile"; p["quantile"] = q
    return p


def collect(df, X):
    sy = df["start_year"].values; grade_ok = gradeable_mask(df).values
    a, lo, mid, hi, cap = [], [], [], [], []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty); tem = (sy == ty) & grade_ok
        if trm.sum() < 100 or tem.sum() < 5: continue
        ytr = df.loc[trm, "salary_curr_pct"].values
        m10 = HistGradientBoostingRegressor(**qparams(0.10)).fit(X[trm], ytr)
        m50 = HistGradientBoostingRegressor(**qparams(0.50)).fit(X[trm], ytr)
        m90 = HistGradientBoostingRegressor(**qparams(0.90)).fit(X[trm], ytr)
        sub = df[tem]; c = sub["cap_curr"].values
        clip = lambda v: np.clip(v, PRED_FLOOR_PCT, PRED_CEIL_PCT)
        p10 = clip(m10.predict(X[tem])); p50 = clip(m50.predict(X[tem])); p90 = clip(m90.predict(X[tem]))
        # enforce monotone q10<=q50<=q90 per row
        p10, p90 = np.minimum(p10, p90), np.maximum(p10, p90)
        p50 = np.clip(p50, p10, p90)
        a.append(sub["salary_curr"].values / c)   # actual as %-of-cap
        lo.append(p10); mid.append(p50); hi.append(p90); cap.append(c)
    return (np.concatenate(a), np.concatenate(lo), np.concatenate(mid),
            np.concatenate(hi), np.concatenate(cap))


def main():
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    X = make_X_augmented(df)
    print(f"{len(df)} contracts\n", flush=True)
    a, lo, mid, hi, cap = collect(df, X)

    inside = (a >= lo) & (a <= hi)
    print("=" * 70, flush=True)
    print("QUANTILE INTERVAL COVERAGE  (target: ~80% inside [q10,q90])", flush=True)
    print("=" * 70, flush=True)
    print(f"  Overall coverage : {np.mean(inside)*100:.1f}%   (n={len(a)})", flush=True)
    width_M = (hi - lo) * CURRENT_CAP_M
    a_M = a * CURRENT_CAP_M
    print(f"  Median band width: ${np.median(width_M):.1f}M", flush=True)
    print("\nBy tier (actual salary, today's $):", flush=True)
    print(f"  {'tier':<11} {'n':>4} {'coverage':>9} {'med width':>10}", flush=True)
    tiers = [("Max",a_M>=40),("BigStar",(a_M>=25)&(a_M<40)),("Mid",(a_M>=15)&(a_M<25)),
             ("Rot",(a_M>=7)&(a_M<15)),("Min",a_M<7)]
    for name, m in tiers:
        k = int(m.sum())
        if k == 0: continue
        print(f"  {name:<11} {k:>4} {np.mean(inside[m])*100:>8.0f}% ${np.median(width_M[m]):>8.1f}M", flush=True)


if __name__ == "__main__":
    main()
