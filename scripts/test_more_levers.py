"""Two untested model-level levers, forward CV 2021-2025 (eval gradeable,
full CBA post-processing). Ship only what beats baseline across years.

A. Recency-weighted training: sample_weight = decay^(ty-1 - signing_year).
   We predict next season; the market evolves, so recent contracts may deserve
   more weight than the equal weighting we use now.
B. GBM + kNN ensemble: blend the gradient-boosted prediction with a k-nearest-
   neighbor (local comps) prediction on standardized features. Helps only if
   the two methods' errors are uncorrelated.

Loads the cached pool.

Usage:
    python -u scripts/test_more_levers.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from build_production_histgbm import (
    HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask, apply_cba_postprocess,
)

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


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
        sub = df[tem]
        a = sub["salary_curr"].values / sub["cap_curr"].values
        y = df.loc[trm, "salary_curr_pct"].values
        folds.append((ty, trm, tem, sub, a, y))

    def w5(deltas_pred):  # helper unused placeholder
        pass

    def score(pred_fn):
        per, ns = [], []
        for ty, trm, tem, sub, a, y in folds:
            p = apply_cba_postprocess(pred_fn(ty, trm, tem, y), sub)
            per.append(np.mean(np.abs(p - a) * 100 <= 5) * 100); ns.append(len(sub))
        return np.array(per), np.array(ns)

    # baseline
    def base_fn(ty, trm, tem, y):
        return HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X[trm], y).predict(X[tem])
    base, ns = score(base_fn)
    base_wt = np.average(base, weights=ns)
    print(f"baseline weighted within-5%: {base_wt:.2f}%   per-year {' '.join(f'{v:.0f}' for v in base)}")

    print("\n=== A. recency-weighted training ===")
    print(f"  {'decay':<8}{'wt Δ5':>8}{'up/dn':>8}{'per-year Δ':>22}")
    for decay in [0.95, 0.90, 0.85, 0.80]:
        def fn(ty, trm, tem, y, decay=decay):
            wgt = decay ** (ty - 1 - sy[trm])
            return HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
                X[trm], y, sample_weight=wgt).predict(X[tem])
        per, _ = score(fn)
        d = per - base; wm = np.average(d, weights=ns)
        up = int((d > 0.01).sum()); dn = int((d < -0.01).sum())
        print(f"  {decay:<8}{wm:>+7.2f}{f'{up}/{dn}':>8}{' '.join(f'{v:+.0f}' for v in d):>22}")

    print("\n=== B. GBM + kNN ensemble (blend weight = GBM share) ===")
    print(f"  {'k / w':<12}{'wt Δ5':>8}{'up/dn':>8}{'per-year Δ':>22}")
    for k in [10, 20]:
        for w in [0.8, 0.7, 0.6]:
            def fn(ty, trm, tem, y, k=k, w=w):
                gbm = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X[trm], y).predict(X[tem])
                sc = StandardScaler().fit(X[trm])
                knn = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(
                    sc.transform(X[trm]), y).predict(sc.transform(X[tem]))
                return w * gbm + (1 - w) * knn
            per, _ = score(fn)
            d = per - base; wm = np.average(d, weights=ns)
            up = int((d > 0.01).sum()); dn = int((d < -0.01).sum())
            print(f"  k={k} w={w:<6}{wm:>+7.2f}{f'{up}/{dn}':>8}{' '.join(f'{v:+.0f}' for v in d):>22}")


if __name__ == "__main__":
    main()
