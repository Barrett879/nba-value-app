"""Biggest miss pattern = young players paid above their production as
restricted free agents / rookie-extension candidates (offer-sheet + matching
market). The model has only a NARROW flag (is_likely_max_ext: top-20 rank, age
22-25). Test a BROAD RFA signal — any player finishing a cheap rookie-scale
deal — as a feature. This is a contract-market mechanism, distinct from the
production stats. Several encodings, forward CV 2021-2025, eval gradeable.

Loads the cached pool.

Usage:
    python -u scripts/test_rfa_feature.py
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
    svc = df["years_in_league"].fillna(0).values
    prev_pct = df["salary_prev_pct"].values
    age = df["age"].fillna(27).values

    # Candidate RFA / rookie-extension signals (all derivable at predict time).
    encodings = {
        "rfa: svc 2-5 & cheap prior": (
            (svc >= 2) & (svc <= 5) & (prev_pct < 0.10)).astype(float),
        "rfa: young & cheap prior":   (
            (age <= 25) & (prev_pct < 0.08)).astype(float),
        "rfa: svc<=4 & prior<8%":     (
            (svc <= 4) & (prev_pct < 0.08)).astype(float),
    }

    folds = []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        sub = df[tem]
        folds.append((ty, trm, tem, sub,
                      sub["salary_curr"].values / sub["cap_curr"].values,
                      df.loc[trm, "salary_curr_pct"].values))

    base = {}
    for ty, trm, tem, sub, a, y in folds:
        m = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X[trm], y)
        p = apply_cba_postprocess(m.predict(X[tem]), sub)
        base[ty] = np.mean(np.abs(p - a) * 100 <= 5) * 100

    print("=" * 72)
    print("FORWARD CV — broad RFA / rookie-extension feature (within-5%)")
    print("=" * 72)
    print(f"  baseline weighted w5: "
          f"{np.average([base[f[0]] for f in folds], weights=[len(f[3]) for f in folds]):.2f}%")
    print(f"  {'encoding':<30}{'wt Δ5':>8}{'up/down':>9}{'per-year Δ':>22}")
    for name, feat in encodings.items():
        Xa = np.hstack([X, feat.reshape(-1, 1)])
        deltas, ns = [], []
        for ty, trm, tem, sub, a, y in folds:
            m = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(Xa[trm], y)
            p = apply_cba_postprocess(m.predict(Xa[tem]), sub)
            deltas.append(np.mean(np.abs(p - a) * 100 <= 5) * 100 - base[ty])
            ns.append(len(sub))
        deltas = np.array(deltas); ns = np.array(ns)
        wm = np.average(deltas, weights=ns)
        up = int((deltas > 0.01).sum()); dn = int((deltas < -0.01).sum())
        per = " ".join(f"{d:+.1f}" for d in deltas)
        flag = "  <-- HELPS" if (wm > 0.1 and up >= dn) else ""
        print(f"  {name:<30}{wm:>+7.2f}{f'{up}/{dn}':>9}{per:>22}{flag}")
    print(f"\n  share of pool flagged (svc2-5 & cheap): {encodings['rfa: svc 2-5 & cheap prior'].mean()*100:.0f}%")


if __name__ == "__main__":
    main()
