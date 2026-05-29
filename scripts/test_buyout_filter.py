"""Should we exclude BUYOUT-RESIDUAL contracts (Ayton-type) the way we exclude
known-bad labels? Two things to answer honestly:

  1. SAFETY: what exactly does the candidate rule catch? List every row so we
     can verify it's only buyout residuals, not legit cheap/paycut deals.
     Candidate: prev >= 20% of cap  AND  curr <= 8% of cap  AND  age <= 28.
     (A prime-age near-max player dropping to ~min only happens via buyout —
      the old team pays the rest. Age gate spares legit vet paycuts: Harden 32,
      Lopez 35/37.)

  2. FORWARD VALUE: does removing them from TRAINING help predict the OTHER
     (legit) contracts? Expanding-window temporal CV, train baseline vs
     train-minus-residuals, evaluate BOTH on the same legit gradeable set
     (residuals removed from eval so we don't just game the denominator).

Usage:
    python -u scripts/test_buyout_filter.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
from build_production_histgbm import (
    make_X_augmented, HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask, apply_cba_postprocess,
)

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def buyout_residual_mask(df):
    # use the pct-of-cap columns build_rows already computed
    prev_pct = df["salary_prev_pct"].values
    curr_pct = df["salary_curr_pct"].values
    age = df["age"].values
    return (prev_pct >= 0.20) & (curr_pct <= 0.08) & (age <= 28)


def main():
    print("Building pool...", flush=True)
    careers = build_career_indexes(playoffs=False)
    df = build_rows(PAIRS, careers, {}, fetch_all_nba_selections()).reset_index(drop=True)
    X = make_X_augmented(df)
    sy = df["start_year"].values
    grade_ok = gradeable_mask(df).values
    resid = buyout_residual_mask(df)

    # ---- 1. SAFETY: list everything the rule catches ----
    print("\n" + "=" * 90)
    print(f"CANDIDATE RULE CATCHES {int(resid.sum())} ROWS  (prev>=20% & curr<=8% & age<=28)")
    print("=" * 90)
    print(f"  {'Player':<24}{'Season':<9}{'prev%':>7}{'curr%':>7}{'age':>5}{'prev$':>8}{'curr$':>8}")
    sub = df[resid].sort_values("start_year")
    for _, r in sub.iterrows():
        pp = r["salary_prev_pct"] * 100
        cp = r["salary_curr_pct"] * 100
        print(f"  {str(r['player'])[:23]:<24}{r['curr']:<9}{pp:>6.1f}%{cp:>6.1f}%"
              f"{r['age']:>5.0f}{r['salary_prev']/1e6:>7.1f}M{r['salary_curr']/1e6:>7.1f}M")

    # ---- 2. FORWARD VALUE: train baseline vs train-minus-residuals ----
    print("\n" + "=" * 90)
    print("FORWARD CV — does removing residuals from TRAINING help predict LEGIT contracts?")
    print("(eval set = gradeable & NOT residual, identical for both models)")
    print("=" * 90)
    print(f"  {'year':<6}{'n_eval':>7}{'base w5':>9}{'filt w5':>9}{'Δw5':>7}"
          f"{'base w10':>10}{'filt w10':>10}{'Δw10':>7}")
    rows = []
    for ty in TEST_YEARS:
        # baseline training = current shipped: all 2012+ prior seasons
        tr_base = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tr_filt = tr_base & ~resid
        evm = (sy == ty) & grade_ok & ~resid
        if tr_base.sum() < 150 or evm.sum() < 10:
            continue
        sub = df[evm]
        a = sub["salary_curr"].values / sub["cap_curr"].values
        m_base = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[tr_base], df.loc[tr_base, "salary_curr_pct"].values)
        m_filt = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[tr_filt], df.loc[tr_filt, "salary_curr_pct"].values)
        p_base = apply_cba_postprocess(m_base.predict(X[evm]), sub)
        p_filt = apply_cba_postprocess(m_filt.predict(X[evm]), sub)
        w5_b = np.mean(np.abs(p_base - a) * 100 <= 5) * 100
        w5_f = np.mean(np.abs(p_filt - a) * 100 <= 5) * 100
        w10_b = np.mean(np.abs(p_base - a) * 100 <= 10) * 100
        w10_f = np.mean(np.abs(p_filt - a) * 100 <= 10) * 100
        rows.append((ty, len(sub), w5_b, w5_f, w10_b, w10_f))
        print(f"  {ty:<6}{len(sub):>7}{w5_b:>8.1f}%{w5_f:>8.1f}%{w5_f-w5_b:>+6.1f}"
              f"{w10_b:>9.1f}%{w10_f:>9.1f}%{w10_f-w10_b:>+6.1f}")

    ns = np.array([r[1] for r in rows])
    d5 = np.array([r[3] - r[2] for r in rows])
    d10 = np.array([r[5] - r[4] for r in rows])
    print("  " + "-" * 60)
    print(f"  weighted Δ within-5%: {np.average(d5, weights=ns):+.2f}pp   "
          f"weighted Δ within-10%: {np.average(d10, weights=ns):+.2f}pp")
    wins5 = int((d5 > 0.01).sum()); loss5 = int((d5 < -0.01).sum())
    print(f"  within-5%: {wins5} years up, {loss5} down, {len(rows)-wins5-loss5} flat")


if __name__ == "__main__":
    main()
