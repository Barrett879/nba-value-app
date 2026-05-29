"""Does removing the VERIFIED buyout contracts from TRAINING make the model
better at predicting real MARKET-rate contracts?

This is the training-data-hygiene question (the actual point of the buyout
work): a buyout deal (a productive player labeled at a buyout minimum) is a
mislabeled example for a model learning production -> market value. Currently
the offseason buyouts (curr >= 2% of cap) survive the artifact filter and sit
in training with their depressed labels.

Honest test — expanding-window temporal CV on 2021-2025:
  baseline  = production training (2012+ minus bad_data)            [incl. buyouts]
  candidate = baseline minus the verified KNOWN_BUYOUTS             [clean labels]
Evaluate BOTH on the same eval set = gradeable AND NOT a buyout (market deals
only — buyouts aren't market value, so grading market prediction against them
is unfair). Ship the exclusion only if it helps across years.

Loads the cached pool (/tmp/pool_df.pkl).

Usage:
    python -u scripts/test_buyout_training.py
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
    _is_bad_data,
)
from utils import is_known_buyout

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def main():
    df = pd.read_pickle("/tmp/pool_df.pkl").reset_index(drop=True)
    X = np.load("/tmp/pool_X.npy")
    sy = df["start_year"].values
    bad = _is_bad_data(df).values
    grade = gradeable_mask(df).values
    buyout = np.array([is_known_buyout(p, s) for p, s in zip(df["player"], df["curr"])])
    print(f"verified buyouts in pool: {int(buyout.sum())}  "
          f"(of which currently in training, i.e. ~bad: {int((buyout & ~bad).sum())})")

    print("\n" + "=" * 78)
    print("FORWARD CV — exclude verified buyouts from TRAINING?  (eval = market deals only)")
    print("=" * 78)
    print(f"  {'year':<6}{'n_eval':>7}{'base w5':>9}{'clean w5':>10}{'Δ5':>6}"
          f"{'base w10':>10}{'clean w10':>11}{'Δ10':>6}{'rm':>4}")
    rows = []
    for ty in TEST_YEARS:
        tr_base = (sy >= TRAINING_START_YEAR) & (sy < ty) & ~bad
        tr_clean = tr_base & ~buyout
        evm = (sy == ty) & grade & ~buyout
        if tr_base.sum() < 100 or evm.sum() < 5:
            continue
        sub = df[evm]
        a = sub["salary_curr"].values / sub["cap_curr"].values
        m_base = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[tr_base], df.loc[tr_base, "salary_curr_pct"].values)
        m_clean = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[tr_clean], df.loc[tr_clean, "salary_curr_pct"].values)
        p_base = apply_cba_postprocess(m_base.predict(X[evm]), sub)
        p_clean = apply_cba_postprocess(m_clean.predict(X[evm]), sub)
        w5b = np.mean(np.abs(p_base - a) * 100 <= 5) * 100
        w5c = np.mean(np.abs(p_clean - a) * 100 <= 5) * 100
        w10b = np.mean(np.abs(p_base - a) * 100 <= 10) * 100
        w10c = np.mean(np.abs(p_clean - a) * 100 <= 10) * 100
        removed = int(((sy >= TRAINING_START_YEAR) & (sy < ty) & ~bad & buyout).sum())
        rows.append((ty, len(sub), w5b, w5c, w10b, w10c))
        print(f"  {ty:<6}{len(sub):>7}{w5b:>8.1f}%{w5c:>9.1f}%{w5c-w5b:>+6.1f}"
              f"{w10b:>9.1f}%{w10c:>10.1f}%{w10c-w10b:>+6.1f}{removed:>4}")

    ns = np.array([r[1] for r in rows])
    d5 = np.array([r[3] - r[2] for r in rows])
    d10 = np.array([r[5] - r[4] for r in rows])
    print("  " + "-" * 68)
    print(f"  weighted Δ within-5%: {np.average(d5, weights=ns):+.2f}pp     "
          f"weighted Δ within-10%: {np.average(d10, weights=ns):+.2f}pp")
    w = int((d5 > 0.01).sum()); l = int((d5 < -0.01).sum())
    print(f"  within-5%: {w} years up, {l} down, {len(rows)-w-l} flat")
    verdict = ("HELPS — exclude buyouts from training." if np.average(d5, weights=ns) > 0.1 and w >= l
               else "WASH/HURT — leave training as-is." if np.average(d5, weights=ns) <= 0.1
               else "MIXED.")
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
