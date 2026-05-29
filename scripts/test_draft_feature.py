"""Does adding DRAFT PEDIGREE (overall pick) as a model feature improve
contract prediction? Hypothesis: the biggest miss pattern is young players
paid on upside (lottery picks landing extensions off thin production). The
model has age/service/Barrett but no draft position — a #1 pick gets paid for
potential the box score doesn't yet show. Trees can learn the interaction
(high pick AND young -> higher).

Honest test — expanding-window temporal CV on 2021-2025: append draft pick to
the 28-feature matrix, train fresh models per fold, compare within-5%/10% on
the gradeable set. Ship only if it helps across years.

Loads the cached pool (/tmp/pool_df.pkl).

Usage:
    python -u scripts/test_draft_feature.py
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
from utils import normalize, build_draft_tier_lookup

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
UNDRAFTED_PICK = 61   # sentinel beyond the 2nd round (no pedigree signal)


def main():
    df = pd.read_pickle("/tmp/pool_df.pkl").reset_index(drop=True)
    X = np.load("/tmp/pool_X.npy")
    sy = df["start_year"].values
    grade = gradeable_mask(df).values

    lookup = build_draft_tier_lookup()
    picks = df["player"].map(
        lambda n: (lookup.get(normalize(str(n)), {}) or {}).get("draft_pick") or UNDRAFTED_PICK
    ).astype(float).values
    matched = np.mean([normalize(str(n)) in lookup for n in df["player"]]) * 100
    print(f"draft lookup matched {matched:.0f}% of pool rows "
          f"(unmatched -> pick {UNDRAFTED_PICK})")
    print(f"pick distribution: lottery(1-14) {np.mean(picks<=14)*100:.0f}%  "
          f"mid {np.mean((picks>14)&(picks<=30))*100:.0f}%  "
          f"2nd/undrafted {np.mean(picks>30)*100:.0f}%")

    X_aug = np.hstack([X, picks.reshape(-1, 1)])

    print("\n" + "=" * 74)
    print("FORWARD CV — add draft pick as a feature?  (within-5% / within-10%)")
    print("=" * 74)
    print(f"  {'year':<6}{'n':>5}{'base w5':>9}{'+draft w5':>11}{'Δ5':>6}"
          f"{'base w10':>10}{'+draft w10':>12}{'Δ10':>6}")
    rows = []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        sub = df[tem]
        a = sub["salary_curr"].values / sub["cap_curr"].values
        y = df.loc[trm, "salary_curr_pct"].values
        m_b = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X[trm], y)
        m_d = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X_aug[trm], y)
        p_b = apply_cba_postprocess(m_b.predict(X[tem]), sub)
        p_d = apply_cba_postprocess(m_d.predict(X_aug[tem]), sub)
        w5b = np.mean(np.abs(p_b - a) * 100 <= 5) * 100
        w5d = np.mean(np.abs(p_d - a) * 100 <= 5) * 100
        w10b = np.mean(np.abs(p_b - a) * 100 <= 10) * 100
        w10d = np.mean(np.abs(p_d - a) * 100 <= 10) * 100
        rows.append((ty, len(sub), w5b, w5d, w10b, w10d))
        print(f"  {ty:<6}{len(sub):>5}{w5b:>8.1f}%{w5d:>10.1f}%{w5d-w5b:>+6.1f}"
              f"{w10b:>9.1f}%{w10d:>11.1f}%{w10d-w10b:>+6.1f}")

    ns = np.array([r[1] for r in rows])
    d5 = np.array([r[3] - r[2] for r in rows])
    d10 = np.array([r[5] - r[4] for r in rows])
    print("  " + "-" * 64)
    print(f"  weighted Δ within-5%: {np.average(d5, weights=ns):+.2f}pp     "
          f"weighted Δ within-10%: {np.average(d10, weights=ns):+.2f}pp")
    w = int((d5 > 0.01).sum()); l = int((d5 < -0.01).sum())
    print(f"  within-5%: {w} years up, {l} down, {len(rows)-w-l} flat")
    wm = np.average(d5, weights=ns)
    print(f"\n  VERDICT: {'HELPS — integrate it.' if wm > 0.1 and w >= l else 'WASH/HURT — drop it.' if wm <= 0.1 else 'MIXED.'}")


if __name__ == "__main__":
    main()
