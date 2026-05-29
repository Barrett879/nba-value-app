"""What happens to accuracy if the max-tier stars are predicted at their
actual max? Three scorings on the same temporal-CV predictions:

  A. RAW            — current validation (clip only). Stars undershot.
  B. ELIGIBILITY    — realistic rule: snap recent-All-NBA players the model
                      already rates highly (>=22% of cap) up to their CBA max
                      tier (25/30/35 by service, +tier for All-NBA). This is
                      what the live app's floor does — no peeking at actuals.
  C. ORACLE         — cheat: snap everyone whose ACTUAL was a max (>=28% of
                      cap) to their tier. The theoretical ceiling of fixing
                      the star undershoot.

Reports within-5%/10% for each, plus how many maxes B fixes vs discount deals
it breaks (the Brunson cost).

Usage:
    python -u scripts/test_max_floor_impact.py
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


def cba_max_pct(service, all_nba):
    elite = (all_nba or 0) >= 1
    s = service or 0
    if s <= 6:  return 0.30 if elite else 0.25
    if s <= 9:  return 0.35 if elite else 0.30
    return 0.35


def main():
    print("Building pool...", flush=True)
    careers = build_career_indexes(playoffs=False)
    df = build_rows(PAIRS, careers, {}, fetch_all_nba_selections()).reset_index(drop=True)
    X = make_X_augmented(df)
    sy = df["start_year"].values
    grade_ok = gradeable_mask(df).values

    raw, elig, oracle, actual, capv = [], [], [], [], []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade_ok
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem]
        cap = sub["cap_curr"].values
        praw_pct = apply_cba_postprocess(reg.predict(X[tem]), sub)  # clip only
        a_pct = sub["salary_curr"].values / cap
        svc = sub["years_in_league"].values
        ann = sub["all_nba_3yr"].values
        for i in range(len(sub)):
            mx = cba_max_pct(svc[i], ann[i])
            r = praw_pct[i]
            # B: eligibility floor (no peeking)
            e = max(r, mx) if (r >= 0.22 and (ann[i] or 0) >= 1) else r
            # C: oracle (peeks at actual)
            o = mx if a_pct[i] >= 0.28 else r
            raw.append(r); elig.append(e); oracle.append(o)
            actual.append(a_pct[i]); capv.append(cap[i])

    raw, elig, oracle = map(np.array, (raw, elig, oracle))
    actual, capv = np.array(actual), np.array(capv)
    n = len(raw)

    def score(pred_pct):
        e = np.abs(pred_pct - actual) * 100  # pp of cap == |err| since both are %cap
        return float(np.mean(e <= 5) * 100), float(np.mean(e <= 10) * 100)

    print("\n" + "=" * 66, flush=True)
    print(f"IMPACT OF PREDICTING STARS AT THEIR MAX  (n={n})", flush=True)
    print("=" * 66, flush=True)
    for label, p in [("A. RAW model (current)", raw),
                     ("B. realistic max-floor (live-app style)", elig),
                     ("C. ORACLE (perfect max knowledge)", oracle)]:
        w5, w10 = score(p)
        print(f"  {label:<42} within-5% {w5:5.1f}%   within-10% {w10:5.1f}%", flush=True)

    # B's tradeoff: maxes fixed vs discounts broken.
    raw_hit = np.abs(raw - actual) * 100 <= 5
    elig_hit = np.abs(elig - actual) * 100 <= 5
    fixed = int((~raw_hit & elig_hit).sum())
    broke = int((raw_hit & ~elig_hit).sum())
    print(f"\n  Realistic floor (B): fixed {fixed} misses, broke {broke} hits "
          f"(net {fixed-broke:+d})", flush=True)
    print(f"  → the 'broke' ones are stars who took a DISCOUNT (Brunson-type).", flush=True)


if __name__ == "__main__":
    main()
