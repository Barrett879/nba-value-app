"""Which buyout-prediction value is MOST accurate against actual buyout
salaries? Test candidate formulas on (a) the 11 verified KNOWN_BUYOUTS and
(b) the broader 105 buyout-shaped cases, scoring mean |error| (pp of cap) and
within-5%. Pick the lowest-error option. Loads the cached pool.

Usage:
    python -u scripts/test_buyout_tiers.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from utils import cba_min_pct, is_known_buyout


def main():
    df = pd.read_pickle("/tmp/pool_df.pkl").reset_index(drop=True)
    svc = df["years_in_league"].fillna(0).values
    actual = df["salary_curr_pct"].values

    known = np.array([is_known_buyout(p, s) for p, s in zip(df["player"], df["curr"])])
    shaped = (df["salary_prev_pct"].values >= 0.12) & (actual <= 0.06) \
        & (df["salary_curr"].values < 0.5 * df["salary_prev"].values)

    # candidate predictions (as % of cap)
    svc_min = np.array([cba_min_pct(s) for s in svc])
    candidates = {
        "veteran min (svc-scaled)": svc_min,
        "flat 2.0%":  np.full(len(df), 0.020),
        "flat 2.5%":  np.full(len(df), 0.025),
        "flat 3.0%":  np.full(len(df), 0.030),
        "flat 3.5% (TMLE)": np.full(len(df), 0.035),
        "flat 4.0%":  np.full(len(df), 0.040),
        "max(min, 3.0%)": np.maximum(svc_min, 0.030),
    }

    for label, mask in [("11 verified KNOWN_BUYOUTS", known),
                        ("105 buyout-shaped cases", shaped)]:
        a = actual[mask]
        print("\n" + "=" * 70)
        print(f"{label}  (n={mask.sum()})   mean actual = {a.mean()*100:.1f}% of cap")
        print("=" * 70)
        print(f"  {'candidate':<26}{'mean|err|':>10}{'median':>9}{'within5%':>10}{'max|err|':>10}")
        best = None
        for cname, pred in candidates.items():
            e = np.abs(pred[mask] - a) * 100
            row = (e.mean(), np.median(e), np.mean(e <= 5) * 100, e.max())
            if best is None or row[0] < best[1][0]:
                best = (cname, row)
            print(f"  {cname:<26}{row[0]:>9.2f}pp{row[1]:>8.2f}{row[2]:>9.0f}%{row[3]:>9.2f}pp")
        print(f"  → lowest mean|err|: {best[0]}  ({best[1][0]:.2f}pp)")

    # per-case for the 11, with the two leading candidates
    print("\n" + "=" * 70)
    print("PER-CASE (11 verified) — veteran-min vs flat-3.0%")
    print("=" * 70)
    print(f"  {'Player':<20}{'Season':<9}{'svc':>4}{'actual%':>9}{'min%':>7}{'3.0%err':>9}{'minErr':>8}")
    for i in np.where(known)[0]:
        am = abs(svc_min[i] - actual[i]) * 100
        a3 = abs(0.030 - actual[i]) * 100
        print(f"  {str(df.iloc[i]['player'])[:19]:<20}{df.iloc[i]['curr']:<9}{svc[i]:>4.0f}"
              f"{actual[i]*100:>8.1f}%{svc_min[i]*100:>6.1f}%{a3:>8.1f}{am:>8.1f}")


if __name__ == "__main__":
    main()
