"""List every BUYOUT-SHAPED salary transition in the dataset so each can be
verified against public buyout records. A buyout signature is a big prior
salary collapsing to a small new one. We cast a WIDE net here (any age) and
verify membership by hand — the goal is a precise KNOWN_BUYOUTS list, not a
heuristic.

Net: prev >= 12% of cap  AND  curr <= 6% of cap  AND  curr < 0.5 * prev.

For each we print prev/curr (% of cap and $), age, service, and what the
veteran-minimum prediction WOULD score (|min - actual| in % of cap) — so we
can immediately see whether "predict the minimum" lands within 5%.

Usage:
    python -u scripts/list_buyout_candidates.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS


# NBA veteran minimum as a fraction of cap, by years of service (approx, modern
# era). Min salary scales with service but flattens; as a share of a ~$140M cap
# it runs ~0.8% (rookie) to ~2.6% (10+ yr vet).
def cba_min_pct(service):
    s = int(service or 0)
    table = {0: 0.010, 1: 0.016, 2: 0.018, 3: 0.019, 4: 0.019,
             5: 0.020, 6: 0.021, 7: 0.022, 8: 0.022, 9: 0.023}
    return table.get(s, 0.026)  # 10+ years ≈ 2.6% of cap


def main():
    careers = build_career_indexes(playoffs=False)
    df = build_rows(PAIRS, careers, {}, fetch_all_nba_selections()).reset_index(drop=True)

    prev_pct = df["salary_prev_pct"].values
    curr_pct = df["salary_curr_pct"].values
    cand = (prev_pct >= 0.12) & (curr_pct <= 0.06) & (df["salary_curr"].values < 0.5 * df["salary_prev"].values)

    sub = df[cand].copy().sort_values(["start_year", "salary_prev_pct"], ascending=[True, False])
    print("=" * 104)
    print(f"BUYOUT-SHAPED TRANSITIONS — {int(cand.sum())} candidates  "
          f"(prev>=12% & curr<=6% & curr<0.5*prev)")
    print("=" * 104)
    print(f"  {'Player':<24}{'Season':<9}{'age':>4}{'svc':>4}"
          f"{'prev%':>7}{'curr%':>7}{'curr$':>8}{'minPred%':>9}{'min err':>9}  within5?")
    n_hit = 0
    for _, r in sub.iterrows():
        svc = r.get("years_in_league", 0)
        mp = cba_min_pct(svc)
        err = abs(mp - r["salary_curr_pct"]) * 100
        hit = err <= 5.0
        n_hit += hit
        print(f"  {str(r['player'])[:23]:<24}{r['curr']:<9}{r['age']:>4.0f}{svc:>4.0f}"
              f"{r['salary_prev_pct']*100:>6.1f}%{r['salary_curr_pct']*100:>6.1f}%"
              f"{r['salary_curr']/1e6:>7.1f}M{mp*100:>8.1f}%{err:>8.1f}%  {'YES' if hit else 'no'}")
    print("  " + "-" * 96)
    print(f"  predicting the veteran minimum would land within 5% of cap on "
          f"{n_hit}/{int(cand.sum())} of these")


if __name__ == "__main__":
    main()
