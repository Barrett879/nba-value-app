"""Build the contract pool (df + X) ONCE, cache to /tmp, and list buyout
candidates. Subsequent buyout scripts load the cache instantly instead of
rebuilding (the build hits the network on a cold cache and is slow).

Progress is printed step-by-step so a hang is visible.

Usage:
    python -u scripts/build_pool_cache.py
"""
import sys, time, warnings, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np

DF_PATH = "/tmp/pool_df.pkl"
X_PATH = "/tmp/pool_X.npy"


def cba_min_pct(service):
    table = {0: 0.010, 1: 0.016, 2: 0.018, 3: 0.019, 4: 0.019,
             5: 0.020, 6: 0.021, 7: 0.022, 8: 0.022, 9: 0.023}
    return table.get(int(service or 0), 0.026)


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:5.1f}s] importing...", flush=True)
    from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
    from build_production_histgbm import make_X_augmented

    print(f"[{time.time()-t0:5.1f}s] build_career_indexes...", flush=True)
    careers = build_career_indexes(playoffs=False)
    print(f"[{time.time()-t0:5.1f}s] fetch_all_nba_selections...", flush=True)
    all_nba = fetch_all_nba_selections()
    print(f"[{time.time()-t0:5.1f}s] build_rows ({len(PAIRS)} pairs)...", flush=True)
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    print(f"[{time.time()-t0:5.1f}s] make_X_augmented...", flush=True)
    X = make_X_augmented(df)
    print(f"[{time.time()-t0:5.1f}s] caching {len(df)} rows...", flush=True)
    df.to_pickle(DF_PATH)
    np.save(X_PATH, X)
    print(f"[{time.time()-t0:5.1f}s] cached -> {DF_PATH}, {X_PATH}", flush=True)

    # ---- candidate listing ----
    prev_pct = df["salary_prev_pct"].values
    curr_pct = df["salary_curr_pct"].values
    cand = (prev_pct >= 0.12) & (curr_pct <= 0.06) & (df["salary_curr"].values < 0.5 * df["salary_prev"].values)
    sub = df[cand].copy().sort_values(["start_year", "salary_prev_pct"], ascending=[True, False])

    print("\n" + "=" * 100, flush=True)
    print(f"BUYOUT-SHAPED TRANSITIONS — {int(cand.sum())} candidates "
          f"(prev>=12% & curr<=6% & curr<0.5*prev)", flush=True)
    print("=" * 100, flush=True)
    print(f"  {'Player':<24}{'Season':<9}{'age':>4}{'svc':>4}{'prev%':>7}{'curr%':>7}"
          f"{'curr$':>8}{'minErr':>8}  w5?", flush=True)
    n_hit = 0
    for _, r in sub.iterrows():
        svc = r.get("years_in_league", 0)
        err = abs(cba_min_pct(svc) - r["salary_curr_pct"]) * 100
        hit = err <= 5.0; n_hit += hit
        print(f"  {str(r['player'])[:23]:<24}{r['curr']:<9}{r['age']:>4.0f}{svc:>4.0f}"
              f"{r['salary_prev_pct']*100:>6.1f}%{r['salary_curr_pct']*100:>6.1f}%"
              f"{r['salary_curr']/1e6:>7.1f}M{err:>7.1f}%  {'Y' if hit else 'n'}", flush=True)
    print("  " + "-" * 90, flush=True)
    print(f"  predict-minimum within 5% of cap on {n_hit}/{int(cand.sum())} candidates "
          f"·  done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
