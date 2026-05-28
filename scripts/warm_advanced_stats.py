"""Warm the advanced-stats parquet cache for every season we need.

The raw-stats contract model uses trailing 3-yr advanced metrics (USG, PIE,
NET_RATING, TS), so we need advanced stats going back ~3 seasons before the
1999 training window — i.e. 1996-97 onward.

Run once; reruns hit the cache instantly.

Usage:
    python -u scripts/warm_advanced_stats.py
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from utils import SEASONS, SALARY_CAP_M, fetch_advanced_stats


def main() -> None:
    # All seasons from 1996-97 through the current one.
    wanted = [s for s in SEASONS if int(s.split("-")[0]) >= 1996]
    wanted = sorted(wanted, key=lambda s: int(s.split("-")[0]))
    print(f"Warming advanced stats for {len(wanted)} seasons "
          f"({wanted[0]} → {wanted[-1]})...", flush=True)

    ok, empty = 0, 0
    for s in wanted:
        t0 = time.time()
        df = fetch_advanced_stats(s, "Regular Season")
        if df.empty:
            empty += 1
            print(f"  {s}: EMPTY", flush=True)
        else:
            ok += 1
            has_pie = "PIE" in df.columns
            print(f"  {s}: {len(df):>3} players, PIE={'Y' if has_pie else 'N'} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    print(f"\nDone. {ok} cached, {empty} empty.", flush=True)


if __name__ == "__main__":
    main()
