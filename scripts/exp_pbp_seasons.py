"""Cache pbpstats PLAYER season totals.

Written to fit D1 against D-LEBRON, but it is NOT an experiment any more: the
file it writes for the live season is where the defensive blend gets
"Charge Fouls Drawn", so this runs as step 3b of scripts/nightly.py. Keep that
in mind before changing what it writes.

A COMPLETE season is fetched once and never again, which is right: its totals
cannot change. The LIVE season is refetched every run, which is the whole
point. Those two rules used to be one rule, `if fp.exists(): continue`, correct
for the ten finished seasons and silently wrong for the one still being played:
player_2026-27.json was fetched on 2026-10-24, two games in, and skipped every
run after. The league's charge total sat at 23 for twelve days and the blend's
damping read that as "nobody draws charges" rather than "this file stopped
counting", so the channel Barrett asked for on 2026-11-02 never once ran on
real data.
"""
import json, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import season_is_live  # noqa: E402

OUT = ROOT / "cache" / "pbpstats"
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0"}


def get(u, tries=4):
    d = 4.0
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(
                urllib.request.Request(u, headers=UA), timeout=60).read())
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"     retry after {type(e).__name__} ({d:.0f}s)", flush=True)
            time.sleep(d)
            d *= 2


def main() -> int:
    live_ok = None
    for y in range(2016, 2028):
        season = f"{y-1}-{str(y)[2:]}"
        fp = OUT / f"player_{season}.json"
        live = season_is_live(season)
        if fp.exists() and not live:
            continue
        try:
            d = get(f"https://api.pbpstats.com/get-totals/nba?Season={season}"
                    f"&SeasonType=Regular%20Season&Type=Player")
            rows = d.get("multi_row_table_data") or []
        except Exception as e:
            print(f"  {season}: FAILED {type(e).__name__}", flush=True)
            if live:
                live_ok = False
            continue
        if live and not rows:
            # Stale beats empty, the same doctrine every other fetcher here
            # follows: an empty payload would zero every count in the blend.
            print(f"  {season}: LIVE and the feed returned 0 rows; keeping "
                  f"{'the existing file' if fp.exists() else 'nothing'}", flush=True)
            live_ok = False
            continue
        fp.write_text(json.dumps(rows))
        gp = max((int(r.get("GamesPlayed") or 0) for r in rows), default=0)
        print(f"  {season}: {len(rows)} players, {len(rows[0]) if rows else 0} fields"
              + (f", max GamesPlayed {gp}  (LIVE, refetched)" if live else ""), flush=True)
        if live:
            live_ok = True
        time.sleep(4)
    print(f"cached: {len(list(OUT.glob('player_*.json')))}")
    # Non-zero ONLY when the live season specifically could not be refreshed, so
    # the nightly can say so out loud without failing the night over it.
    return 0 if live_ok is not False else 1


if __name__ == "__main__":
    sys.exit(main())
