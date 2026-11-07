#!/usr/bin/env python3
"""Build the per-stint splits artifact for the scored season.

WHY THIS EXISTS. pages/Rankings.py's Splits View calls utils.build_splits_data,
which reads a committed pickle and otherwise falls through to
build_splits_data_live -- thirty live calls to the NBA stats API, one per team.
That fallback works on a laptop and fails on Render every single time, because
the stats API is not reachable from there. Every team fetch returned None, the
live builder handed back a bare DataFrame with no columns, and the page raised
KeyError: 'total_min', which production rendered as "this app has encountered
an error".

Reader and writer had also drifted apart: load_splits_from_disk looks for
splits_<season>_<FORMULA_VERSION>.pkl while every committed file is named
splits_<season>.pkl, so the disk branch had been dead for every season.

So: build the artifact here, commit it, let the nightly keep it fresh. Same
shape as the existing pickles ({"data": DataFrame, "fetched_at": ...}) and the
same revisioned filename rule as every other cache file that contains current
-season numbers, because Render's /data disk seeds a file only when its NAME is
missing.

Usage:  python3 scripts/build_splits.py [SEASON]
"""
from __future__ import annotations

import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ATTEMPTS = 10       # whole-build retries before giving up
PAUSE_S = 60        # between them, to let a rate limit clear

from nba_api.stats.static import teams as nba_teams_static  # noqa: E402

from utils import (  # noqa: E402
    CACHE_DIR, DATA_REV, FORMULA_VERSION, SEASONS, build_ranked_projected,
    build_splits_data_live, normalize, season_is_live,
)

# .__wrapped__ because build_splits_data_live is @st.cache_data(ttl=3600).
# Retrying the CACHED function is not retrying anything: the first six-attempt
# run here printed "21 of 30 teams" six times with identical stint counts,
# which is not a flaky API, it is one answer returned six times. The repo
# already reaches past the decorator this way for the same reason
# (fetch_bref_positions.__wrapped__).
_build_live = getattr(build_splits_data_live, "__wrapped__", build_splits_data_live)


ALL_TEAMS = {t["abbreviation"] for t in nba_teams_static.get_teams()}


def main() -> int:
    season = sys.argv[1] if len(sys.argv) > 1 else SEASONS[0]

    # The FULL pool, not a threshold-filtered slice. The page applies its own
    # minutes threshold to the splits frame afterwards, so building from a
    # filtered lookup would bake one slider position into the artifact.
    board = build_ranked_projected(season)
    if board is None or board.empty:
        print(f"FAIL  no board for {season}, nothing to build splits from")
        return 1
    salary_lookup = tuple(
        (normalize(row["Player"]), row["salary"]) for _, row in board.iterrows()
    )
    print(f"season {season} - salary lookup carries {len(salary_lookup)} players")

    # Patience, because the stats API rate-limits a caller who asks for thirty
    # team-seasons in a row. build_splits_data_live is bounded on purpose (a
    # dead API must never hang a page request), so the retrying belongs here in
    # the batch job rather than in the function the page calls. A partial
    # answer is never merged across attempts: ranks are computed over the whole
    # pool, so stitching two runs together would produce ranks that are right
    # for neither.
    df = None
    for attempt in range(1, ATTEMPTS + 1):
        got = _build_live(season, salary_lookup)
        if got is None or got.empty or "total_min" not in got.columns:
            print(f"attempt {attempt}/{ATTEMPTS}: the API returned nothing at all")
        else:
            seen = set(got.loc[got["Team"] != "TOT", "Team"])
            missing = sorted(ALL_TEAMS - seen)
            print(f"attempt {attempt}/{ATTEMPTS}: {len(got)} stints, "
                  f"{got['Player'].nunique()} players, {len(seen)} of 30 teams"
                  + (f", missing {' '.join(missing)}" if missing else ""))
            if not missing:
                df = got
                break
        if attempt < ATTEMPTS:
            time.sleep(PAUSE_S)

    if df is None:
        # Refuse to write. A file with no columns, or one missing a whole
        # team, is worse than no file: the reader would find it and serve it,
        # and a team's players would simply not be in Splits View.
        print(f"FAIL  no complete 30-team answer in {ATTEMPTS} attempts; "
              f"NOT writing a partial artifact")
        return 1

    stints = len(df)
    players = df["Player"].nunique()
    tots = int((df["Team"] == "TOT").sum())
    teams = df.loc[df["Team"] != "TOT", "Team"].nunique()
    print(f"built {stints} stints, {players} players, {teams} teams, {tots} TOT rows")

    key = season.replace("-", "_")
    stamp = {"data": df, "fetched_at": datetime.now(timezone.utc).isoformat()}
    base = CACHE_DIR / f"splits_{key}_{FORMULA_VERSION}.pkl"

    # A FINISHED season's splits never change again, so it gets ONE stable
    # filename and no DATA_REV. That matters on Render, where the /data disk
    # seeds a cache file only when its NAME is missing: a name that never moves
    # is seeded once and is then correct forever, while a revisioned name for
    # a static season would ship a new copy every night for no reason.
    if not season_is_live(season):
        with open(base, "wb") as f:
            pickle.dump(stamp, f)
        print(f"WROTE {base.name}  (finished season, one stable name)")
        return 0

    # The LIVE season moves every night, so it needs the revisioned name, which
    # is the only way a new copy reaches that same /data disk.
    revved = CACHE_DIR / f"splits_{key}_{FORMULA_VERSION}_{DATA_REV}.pkl"
    with open(revved, "wb") as f:
        pickle.dump(stamp, f)
    print(f"WROTE {revved.name}")

    # Plus a one-time floor under the un-revisioned name, which the reader
    # falls back to, so a night where the stats API is down degrades to
    # stale-but-present rather than to no Splits View at all. Written only when
    # absent, for the seeding reason above.
    if not base.exists():
        with open(base, "wb") as f:
            pickle.dump(stamp, f)
        print(f"WROTE {base.name}  (fallback floor, written once)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
