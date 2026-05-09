"""
Run once from Render Shell to pre-populate /data/cache with all season parquets.
    cd /app && python seed_cache.py

After this runs, every page load reads from disk instead of hitting the NBA API.

If the script gets rate-limited and bails partway through, just re-run it —
disk caches make seasons that already finished a no-op, so it picks up where
it left off. Pre-1996 seasons hit BBRef hardest (per-game stats + 30 team
salary pages + playoff per-game), so expect 60-90 sec each on those years.
"""
import sys
import time
from pathlib import Path

# Patch Streamlit cache decorators before importing utils so they become
# plain pass-throughs — the disk-writing logic inside each function still runs.
import streamlit as st

def _passthrough(*dargs, **dkwargs):
    def decorator(func):
        return func
    if len(dargs) == 1 and callable(dargs[0]):
        return dargs[0]
    return decorator

st.cache_data     = _passthrough
st.cache_resource = _passthrough

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    SEASONS, CACHE_DIR, season_to_espn_year,
    build_raw, apply_rankings, apply_projections,
    fetch_bref_positions, fetch_next_year_contracts, fetch_rookie_scale_players,
    fetch_playoff_rounds,
)

print(f"\nCache directory: {CACHE_DIR}")
print(f"Seeding {len(SEASONS)} seasons...\n")

total_start = time.time()

for i, season in enumerate(SEASONS, 1):
    raw_path = CACHE_DIR / f"raw_{season.replace('-', '_')}.parquet"
    age_str  = ""
    if raw_path.exists():
        age_h = (time.time() - raw_path.stat().st_mtime) / 3600
        if age_h < 24 * 30:  # 30 days
            age_str = f"  (raw cached, {age_h:.0f}h old)"

    print(f"[{i:2}/{len(SEASONS)}] {season}{age_str}")

    # ── Raw player stats + salary data ────────────────────────────────────────
    t0 = time.time()
    try:
        raw = build_raw(season)
        print(f"         raw       {len(raw):3d} players  {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"         raw       ERROR: {e}")
        raw = None

    # ── ESPN position map ──────────────────────────────────────────────────────
    t0 = time.time()
    try:
        fetch_bref_positions(season_to_espn_year(season), cache_v=3)
        print(f"         positions  ok  {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"         positions  ERROR: {e}")

    # ── Next-year contracts ────────────────────────────────────────────────────
    t0 = time.time()
    try:
        fetch_next_year_contracts(season_to_espn_year(season), cache_v=7)
        print(f"         contracts  ok  {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"         contracts  ERROR: {e}")

    # ── Rookie scale ───────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        fetch_rookie_scale_players(season)
        print(f"         rookies    ok  {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"         rookies    ERROR: {e}")

    # ── Playoff rounds (BBRef scrape, 1 page per season) ──────────────────────
    t0 = time.time()
    try:
        rounds = fetch_playoff_rounds(season)
        if rounds:
            print(f"         pl_rounds  {len(rounds):2d} teams   {time.time()-t0:.1f}s")
        else:
            print(f"         pl_rounds  (none — pre-bracket era?)  {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"         pl_rounds  ERROR: {e}")

    # ── Playoff raw stats ─────────────────────────────────────────────────────
    # Skip the current season if no playoffs have happened yet — empty
    # response is normal and shouldn't pollute the log.
    t0 = time.time()
    try:
        playoff_raw = build_raw(season, playoffs=True)
        n = len(playoff_raw)
        if n:
            print(f"         playoff   {n:3d} players  {time.time()-t0:.1f}s")
        else:
            print(f"         playoff    (no data — season in progress?)  {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"         playoff    ERROR: {e}")

    # Be polite to BBRef between seasons — they rate-limit hard. Brief sleep
    # for modern seasons (NBA Stats API only), longer for pre-1996 / pre-2000
    # which hit BBRef heavily for stats + salaries + playoff data.
    is_bref_heavy = int(season.split("-")[0]) < 2000
    inter_sleep = 4.0 if is_bref_heavy else 1.0
    time.sleep(inter_sleep)

    print()

elapsed = time.time() - total_start
print(f"Done in {elapsed:.0f}s. Restart the Streamlit service and loads should be fast.\n")
