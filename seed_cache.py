"""
Run once from Render Shell to pre-populate /data/cache with all season parquets.
    cd /app && python seed_cache.py

After this runs, every page load reads from disk instead of hitting the NBA API.
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

    print()

elapsed = time.time() - total_start
print(f"Done in {elapsed:.0f}s. Restart the Streamlit service and loads should be fast.\n")
