#!/usr/bin/env python3
"""Refresh cache/contract_end_years_v2.pkl from Basketball Reference.

Why this exists as a script rather than a one-liner: the contract cache is
committed and deployed, so refreshing it is a deliberate act with a verification
step, not a side effect of running the app.

Two things it handles that a bare call to fetch_contract_end_years() does not:

  * BBRef rate limiting. Their limit is roughly 20 requests a minute and a
    violation blocks the IP for up to an hour -- long enough that a scrape
    started into a block just burns time in backoff. This waits for a probe to
    come back 200 before it starts, then paces at 5s a team (12/min).

  * The freshness gate. fetch_contract_end_years short-circuits on a cache
    younger than a day, so a plain call after a recent run is a no-op. This
    bypasses the gate WITHOUT moving the file: the fetcher needs the old cache
    in place to carry rows forward for any team it cannot read. block=True
    forces the scrape inline -- the app path serves stale and refreshes in a
    background thread, which would let this script exit before the scrape ran.

Prints a coverage report against data/master_roster.csv at the end. The cache is
only rewritten if the scrape is at least as complete as what is already there --
that guard lives in fetch_contract_end_years itself.

Usage:  python3 scripts/refresh_contract_years.py [--wait-minutes N]
"""
import argparse
import collections
import csv
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests                                             # noqa: E402
import utils                                                # noqa: E402
from utils import NameIndex, _BREF_UA                       # noqa: E402

PROBE = "https://www.basketball-reference.com/contracts/WAS.html"
BOOKS_START = 2026        # the season these pages are about: 2026-27 is year one
MIN_PACE = 5.0            # seconds between team pages, ~12 requests/minute


def wait_for_bbref(max_minutes: int) -> bool:
    """Poll until BBRef answers 200, or give up. Returns True if it is up."""
    deadline = time.time() + max_minutes * 60
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = requests.get(PROBE, headers={"User-Agent": _BREF_UA}, timeout=20)
            if r.status_code == 200:
                print(f"  BBRef reachable after {attempt} probe(s)", flush=True)
                return True
            print(f"  probe {attempt}: http {r.status_code}, waiting 60s", flush=True)
        except Exception as e:
            print(f"  probe {attempt} failed: {e}", flush=True)
        time.sleep(60)
    return False


def coverage_report(cey: dict) -> None:
    """How much of the 2026-27 books this cache can actually speak to."""
    idx = NameIndex(cey)
    path = ROOT / "data" / "master_roster.csv"
    rows = [r for r in csv.DictReader(
        ln for ln in open(path) if not ln.lstrip().startswith("#")) if r.get("player")]
    under = [r for r in rows if r.get("kind") in ("standard", "two_way")]
    usable = stale = absent = 0
    stale_names = []
    for r in under:
        ci = idx.get(r["player"])
        if ci is None:
            absent += 1
            continue
        try:
            end = int(str(ci.get("end_season", "0-0")).split("-")[0])
        except ValueError:
            absent += 1
            continue
        if end < BOOKS_START:
            stale += 1
            stale_names.append(f"{r['team']} {r['player']} (ends {ci['end_season']})")
        else:
            usable += 1
    n = len(under) or 1
    print(f"\nCOVERAGE against the {BOOKS_START}-{str(BOOKS_START+1)[-2:]} books "
          f"({len(under)} players under contract)")
    print(f"  usable : {usable:4d}  ({100*usable/n:.0f}%)")
    print(f"  stale  : {stale:4d}  (cache holds a deal that ended before the season)")
    print(f"  absent : {absent:4d}")
    for s in stale_names[:10]:
        print(f"      stale e.g. {s}")

    teams = collections.Counter(v.get("current_team") for v in cey.values())
    missing = sorted(set(utils._BBREF_TEAMS) - set(teams))
    print(f"\n  cache: {len(cey)} rows across {len(teams)} teams"
          f"{'  MISSING: ' + ','.join(missing) if missing else ''}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-minutes", type=int, default=75,
                    help="how long to wait for BBRef's rate limit to clear")
    args = ap.parse_args()

    path = utils._dc_path("contract_end_years_v2.pkl")
    before = {}
    if path.exists():
        with open(path, "rb") as fh:
            before = pickle.load(fh)
    print(f"cache before: {len(before)} rows, "
          f"{len({v.get('current_team') for v in before.values()})} teams")

    print(f"waiting for BBRef (up to {args.wait_minutes}m)...", flush=True)
    if not wait_for_bbref(args.wait_minutes):
        print("BBRef still rate limiting; cache left untouched.")
        coverage_report(before)
        return 1

    # Bypass the day-old freshness gate, but leave the file where it is so the
    # fetcher can fall back on it per team.
    utils._dc_fresh = lambda *a, **k: False
    real_sleep = time.sleep
    utils.time.sleep = lambda s: real_sleep(max(s, MIN_PACE))

    print(f"scraping 30 team pages at {MIN_PACE}s apiece...", flush=True)
    t0 = time.time()
    cey = utils.fetch_contract_end_years(block=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)

    after = {}
    if path.exists():
        with open(path, "rb") as fh:
            after = pickle.load(fh)
    print(f"cache after: {len(after)} rows, "
          f"{len({v.get('current_team') for v in after.values()})} teams "
          f"({'REWRITTEN' if after != before else 'unchanged'})")
    coverage_report(cey)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
