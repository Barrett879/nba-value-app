"""Scrape Spotrac's per-year NBA free-agent tables into one contract-type ledger.

Each yearly page (https://www.spotrac.com/nba/free-agents/_/year/YYYY) lists every
free-agent signing for that offseason with the team the player came *from* and the
team he signed *with*. That From/To pair is the piece the value model has never had:

    from_team == to_team  ->  the incumbent re-signed him (Bird rights, cap hold)
    from_team != to_team  ->  he changed teams on the open market

Those two populations price very differently, and until now every signing in the
training set was treated as one undifferentiated "contract".

Output: data/contract_types_2012_2026.csv

Note on what this is and is not. Presence on the page means "signed as a free agent
that offseason". Rookie-scale deals and veteran extensions signed under contract do
not appear here at all, so this separates *re-sign vs move*, not *extension vs
open-market signing*. The absence of a player from a year is itself a weak signal
that any deal he signed that year was an extension, but it is not stated by Spotrac
and should not be treated as one.

Usage:
    python scripts/scrape_contract_types.py            # 2012-2026, cached pages reused
    python scripts/scrape_contract_types.py --refresh  # re-download every page
    python scripts/scrape_contract_types.py --years 2025 2026
"""

from __future__ import annotations

import argparse
import csv
import html as _html
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "contract_types_2012_2026.csv"
CACHE = ROOT / "cache" / "spotrac_free_agents"

# Spotrac 403s a bare urllib/requests call. A normal browser User-Agent gets a 200.
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
URL = "https://www.spotrac.com/nba/free-agents/_/year/{year}"

FIRST_YEAR, LAST_YEAR = 2012, 2026
PACE_SECONDS = 3.0  # Spotrac throttles bursts; 3s between fetches has run clean

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r"<td\b[^>]*>(.*?)</td>", re.S)
_TEAM = re.compile(r'<span class="d-none">([A-Z]{2,3})</span>')
_PLAYER = re.compile(r'/nba/player/_/id/(\d+)/[^"]*"[^>]*>(.*?)</a>', re.S)


def _text(fragment: str) -> str:
    return _html.unescape(re.sub(r"<[^>]+>", " ", fragment)).strip()


def _money(fragment: str) -> str:
    digits = re.sub(r"[^0-9]", "", _text(fragment))
    return digits or ""


def fetch(year: int, refresh: bool = False) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{year}.html"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8", errors="replace")
    out = subprocess.run(
        ["curl", "-sS", "--fail", "-H", f"User-Agent: {UA}",
         "-H", "Accept: text/html,application/xhtml+xml",
         "-H", "Accept-Language: en-US,en;q=0.9", URL.format(year=year)],
        capture_output=True, text=True, timeout=90,
    )
    if out.returncode != 0:
        raise RuntimeError(f"{year}: curl failed - {out.stderr.strip()[:200]}")
    path.write_text(out.stdout, encoding="utf-8")
    return out.stdout


def parse(year: int, page: str) -> list[dict]:
    body = page[page.find("<tbody"):page.find("</tbody>")]
    rows = []
    for chunk in _ROW.findall(body):
        cells = _CELL.findall(chunk)
        if len(cells) < 9:
            continue
        player = _PLAYER.search(cells[3])
        if not player:
            continue
        from_m, to_m = _TEAM.search(cells[0]), _TEAM.search(cells[2])
        from_team = from_m.group(1) if from_m else ""
        to_team = to_m.group(1) if to_m else ""
        if from_team and to_team:
            kind = "re_sign" if from_team == to_team else "move"
        else:
            # No "from" team means he was not on an NBA roster when he signed:
            # overseas, G-League, or out of the league. Spotrac leaves the cell
            # blank rather than naming his last NBA club.
            kind = "from_outside"
        rows.append({
            "year": year,
            "spotrac_id": player.group(1),
            "player": _text(player.group(2)),
            "from_team": from_team,
            "to_team": to_team,
            "kind": kind,
            "pos": _text(cells[4]),
            "years": _text(cells[5]),
            "value": _money(cells[6]),
            "aav": _money(cells[7]),
            "status": _text(cells[8]),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    years = args.years or list(range(FIRST_YEAR, LAST_YEAR + 1))
    all_rows: list[dict] = []
    for n, year in enumerate(years):
        if n and (args.refresh or not (CACHE / f"{year}.html").exists()):
            time.sleep(PACE_SECONDS)
        rows = parse(year, fetch(year, args.refresh))
        print(f"{year}: {len(rows)} signings", flush=True)
        all_rows.extend(rows)

    # The tables occasionally repeat a row verbatim (a player listed under two
    # transaction entries for the same deal). Same year + id + value = one deal.
    seen, deduped = set(), []
    for r in all_rows:
        key = (r["year"], r["spotrac_id"], r["value"], r["from_team"], r["to_team"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(deduped[0].keys()))
        w.writeheader()
        w.writerows(deduped)

    re_signs = sum(r["kind"] == "re_sign" for r in deduped)
    moves = sum(r["kind"] == "move" for r in deduped)
    outside = len(deduped) - re_signs - moves
    print(f"\n{OUT.relative_to(ROOT)}: {len(deduped)} rows "
          f"({len(all_rows) - len(deduped)} duplicates dropped)")
    print(f"  re-signed with incumbent : {re_signs}")
    print(f"  changed teams            : {moves}")
    print(f"  signed from outside NBA  : {outside}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
