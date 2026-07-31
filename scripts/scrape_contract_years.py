#!/usr/bin/env python3
"""Scrape years-remaining + option type per player from Spotrac's team yearly pages.

Why Spotrac and not Basketball Reference: BBRef's contract tables are a season
behind for anyone who signed in the 2026 offseason (Trae Young still shows the
$49M option he declined, not the 4yr/$212M he signed), and its rate limiting
blocks an IP for the best part of an hour. Spotrac is what master_roster.csv
already calls canonical, it is current, and a browser User-Agent gets a 200.

The shape of the source: /nba/<team-slug>/yearly renders one column per future
season and one row per player, and each salary cell carries a "pill" class that
says what kind of season it is:

    pill-club   -> team option        pill-ufa -> hits unrestricted FA that year
    pill-player -> player option      pill-rfa -> restricted FA
    (neither)   -> guaranteed / non-guaranteed money owed

So years remaining is "how many season columns from 2026-27 onward hold real
money", and the option flag is the pill on the LAST of those columns. That is
precisely the "1+TO" vs "2" distinction.

Output: data/contract_years_2026_27.csv
    team, player, years_left, last_year_type, end_season, seasons

Usage:
    python scripts/scrape_contract_years.py              # cached pages reused
    python scripts/scrape_contract_years.py --refresh    # re-download every page
    python scripts/scrape_contract_years.py --teams WAS OKC
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
OUT = ROOT / "data" / "contract_years_2026_27.csv"
CACHE = ROOT / "cache" / "spotrac_yearly"

# Spotrac 403s a bare urllib/requests call; a normal browser UA gets a 200.
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
PACE_SECONDS = 3.0        # Spotrac throttles bursts; 3s has run clean before
BOOKS_START = 2026        # 2026-27 is "year one" for these pages

# app abbreviation -> spotrac slug
TEAMS = {
    "ATL": "atlanta-hawks",        "BOS": "boston-celtics",
    "BKN": "brooklyn-nets",        "CHA": "charlotte-hornets",
    "CHI": "chicago-bulls",        "CLE": "cleveland-cavaliers",
    "DAL": "dallas-mavericks",     "DEN": "denver-nuggets",
    "DET": "detroit-pistons",      "GSW": "golden-state-warriors",
    "HOU": "houston-rockets",      "IND": "indiana-pacers",
    "LAC": "la-clippers",          "LAL": "los-angeles-lakers",
    "MEM": "memphis-grizzlies",    "MIA": "miami-heat",
    "MIL": "milwaukee-bucks",      "MIN": "minnesota-timberwolves",
    "NOP": "new-orleans-pelicans", "NYK": "new-york-knicks",
    "OKC": "oklahoma-city-thunder", "ORL": "orlando-magic",
    "PHI": "philadelphia-76ers",   "PHX": "phoenix-suns",
    "POR": "portland-trail-blazers", "SAC": "sacramento-kings",
    "SAS": "san-antonio-spurs",    "TOR": "toronto-raptors",
    "UTA": "utah-jazz",            "WAS": "washington-wizards",
}

_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_TD = re.compile(r"<td\b[^>]*>.*?</td>", re.S)
_PLAYER = re.compile(r'/nba/player/_/id/\d+/[^"]*"[^>]*>\s*([^<]+?)\s*</a>', re.S)
_PILL = re.compile(r"class='pill ([^']*)'|class=\"pill ([^\"]*)\"")
_SEASON_HDR = re.compile(r"(20\d\d)-\d\d")
_MARK = {"guaranteed": "G", "team_option": "T",
         "player_option": "P", "two_way": "W"}


def fetch(team: str, slug: str, refresh: bool) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{team}.html"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8", errors="replace")
    url = f"https://www.spotrac.com/nba/{slug}/yearly"
    time.sleep(PACE_SECONDS)
    # Ask for the status code too: a wrong slug 404s and a throttle returns an
    # empty 200, and those want different fixes. Reporting "throttled?" for both
    # sent me looking at rate limits when the Clippers were simply at
    # /nba/la-clippers, not /nba/los-angeles-clippers.
    res = subprocess.run(["curl", "-s", "-A", UA, "-w", "\n%{http_code}", url],
                         capture_output=True, text=True)
    body, _, code = (res.stdout or "").rpartition("\n")
    if code.strip() != "200":
        raise RuntimeError(f"{team}: http {code.strip() or '?'} for {url}")
    if len(body) < 50_000:
        raise RuntimeError(f"{team}: http 200 but only {len(body)} bytes — throttled")
    path.write_text(body, encoding="utf-8")
    return body


def parse(team: str, page: str) -> list[dict]:
    """One row per player on the active roster, with years left and option type."""
    # Only the Active Roster table. Later tables are dead money / two-way / FA
    # pools and would double-count or add people who are not under contract.
    start = page.find("Active Roster")
    if start == -1:
        raise RuntimeError(f"{team}: no Active Roster section")
    chunk = page[start:]
    end = chunk.find("<h2>", 20)
    if end != -1:
        chunk = chunk[:end]

    head = re.search(r"<thead.*?</thead>", chunk, re.S)
    if not head:
        raise RuntimeError(f"{team}: no header row")
    seasons = _SEASON_HDR.findall(head.group(0))
    if not seasons:
        raise RuntimeError(f"{team}: no season columns")

    out: list[dict] = []
    for tr in _TR.findall(chunk):
        if "/nba/player/" not in tr:
            continue
        m = _PLAYER.search(tr)
        if not m:
            continue
        name = _html.unescape(m.group(1)).strip()
        cells = _TD.findall(tr)
        # The season columns are the trailing N cells (player, pos, age lead).
        season_cells = cells[-len(seasons):] if len(cells) >= len(seasons) else []
        run: list[tuple[str, str]] = []
        for season, cell in zip(seasons, season_cells):
            pm = _PILL.search(cell)
            cls = (pm.group(1) or pm.group(2) or "") if pm else ""
            text = _html.unescape(re.sub(r"<[^>]+>", " ", cell))
            has_money = bool(re.search(r"\$[\d,]+", cell))
            # A two-way is a real roster year, it just shows the words rather
            # than a figure. Without this the player is silently dropped.
            is_two_way = "Two-Way" in text
            if "pill-ufa" in cls or "pill-rfa" in cls:
                break                      # he reaches free agency here: contract over
            if not has_money and not is_two_way:
                break                      # empty column: nothing owed beyond this
            kind = ("two_way" if is_two_way else
                    "team_option" if "pill-club" in cls else
                    "player_option" if "pill-player" in cls else "guaranteed")
            run.append((season, kind))
        if not run:
            continue
        end_season = run[-1][0]
        out.append({
            "team": team,
            "player": name,
            "years_left": len(run),
            "last_year_type": run[-1][1],
            "end_season": f"{end_season}-{str(int(end_season)+1)[-2:]}",
            "seasons": "|".join(f"{s}:{_MARK[k]}" for s, k in run),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download every page")
    ap.add_argument("--teams", nargs="*", help="limit to these abbreviations")
    args = ap.parse_args()

    todo = {t: s for t, s in TEAMS.items() if not args.teams or t in args.teams}
    rows: list[dict] = []
    failed: list[str] = []
    for i, (team, slug) in enumerate(sorted(todo.items()), 1):
        try:
            page = fetch(team, slug, args.refresh)
            got = parse(team, page)
            rows.extend(got)
            print(f"  [{i:2}/{len(todo)}] {team}: {len(got)} players", flush=True)
        except Exception as e:
            failed.append(team)
            print(f"  [{i:2}/{len(todo)}] {team}: FAILED — {e}", flush=True)

    if failed:
        print(f"\nFAILED teams: {','.join(failed)} — not writing a partial file")
        return 1

    # Spotrac uses legal names where the roster uses the name everyone knows
    # (Nah'Shon vs Bones Hyland, Mohamed vs Mo Bamba). Emit the ROSTER's name so
    # this file joins onto master_roster.csv, keeping Spotrac's for provenance.
    #
    # Matching on bare surname is NOT safe and was wrong here first time round:
    # "Jaime Jaquez Jr.".split()[-1] is "Jr.", so every suffixed player on a team
    # collided, and Jaylin/Kenrich/Jalen Williams all collapsed onto one man.
    # So: strip the suffix, only consider players neither side has already
    # matched exactly, and only rename when the surname is UNIQUE among those.
    def surname(name: str) -> str:
        parts = [w for w in name.replace(".", "").split()
                 if w.lower() not in ("jr", "sr", "ii", "iii", "iv", "v")]
        return parts[-1].lower() if parts else name.lower()

    def key(name: str) -> str:
        return re.sub(r"[^a-z]", "", name.lower())

    roster: dict[str, list[str]] = {}
    mr = ROOT / "data" / "master_roster.csv"
    if mr.exists():
        for r in csv.DictReader(ln for ln in open(mr) if not ln.lstrip().startswith("#")):
            if r.get("player") and r.get("team"):
                roster.setdefault(r["team"].strip(), []).append(r["player"].strip())

    for r in rows:
        r["spotrac_player"] = r["player"]
    matched_roster = {(r["team"], key(r["player"])) for r in rows}
    renamed = 0
    for r in rows:
        if (r["team"], key(r["player"])) in {
                (t, key(n)) for t, ns in roster.items() for n in ns}:
            continue                                   # already the roster spelling
        cands = [n for n in roster.get(r["team"], [])
                 if surname(n) == surname(r["player"])
                 and (r["team"], key(n)) not in matched_roster]
        if len(cands) == 1:                            # unambiguous alias only
            r["player"] = cands[0]
            matched_roster.add((r["team"], key(cands[0])))
            renamed += 1
    if renamed:
        print(f"  reconciled {renamed} name(s) to the roster spelling")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        fh.write("# Years remaining on every contract on the 2026-27 books, per player.\n")
        fh.write("# Source: spotrac.com/nba/<team>/yearly (canonical, and current --\n")
        fh.write("#   BBRef's contract tables still show pre-2026-offseason deals).\n")
        fh.write("# years_left counts seasons from 2026-27 forward that still owe money;\n")
        fh.write("#   last_year_type is what the FINAL of those seasons is, so a label\n")
        fh.write("#   reads '2' for two guaranteed years and '1+TO' for one guaranteed\n")
        fh.write("#   year followed by a team option.\n")
        fh.write("# seasons: per-year detail, G=guaranteed T=team option P=player option.\n")
        w = csv.DictWriter(fh, fieldnames=["team", "player", "years_left",
                                           "last_year_type", "end_season", "seasons",
                                           "spotrac_player"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["team"], -r["years_left"], r["player"])):
            w.writerow(r)
    print(f"\nwrote {len(rows)} rows across {len({r['team'] for r in rows})} teams -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
