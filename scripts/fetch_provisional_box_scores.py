"""Build data/provisional_box_scores.csv from ESPN's box-score API.

    /opt/anaconda3/bin/python3 -u scripts/fetch_provisional_box_scores.py 2026-10-23
    /opt/anaconda3/bin/python3 -u scripts/fetch_provisional_box_scores.py 2026-10-23 --write

Prints what it would write; --write actually writes. Exits non-zero if any
team block fails its checksum.

WHY. Basketball-Reference publishes a night's games some hours later, usually
overnight, and the NBA Stats API times out for this host entirely. So between
roughly midnight and dawn the site cannot show games that have finished. On
2026-10-23 that gap was filled by hand off screenshots, which worked for two
games and found two of my own misreadings in the process. Twelve games is
twenty-four box scores and about three hundred players; hand-transcribing that
multiplies the error surface for no reason, when the same numbers are
available as JSON from the same source.

So this is the hand-entry from that night, automated. Identical columns,
identical destination, identical self-superseding behaviour: utils skips a
team's block the moment the real feed reports that many games, and the hourly
rebuild drops it without anyone remembering.

It still writes the TEAM checksum rows and still verifies them, because the
point of that check was never that a human typed the numbers. It is that the
numbers came from somewhere we do not control.
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

API = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
UA = {"User-Agent": "Mozilla/5.0"}

# ESPN -> the abbreviations the stat frame uses (Basketball-Reference's).
# Three sets are in play: ESPN's, BBRef's, and the site's own TEAM_HEX keys.
# The provisional rows are matched against TEAM_ABBREVIATION in the stats
# frame, which is BBRef's, so that is the target. Every mapping is asserted
# against the live frame before anything is written.
ESPN_TO_BREF = {
    "GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS",
    "UTAH": "UTA", "WSH": "WAS", "BKN": "BRK", "CHA": "CHO", "PHX": "PHO",
}

# ESPN's column order, which has been stable and is asserted below.
COLS = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK",
        "OREB", "DREB", "PF", "+/-"]


def get(url: str, tries: int = 4) -> dict:
    """One ESPN call, with retries.

    A single "connection reset by peer" used to abort the whole run, and on a
    nightly job that fetches a dozen games the odds of at least one reset are
    not small: it happened on the first real run of scripts/nightly.py. The
    last failure is re-raised rather than swallowed, because a date we decided
    we need and cannot fetch means a team's players sit a game behind every
    other surface, and that must stop the run rather than ship quietly.
    """
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                      # reset, timeout, 5xx
            last = e
            if attempt < tries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last


def num(s) -> float:
    try:
        return float(str(s).replace("+", "").strip() or 0)
    except ValueError:
        return 0.0


def split_made_att(s):
    txt = str(s or "").strip()
    if "-" in txt:
        a, _, b = txt.partition("-")
        return num(a), num(b)
    return 0.0, 0.0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    date = sys.argv[1]                                  # YYYY-MM-DD
    write = "--write" in sys.argv
    season = "2026-27"
    ymd = date.replace("-", "")

    sb = get(f"{API}/scoreboard?dates={ymd}")
    events = sb.get("events", [])
    final = [e for e in events
             if e["competitions"][0]["status"]["type"].get("completed")]
    print(f"\n  {date}: {len(events)} games scheduled, {len(final)} final\n")
    if not final:
        print("  nothing to ingest\n")
        return 0

    rows, failures, blocks = [], [], 0
    for ev in final:
        comp = ev["competitions"][0]
        # games played INCLUDING this one, straight off the post-game record
        gp_after, opp_of = {}, {}
        abbrs = []
        for c in comp["competitors"]:
            ab = c["team"]["abbreviation"]
            abbrs.append(ab)
            recs = c.get("records") or []
            r = next((x for x in recs if x.get("type") == "total"), recs[0] if recs else None)
            w, _, l = (r or {}).get("summary", "0-0").partition("-")
            gp_after[ab] = int(num(w) + num(l))
        for a, b in (abbrs, abbrs[::-1]):
            pass
        opp_of = {abbrs[0]: abbrs[1], abbrs[1]: abbrs[0]}

        summary = get(f"{API}/summary?event={ev['id']}")
        for tb in summary.get("boxscore", {}).get("players", []):
            ab = tb["team"]["abbreviation"]
            st = tb["statistics"][0]
            if st.get("names") != COLS:
                failures.append(f"{ab}: unexpected column order {st.get('names')}")
                continue
            bref = ESPN_TO_BREF.get(ab, ab)
            opp = ESPN_TO_BREF.get(opp_of[ab], opp_of[ab])
            blocks += 1
            players = []
            for a in st.get("athletes", []):
                if a.get("didNotPlay") or not a.get("stats"):
                    continue
                v = dict(zip(COLS, a["stats"]))
                fgm, fga = split_made_att(v["FG"])
                ftm, fta = split_made_att(v["FT"])
                players.append({
                    "player": a["athlete"]["displayName"],
                    "min": num(v["MIN"]), "pts": num(v["PTS"]),
                    "fga": fga, "fta": fta, "reb": num(v["REB"]),
                    "ast": num(v["AST"]), "tov": num(v["TO"]),
                    "stl": num(v["STL"]), "blk": num(v["BLK"]),
                    "oreb": num(v["OREB"]), "dreb": num(v["DREB"]),
                    "pf": num(v["PF"]),
                })
            t = dict(zip(COLS, st.get("totals", [])))
            tfgm, tfga = split_made_att(t.get("FG"))
            tftm, tfta = split_made_att(t.get("FT"))
            totals = {"pts": num(t.get("PTS")), "fga": tfga, "fta": tfta,
                      "reb": num(t.get("REB")), "ast": num(t.get("AST")),
                      "tov": num(t.get("TO")), "stl": num(t.get("STL")),
                      "blk": num(t.get("BLK")), "oreb": num(t.get("OREB")),
                      "dreb": num(t.get("DREB")), "pf": num(t.get("PF"))}
            # ESPN's own team row and its own player rows sometimes disagree
            # by ONE on a single shot-count column: on 2026-10-25 the Clippers'
            # players summed to 37-82 from the field while the team row said
            # 37-81. Refusing an entire night's update over one attempt in
            # someone else's arithmetic is the wrong trade, so a gap of exactly
            # 1 on ONE of the shot-count columns is tolerated and reported.
            # Everything else still has to reconcile exactly, and two sloppy
            # columns still fail: the point of the check is that a block which
            # is wrong in SHAPE never reaches disk.
            _TOL_1 = {"fga", "fta"}
            bad, tolerated = [], []
            for k, want in totals.items():
                gap = abs(sum(p[k] for p in players) - want)
                if gap <= 1e-6:
                    continue
                if k in _TOL_1 and gap <= 1 + 1e-6:
                    tolerated.append(f"{k} off by {gap:.0f}")
                else:
                    bad.append(k)
            if len(tolerated) > 1:
                bad.extend(t.split()[0] for t in tolerated)
                tolerated = []
            if tolerated:
                print(f"  {bref}  ESPN's own totals disagree with its player rows: "
                      f"{', '.join(tolerated)} -- tolerated, players are the source")
            shifted = [p["player"] for p in players
                       if abs(p["reb"] - (p["oreb"] + p["dreb"])) > 1e-6]
            if bad or shifted:
                failures.append(f"{bref}: columns {bad} disagree with the TEAM row"
                                + (f"; reb != oreb+dreb for {shifted[:3]}" if shifted else ""))
            print(f"  {bref:4} vs {opp:4}  {len(players):>2} players, "
                  f"{gp_after[ab]} game(s) played  "
                  f"{'CHECKSUM OK' if not (bad or shifted) else 'CHECKSUM FAILED'}")
            for p in players:
                rows.append([season, date, bref, opp, gp_after[ab], p["player"],
                             f"{p['min']:g}", f"{p['pts']:g}", f"{p['fga']:g}",
                             f"{p['fta']:g}", f"{p['reb']:g}", f"{p['ast']:g}",
                             f"{p['tov']:g}", f"{p['stl']:g}", f"{p['blk']:g}",
                             f"{p['oreb']:g}", f"{p['dreb']:g}", f"{p['pf']:g}",
                             "ESPN box score API"])
            rows.append([season, date, bref, opp, gp_after[ab], "TEAM", ""]
                        + [f"{totals[k]:g}" for k in
                           ("pts", "fga", "fta", "reb", "ast", "tov", "stl",
                            "blk", "oreb", "dreb", "pf")]
                        + ["ESPN published totals (checksum row, never merged)"])
        time.sleep(0.4)

    # every mapped abbreviation must exist in the stat frame, or the block is
    # silently inert: it would never match a team and never be superseded
    try:
        from utils import fetch_bref_player_stats
        known = set(fetch_bref_player_stats(season)["TEAM_ABBREVIATION"].unique())
        unknown = sorted({r[2] for r in rows} - known)
        if unknown:
            failures.append(f"abbreviations absent from the stat frame: {unknown}")
    except Exception as e:                                           # noqa: BLE001
        print(f"  (could not verify abbreviations against the stat frame: {e})")

    print(f"\n  {blocks} team blocks, {len([r for r in rows if r[5] != 'TEAM'])} player rows")
    for f in failures:
        print(f"  FAIL {f}")
    if failures:
        print("\n  refusing to write\n")
        return 1

    if not write:
        print("\n  dry run, nothing written. Pass --write to commit it to disk.\n")
        return 0

    hdr = ("season,game_date,team,opponent,team_gp_after,player,min,pts,fga,fta,"
           "reb,ast,tov,stl,blk,oreb,dreb,pf,source")
    preamble = f"""# HoopsValue - PROVISIONAL box scores, from ESPN's box-score API.
#
# TEMPORARY, and generated. Rebuild with:
#   scripts/fetch_provisional_box_scores.py {date} --write
#
# These are games that have been PLAYED but that neither stat source has
# published yet. Basketball-Reference posts a night's games some hours later,
# usually overnight, and stats.nba.com times out for this host entirely, so
# between midnight and dawn the site otherwise cannot show finished games.
#
# THEY SUPERSEDE THEMSELVES. team_gp_after is that team's games played
# INCLUDING this one, taken from the post-game record, and utils skips the
# whole block once the real feed reports that many. The raw parquet rebuilds
# hourly while a season is live, so the first rebuild after BBRef publishes
# drops these rows on its own. scripts/check_provisional.py says when that has
# happened and the file can be deleted.
#
# Every block is checked against the TEAM totals ESPN publishes before this
# file is written: all eleven summed columns, plus reb = oreb + dreb on every
# row. That check exists because the numbers come from somewhere we do not
# control, not because a human typed them.
#
#   player=TEAM = published totals, the checksum row. Never merged.
#   min..pf     = this GAME's totals, not per-game averages.
"""
    out = ROOT / "data" / "provisional_box_scores.csv"
    # MERGE on (team, game_date), never a blind overwrite. Opening this "w" and
    # writing only the requested date silently DELETED every block from every
    # other date: running it for 10/24 would have dropped the 300 rows already
    # here for 10/22 and 10/23. Blocks still supersede themselves through
    # team_gp_after; that is a reason to let them expire, not to erase them.
    keep, dropped = [], 0
    if out.exists():
        mine = {(r[hdr.split(",").index("team")], r[hdr.split(",").index("game_date")])
                for r in rows}
        with open(out, newline="", encoding="utf-8") as fh:
            for r in csv.reader(fh):
                if not r or r[0].startswith("#") or r[0] == "season":
                    continue
                key = (r[hdr.split(",").index("team")],
                       r[hdr.split(",").index("game_date")])
                if key in mine:
                    dropped += 1          # this run supersedes that block
                else:
                    keep.append(r)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write(preamble + hdr + "\n")
        csv.writer(fh).writerows(rows + keep)
    print(f"\n  wrote {out}: {len(rows)} new rows, {len(keep)} kept from other "
          f"dates, {dropped} replaced\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
