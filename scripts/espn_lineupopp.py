"""Rebuild a pbpstats-shaped LineupOpponent file from ESPN's play-by-play.

    /opt/anaconda3/bin/python3 -u scripts/espn_lineupopp.py validate [EVENT_ID ...]
    /opt/anaconda3/bin/python3 -u scripts/espn_lineupopp.py write GAME_ID EVENT_ID

Why this exists: pbpstats never PROCESSED two games that were played
(HOU@SAS and LAC@LAL on 2026-10-23), so four teams carry a box line a game
ahead of their lineup data and Walker Kessler was scored off one of his two
games. Fetching those games from pbpstats by id returns no stats block, so the
only route is to rebuild them from another source.

The output deliberately mimics pbpstats' own file shape, including its least
obvious convention: the rows under "Home" are named with HOME players but carry
the AWAY team's offensive outcomes, because they describe what the opponent did
while that lineup defended. Verified against the real files, not assumed.

Geometry: ESPN's shot coordinates are feet with the basket at (25, 0), derived
by regressing 247 shots against the distance stated in their own text (mean
absolute error 0.64 ft, which is ESPN's own integer rounding).

NOTHING is written until `validate` shows the rebuild agrees with pbpstats on
games BOTH sources have. Mixing two measurement pipelines inside one on/off
column is worse than a missing game, so the agreement is the whole gate.
"""
from __future__ import annotations
import json, math, re, sys, urllib.request
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PB = ROOT / "cache" / "pbpstats"
UA = {"User-Agent": "Mozilla/5.0"}

BX, BY = 25.0, 0.0                 # basket, in ESPN's coordinate feet
RIM, SHORT = 4.0, 14.0             # at-rim and short-mid-range cutoffs
CORNER_Y = 14.0                    # a three below this y is a corner three

SHOT_WORDS = ("Shot", "Jump", "Layup", "Dunk", "Hook", "Tip", "Floating",
              "Finger Roll", "Fade Away", "Bank")


def get(url, tries=4):
    d = 3.0
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=40).read())
        except Exception:
            if i == tries - 1:
                raise
            import time
            time.sleep(d); d *= 1.8


def is_shot(t):
    return any(w in t for w in SHOT_WORDS) and "Free Throw" not in t


def zone(x, y, three):
    """pbpstats' five zones from a coordinate."""
    d = math.hypot(x - BX, y - BY)
    if three:
        return "Corner3" if y <= CORNER_Y else "Arc3"
    if d <= RIM:
        return "AtRim"
    if d <= SHORT:
        return "ShortMidRange"
    return "LongMidRange"


def build(event_id, local_date=None):
    s = get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
            f"summary?event={event_id}")
    teams, starters, names, side_of = {}, {}, {}, {}
    for blk in s["boxscore"]["players"]:
        tid = str(blk["team"]["id"])
        teams[tid] = blk["team"]["abbreviation"]
        st = set()
        for a in (blk.get("statistics") or [{}])[0].get("athletes") or []:
            aid = str(a["athlete"]["id"])
            names[aid] = a["athlete"].get("displayName")
            if a.get("starter"):
                st.add(aid)
        starters[tid] = st
    comp = s["header"]["competitions"][0]
    for c in comp["competitors"]:
        side_of[str(c["team"]["id"])] = c["homeAway"].capitalize()
    # comp["date"] is the UTC tip-off, so a 10/23 evening game in the US reads
    # as 10/24 and both the dedupe key and the game ordering go wrong. The
    # caller passes the date ESPN's own SCOREBOARD filed it under.
    date = local_date or str(comp.get("date", ""))[:10]

    on = {t: set(starters[t]) for t in teams}
    ids = list(teams)
    other = {ids[0]: ids[1], ids[1]: ids[0]}
    agg = defaultdict(lambda: defaultdict(float))   # (defending tid, lineup) -> stats
    viol = defaultdict(int)

    def lineup_key(tid):
        return ", ".join(sorted(names.get(a, a) for a in on[tid]))

    def add(off_tid, field, v=1.0):
        """Credit an OFFENSIVE outcome to the DEFENDING team's current lineup."""
        dtid = other[off_tid]
        agg[(dtid, lineup_key(dtid))][field] += v

    last_off = None
    prev_total = 0
    for p in s["plays"]:
        a_sc, h_sc = p.get("awayScore"), p.get("homeScore")
        if a_sc is None or h_sc is None:
            sc_delta = 0
        else:
            sc_delta = (a_sc + h_sc) - prev_total
            prev_total = a_sc + h_sc
        ttype = (p.get("type") or {}).get("text") or ""
        tid = str((p.get("team") or {}).get("id") or "")
        txt = p.get("text") or ""
        parts = [str(x["athlete"]["id"]) for x in (p.get("participants") or [])
                 if x.get("athlete")]

        if ttype == "Substitution" and tid in on:
            m = re.match(r"(.+?) enters the game for (.+)", txt)
            cin = cout = None
            if m:
                for aid in parts:
                    if names.get(aid) == m.group(1).strip():
                        cin = aid
                    elif names.get(aid) == m.group(2).strip():
                        cout = aid
            if cin is None or cout is None:
                if len(parts) == 2:
                    cin, cout = parts[0], parts[1]
            if cin and cout:
                on[tid].discard(cout); on[tid].add(cin)
            continue

        if tid in on and parts and ttype not in ("End Period", "Full Timeout",
                                                 "Short Timeout", "Official Timeout"):
            if parts[0] not in on[tid]:
                viol[tid] += 1
                on[tid].add(parts[0])
                if len(on[tid]) > 5:                # drop whoever we can least justify
                    on[tid].discard(sorted(on[tid] - {parts[0]})[0])

        if tid not in on:
            continue
        shot = is_shot(ttype)
        made = bool(p.get("scoringPlay"))
        c = p.get("coordinate") or {}
        x, y = c.get("x"), c.get("y")
        has_xy = isinstance(x, int) and isinstance(y, int) and abs(x) < 200 and abs(y) < 200
        # A made shot's value is unambiguous from the score delta; ESPN's text
        # says "three point" on only 27 of 29 made threes, so trusting the text
        # misclassified about 9% of threes as twos. A MISSED shot has no delta,
        # so fall back to geometry: 22 ft in the corner, 23.75 on the arc.
        if made and sc_delta in (2, 3):
            three = sc_delta == 3
        elif has_xy:
            dd = math.hypot(float(x) - BX, float(y) - BY)
            three = dd >= (22.0 if float(y) <= CORNER_Y else 23.75)
        else:
            three = "three point" in txt.lower() or "3-pt" in txt.lower()

        if shot:
            add(tid, "FG3A" if three else "FG2A")
            if made:
                add(tid, "FG3M" if three else "FG2M")
                add(tid, "Points", 3.0 if three else 2.0)
            if has_xy:
                z = zone(float(x), float(y), three)
                add(tid, f"{z}FGA")
                if made:
                    add(tid, f"{z}FGM")
            # a made field goal ends the possession
            if made:
                add(tid, "OffPoss"); last_off = tid
        elif "Free Throw" in ttype:
            add(tid, "FTA")
            if made:
                add(tid, "Points", 1.0)
            # Only a MADE last free throw ends the possession here. A missed one
            # is ended by the Defensive Rebound branch below, so counting both
            # would double the possession.
            m = re.search(r"(\d+) of (\d+)", ttype)
            if made and m and m.group(1) == m.group(2):
                add(tid, "OffPoss"); last_off = tid
        elif "Turnover" in ttype or ttype == "Traveling":
            add(tid, "Turnovers")
            if "Bad Pass" in ttype or "Lost Ball" in ttype:
                add(tid, "LiveBallTurnovers")
            add(tid, "OffPoss"); last_off = tid
        elif ttype == "Offensive Rebound":
            # Playerless rebounds are TEAM rebounds and are EXCLUDED, which
            # reproduces ESPN's own box score exactly. That still lands 11-21%
            # under pbpstats; including them overshot by 13-48%, so pbpstats
            # excludes them too and the residual gap is some other definitional
            # choice of theirs. Left as the one known-biased channel: it feeds
            # d_oreb, which only options 7 and 11 use. The box proxy and option
            # 13 -- the two the blend is built from -- do not touch it.
            if parts:
                add(tid, "OffRebounds")
        elif ttype == "Defensive Rebound":
            # the possession that just ended belonged to the OTHER team, and
            # that is true of a team rebound too, so this is not gated on parts
            add(other[tid], "OffPoss"); last_off = other[tid]
        elif "Foul" in ttype:
            # ESPN's team on a foul is the team that COMMITTED it, and a row
            # carries the OPPONENT's stat line, so a foul by X belongs in the
            # row of X's opponent -- which is exactly what add(X, ...) does.
            add(tid, "Fouls")

    # ESPN's abbreviations differ from the ones pbpstats uses on several teams.
    # Writing ESPN's would create a phantom team: HOU@SAS would attribute every
    # Spurs player to "SA", splitting their season across two labels.
    E2N = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA",
           "WSH": "WAS", "PHO": "PHX", "BKN": "BKN", "CHA": "CHA"}
    teams = {t: E2N.get(ab, ab) for t, ab in teams.items()}
    out = {"stats": {}, "home_team_abbreviation": None, "away_team_abbreviation": None,
           "date": date, "league": "nba", "season": "", "season_type": "Regular Season",
           "_source": "espn", "_event_id": str(event_id),
           "_lineup_violations": {teams[t]: viol[t] for t in teams}}
    for tid, ab in teams.items():
        out[f"{side_of[tid].lower()}_team_abbreviation"] = ab
    rows = defaultdict(list)
    for (dtid, key), st in agg.items():
        d = {"Name": key}
        d.update({k: round(v, 4) for k, v in st.items()})
        d.setdefault("OffPoss", 0.0)
        d["DefPoss"] = d["OffPoss"]
        rows[side_of[dtid]].append(d)
    for side in ("Home", "Away"):
        out["stats"][side] = {"FullGame": rows.get(side, [])}
    return out, teams, side_of


# ─────────────────────────────────────────────────────────────────────────────

FIELDS = ("OffPoss", "Points", "FG2A", "FG3A", "AtRimFGA", "Turnovers",
          "OffRebounds", "FTA")

# ESPN's own team box score, which my play-walk must reproduce. Any gap to
# pbpstats AFTER this matches is pbpstats' definition, not my parse error.
BOX_MAP = {"PTS": "Points", "3PT": "FG3", "FG": "FG", "FT": "FTA",
           "TO": "Turnovers", "OREB": "OffRebounds"}


def box_totals(event_id):
    s = get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
            f"summary?event={event_id}")
    out = {}
    for t in s["boxscore"]["teams"]:
        ab = t["team"]["abbreviation"]
        d = {}
        for st in t.get("statistics") or []:
            d[st.get("name") or st.get("abbreviation")] = st.get("displayValue")
        out[ab] = d
    return out


def selfcheck(events):
    """Does the play-walk reproduce ESPN's own team totals?"""
    print(f"{'game':12} {'team':5} {'field':12} {'walk':>8} {'box':>8} {'diff':>7}")
    print("-" * 60)
    for ev in events:
        doc, teams, side_of = build(ev)
        box = box_totals(ev)
        for side in ("Home", "Away"):
            rows = (doc["stats"].get(side) or {}).get("FullGame") or []
            tot = defaultdict(float)
            for r in rows:
                for k, v in r.items():
                    if k != "Name":
                        try:
                            tot[k] += float(v)
                        except (TypeError, ValueError):
                            pass
            # rows under Home carry the AWAY team's stats
            opp_side = "Away" if side == "Home" else "Home"
            ab = doc[f"{opp_side.lower()}_team_abbreviation"]
            b = box.get(ab, {})
            def bx(key):
                v = b.get(key)
                if not v:
                    return None
                return float(str(v).split("-")[0]) if "-" in str(v) else float(v)
            checks = [("Points", bx("points") or bx("PTS")),
                      ("Turnovers", bx("turnovers") or bx("TO")),
                      ("OffRebounds", bx("offensiveRebounds") or bx("OREB")),
                      ("FG3M", bx("threePointFieldGoalsMade-threePointFieldGoalsAttempted")
                       or bx("3PT")),
                      ("FTA", None)]
            for fld, want in checks:
                if want is None:
                    continue
                got = tot.get(fld, 0.0)
                print(f"{doc['away_team_abbreviation']}@{doc['home_team_abbreviation']:6} "
                      f"{ab:5} {fld:12} {got:8.0f} {want:8.0f} {got - want:7.0f}")
        print()


def per_player(doc):
    """{player: {field: total}} summed over every lineup the player appears in."""
    out = defaultdict(lambda: defaultdict(float))
    for side in ("Home", "Away"):
        for r in (doc["stats"].get(side) or {}).get("FullGame") or []:
            who = [x.strip() for x in str(r.get("Name", "")).split(",") if x.strip()]
            for w in who:
                for f in FIELDS:
                    try:
                        out[w][f] += float(r.get(f) or 0)
                    except (TypeError, ValueError):
                        pass
    return out


def validate(events):
    """Compare the rebuild against pbpstats' own file for the same game."""
    import glob
    by_pair = {}
    for f in sorted(glob.glob(str(PB / "lineupopp_00226*.json"))):
        d = json.loads(Path(f).read_text())
        by_pair[(d.get("away_team_abbreviation"), d.get("home_team_abbreviation"))] = d
    # ESPN uses a few different abbreviations
    E2N = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA",
           "WSH": "WAS", "PHO": "PHX"}
    print(f"{'game':14} {'field':12} {'espn':>10} {'pbpstats':>10} {'diff':>8} {'rel':>7}")
    print("-" * 70)
    for ev in events:
        doc, teams, side_of = build(ev)
        a = E2N.get(doc["away_team_abbreviation"], doc["away_team_abbreviation"])
        h = E2N.get(doc["home_team_abbreviation"], doc["home_team_abbreviation"])
        ref = by_pair.get((a, h))
        label = f"{a}@{h}"
        if ref is None:
            print(f"{label:14} not in pbpstats, cannot validate "
                  f"(violations {doc['_lineup_violations']})")
            continue
        mine, theirs = per_player(doc), per_player(ref)
        shared = sorted(set(mine) & set(theirs))
        print(f"{label:14} players matched {len(shared)} of "
              f"{len(theirs)} in pbpstats  violations {doc['_lineup_violations']}")
        for f in FIELDS:
            m = sum(mine[p][f] for p in shared)
            t = sum(theirs[p][f] for p in shared)
            rel = (m - t) / t if t else float("nan")
            print(f"{'':14} {f:12} {m:10.0f} {t:10.0f} {m - t:8.0f} {rel:+7.1%}")
        print()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if mode == "selfcheck":
        selfcheck(sys.argv[2:] or ["401909850"])
    elif mode == "validate":
        evs = sys.argv[2:] or ["401909850", "401909852", "401909848"]
        validate(evs)
    elif mode == "backfill":
        # Find every played game the lineup feed lacks, look its ESPN event up
        # under the date ESPN filed it, and write it. No ids by hand, and the
        # date is the scoreboard's, not the UTC tip-off.
        sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
        import logging
        logging.disable(logging.WARNING)
        import def_lib as L
        logging.disable(logging.NOTSET)
        _e, _f, missing = L.feed_gap("2026-27")
        want = {(m, d) for m, d in missing}
        if not want:
            print("nothing missing")
            sys.exit(0)
        E2N = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA",
               "WSH": "WAS", "PHO": "PHX"}
        wrote = 0
        for d in sorted({d for _m, d in want}):
            sb = get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
                     f"scoreboard?dates={d}")
            for x in sb.get("events", []):
                c = x["competitions"][0]
                if not c["status"]["type"].get("completed"):
                    continue
                if (x.get("season") or {}).get("type") != 2:
                    continue
                ab = {y["homeAway"]: y["team"]["abbreviation"] for y in c["competitors"]}
                a = E2N.get(ab["away"], ab["away"]); h = E2N.get(ab["home"], ab["home"])
                if (f"{a}@{h}", d) not in want:
                    continue
                iso = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                doc, _t, _s = build(x["id"], local_date=iso)
                gid = f"E{x['id']}"
                doc["GameId"] = gid
                doc["_matchup"] = f"{a}@{h}"
                (PB / f"lineupopp_{gid}.json").write_text(json.dumps(doc))
                v = doc["_lineup_violations"]
                flag = "  HIGH VIOLATIONS" if max(v.values()) > 6 else ""
                print(f"wrote lineupopp_{gid}.json  {a}@{h}  {iso}  violations {v}{flag}")
                wrote += 1
        print(f"\n{wrote} games rebuilt from ESPN")
    elif mode == "write":
        # The GameId is SYNTHETIC on purpose: "E" + the ESPN event id. Guessing
        # a real NBA id from gaps in the sequence (which is what the first two
        # backfills did) risks colliding with a different real game, and once a
        # file exists under that id the ingest skips it forever, silently
        # preferring the rebuild over the real thing. A synthetic id cannot
        # collide, and def_lib.games() drops it the moment the real game lands.
        for ev in sys.argv[2:]:
            doc, _t, _s = build(ev)
            gid = f"E{ev}"
            doc["GameId"] = gid
            doc["_matchup"] = (f"{doc['away_team_abbreviation']}@"
                               f"{doc['home_team_abbreviation']}")
            (PB / f"lineupopp_{gid}.json").write_text(json.dumps(doc))
            print(f"wrote lineupopp_{gid}.json  {doc['_matchup']}  {doc['date']}  "
                  f"violations {doc['_lineup_violations']}")
    else:
        print(__doc__)
