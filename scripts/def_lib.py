"""Shared loading for the defensive-metric options. Offline; writes nothing the site reads.

One place to build the three things every option needs:
  on/off opponent outcomes per player, from per-game lineup files
  per-player box and pbpstats counts
  role weight (share of minutes at PF/C) from BBRef's play-by-play table

Per-game files are used rather than the season endpoint because the season
endpoint returns exactly 500 lineups sorted by minutes and silently truncates
the tail, which biases every off-floor aggregate built from it.
"""
from __future__ import annotations
import json, sys
from functools import lru_cache
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from utils import normalize

PB = ROOT / "cache" / "pbpstats"
DEF = ROOT / "cache" / "bref_def"

# A "lineupopp" row is the OPPONENT's stat line while this lineup was on the
# floor (verified: the BOS lineups in 2024-11-12 BOS/ATL carry ATL's 40 twos and
# 23 rim attempts, not BOS's 18 threes). So the possessions our player actually
# defended are the opponent's OffPoss, not the row's DefPoss -- which is our own
# team's offensive possessions. They agree at the median and diverge by >15% in
# 5% of lineup rows, so DefPoss was a usable but wrong denominator.
OPP = ("OffPoss", "DefPoss", "Points", "AtRimFGA", "AtRimFGM",
       "ShortMidRangeFGA", "ShortMidRangeFGM",
       "LongMidRangeFGA", "LongMidRangeFGM", "Arc3FGA", "Arc3FGM",
       "Corner3FGA", "Corner3FGM", "FG2A", "FG3A", "Turnovers",
       "LiveBallTurnovers", "OffRebounds", "FTA", "Fouls")

# league points per shot by zone, used to price a shot-diet shift
PPS = {"rim": 1.30, "smid": 0.80, "lmid": 0.82, "arc3": 1.06, "cor3": 1.17}


@lru_cache(maxsize=None)
def games(season):
    """Games with a lineup file, oldest first, real sources preferred.

    A game rebuilt from ESPN carries a synthetic "E<event>" id. If pbpstats
    later processes the same matchup on the same date, the real file must win
    and the rebuild must disappear -- otherwise the rebuild shadows better data
    forever. Enforced here, on the READ path, because a rule applied only at
    ingest is a rule that gets forgotten the next time something writes a file.
    """
    gf = PB / f"games_{season}.json"
    if not gf.exists():
        return []
    gs = sorted(json.loads(gf.read_text()),
                key=lambda g: (str(g.get("Date")), str(g.get("GameId"))))
    gs = [g for g in gs if (PB / f"lineupopp_{g['GameId']}.json").exists()]
    real = set()
    for g in gs:
        gid = str(g.get("GameId"))
        if gid.startswith("E"):
            continue
        d = json.loads((PB / f"lineupopp_{gid}.json").read_text())
        real.add((str(d.get("date"))[:10].replace("-", ""),
                  f"{d.get('away_team_abbreviation')}@{d.get('home_team_abbreviation')}"))
    out = []
    for g in gs:
        gid = str(g.get("GameId"))
        if gid.startswith("E"):
            d = json.loads((PB / f"lineupopp_{gid}.json").read_text())
            key = (str(d.get("date"))[:10].replace("-", ""),
                   f"{d.get('away_team_abbreviation')}@{d.get('home_team_abbreviation')}")
            if key in real:
                continue
        out.append(g)
    return out


@lru_cache(maxsize=None)
def _canon_names(season):
    """{raw spelling: canonical spelling} for one season's lineup files.

    pbpstats carries the same player under two spellings when the accent is
    dropped ("Moussa Diabate" / "Moussa Diabate" with the acute), which splits
    his on/off in two and makes BOTH halves wrong -- the same defect class as
    the BBRef mojibake, arriving through a different source. Folding happens
    here, at the one place every option reads lineups through, because a fold
    applied per call site decays the next time a call site is added.

    Grouping is by the BOX TABLE's key, not by normalize(), because the feed
    also renames a player outright mid-season: 2024-25 carries both "Alex Sarr"
    and "Alexandre Sarr", both "Bub Carrington" and "Carlton Carrington", and
    both "Jakob Poeltl" and "Jakob Poltl". Those normalize differently, so a
    normalize()-keyed fold leaves each player split across two rows with two
    partial on/off splits and no duplicate-key warning to show it.

    The canonical spelling is the most frequent one, ties broken alphabetically
    so the choice is deterministic rather than filesystem-order dependent.
    """
    counts = defaultdict(lambda: defaultdict(int))
    for g in games(season):
        d = json.loads((PB / f"lineupopp_{g['GameId']}.json").read_text())
        for side in ("Home", "Away"):
            blk = d["stats"].get(side)
            for r in (blk.get("FullGame") if isinstance(blk, dict) else []) or []:
                for x in str(r.get("Name", "")).split(","):
                    x = x.strip()
                    if x:
                        counts[box_key(season, x)][x] += 1
    out = {}
    for variants in counts.values():
        best = sorted(variants.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for raw in variants:
            out[raw] = best
    return out


def lineup_rows(season, max_team_games=None, min_team_games=0):
    """Yield (game_id, team_abbrev, rows) per team-game, oldest first.

    max_team_games caps each team at its first N games, which is how a
    back-test replays "what did this metric say after N games?", and
    min_team_games skips that many first, so the two together select a WINDOW.
    A forward holdout needs both: fit on [0, k), score on [k, n). Every option
    reads its lineups through here so the cap cannot be applied in one place
    and forgotten in another.
    """
    seen = defaultdict(int)
    canon = _canon_names(season)
    for g in games(season):
        d = json.loads((PB / f"lineupopp_{g['GameId']}.json").read_text())
        for side in ("Home", "Away"):
            blk = d["stats"].get(side)
            rows = blk.get("FullGame") if isinstance(blk, dict) else None
            if not rows:
                continue
            ab = d.get(f"{side.lower()}_team_abbreviation")
            if max_team_games and seen[ab] >= max_team_games:
                continue
            n_before = seen[ab]
            seen[ab] += 1
            if n_before < min_team_games:
                continue
            for r in rows:
                parts = [x.strip() for x in str(r.get("Name", "")).split(",") if x.strip()]
                r["Name"] = ", ".join(canon.get(x, x) for x in parts)
            yield g["GameId"], ab, rows


@lru_cache(maxsize=None)
def onoff(season, max_team_games=None, min_team_games=0):
    """{player: (on, off, team, games)} of opponent outcomes, from complete lineups."""
    on = defaultdict(lambda: defaultdict(float))
    off = defaultdict(lambda: defaultdict(float))
    team, pg = {}, defaultdict(set)
    for gid, ab, rows in lineup_rows(season, max_team_games, min_team_games):
        roster = set()
        for r in rows:
            roster |= {x.strip() for x in str(r.get("Name", "")).split(",") if x.strip()}
        for p in roster:
            team[p] = ab
            pg[p].add(gid)
        for r in rows:
            here = {x.strip() for x in str(r.get("Name", "")).split(",") if x.strip()}
            for p in roster:
                t = on[p] if p in here else off[p]
                for k in OPP:
                    try:
                        t[k] += float(r.get(k) or 0)
                    except (TypeError, ValueError):
                        pass
    return {p: (dict(on[p]), dict(off[p]), team.get(p, "?"), len(pg[p])) for p in on}


def zones(d):
    """Opponent shot shares by zone, and the points-per-shot their diet implies."""
    shots = d.get("FG2A", 0) + d.get("FG3A", 0)
    if shots <= 0:
        return None
    z = {"rim": d.get("AtRimFGA", 0), "smid": d.get("ShortMidRangeFGA", 0),
         "lmid": d.get("LongMidRangeFGA", 0), "arc3": d.get("Arc3FGA", 0),
         "cor3": d.get("Corner3FGA", 0)}
    tot = sum(z.values()) or shots
    share = {k: v / tot for k, v in z.items()}
    return share, sum(share[k] * PPS[k] for k in share)


@lru_cache(maxsize=None)
def players(season):
    """Box + pbpstats counts + BBRef rates + role weight, one row per player.

    Memoised, so callers must not mutate the frame they get back.
    """
    from utils import fetch_bref_player_stats, _merge_provisional_box_scores
    ls = ROOT / "cache" / f"league_stats_{season.replace('-', '_')}.parquet"
    cols = ["PLAYER_ID", "PLAYER_NAME", "GP", "MIN", "STL", "BLK", "DREB", "OREB", "PF", "TOV"]
    raw = pd.read_parquet(ls) if ls.exists() else fetch_bref_player_stats(season)
    # Fold in the provisional rows, exactly as the site's own build does. Without
    # this the box half of every score stays behind the lineup half whenever the
    # lineups have been rebuilt from ESPN ahead of BBRef -- which is the same
    # asymmetry that put Walker Kessler top of the board off one of two games,
    # just pointing the other way. The merge is a no-op once the feed catches up.
    try:
        raw = _merge_provisional_box_scores(raw, season)
    except Exception:
        pass
    box = raw[cols].copy()
    box["key"] = box["PLAYER_NAME"].map(normalize)
    fp = PB / f"player_{season}.json"
    if fp.exists():
        p = pd.DataFrame(json.loads(fp.read_text()))
        if "Name" in p.columns:
            p["key"] = p["Name"].map(normalize)
            CNT = ["Blocks", "Steals", "DefRebounds", "Fouls", "Charge Fouls",
                   "BadPassSteals", "LostBallSteals"]
            for c in CNT + ["DefPoss", "DefFGReboundPct"]:
                p[c] = pd.to_numeric(p[c], errors="coerce") if c in p.columns else 0.0
            p[CNT] = p[CNT].fillna(0.0)
            p["DefFGReboundPct"] = p["DefFGReboundPct"].fillna(0.0)
            box = box.merge(p[["key"] + CNT + ["DefPoss", "DefFGReboundPct"]]
                            .drop_duplicates("key"), on="key", how="left")
    for tid, ren in (("advanced", {"DBPM": "dbpm", "DWS": "dws", "BLK%": "blk_pct",
                                   "STL%": "stl_pct", "DRB%": "drb_pct"}),
                     ("pbp_stats", {"On-Off": "on_off", "PF%": "pfs", "C%": "cs"})):
        f = DEF / f"{tid}_{season}.parquet"
        if not f.exists():
            continue
        a = pd.read_parquet(f)
        a.columns = [str(c) for c in a.columns]
        a = a.loc[:, ~a.columns.duplicated()]
        a["key"] = a["Player"].map(normalize)
        keep = ["key"] + [c for c in ren if c in a.columns]
        box = box.merge(a[keep].rename(columns=ren).drop_duplicates("key"), on="key", how="left")
    for c in ("dbpm", "dws", "blk_pct", "stl_pct", "drb_pct", "on_off", "pfs", "cs"):
        if c not in box.columns:
            box[c] = np.nan
        box[c] = pd.to_numeric(box[c], errors="coerce")
    box["big"] = ((box.pfs.fillna(0) + box.cs.fillna(0)) / 100).clip(0, 1)
    box["Season"] = season
    return box


# pbpstats and the box feed disagree on these beyond what NameIndex can bridge:
# it requires a matching surname and a prefix relation, and "Poeltl"/"Poltl" is a
# transliteration of the same umlaut while "Bub"/"Carlton" is a nickname with no
# letters in common. Everything else -- the "III" suffix on Butler, "Alex" for
# "Alexandre", the dropped "Jr." on Boston -- NameIndex already resolves.
DEF_ALIASES = {"jakob poltl": "Jakob Poeltl", "carlton carrington": "Bub Carrington"}


@lru_cache(maxsize=None)
def box_index(season):
    """Name-tolerant map from any spelling to the box table's own key.

    A bare normalize() key silently drops the players whose two feeds disagree,
    and because build() fills an unmatched row with ZEROS rather than dropping
    it, the miss surfaces as a plausible-looking score instead of an error: five
    such players held half the shipped proxy's bottom ten on 2024-25.
    """
    from utils import NameIndex
    box = players(season)
    return NameIndex({n: k for n, k in zip(box.PLAYER_NAME, box.key)})


def box_key(season, name):
    """The box table's key for a lineup spelling; the bare key if unmatched."""
    alias = DEF_ALIASES.get(normalize(name), name)
    return box_index(season).get(alias) or normalize(alias)


# ESPN's abbreviations differ from the NBA's on nine teams. Only the ones that
# differ are listed; anything absent passes through unchanged.
_ESPN_TO_NBA = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS",
                "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX"}
_ESPN_DIR = ROOT / "cache" / "espn_sched"


def espn_completed(season, today=None):
    """{(away, home, date): date} for every COMPLETED game of the season, per ESPN.

    This exists because pbpstats' own get-games list is INCOMPLETE and saying so
    costs nothing: on 2026-10-24 it listed 26 games while ESPN had 28 completed
    through the same date, missing HOU@SAS and LAC@LAL from 10/23 -- games it had
    not processed at all, so they could not be fetched by id either. Four teams
    were therefore short a played game, and the ingest reported "fetched 0, no
    errors" because it only ever compared itself to that list.

    THE DATE IS PART OF THE KEY, and that is the whole point. This was keyed on
    (away, home) alone, which silently cannot hold a repeat matchup: the second
    meeting overwrote the first and the game ceased to exist for everything
    downstream. By 2026-10-31 twelve pairs had met twice, among them CHI at BOS
    on the 26th and the 30th and ORL at TOR on the 28th and the 30th, so
    Boston, Chicago, Orlando and Toronto were each counted one game short.
    check_board_gp then reported those four teams as having played MORE games
    than the schedule -- the board was right and the "truth" it was checked
    against was missing games. A dict keyed by matchup is a dict that assumes
    teams meet once a season.

    Each past date is cached, since a finished date never changes; the last two
    days are always refetched in case a late game went final.
    """
    import datetime as _d, urllib.request
    _ESPN_DIR.mkdir(parents=True, exist_ok=True)
    today = today or _d.date.today()
    start = _d.date(int(season[:4]), 10, 1)
    out = {}
    day = start
    while day <= today:
        k = day.strftime("%Y%m%d")
        fp = _ESPN_DIR / f"{k}.json"
        fresh = (today - day).days <= 1
        if fp.exists() and not fresh:
            try:
                for a, h, d in json.loads(fp.read_text()):
                    out[(a, h, d)] = d
                day += _d.timedelta(days=1)
                continue
            except Exception:
                pass
        try:
            ev = json.loads(urllib.request.urlopen(urllib.request.Request(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
                f"scoreboard?dates={k}",
                headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read()).get("events", [])
        except Exception:
            day += _d.timedelta(days=1)
            continue
        rows = []
        for e in ev:
            c = e["competitions"][0]
            if not c["status"]["type"].get("completed"):
                continue
            # ESPN's scoreboard mixes in PRESEASON (season.type 1), which is how
            # a first pass of this check "found" 71 missing games and a team
            # called LON. Regular season only.
            if (e.get("season") or {}).get("type") != 2:
                continue
            ab = {x["homeAway"]: x["team"]["abbreviation"] for x in c["competitors"]}
            a = _ESPN_TO_NBA.get(ab["away"], ab["away"])
            h = _ESPN_TO_NBA.get(ab["home"], ab["home"])
            rows.append([a, h, k])
            out[(a, h, k)] = k
        if not fresh:
            fp.write_text(json.dumps(rows))
        day += _d.timedelta(days=1)
    return out


def feed_gap(season, today=None):
    """What the lineup feed is MISSING against ESPN's completed games.

    Returns (per_team_espn, per_team_feed, missing_matchups). Absence in the
    lineup feed must SHOW, not be read as "that team played fewer games".
    """
    from collections import Counter
    played = espn_completed(season, today)
    have = set()
    feed = Counter()
    for _gid, ab, _rows in lineup_rows(season):
        feed[ab] += 1
    for g in games(season):
        d = json.loads((PB / f"lineupopp_{g['GameId']}.json").read_text())
        # The DATE is part of the identity here too: without it the second
        # meeting of a pair looks like one the feed already has, and a game
        # that is genuinely missing is never rebuilt. The lineup files carry
        # their own date; ESPN's keys are YYYYMMDD, so normalise.
        have.add((d.get("away_team_abbreviation"), d.get("home_team_abbreviation"),
                  str(d.get("date") or "").replace("-", "")))
    espn = Counter()
    for (a, h, _d) in played:
        espn[a] += 1
        espn[h] += 1
    missing = sorted((f"{a}@{h}", d) for (a, h, _k), d in played.items()
                     if (a, h, d) not in have)
    return dict(espn), dict(feed), missing


@lru_cache(maxsize=None)
def career_possessions(season, poss_per_min=2.08):
    """{box key: estimated career defensive possessions BEFORE this season}.

    Option 4 applies DARKO's w = career/(career+6000), which is meant to read a
    thirteen-year rim protector off deterrence and a rookie off his box line.
    Its original stand-in for career possessions was `dp_on * 41 * 3`, which at
    three games is a function of CURRENT-SEASON MINUTES: a high-minute rookie
    scored a higher deterrence weight than a low-minute veteran, the exact
    inversion of the intent. This uses the 100-odd cached league_stats seasons
    instead, back to 1973-74.

    Regular season only, and minutes converted at roughly 2.08 defensive
    possessions per minute (about 100 possessions per 48-minute team game).
    Returns {} rather than raising if the caches are not there.
    """
    import re
    acc = {}
    cur = int(season[:4])
    for f in sorted((ROOT / "cache").glob("league_stats_*.parquet")):
        m = re.fullmatch(r"league_stats_(\d{4})_(\d{2})", f.stem)
        if not m or int(m.group(1)) >= cur:      # strictly prior seasons
            continue
        try:
            d = pd.read_parquet(f, columns=["PLAYER_NAME", "GP", "MIN"])
        except Exception:
            continue
        gp = pd.to_numeric(d["GP"], errors="coerce").fillna(0)
        mn = pd.to_numeric(d["MIN"], errors="coerce").fillna(0)
        for n, v in zip(d["PLAYER_NAME"], gp * mn):
            if v and v == v:
                k = normalize(str(n))
                acc[k] = acc.get(k, 0.0) + float(v)
    if not acc:
        return {}
    from utils import NameIndex
    ni = NameIndex({n: n for n in acc})
    out = {}
    for pk in players(season)["key"]:
        hit = ni.get(pk)
        out[pk] = acc.get(hit, 0.0) * poss_per_min if hit else 0.0
    return out


def z(s):
    s = pd.Series(s, dtype=float)
    sd = s.std()
    return (s - s.mean()) / (sd if sd and sd == sd else 1.0)
