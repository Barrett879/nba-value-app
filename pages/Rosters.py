"""Rosters: the real 2026-27 books, team by team.

Every roster spot on one board -- guaranteed deals, two-ways and the team's
own unsigned free agents -- each with its contract line, Barrett Score and
model value, over a cap picture that reads against the tax line and both
aprons. The roster spine is data/master_roster.csv (hand-verified from
Spotrac, all 30 teams), NOT the projected board: cache/team_pages.json
carries 12 model-projected signings that nobody has actually signed, and
payroll here has to be what a team really owes.

All 30 teams ship in one payload and switching teams is instant: the page
renders a srcdoc component (templates/rosters.html), same pattern as the
Trade Machine, so there is no server round trip per team.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import datetime as _dt
import json
import re

import streamlit as st
import streamlit.components.v1 as components

from utils import (BLEND_MIN_MINUTES, NameIndex, NBA_DIVISIONS, SEASONS, TEAM_CONFERENCE,
                   TEAM_HEX, TEAM_HEX_2,
                   _headshot_id_map, fetch_dlebron, fetch_injuries, injury_is_out,
                   render_footer, render_nav,
                   render_page_chrome, script_json, _bootstrap_warm,
                   unavailable_players)

st.set_page_config(page_title="Rosters", page_icon="static/favicon.svg", layout="wide")
render_page_chrome()
_bootstrap_warm()
render_nav("Rosters")

st.title("Rosters")
st.caption(
    "The real 2026-27 books, team by team. Every roster spot with its contract, "
    "Barrett Score and model value. Payroll is what a team actually owes, read "
    "against the cap, the tax line and both aprons."
)

_ROOT = Path(__file__).parent.parent
# Used to expire lapsed trade exceptions. A frozen literal here silently
# resurrects TPEs that have already expired.
_TODAY = _dt.date.today().isoformat()
# Which season a card shows is the SITE's decision, not this page's.
# utils.BLEND_MIN_MINUTES is the one knob that makes it (Barrett, 2026-10-29:
# the current season everywhere, with only a player who has not played at all
# falling back), and the hub and the team pages both read it.
#
# A local 500 here meant this page never got that decision. On 2026-10-30 the
# season's minutes leader was at 201, so NOT ONE of 396 scored players cleared
# the floor: every card on every depth chart showed a 2025-26 number tagged
# (25-26) while the hub showed 2026-27 for the same player. Luka read 44.0
# against the hub's 50.9.
#
# The floor was also only ever applied to the CURRENT season -- nothing floored
# the season it fell back TO. Walker Kessler is the player the original comment
# cited as the reason for the floor, 5 games and 154 minutes in 2025-26, and
# the rule rejected his 152 current-season minutes and then printed that 154
# minute sample instead. Same sample size, opposite verdict.
#
# The argument for a higher threshold is real and is written out in full at
# utils.build_ranked_blended. It belongs in that one place, where changing it
# moves every surface together.
MIN_MINUTES = BLEND_MIN_MINUTES
# feeds spell a few teams the old way
_ABBR_FIX = {"PHO": "PHX", "CHO": "CHA", "BRK": "BKN", "NOH": "NOP"}
# every player lands in exactly one depth-chart column; master_roster carries a
# handful of bare "G" and "F" listings on two-way and free-agent rows
_SLOT = {"PG": "PG", "SG": "SG", "SF": "SF", "PF": "PF", "C": "C", "G": "PG", "F": "SF"}

# conference / division, for grouping the team strip. Shared with the Trade
# Machine's team menu, so it lives in utils rather than here.
_DIVS = NBA_DIVISIONS


def _read_csv(path: Path) -> list:
    """CSV rows with the leading # provenance comments stripped."""
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


@st.cache_data(show_spinner=False)
def _payload() -> str:
    cfg = {k: v for k, v in json.loads(
        (_ROOT / "data" / "cba_config_2026_27.json").read_text()).items()
        if not k.startswith("_")}
    master = _read_csv(_ROOT / "data" / "master_roster.csv")
    asof = next((r["asof"] for r in master if r.get("asof")), "")

    # model numbers (market value, value delta, Barrett Score) come from the
    # projected board, joined by NameIndex: the feeds spell "C.J. McCollum"
    # and "Nic Claxton" two different ways and an exact key silently drops them
    board = json.loads((_ROOT / "cache" / "team_pages.json").read_text())
    model = NameIndex()
    for _t in board["teams"].values():
        for _p in _t["players"]:
            model.add(_p["n"], _p)

    heads = NameIndex(_headshot_id_map())

    # Barrett Scores: the most recent season in which the player cleared
    # MIN_MINUTES, which is the site-wide knob (see its definition above). At
    # the knob's current value of 0 that means the current season for anyone
    # who has played at all, and the last complete season for anyone who has
    # not, tagged so the card says which -- the same rule, from the same
    # constant, as utils.build_ranked_blended and so as the hub and the team
    # pages. Raise the knob and every surface reaches back together.
    # DERIVED, never literal. These were written as "2025-26" and "2024-25"
    # when 2025-26 was the season being scored. After the flip to 2026-27 the
    # literals still parsed and still loaded, so nothing failed: the page just
    # presented last season's score UNTAGGED, as though it were this season's.
    # That is the rollover's signature failure, a wrong number with no error.
    _CUR   = SEASONS[0]                              # the season being scored
    _PRIOR = SEASONS[1]                              # the last complete season

    def _tag(season):
        """2025-26 -> 25-26, the label the card prints."""
        return season[2:]

    scores, prior_scores, short = NameIndex(), NameIndex(), NameIndex()
    extra_scores: dict = {}                          # season -> NameIndex, for overrides
    try:
        from utils import build_raw
        for _r in build_raw(_PRIOR).itertuples():
            prior_scores.add(_r.Player, (round(float(_r.barrett_score), 1), _tag(_PRIOR),
                                         float(_r.total_min)))
        for _r in build_raw(_CUR).itertuples():
            v = (round(float(_r.barrett_score), 1), None, float(_r.total_min))
            if float(_r.total_min) >= MIN_MINUTES:
                scores.add(_r.Player, v)
            else:
                short.add(_r.Player, v)              # keep as the last resort
    except Exception:                                # never block the page
        pass

    # Hand-set depth-chart order (data/depth_order.csv). Keyed by team as well
    # as name so a pin can never leak onto a same-named player elsewhere. The
    # rank rides on the player payload; the template sorts pinned players above
    # the score-ranked rest.
    # Current injury designations (utils.fetch_injuries, ESPN, 4h serve-then-
    # refresh). Resolved through NameIndex, never a bare normalize() key: ESPN
    # prints "Jimmy Butler III" where the roster says "Jimmy Butler", and an
    # exact key reports a torn ACL as healthy.
    try:
        _inj_idx = NameIndex(fetch_injuries())
    except Exception:                                # never block the page
        _inj_idx = NameIndex()

    # Hand-set position overrides (data/position_overrides.csv). EDITORIAL, so
    # they live outside master_roster, whose pos is scraped roster data a
    # verification round would re-write. The FIRST position decides the depth
    # column, so most of these rows exist to move a player between columns.
    #
    # A row may carry alt_pos + alt_when_out, which swaps the position only
    # while the named player is listed OUT. Keyed to the injury feed rather
    # than a date so it reverts by itself the day he is back: Draymond Green is
    # a centre only while Porzingis is out, Derrick Jones Jr. a small forward
    # only while Ingram is out.
    _posov = NameIndex()
    for _row in _read_csv(_ROOT / "data" / "position_overrides.csv"):
        _nm = (_row.get("player") or "").strip()
        _pp = (_row.get("pos") or "").strip()
        if not _nm or not _pp:
            continue
        _alt = (_row.get("alt_pos") or "").strip()
        _dep = (_row.get("alt_when_out") or "").strip()
        _use = _pp
        if _alt and _dep:
            _di = _inj_idx.get(_dep)
            if _di and injury_is_out(_di.get("status", "")):
                _use = _alt
        _posov.add(_nm, _use)

    _pin = {}
    for _row in _read_csv(_ROOT / "data" / "depth_order.csv"):
        _tm, _nm = (_row.get("team") or "").strip(), (_row.get("player") or "").strip()
        try:
            _rk = int((_row.get("rank") or "").strip())
        except ValueError:
            continue
        if _tm and _nm:
            _pin.setdefault(_tm, NameIndex()).add(_nm, (_rk, (_row.get("note") or "").strip()))

    # Hand-picked season overrides (data/score_season_override.csv). The minutes
    # floor cannot catch a player who kept a starter's MINUTES across a fraction
    # of the SEASON -- Keegan Murray's 23 games at ~34 a night clear 500 easily,
    # and the 12.0 that produced ranked him behind a bench big on his own chart.
    # An override forces the season; the card still labels which year it is.
    _season_ovr = NameIndex()
    _ovr_seasons: set = set()          # NameIndex has no values(); collect while reading
    for _row in _read_csv(_ROOT / "data" / "score_season_override.csv"):
        _nm, _sn = (_row.get("player") or "").strip(), (_row.get("season") or "").strip()
        if _nm and _sn:
            _season_ovr.add(_nm, _sn)
            _ovr_seasons.add(_sn)

    # A row may pin a season that is neither the scored nor the last complete
    # one, and after a rollover it usually does: Keegan Murray's row points at
    # 2024-25, which stopped being loaded the moment 2026-27 became SEASONS[0].
    # Load those on demand so an override never goes quietly dead.
    try:
        from utils import build_raw as _br
        for _sn in _ovr_seasons - {_CUR, _PRIOR}:
            _idx = NameIndex()
            for _r in _br(_sn).itertuples():
                _idx.add(_r.Player, (round(float(_r.barrett_score), 1), _tag(_sn),
                                     float(_r.total_min)))
            extra_scores[_sn] = _idx
    except Exception:
        pass

    def score_of(name):
        """(score, season tag) under the minutes floor, or the pinned season."""
        want = _season_ovr.get(name)
        if want:
            # 2025-26 lives in `scores` or `short` depending on the floor; the
            # tag comes from the source, so an override to the current season
            # stays untagged exactly like an ordinary current score.
            _srcs = ((prior_scores,) if want == _PRIOR
                     else (scores, short) if want == _CUR
                     else ((extra_scores[want],) if want in extra_scores else ()))
            for src in _srcs:
                hit = src.get(name)
                if hit:
                    return hit[0], hit[1]
        for src in (scores, prior_scores, short):
            hit = src.get(name)
            if hit:
                return hit[0], hit[1]
        return None, None

    # Market value has to be priced off the SAME score the row displays.
    #
    # cache/team_pages.json prices every player off the current season with no
    # minutes floor, while score_of() honours the 500-minute floor and reaches
    # back a season when it has to. Those two disagreeing put a 2024-25 score
    # next to a value derived from a 2025-26 sprained-ankle sample and printed a
    # verdict on it: Trae Young (384 minutes in 2025-26) rendered as
    # "35.4 BS (24-25) · $49.5M · $35.4M over value" when the honest gap is
    # about $5M. 21 players on the 2026-27 books were affected.
    #
    # So rather than read the precomputed value, map the FLOORED score back
    # through the same score -> projected-salary curve the board is built from.
    # For a player whose score already comes from the current season this
    # reproduces the board value exactly; for everyone else it prices the season
    # actually being shown.
    # One curve per season, because the row's value has to be priced off the
    # SAME season its score came from. A single hardcoded curve was right only
    # while every score came from one season; after the rollover most scores
    # come from _PRIOR while a few will cross the floor in _CUR, and pricing
    # those against the wrong year is the mismatch described just above.
    _curves: dict = {}

    def _curve_for(season):
        if season in _curves:
            return _curves[season]
        curve = None
        try:
            from utils import build_ranked_projected
            _proj = build_ranked_projected(season)[["barrett_score", "projected_salary"]]
            _proj = _proj.dropna().sort_values("barrett_score")
            if len(_proj):
                curve = (_proj["barrett_score"].to_numpy(dtype=float),
                         _proj["projected_salary"].to_numpy(dtype=float) / 1e6)
        except Exception:
            curve = None
        _curves[season] = curve
        return curve

    def value_of(score, tag=None):
        """Projected market salary ($M) for a Barrett Score, priced off the
        season that score came from. None when that curve is unavailable.

        `tag` is the card's season label ("25-26"), or None for the scored
        season, which is exactly what score_of returns alongside the score.
        """
        if score is None:
            return None
        season = _CUR if not tag else next(
            (s for s in SEASONS if _tag(s) == tag), _CUR)
        curve = _curve_for(season)
        if curve is None or not len(curve[0]):
            # An incomplete season may have no usable curve yet; fall back to
            # the last complete one rather than printing no value at all.
            curve = _curve_for(_PRIOR)
        if curve is None or not len(curve[0]):
            return None
        import numpy as _np
        xs, ys = curve
        return round(float(_np.interp(float(score), xs, ys)), 1)

    # last season's per-game line + D-LEBRON
    box = NameIndex()
    try:
        import pandas as pd
        _ls = pd.read_parquet(
            _ROOT / "cache" / f"league_stats_{_PRIOR.replace('-', '_')}.parquet")
        _dl = fetch_dlebron(_PRIOR)
        for _r in _ls.itertuples():
            box.add(_r.PLAYER_NAME, {
                "age": int(_r.AGE) if getattr(_r, "AGE", None) == getattr(_r, "AGE", None)
                       and getattr(_r, "AGE", None) is not None else None,
                "gp": int(_r.GP), "mpg": round(float(_r.MIN), 1),
                "ppg": round(float(_r.PTS), 1), "apg": round(float(_r.AST), 1),
                "rpg": round(float(_r.REB), 1),
                "dl": (round(float(_dl[_r.PLAYER_ID]), 1)
                       if _dl.get(_r.PLAYER_ID) else None)})
    except Exception:
        pass

    # free-agent market values (the master notes carry an est value too, but
    # the pool is the same number the rest of the site quotes)
    fa_val = NameIndex()
    _fp = _ROOT / "cache" / "fa_pool_v1.json"
    if _fp.exists():
        for f in json.loads(_fp.read_text())["fas"]:
            if f.get("n"):
                fa_val.add(f["n"], f)
    # the board carries a few notable unsigned players the pool file misses
    extra_fas = {}
    for _ab, _t in board["teams"].items():
        for _u in _t.get("unsigned", []):
            if _u.get("n"):
                fa_val.add(_u["n"], {"barrett": _u.get("barrett")})
                extra_fas.setdefault(_ab, []).append(_u["n"])

    hard_caps = {}
    for r in _read_csv(_ROOT / "data" / "hard_caps_2026_27.csv"):
        if r.get("team") and (r.get("cap") or "").strip() in ("apron1", "apron2"):
            hard_caps[r["team"].strip()] = {
                "cap": r["cap"].strip(), "why": (r.get("trigger") or "").strip()}

    # Dead money: salary still owed to players who are no longer on the roster.
    # It is not a roster spot but it IS on the cap, so committed salary has to
    # carry it -- Memphis owes Caldwell-Pope $17.7M in 2026-27 not to play, and
    # a payroll that leaves it out reads $17.7M cheaper than the team really is.
    dead = {}
    for r in _read_csv(_ROOT / "data" / "dead_money_2026_27.csv"):
        if r.get("team") and r.get("amount_M"):
            dead.setdefault(r["team"].strip(), []).append({
                "n": (r.get("player") or "").strip(),
                "amt": round(float(r["amount_M"]), 2),
                "why": (r.get("status") or "").strip(),
                "note": (r.get("notes") or "").strip()})
    for v in dead.values():
        v.sort(key=lambda x: -x["amt"])

    # Last season's finished record, for the leaderboard beside the team picker.
    # mov (average point differential) is carried because the 2025-26 season ended
    # with New York and the Lakers both 53-29: a leaderboard cut at six has to
    # break that tie on something, and list order is not a reason.
    records = {}
    _rec_path = _ROOT / "data" / f"team_records_{_PRIOR.replace('-', '_')}.csv"
    for r in (_read_csv(_rec_path) if _rec_path.exists() else []):
        if r.get("team") and r.get("wins"):
            records[r["team"].strip()] = {
                "w": int(r["wins"]), "l": int(r["losses"]),
                "mov": float(r["mov"]), "conf": (r.get("conf") or "").strip()}

    # Years still owed on each contract, and what the final year is. Keyed by
    # (team, name) rather than name alone because the roster carries a handful
    # of same-surname pairs and a bare name key is how players quietly lose data
    # here. Absent is normal and means "no contract on file" -- 6 two-ways and 5
    # players Spotrac still lists as pending, which render no chip at all rather
    # than a guess.
    years: dict[str, NameIndex] = {}
    for r in _read_csv(_ROOT / "data" / "contract_years_2026_27.csv"):
        if r.get("player") and r.get("years_left"):
            years.setdefault(r["team"].strip(), NameIndex()).add(
                r["player"], {
                    "n": int(r["years_left"]),
                    "g": int(r.get("guaranteed_years") or 0),
                    "kind": (r.get("last_year_type") or "guaranteed").strip(),
                    "seasons": (r.get("seasons") or "").strip(),
                })

    tpes = {}
    for r in _read_csv(_ROOT / "data" / "trade_exceptions_2026_27.csv"):
        exp = (r.get("expires") or "").strip()
        if exp and exp < _TODAY:          # already lapsed, not capital any more
            continue
        if r.get("team") and r.get("amount_M"):
            tpes.setdefault(r["team"].strip(), []).append({
                "amt": round(float(r["amount_M"]), 2),
                "player": (r.get("player") or "").strip(),
                "expires": exp})
    for v in tpes.values():
        v.sort(key=lambda x: -x["amt"])

    # draft picks the team controls, split by round
    picks = {}
    _pl = _ROOT / "cache" / "pick_ledger.json"
    if _pl.exists():
        led = json.loads(_pl.read_text())["teams"]
        for abbr, t in led.items():
            rows = []
            for pk in t.get("controls", []):
                holder = (pk.get("swap_holder") or "").strip()
                with_ = (pk.get("swap_with") or "").strip()
                swap = ""
                if with_ or holder:
                    swap = ("in" if holder == abbr
                            else "out" if holder else "unknown")
                rows.append({
                    "year": pk["year"], "origin": pk["origin"],
                    "round": pk.get("round", 1),
                    "own": pk["origin"] == abbr,
                    "protection": (pk.get("protection") or "").strip(),
                    "swap": swap, "swap_with": with_, "swap_holder": holder,
                    "notes": (pk.get("notes") or "").strip()})
            rows.sort(key=lambda p: (p["round"], p["year"], p["origin"]))
            picks[abbr] = rows

    # option decisions already made this offseason, keyed by player
    opts = NameIndex()
    for r in _read_csv(_ROOT / "data" / "option_decisions_2026.csv"):
        if r.get("player"):
            opts.add(r["player"], {"d": (r.get("decision") or "").strip(),
                                   "note": (r.get("note") or "").strip()})

    _conf = TEAM_CONFERENCE                      # derived once, in utils
    _div = {ab: d for ds in _DIVS.values() for d, abs_ in ds.items() for ab in abs_}

    # Retirements, joined through NameIndex because the file's spelling and the
    # roster's need not agree.
    _GONE = NameIndex(unavailable_players())

    by_team = {}
    for r in master:
        ab = (r.get("team") or "").strip()
        if not ab:
            continue
        by_team.setdefault(ab, []).append(r)

    teams = {}
    for abbr, rows in by_team.items():
        players = []
        for r in rows:
            name = (r.get("player") or "").strip()
            kind = (r.get("kind") or "standard").strip()
            sal = float(r.get("salary_M") or 0)
            note = (r.get("notes") or "").strip()
            # A retired player holds no roster spot and is not a free agent the
            # rights holder could re-sign, so he is on none of this team's books.
            # The row stays in master_roster because it carries the dead money
            # and the announcement; it just does not belong on a team page.
            if _GONE.get(name):
                continue
            m = model.get(name) or {}
            fv = fa_val.get(name) or {}
            pid = heads.get(name)
            # score first: the value is priced off it, so it has to be resolved
            # before the value is chosen
            bar, bar_yr = score_of(name)         # minutes floor wins outright
            if bar is None:
                bar, bar_yr = m.get("barrett"), m.get("bs_yr")
            if bar is None:
                bar, bar_yr = fv.get("barrett"), fv.get("bs_yr")
            # price the score that is actually on the row; fall back to the
            # board value only when the curve is unavailable
            value = value_of(bar, bar_yr)
            if value is None:
                value = m.get("value")
            if kind == "free_agent":
                value = fv.get("value")
                if value is None:                      # "est value $12.3M"
                    _mm = re.search(r"\$([0-9.]+)M", note)
                    value = float(_mm.group(1)) if _mm else None
                sal = 0.0
            o = opts.get(name) or {}
            pos = (_posov.get(name) or (r.get("pos") or "")).strip()
            _pn = (_pin.get(abbr) or NameIndex()).get(name)
            _ij = _inj_idx.get(name)
            players.append({
                "n": name, "kind": kind,
                "pos": pos,
                "slot": _SLOT.get(pos.split("/")[0].upper(), ""),
                "dord": _pn[0] if _pn else None,
                "dnote": (_pn[1] or None) if _pn else None,
                # inj: the short badge. iout: whether it takes him OUT of the
                # rotation, which is the only thing that reorders a column.
                "inj": _ij["status"] if _ij else None,
                "iret": (_ij.get("ret") or None) if _ij else None,
                "icmt": (_ij.get("comment") or None) if _ij else None,
                "iout": bool(_ij and injury_is_out(_ij["status"])),
                "salary": round(sal, 2),
                "note": note,
                "value": round(value, 1) if value is not None else None,
                "vd": (round(sal - value, 1)
                       if (value is not None and kind == "standard") else None),
                "barrett": bar, "bs_yr": bar_yr,
                **(box.get(name) or {}),
                "opt": o.get("d") or None,
                "opt_note": o.get("note") or None,
                "yrs": (years.get(abbr) or NameIndex()).get(name),
                "headshot": (f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"
                             if pid else None),
            })
        # a few notable unsigned players ride only on the board's "unsigned"
        # list, not in the master free-agent rows; add the ones this roster does
        # not already carry (NameIndex dedupe: the board says "CJ McCollum"
        # where the roster says "C.J. McCollum", and he is signed, not an FA)
        have = NameIndex()
        for p in players:
            have.add(p["n"], True)
        for nm in extra_fas.get(abbr, []):
            if have.get(nm) or _GONE.get(nm):
                continue
            fv = fa_val.get(nm) or {}
            pid = heads.get(nm)
            players.append({
                "n": nm, "kind": "free_agent", "pos": "", "slot": "",
                "salary": 0.0, "note": "unsigned free agent",
                "value": round(fv["value"], 1) if fv.get("value") is not None else None,
                "vd": None, "barrett": fv.get("barrett"), "bs_yr": fv.get("bs_yr"),
                **(box.get(nm) or {}),
                "opt": None, "opt_note": None,
                "headshot": (f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"
                             if pid else None)})
            have.add(nm, True)

        std = [p for p in players if p["kind"] == "standard"]
        tw = [p for p in players if p["kind"] == "two_way"]
        fas = [p for p in players if p["kind"] == "free_agent"]
        std.sort(key=lambda p: -p["salary"])
        tw.sort(key=lambda p: -(p["barrett"] or 0))
        fas.sort(key=lambda p: -(p["value"] or 0))
        dead_rows = dead.get(abbr, [])
        dead_tot = round(sum(d["amt"] for d in dead_rows), 1)
        pay_signed = round(sum(p["salary"] for p in std), 1)
        payroll = round(pay_signed + dead_tot, 1)
        # Surplus compares like with like: a player the model has no 2026-27
        # value for (he missed all of last season, so there is nothing to price
        # off) is left out of BOTH sides. Counting his salary against a zero
        # value would read as a huge overpay -- Haliburton alone would have put
        # Indiana $49M underwater.
        priced = [p for p in std if p["value"] is not None]
        val_tot = round(sum(p["value"] for p in priced), 1)
        pay_priced = round(sum(p["salary"] for p in priced), 1)
        unpriced = [p for p in std if p["value"] is None]
        hc = hard_caps.get(abbr)
        teams[abbr] = {
            "abbr": abbr,
            "name": board["teams"].get(abbr, {}).get("name", abbr),
            "conf": _conf.get(abbr, ""), "div": _div.get(abbr, ""),
            "players": std + tw + fas,
            "payroll": payroll,
            "pay_signed": pay_signed,
            "dead": {"total": dead_tot, "rows": dead_rows},
            "value_total": val_tot,
            "pay_priced": pay_priced,
            "surplus": round(val_tot - pay_priced, 1),
            "unpriced": {"n": len(unpriced),
                         "salary": round(sum(p["salary"] for p in unpriced), 1)},
            "counts": {"std": len(std), "tw": len(tw), "fa": len(fas)},
            "hard_cap": (cfg["apron1"] if hc["cap"] == "apron1" else cfg["apron2"]) if hc else None,
            "hard_cap_conflict": bool(hc and payroll > (
                cfg["apron1"] if hc["cap"] == "apron1" else cfg["apron2"])),
            "hard_cap_label": ({"apron1": "1st apron", "apron2": "2nd apron"}[hc["cap"]]
                               if hc else None),
            "hard_cap_why": hc["why"] if hc else None,
            "tpes": tpes.get(abbr, []),
            "picks": picks.get(abbr, []),
        }

    from nba_api.stats.static import teams as _nbat
    logos = {t["abbreviation"]: f"https://cdn.nba.com/logos/nba/{t['id']}/global/L/logo.svg"
             for t in _nbat.get_teams()}

    return json.dumps({
        "season": _CUR, "asof": asof, "cfg": cfg, "logos": logos,
        "colors": dict(TEAM_HEX), "colors2": dict(TEAM_HEX_2),
        "divs": _DIVS, "teams": teams,
        "records": records, "records_season": _PRIOR,
        "abbrs": sorted(teams),
    }, separators=(",", ":"))


def _payload_with_request_state() -> str:
    """Splice the per-request team (from ?team=) into the cached payload.

    The value is checked against the 30 real abbreviations before it is spliced:
    anything else becomes None. A query param that reaches a <script> block
    unvalidated is an XSS hole, and script_json() below is the second lock.
    """
    want = (st.query_params.get("team", "") or "").upper()
    want = _ABBR_FIX.get(want, want)
    cached = _payload()
    if want and f'"{want}":' not in cached:      # not one of the 30 teams
        want = ""
    return cached[:-1] + ',"initial":' + json.dumps(want or None) + "}"


_DARK = """--panel:#15171d;--card:#1b1f28;--line:#262a33;--track:#242833;
--fg1:#e6e9f2;--fg2:#c3c8d4;--fg3:#a7adbb;--fg4:#8a8f9c;--fg5:#6b7079;
--teal:#16d4c1;--good:#2ecc71;--bad:#e74c3c;--amberc:#f0b35b;--blue:#4c8dff;
--tintg:rgba(22,212,193,.08);--tintb:rgba(231,76,60,.10);--tintz:rgba(255,255,255,.02);"""
_LIGHT = """--panel:#ffffff;--card:#f7f8fa;--line:#e3e6eb;--track:#eceef2;
--fg1:#16233f;--fg2:#3a4150;--fg3:#5b6472;--fg4:#6b7280;--fg5:#9aa0ac;
--teal:#0fae9d;--good:#16a34a;--bad:#e0483a;--amberc:#b97f24;--blue:#2563eb;
--tintg:rgba(15,174,157,.08);--tintb:rgba(224,72,58,.08);--tintz:rgba(22,35,63,.025);"""

# BOTH palettes ship in one byte-stable srcdoc and the component detects the
# parent's theme at runtime. Injecting only the active palette would change the
# html on every theme toggle, which forces Streamlit to remount the iframe.
_html = ((_ROOT / "templates" / "rosters.html").read_text()
         .replace("__THEME__", _DARK)
         .replace("__THEME_LIGHT__", _LIGHT)
         .replace("__DATA__", script_json(_payload_with_request_state())))
components.html(_html, height=760, scrolling=False)

# Server-rendered team index. The board above lives in a component iframe, which
# crawlers do not read and a no-JS visitor never sees, so the same 30 teams also
# ship as plain crawlable HTML with their real payrolls.
_wall = json.loads(_payload())
_rows = sorted(_wall["teams"].values(), key=lambda t: -t["payroll"])
_cards = "".join(
    f'<a class="rw-card" href="/Rosters?team={t["abbr"]}" target="_top" '
    f'style="--tc:{TEAM_HEX.get(t["abbr"], "#888")}">'
    f'<span class="rw-name">{t["name"]}</span>'
    f'<span class="rw-meta">&#36;{t["payroll"]:.1f}M &middot; '
    f'{t["counts"]["std"]} signed, {t["counts"]["tw"]} two-way</span></a>'
    for t in _rows)
st.markdown(
    "<style>"
    ".rw-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));"
    "gap:.5rem;margin:.4rem 0 1.2rem}"
    ".rw-card{display:block;background:var(--panel-solid);border:1px solid var(--panel-line);"
    "border-left:4px solid var(--tc);border-radius:10px;padding:.5rem .7rem;"
    "text-decoration:none!important}"
    ".rw-card:hover{border-color:var(--tc)}"
    ".rw-name{display:block;font-weight:700;color:var(--fg-1);font-size:.88rem}"
    ".rw-meta{display:block;color:var(--fg-4);font-size:.75rem;font-variant-numeric:tabular-nums}"
    "</style>"
    f'<div class="rw-grid">{_cards}</div>',
    unsafe_allow_html=True)

st.caption(
    "Rosters are the verified 2026-27 books as of the date shown, not a "
    "projection: model-projected signings are excluded, so payroll is only "
    "what a team has actually committed. Two-way deals do not count against "
    f"the cap. Model value reads {SEASONS[0]} production against {SEASONS[0]} money. This "
    "early in a season that is a small sample, and a player who has not played "
    "yet carries his last complete season instead, marked with the year; a "
    "player with neither carries no value and is left out of the surplus on "
    f"both sides. Committed salary is signed {SEASONS[0]} contracts plus "
    "dead money still owed to waived and bought-out players, which counts "
    "against the cap the same way a roster player does. It does not include "
    "cap holds for the team's own free agents or empty-roster-spot charges."
)

render_footer()
