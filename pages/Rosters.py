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
import json
import re

import streamlit as st
import streamlit.components.v1 as components

from utils import (NameIndex, TEAM_HEX, _headshot_id_map, fetch_dlebron,
                   render_footer, render_nav, render_page_chrome, script_json,
                   _bootstrap_warm)

st.set_page_config(page_title="Rosters", page_icon="static/favicon.svg", layout="wide")
render_page_chrome()
_bootstrap_warm()
render_nav("Rosters")

st.title("Rosters")
st.caption(
    "The real 2026-27 books, team by team. Every roster spot with its contract, "
    "Barrett Score and model value: guaranteed deals, two-ways, and the team's "
    "own unsigned free agents. Payroll is what a team actually owes, read "
    "against the cap, the tax line and both aprons."
)

_ROOT = Path(__file__).parent.parent
_TODAY = "2026-07-28"
# feeds spell a few teams the old way
_ABBR_FIX = {"PHO": "PHX", "CHO": "CHA", "BRK": "BKN", "NOH": "NOP"}
# every player lands in exactly one depth-chart column; master_roster carries a
# handful of bare "G" and "F" listings on two-way and free-agent rows
_SLOT = {"PG": "PG", "SG": "SG", "SF": "SF", "PF": "PF", "C": "C", "G": "PG", "F": "SF"}

# conference / division, for grouping the team strip
_DIVS = {
    "East": {
        "Atlantic": ["BOS", "BKN", "NYK", "PHI", "TOR"],
        "Central": ["CHI", "CLE", "DET", "IND", "MIL"],
        "Southeast": ["ATL", "CHA", "MIA", "ORL", "WAS"],
    },
    "West": {
        "Northwest": ["DEN", "MIN", "OKC", "POR", "UTA"],
        "Pacific": ["GSW", "LAC", "LAL", "PHX", "SAC"],
        "Southwest": ["DAL", "HOU", "MEM", "NOP", "SAS"],
    },
}


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

    # Barrett Scores for anyone the projected board does not carry (two-way
    # players, mostly). Same fallback the roster pipeline uses: this season
    # first, then last season tagged so a card can say which year it is.
    scores = NameIndex()
    try:
        from utils import build_raw
        for _season, _tag in (("2025-26", None), ("2024-25", "24-25")):
            for _r in build_raw(_season).itertuples():
                scores.add(_r.Player, (round(float(_r.barrett_score), 1), _tag))
    except Exception:                                # never block the page
        pass

    # last season's per-game line + D-LEBRON
    box = NameIndex()
    try:
        import pandas as pd
        _ls = pd.read_parquet(_ROOT / "cache" / "league_stats_2025_26.parquet")
        _dl = fetch_dlebron("2025-26")
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

    _conf = {ab: c for c, ds in _DIVS.items() for d in ds.values() for ab in d}
    _div = {ab: d for ds in _DIVS.values() for d, abs_ in ds.items() for ab in abs_}

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
            m = model.get(name) or {}
            fv = fa_val.get(name) or {}
            value = m.get("value")
            if kind == "free_agent":
                value = fv.get("value")
                if value is None:                      # "est value $12.3M"
                    _mm = re.search(r"\$([0-9.]+)M", note)
                    value = float(_mm.group(1)) if _mm else None
                sal = 0.0
            pid = heads.get(name)
            bar, bar_yr = m.get("barrett"), m.get("bs_yr")
            if bar is None:
                bar, bar_yr = fv.get("barrett"), fv.get("bs_yr")
            if bar is None:
                bar, bar_yr = scores.get(name) or (None, None)
            o = opts.get(name) or {}
            pos = (r.get("pos") or "").strip()
            players.append({
                "n": name, "kind": kind,
                "pos": pos,
                "slot": _SLOT.get(pos.split("/")[0].upper(), ""),
                "salary": round(sal, 2),
                "note": note,
                "value": round(value, 1) if value is not None else None,
                "vd": (round(sal - value, 1)
                       if (value is not None and kind == "standard") else None),
                "barrett": bar, "bs_yr": bar_yr,
                **(box.get(name) or {}),
                "opt": o.get("d") or None,
                "opt_note": o.get("note") or None,
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
            if have.get(nm):
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
        payroll = round(sum(p["salary"] for p in std), 1)
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
        "season": "2026-27", "asof": asof, "cfg": cfg, "logos": logos,
        "colors": dict(TEAM_HEX), "divs": _DIVS, "teams": teams,
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
    "the cap. Model value reads 2025-26 production against 2026-27 money, so "
    "a star who was hurt last season looks overpaid here, and a player who "
    "did not play at all carries no value at all and is left out of the "
    "surplus on both sides. Committed salary counts signed 2026-27 contracts "
    "only: it does not include dead money from waived players, cap holds for "
    "the team's own free agents, or empty-roster-spot charges."
)

render_footer()
