"""Trade Machine: Fanspo-style interactive trade builder.

Pick teams, see full rosters as cards, and drag players/picks between teams
(or use the arrow buttons -- also the touch path). Verdicts render instantly:
the 2023-CBA rules engine runs IN THE BROWSER as a line-for-line JS port of
trade_rules.py. Parity vectors exported from the verified Python engine
(scripts/export_trade_vectors.py) execute on every load; any divergence shows
a loud red banner instead of silently wrong verdicts.

All data is precomputed JSON injected at render -- no request-path model work.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import json

import streamlit as st
import streamlit.components.v1 as components

from utils import (_headshot_id_map, normalize, render_nav, render_page_chrome,
                   _bootstrap_warm)

st.set_page_config(page_title="Trade Machine", page_icon="static/favicon.svg", layout="wide")
render_page_chrome()
_bootstrap_warm()
render_nav("Trade Machine")

_ROOT = Path(__file__).parent.parent
_TODAY = "2026-07-27"

st.title("Trade Machine")
st.caption(
    "Pick two to four teams, then drag players and picks between them (or use "
    "the arrow buttons). The verdict updates instantly under the 2023 CBA: "
    "salary matching, apron rules, hard caps, the Stepien rule, and trade "
    "timing, each explained in plain English. Rosters are the real 2026-27 "
    "books; projected (unsigned) players are not tradable."
)


@st.cache_data(show_spinner=False)
def _payload() -> str:
    teams_raw = json.loads((_ROOT / "cache" / "team_pages.json").read_text())["teams"]
    ledger = json.loads((_ROOT / "cache" / "pick_ledger.json").read_text())["teams"]
    vectors = json.loads((_ROOT / "cache" / "trade_vectors.json").read_text())
    cfg = {k: v for k, v in json.loads(
        (_ROOT / "data" / "cba_config_2026_27.json").read_text()).items()
        if not k.startswith("_")}
    heads = _headshot_id_map()
    # verified master list of 2026-27 hard caps (apron caps triggered by
    # offseason mechanics); missing file or team = not hard-capped
    hard_caps = {}
    hc_path = _ROOT / "data" / "hard_caps_2026_27.csv"
    if hc_path.exists():
        with hc_path.open() as f:
            for r in csv.DictReader(x for x in f if not x.lstrip().startswith("#")):
                if r.get("team") and r.get("cap", "").strip() in ("apron1", "apron2"):
                    hard_caps[r["team"].strip()] = {
                        "cap": r["cap"].strip(),
                        "why": (r.get("trigger") or "").strip()}
    # verified master list of outstanding traded-player exceptions; shown as
    # info on team panels (verdicts do not model TPE absorption yet)
    tpes = {}
    tpe_path = _ROOT / "data" / "trade_exceptions_2026_27.csv"
    if tpe_path.exists():
        with tpe_path.open() as f:
            for r in csv.DictReader(x for x in f if not x.lstrip().startswith("#")):
                if r.get("team") and r.get("amount_M"):
                    created = (r.get("created") or "").strip()
                    tpes.setdefault(r["team"].strip(), []).append({
                        "amt": float(r["amount_M"]),
                        "player": (r.get("player") or "").strip(),
                        "expires": (r.get("expires") or "").strip(),
                        # created before this league year (or unknown, taken
                        # conservatively) = prior-season -> apron-1 trigger
                        "prior": not created or created < "2026-07"})
    for v in tpes.values():
        v.sort(key=lambda x: -x["amt"])
    signings = {}
    with (_ROOT / "data" / "real_signings_2026.csv").open() as f:
        for r in csv.DictReader(x for x in f if not x.lstrip().startswith("#")):
            if r.get("player") and r.get("signed_date"):
                d = r["signed_date"].strip()
                signings[normalize(r["player"])] = {
                    "date": d + "-01" if len(d) == 7 else d,
                    "type": (r.get("type") or "").strip().lower()}
    teams = {}
    for abbr, t in teams_raw.items():
        players = []
        for p in t["players"]:
            if not p.get("real", True) or p["role"] == "Draft pick" or p.get("salary") is None:
                continue
            n = normalize(p["n"])
            sig = signings.get(n)
            pid = heads.get(n)
            players.append({
                "n": p["n"], "salary": float(p["salary"]),
                "pos": (p.get("pos") or "").split("/")[0],
                "value": p.get("value"), "barrett": p.get("barrett"),
                "headshot": (f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"
                             if pid else None),
                "signed_date": (sig or {}).get("date"),
                "signed_via": ("bird_resign" if (sig or {}).get("type") == "resign"
                               else "standard"),
            })
        players.sort(key=lambda x: -x["salary"])
        picks = []
        for pk in ledger.get(abbr, {}).get("controls", []):
            low = (pk.get("notes") or "").lower()
            if "frozen" in low or "untradeable" in low or "untradable" in low:
                continue
            swap_only = bool(pk.get("swap_with")) and pk["origin"] == abbr
            entry = {"year": pk["year"], "origin": pk["origin"],
                     "round": pk.get("round", 1),
                     "protection": pk.get("protection", "")}
            if swap_only:
                # a swap-encumbered own pick is the residual (worse of the
                # two); it can move, but ONLY as a further swap right so the
                # year stays Stepien-covered
                entry["swap_only"] = True
                entry["swap_with"] = pk.get("swap_with", "").strip()
            picks.append(entry)
        hc = hard_caps.get(abbr)
        teams[abbr] = {"name": t["name"], "payroll": t["payroll"], "size": t["size"],
                       "covered": ledger.get(abbr, {}).get("covered", []),
                       "players": players, "picks": picks,
                       "hard_cap": (cfg["apron1"] if hc["cap"] == "apron1"
                                    else cfg["apron2"]) if hc else None,
                       "hard_cap_label": ({"apron1": "1st apron",
                                           "apron2": "2nd apron"}[hc["cap"]]
                                          if hc else None),
                       "hard_cap_why": hc["why"] if hc else None,
                       "tpes": tpes.get(abbr, [])}
    # held SWAP RIGHTS: surface each right as a tradable card on the HOLDER's
    # panel. swap_with names the COUNTERPARTY, not the holder -- two-team
    # swaps are often recorded on BOTH origins' rows, and reading swap_with
    # as the holder once flipped the Kessler swaps (a false "LAL holds UTA
    # 2030 swap" card). Resolution order: an explicit swap_holder wins; a
    # single-sided swap_with means that team holds the right; a reciprocal
    # recording with no explicit holder emits nothing (ambiguous webs stay
    # in notes). The origin's encumbered own pick stays hidden as before.
    _swaps = {(p["origin"], p["year"]): (p.get("swap_with") or "").strip()
              for led in ledger.values() for p in led.get("controls", [])
              if p.get("round", 1) == 1 and p["controlled_by"] == p["origin"]}
    for abbr, led in ledger.items():
        for pk in led.get("controls", []):
            if pk["controlled_by"] != pk["origin"] or pk.get("round", 1) != 1:
                continue
            sw = (pk.get("swap_with") or "").strip()
            holder = (pk.get("swap_holder") or "").strip()
            if not holder and sw in teams and sw != pk["origin"]:
                reciprocal = _swaps.get((sw, pk["year"])) == pk["origin"]
                holder = "" if reciprocal else sw
            if holder in teams and holder != pk["origin"]:
                teams[holder]["picks"].append({
                    "year": pk["year"], "origin": pk["origin"], "round": 1,
                    "protection": pk.get("protection", ""), "swap_right": True})
    for t in teams.values():
        t["picks"].sort(key=lambda p: (p["year"], p["origin"], p.get("round", 1)))
    # free agents each team can sign-and-trade (its OWN FAs; est value seeds
    # the editable starting salary, minimum-level guys default to the vet min)
    fas = {}
    fa_path = _ROOT / "cache" / "fa_pool_v1.json"
    if fa_path.exists():
        for f in json.loads(fa_path.read_text())["fas"]:
            ab = f.get("team", "")
            if ab not in teams or not f.get("n"):
                continue
            pid = heads.get(normalize(f["n"]))
            fas.setdefault(ab, []).append({
                "n": f["n"], "pos": (f.get("pos") or "").strip("—- "),
                "value": round(f.get("value") or 2.1, 1),
                "headshot": (f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"
                             if pid else None)})
        for v in fas.values():
            v.sort(key=lambda x: -x["value"])
    from nba_api.stats.static import teams as _nbat
    logos = {t["abbreviation"]:
             f"https://cdn.nba.com/logos/nba/{t['id']}/global/L/logo.svg"
             for t in _nbat.get_teams()}
    return json.dumps({
        "abbrs": sorted(teams), "teams": teams, "cfg": cfg, "vectors": vectors,
        "logos": logos, "fas": fas, "today": _TODAY,
        "default_teams": ["LAL", "BKN"],
    }, separators=(",", ":"))


def _payload_with_request_state() -> str:
    """Splice per-request fields (share token from the URL, share base) into
    the cached static payload without reparsing it."""
    token = st.query_params.get("trade", "")
    extras = (',"initial":' + json.dumps(token or None)
              + ',"share_base":"https://hoopsvalue.com/Trade_Machine"}')
    return _payload()[:-1] + extras


_DARK = """--panel:#15171d;--card:#1b1f28;--line:#262a33;--track:#242833;
--fg1:#e6e9f2;--fg2:#c3c8d4;--fg3:#a7adbb;--fg4:#8a8f9c;--fg5:#6b7079;
--teal:#16d4c1;--good:#2ecc71;--bad:#e74c3c;--amberc:#f0b35b;--blue:#4c8dff;
--tintg:rgba(22,212,193,.08);--tintb:rgba(231,76,60,.10);--tintz:rgba(255,255,255,.02);"""
_LIGHT = """--panel:#ffffff;--card:#f7f8fa;--line:#e3e6eb;--track:#eceef2;
--fg1:#16233f;--fg2:#3a4150;--fg3:#5b6472;--fg4:#6b7280;--fg5:#9aa0ac;
--teal:#0fae9d;--good:#16a34a;--bad:#e0483a;--amberc:#b97f24;--blue:#2563eb;
--tintg:rgba(15,174,157,.08);--tintb:rgba(224,72,58,.08);--tintz:rgba(22,35,63,.025);"""

# BOTH palettes ship in one byte-stable srcdoc and the component detects the
# parent's theme at runtime. Injecting only the active palette made the html
# change on every theme toggle, which forced Streamlit to remount the iframe;
# racing remounts during quick light/dark cycling could leave it blank.
_html = ((_ROOT / "templates" / "trade_machine.html").read_text()
         .replace("__THEME__", _DARK)
         .replace("__THEME_LIGHT__", _LIGHT)
         .replace("__DATA__", _payload_with_request_state()))
# Initial height only: the component resizes its own frame to fit the full
# rosters (window.frameElement, same-origin srcdoc), so nothing scrolls inside.
components.html(_html, height=1000, scrolling=False)

st.caption(
    "Legality uses the verified 2023 CBA rules; ambiguous edge cases resolve "
    "conservatively. The in-browser engine proves itself against the verified "
    "Python engine's test vectors on every load. Player values are the "
    "model's 2026-27 market values."
)
