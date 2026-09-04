"""Redraft: strip every team status off every player and rebuild rosters from
scratch under the real 2026-27 cap.

Every player in the league becomes available, his contract comes with him, and
you fill 15 slots while the cap bar tracks you against all five lines (floor,
cap, tax, first apron, second apron). Set how many franchises are drafting and
you can rebuild the whole league, one team at a time.

The board is TAP-driven, not drag-driven. The Trade Machine's drag-and-drop
cannot fire from touch on iOS, so a phone visitor cannot move a single asset
there; this page is built on clicks and behaves identically on a phone.

All data is precomputed (cache/redraft_pool_v1.json via
scripts/build_redraft_pool.py) -- no model work on the request path.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import streamlit as st
import streamlit.components.v1 as components

from utils import (TEAM_HEX, render_nav, render_page_chrome,
                   script_json, _bootstrap_warm)

st.set_page_config(page_title="Redraft", page_icon="static/favicon.svg", layout="wide")
render_page_chrome()
_bootstrap_warm()
render_nav("Redraft")

_ROOT = Path(__file__).parent.parent

st.title("Redraft")
st.caption(
    "Every player in the league is available and his contract comes with him. "
    "Fill 15 roster spots and watch the bar: floor, cap, tax, first apron, "
    "second apron. Set how many teams are drafting to rebuild the whole league."
)


@st.cache_data(show_spinner=False)
def _payload() -> str:
    pool = json.loads((_ROOT / "cache" / "redraft_pool_v1.json").read_text())
    cfg = {k: v for k, v in json.loads(
        (_ROOT / "data" / "cba_config_2026_27.json").read_text()).items()
        if not k.startswith("_")}
    # full franchise names come from the roster board's cache, the same place
    # the Rosters page reads them, so the two pages can never disagree
    names = {a: t.get("name", a) for a, t in
             json.loads((_ROOT / "cache" / "team_pages.json").read_text())["teams"].items()}
    abbrs = sorted(TEAM_HEX)
    return json.dumps({
        "cfg": cfg,
        "players": pool["players"],
        "value_season": pool.get("value_season", ""),
        "abbrs": abbrs,
        "colors": {a: TEAM_HEX[a] for a in abbrs},
        "names": {a: names.get(a, a) for a in abbrs},
    }, separators=(",", ":"))


_DARK = """--panel:#141a24;--card:#1a212d;--line:#242c3a;--track:#212936;
--fg1:#e8ecf3;--fg2:#c2c9d6;--fg3:#98a1b2;--fg4:#7b8497;--fg5:#5d6678;
--teal:#16d4c1;--good:#2ecc71;--bad:#e74c3c;--amberc:#f0b35b;--blue:#4c8dff;
--tintg:rgba(22,212,193,.08);--tintb:rgba(231,76,60,.10);--tintz:rgba(255,255,255,.02);"""
_LIGHT = """--panel:#ffffff;--card:#f7f8fa;--line:#e3e6eb;--track:#eceef2;
--fg1:#16233f;--fg2:#3a4150;--fg3:#5b6472;--fg4:#6b7280;--fg5:#9aa0ac;
--teal:#0fae9d;--good:#16a34a;--bad:#e0483a;--amberc:#b97f24;--blue:#2563eb;
--tintg:rgba(15,174,157,.08);--tintb:rgba(224,72,58,.08);--tintz:rgba(22,35,63,.025);"""

# Both palettes ship in one byte-stable srcdoc and the component picks the
# active one at runtime; injecting only one would change the html on every
# theme toggle and force Streamlit to remount the iframe.
_html = ((_ROOT / "templates" / "redraft.html").read_text()
         .replace("__THEME__", _DARK)
         .replace("__THEME_LIGHT__", _LIGHT)
         .replace("__DATA__", script_json(_payload())))
components.html(_html, height=980, scrolling=False)

from utils import render_footer  # noqa: E402
render_footer()
