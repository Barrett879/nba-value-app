import sys
import html
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from utils import (
    _bootstrap_warm,
    THEME_DEFAULT_DARK,
    build_ranked_projected,
    fetch_next_year_contracts,
    fetch_rookie_scale_players,
    fetch_player_career_with_rank,
    fmt_next_contract, classify_fa_status,
    season_to_espn_year,
    normalize,
    SEASONS,
    DEFAULT_MIN_THRESHOLD,
    get_all_player_names,
    HISTORICAL_TRADES,
    trade_side_summary,
    _PLAYOFF_HELP,
    inject_theme,
    render_theme_toggle,
    COMMON_CSS, render_nav, TEAM_HEX, HV_KIT_CSS,
)

# Featured players for the Legacy preview overlay on the home page.
# IDs come from nba_api.stats.static.players. One per major era — five eras:
# Jordan ('80s/'90s), Kobe ('00s), LeBron ('10s), Curry (3-pt revolution),
# Jokić (modern bigs).
LEGACY_FEATURED = [
    {"name": "Michael Jordan",  "id":    893, "color": "#f1c40f"},  # gold
    {"name": "Kobe Bryant",     "id":    977, "color": "#9b59b6"},  # purple
    {"name": "LeBron James",    "id":   2544, "color": "#e63946"},  # red
    {"name": "Stephen Curry",   "id": 201939, "color": "#16d4c1"},  # teal — splash
    {"name": "Nikola Jokić",    "id": 203999, "color": "#7ec8e8"},  # blue
]


# Start warming all season caches the moment the server boots —
# before any user arrives, so the first visitor doesn't pay the cost.
_bootstrap_warm()

st.set_page_config(page_title="HoopsValue", page_icon="static/favicon.svg", layout="wide")

# Theme tokens (light/dark) — home is self-contained chrome (no render_page_chrome),
# so it injects the tokens itself. Must run right after set_page_config.
inject_theme()

# Shared chrome CSS (fixed top-nav styling lives here). Injected BEFORE the
# homepage-specific block below, so the homepage's own overrides still win.
st.markdown(COMMON_CSS, unsafe_allow_html=True)
st.markdown(HV_KIT_CSS, unsafe_allow_html=True)  # rails, mini faces, team dots

# ── Page chrome (background, hide Streamlit UI) ────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: var(--app-bg) !important;   /* flat, court photo removed (design refresh) */
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] { background: transparent !important; }

    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    .stApp > header {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        visibility: hidden !important;
    }
    [data-testid="stToolbar"]        { display: none !important; height: 0 !important; }
    [data-testid="stDecoration"]     { display: none !important; height: 0 !important; }
    [data-testid="stStatusWidget"]   { display: none !important; }
    [data-testid="stAppViewerBadge"] { display: none !important; }
    [data-testid="stBottom"]         { display: none !important; }
    [data-testid="stSidebarNav"]     { display: none !important; }
    [data-testid="stSidebar"]        { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    .viewerBadge_container__r5tak    { display: none !important; }
    .styles_viewerBadge__CvC9N       { display: none !important; }
    #MainMenu, footer                { visibility: hidden; }

    .block-container,
    .main .block-container,
    section.main > .block-container,
    [data-testid="stMain"] .block-container,
    [data-testid="stMainBlockContainer"],
    section[data-testid="stMain"] > .block-container {
        padding-top: 5.2rem !important;   /* clear the fixed top-nav with breathing room */
        padding-bottom: 1rem !important;
        /* Spotrac-style gutters: content uses most of the screen but always
           floats with generous whitespace off the browser edges. */
        padding-left: 4.5rem;
        padding-right: 4.5rem;
        max-width: 1850px;
    }
    @media (max-width: 900px) {
        .block-container,
        [data-testid="stMain"] .block-container,
        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    .stApp { padding-top: 0 !important; }

    /* Horizontal tab strips, one per nav page */
    a.tab-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--panel);
        border: 1px solid var(--panel-line);
        border-left: 3px solid var(--accent, var(--accent-red));
        border-radius: 8px;
        padding: 0.85rem 1.2rem;
        text-decoration: none;
        margin-bottom: 0.55rem;
        transition: background-color 0.15s, border-color 0.15s, transform 0.1s;
        backdrop-filter: blur(2px);
    }
    a.tab-strip:hover {
        background: var(--panel-hover);
        border-color: var(--accent, var(--accent-red));
        text-decoration: none;
        transform: translateX(2px);
    }
    .tab-strip-name {
        color: var(--fg-1);
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-right: 1.2rem;
        flex-shrink: 0;
        min-width: 165px;
    }
    .tab-strip-desc {
        color: var(--fg-3);
        font-size: 0.82rem;
        flex-grow: 1;
        line-height: 1.35;
    }
    .tab-strip-arrow {
        color: var(--accent, var(--accent-red));
        font-size: 1.2rem;
        font-weight: 600;
        margin-left: 1rem;
        flex-shrink: 0;
    }

    /* Streamlit expander styling, make it sit tight under each strip */
    div[data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
        margin-top: -0.4rem;
        margin-bottom: 0.6rem;
    }
    div[data-testid="stExpander"] details {
        border: none !important;
        background: transparent !important;
    }
    div[data-testid="stExpander"] summary {
        color: var(--fg-4) !important;
        font-size: 0.78rem !important;
        padding-left: 1.2rem !important;
        background: transparent !important;
    }
    div[data-testid="stExpander"] summary:hover { color: var(--fg-2) !important; }
    .preview-box {
        background: rgba(0, 0, 0, 0.3);
        border-left: 2px solid var(--hairline);
        border-radius: 4px;
        padding: 0.8rem 1rem;
        margin-left: 1rem;
        margin-top: 0.3rem;
    }

    /* Expandable Explore-Deeper strips (raw HTML <details>), same visual
       treatment as the old tab-strip header but the strip is the expand
       trigger instead of a direct navigation link. CTA inside the body
       handles the actual navigation. */
    details.explore-strip {
        background: var(--panel);
        border: 1px solid var(--panel-line);
        border-left: 3px solid var(--accent, var(--accent-red));
        border-radius: 8px;
        margin-bottom: 0.55rem;
        overflow: hidden;
        backdrop-filter: blur(2px);
    }
    summary.explore-strip-summary {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.85rem 1.2rem;
        cursor: pointer;
        list-style: none;
        transition: background-color 0.15s;
    }
    summary.explore-strip-summary::-webkit-details-marker { display: none; }
    summary.explore-strip-summary::marker { content: ""; }
    summary.explore-strip-summary:hover {
        background: var(--panel-hover);
    }
    details[open].explore-strip > summary.explore-strip-summary {
        border-bottom: 1px solid var(--hairline);
    }
    .strip-arrow {
        color: var(--accent, var(--accent-red));
        font-size: 1.1rem;
        font-weight: 600;
        margin-left: 1rem;
        flex-shrink: 0;
        transition: transform 0.2s ease;
    }
    details[open].explore-strip .strip-arrow { transform: rotate(180deg); }

    .explore-strip-body {
        padding: 1rem 1.2rem 1.1rem;
        background: var(--panel-2);
    }
    a.goto-btn {
        display: inline-block;
        background: var(--accent, var(--accent-red));
        color: var(--fg-1) !important;
        padding: 0.45rem 1.1rem;
        border-radius: 6px;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 0.8rem;
        transition: opacity 0.15s, transform 0.1s;
    }
    a.goto-btn:hover {
        opacity: 0.88;
        transform: translateX(2px);
        text-decoration: none !important;
    }

    /* Playoff-mode toggle pinned to the top-right (same CSS as the
       inner pages, the home page doesn't have the nav bar but still
       needs the toggle in the same visual slot). */
    .st-key-playoff_nav_toggle {
        position: fixed !important;
        top: 0.45rem !important;
        right: 1rem !important;
        z-index: 10001 !important;
        margin: 0 !important;
        padding: 0 !important;
        width: auto !important;
        background: transparent !important;
    }
    .st-key-playoff_nav_toggle [data-testid="stToggle"] {
        background: transparent;
        padding: 0;
    }
    .st-key-playoff_nav_toggle label p {
        color: var(--fg-3) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    .st-key-playoff_nav_toggle:hover label p { color: var(--fg-1) !important; }

    /* Theme (brightness) button, pinned to the far top-right (home has no nav
       bar, but the button sits in the same slot as on the inner pages). */
    .st-key-theme_nav_toggle {
        position: fixed !important;
        top: 0.45rem !important;
        right: 1rem !important;
        z-index: 10001 !important;
        margin: 0 !important;
        padding: 0 !important;
        width: auto !important;
        background: transparent !important;
    }
    .st-key-theme_nav_toggle button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.1rem 0.3rem !important;
        min-height: 0 !important;
        height: auto !important;
        line-height: 1 !important;
        font-size: 1.3rem !important;
        color: var(--fg-3) !important;
    }
    .st-key-theme_nav_toggle button [data-testid="stIconMaterial"] { font-size: 1.3rem !important; }
    .st-key-theme_nav_toggle button:hover { color: var(--fg-1) !important; background: transparent !important; }
    .st-key-theme_nav_toggle button:active,
    .st-key-theme_nav_toggle button:focus,
    .st-key-theme_nav_toggle button:focus-visible {
        box-shadow: none !important; background: transparent !important; color: var(--fg-1) !important;
    }

    /* Rail section headers: kicker + title + count pill + rule line. The page's
       unifying device (front page strip, player hub, the board). */
    .hv-rail{display:flex;align-items:center;gap:.65rem;margin:0;padding:2.4rem 0 .7rem;}
    .hv-rail::before{content:"";width:4px;height:1.05em;background:var(--accent-red);
        border-radius:2px;flex:0 0 auto;}
    .hv-rail .k{font-size:.64rem;font-weight:800;letter-spacing:.11em;
        text-transform:uppercase;color:var(--fg-4);white-space:nowrap;}
    .hv-rail .t{font-size:1.02rem;font-weight:800;color:var(--fg-1);white-space:nowrap;}
    .hv-rail .n{font-size:.72rem;font-weight:700;color:var(--fg-4);background:var(--panel-2);
        border:1px solid var(--panel-line);border-radius:99px;padding:.1rem .55rem;
        font-variant-numeric:tabular-nums;white-space:nowrap;}
    .hv-rail .meta{font-size:.72rem;color:var(--fg-4);white-space:nowrap;}
    .hv-rail::after{content:"";flex:1;height:1px;background:var(--hairline);order:9;}
    .hv-rail .meta{order:10;}

    /* Front Page strip: 4 clickable feature cards under the search. */
    .fp-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem;
        margin:0;padding:.25rem 0 .8rem;}
    @media(max-width:1100px){.fp-grid{grid-template-columns:1fr 1fr;}}
    a.fp-card{position:relative;display:flex;align-items:center;gap:.8rem;
        background:var(--panel-solid);border:1px solid var(--panel-line);
        border-left:4px solid var(--team,var(--accent-teal));border-radius:12px;
        padding:.8rem 1rem;box-shadow:var(--shadow-card);text-decoration:none !important;
        min-height:104px;}
    a.fp-card:hover{background:var(--panel-hover);border-color:var(--team,var(--accent-teal));}
    img.fp-face{width:52px;height:52px;border-radius:50%;object-fit:cover;
        object-position:center 12%;background:var(--panel-2);
        border:2px solid var(--team,var(--panel-line));flex:0 0 auto;}
    .fp-card .k{font-size:.64rem;font-weight:800;letter-spacing:.1em;
        text-transform:uppercase;color:var(--fg-4);}
    .fp-card .nm{font-size:.95rem;font-weight:800;color:var(--fg-1);line-height:1.15;
        margin:.1rem 0;}
    .fp-card .v{font-size:1.35rem;font-weight:800;font-variant-numeric:tabular-nums;
        line-height:1.1;}
    .fp-card .v.good{color:var(--value-good);}
    .fp-card .v.bad{color:var(--value-bad);}
    .fp-card .v.teal{color:var(--accent-teal);}
    .fp-card .sub{font-size:.72rem;color:var(--fg-3);}
</style>
""", unsafe_allow_html=True)

# Full top nav (tabs + theme toggle), same bar as every other page. No tab is
# "active" on the homepage — the Home link itself marks where you are.
render_nav("")

# ── Hero — HoopsValue logo + tagline ────────────────────────────────────────
# Streamlit serves static files via enableStaticServing in config.toml, but
# inline <img src="./app/static/..."> doesn't resolve cleanly inside HTML
# markdown (CSS background-image works, <img> doesn't — different base URL
# handling). st.image() loads the file directly and avoids the path issue.
# Premium wordmark logo (design refresh) — HTML/SVG via st.markdown. The metals
# come from CSS vars so the coming light theme can retune them per mode.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Manrope:wght@500;600;700&display=swap');
/* logo metals (--logo-copper/-sage/-tag) come from the theme tokens so they
   retune per mode; see utils.THEME_BASE_CSS / THEME_LIGHT_CSS */
.hv-logo-wrap{display:flex;justify-content:center;padding:0.55rem 0 1.0rem;margin-bottom:0.9rem;}
.hv-logo{display:inline-flex;flex-direction:column;align-items:center;font-size:46px;gap:3px;user-select:none}
.hv-wm{display:inline-flex;align-items:center;font-family:"Space Grotesk",sans-serif;font-weight:700;line-height:1;letter-spacing:-.035em}
.hv-wm .cu{color:var(--logo-copper)}
.hv-wm .sg{color:var(--logo-sage)}
.hv-ball{width:.86em;height:.86em;margin:0 -.02em;position:relative;top:.02em;color:var(--logo-copper);flex:0 0 auto}
.hv-tag{display:flex;align-items:center;gap:.8em;font-family:"Manrope",sans-serif;font-weight:600;font-size:.185em;letter-spacing:.34em;text-transform:uppercase;color:var(--logo-tag);white-space:nowrap}
.hv-tag::before,.hv-tag::after{content:"";height:1px;width:3.1em;background:currentColor;opacity:.45}
</style>
<div class="hv-logo-wrap"><div class="hv-logo">
  <div class="hv-wm">
    <span class="cu">HO</span>
    <svg class="hv-ball" viewBox="0 0 100 100"><g fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round"><circle cx="50" cy="50" r="46"/><path d="M50 4 V96 M4 50 H96 M20 14 Q42 50 20 86 M80 14 Q58 50 80 86"/></g></svg>
    <span class="cu">PS</span><span class="sg">VALUE</span>
  </div>
  <div class="hv-tag">NBA Contract Value</div>
</div></div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding: 0 0 0.4rem 0; margin-bottom: 0.6rem;">
    <div style="font-size:0.8rem; color:var(--fg-2); max-width:820px; margin:0 auto; line-height:1.45;">
        Every NBA player since 1973, ranked by the <b style="color:var(--fg-1);">Barrett Score</b> · find the steals · expose the overpays · settle the GOAT debate
    </div>
</div>
""", unsafe_allow_html=True)

# ── Search hero ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSelectbox"][data-baseweb] div[role="combobox"] {
    background: var(--panel-solid) !important;
    border: 1px solid var(--panel-line) !important;
    border-radius: 10px !important;
    box-shadow: var(--shadow-card);
}
[data-testid="stSelectbox"]:focus-within div[role="combobox"] {
    border: 2px solid var(--sky) !important;
}
.home-search-label {
    font-size: 0.7rem;
    color: var(--fg-2);
    text-align: center;
    margin: 0.6rem 0 0.3rem;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

_, _search_col, _ = st.columns([1, 2.6, 1])
with _search_col:
    st.markdown(
        '<div class="home-search-label">SEARCH ANY PLAYER · CAREER ARCS · HEAD-TO-HEAD COMPARISONS · 1973 → TODAY</div>',
        unsafe_allow_html=True,
    )
    _all_player_names = get_all_player_names() or []
    _picked = st.selectbox(
        "Search any player",
        options=_all_player_names,
        index=None,
        placeholder="Type a name: LeBron, Jordan, Magic, Jokić, Wembanyama…",
        label_visibility="collapsed",
        key="home_search_select",
    )
    if _picked:
        st.session_state["search_player"] = _picked
        try:
            st.switch_page("pages/Search.py")
        except Exception:
            st.markdown(
                f'<a href="/Search" target="_top" style="color:var(--sky); text-decoration: underline;">'
                f'Click here to view {_picked}\'s profile →</a>',
                unsafe_allow_html=True,
            )

st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Player Hub — the homepage IS the player list. Selecting a player (row link or
# ?player= deep link) opens a hub panel: score + rank, predicted contract, FA
# outcome, career arc, and jump buttons into the full tabs. HARD RULE: everything
# below reads precomputed caches only — no live model runs, no network.
# ══════════════════════════════════════════════════════════════════════════════
import json as _json
import csv as _csv
from urllib.parse import quote as _urlquote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    CACHE_DIR, html_table, theme_fig, get_player_draft_info,
    fetch_bref_positions, HV_TABLE_CSS, _HV_SORT_SCRIPT, get_player_contract_info,
    fetch_league_stats, SALARY_CAP_M, get_max_contract_eligibility,
    FACE_GUARD_SCRIPT,
    face_img as _face_img, render_rail as _rail, spark_svg as _spark_svg,
    hex_rgba as _hex_rgba, hex_darken as _hex_darken, hex_is_light as _hex_is_light,
)
import streamlit.components.v1 as _components
import team_suitors as _ts

# The homepage is self-contained chrome (no render_page_chrome), so it must emit
# the themed-table CSS itself — without it the table has no overflow container
# and spills over the footer — plus the delegated click-to-sort script.
st.markdown(HV_TABLE_CSS, unsafe_allow_html=True)
_components.html(_HV_SORT_SCRIPT, height=0)

_components.html(FACE_GUARD_SCRIPT, height=0)     # hide 404 headshots

_HUB_SEASON = SEASONS[0]
_NEXT_SEASON = f"{int(_HUB_SEASON[:4]) + 1}-{(int(_HUB_SEASON[:4]) + 2) % 100:02d}"


@st.cache_data(ttl=3600, show_spinner=False)
def _actual_max_norms(candidates: tuple) -> set:
    """Norms whose ACTUAL next-season salary sits at/above their own CBA max
    (25/30/35% of the 2026-27 cap by service years). Mid-contract max deals
    with built-in raises exceed the new-deal max, so >= catches them too.
    Only called for the handful of salaries above the 25% floor."""
    _cap = SALARY_CAP_M.get(_NEXT_SEASON, 165.0)
    out = set()
    for _nm, _player, _nx in candidates:
        try:
            _e = get_max_contract_eligibility(_player, _HUB_SEASON)
            if _nx >= _e["max_pct"] * _cap - 0.05:
                out.add(_nm)
        except Exception:
            pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _hub_pcv() -> dict:
    """Full-pool predicted contracts (scripts/build_player_hub.py). {} if absent."""
    try:
        # REPO copy, not CACHE_DIR: /data on Render only seeds MISSING files, so it
        # keeps serving the stale first-ever copy of repo-authored caches like this one.
        return _json.loads((Path(__file__).parent / "cache" / "player_hub_pcv_v1.json")
                           .read_text()).get("players", {})
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _hub_signings() -> dict:
    """Real 2026 signings keyed by normalized name (accuracy tracker cache)."""
    try:
        d = _json.loads((Path(__file__).parent / "cache" / "accuracy_tracker_v1.json").read_text())
        return {normalize(s["player"]): s for s in d.get("signings", [])}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _hub_decisions() -> dict:
    """Option decisions {norm: (decision, figure_M)} from data/option_decisions_2026.csv."""
    out = {}
    try:
        with open(Path(__file__).parent / "data" / "option_decisions_2026.csv") as fh:
            for r in _csv.DictReader(l for l in fh if l.strip() and not l.lstrip().startswith("#")):
                if r.get("player"):
                    try:
                        fig = float(r.get("figure_M") or 0) or None
                    except ValueError:
                        fig = None
                    out[normalize(r["player"])] = ((r.get("decision") or "").strip(), fig)
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _hub_career() -> pd.DataFrame:
    """Per-season score/rank/salary for every player 1973→today (combined parquet)."""
    cols = ["Player", "Season", "barrett_score", "score_rank", "salary"]
    try:
        df = pd.read_parquet(CACHE_DIR / "all_seasons_0_v7.parquet", columns=cols)
        df["norm"] = df["Player"].map(normalize)
        return df
    except Exception:
        return pd.DataFrame(columns=cols + ["norm"])


@st.cache_data(ttl=3600, show_spinner=False)
def _hub_career_agg() -> pd.DataFrame:
    """Career aggregates per player (for the Career Twins quadrant): seasons played,
    average + peak Barrett Score, peak season, best rank, top salary."""
    car = _hub_career()
    if car.empty:
        return pd.DataFrame()
    g = car.groupby("norm").agg(
        Player=("Player", "last"), yrs=("Season", "nunique"),
        avg=("barrett_score", "mean"), peak=("barrett_score", "max"),
        best_rank=("score_rank", "min"), top_sal=("salary", "max"),
    )
    peak_season = car.loc[car.groupby("norm")["barrett_score"].idxmax()].set_index("norm")["Season"]
    g["peak_season"] = peak_season
    return g.reset_index()


@st.cache_data(ttl=3600, show_spinner=False)
def _hub_counting() -> dict:
    """Per-game box-score line for the current season, keyed by normalized name
    (the same warm league-stats parquet the pool build reads)."""
    try:
        _df = fetch_league_stats(_HUB_SEASON)
        return {normalize(str(r["PLAYER_NAME"])):
                (float(r["PTS"]), float(r["REB"]), float(r["AST"]),
                 float(r["STL"]), float(r["BLK"]))
                for _, r in _df.iterrows()}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _hub_salary_supplement() -> dict:
    """Verified 2026-27 salaries for players the scraped next-year feed omits
    (data/salary_supplement_2026_27.csv, hand-verified rows with sources)."""
    out = {}
    try:
        with open(Path(__file__).parent / "data" / "salary_supplement_2026_27.csv") as fh:
            for r in _csv.DictReader(l for l in fh if l.strip() and not l.lstrip().startswith("#")):
                if r.get("player") and r.get("salary_M"):
                    try:
                        out[normalize(r["player"])] = float(r["salary_M"])
                    except ValueError:
                        pass
    except Exception:
        pass
    return out


_OUTCOME_LABEL = {"po_in": "PO Opt In", "po_out": "PO Opt Out",
                  "to_in": "TO Picked Up", "to_out": "TO Declined"}


def _hub_outcome(norm_name: str, status: str | None) -> str | None:
    """Compact resolution string ('PO Opt Out · Signed $41.3M') or None."""
    s = _hub_signings().get(norm_name)
    dec, fig = _hub_decisions().get(norm_name, (None, None))
    if not dec and s:
        if status == "Player Option":
            dec = "po_out"
        elif status == "Team Option":
            dec = "to_out"
    parts = []
    if dec in _OUTCOME_LABEL:
        lbl = _OUTCOME_LABEL[dec]
        if dec in ("po_in", "to_in") and fig:
            lbl += f" · ${fig:.1f}M"
        parts.append(lbl)
    if s:
        parts.append(f"Signed ${s['actual_M']:.1f}M" if s.get("actual_M") is not None else "Signed")
    return " · ".join(parts) if parts else None


# ── Assemble the hub frame (warm caches only) ─────────────────────────────────
_pool = build_ranked_projected(_HUB_SEASON)
_pool = _pool.sort_values("barrett_score", ascending=False).reset_index(drop=True)

_bref_pos = fetch_bref_positions(season_to_espn_year(_HUB_SEASON), cache_v=3)
_pos2k = _ts.load_player_positions()
_nc = fetch_next_year_contracts(season_to_espn_year(_HUB_SEASON), cache_v=7)
_rookies = fetch_rookie_scale_players(_HUB_SEASON)
_pcv_by = _hub_pcv()
_max_norms = {n for n, p in _pcv_by.items() if p.get("is_max")}
_amax_floor = 0.25 * SALARY_CAP_M.get(_NEXT_SEASON, 165.0) - 0.1

_hub_rows = []
for _i, _r in _pool.iterrows():
    _nm = str(_r["Player"])
    _n = normalize(_nm)
    _status = classify_fa_status(_nm, fmt_next_contract(_nm, _nc), _rookies, _HUB_SEASON)
    if _status is None and _n in _hub_signings():
        _status = "Signed"          # tracked 2026 signing that came off the board
    _sg = _hub_signings().get(_n)
    _nx = _nc.get(_n)
    _next_M = (_sg.get("actual_M") if _sg and _sg.get("actual_M") is not None
               else (float(_nx["salary"]) / 1e6 if _nx and _nx.get("salary") else None))
    if _next_M is None:
        # Exercised options carry their figure in the decisions file; last
        # resort is the hand-verified supplement for feed-omitted players.
        _d, _fig = _hub_decisions().get(_n, (None, None))
        if _d in ("po_in", "to_in") and _fig:
            _next_M = float(_fig)
        else:
            _next_M = _hub_salary_supplement().get(_n)
    _hub_rows.append({
        "norm": _n, "Player": _nm, "Team": _r["Team"],
        "Pos": _ts.resolve_position(_nm, _bref_pos.get(_n, ""), _pos2k),
        "Status": _status or "—",
        "Barrett Score": float(_r["barrett_score"]),
        "Salary": float(_r["salary"]) / 1e6,
        "Predicted": (_pcv_by.get(_n) or {}).get("pcv_M"),
        # extra depth for the Right Now quadrant
        "GP": int(_r.get("GP") or 0), "MPG": float(_r.get("MPG") or 0),
        "TS": float(_r.get("ts_pct") or 0), "DLEB": float(_r.get("d_lebron") or 0),
        "SalRank": int(_r.get("salary_rank") or 0),
        "Avail": float(_r.get("avail_mult") or 0),
        "Next": _next_M,
        "ProjValue": float(_r.get("projected_salary") or 0) / 1e6,
        "DeltaMkt": float(_r.get("value_diff") or 0) / 1e6,
    })
_hub_df = pd.DataFrame(_hub_rows)
_hub_df.insert(0, "#", range(1, len(_hub_df) + 1))
_amax_norms = _actual_max_norms(tuple(
    (r["norm"], r["Player"], float(r["Next"])) for r in _hub_rows
    if r.get("Next") and r["Next"] >= _amax_floor))
_by_norm = {r["norm"]: dict(r, rank=i + 1) for i, r in enumerate(_hub_rows)}

_FA_SET = {"UFA", "RFA", "Player Option", "Team Option"}

# ── Front Page strip: four clickable feature cards under the search ───────────
if not _hub_df.empty:
    def _fp_card(kicker: str, name: str, team: str, value_html: str, sub: str) -> str:
        _hx = TEAM_HEX.get(str(team), "")
        _style = f' style="--team:{_hx}"' if _hx else ""
        return (f'<a class="fp-card" href="/?player={_urlquote(str(name))}" target="_top"{_style}>'
                f'{_face_img(str(name), "fp-face")}'
                f'<span style="min-width:0">'
                f'<span class="k" style="display:block">{html.escape(kicker)}</span>'
                f'<span class="nm" style="display:block">{html.escape(str(name))}</span>'
                f'{value_html}'
                f'<span class="sub" style="display:block">{sub}</span>'
                f'</span></a>')

    _r0 = _hub_df.iloc[0]
    _stl = _hub_df[_hub_df["Salary"] >= 2.0]          # keep rookie-min noise off the card
    _stl = _stl.loc[_stl["DeltaMkt"].idxmin()]
    _ovp = _hub_df.loc[_hub_df["DeltaMkt"].idxmax()]
    _fa_df = _hub_df[_hub_df["Status"].isin(_FA_SET)]

    _rail_meta = None
    try:
        _sc = (_json.loads((Path(__file__).parent / "cache" / "accuracy_tracker_v1.json")
                           .read_text()).get("scorecard") or {})
        if _sc.get("n"):
            _rail_meta = f"model: {_sc['within_4M']:.0f}% of {_sc['n']} real 2026 deals within $4M"
    except Exception:
        pass
    _rail("Front page", "Today around the league", meta=_rail_meta)

    _cards = [
        _fp_card("Best right now", _r0["Player"], _r0["Team"],
                 f'<span class="v teal" style="display:block">{_r0["Barrett Score"]:.2f}</span>',
                 f'League #1 · ${_r0["Salary"]:.1f}M salary'),
        _fp_card("Biggest steal", _stl["Player"], _stl["Team"],
                 f'<span class="v good" style="display:block">-${abs(_stl["DeltaMkt"]):.1f}M</span>',
                 f'paid ${_stl["Salary"]:.1f}M · worth ${_stl["ProjValue"]:.1f}M'),
        _fp_card("Most overpaid", _ovp["Player"], _ovp["Team"],
                 f'<span class="v bad" style="display:block">+${_ovp["DeltaMkt"]:.1f}M</span>',
                 f'paid ${_ovp["Salary"]:.1f}M · worth ${_ovp["ProjValue"]:.1f}M'),
    ]
    if len(_fa_df):
        _fa_top = _fa_df.iloc[0]
        _cards.append(
            _fp_card("FA watch · 2026", _fa_top["Player"], _fa_top["Team"],
                     f'<span class="v teal" style="display:block">{_fa_top["Barrett Score"]:.2f}</span>',
                     "best available free agent right now"))
    st.markdown('<div class="fp-grid">' + "".join(_cards) + "</div>", unsafe_allow_html=True)

# ── Selection from ?player= ──────────────────────────────────────────────────
_sel = None
if "player" in st.query_params:
    _sel = _by_norm.get(normalize(st.query_params.get("player", "")))

# ── Hub panel — four quadrants ────────────────────────────────────────────────
# Q1 Right Now (score/rank/contract) · Q2 Career (scores + contracts by season)
# Q3 Similar Today (closest current Barrett Scores) · Q4 Career Twins (closest
# career arcs, all eras). Native bordered containers (st.container(border=True))
# hold each quadrant — raw <div> cards can't wrap Streamlit elements (charts,
# tables): markdown auto-closes them and the layout shatters.
if _sel:
    _n = _sel["norm"]
    _pv = _pcv_by.get(_n) or {}
    _draft = get_player_draft_info(_sel["Player"])
    _draft_txt = (f"{_draft['draft_tier']} · #{_draft['draft_pick']} in {_draft['draft_year']}"
                  if _draft.get("draft_pick") else "Undrafted")
    _outcome = _hub_outcome(_n, _sel["Status"])
    _q = _urlquote(_sel["Player"])
    _pred_txt = ((('<span class="hv-chip max">MAX</span>' if _pv.get("is_max") else "")
                  + f"${_pv['pcv_M']:.1f}M")
                 if _pv.get("pcv_M") is not None else "—")

    _team = str(_sel["Team"])
    _thx = TEAM_HEX.get(_team, "")
    _team_style = f"--team:{_thx};" if _thx else ""
    _wash = _hex_rgba(_thx, 0.08) if _thx else "transparent"
    _rule = _hex_rgba(_thx, 0.35) if _thx else "var(--hairline-soft)"
    # One-line verdict: production rank vs pay rank vs market, plus the FA hook.
    _vbits = []
    if _sel.get("SalRank"):
        _vbits.append(f'Plays like #{_sel["rank"]}, paid like #{_sel["SalRank"]}')
    _vdm = _sel["DeltaMkt"]
    if _vdm <= -3:
        _vbits.append(f"${abs(_vdm):.1f}M below market value")
    elif _vdm >= 3:
        _vbits.append(f"${_vdm:.1f}M over market value")
    else:
        _vbits.append("paid about right")
    if str(_sel["Status"]) in _FA_SET:
        _vbits.append("hits the market in 2026")
    _verdict = " · ".join(_vbits)
    _STATUS_CHIP = {"UFA": ("ufa", "UFA"), "RFA": ("rfa", "RFA"),
                    "Player Option": ("po", "PLAYER OPTION"),
                    "Team Option": ("to", "TEAM OPTION"), "Signed": ("signed", "SIGNED")}
    _chip = _STATUS_CHIP.get(str(_sel["Status"]))
    _status_html = (f'<span class="hv-chip {_chip[0]}">{_chip[1]}</span>' if _chip
                    else (f'<b>{html.escape(str(_sel["Status"]))}</b>'
                          if str(_sel["Status"]) not in ("", "—") else ""))
    if _outcome:
        _status_html = (_status_html + " · " if _status_html else "") + html.escape(_outcome)
    _status_seg = f"&nbsp;·&nbsp; {_status_html}" if _status_html else ""

    st.markdown(f"""
<style>
/* Selected-player masthead: headshot + name + meta, team-color rail + watermark. */
.hub-banner {{ display: flex; align-items: center; gap: 1rem;
  background: linear-gradient(90deg, {_wash}, transparent 45%), var(--panel-solid);
  border: 1px solid var(--panel-line);
  border-left: 4px solid var(--team, var(--accent-teal));
  border-radius: 14px; padding: 0.7rem 1.2rem; box-shadow: var(--shadow-card);
  position: relative; overflow: hidden; margin-bottom: 0.8rem; }}
img.hub-face {{ width: 64px; height: 64px; border-radius: 50%; object-fit: cover;
  object-position: center 12%; background: var(--panel-2);
  border: 2px solid var(--team, var(--panel-line)); flex: 0 0 auto; }}
.hub-banner .nm {{ font-size: 1.6rem; font-weight: 800; letter-spacing: -0.01em;
  color: var(--fg-1); line-height: 1.15; }}
.hub-banner .meta {{ color: var(--fg-3); font-size: 0.85rem; }}
.hub-banner .rank {{ margin-left: auto; text-align: right;
  font-variant-numeric: tabular-nums; position: relative; z-index: 1; }}
.hub-banner .rank .v {{ font-size: 1.5rem; font-weight: 800; color: var(--accent-teal); }}
.hub-banner .rank .l {{ display: block; font-size: 0.62rem; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--fg-4); }}
.hub-banner::after {{ content: attr(data-team); position: absolute; right: 4.5rem;
  top: 50%; transform: translateY(-50%); font-family: "Space Grotesk", sans-serif;
  font-size: 4.2rem; font-weight: 800; letter-spacing: -0.04em;
  color: var(--team, var(--accent-teal)); opacity: 0.07; pointer-events: none; }}
/* Skin the four quadrant containers like themed cards (scoped by key so other
   bordered containers on the page stay native). Streamlit's st-key class can
   land on the wrapper itself or a descendant depending on version: cover both. */
[data-testid="stLayoutWrapper"]:has(> .st-key-hub_q1),
[data-testid="stLayoutWrapper"]:has(> .st-key-hub_q2),
[data-testid="stLayoutWrapper"]:has(> .st-key-hub_q3),
[data-testid="stLayoutWrapper"]:has(> .st-key-hub_q4),
[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-hub_q1),
[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-hub_q2),
[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-hub_q3),
[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-hub_q4) {{
  background: var(--panel-solid); border: 1px solid var(--panel-line) !important;
  border-radius: 14px !important; box-shadow: var(--shadow-card);
  height: 580px !important; max-height: 580px !important;
  overflow-y: auto !important; padding: 0.55rem 0.8rem 3.3rem !important;
  position: relative; margin-bottom: 0.9rem; }}
/* Jump buttons pin to the bottom-left of every quadrant card. Streamlit's
   element containers are positioned, which would capture the absolute button,
   so the containers on the button's ancestor chain go static. */
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-hub_q"]) [data-testid="stElementContainer"]:has(.hub-go),
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-hub_q"]) [data-testid="stMarkdown"]:has(.hub-go),
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-hub_q"]) [data-testid="stMarkdownContainer"]:has(.hub-go) {{
  position: static !important; }}
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-hub_q"]) .hub-go {{
  position: absolute; left: 0.9rem; bottom: 0.7rem; z-index: 2; margin: 0; }}
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-hub_q"]) .hub-go a {{
  margin-top: 0; }}
/* Per-quadrant identity accents. */
[data-testid="stLayoutWrapper"]:has(> .st-key-hub_q1),
[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-hub_q1) {{
  border-top: 3px solid var(--accent-teal) !important; }}
[data-testid="stLayoutWrapper"]:has(> .st-key-hub_q2),
[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-hub_q2) {{
  border-top: 3px solid var(--logo-copper) !important; }}
[data-testid="stLayoutWrapper"]:has(> .st-key-hub_q3),
[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-hub_q3) {{
  border-top: 3px solid var(--sky) !important; }}
[data-testid="stLayoutWrapper"]:has(> .st-key-hub_q4),
[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-hub_q4) {{
  border-top: 3px solid var(--logo-sage) !important; }}
[data-testid="stVerticalBlockBorderWrapper"] .hv-table-wrap,
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-hub_q"]) .hv-table-wrap {{ margin: 0.3rem 0 0.5rem; }}
.hub-qh {{ display: flex; align-items: center; gap: 0.5rem; font-size: 0.7rem;
  font-weight: 800; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--fg-4); margin-bottom: 0.35rem; }}
.hub-qh::after {{ content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, {_rule}, var(--hairline-soft)); }}
.hub-qh b {{ color: var(--accent-teal); }}
.hub-stats {{ display: flex; gap: 1.7rem; flex-wrap: wrap; margin-top: 0.3rem; }}
.hub-stat .v {{ font-size: 1.55rem; font-weight: 800; color: var(--fg-1); }}
.hub-stat .l {{ font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--fg-4); font-weight: 600; margin-top: 0.1rem; }}
.hub-ladder {{ margin-top: 1.4rem; }}
.hub-ladder .lrow {{ display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.28rem; }}
.hub-ladder .ll {{ width: 66px; font-size: 0.64rem; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--fg-4); font-weight: 700; flex: 0 0 auto; }}
.hub-ladder .lbar {{ flex: 1; height: 13px; background: var(--hairline-soft);
  border-radius: 7px; overflow: hidden; position: relative; }}
.hub-ladder .lbar .tickm {{ position: absolute; top: -1px; bottom: -1px; width: 2px;
  background: var(--fg-3); opacity: 0.85; z-index: 2; border-radius: 1px; }}
.hub-ladder .lbar div {{ height: 100%; border-radius: 7px; }}
.hub-ladder .lbar .lband {{ position: absolute; top: 0; height: 100%;
  background: var(--bar-tint); border-radius: 0; }}
.hub-ladder .lv {{ width: 72px; text-align: right; font-size: 0.78rem; font-weight: 700;
  color: var(--fg-2); font-variant-numeric: tabular-nums; flex: 0 0 auto; line-height: 1.15; }}
.hub-ladder .lv .avg {{ display: block; font-size: 0.6rem; font-weight: 600;
  color: var(--fg-5); }}
.hub-note {{ color: var(--fg-3); font-size: 0.78rem; margin-top: 0.8rem; }}
.hub-go a {{ display: inline-block; margin-top: 0.6rem; background: var(--panel);
  border: 1px solid var(--panel-line); border-radius: 9px; padding: 0.4rem 0.85rem;
  font-size: 0.8rem; font-weight: 700; color: var(--sky); text-decoration: none; }}
.hub-go a:hover {{ border-color: var(--sky); }}
.hv-plink {{ color: var(--sky); font-weight: 700; text-decoration: none; }}
.hv-plink:hover {{ text-decoration: underline; }}
</style>
<div class="hub-banner" style="{_team_style}" data-team="{html.escape(_team, quote=True)}">
  {_face_img(_sel["Player"], "hub-face")}
  <span style="min-width:0">
    <span class="nm" style="display:block">{html.escape(_sel["Player"])}</span>
    <span class="meta">{html.escape(_team)} · {html.escape(str(_sel["Pos"]))} · {html.escape(_draft_txt)}
{_status_seg}</span>
    <span style="display:block;font-style:italic;color:var(--fg-3);font-size:0.82rem;margin-top:2px">{html.escape(_verdict)}</span>
  </span>
  <span class="rank"><span class="v">#{_sel["rank"]}</span><span class="l">2025-26 rank</span></span>
</div>
""", unsafe_allow_html=True)

    _car = _hub_career()
    _mine = _car[_car["norm"] == _n].sort_values("Season")

    _left, _right = st.columns(2)

    with _left:   # ── Quadrant 1: Right Now ─────────────────────────────────────
        # Salary vs market-anchor verdict (value_diff = actual − projected; negative = underpaid)
        _dm = _sel["DeltaMkt"]
        _d_color = "var(--value-good)" if _dm < 0 else ("var(--value-bad)" if _dm > 0 else "var(--fg-2)")
        _d_txt = f"{'+' if _dm > 0 else '−' if _dm < 0 else ''}${abs(_dm):.1f}M"
        _d_lbl = "Underpaid" if _dm < 0 else ("Overpaid" if _dm > 0 else "At market")
        _ci = get_player_contract_info(_sel["Player"]) or {}
        _deal_line = (f'<div class="hub-note">Current deal runs through <b>{html.escape(str(_ci["end_season"]))}</b>'
                      f' · next contract window <b>{html.escape(str(_ci.get("signing_season") or "now"))}</b>.</div>'
                      if _ci.get("end_season") else
                      '<div class="hub-note">No future salary on the books · signing his next deal now.</div>')
        _bx = _hub_counting().get(_n)
        # Position profile: each stat as a percentile bar among the player's
        # primary-position peers this season (tick on the track = median).
        _pos_prim = str(_sel["Pos"]).split("/")[0].strip()
        _bx_all = _hub_counting()

        # Rotation-minutes floor: low-minute players (two-ways, garbage time)
        # distort the min/avg/max endpoints, especially on rate stats where a
        # 5-minute fluke sets an unreachable TS% ceiling. Peers must clear
        # _POS_MPG_MIN; if that leaves too thin a pool, fall back to everyone.
        _POS_MPG_MIN = 15.0
        _pos_peers = [_r0 for _r0 in _hub_rows
                      if str(_r0["Pos"]).split("/")[0].strip() == _pos_prim
                      and float(_r0.get("MPG") or 0) >= _POS_MPG_MIN]
        if len(_pos_peers) < 12:
            _pos_peers = [_r0 for _r0 in _hub_rows
                          if str(_r0["Pos"]).split("/")[0].strip() == _pos_prim]

        def _pos_pool(idx=None, col=None):
            _vals = []
            for _r0 in _pos_peers:
                if col is not None:
                    _vals.append(float(_r0[col]))
                else:
                    _b0 = _bx_all.get(_r0["norm"])
                    if _b0:
                        _vals.append(float(_b0[idx]))
            return _vals

        # Center of the position track (the tick + the small number). Flip
        # _CENTER_STAT between "median" and "mean" to switch the whole profile.
        _CENTER_STAT = "median"
        _center_lbl = "med" if _CENTER_STAT == "median" else "avg"
        _center_word = "median" if _CENTER_STAT == "median" else "average"

        def _avg_of(pool):
            if not pool:
                return None
            if _CENTER_STAT == "median":
                _s = sorted(pool); _m = len(_s)
                return _s[_m // 2] if _m % 2 else (_s[_m // 2 - 1] + _s[_m // 2]) / 2
            return sum(pool) / len(pool)

        def _scale_of(v, pool):
            """0 = position low, 100 = position high, 50 = position center
            (mean or median per _CENTER_STAT). Each half is linear, so
            bar-vs-tick reads as above/below the center player."""
            if not pool:
                return 50.0
            _lo, _hi = min(pool), max(pool)
            _av = _avg_of(pool)
            if v >= _hi:
                return 100.0
            if v <= _lo:
                return 0.0
            if v <= _av:
                return 50.0 * (v - _lo) / (_av - _lo) if _av > _lo else 50.0
            return 50.0 + 50.0 * (v - _av) / (_hi - _av) if _hi > _av else 50.0

        _prof = []
        if _bx:
            for _lbl, _idx in [("Points", 0), ("Rebounds", 1), ("Assists", 2),
                               ("Steals", 3), ("Blocks", 4)]:
                _v = float(_bx[_idx])
                _pl = _pos_pool(idx=_idx)
                _av = _avg_of(_pl)
                _prof.append((_lbl, f"{_v:.1f}", _scale_of(_v, _pl),
                              f"{_av:.1f}" if _av is not None else ""))
        _ts_pool = _pos_pool(col="TS")
        _ts_avg = _avg_of(_ts_pool)
        _prof.append(("TS%", f"{_sel['TS'] * 100:.1f}%", _scale_of(float(_sel["TS"]), _ts_pool),
                      f"{_ts_avg * 100:.1f}%" if _ts_avg is not None else ""))
        _dl_pool = _pos_pool(col="DLEB")
        _dl_avg = _avg_of(_dl_pool)
        _prof.append(("D-LEBRON", f"{_sel['DLEB']:+.1f}", _scale_of(float(_sel["DLEB"]), _dl_pool),
                      f"{_dl_avg:+.1f}" if _dl_avg is not None else ""))
        _bar_c = _thx or "var(--accent-teal)"
        _ladder = "".join(
            f'<div class="lrow"><span class="ll">{_l}</span>'
            f'<div class="lbar"><span class="tickm" style="left:50%"></span>'
            f'<div style="width:{max(2.0, _p):.0f}%;background:{_bar_c};position:relative;z-index:0"></div></div>'
            f'<span class="lv">{_vtxt}'
            + (f'<span class="avg">{_center_lbl} {_atxt}</span>' if _atxt else "")
            + '</span></div>'
            for _l, _vtxt, _p, _atxt in _prof)
        _avail_txt = f"{_sel['Avail'] * 100:.0f}%" if _sel.get("Avail") else "—"
        _salrank_txt = f"#{_sel['SalRank']}" if _sel.get("SalRank") else "—"
        # Named anchors: who owns the paycheck at his production rank, and who
        # produces at his pay rank. Turns the abstract footnote into people.
        _anchor_note = ""
        try:
            _mk = _hub_df[_hub_df["SalRank"] == _sel["rank"]]
            _pd_ = _hub_df.iloc[_sel["SalRank"] - 1] if 0 < _sel["SalRank"] <= len(_hub_df) else None
            _mk_nm = str(_mk.iloc[0]["Player"]) if len(_mk) else ""
            _pd_nm = str(_pd_["Player"]) if _pd_ is not None else ""
            _bits = []
            if _mk_nm and normalize(_mk_nm) != _n:
                _bits.append(f"market value = the #{_sel['rank']} paycheck ({html.escape(_mk_nm)}'s money)")
            if _pd_nm and normalize(_pd_nm) != _n:
                _bits.append(f"his own salary is #{_sel['SalRank']}: {html.escape(_pd_nm)} territory")
            if _bits:
                _anchor_note = '<div class="hub-note">In names: ' + " · ".join(_bits) + ".</div>"
        except Exception:
            _anchor_note = ""
        with st.container(border=True, key="hub_q1"):
            st.markdown(f"""
<div class="hub-qh">Right now · <b>2025-26</b></div>
<div class="hub-stats" style="margin-top:0.7rem">
  <div class="hub-stat"><div class="v" style="color:var(--accent-teal)">{_sel["Barrett Score"]:.2f}</div><div class="l">Barrett Score</div></div>
  <div class="hub-stat"><div class="v" style="color:var(--accent-teal)">#{_sel["rank"]}</div><div class="l">League rank</div></div>
  <div class="hub-stat"><div class="v">{_sel["GP"]} · {_sel["MPG"]:.1f}</div><div class="l">GP · MPG</div></div>
  <div class="hub-stat"><div class="v" style="color:var(--accent-teal)">{_pred_txt}</div><div class="l" title="The model's projection for a NEW deal signed today, at next season's cap">Predicted contract</div></div>
</div>
<div class="hub-ladder">{_ladder}</div>
<div class="hub-note" style="margin-top:0.3rem">Track runs lowest to highest {html.escape(_pos_prim)} this season · middle tick = the {_center_word} {html.escape(_pos_prim)} (small number).</div>
<div class="hub-stats" style="margin-top:1.1rem">
  <div class="hub-stat"><div class="v">${_sel["Salary"]:.1f}M</div><div class="l">Salary</div></div>
  <div class="hub-stat"><div class="v" style="color:{_d_color}">{_d_txt}</div><div class="l">{_d_lbl} vs market</div></div>
  <div class="hub-stat"><div class="v">{_salrank_txt}</div><div class="l">Salary rank · paid like</div></div>
  <div class="hub-stat"><div class="v" style="color:{'var(--value-good)' if _sel.get("Avail", 0) >= 0.85 else ('var(--value-bad)' if 0 < _sel.get("Avail", 0) < 0.6 else 'var(--fg-1)')}">{_avail_txt}</div><div class="l">Availability</div></div>
</div>
{_deal_line}
{_anchor_note}
<div class="hub-go"><a href="/Contract_Predictor?player={_q}" target="_top">Full contract prediction →</a></div>
""", unsafe_allow_html=True)

    with _right:   # ── Quadrant 2: Career ────────────────────────────────────────
        with st.container(border=True, key="hub_q2"):
            st.markdown('<div class="hub-qh">Career · <b>scores & contracts</b></div>',
                        unsafe_allow_html=True)
            if len(_mine) >= 2:
                # Team-colored score line over muted salary bars: one graphic,
                # the whole pay-vs-production story. Plotly can't read CSS vars,
                # so tints come from TEAM_HEX server-side; light golds (DEN/IND/
                # UTA/NOP) get darkened for line contrast.
                _chex = _thx or "#7ec8e8"
                _line_hex = _hex_darken(_chex, 0.72) if _hex_is_light(_chex) else _chex
                _pk = _mine.loc[_mine["barrett_score"].idxmax()]
                _fig = go.Figure()
                _fig.add_bar(x=_mine["Season"], y=_mine["salary"] / 1e6, yaxis="y2",
                             marker_color=_hex_rgba(_chex, 0.22),
                             hovertemplate="$%{y:.1f}M<extra></extra>")
                _base = float(_mine["barrett_score"].min()) - 2
                _fig.add_scatter(x=_mine["Season"], y=[_base] * len(_mine),
                                 mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False)
                _fig.add_scatter(x=_mine["Season"], y=_mine["barrett_score"],
                                 mode="lines+markers",
                                 line=dict(color=_line_hex, width=2.5),
                                 marker=dict(color=_line_hex, size=5),
                                 fill="tonexty", fillcolor=_hex_rgba(_chex, 0.10),
                                 hovertemplate="%{y:.2f}<extra></extra>")
                _fig.add_scatter(x=[_pk["Season"]], y=[_pk["barrett_score"]],
                                 mode="markers",
                                 marker=dict(color="#f1c40f", size=11,
                                             line=dict(color="#b8860b", width=1)),
                                 hovertemplate="Peak: %{y:.2f}<extra></extra>")
                _fig.update_layout(
                    height=175, margin=dict(t=6, b=6, l=6, r=6), showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis2=dict(overlaying="y", side="right", showticklabels=False,
                                showgrid=False, rangemode="tozero",
                                range=[0, float(_mine["salary"].max()) / 1e6 * 3.2]),
                )
                # Season strings like "2003-04" parse as YYYY-MM dates, so plotly
                # infers a DATE axis from early-career values and silently drops
                # "2012-13"+ (months 13-26 do not exist). Force categories; thin
                # the tick labels for long careers so they do not collide.
                _fig.update_xaxes(type="category",
                                  dtick=max(1, len(_mine) // 8),
                                  tickfont=dict(size=9))
                st.plotly_chart(theme_fig(_fig), use_container_width=True,
                                config={"displayModeBar": False})
            _ct = _mine.sort_values("Season", ascending=False)[
                ["Season", "barrett_score", "score_rank", "salary"]].copy()
            _ct.columns = ["Season", "Score", "Rank", "Salary"]
            _ct["Salary"] = _ct["Salary"] / 1e6
            _pk_score = float(_ct["Score"].max()) if len(_ct) else 0.0
            html_table(
                _ct,
                formatters={"Score": lambda v: f"{v:.2f}",
                            "Rank": lambda v: f"#{int(v)}" if v == v else "—",
                            "Salary": lambda v: f"${v:.1f}M"},
                styles={"Score": lambda v, _r: ("color:var(--gold);font-weight:800"
                                                if v == _pk_score else "")},
                row_style=lambda rd: ("background:rgba(241,196,15,0.07)"
                                      if rd.get("Score") == _pk_score else ""),
                aligns={"Score": "right", "Rank": "right", "Salary": "right"},
                numeric={"Score", "Rank", "Salary"},
                height=216,
            )
            st.markdown(f'<div class="hub-go"><a href="/Search?player={_q}" target="_top">'
                        f'Full profile & career →</a></div>', unsafe_allow_html=True)

    with _left:   # ── Quadrant 3: Similar Today ────────────────────────────────
        with st.container(border=True, key="hub_q3"):
            st.markdown('<div class="hub-qh">Similar players · <b>today</b></div>',
                        unsafe_allow_html=True)
            _sim = (_hub_df.assign(_d=(_hub_df["Barrett Score"] - _sel["Barrett Score"]).abs())
                    .loc[lambda d: d["norm"] != _n]
                    .nsmallest(10, "_d"))
            # Pin the selected player into the list so the comparison has an anchor.
            _sim = (pd.concat([_sim, _hub_df[_hub_df["norm"] == _n]])
                    .sort_values("Barrett Score", ascending=False))
            _sim_view = _sim[["#", "Player", "Team", "Barrett Score", "Salary", "Predicted"]].copy()
            _s_lo = float(_sim_view["Barrett Score"].min())
            _s_hi = float(_sim_view["Barrett Score"].max())
            _s_rng = (_s_hi - _s_lo) or 1.0

            def _sim_player_cell(v, r):
                _tm = str(r.get("Team", ""))
                _ring = TEAM_HEX.get(_tm, "")
                _ring_st = f' style="box-shadow:inset 0 0 0 2px {_ring}"' if _ring else ""
                return (f'<span class="hv-mini-wrap"{_ring_st}>{_face_img(str(v), "hv-mini-face")}</span>'
                        f'<a class="hv-plink" href="/?player={_urlquote(str(v))}" '
                        f'target="_top">{html.escape(str(v))}</a>')

            html_table(
                _sim_view,
                formatters={
                    "Player": _sim_player_cell,
                    "Team": lambda v: (f'<span class="tdot tdot-{html.escape(str(v), quote=True)}"></span>'
                                       f'{html.escape(str(v))}'),
                    "Barrett Score": lambda v: f"{v:.2f}",
                    "Salary": lambda v: f"${v:.1f}M",
                    "Predicted": lambda v, r: ("—" if v is None or (isinstance(v, float) and v != v)
                                               else ('<span class="hv-chip max">MAX</span>'
                                                     if normalize(str(r.get("Player", ""))) in _max_norms else "")
                                                    + f"${v:.1f}M"),
                },
                raw={"Player", "Team", "Predicted"},
                styles={
                    "Barrett Score": lambda v, _r: (
                        f"background:linear-gradient(90deg,var(--bar-tint) "
                        f"{15 + (v - _s_lo) / _s_rng * 85:.0f}%,transparent 0)"),
                    "Predicted": lambda v, _r: ("color:var(--fg-6)" if v is None or (isinstance(v, float) and v != v)
                                                else "color:var(--accent-teal)"),
                },
                row_style=lambda rd: ((f"background:{_hex_rgba(_thx, 0.10)};font-weight:600" if _thx
                                       else "background:var(--panel-hover);font-weight:600")
                                      if normalize(str(rd.get("Player", ""))) == _n else ""),
                aligns={"#": "right", "Barrett Score": "right", "Salary": "right", "Predicted": "right"},
                numeric={"#", "Barrett Score", "Salary", "Predicted"},
                height=404,
            )
            st.markdown('<div class="hub-note">Closest current Barrett Scores in the 2025-26 pool.</div>'
                        '<div class="hub-go"><a href="/Rankings" target="_top">Full current rankings →</a></div>',
                        unsafe_allow_html=True)

    with _right:   # ── Quadrant 4: Career Twins (all eras) ──────────────────────
        with st.container(border=True, key="hub_q4"):
            st.markdown('<div class="hub-qh">Career twins · <b>1973 → today</b></div>',
                        unsafe_allow_html=True)
            _agg = _hub_career_agg()
            if not _agg.empty and (_agg["norm"] == _n).any():
                _me = _agg[_agg["norm"] == _n].iloc[0]
                _tw = (_agg[(_agg["norm"] != _n) & (_agg["yrs"] >= 3)]
                       .assign(_d=lambda d: (d["avg"] - _me["avg"]).abs())
                       .nsmallest(10, "_d"))
                # Pin the selected player among his twins.
                _tw = (pd.concat([_tw, _agg[_agg["norm"] == _n]])
                       .sort_values("avg", ascending=False))
                # Career-shape sparklines, from the same parquet (one groupby pass).
                _arc_norms = set(_tw["norm"])
                _arcs = {nm: _spark_svg(g.sort_values("Season")["barrett_score"].tolist())
                         for nm, g in _car[_car["norm"].isin(_arc_norms)].groupby("norm")}
                _tw_view = _tw[["Player", "norm", "avg", "peak", "best_rank", "top_sal"]].copy()
                _tw_view["Arc"] = _tw_view["norm"].map(_arcs).fillna("")
                _tw_view = _tw_view.drop(columns=["norm"])[
                    ["Player", "Arc", "avg", "peak", "best_rank", "top_sal"]]
                _tw_view.columns = ["Player", "Arc", "Avg Score", "Peak", "Best Rank", "Top Salary"]
                _tw_view["Top Salary"] = _tw_view["Top Salary"] / 1e6
                _t_lo = float(_tw_view["Avg Score"].min())
                _t_rng = (float(_tw_view["Avg Score"].max()) - _t_lo) or 1.0
                html_table(
                    _tw_view,
                    formatters={
                        "Player": lambda v: (f'<span class="hv-mini-wrap">{_face_img(str(v), "hv-mini-face")}</span>'
                                             f'<a class="hv-plink" href="/Search?player={_urlquote(str(v))}" '
                                             f'target="_top">{html.escape(str(v))}</a>'),
                        "Avg Score": lambda v: f"{v:.2f}",
                        "Peak": lambda v: f"{v:.2f}",
                        "Best Rank": lambda v: f"#{int(v)}" if v == v else "—",
                        "Top Salary": lambda v: f"${v:.1f}M",
                    },
                    raw={"Player", "Arc"},
                    styles={
                        "Avg Score": lambda v, _r: (
                            f"background:linear-gradient(90deg,var(--bar-tint) "
                            f"{15 + (v - _t_lo) / _t_rng * 85:.0f}%,transparent 0)"),
                    },
                    row_style=lambda rd: ((f"background:{_hex_rgba(_thx, 0.10)};font-weight:600" if _thx
                                           else "background:var(--panel-hover);font-weight:600")
                                          if normalize(str(rd.get("Player", ""))) == _n else ""),
                    aligns={"Avg Score": "right", "Peak": "right", "Best Rank": "right",
                            "Top Salary": "right"},
                    numeric={"Avg Score", "Peak", "Best Rank", "Top Salary"},
                    height=404,
                )
                st.markdown(f'<div class="hub-note">Closest career averages, all eras: '
                            f'{html.escape(_sel["Player"])} at {_me["avg"]:.2f} over '
                            f'{int(_me["yrs"])} season{"s" if _me["yrs"] != 1 else ""}.</div>'
                            f'<div class="hub-go"><a href="/Legacy" target="_top">Compare eras in Legacy →</a></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="hub-note">No career history on file yet.</div>',
                            unsafe_allow_html=True)

# ── The board ─────────────────────────────────────────────────────────────────
# Rail header + quick-filter pills + the value-coded table, all in one fragment
# so a filter click re-renders only the board (the sort script is document-
# delegated and survives fragment re-renders).
_BOARD_VIEWS = ["All", "Bargains", "Overpays", "Free agents", "Max tier"]
_SCORE_MAX = float(_hub_df["Barrett Score"].max() or 0) or 1.0

st.markdown("""
<style>
.st-key-board_view [data-testid="stButtonGroup"] button{border-radius:999px;
    font-weight:700;font-size:.78rem;background:var(--panel-solid);
    border:1px solid var(--panel-line);color:var(--fg-3);}
.st-key-board_view [data-testid="stButtonGroup"] button:hover{color:var(--fg-1);
    border-color:var(--fg-5);}
.st-key-board_view [data-testid="stButtonGroup"] button[kind="pillsActive"]{
    color:var(--accent-teal);border-color:var(--accent-teal);
    background:var(--panel-hover);}
.st-key-board_view [data-testid="stButtonGroup"] button p{font-size:.78rem !important;}
.hv-plink{color:var(--sky);font-weight:700;text-decoration:none}
.hv-plink:hover{text-decoration:underline}
/* Streamlit 1.51 puts margin-bottom:-16px on every stMarkdownContainer, which
   swallows section spacing under the zeroed root gap. Neutralize it for the
   page-level rhythm blocks only (rails + card grid). */
[data-testid="stMarkdownContainer"]:has(.hv-rail),
[data-testid="stMarkdownContainer"]:has(.fp-grid),
[data-testid="stMarkdownContainer"]:has(.hub-banner){margin-bottom:0 !important;}
/* Board fragment: its inner block keeps the default 1rem gap (1.51 nests the
   columns row under a stLayoutWrapper). Tighten pills-to-table; the :not
   guard keeps the rule off the root block, whose gap must stay 0. */
[data-testid="stVerticalBlock"]:not(:has(.st-key-theme_nav_toggle)):has(> [data-testid="stLayoutWrapper"] .st-key-board_view){
    gap:0.35rem !important;}
[data-testid="stVerticalBlock"]:not(:has(.st-key-theme_nav_toggle)):has(> [data-testid="stLayoutWrapper"] .st-key-board_view) .hv-table-wrap{
    margin-top:0.25rem;}
</style>
""", unsafe_allow_html=True)


@st.fragment
def _board():
    _pick = st.session_state.get("board_view") or "All"
    if _pick == "Bargains":
        _df = _hub_df[_hub_df["DeltaMkt"] <= -5].sort_values("DeltaMkt")
    elif _pick == "Overpays":
        _df = _hub_df[_hub_df["DeltaMkt"] >= 5].sort_values("DeltaMkt", ascending=False)
    elif _pick == "Free agents":
        _df = _hub_df[_hub_df["Status"].isin(_FA_SET)]
    elif _pick == "Max tier":
        _df = _hub_df[_hub_df["norm"].isin(_max_norms)]
    else:
        _df = _hub_df
    _rail("The board", "2025-26 Player Board", count=f"{len(_df)} players")
    st.pills("View", _BOARD_VIEWS, default="All", key="board_view",
             label_visibility="collapsed")

    _view = _df
    _view = _view[["#", "Player", "Team", "Pos", "Barrett Score", "Salary",
                   "ProjValue", "DeltaMkt", "Predicted", "Next"]].rename(columns={
        "Barrett Score": "2025-26 Barrett Score", "Salary": "2025-26 Salary",
        "ProjValue": "2025-26 Value", "DeltaMkt": "+/-",
        "Predicted": "2026-27 Predicted", "Next": "Actual 2026-27 Salary"})
    html_table(
        _view,
        formatters={
            "Player": lambda v: (f'<a class="hv-plink" href="/?player={_urlquote(str(v))}" '
                                 f'target="_top">{html.escape(str(v))}</a>'),
            "Team": lambda v: (f'<span class="tdot tdot-{html.escape(str(v), quote=True)}"></span>'
                               f'{html.escape(str(v))}'),
            "2025-26 Barrett Score": lambda v: f"{v:.2f}",
            "2025-26 Salary": lambda v: f"${v:.2f}M",
            "2025-26 Value": lambda v: ("—" if v is None or (isinstance(v, float) and v != v) or v == 0
                                        else f"${v:.2f}M"),
            "Actual 2026-27 Salary": lambda v, r: ("—" if v is None or (isinstance(v, float) and v != v) or v == 0
                                                   else ('<span class="hv-chip max">MAX</span>'
                                                         if normalize(str(r.get("Player", ""))) in _amax_norms else "")
                                                        + f"${v:.2f}M"),
            "+/-": lambda v: ("—" if v is None or (isinstance(v, float) and v != v) or v == 0
                              else ("-" if v < 0 else "+") + f"${abs(v):.1f}M"),
            "2026-27 Predicted": lambda v, r: ("—" if v is None or (isinstance(v, float) and v != v)
                                               else ('<span class="hv-chip max">MAX</span>'
                                                     if normalize(str(r.get("Player", ""))) in _max_norms else "")
                                                    + f"${v:.1f}M"),
        },
        raw={"Player", "Team", "2026-27 Predicted", "Actual 2026-27 Salary"},
        styles={
            "2025-26 Barrett Score": lambda v, _r: (
                "" if v is None or (isinstance(v, float) and v != v) else
                f"background:linear-gradient(90deg,var(--bar-tint) "
                f"{max(4, min(100, v / _SCORE_MAX * 100)):.0f}%,transparent 0)"),
            "+/-": lambda v, _r: ("color:var(--fg-6)" if v is None or (isinstance(v, float) and v != v) or v == 0
                                  else ("color:var(--value-good);font-weight:700" if v < 0
                                        else "color:var(--value-bad);font-weight:700")),
            "2026-27 Predicted": lambda v, _r: ("color:var(--fg-6)" if v is None or (isinstance(v, float) and v != v)
                                                else "color:var(--accent-teal)"),
        },
        aligns={"#": "right", "2025-26 Barrett Score": "right", "2025-26 Salary": "right",
                "2025-26 Value": "right", "+/-": "right", "2026-27 Predicted": "right",
                "Actual 2026-27 Salary": "right"},
        numeric={"#", "2025-26 Barrett Score", "2025-26 Salary", "2025-26 Value",
                 "+/-", "2026-27 Predicted", "Actual 2026-27 Salary"},
        helps={
            "2025-26 Barrett Score": "Base Score × Availability Multiplier. Higher = more valuable.",
            "2025-26 Value": "Market value from Current Rankings: the salary of the player at the same Barrett Score rank this season.",
            "+/-": "2025-26 Salary minus 2025-26 Value. Negative = underpaid.",
            "2026-27 Predicted": "The Contract Predictor's model projection: what a NEW deal signed today would pay, at next season's cap.",
            "Actual 2026-27 Salary": "What is really on the books for 2026-27: existing contract, exercised option, or the actual new deal signed this summer. Blank = nothing signed yet.",
        },
        height=min(760, max(260, len(_view) * 38 + 46)),
    )


_board()


# ── Footer ────────────────────────────────────────────────────────────────────
from utils import render_footer
render_footer()
