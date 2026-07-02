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
        padding-top: 0.6rem !important;
        padding-bottom: 1rem !important;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
        max-width: 1100px;
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

    /* Landing-page hero cards (Best Right Now / Biggest Steal / Most Overpaid) */
    .home-hero-card {
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        text-align: center;
        height: 100%;
        backdrop-filter: blur(2px);
    }
    .hh-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: .09em; opacity: .65; margin-bottom: .25rem; color: var(--fg-1); }
    .hh-name  { font-size: 1.15rem; font-weight: 800; line-height: 1.2; color: var(--fg-1); }
    .hh-sub   { font-size: 0.78rem; margin-top: .35rem; opacity: .75; color: var(--fg-1); }
</style>
""", unsafe_allow_html=True)

# Theme button pinned top-right via CSS. Playoff mode is NOT shown here — it
# only appears on pages where it changes the content.
with st.container(key="theme_nav_toggle"):
    render_theme_toggle()

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
.hv-logo-wrap{display:flex;justify-content:center;padding:0.4rem 0 0.1rem;}
.hv-logo{display:inline-flex;flex-direction:column;align-items:center;font-size:60px;gap:3px;user-select:none}
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
<div style="text-align:center; padding: 0 0 0.6rem 0;">
    <div style="font-size:0.88rem; color:var(--fg-2); max-width:760px; margin:0.4rem auto 0; line-height:1.45;">
        Every NBA player since 1973, ranked by the <b style="color:var(--fg-1);">Barrett Score</b>. On-court production sized up against every paycheck.<br><span style="color:var(--fg-2);">Compare any two eras · find the steals · expose the overpays · settle the GOAT debate.</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Search hero ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSelectbox"][data-baseweb] div[role="combobox"] {
    background: var(--panel) !important;
    border: 2px solid rgba(126, 200, 232, 0.55) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(6px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}
.home-search-label {
    font-size: 0.78rem;
    color: var(--fg-2);
    text-align: center;
    margin-bottom: 0.35rem;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

_, _search_col, _ = st.columns([1, 2, 1])
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

from utils import (
    CACHE_DIR, html_table, theme_fig, get_player_draft_info,
    fetch_bref_positions,
)
import team_suitors as _ts

_HUB_SEASON = SEASONS[0]


@st.cache_data(ttl=3600, show_spinner=False)
def _hub_pcv() -> dict:
    """Full-pool predicted contracts (scripts/build_player_hub.py). {} if absent."""
    try:
        return _json.loads((CACHE_DIR / "player_hub_pcv_v1.json").read_text()).get("players", {})
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _hub_signings() -> dict:
    """Real 2026 signings keyed by normalized name (accuracy tracker cache)."""
    try:
        d = _json.loads((CACHE_DIR / "accuracy_tracker_v1.json").read_text())
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
    """Per-season Barrett Scores for sparklines (combined all-seasons parquet)."""
    try:
        df = pd.read_parquet(CACHE_DIR / "all_seasons_0_v7.parquet",
                             columns=["Player", "Season", "barrett_score"])
        df["norm"] = df["Player"].map(normalize)
        return df
    except Exception:
        return pd.DataFrame(columns=["Player", "Season", "barrett_score", "norm"])


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

_hub_rows = []
for _i, _r in _pool.iterrows():
    _nm = str(_r["Player"])
    _n = normalize(_nm)
    _status = classify_fa_status(_nm, fmt_next_contract(_nm, _nc), _rookies, _HUB_SEASON)
    if _status is None and _n in _hub_signings():
        _status = "Signed"          # tracked 2026 signing that came off the board
    _hub_rows.append({
        "norm": _n, "Player": _nm, "Team": _r["Team"],
        "Pos": _ts.resolve_position(_nm, _bref_pos.get(_n, ""), _pos2k),
        "Status": _status or "—",
        "Barrett Score": float(_r["barrett_score"]),
        "Salary": float(_r["salary"]) / 1e6,
        "Predicted": (_pcv_by.get(_n) or {}).get("pcv_M"),
    })
_hub_df = pd.DataFrame(_hub_rows)
_hub_df.insert(0, "#", range(1, len(_hub_df) + 1))
_by_norm = {r["norm"]: dict(r, rank=i + 1) for i, r in enumerate(_hub_rows)}

# ── Selection from ?player= ──────────────────────────────────────────────────
_sel = None
if "player" in st.query_params:
    _sel = _by_norm.get(normalize(st.query_params.get("player", "")))

# ── Hub panel ─────────────────────────────────────────────────────────────────
if _sel:
    _n = _sel["norm"]
    _pv = _pcv_by.get(_n) or {}
    _draft = get_player_draft_info(_sel["Player"])
    _draft_txt = (f"{_draft['draft_tier']} · #{_draft['draft_pick']} in {_draft['draft_year']}"
                  if _draft.get("draft_pick") else "Undrafted")
    _outcome = _hub_outcome(_n, _sel["Status"])
    _status_txt = _sel["Status"]
    _pred_txt = (f"${_pv['pcv_M']:.1f}M" if _pv.get("pcv_M") is not None else "—")
    _band_txt = (f"${_pv['low_M']:.0f}–{_pv['high_M']:.0f}M"
                 if _pv.get("pcv_M") is not None and _pv.get("low_M") is not None else "")
    _q = _urlquote(_sel["Player"])
    st.markdown(f"""
<style>
.hub-card {{ background: var(--panel-solid); border: 1px solid var(--panel-line);
  border-radius: 14px; padding: 1.1rem 1.4rem; box-shadow: var(--shadow-card);
  margin-bottom: 0.8rem; }}
.hub-name {{ font-size: 1.6rem; font-weight: 800; color: var(--fg-1); }}
.hub-sub  {{ color: var(--fg-3); font-size: 0.85rem; margin-top: 0.15rem; }}
.hub-stats {{ display: flex; gap: 2.2rem; flex-wrap: wrap; margin-top: 0.9rem; }}
.hub-stat .v {{ font-size: 1.45rem; font-weight: 800; color: var(--accent-teal); }}
.hub-stat .l {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--fg-4); font-weight: 600; margin-top: 0.1rem; }}
.hub-links {{ margin-top: 1rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }}
.hub-links a {{ background: var(--panel); border: 1px solid var(--panel-line);
  border-radius: 9px; padding: 0.45rem 0.9rem; font-size: 0.82rem; font-weight: 700;
  color: var(--sky); text-decoration: none; }}
.hub-links a:hover {{ border-color: var(--sky); }}
</style>
<div class="hub-card">
  <div class="hub-name">{html.escape(_sel["Player"])}</div>
  <div class="hub-sub">{html.escape(str(_sel["Team"]))} · {html.escape(str(_sel["Pos"]))} · {html.escape(_draft_txt)}
    &nbsp;·&nbsp; Status: <b>{html.escape(str(_status_txt))}</b>{(" — " + html.escape(_outcome)) if _outcome else ""}</div>
  <div class="hub-stats">
    <div class="hub-stat"><div class="v">{_sel["Barrett Score"]:.2f}</div><div class="l">Barrett Score · #{_sel["rank"]}</div></div>
    <div class="hub-stat"><div class="v">${_sel["Salary"]:.1f}M</div><div class="l">2025-26 Salary</div></div>
    <div class="hub-stat"><div class="v">{_pred_txt}</div><div class="l">Predicted contract{(" · " + _band_txt) if _band_txt else ""}</div></div>
  </div>
  <div class="hub-links">
    <a href="/Contract_Predictor?player={_q}" target="_top">Full contract prediction →</a>
    <a href="/Search?player={_q}" target="_top">Player profile & career →</a>
    <a href="/Free_Agent_Class" target="_top">Free agent class →</a>
  </div>
</div>
""", unsafe_allow_html=True)

    _car = _hub_career()
    _mine = _car[_car["norm"] == _n].sort_values("Season")
    if len(_mine) >= 2:
        _fig = px.line(_mine, x="Season", y="barrett_score", markers=True,
                       labels={"barrett_score": "Barrett Score", "Season": ""}, height=220)
        _fig.update_layout(margin=dict(t=10, b=10, l=10, r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(theme_fig(_fig), use_container_width=True,
                        config={"displayModeBar": False})

# ── The list ──────────────────────────────────────────────────────────────────
_lst_l, _lst_r = st.columns([3, 1])
with _lst_l:
    st.markdown(
        f"<div style='font-size:1.05rem;font-weight:800;color:var(--fg-1);margin:0.4rem 0 0.3rem'>"
        f"2025-26 Player Board <span style='color:var(--fg-4);font-weight:600;font-size:0.8rem'>"
        f"· {len(_hub_df)} players · click a name</span></div>",
        unsafe_allow_html=True)
with _lst_r:
    _show_all = st.checkbox(f"Show all {len(_hub_df)}", value=False, key="hub_show_all")

_view = _hub_df if _show_all else _hub_df.head(100)
html_table(
    _view.drop(columns=["norm"]),
    formatters={
        "Player": lambda v: (f'<a class="hv-plink" href="/?player={_urlquote(str(v))}" '
                             f'target="_top">{html.escape(str(v))}</a>'),
        "Barrett Score": lambda v: f"{v:.2f}",
        "Salary": lambda v: f"${v:.2f}M",
        "Predicted": lambda v: ("—" if v is None or (isinstance(v, float) and v != v)
                                else f"${v:.1f}M"),
    },
    raw={"Player"},
    styles={
        "Status": lambda v, _r: {
            "UFA": "color:var(--fg-3)", "RFA": "color:var(--value-good);font-weight:700",
            "Player Option": "color:var(--blue);font-weight:700",
            "Team Option": "color:var(--orange);font-weight:700",
            "Signed": "color:var(--accent-teal);font-weight:700",
        }.get(str(v), "color:var(--fg-6)"),
        "Predicted": lambda v, _r: "color:var(--fg-6)" if v is None or (isinstance(v, float) and v != v) else "color:var(--accent-teal)",
    },
    aligns={"#": "right", "Barrett Score": "right", "Salary": "right", "Predicted": "right"},
    numeric={"#", "Barrett Score", "Salary", "Predicted"},
    helps={
        "Barrett Score": "Base Score × Availability Multiplier. Higher = more valuable.",
        "Predicted": "The Contract Predictor's model projection — what this player would sign for today.",
        "Status": "This offseason: UFA/RFA/option holder, Signed = came off the board, — = under contract.",
    },
    height=min(760, max(260, len(_view) * 38 + 46)),
)
st.markdown(
    "<style>.hv-plink{color:var(--sky);font-weight:700;text-decoration:none}"
    ".hv-plink:hover{text-decoration:underline}</style>",
    unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
from utils import render_footer
render_footer()
