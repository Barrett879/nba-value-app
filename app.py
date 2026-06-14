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
# SVG chart helpers (used by tab previews)
# ══════════════════════════════════════════════════════════════════════════════
def _esc(s) -> str:
    return html.escape(str(s))


def _hbar_chart(items, w=460, h=140, label_w=140, color_default="#e63946"):
    if not items:
        return ""
    n = len(items)
    pad_top, pad_bot = 6, 6
    avail_h = h - pad_top - pad_bot
    bar_h = avail_h / n - 4
    vmax = max(it["value"] for it in items) or 1.0
    chart_x = label_w
    chart_w = w - label_w - 50
    parts = []
    for i, it in enumerate(items):
        y = pad_top + i * (avail_h / n)
        bw = (it["value"] / vmax) * chart_w if vmax else 0
        c = it.get("color", color_default)
        parts.append(
            f'<text x="6" y="{y + bar_h/2 + 4:.1f}" style="fill:var(--fg-3)" font-size="11" '
            f'font-family="system-ui">{_esc(it["label"])}</text>'
        )
        parts.append(
            f'<rect x="{chart_x}" y="{y:.1f}" rx="3" ry="3" '
            f'width="{max(bw, 1):.1f}" height="{bar_h:.1f}" fill="{c}" opacity="0.88"/>'
        )
        parts.append(
            f'<text x="{chart_x + bw + 6:.1f}" y="{y + bar_h/2 + 4:.1f}" '
            f'style="fill:var(--fg-1)" font-size="11" font-weight="700" '
            f'font-family="system-ui">{_esc(it["value_str"])}</text>'
        )
    return f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" style="width:100%; height:{h}px;">{"".join(parts)}</svg>'


def _diverging_bars(rows, w=460, h=140):
    if not rows:
        return ""
    n = len(rows)
    pad_top, pad_bot = 6, 6
    avail_h = h - pad_top - pad_bot
    bar_h = avail_h / n - 4
    vmax = max(abs(r["value"]) for r in rows) or 1.0
    cx = w / 2
    half_w = w / 2 - 70
    parts = [f'<line x1="{cx}" y1="{pad_top - 2}" x2="{cx}" y2="{h - pad_bot + 2}" '
             f'stroke="rgba(255,255,255,0.12)" stroke-width="1"/>']
    for i, r in enumerate(rows):
        y = pad_top + i * (avail_h / n)
        bw = (abs(r["value"]) / vmax) * half_w
        if r["side"] == "neg":
            x_rect, x_lbl, anchor_lbl, x_val, anchor_val = cx - bw, cx - bw - 6, "end", cx + 4, "start"
        else:
            x_rect, x_lbl, anchor_lbl, x_val, anchor_val = cx, cx + bw + 6, "start", cx - 4, "end"
        parts.append(
            f'<rect x="{x_rect:.1f}" y="{y:.1f}" rx="3" ry="3" '
            f'width="{max(bw, 1):.1f}" height="{bar_h:.1f}" fill="{r["color"]}" opacity="0.88"/>'
        )
        parts.append(
            f'<text x="{x_lbl:.1f}" y="{y + bar_h/2 + 4:.1f}" text-anchor="{anchor_lbl}" '
            f'style="fill:var(--fg-3)" font-size="10" font-family="system-ui">{_esc(r["label"])}</text>'
        )
        parts.append(
            f'<text x="{x_val:.1f}" y="{y + bar_h/2 + 4:.1f}" text-anchor="{anchor_val}" '
            f'style="fill:var(--fg-1)" font-size="10" font-weight="700" '
            f'font-family="system-ui">{_esc(r["value_str"])}</text>'
        )
    return f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" style="width:100%; height:{h}px;">{"".join(parts)}</svg>'


def _interp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    return f"#{int(r1+(r2-r1)*t):02x}{int(g1+(g2-g1)*t):02x}{int(b1+(b2-b1)*t):02x}"


def _value_color(v, vmin, vmax, low="#e74c3c", mid="#f1c40f", high="#2ecc71"):
    if vmax <= vmin:
        return mid
    t = (v - vmin) / (vmax - vmin)
    return _interp_color(low, mid, t * 2) if t < 0.5 else _interp_color(mid, high, (t - 0.5) * 2)


def _multi_sparkline(series_list, w=460, h=160, dots=True, halo=None):
    """Overlay multiple career arcs aligned by career year.
    series_list: list of dicts {name, color, career: [{season, score, rank, total}, ...]}.
    Player labels render as a colored legend across the top.
    Each dot has a native browser tooltip (SVG <title>) showing
    season · Barrett Score · rank that year.
    """
    valid = [s for s in series_list if s.get("career")]
    if not valid:
        return ""
    pad_left, pad_right = 14, 14
    pad_top, pad_bot = 26, 18  # extra top pad for legend
    chart_w = w - pad_left - pad_right
    chart_h = h - pad_top - pad_bot

    all_vals = [pt["score"] for s in valid for pt in s["career"]]
    vmin, vmax = min(all_vals), max(all_vals)
    rng = (vmax - vmin) or 1.0
    max_len = max(len(s["career"]) for s in valid)
    if max_len < 2:
        return ""

    parts = []

    # Legend across the top — one tspan per player, colored to match its line
    legend_segments = []
    for i, s in enumerate(valid):
        last_name = s["name"].split()[-1]
        if i > 0:
            legend_segments.append('<tspan fill="#666"> · </tspan>')
        legend_segments.append(
            f'<tspan fill="{s["color"]}" font-weight="700">{_esc(last_name)}</tspan>'
        )
    parts.append(
        f'<text x="{w/2:.1f}" y="14" text-anchor="middle" '
        f'font-size="11" font-family="system-ui">'
        + "".join(legend_segments) + "</text>"
    )

    # Lines + dots per player
    for s in valid:
        coords = []
        for i, pt in enumerate(s["career"]):
            x = pad_left + (i / (max_len - 1)) * chart_w
            y = pad_top + chart_h - ((pt["score"] - vmin) / rng) * chart_h
            coords.append((x, y, pt))
        line_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
        # Optional halo: a wider under-stroke in a surface-contrasting color so a
        # bright series line stays legible on a low-contrast (e.g. light) surface.
        if halo:
            parts.append(
                f'<polyline points="{line_pts}" fill="none" stroke="{halo}" '
                f'stroke-width="4.4" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        parts.append(
            f'<polyline points="{line_pts}" fill="none" stroke="{s["color"]}" '
            f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>'
        )
        if not dots:
            continue  # lines-only overlay (multi-player preview): skip dots/tooltips
        for x, y, pt in coords:
            tooltip = (
                f'{_esc(s["name"])} · {_esc(pt["season"])}'
                f'  ·  Barrett {pt["score"]:.2f}'
                f'  ·  Rank #{pt["rank"]}/{pt["total"]}'
            )
            # Single visible dot — bigger (r=4.5) for an easy hover target.
            # <title> as a direct child triggers the OS-native tooltip in
            # every browser. Putting it inside a <g> with a transparent
            # overlay was unreliable across browsers.
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{s["color"]}" '
                f'stroke="#14142a" stroke-width="1.2" style="cursor:help;">'
                f'<title>{tooltip}</title>'
                f'</circle>'
            )

    # X-axis hint
    parts.append(
        f'<text x="{pad_left:.1f}" y="{h - 4}" '
        f'fill="#777" font-size="9" font-family="system-ui">Year 1</text>'
    )
    parts.append(
        f'<text x="{w - pad_right:.1f}" y="{h - 4}" text-anchor="end" '
        f'fill="#777" font-size="9" font-family="system-ui">Year {max_len}</text>'
    )

    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%; height:{h}px;">{"".join(parts)}</svg>'
    )


def _fa_category_chart(items, w=460, h=140):
    if not items:
        return ""
    n = len(items)
    total = sum(it["count"] for it in items)
    chart_y = 22
    chart_h = h - chart_y - 6
    row_h = chart_h / n
    bar_h = row_h * 0.62
    label_w = 92
    chart_x = label_w
    chart_w = w - chart_x - 50
    avg_max = max(it["avg_score"] for it in items) or 1.0
    parts = [
        f'<text x="{w/2:.1f}" y="11" text-anchor="middle" style="fill:var(--fg-3)" '
        f'font-size="10.5" font-family="system-ui">'
        f'<tspan style="fill:var(--fg-1)" font-weight="700">{int(total)}</tspan> free agents · avg Barrett Score</text>'
    ]
    for i, it in enumerate(items):
        y = chart_y + i * row_h + (row_h - bar_h) / 2
        bw = (it["avg_score"] / avg_max) * chart_w if avg_max else 0
        parts.append(
            f'<text x="6" y="{y + bar_h/2 + 4:.1f}" font-size="10.5" font-family="system-ui">'
            f'<tspan fill="{it["color"]}" font-weight="700">{_esc(it["label"])}</tspan> '
            f'<tspan fill="#888"> ({int(it["count"])})</tspan></text>'
        )
        parts.append(
            f'<rect x="{chart_x:.1f}" y="{y:.1f}" rx="3" ry="3" '
            f'width="{max(bw, 1):.1f}" height="{bar_h:.1f}" fill="{it["color"]}" opacity="0.88"/>'
        )
        parts.append(
            f'<text x="{chart_x + bw + 5:.1f}" y="{y + bar_h/2 + 4:.1f}" '
            f'style="fill:var(--fg-1)" font-size="10.5" font-weight="700" font-family="system-ui">'
            f'{it["avg_score"]:.1f}</text>'
        )
    return f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" style="width:100%; height:{h}px;">{"".join(parts)}</svg>'


# ══════════════════════════════════════════════════════════════════════════════
# Compute live data for previews. Cached so reruns (theme/playoff toggles,
# search keystrokes) don't re-run the whole assembly — that recompute was what
# made toggling dark mode feel like a full page refresh. Depends only on the
# current season, so a flat cache keyed on the season string is correct.
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def _compute_charts():
    try:
        df = build_ranked_projected(SEASONS[0])
        df = df[df["total_min"] >= DEFAULT_MIN_THRESHOLD].copy()
        if df.empty:
            return None

        # Value picks (steal/overpaid) only make sense for players on a real
        # salary — a $0/unsigned row would surface as an infinite "steal".
        # Mirrors the salary>0 guard on the Rankings/Legacy pages.
        df_paid = df[df["salary"] > 0] if "salary" in df.columns else df
        if df_paid.empty:
            df_paid = df

        top10 = df.nsmallest(10, "score_rank")[["Player", "barrett_score"]].values.tolist()
        steals_3 = df_paid.nsmallest(3, "value_diff")[["Player", "value_diff"]].values.tolist()
        overpaid_3 = df_paid.nlargest(3, "value_diff")[["Player", "value_diff"]].values.tolist()

        # Hero-card data (#1 in each category, with team info for the subtitle).
        # Kept separate from the 2-tuple top10/steals_3/overpaid_3 above so the
        # strip-preview consumers (which expect 2-tuples) don't need to change.
        _b = df.nsmallest(1, "score_rank").iloc[0]
        _s = df_paid.nsmallest(1, "value_diff").iloc[0]
        _o = df_paid.nlargest(1, "value_diff").iloc[0]
        best_card = {"name": str(_b["Player"]), "team": str(_b["Team"]),
                     "score": float(_b["barrett_score"])}
        steal_card = {"name": str(_s["Player"]), "team": str(_s["Team"]),
                      "value_diff": float(_s["value_diff"])}
        overpaid_card = {"name": str(_o["Player"]), "team": str(_o["Team"]),
                         "value_diff": float(_o["value_diff"])}

        team_eff = (
            df.groupby("Team")
              .apply(lambda g: (g["salary"].sum() - g["projected_salary"].sum()) / 1e6)
              .sort_values()
        )
        best_teams = list(team_eff.head(3).items())
        worst_teams = list(team_eff.tail(3).items())

        next_contracts = fetch_next_year_contracts(season_to_espn_year(SEASONS[0]), cache_v=7)
        rookie_scale = fetch_rookie_scale_players(SEASONS[0])

        ufa, rfa, po, to = [], [], [], []
        _fa_lists = {"UFA": ufa, "RFA": rfa, "Player Option": po, "Team Option": to}
        for _, row in df[["Player", "barrett_score"]].iterrows():
            name = row["Player"]
            _st = classify_fa_status(name, fmt_next_contract(name, next_contracts),
                                     rookie_scale, SEASONS[0])
            if _st in _fa_lists:
                _fa_lists[_st].append(float(row["barrett_score"]))

        def _avg(xs):
            return sum(xs) / len(xs) if xs else 0.0

        # Pull career arcs for all featured Legacy players. Each call reads
        # disk-cached parquets — no API hits at view time. Includes rank +
        # total players per season for the hover tooltips.
        legacy_series = []
        for entry in LEGACY_FEATURED:
            try:
                career = fetch_player_career_with_rank(entry["id"])
            except Exception:
                career = []
            legacy_series.append({
                "name":   entry["name"],
                "color":  entry["color"],
                "career": career,  # list of {'season', 'score', 'rank', 'total'}
            })

        return {
            "top10": top10,
            "best_card":     best_card,
            "steal_card":    steal_card,
            "overpaid_card": overpaid_card,
            "steals_3": [(str(p), float(v)) for p, v in steals_3],
            "overpaid_3": [(str(p), float(v)) for p, v in overpaid_3],
            "best_teams": [(str(t), float(v)) for t, v in best_teams],
            "worst_teams": [(str(t), float(v)) for t, v in worst_teams],
            "fa_categories": [
                {"label": "UFA", "count": len(ufa), "avg_score": _avg(ufa), "color": "#aaaaaa"},
                {"label": "RFA", "count": len(rfa), "avg_score": _avg(rfa), "color": "#2ecc71"},
                {"label": "PO", "count": len(po), "avg_score": _avg(po), "color": "#3498db"},
                {"label": "TO", "count": len(to), "avg_score": _avg(to), "color": "#f39c12"},
            ],
            "legacy_series":     legacy_series,
            "n_indexed_players": len(_all_player_names) if _all_player_names else 0,
        }
    except Exception:
        return None


_p = _compute_charts()


# ══════════════════════════════════════════════════════════════════════════════
# 3 hero cards — Best Player Right Now / Biggest Steal / Most Overpaid
# Rendered above the strips so they're the first big visual after the search.
# ══════════════════════════════════════════════════════════════════════════════
if _p and _p.get("best_card") and _p.get("steal_card") and _p.get("overpaid_card"):
    _bc, _sc, _oc = _p["best_card"], _p["steal_card"], _p["overpaid_card"]
    hc1, hc2, hc3 = st.columns(3, gap="medium")
    with hc1:
        st.markdown(f"""
        <div class="home-hero-card" style="background:var(--tint-good); border:1px solid var(--value-good);">
            <div class="hh-label">Best Player Right Now</div>
            <div class="hh-name">{_bc["name"]}</div>
            <div class="hh-sub">{_bc["team"]} · Barrett Score {_bc["score"]:.1f}</div>
        </div>""", unsafe_allow_html=True)
    with hc2:
        st.markdown(f"""
        <div class="home-hero-card" style="background:var(--tint-good); border:1px solid var(--value-good);">
            <div class="hh-label">Biggest Steal</div>
            <div class="hh-name">{_sc["name"]}</div>
            <div class="hh-sub">{_sc["team"]} · ${abs(_sc["value_diff"])/1e6:.1f}M below market value</div>
        </div>""", unsafe_allow_html=True)
    with hc3:
        st.markdown(f"""
        <div class="home-hero-card" style="background:var(--tint-bad); border:1px solid var(--value-bad);">
            <div class="hh-label">Most Overpaid</div>
            <div class="hh-name">{_oc["name"]}</div>
            <div class="hh-sub">{_oc["team"]} · ${_oc["value_diff"]/1e6:.1f}M above market value</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Render one expandable strip — click the header to open the preview inline,
# 'Open X →' button inside the body handles navigation.
# Using a raw HTML <details> element so the entire thing is one Streamlit
# markdown call (native browser open/close, no script reruns, instant).
# ══════════════════════════════════════════════════════════════════════════════
def render_strip(name: str, href: str, accent: str, description: str, preview_html: str):
    # The whole strip must reach st.markdown as ONE unbroken HTML block: a
    # blank (or whitespace-only) line ends the block, and any indented line
    # after that renders as a markdown code block (raw-HTML leak). So drop
    # empty lines from the preview and keep the template itself unindented.
    preview_html = "\n".join(l.strip() for l in preview_html.splitlines() if l.strip())
    st.markdown(
        f'<details class="explore-strip" style="--accent:{accent};">'
        f'<summary class="explore-strip-summary">'
        f'<div class="tab-strip-name">{name}</div>'
        f'<div class="tab-strip-desc">{description}</div>'
        f'<span class="strip-arrow">▾</span>'
        f'</summary>'
        f'<div class="explore-strip-body">'
        f'{preview_html}'
        f'<a class="goto-btn" href="{href}" target="_top">Open {name} →</a>'
        f'</div>'
        f'</details>',
        unsafe_allow_html=True,
    )


# ── Rankings preview ─────────────────────────────────────────────────────────
if _p:
    # Gold → bronze gradient for ranks 1 → 10
    rank_colors = [
        "#f1c40f", "#ecbe1a", "#e3b121", "#d3a02a", "#bf8e34",
        "#a87c3a", "#916b3d", "#7a5b3c", "#634c39", "#4d3e35",
    ]
    rankings_preview = _hbar_chart([
        {
            "label":     f"{i+1}. {name.split()[-1] if len(name.split()) > 1 else name}",
            "value":     score,
            "value_str": f"{score:.1f}",
            "color":     rank_colors[i] if i < len(rank_colors) else rank_colors[-1],
        }
        for i, (name, score) in enumerate(_p["top10"])
    ], w=460, h=260, label_w=150) + '<div style="text-align:center; font-size:0.7rem; color:var(--fg-5); margin-top:0.4rem;">Top 10 by Barrett Score · this season</div>'

    vis_rows = []
    for n_, vd in _p["steals_3"]:
        amt = abs(vd) / 1e6
        vis_rows.append({"label": n_.split()[-1], "value": amt, "value_str": f"-${amt:.1f}M",
                         "color": "#2ecc71", "side": "neg"})
    for n_, vd in _p["overpaid_3"]:
        amt = vd / 1e6
        vis_rows.append({"label": n_.split()[-1], "value": amt, "value_str": f"+${amt:.1f}M",
                         "color": "#e74c3c", "side": "pos"})

    team_rows = []
    for t, v in _p["best_teams"]:
        team_rows.append({"label": t, "value": abs(v), "value_str": f"-${abs(v):.1f}M",
                          "color": "#2ecc71", "side": "neg"})
    for t, v in _p["worst_teams"]:
        team_rows.append({"label": t, "value": abs(v), "value_str": f"+${abs(v):.1f}M",
                          "color": "#e74c3c", "side": "pos"})

    teams_preview = _diverging_bars(team_rows) + '<div style="text-align:center; font-size:0.7rem; color:var(--fg-5); margin-top:0.4rem;">Net payroll efficiency · green = team is winning the value game</div>'
    fa_preview = _fa_category_chart(_p["fa_categories"]) + '<div style="text-align:center; font-size:0.7rem; color:var(--fg-5); margin-top:0.4rem;">Free-agent class breakdown · this offseason</div>'
    # Legacy preview is rendered specially below (needs an interactive radio
    # for the user to pick a featured player), so we just stash the data here.
    legacy_preview = None

    # Trades preview — Harden→Houston featured trade. Color the bars based on
    # the editorial 'winner' field (grounded in actual outcomes), NOT raw
    # Barrett-Score sums — those would mislead for trades like AD-to-Lakers
    # where the Lakers won a title with a side that had a lower Barrett sum.
    def _build_trades_preview():
        if not HISTORICAL_TRADES:
            return "<em>No featured trades.</em>"
        pick = next((t for t in HISTORICAL_TRADES if "Harden" in t["name"]), HISTORICAL_TRADES[0])
        season = pick["season"]
        sum_a = trade_side_summary(tuple(pick["side_a"]), season)
        sum_b = trade_side_summary(tuple(pick["side_b"]), season)
        a_total, b_total = sum_a["barrett_total"], sum_b["barrett_total"]
        if a_total == 0 and b_total == 0:
            return "<em>Data still seeding…</em>"
        winner = pick.get("winner", "wash")
        a_color = "#2ecc71" if winner == "side_a" else ("#888" if winner == "wash" else "#e63946")
        b_color = "#2ecc71" if winner == "side_b" else ("#888" if winner == "wash" else "#e63946")
        rows = [
            {"label": pick["side_a_team"].split()[-1], "value": max(a_total, 0.5),
             "value_str": f"{a_total:.1f}",
             "color": a_color, "side": "pos"},
            {"label": pick["side_b_team"].split()[-1], "value": max(b_total, 0.5),
             "value_str": f"{b_total:.1f}",
             "color": b_color, "side": "pos"},
        ]
        return _hbar_chart(rows, w=460, h=110, label_w=110) + f'<div style="text-align:center; font-size:0.7rem; color:var(--fg-5); margin-top:0.4rem;">Featured trade · {pick["name"]}</div>'

    trades_preview = _build_trades_preview()
else:
    rankings_preview = teams_preview = fa_preview = trades_preview = "<em>Loading live data…</em>"


# ══════════════════════════════════════════════════════════════════════════════
# Explore Deeper — expandable strips. Click any strip to expand its preview
# inline; click the "Open X →" button inside to navigate to that page.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div style='margin-top:1.2rem; margin-bottom:0.5rem; "
    "font-size:0.78rem; color:var(--fg-3); letter-spacing:0.08em; "
    "text-transform:uppercase;'>Explore deeper</div>",
    unsafe_allow_html=True,
)
render_strip(
    name="Current Rankings",
    href="/Rankings",
    accent="#e63946",
    description="Who's the best NBA player right now? Every player ranked by Barrett Score this season.",
    preview_html=rankings_preview,
)

def _star_points(cx, cy, r):
    pts = []
    for k in range(10):
        a = math.radians(-90 + k * 36)
        rad = r if k % 2 == 0 else r * 0.42
        pts.append(f"{cx + rad * math.cos(a):.1f},{cy + rad * math.sin(a):.1f}")
    return " ".join(pts)


def _legacy_compare_svg(valid, dark=False):
    """Career-arc comparison chart for the Legacy preview — line + dot markers,
    horizontal gridlines, a Barrett-Score y-axis, a peak star per player and a
    legend: a compact SVG echo of the full Plotly head-to-head chart on the
    Search page."""
    valid = [s for s in valid if s.get("career")]
    if not valid:
        return ""
    W, H = 860, 300
    ML, MR, MT, MB = 50, 18, 16, 60
    pw, ph = W - ML - MR, H - MT - MB
    grid = "rgba(255,255,255,0.07)" if dark else "rgba(20,22,40,0.07)"
    axfg = "#888888" if dark else "#9aa0ab"
    txfg = "#aaaaaa" if dark else "#585c68"
    surf = "#1a1a2e" if dark else "#eef1f4"
    ytop = max(10, int(math.ceil(max(pt["score"] for s in valid for pt in s["career"]) / 10.0)) * 10)
    maxlen = max(len(s["career"]) for s in valid)

    def X(i):
        return ML + (i / (maxlen - 1)) * pw if maxlen > 1 else ML + pw / 2

    def Y(v):
        return MT + ph - (v / ytop) * ph

    p = []
    yt = 0
    while yt <= ytop:
        gy = Y(yt)
        p.append(f'<line x1="{ML}" y1="{gy:.1f}" x2="{ML + pw}" y2="{gy:.1f}" stroke="{grid}" stroke-width="1"/>')
        p.append(f'<text x="{ML - 8}" y="{gy + 3.5:.1f}" text-anchor="end" font-size="11" fill="{axfg}" font-family="system-ui">{yt}</text>')
        yt += 10
    _ym = MT + ph / 2
    p.append(f'<text x="13" y="{_ym:.1f}" text-anchor="middle" font-size="11" fill="{txfg}" '
             f'font-family="system-ui" transform="rotate(-90 13 {_ym:.1f})">Barrett Score</text>')
    for xi in range(maxlen):
        p.append(f'<text x="{X(xi):.1f}" y="{MT + ph + 17:.1f}" text-anchor="middle" font-size="10" '
                 f'fill="{axfg}" font-family="system-ui">{xi + 1}</text>')
    p.append(f'<text x="{ML + pw / 2:.1f}" y="{MT + ph + 33:.1f}" text-anchor="middle" font-size="11" '
             f'fill="{txfg}" font-family="system-ui">Career year (Year 1 = first season)</text>')
    for s in valid:
        col = s["color"]
        pts = [(X(i), Y(pt["score"])) for i, pt in enumerate(s["career"])]
        p.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" '
                 f'stroke="{col}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>')
        for x, y in pts:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{col}" stroke="{surf}" stroke-width="1"/>')
        pk = max(range(len(s["career"])), key=lambda i: s["career"][i]["score"])
        p.append(f'<polygon points="{_star_points(X(pk), Y(s["career"][pk]["score"]), 9)}" '
                 f'fill="{col}" stroke="{surf}" stroke-width="1"/>')
    segs = []
    for i, s in enumerate(valid):
        if i:
            segs.append('<tspan fill="#999">     </tspan>')
        segs.append(f'<tspan fill="{s["color"]}" font-weight="600">● {_esc(s["name"])}</tspan>')
    p.append(f'<text x="{W / 2:.1f}" y="{H - 8:.1f}" text-anchor="middle" font-size="12.5" '
             f'font-family="system-ui">{"".join(segs)}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" '
            f'style="width:100%; height:auto; max-height:340px;">{"".join(p)}</svg>')


# Legacy uses the SAME render_strip <details> as every other strip, so it looks
# and collapses identically. An interactive per-player picker can't be embedded
# inline and made to match the other strips' chrome reliably in Streamlit
# (inline HTML resets / strips scripts; a components.html iframe loads the app
# homepage into itself on Render restarts; a styled st.expander won't render as a
# bordered card). So the preview is a static LeBron-vs-Jordan comparison chart;
# the full per-player picker lives on the Legacy page (the "Open Legacy" button).
_LEGACY_DESC = ("53 seasons of NBA history: all-time greats, era leaderboards, "
                "team Mount Rushmores, draft classes.")

_legacy_valid = [s for s in _p.get("legacy_series", []) if s.get("career")] if _p else []
if _legacy_valid:
    _dark = st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
    # Legible per-theme colours (bright gold/red on dark, deeper on the light strip).
    _lcolors = {"Michael Jordan": "#f1c40f" if _dark else "#b8860b",
                "LeBron James":   "#ef5350" if _dark else "#d83a4a"}
    _pair = [{**s, "color": _lcolors.get(s["name"], s["color"])}
             for nm in ("Michael Jordan", "LeBron James")
             for s in _legacy_valid if s["name"] == nm]
    _legacy_preview = _legacy_compare_svg(_pair or _legacy_valid[:2], dark=_dark)
else:
    _legacy_preview = "<em>Loading live data...</em>"

render_strip(name="Legacy", href="/Legacy", accent="#f1c40f", description=_LEGACY_DESC, preview_html=_legacy_preview)

render_strip(
    name="Team Analysis",
    href="/Team_Analysis",
    accent="#3498db",
    description="Which front offices are getting the most for their money? Payroll efficiency by team.",
    preview_html=teams_preview,
)

render_strip(
    name="Contract Predictor",
    href="/Contract_Predictor",
    accent="var(--accent-teal)",
    description="What would any player command on a new deal signed today? A model prices their production against next season's cap.",
    preview_html=(
        '<div style="font-size:0.85rem; color:var(--fg-2); line-height:1.55;">'
        "Type any player's name to project the contract they'd sign "
        '<b style="color:var(--fg-1);">today</b>, their 2025-26 production valued '
        "against next season's salary cap."
        '<div style="margin-top:0.7rem; display:flex; gap:1.4rem; flex-wrap:wrap; '
        'font-size:0.78rem; color:var(--fg-3);">'
        '<span><b style="color:var(--accent-teal);">HistGBM</b> · 1,900+ modern signings</span>'
        '<span><b style="color:var(--value-good);">89%</b> within 5% of the cap</span>'
        '<span>supermax · vet-discount · rookie-scale aware</span>'
        '</div></div>'
    ),
)

# Trades strip removed — page disabled. Restore by uncommenting and adding
# back to _NAV_PAGES in utils.py + un-disabling pages/Trades.py.
# render_strip(
#     name="Trades",
#     href="/Trades",
#     accent="#9b59b6",
#     description="Stack any two trade sides head-to-head — past trades or your own. Who actually came out ahead?",
#     preview_html=trades_preview,
# )

render_strip(
    name="Current Free Agents",
    href="/Free_Agent_Class",
    accent="#2ecc71",
    description="Every player hitting the market this offseason: UFAs, RFAs, options. What they're worth.",
    preview_html=fa_preview,
)


# ── Footer ────────────────────────────────────────────────────────────────────
from utils import render_footer
render_footer()
