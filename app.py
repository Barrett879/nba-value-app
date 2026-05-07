import sys
import html
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from utils import (
    _bootstrap_warm,
    build_ranked_projected,
    fetch_next_year_contracts,
    fetch_rookie_scale_players,
    fetch_career_trend,
    fmt_next_contract,
    season_to_espn_year,
    normalize,
    SEASONS,
    DEFAULT_MIN_THRESHOLD,
    get_all_player_names,
    HISTORICAL_TRADES,
    trade_side_summary,
)

# LeBron James' NBA Stats player ID — used for the Legacy career arc sparkline
LEBRON_PLAYER_ID = 2544

# Start warming all season caches the moment the server boots —
# before any user arrives, so the first visitor doesn't pay the cost.
_bootstrap_warm()

st.set_page_config(page_title="Barrett Score", layout="wide")

# ── Hide Streamlit chrome & set light-gray page background ────────────────────
st.markdown("""
<style>
    /* Court-image background with a dark overlay for text contrast.
       Image lives at static/court.jpg and is served by Streamlit when
       enableStaticServing = true (set in .streamlit/config.toml). */
    .stApp {
        background-image:
            linear-gradient(rgba(8, 8, 16, 0.78), rgba(8, 8, 16, 0.86)),
            url("./app/static/LightCourt.jpeg") !important;
        background-size: cover !important;
        background-position: center top !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
        background-color: #0a0a14 !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] { background: transparent !important; }

    /* Aggressively kill the Streamlit header/toolbar height */
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

    /* Tight top padding — page should fit one viewport without scroll */
    .block-container,
    .main .block-container,
    section.main > .block-container,
    [data-testid="stMain"] .block-container,
    [data-testid="stMainBlockContainer"],
    section[data-testid="stMain"] > .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
        max-width: 100%;
    }
    .stApp { padding-top: 0 !important; }

    /* Clickable nav card — dark "data panel" on the light page */
    a.nav-card {
        display: flex;
        flex-direction: column;
        background: #14142a;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 0.95rem 1rem 0.85rem 1rem;
        text-decoration: none;
        transition: border-color 0.2s, transform 0.15s, box-shadow 0.2s;
        cursor: pointer;
        min-height: 300px;
        box-sizing: border-box;
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(20, 20, 42, 0.06);
    }
    a.nav-card:hover {
        border-color: var(--accent, #e63946);
        transform: translateY(-2px);
        text-decoration: none;
        box-shadow: 0 8px 22px rgba(20, 20, 42, 0.18);
    }
    a.nav-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--accent, #e63946);
        opacity: 0.85;
    }
    .nav-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #fff;
        text-align: center;
    }
    .nav-desc {
        font-size: 0.74rem;
        color: #888;
        line-height: 1.35;
        text-align: center;
        margin-bottom: 0.45rem;
    }
    .nav-chart {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 0.3rem 0.15rem;
        background: rgba(0, 0, 0, 0.25);
        border-radius: 8px;
        margin-bottom: 0.6rem;
        min-height: 130px;
    }
    .nav-chart svg { display: block; }
    .nav-chart-empty {
        font-size: 0.75rem;
        color: #666;
        text-align: center;
        font-style: italic;
        padding: 0.8rem;
    }
    .nav-chart-caption {
        font-size: 0.65rem;
        color: #777;
        text-align: center;
        font-style: italic;
        margin-top: 0.15rem;
    }

    .nav-cta {
        align-self: center;
        background: #e63946;
        color: #fff !important;
        border-radius: 8px;
        padding: 0.35rem 1.2rem;
        font-weight: 600;
        font-size: 0.78rem;
        flex-shrink: 0;
    }
    a.nav-card:hover .nav-cta { background: #c1121f; }
</style>
""", unsafe_allow_html=True)

# ── Hero — title + intro blurb ────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 0.4rem 0 0.7rem 0;">
    <div style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, system-ui, sans-serif;
        font-size: 2.4rem;
        line-height: 1;
        letter-spacing: 0.1em;
        text-shadow: 0 3px 14px rgba(0,0,0,0.55);
        white-space: nowrap;
    ">
        <span style="color: #c8cdd6; font-weight: 600;">THE&nbsp;</span><span style="color: #ffffff; font-weight: 800;">BARRETT&nbsp;</span><span style="color: #7ec8e8; font-weight: 800;">SCORE</span>
    </div>
    <div style="font-size:0.95rem; color:#d6d6dc; margin-top:0.7rem; max-width:1100px; margin-left:auto; margin-right:auto; line-height:1.5; text-shadow: 0 1px 6px rgba(0,0,0,0.5);">
        Ever wonder which NBA stars are quietly outplaying their contracts and which ones aren't earning their paycheck?<br>
        Welcome to <b style="color:#fff;">The Barrett Score</b>.<br>
        We calculate scoring, playmaking, defense, and efficiency into a single value, then we put every player's on-court impact next to what they're paid.<br>
        <span style="color:#f1c40f; font-weight:700;">That's the Barrett Score.</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Search hero — centerpiece on the home page
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Style the search box to feel like a hero element */
[data-testid="stSelectbox"][data-baseweb] div[role="combobox"] {
    background: rgba(20, 20, 42, 0.7) !important;
    border: 2px solid rgba(126, 200, 232, 0.55) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(6px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}
.home-search-wrap {
    max-width: 760px;
    margin: 0.7rem auto 1.2rem auto;
}
.home-search-label {
    font-size: 0.85rem;
    color: #d6d6dc;
    text-align: center;
    margin-bottom: 0.4rem;
    text-shadow: 0 1px 6px rgba(0,0,0,0.6);
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

_, _search_col, _ = st.columns([1, 2, 1])
with _search_col:
    st.markdown(
        '<div class="home-search-label">SEARCH ANY PLAYER · 1984 → today</div>',
        unsafe_allow_html=True,
    )
    _all_player_names = get_all_player_names() or []
    _picked = st.selectbox(
        "Search any player",
        options=_all_player_names,
        index=None,
        placeholder="Type a name — LeBron, Jordan, Magic, Jokić, Wembanyama…",
        label_visibility="collapsed",
        key="home_search_select",
    )
    if _picked:
        st.session_state["search_player"] = _picked
        try:
            st.switch_page("pages/Search.py")
        except Exception:
            # Older Streamlit fallback — render a link the user can click
            st.markdown(
                f'<a href="/Search" target="_top" style="color:#7ec8e8; text-decoration: underline;">'
                f'Click here to view {_picked}\'s profile →</a>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# SVG chart helpers
# ══════════════════════════════════════════════════════════════════════════════
def _esc(s) -> str:
    return html.escape(str(s))


def _hbar_chart(items, w=320, h=130, label_w=130, color_default="#e63946"):
    """Horizontal bar chart. items: list of dicts {label, value, value_str, color}."""
    if not items:
        return ""
    n = len(items)
    pad_top, pad_bot = 6, 6
    avail_h = h - pad_top - pad_bot
    bar_h   = avail_h / n - 4
    vmax    = max(it["value"] for it in items) or 1.0
    chart_x = label_w
    chart_w = w - label_w - 50

    parts = []
    for i, it in enumerate(items):
        y  = pad_top + i * (avail_h / n)
        bw = (it["value"] / vmax) * chart_w if vmax else 0
        c  = it.get("color", color_default)
        parts.append(
            f'<text x="6" y="{y + bar_h/2 + 4:.1f}" fill="#cfcfd6" font-size="11" '
            f'font-family="system-ui, -apple-system, sans-serif">{_esc(it["label"])}</text>'
        )
        parts.append(
            f'<rect x="{chart_x}" y="{y:.1f}" rx="3" ry="3" '
            f'width="{max(bw, 1):.1f}" height="{bar_h:.1f}" fill="{c}" opacity="0.88"/>'
        )
        parts.append(
            f'<text x="{chart_x + bw + 6:.1f}" y="{y + bar_h/2 + 4:.1f}" '
            f'fill="#fff" font-size="11" font-weight="700" '
            f'font-family="system-ui, -apple-system, sans-serif">{_esc(it["value_str"])}</text>'
        )
    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%; height:{h}px;">{"".join(parts)}</svg>'
    )


def _diverging_bars(rows, w=320, h=130):
    """rows: list of dicts {label, value, value_str, color, side: 'pos'|'neg'}."""
    if not rows:
        return ""
    n = len(rows)
    pad_top, pad_bot = 6, 6
    avail_h = h - pad_top - pad_bot
    bar_h   = avail_h / n - 4
    vmax    = max(abs(r["value"]) for r in rows) or 1.0
    cx      = w / 2
    half_w  = w / 2 - 70

    parts = [f'<line x1="{cx}" y1="{pad_top - 2}" x2="{cx}" y2="{h - pad_bot + 2}" '
             f'stroke="rgba(255,255,255,0.12)" stroke-width="1"/>']
    for i, r in enumerate(rows):
        y  = pad_top + i * (avail_h / n)
        bw = (abs(r["value"]) / vmax) * half_w
        if r["side"] == "neg":
            x_rect = cx - bw
            x_lbl  = cx - bw - 6
            anchor_lbl = "end"
            x_val  = cx + 4
            anchor_val = "start"
        else:
            x_rect = cx
            x_lbl  = cx + bw + 6
            anchor_lbl = "start"
            x_val  = cx - 4
            anchor_val = "end"

        parts.append(
            f'<rect x="{x_rect:.1f}" y="{y:.1f}" rx="3" ry="3" '
            f'width="{max(bw, 1):.1f}" height="{bar_h:.1f}" fill="{r["color"]}" opacity="0.88"/>'
        )
        parts.append(
            f'<text x="{x_lbl:.1f}" y="{y + bar_h/2 + 4:.1f}" text-anchor="{anchor_lbl}" '
            f'fill="#cfcfd6" font-size="10" '
            f'font-family="system-ui, -apple-system, sans-serif">{_esc(r["label"])}</text>'
        )
        parts.append(
            f'<text x="{x_val:.1f}" y="{y + bar_h/2 + 4:.1f}" text-anchor="{anchor_val}" '
            f'fill="#fff" font-size="10" font-weight="700" '
            f'font-family="system-ui, -apple-system, sans-serif">{_esc(r["value_str"])}</text>'
        )
    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%; height:{h}px;">{"".join(parts)}</svg>'
    )


def _interp_color(c1: str, c2: str, t: float) -> str:
    """Linear interpolate between two #rrggbb colors."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _value_color(v: float, vmin: float, vmax: float,
                 low="#e74c3c", mid="#f1c40f", high="#2ecc71") -> str:
    """Map a value to a color along a low → mid → high gradient."""
    if vmax <= vmin:
        return mid
    t = (v - vmin) / (vmax - vmin)
    if t < 0.5:
        return _interp_color(low, mid, t * 2)
    return _interp_color(mid, high, (t - 0.5) * 2)


def _sparkline_gradient(points, w=320, h=130, gradient_id="lebron-grad"):
    """Sparkline whose line + dots are colored by value (red→gold→green)."""
    if not points:
        return ""
    pad_left, pad_right = 36, 36
    pad_top, pad_bot    = 12, 22

    n = len(points)
    vals = [p[1] for p in points]
    vmin = min(vals)
    vmax = max(vals)
    rng  = (vmax - vmin) or 1.0

    chart_w = w - pad_left - pad_right
    chart_h = h - pad_top  - pad_bot

    coords = []
    for i, (_, v) in enumerate(points):
        x = pad_left + (i / (n - 1)) * chart_w if n > 1 else w / 2
        y = pad_top  + chart_h - ((v - vmin) / rng) * chart_h
        coords.append((x, y))

    line_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area_points = (
        f"{coords[0][0]:.1f},{pad_top + chart_h} "
        + line_points
        + f" {coords[-1][0]:.1f},{pad_top + chart_h}"
    )

    # Vertical gradient — high y (low score) gets red, low y (high score) gets green.
    # The gradient bbox covers the polyline's bounds, so it auto-stretches across data range.
    defs = (
        f'<defs>'
        f'<linearGradient id="{gradient_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%"   stop-color="#2ecc71"/>'
        f'<stop offset="50%"  stop-color="#f1c40f"/>'
        f'<stop offset="100%" stop-color="#e74c3c"/>'
        f'</linearGradient>'
        f'<linearGradient id="{gradient_id}-fill" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%"   stop-color="#2ecc71" stop-opacity="0.18"/>'
        f'<stop offset="50%"  stop-color="#f1c40f" stop-opacity="0.10"/>'
        f'<stop offset="100%" stop-color="#e74c3c" stop-opacity="0.05"/>'
        f'</linearGradient>'
        f'</defs>'
    )

    # Per-dot color based on its value
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" '
        f'fill="{_value_color(v, vmin, vmax)}" stroke="#14142a" stroke-width="1.5"/>'
        for (x, y), v in zip(coords, vals)
    )

    first_lbl, last_lbl = points[0][0], points[-1][0]
    first_val, last_val = vals[0], vals[-1]

    label_first = (
        f'<text x="{coords[0][0]:.1f}" y="{h - 5}" text-anchor="middle" '
        f'fill="#888" font-size="10" font-family="system-ui">{_esc(first_lbl)}</text>'
    )
    label_last = (
        f'<text x="{coords[-1][0]:.1f}" y="{h - 5}" text-anchor="middle" '
        f'fill="#888" font-size="10" font-family="system-ui">{_esc(last_lbl)}</text>'
    )
    val_first = (
        f'<text x="{coords[0][0] - 7:.1f}" y="{coords[0][1] + 4:.1f}" text-anchor="end" '
        f'fill="{_value_color(first_val, vmin, vmax)}" '
        f'font-size="11" font-weight="700" font-family="system-ui">{first_val:.1f}</text>'
    )
    val_last = (
        f'<text x="{coords[-1][0] + 7:.1f}" y="{coords[-1][1] + 4:.1f}" text-anchor="start" '
        f'fill="{_value_color(last_val, vmin, vmax)}" '
        f'font-size="11" font-weight="700" font-family="system-ui">{last_val:.1f}</text>'
    )

    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%; height:{h}px;">'
        f'{defs}'
        f'<polygon points="{area_points}" fill="url(#{gradient_id}-fill)"/>'
        f'<polyline points="{line_points}" fill="none" stroke="url(#{gradient_id})" '
        f'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
        f'{dots}{label_first}{label_last}{val_first}{val_last}'
        f'</svg>'
    )


def _fa_category_chart(items, w=320, h=130):
    """4 horizontal bars of avg Barrett Score per FA category, with count next to label."""
    if not items:
        return ""
    n      = len(items)
    total  = sum(it["count"] for it in items)
    header_y = 11
    chart_y  = 22
    chart_h  = h - chart_y - 6
    row_h    = chart_h / n
    bar_h    = row_h * 0.62

    label_w = 92
    chart_x = label_w
    chart_w = w - chart_x - 50
    avg_max = max(it["avg_score"] for it in items) or 1.0

    parts = [
        f'<text x="{w/2:.1f}" y="{header_y}" text-anchor="middle" fill="#cfcfd6" '
        f'font-size="10.5" font-family="system-ui">'
        f'<tspan fill="#fff" font-weight="700">{int(total)}</tspan> free agents · avg Barrett Score</text>'
    ]
    for i, it in enumerate(items):
        y  = chart_y + i * row_h + (row_h - bar_h) / 2
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
            f'fill="#fff" font-size="10.5" font-weight="700" font-family="system-ui">'
            f'{it["avg_score"]:.1f}</text>'
        )
    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%; height:{h}px;">{"".join(parts)}</svg>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# Compute live data for charts
# ══════════════════════════════════════════════════════════════════════════════
def _compute_charts():
    try:
        df = build_ranked_projected(SEASONS[0])
        df = df[df["total_min"] >= DEFAULT_MIN_THRESHOLD].copy()
        if df.empty:
            return None

        top5 = df.nsmallest(5, "score_rank")[["Player", "barrett_score"]].values.tolist()

        steals_3   = df.nsmallest(3, "value_diff")[["Player", "value_diff"]].values.tolist()
        overpaid_3 = df.nlargest(3,  "value_diff")[["Player", "value_diff"]].values.tolist()

        team_eff = (
            df.groupby("Team")
              .apply(lambda g: (g["salary"].sum() - g["projected_salary"].sum()) / 1e6)
              .sort_values()
        )
        best_teams  = list(team_eff.head(3).items())
        worst_teams = list(team_eff.tail(3).items())

        next_contracts = fetch_next_year_contracts(season_to_espn_year(SEASONS[0]), cache_v=7)
        rookie_scale   = fetch_rookie_scale_players(SEASONS[0])

        ufa_scores, rfa_scores, po_scores, to_scores = [], [], [], []
        for _, row in df[["Player", "barrett_score"]].iterrows():
            name  = row["Player"]
            score = float(row["barrett_score"])
            nc    = fmt_next_contract(name, next_contracts)
            if nc == "RFA":
                rfa_scores.append(score)
            elif nc == "—":
                if normalize(name) in rookie_scale:
                    rfa_scores.append(score)
                else:
                    ufa_scores.append(score)
            elif " PO" in nc:
                po_scores.append(score)
            elif " TO" in nc:
                to_scores.append(score)

        def _avg(xs):
            return sum(xs) / len(xs) if xs else 0.0

        n_ufa, n_rfa, n_po, n_to = len(ufa_scores), len(rfa_scores), len(po_scores), len(to_scores)
        avg_ufa, avg_rfa, avg_po, avg_to = _avg(ufa_scores), _avg(rfa_scores), _avg(po_scores), _avg(to_scores)

        try:
            lebron_df = fetch_career_trend(LEBRON_PLAYER_ID, num_seasons=len(SEASONS))
            lebron_career = [
                (str(row["Season"]).split("-")[0], float(row["barrett_score"]))
                for _, row in lebron_df.iterrows()
            ]
        except Exception:
            lebron_career = []

        return {
            "season":       SEASONS[0],
            "n_players":    len(df),
            "top5":         top5,
            "steals_3":     [(str(p), float(v)) for p, v in steals_3],
            "overpaid_3":   [(str(p), float(v)) for p, v in overpaid_3],
            "best_teams":   [(str(t), float(v)) for t, v in best_teams],
            "worst_teams": [(str(t), float(v)) for t, v in worst_teams],
            "fa_categories": [
                {"label": "UFA", "count": n_ufa, "avg_score": avg_ufa, "color": "#aaaaaa"},
                {"label": "RFA", "count": n_rfa, "avg_score": avg_rfa, "color": "#2ecc71"},
                {"label": "PO",  "count": n_po,  "avg_score": avg_po,  "color": "#3498db"},
                {"label": "TO",  "count": n_to,  "avg_score": avg_to,  "color": "#f39c12"},
            ],
            "lebron_career": lebron_career,
            "n_indexed_players": len(_all_player_names) if _all_player_names else 0,
        }
    except Exception:
        return None


_p = _compute_charts()


def _wrap_chart(svg: str) -> str:
    if not svg:
        return '<div class="nav-chart"><div class="nav-chart-empty">Loading live data…</div></div>'
    return f'<div class="nav-chart">{svg}</div>'


# ── Build per-card chart HTML ─────────────────────────────────────────────────
if _p:
    rank_colors = ["#f1c40f", "#e8b923", "#c9a02e", "#a78232", "#7a6334"]
    rankings_chart = _wrap_chart(_hbar_chart([
        {
            "label":     f"{i+1}. {name.split()[-1] if len(name.split()) > 1 else name}",
            "value":     score,
            "value_str": f"{score:.1f}",
            "color":     rank_colors[i],
        }
        for i, (name, score) in enumerate(_p["top5"])
    ]))

    vis_rows = []
    for name, vd in _p["steals_3"]:
        amt = abs(vd) / 1e6
        vis_rows.append({
            "label":     name.split()[-1],
            "value":     amt,
            "value_str": f"-${amt:.1f}M",
            "color":     "#2ecc71",
            "side":      "neg",
        })
    for name, vd in _p["overpaid_3"]:
        amt = vd / 1e6
        vis_rows.append({
            "label":     name.split()[-1],
            "value":     amt,
            "value_str": f"+${amt:.1f}M",
            "color":     "#e74c3c",
            "side":      "pos",
        })
    vis_chart = _wrap_chart(_diverging_bars(vis_rows))

    team_rows = []
    for t, v in _p["best_teams"]:
        team_rows.append({
            "label":     t, "value": abs(v),
            "value_str": f"-${abs(v):.1f}M",
            "color":     "#2ecc71", "side": "neg",
        })
    for t, v in _p["worst_teams"]:
        team_rows.append({
            "label":     t, "value": abs(v),
            "value_str": f"+${abs(v):.1f}M",
            "color":     "#e74c3c", "side": "pos",
        })
    team_chart = _wrap_chart(_diverging_bars(team_rows))

    fa_chart = _wrap_chart(_fa_category_chart(_p["fa_categories"]))
else:
    rankings_chart = vis_chart = team_chart = fa_chart = _wrap_chart("")

if _p and _p["lebron_career"]:
    _lebron_svg = _sparkline_gradient(_p["lebron_career"])
    legacy_chart = (
        '<div class="nav-chart">'
        f'{_lebron_svg}'
        '<div class="nav-chart-caption">Featured arc — LeBron James</div>'
        '</div>'
    )
else:
    legacy_chart = _wrap_chart("")


# ── Search card teaser — list of categories ──────────────────────────────────
_n_indexed = (_p or {}).get("n_indexed_players", 0)
search_chart = (
    '<div class="nav-chart" style="justify-content: center; padding: 1rem 0.6rem;">'
    f'  <div style="text-align:center; font-size:1.5rem; font-weight:800; color:#7ec8e8; line-height:1;">'
    f'    {_n_indexed:,}'
    f'  </div>'
    f'  <div style="text-align:center; font-size:0.78rem; color:#aaa; margin-top:0.3rem;">'
    f'    players indexed'
    f'  </div>'
    f'  <div style="text-align:center; font-size:0.72rem; color:#888; margin-top:0.7rem; line-height:1.5;">'
    f'    All-time greats · Active stars<br>Pre-1996 legends · 1984 → today'
    f'  </div>'
    '</div>'
)


# ── Trades card teaser — show one famous historical trade with the math ──────
def _build_trades_teaser():
    if not HISTORICAL_TRADES:
        return _wrap_chart("")
    # Pick the Harden→Houston trade — one of the most famously lopsided
    pick = next((t for t in HISTORICAL_TRADES if "Harden" in t["name"]), HISTORICAL_TRADES[0])
    season = pick["season"]
    sum_a = trade_side_summary(tuple(pick["side_a"]), season)
    sum_b = trade_side_summary(tuple(pick["side_b"]), season)
    a_total = sum_a["barrett_total"]
    b_total = sum_b["barrett_total"]
    if a_total == 0 and b_total == 0:
        return _wrap_chart("")  # data not yet seeded
    rows = [
        {"label": pick["side_a_team"].split()[-1], "value": a_total,
         "value_str": f"{a_total:.1f}",
         "color": "#2ecc71" if a_total >= b_total else "#e63946", "side": "pos"},
        {"label": pick["side_b_team"].split()[-1], "value": b_total,
         "value_str": f"{b_total:.1f}",
         "color": "#2ecc71" if b_total > a_total else "#e63946", "side": "pos"},
    ]
    body = _hbar_chart(rows, w=320, h=110, label_w=110)
    caption = f'Featured trade — {pick["name"]}'
    return (
        '<div class="nav-chart">'
        f'{body}'
        f'<div class="nav-chart-caption">{caption}</div>'
        '</div>'
    )

trades_chart = _build_trades_teaser() if _p else _wrap_chart("")


# ── Nav cards — row 1: Rankings · Search · Legacy ────────────────────────────
r1c1, r1c2, r1c3 = st.columns(3, gap="medium")
with r1c1:
    st.markdown(f"""
    <a class="nav-card" href="/Rankings" target="_top" style="--accent:#e63946;">
        <div class="nav-title">Current Rankings</div>
        <div class="nav-desc">Who's the best NBA player right now? See where every player stacks up this season.</div>
        {rankings_chart}
        <span class="nav-cta">Open Rankings →</span>
    </a>
    """, unsafe_allow_html=True)
with r1c2:
    st.markdown(f"""
    <a class="nav-card" href="/Search" target="_top" style="--accent:#7ec8e8;">
        <div class="nav-title">Search Player</div>
        <div class="nav-desc">Find any player from 1984 to today — full career arcs and per-season stats.</div>
        {search_chart}
        <span class="nav-cta">Open Search →</span>
    </a>
    """, unsafe_allow_html=True)
with r1c3:
    st.markdown(f"""
    <a class="nav-card" href="/Legacy" target="_top" style="--accent:#f1c40f;">
        <div class="nav-title">Legacy</div>
        <div class="nav-desc">42 seasons of NBA history — career arcs, all-time greats, and team legends.</div>
        {legacy_chart}
        <span class="nav-cta">Open Legacy →</span>
    </a>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

# ── Nav cards — row 2: Team Analysis · Trades · Free Agency ──────────────────
r2c1, r2c2, r2c3 = st.columns(3, gap="medium")
with r2c1:
    st.markdown(f"""
    <a class="nav-card" href="/Team_Analysis" target="_top" style="--accent:#3498db;">
        <div class="nav-title">Team Analysis</div>
        <div class="nav-desc">Which front offices are getting the most for their money?</div>
        {team_chart}
        <span class="nav-cta">Open Teams →</span>
    </a>
    """, unsafe_allow_html=True)
with r2c2:
    st.markdown(f"""
    <a class="nav-card" href="/Trades" target="_top" style="--accent:#9b59b6;">
        <div class="nav-title">Trades</div>
        <div class="nav-desc">Stack any two trade sides head-to-head — past trades or your own.</div>
        {trades_chart}
        <span class="nav-cta">Open Trades →</span>
    </a>
    """, unsafe_allow_html=True)
with r2c3:
    st.markdown(f"""
    <a class="nav-card" href="/Free_Agent_Class" target="_top" style="--accent:#2ecc71;">
        <div class="nav-title">Current Free Agents</div>
        <div class="nav-desc">Every player hitting the market this offseason — and what they're worth.</div>
        {fa_chart}
        <span class="nav-cta">Open Free Agency →</span>
    </a>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:0.6rem; color:#888; font-size:0.7rem;">
    barrettscore.com &nbsp;·&nbsp; Data from NBA Stats API &nbsp;·&nbsp; Updated daily
</div>
""", unsafe_allow_html=True)
