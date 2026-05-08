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
    fetch_player_career_with_rank,
    fmt_next_contract,
    season_to_espn_year,
    normalize,
    SEASONS,
    DEFAULT_MIN_THRESHOLD,
    get_all_player_names,
    HISTORICAL_TRADES,
    trade_side_summary,
)

# Featured players for the Legacy preview overlay on the home page.
# IDs come from nba_api.stats.static.players. Order matters — earliest first
# so older legends sit at the bottom of the legend.
LEGACY_FEATURED = [
    {"name": "Michael Jordan",  "id":    893, "color": "#f1c40f"},  # gold
    {"name": "Kobe Bryant",     "id":    977, "color": "#9b59b6"},  # purple
    {"name": "LeBron James",    "id":   2544, "color": "#e63946"},  # red
    {"name": "Nikola Jokić",    "id": 203999, "color": "#7ec8e8"},  # blue
]

# Start warming all season caches the moment the server boots —
# before any user arrives, so the first visitor doesn't pay the cost.
_bootstrap_warm()

st.set_page_config(page_title="Barrett Score", layout="wide")

# ── Page chrome (background, hide Streamlit UI) ────────────────────────────────
st.markdown("""
<style>
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

    /* Horizontal tab strips — one per nav page */
    a.tab-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(20, 20, 42, 0.55);
        border: 1px solid rgba(80, 80, 110, 0.35);
        border-left: 3px solid var(--accent, #e63946);
        border-radius: 8px;
        padding: 0.85rem 1.2rem;
        text-decoration: none;
        margin-bottom: 0.55rem;
        transition: background-color 0.15s, border-color 0.15s, transform 0.1s;
        backdrop-filter: blur(2px);
    }
    a.tab-strip:hover {
        background: rgba(30, 30, 56, 0.85);
        border-color: var(--accent, #e63946);
        text-decoration: none;
        transform: translateX(2px);
    }
    .tab-strip-name {
        color: #fff;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-right: 1.2rem;
        flex-shrink: 0;
        min-width: 165px;
    }
    .tab-strip-desc {
        color: #aaa;
        font-size: 0.82rem;
        flex-grow: 1;
        line-height: 1.35;
    }
    .tab-strip-arrow {
        color: var(--accent, #e63946);
        font-size: 1.2rem;
        font-weight: 600;
        margin-left: 1rem;
        flex-shrink: 0;
    }

    /* Streamlit expander styling — make it sit tight under each strip */
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
        color: #888 !important;
        font-size: 0.78rem !important;
        padding-left: 1.2rem !important;
        background: transparent !important;
    }
    div[data-testid="stExpander"] summary:hover { color: #ddd !important; }
    .preview-box {
        background: rgba(0, 0, 0, 0.3);
        border-left: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        padding: 0.8rem 1rem;
        margin-left: 1rem;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero — title + intro blurb ────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 0.4rem 0 0.6rem 0;">
    <div style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, system-ui, sans-serif;
        font-size: 2.2rem;
        line-height: 1;
        letter-spacing: 0.1em;
        text-shadow: 0 3px 14px rgba(0,0,0,0.55);
        white-space: nowrap;
    ">
        <span style="color: #c8cdd6; font-weight: 600;">THE&nbsp;</span><span style="color: #ffffff; font-weight: 800;">BARRETT&nbsp;</span><span style="color: #7ec8e8; font-weight: 800;">SCORE</span>
    </div>
    <div style="font-size:0.88rem; color:#cdcdd5; margin-top:0.55rem; max-width:760px; margin-left:auto; margin-right:auto; line-height:1.45; text-shadow: 0 1px 6px rgba(0,0,0,0.5);">
        Scoring, playmaking, defense, and efficiency — distilled into one number, then put next to what each player gets paid.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Search hero ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSelectbox"][data-baseweb] div[role="combobox"] {
    background: rgba(20, 20, 42, 0.7) !important;
    border: 2px solid rgba(126, 200, 232, 0.55) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(6px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}
.home-search-label {
    font-size: 0.78rem;
    color: #d0d0d6;
    text-align: center;
    margin-bottom: 0.35rem;
    text-shadow: 0 1px 6px rgba(0,0,0,0.6);
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

_, _search_col, _ = st.columns([1, 2, 1])
with _search_col:
    st.markdown(
        '<div class="home-search-label">SEARCH ANY PLAYER · CAREER ARCS · HEAD-TO-HEAD COMPARISONS · 1984 → TODAY</div>',
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
            st.markdown(
                f'<a href="/Search" target="_top" style="color:#7ec8e8; text-decoration: underline;">'
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
            f'<text x="6" y="{y + bar_h/2 + 4:.1f}" fill="#cfcfd6" font-size="11" '
            f'font-family="system-ui">{_esc(it["label"])}</text>'
        )
        parts.append(
            f'<rect x="{chart_x}" y="{y:.1f}" rx="3" ry="3" '
            f'width="{max(bw, 1):.1f}" height="{bar_h:.1f}" fill="{c}" opacity="0.88"/>'
        )
        parts.append(
            f'<text x="{chart_x + bw + 6:.1f}" y="{y + bar_h/2 + 4:.1f}" '
            f'fill="#fff" font-size="11" font-weight="700" '
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
            f'fill="#cfcfd6" font-size="10" font-family="system-ui">{_esc(r["label"])}</text>'
        )
        parts.append(
            f'<text x="{x_val:.1f}" y="{y + bar_h/2 + 4:.1f}" text-anchor="{anchor_val}" '
            f'fill="#fff" font-size="10" font-weight="700" '
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


def _multi_sparkline(series_list, w=460, h=160):
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
        parts.append(
            f'<polyline points="{line_pts}" fill="none" stroke="{s["color"]}" '
            f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>'
        )
        for x, y, pt in coords:
            tooltip = (
                f'{_esc(s["name"])} · {_esc(pt["season"])}'
                f'  —  Barrett {pt["score"]:.2f}'
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
        f'<text x="{w/2:.1f}" y="11" text-anchor="middle" fill="#cfcfd6" '
        f'font-size="10.5" font-family="system-ui">'
        f'<tspan fill="#fff" font-weight="700">{int(total)}</tspan> free agents · avg Barrett Score</text>'
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
            f'fill="#fff" font-size="10.5" font-weight="700" font-family="system-ui">'
            f'{it["avg_score"]:.1f}</text>'
        )
    return f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" style="width:100%; height:{h}px;">{"".join(parts)}</svg>'


# ══════════════════════════════════════════════════════════════════════════════
# Compute live data for previews (runs once, cached output)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_charts():
    try:
        df = build_ranked_projected(SEASONS[0])
        df = df[df["total_min"] >= DEFAULT_MIN_THRESHOLD].copy()
        if df.empty:
            return None

        top10 = df.nsmallest(10, "score_rank")[["Player", "barrett_score"]].values.tolist()
        steals_3 = df.nsmallest(3, "value_diff")[["Player", "value_diff"]].values.tolist()
        overpaid_3 = df.nlargest(3, "value_diff")[["Player", "value_diff"]].values.tolist()

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
        for _, row in df[["Player", "barrett_score"]].iterrows():
            name = row["Player"]
            score = float(row["barrett_score"])
            nc = fmt_next_contract(name, next_contracts)
            if nc == "RFA":
                rfa.append(score)
            elif nc == "—":
                if normalize(name) in rookie_scale:
                    rfa.append(score)
                else:
                    ufa.append(score)
            elif " PO" in nc:
                po.append(score)
            elif " TO" in nc:
                to.append(score)

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
# Render one tab strip + collapsible preview
# ══════════════════════════════════════════════════════════════════════════════
def render_strip(name: str, href: str, accent: str, description: str, preview_html: str):
    st.markdown(f"""
    <a class="tab-strip" href="{href}" target="_top" style="--accent:{accent};">
        <div class="tab-strip-name">{name}</div>
        <div class="tab-strip-desc">{description}</div>
        <span class="tab-strip-arrow">→</span>
    </a>
    """, unsafe_allow_html=True)
    with st.expander("Preview", expanded=False):
        st.markdown(f'<div class="preview-box">{preview_html}</div>', unsafe_allow_html=True)


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
    ], w=460, h=260, label_w=150) + '<div style="text-align:center; font-size:0.7rem; color:#777; margin-top:0.4rem;">Top 10 by Barrett Score · this season</div>'

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

    teams_preview = _diverging_bars(team_rows) + '<div style="text-align:center; font-size:0.7rem; color:#777; margin-top:0.4rem;">Net payroll efficiency · green = team is winning the value game</div>'
    fa_preview = _fa_category_chart(_p["fa_categories"]) + '<div style="text-align:center; font-size:0.7rem; color:#777; margin-top:0.4rem;">Free-agent class breakdown · this offseason</div>'
    # Legacy preview is rendered specially below (needs an interactive radio
    # for the user to pick a featured player), so we just stash the data here.
    legacy_preview = None

    # Trades preview — Harden→Houston featured trade
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
        rows = [
            {"label": pick["side_a_team"].split()[-1], "value": a_total,
             "value_str": f"{a_total:.1f}",
             "color": "#2ecc71" if a_total >= b_total else "#e63946", "side": "pos"},
            {"label": pick["side_b_team"].split()[-1], "value": b_total,
             "value_str": f"{b_total:.1f}",
             "color": "#2ecc71" if b_total > a_total else "#e63946", "side": "pos"},
        ]
        return _hbar_chart(rows, w=460, h=110, label_w=110) + f'<div style="text-align:center; font-size:0.7rem; color:#777; margin-top:0.4rem;">Featured trade — {pick["name"]}</div>'

    trades_preview = _build_trades_preview()
else:
    rankings_preview = teams_preview = fa_preview = trades_preview = "<em>Loading live data…</em>"


# ══════════════════════════════════════════════════════════════════════════════
# Render 6 tab strips
# ══════════════════════════════════════════════════════════════════════════════
render_strip(
    name="Current Rankings",
    href="/Rankings",
    accent="#e63946",
    description="Who's the best NBA player right now? Every player ranked by Barrett Score this season.",
    preview_html=rankings_preview,
)

# ── Legacy strip — special-cased for the interactive player picker ───────────
st.markdown(f"""
<a class="tab-strip" href="/Legacy" target="_top" style="--accent:#f1c40f;">
    <div class="tab-strip-name">Legacy</div>
    <div class="tab-strip-desc">42 seasons of NBA history — all-time greats, era leaderboards, team Mount Rushmores, draft classes.</div>
    <span class="tab-strip-arrow">→</span>
</a>
""", unsafe_allow_html=True)
with st.expander("Preview", expanded=False):
    legacy_series = (_p or {}).get("legacy_series", [])
    if not legacy_series:
        st.markdown('<em>Loading live data…</em>', unsafe_allow_html=True)
    else:
        names = [s["name"] for s in legacy_series]
        # Default to LeBron (index 2 in LEGACY_FEATURED)
        default_name = "LeBron James" if "LeBron James" in names else names[0]
        picked = st.radio(
            "Featured player",
            options=names,
            index=names.index(default_name),
            horizontal=True,
            label_visibility="collapsed",
            key="legacy_preview_pick",
        )
        chosen = next((s for s in legacy_series if s["name"] == picked), None)
        if chosen and chosen["career"]:
            chart_html = _multi_sparkline([chosen])
            n_seasons = len(chosen["career"])
            first = chosen["career"][0]["season"]
            last  = chosen["career"][-1]["season"]
            st.markdown(
                f'<div class="preview-box">'
                f'{chart_html}'
                f'<div style="text-align:center; font-size:0.7rem; color:#777; margin-top:0.4rem;">'
                f'{picked} · {n_seasons} seasons · {first} → {last}  ·  '
                f'<span style="color:#999;">hover any dot for details</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<em>No data on disk yet for {picked} — try after re-seeding.</em>',
                unsafe_allow_html=True,
            )

render_strip(
    name="Team Analysis",
    href="/Team_Analysis",
    accent="#3498db",
    description="Which front offices are getting the most for their money? Payroll efficiency by team.",
    preview_html=teams_preview,
)

render_strip(
    name="Trades",
    href="/Trades",
    accent="#9b59b6",
    description="Stack any two trade sides head-to-head — past trades or your own. Who actually came out ahead?",
    preview_html=trades_preview,
)

render_strip(
    name="Current Free Agents",
    href="/Free_Agent_Class",
    accent="#2ecc71",
    description="Every player hitting the market this offseason — UFAs, RFAs, options. What they're worth.",
    preview_html=fa_preview,
)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:1rem; color:#888; font-size:0.7rem;">
    barrettscore.com &nbsp;·&nbsp; Data from NBA Stats API &nbsp;·&nbsp; Updated daily
</div>
""", unsafe_allow_html=True)
