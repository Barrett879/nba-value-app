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

LEBRON_PLAYER_ID = 2544

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


def _sparkline_gradient(points, w=460, h=140, gradient_id="lebron-grad"):
    if not points:
        return ""
    pad_left, pad_right = 36, 36
    pad_top, pad_bot = 12, 22
    n = len(points)
    vals = [p[1] for p in points]
    vmin, vmax = min(vals), max(vals)
    rng = (vmax - vmin) or 1.0
    chart_w = w - pad_left - pad_right
    chart_h = h - pad_top - pad_bot
    coords = []
    for i, (_, v) in enumerate(points):
        x = pad_left + (i / (n - 1)) * chart_w if n > 1 else w / 2
        y = pad_top + chart_h - ((v - vmin) / rng) * chart_h
        coords.append((x, y))
    line_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area_points = (
        f"{coords[0][0]:.1f},{pad_top + chart_h} " + line_points +
        f" {coords[-1][0]:.1f},{pad_top + chart_h}"
    )
    defs = (
        f'<defs>'
        f'<linearGradient id="{gradient_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="#2ecc71"/><stop offset="50%" stop-color="#f1c40f"/>'
        f'<stop offset="100%" stop-color="#e74c3c"/></linearGradient>'
        f'<linearGradient id="{gradient_id}-fill" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="#2ecc71" stop-opacity="0.18"/>'
        f'<stop offset="50%" stop-color="#f1c40f" stop-opacity="0.10"/>'
        f'<stop offset="100%" stop-color="#e74c3c" stop-opacity="0.05"/></linearGradient></defs>'
    )
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{_value_color(v, vmin, vmax)}" '
        f'stroke="#14142a" stroke-width="1.5"/>'
        for (x, y), v in zip(coords, vals)
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%; height:{h}px;">{defs}'
        f'<polygon points="{area_points}" fill="url(#{gradient_id}-fill)"/>'
        f'<polyline points="{line_points}" fill="none" stroke="url(#{gradient_id})" '
        f'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>{dots}</svg>'
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

        top5 = df.nsmallest(5, "score_rank")[["Player", "barrett_score"]].values.tolist()
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

        try:
            lebron_df = fetch_career_trend(LEBRON_PLAYER_ID, num_seasons=len(SEASONS))
            lebron_career = [
                (str(row["Season"]).split("-")[0], float(row["barrett_score"]))
                for _, row in lebron_df.iterrows()
            ]
        except Exception:
            lebron_career = []

        return {
            "top5": top5,
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
            "lebron_career": lebron_career,
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
    rank_colors = ["#f1c40f", "#e8b923", "#c9a02e", "#a78232", "#7a6334"]
    rankings_preview = _hbar_chart([
        {
            "label":     f"{i+1}. {name.split()[-1] if len(name.split()) > 1 else name}",
            "value":     score,
            "value_str": f"{score:.1f}",
            "color":     rank_colors[i],
        }
        for i, (name, score) in enumerate(_p["top5"])
    ]) + '<div style="text-align:center; font-size:0.7rem; color:#777; margin-top:0.4rem;">Top 5 by Barrett Score · this season</div>'

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
    legacy_preview = _sparkline_gradient(_p["lebron_career"]) + '<div style="text-align:center; font-size:0.7rem; color:#777; margin-top:0.4rem;">LeBron\'s career arc · 22 seasons</div>' if _p["lebron_career"] else "<em>Loading…</em>"

    n_indexed = _p.get("n_indexed_players", 0)
    search_preview = (
        f'<div style="text-align:center; padding: 0.6rem 0.5rem;">'
        f'  <div style="font-size:1.6rem; font-weight:800; color:#7ec8e8; line-height:1;">'
        f'    {n_indexed:,}'
        f'  </div>'
        f'  <div style="font-size:0.78rem; color:#aaa; margin-top:0.3rem;">'
        f'    players indexed · 1984 → today'
        f'  </div>'
        f'  <div style="font-size:0.72rem; color:#888; margin-top:0.6rem; line-height:1.4;">'
        f'    All-time greats · Active stars · Pre-1996 legends · Two-way contracts'
        f'  </div>'
        f'</div>'
    )

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
    rankings_preview = teams_preview = fa_preview = legacy_preview = search_preview = trades_preview = "<em>Loading live data…</em>"


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

render_strip(
    name="Search Player",
    href="/Search",
    accent="#7ec8e8",
    description="Find any player from 1984 to today — career arcs, peak seasons, head-to-head comparisons.",
    preview_html=search_preview,
)

render_strip(
    name="Legacy",
    href="/Legacy",
    accent="#f1c40f",
    description="42 seasons of NBA history — all-time greats, era leaderboards, team Mount Rushmores, draft classes.",
    preview_html=legacy_preview,
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
