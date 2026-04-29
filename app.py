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
    fmt_next_contract,
    season_to_espn_year,
    normalize,
    SEASONS,
    DEFAULT_MIN_THRESHOLD,
)

# Start warming all season caches the moment the server boots —
# before any user arrives, so the first visitor doesn't pay the cost.
_bootstrap_warm()

st.set_page_config(page_title="Barrett Score", layout="wide")

# ── Hide Streamlit chrome & sidebar nav ────────────────────────────────────────
st.markdown("""
<style>
    /* Aggressively kill the Streamlit header/toolbar height that creates the giant top gap */
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

    /* Cover every selector Streamlit has used for the main block padding */
    .block-container,
    .main .block-container,
    section.main > .block-container,
    [data-testid="stMain"] .block-container,
    [data-testid="stMainBlockContainer"],
    section[data-testid="stMain"] > .block-container {
        padding-top: 0.75rem !important;
        padding-bottom: 1rem !important;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
        max-width: 100%;
    }
    .stApp { padding-top: 0 !important; }

    /* Clickable nav card */
    a.nav-card {
        display: flex;
        flex-direction: column;
        background: #14142a;
        border: 1px solid #2a2a4a;
        border-radius: 14px;
        padding: 1.4rem 1.3rem 1.2rem 1.3rem;
        text-decoration: none;
        transition: border-color 0.2s, transform 0.15s, box-shadow 0.2s;
        cursor: pointer;
        min-height: 380px;
        box-sizing: border-box;
        position: relative;
        overflow: hidden;
    }
    a.nav-card:hover {
        border-color: var(--accent, #e63946);
        transform: translateY(-3px);
        text-decoration: none;
        box-shadow: 0 10px 28px rgba(230, 57, 70, 0.14);
    }
    /* Color accent stripe at the top of each card */
    a.nav-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--accent, #e63946);
        opacity: 0.85;
    }
    .nav-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        color: #fff;
        text-align: center;
    }
    .nav-desc {
        font-size: 0.78rem;
        color: #888;
        line-height: 1.4;
        text-align: center;
        margin-bottom: 0.7rem;
    }
    /* Chart container — fills the middle of the card */
    .nav-chart {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 0.4rem 0.2rem;
        background: rgba(0, 0, 0, 0.25);
        border-radius: 10px;
        margin-bottom: 0.9rem;
        min-height: 160px;
    }
    .nav-chart svg { display: block; }
    .nav-chart-empty {
        font-size: 0.78rem;
        color: #666;
        text-align: center;
        font-style: italic;
        padding: 1rem;
    }

    .nav-cta {
        align-self: center;
        background: #e63946;
        color: #fff !important;
        border-radius: 8px;
        padding: 0.45rem 1.4rem;
        font-weight: 600;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    a.nav-card:hover .nav-cta { background: #c1121f; }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 0.25rem 0 1rem 0;">
    <div style="font-size:2.5rem; font-weight:800; letter-spacing:-1px; color:#fff; line-height:1.1;">
        Barrett Score
    </div>
    <div style="font-size:1rem; color:#999; margin-top:0.35rem;">
        A stat-driven NBA player valuation tool — who's underpaid, overpaid, and available.
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SVG chart helpers
# ══════════════════════════════════════════════════════════════════════════════
def _esc(s) -> str:
    return html.escape(str(s))


def _hbar_chart(items, w=320, h=170, label_w=130, color_default="#e63946"):
    """Horizontal bar chart. items: list of dicts {label, value, value_str, color}."""
    if not items:
        return ""
    n = len(items)
    pad_top, pad_bot = 8, 8
    avail_h = h - pad_top - pad_bot
    bar_h   = avail_h / n - 5
    vmax    = max(it["value"] for it in items) or 1.0
    chart_x = label_w
    chart_w = w - label_w - 50  # leave 50px for value label

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


def _diverging_bars(rows, w=320, h=170):
    """rows: list of dicts {label, value, value_str, color, side: 'pos'|'neg'}.
    Each bar grows outward from a vertical center axis.
    """
    if not rows:
        return ""
    n = len(rows)
    pad_top, pad_bot = 8, 8
    avail_h = h - pad_top - pad_bot
    bar_h   = avail_h / n - 5
    vmax    = max(abs(r["value"]) for r in rows) or 1.0
    cx      = w / 2
    half_w  = w / 2 - 70  # leave room for labels

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
            f'fill="#cfcfd6" font-size="11" '
            f'font-family="system-ui, -apple-system, sans-serif">{_esc(r["label"])}</text>'
        )
        parts.append(
            f'<text x="{x_val:.1f}" y="{y + bar_h/2 + 4:.1f}" text-anchor="{anchor_val}" '
            f'fill="#fff" font-size="11" font-weight="700" '
            f'font-family="system-ui, -apple-system, sans-serif">{_esc(r["value_str"])}</text>'
        )
    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%; height:{h}px;">{"".join(parts)}</svg>'
    )


def _sparkline(points, labels=None, w=320, h=170, color="#f1c40f"):
    """points: list of (label, value). Renders a line + dots + endpoint labels."""
    if not points:
        return ""
    pad_x, pad_y = 14, 24
    n = len(points)
    vals = [p[1] for p in points]
    vmin = min(vals)
    vmax = max(vals)
    rng  = (vmax - vmin) or 1.0

    chart_w = w - pad_x * 2
    chart_h = h - pad_y * 2

    coords = []
    for i, (_, v) in enumerate(points):
        x = pad_x + (i / (n - 1)) * chart_w if n > 1 else w / 2
        y = pad_y + chart_h - ((v - vmin) / rng) * chart_h
        coords.append((x, y))

    line_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area_points = f"{coords[0][0]:.1f},{pad_y + chart_h} " + line_points + f" {coords[-1][0]:.1f},{pad_y + chart_h}"

    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" stroke="#14142a" stroke-width="1.5"/>'
        for x, y in coords
    )

    # Labels at first and last point + min/max value badges
    first_lbl = points[0][0]
    last_lbl  = points[-1][0]
    first_val = vals[0]
    last_val  = vals[-1]

    label_first = (
        f'<text x="{coords[0][0]:.1f}" y="{h - 6}" text-anchor="middle" '
        f'fill="#888" font-size="10" font-family="system-ui">{_esc(first_lbl)}</text>'
    )
    label_last = (
        f'<text x="{coords[-1][0]:.1f}" y="{h - 6}" text-anchor="middle" '
        f'fill="#888" font-size="10" font-family="system-ui">{_esc(last_lbl)}</text>'
    )

    val_first = (
        f'<text x="{coords[0][0]:.1f}" y="{coords[0][1] - 8:.1f}" text-anchor="middle" '
        f'fill="#fff" font-size="10" font-weight="700" font-family="system-ui">{first_val:.1f}</text>'
    )
    val_last = (
        f'<text x="{coords[-1][0]:.1f}" y="{coords[-1][1] - 8:.1f}" text-anchor="middle" '
        f'fill="{color}" font-size="11" font-weight="700" font-family="system-ui">{last_val:.1f}</text>'
    )

    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%; height:{h}px;">'
        f'<polygon points="{area_points}" fill="{color}" opacity="0.10"/>'
        f'<polyline points="{line_points}" fill="none" stroke="{color}" stroke-width="2.2" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'{dots}{label_first}{label_last}{val_first}{val_last}'
        f'</svg>'
    )


def _stacked_bar(segments, total_label="Total", w=320, h=170):
    """segments: list of dicts {label, value, color}. Renders a single horizontal stacked bar
    with a legend underneath.
    """
    total = sum(s["value"] for s in segments) or 1
    bar_x = 14
    bar_y = 24
    bar_w = w - 28
    bar_h = 36

    # Title text
    parts = [
        f'<text x="{w/2:.1f}" y="14" text-anchor="middle" fill="#cfcfd6" font-size="11" '
        f'font-family="system-ui">{_esc(total_label)}: '
        f'<tspan fill="#fff" font-weight="700">{int(total)}</tspan></text>'
    ]

    # Stacked rect
    x_cursor = bar_x
    for s in segments:
        seg_w = (s["value"] / total) * bar_w if total else 0
        parts.append(
            f'<rect x="{x_cursor:.1f}" y="{bar_y}" width="{seg_w:.1f}" height="{bar_h}" '
            f'fill="{s["color"]}" opacity="0.92"/>'
        )
        if seg_w > 30 and s["value"] > 0:
            parts.append(
                f'<text x="{x_cursor + seg_w/2:.1f}" y="{bar_y + bar_h/2 + 4:.1f}" '
                f'text-anchor="middle" fill="#fff" font-size="11" font-weight="700" '
                f'font-family="system-ui">{int(s["value"])}</text>'
            )
        x_cursor += seg_w

    # Legend
    legend_y = bar_y + bar_h + 22
    item_w   = w / max(len(segments), 1)
    for i, s in enumerate(segments):
        cx = i * item_w + 16
        parts.append(
            f'<rect x="{cx:.1f}" y="{legend_y - 8}" width="10" height="10" rx="2" fill="{s["color"]}"/>'
        )
        parts.append(
            f'<text x="{cx + 14:.1f}" y="{legend_y + 1:.1f}" fill="#bbb" font-size="11" '
            f'font-family="system-ui">{_esc(s["label"])} '
            f'<tspan fill="#fff" font-weight="700">{int(s["value"])}</tspan></text>'
        )

    return (
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%; height:{h}px;">{"".join(parts)}</svg>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# Compute live data for charts (from already-warmed cache)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_charts():
    try:
        df = build_ranked_projected(SEASONS[0])
        df = df[df["total_min"] >= DEFAULT_MIN_THRESHOLD].copy()
        if df.empty:
            return None

        # Top 5 by Barrett Score
        top5 = df.nsmallest(5, "score_rank")[["Player", "barrett_score"]].values.tolist()

        # Biggest steal + most overpaid
        steal_row    = df.loc[df["value_diff"].idxmin()]
        overpaid_row = df.loc[df["value_diff"].idxmax()]

        # Team payroll efficiency: top 3 efficient + bottom 3 inefficient
        team_eff = (
            df.groupby("Team")
              .apply(lambda g: (g["salary"].sum() - g["projected_salary"].sum()) / 1e6)
              .sort_values()
        )
        # 3 most efficient (most negative = underspending)
        best_teams  = list(team_eff.head(3).items())
        # 3 least efficient (most positive = overspending)
        worst_teams = list(team_eff.tail(3).items())

        # Free agent breakdown
        next_contracts = fetch_next_year_contracts(season_to_espn_year(SEASONS[0]), cache_v=7)
        rookie_scale   = fetch_rookie_scale_players(SEASONS[0])

        n_ufa = n_rfa = n_po = n_to = 0
        for name in df["Player"]:
            nc = fmt_next_contract(name, next_contracts)
            if nc == "RFA":
                n_rfa += 1
            elif nc == "—":
                # rookie scale player about to become RFA, otherwise UFA
                if normalize(name) in rookie_scale:
                    n_rfa += 1
                else:
                    n_ufa += 1
            elif " PO" in nc:
                n_po += 1
            elif " TO" in nc:
                n_to += 1

        # Top barrett-score per season for last 10 seasons (sparkline)
        # Each call is a cache hit since seasons get warmed in the background.
        spark = []
        for s in list(reversed(SEASONS[:10])):  # chronological order, oldest first
            try:
                sdf = build_ranked_projected(s)
                sdf = sdf[sdf["total_min"] >= DEFAULT_MIN_THRESHOLD]
                if not sdf.empty:
                    spark.append((s.split("-")[0], float(sdf["barrett_score"].max())))
            except Exception:
                pass

        return {
            "season":       SEASONS[0],
            "n_players":    len(df),
            "top5":         top5,
            "steal_name":   str(steal_row["Player"]),
            "steal_amt":    abs(float(steal_row["value_diff"])) / 1e6,
            "over_name":    str(overpaid_row["Player"]),
            "over_amt":     float(overpaid_row["value_diff"]) / 1e6,
            "best_teams":   [(str(t), float(v)) for t, v in best_teams],
            "worst_teams":  [(str(t), float(v)) for t, v in worst_teams],
            "fa_segments":  [
                {"label": "UFA", "value": n_ufa, "color": "#aaaaaa"},
                {"label": "RFA", "value": n_rfa, "color": "#2ecc71"},
                {"label": "PO",  "value": n_po,  "color": "#3498db"},
                {"label": "TO",  "value": n_to,  "color": "#f39c12"},
            ],
            "spark":        spark,
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
    # Rankings — top 5 horizontal bars (gradient gold→muted)
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

    # Visualizer — diverging-style bars (steal vs overpaid)
    vis_chart = _wrap_chart(_diverging_bars([
        {
            "label":     _p["steal_name"].split()[-1],
            "value":     _p["steal_amt"],
            "value_str": f"-${_p['steal_amt']:.1f}M",
            "color":     "#2ecc71",
            "side":      "neg",
        },
        {
            "label":     _p["over_name"].split()[-1],
            "value":     _p["over_amt"],
            "value_str": f"+${_p['over_amt']:.1f}M",
            "color":     "#e74c3c",
            "side":      "pos",
        },
    ]))

    # Team Analysis — top 3 efficient (green, neg side) + 3 inefficient (red, pos side)
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

    # Free Agents — stacked bar with 4 segments
    fa_chart = _wrap_chart(_stacked_bar(_p["fa_segments"], total_label="Free Agents"))
else:
    rankings_chart = vis_chart = team_chart = fa_chart = _wrap_chart("")

# Legacy — sparkline of top Barrett Score per season for the last 10 seasons
if _p and _p["spark"]:
    legacy_chart = _wrap_chart(_sparkline(_p["spark"], color="#f1c40f"))
else:
    legacy_chart = _wrap_chart("")


# ── Nav cards — row 1: Rankings + Legacy (2 centered) ─────────────────────────
_, col1, col2, _ = st.columns([0.5, 1, 1, 0.5], gap="medium")
with col1:
    st.markdown(f"""
    <a class="nav-card" href="/Rankings" target="_top" style="--accent:#e63946;">
        <div class="nav-title">Current Rankings</div>
        <div class="nav-desc">Top 5 by Barrett Score · {_p['n_players'] if _p else '—'} players ranked</div>
        {rankings_chart}
        <span class="nav-cta">Open Rankings →</span>
    </a>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <a class="nav-card" href="/Legacy" target="_top" style="--accent:#f1c40f;">
        <div class="nav-title">Legacy</div>
        <div class="nav-desc">Best Barrett Score per season · last 10 years</div>
        {legacy_chart}
        <span class="nav-cta">Open Legacy →</span>
    </a>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)

# ── Nav cards — row 2: Visualizer, Team Analysis, Free Agent Class ────────────
col3, col4, col5 = st.columns(3, gap="medium")
with col3:
    st.markdown(f"""
    <a class="nav-card" href="/Salary_Projector" target="_top" style="--accent:#9b59b6;">
        <div class="nav-title">Visualizer</div>
        <div class="nav-desc">Biggest steal vs most overpaid this season</div>
        {vis_chart}
        <span class="nav-cta">Open Visualizer →</span>
    </a>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <a class="nav-card" href="/Team_Analysis" target="_top" style="--accent:#3498db;">
        <div class="nav-title">Team Analysis</div>
        <div class="nav-desc">Best vs worst payroll efficiency · current season</div>
        {team_chart}
        <span class="nav-cta">Open Teams →</span>
    </a>
    """, unsafe_allow_html=True)
with col5:
    st.markdown(f"""
    <a class="nav-card" href="/Free_Agent_Class" target="_top" style="--accent:#2ecc71;">
        <div class="nav-title">Current Free Agents</div>
        <div class="nav-desc">UFA · RFA · player options · team options</div>
        {fa_chart}
        <span class="nav-cta">Open Free Agency →</span>
    </a>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:2rem; color:#555; font-size:0.78rem;">
    barrettscore.com &nbsp;·&nbsp; Data from NBA Stats API &nbsp;·&nbsp; Updated daily
</div>
""", unsafe_allow_html=True)
