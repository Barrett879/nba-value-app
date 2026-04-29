import sys
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
    .main .block-container { padding-left: 0.5rem; padding-right: 0.5rem; max-width: 100%; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"]        { display: none !important; }
    [data-testid="stDecoration"]     { display: none !important; }
    [data-testid="stStatusWidget"]   { display: none !important; }
    [data-testid="stAppViewerBadge"] { display: none !important; }
    [data-testid="stBottom"]         { display: none !important; }
    [data-testid="stSidebarNav"]     { display: none !important; }
    [data-testid="stSidebar"]        { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    .viewerBadge_container__r5tak    { display: none !important; }
    .styles_viewerBadge__CvC9N       { display: none !important; }

    /* Clickable nav card */
    a.nav-card {
        display: flex;
        flex-direction: column;
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.6rem 1.4rem 1.4rem 1.4rem;
        text-decoration: none;
        transition: border-color 0.2s, transform 0.15s, box-shadow 0.2s;
        cursor: pointer;
        min-height: 300px;
        box-sizing: border-box;
        position: relative;
        overflow: hidden;
    }
    a.nav-card:hover {
        border-color: #e63946;
        transform: translateY(-3px);
        text-decoration: none;
        box-shadow: 0 8px 24px rgba(230, 57, 70, 0.12);
    }
    /* Color accent stripe at the top of each card */
    a.nav-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--accent, #e63946);
        opacity: 0.8;
    }
    .nav-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        color: #fff;
        text-align: center;
    }
    .nav-desc {
        font-size: 0.82rem;
        color: #999;
        line-height: 1.45;
        text-align: center;
        margin-bottom: 0.9rem;
    }
    /* Preview block — between description and CTA */
    .nav-preview {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 0.45rem;
        padding: 0.8rem 0.5rem;
        background: rgba(0, 0, 0, 0.18);
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .nav-preview-row {
        display: flex;
        align-items: center;
        font-size: 0.82rem;
        padding: 0 0.6rem;
    }
    .nav-preview-row .label  { color: #aaa; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 0.5rem; }
    .nav-preview-row .value  { color: #fff; font-weight: 600; font-variant-numeric: tabular-nums; }
    .nav-preview-row .value.green { color: #2ecc71; }
    .nav-preview-row .value.red   { color: #e74c3c; }
    .nav-preview-row .value.gold  { color: #f1c40f; }
    .nav-preview-row .value.blue  { color: #4cc9f0; }
    .nav-preview-empty {
        font-size: 0.78rem;
        color: #777;
        text-align: center;
        font-style: italic;
        padding: 0.6rem;
    }

    .nav-cta   {
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
<div style="text-align:center; padding: 2.5rem 0 1.75rem 0;">
    <div style="font-size:3rem; font-weight:800; letter-spacing:-1px; color:#fff;">
        Barrett Score
    </div>
    <div style="font-size:1.05rem; color:#aaa; margin-top:0.5rem;">
        A stat-driven NBA player valuation tool — who's underpaid, overpaid, and available.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Compute live previews from already-warmed cache ───────────────────────────
def _compute_previews():
    """Pull a few fast stats from the current-season cache for the home cards.
    Everything here is in-memory cache hits (build_ranked_projected etc. are warmed
    by _bootstrap_warm above), so this adds < ~50ms to home-page render."""
    try:
        df = build_ranked_projected(SEASONS[0])
        df = df[df["total_min"] >= DEFAULT_MIN_THRESHOLD].copy()
        if df.empty:
            return None

        top3 = df.nsmallest(3, "score_rank")[["Player", "barrett_score"]].values.tolist()

        steal_row    = df.loc[df["value_diff"].idxmin()]
        overpaid_row = df.loc[df["value_diff"].idxmax()]

        team_eff = (
            df.groupby("Team")
              .apply(lambda g: (g["salary"].sum() - g["projected_salary"].sum()) / 1e6)
              .sort_values()
        )
        best_team_name  = team_eff.index[0]
        worst_team_name = team_eff.index[-1]
        best_team_val   = float(team_eff.iloc[0])
        worst_team_val  = float(team_eff.iloc[-1])

        next_contracts = fetch_next_year_contracts(season_to_espn_year(SEASONS[0]), cache_v=7)
        rookie_scale   = fetch_rookie_scale_players(SEASONS[0])

        def _is_fa(player_name: str) -> bool:
            nc = fmt_next_contract(player_name, next_contracts)
            if nc == "RFA":
                return True
            if nc == "—":
                # UFA, or rookie-scale player about to become RFA
                return True
            if " PO" in nc or " TO" in nc:
                return True
            return False

        fa_mask  = df["Player"].apply(_is_fa)
        fa_count = int(fa_mask.sum())
        if fa_mask.any():
            top_fa_row = df[fa_mask].sort_values("barrett_score", ascending=False).iloc[0]
            top_fa     = str(top_fa_row["Player"])
        else:
            top_fa = "—"

        return {
            "season":        SEASONS[0],
            "n_players":     len(df),
            "top3":          top3,
            "steal_name":    str(steal_row["Player"]),
            "steal_amt":     abs(float(steal_row["value_diff"])) / 1e6,
            "over_name":     str(overpaid_row["Player"]),
            "over_amt":      float(overpaid_row["value_diff"]) / 1e6,
            "best_team":     str(best_team_name),
            "best_team_val": best_team_val,
            "worst_team":    str(worst_team_name),
            "worst_team_val": worst_team_val,
            "fa_count":      fa_count,
            "top_fa":        top_fa,
        }
    except Exception:
        return None


_p = _compute_previews()


def _preview_block(rows):
    """rows is a list of (label, value, css_class)."""
    if not rows:
        return '<div class="nav-preview"><div class="nav-preview-empty">Loading live data…</div></div>'
    parts = ['<div class="nav-preview">']
    for r in rows:
        label, value, cls = (r + ("",))[:3] if len(r) < 3 else r
        parts.append(
            f'<div class="nav-preview-row">'
            f'<span class="label">{label}</span>'
            f'<span class="value {cls}">{value}</span>'
            f'</div>'
        )
    parts.append("</div>")
    return "".join(parts)


# ── Build per-card previews ───────────────────────────────────────────────────
if _p:
    rankings_preview = _preview_block([
        (f"1. {_p['top3'][0][0]}", f"{_p['top3'][0][1]:.1f}", "gold"),
        (f"2. {_p['top3'][1][0]}", f"{_p['top3'][1][1]:.1f}", ""),
        (f"3. {_p['top3'][2][0]}", f"{_p['top3'][2][1]:.1f}", ""),
    ])
    visualizer_preview = _preview_block([
        ("Biggest steal",  f"{_p['steal_name']}",    "green"),
        ("",               f"−${_p['steal_amt']:.1f}M",  "green"),
        ("Most overpaid",  f"{_p['over_name']}",     "red"),
        ("",               f"+${_p['over_amt']:.1f}M",   "red"),
    ])
    team_preview = _preview_block([
        ("Best efficiency",  f"{_p['best_team']}",      "green"),
        ("",                 f"${_p['best_team_val']:+.1f}M",  "green"),
        ("Worst efficiency", f"{_p['worst_team']}",     "red"),
        ("",                 f"${_p['worst_team_val']:+.1f}M", "red"),
    ])
    fa_preview = _preview_block([
        ("Free agents",   f"{_p['fa_count']}",  "blue"),
        ("Top FA",        f"{_p['top_fa']}",    ""),
        ("Season",        f"{_p['season']}",    ""),
    ])
else:
    rankings_preview = visualizer_preview = team_preview = fa_preview = _preview_block([])

# Legacy preview is a static feature list (live all-seasons data is too heavy for the home page)
legacy_preview = _preview_block([
    ("Seasons",         "2006 → 2025", "gold"),
    ("Career arcs",     "Every player",  ""),
    ("All-time ranks",  "Top 100",       ""),
    ("Mount Rushmores", "All 30 teams",  ""),
])

# ── Nav cards — row 1: Rankings + Legacy (2 centered) ─────────────────────────
_, col1, col2, _ = st.columns([0.5, 1, 1, 0.5], gap="medium")
with col1:
    st.markdown(f"""
    <a class="nav-card" href="/Rankings" target="_top" style="--accent:#e63946;">
        <div class="nav-title">Current Rankings</div>
        <div class="nav-desc">Every NBA player ranked by Barrett Score — filter by team, position, season.</div>
        {rankings_preview}
        <span class="nav-cta">Open Rankings →</span>
    </a>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <a class="nav-card" href="/Legacy" target="_top" style="--accent:#f1c40f;">
        <div class="nav-title">Legacy</div>
        <div class="nav-desc">The historical record — all-time rankings, career arcs, era leaderboards, and more.</div>
        {legacy_preview}
        <span class="nav-cta">Open Legacy →</span>
    </a>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)

# ── Nav cards — row 2: Visualizer, Team Analysis, Free Agent Class ────────────
col3, col4, col5 = st.columns(3, gap="medium")
with col3:
    st.markdown(f"""
    <a class="nav-card" href="/Salary_Projector" target="_top" style="--accent:#9b59b6;">
        <div class="nav-title">Visualizer</div>
        <div class="nav-desc">What every player should earn based on their Barrett Score rank vs actual contract.</div>
        {visualizer_preview}
        <span class="nav-cta">Open Visualizer →</span>
    </a>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <a class="nav-card" href="/Team_Analysis" target="_top" style="--accent:#3498db;">
        <div class="nav-title">Team Analysis</div>
        <div class="nav-desc">Aggregate Barrett Scores by team — best and worst roster construction in the league.</div>
        {team_preview}
        <span class="nav-cta">Open Teams →</span>
    </a>
    """, unsafe_allow_html=True)
with col5:
    st.markdown(f"""
    <a class="nav-card" href="/Free_Agent_Class" target="_top" style="--accent:#2ecc71;">
        <div class="nav-title">Current Free Agents</div>
        <div class="nav-desc">UFAs, player options, and team options ranked by Barrett Score — a GM's draft board.</div>
        {fa_preview}
        <span class="nav-cta">Open Free Agency →</span>
    </a>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; color:#555; font-size:0.8rem;">
    barrettscore.com &nbsp;·&nbsp; Data from NBA Stats API &nbsp;·&nbsp; Updated daily
</div>
""", unsafe_allow_html=True)
