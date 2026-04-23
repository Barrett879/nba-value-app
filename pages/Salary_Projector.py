import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from utils import (
    COMMON_CSS, SEASONS, DEFAULT_MIN_THRESHOLD,
    normalize, season_to_espn_year,
    build_raw, apply_rankings, apply_projections,
    fetch_bref_positions, fetch_next_year_contracts, fetch_rookie_scale_players,
    color_value_diff, render_nav,
)

st.set_page_config(page_title="Barrett Score — Salary Projector", layout="wide", page_icon="🏀")

st.markdown(COMMON_CSS, unsafe_allow_html=True)

components.html("""
<script>
    function hideBadge() {
        try {
            const doc = window.parent.document;
            [
                '[data-testid="stAppViewerBadge"]',
                '[data-testid="stBottom"]',
                '[data-testid="stToolbar"]',
                '[data-testid="stStatusWidget"]',
                '[class*="viewerBadge"]',
                '[class*="ViewerBadge"]',
            ].forEach(sel => doc.querySelectorAll(sel).forEach(el => el.remove()));
        } catch(e) {}
    }
    hideBadge();
    new MutationObserver(hideBadge).observe(document.documentElement, { childList: true, subtree: true });
</script>
""", height=0)

render_nav("💰 Salary Projector")

st.title("Barrett Score — Salary Projector")

# ── Season selector ────────────────────────────────────────────────────────────
ctrl_l, ctrl_mid, ctrl_r = st.columns([1, 1, 1])
with ctrl_l:
    season = st.selectbox("Season", SEASONS, index=0)
with ctrl_r:
    min_threshold = st.slider(
        "Min total minutes", min_value=0, max_value=1500,
        value=DEFAULT_MIN_THRESHOLD, step=50,
        help="Hides players below this threshold. Ranks are always computed on the full pool.",
    )

# ── Data loading ───────────────────────────────────────────────────────────────
raw = build_raw(season)
df = apply_rankings(raw)
df = apply_projections(df)
df = df[df["total_min"] >= min_threshold]

_bref_positions = fetch_bref_positions(season_to_espn_year(season), cache_v=3)
df["position"] = df["Player"].map(
    lambda n: _bref_positions.get(normalize(n), "")
)

# ══════════════════════════════════════════════════════════════════════════════
# Salary Projector content
# ══════════════════════════════════════════════════════════════════════════════

st.caption(
    "Projected salary = the actual salary of whoever holds the same rank position by pay. "
    "If Jokic is #1 by Barrett Score, he deserves the #1 salary (Curry's contract). "
    "No invented numbers — every projected salary is a real contract on the books."
)

proj = df.copy()
proj["Actual $M"]     = proj["salary"] / 1e6
proj["Proj. $M"]      = proj["projected_salary"] / 1e6
proj["Δ $M"]          = proj["value_diff"] / 1e6
proj["Barrett Score"] = proj["barrett_score"].round(2)
proj["Score Rank"]    = proj["score_rank"]

ph_col, ph_sort = st.columns([2, 1])
with ph_col:
    proj_search = st.text_input("Highlight player", "", key="proj_search",
                                placeholder="Type a name to highlight on the chart…")
with ph_sort:
    proj_sort = st.selectbox("Table order", ["Most Overpaid", "Most Underpaid"], key="proj_sort")

axis_max = proj[["Actual $M", "Proj. $M"]].max().max() * 1.05

fig = px.scatter(
    proj,
    x="Proj. $M",
    y="Actual $M",
    color="Δ $M",
    color_continuous_scale="RdYlGn_r",
    color_continuous_midpoint=0,
    hover_name="Player",
    hover_data={
        "Team": True,
        "Score Rank": True,
        "Barrett Score": ":.2f",
        "Actual $M": ":.1f",
        "Proj. $M": ":.1f",
        "Δ $M": ":.1f",
    },
    labels={"Proj. $M": "Projected Salary ($M)", "Actual $M": "Actual Salary ($M)"},
    height=520,
)

fig.add_shape(
    type="line",
    x0=0, y0=0, x1=axis_max, y1=axis_max,
    line=dict(color="rgba(255,255,255,0.5)", width=2, dash="dash"),
)
fig.add_annotation(
    x=axis_max * 0.72, y=axis_max * 0.82,
    text="Fairly paid", showarrow=False,
    font=dict(color="rgba(255,255,255,0.5)", size=11),
    textangle=-40,
)

if proj_search:
    hi = proj[proj["Player"].str.contains(proj_search, case=False)]
    fig.add_traces(px.scatter(
        hi, x="Proj. $M", y="Actual $M",
        text="Player",
    ).update_traces(
        marker=dict(size=14, color="yellow", line=dict(color="black", width=1)),
        textposition="top center",
        name="Highlighted",
    ).data)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.15)",
    font_color="white",
    coloraxis_colorbar=dict(title="Δ ($M)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickprefix="$", ticksuffix="M",
               range=[0, axis_max]),
    yaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickprefix="$", ticksuffix="M",
               range=[0, axis_max]),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

tbl = proj[["Player", "Team", "Score Rank", "Barrett Score",
            "Proj. $M", "Actual $M", "Δ $M"]].copy()
tbl["Proj. $M"]  = tbl["Proj. $M"].round(2)
tbl["Actual $M"] = tbl["Actual $M"].round(2)
tbl["Δ $M"]      = tbl["Δ $M"].round(2)

if proj_sort == "Most Overpaid":
    tbl = tbl.sort_values("Δ $M", ascending=False)
else:
    tbl = tbl.sort_values("Δ $M", ascending=True)
tbl = tbl.reset_index(drop=True)
tbl.insert(0, "#", range(1, len(tbl) + 1))


def color_delta(val):
    try:
        n = float(val)
    except (ValueError, TypeError):
        return ""
    if n > 20:  return "color: #e74c3c; font-weight: bold"
    if n > 5:   return "color: #f1a8a8"
    if n < -20: return "color: #2ecc71; font-weight: bold"
    if n < -5:  return "color: #a8e6a8"
    return ""


st.dataframe(
    tbl.style.map(color_delta, subset=["Δ $M"]),
    column_config={
        "Proj. $M":  st.column_config.NumberColumn(format="$%.2fM"),
        "Actual $M": st.column_config.NumberColumn(format="$%.2fM"),
        "Δ $M":      st.column_config.NumberColumn(format="$%.2fM"),
        "Barrett Score": st.column_config.NumberColumn(format="%.2f"),
    },
    use_container_width=True,
    hide_index=True,
    height=500,
)
st.caption(
    "**Proj. Salary** = salary of whoever is the same rank by pay (e.g. score rank #1 → highest salary on the roster). "
    "**Δ $M** = Actual − Projected. Positive (red) = overpaid. Negative (green) = underpaid."
)
