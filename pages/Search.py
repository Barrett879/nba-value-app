import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from utils import (
    COMMON_CSS,
    get_all_player_names, fetch_player_full_career,
    render_nav, _bootstrap_warm,
)

st.set_page_config(page_title="Barrett Score — Search Player", layout="wide")
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

_bootstrap_warm()
render_nav("Search Player")

st.title("Search Player")
st.caption("Find any player who's appeared in the league — career arcs, season-by-season stats, peak years.")

# ── Search box ─────────────────────────────────────────────────────────────────
all_names = get_all_player_names()
if not all_names:
    st.error("Player database not yet loaded. Try again in a moment.")
    st.stop()

selected = st.selectbox(
    "Type a player name…",
    options=all_names,
    index=None,
    placeholder="Try LeBron James, Michael Jordan, Nikola Jokić…",
    key="player_search_select",
)

if not selected:
    st.info(
        f"**{len(all_names):,} players** indexed across "
        f"every season we have data for. Names are sorted by career-average "
        "Barrett Score, so the legends rise to the top."
    )
    st.stop()

# ── Load full career data ──────────────────────────────────────────────────────
with st.spinner(f"Loading {selected}'s career…"):
    career = fetch_player_full_career(selected)

if career.empty:
    st.warning(f"No career data found for {selected}.")
    st.stop()

# ── Header summary ─────────────────────────────────────────────────────────────
n_seasons   = len(career)
first_yr    = career["Season"].iloc[0].split("-")[0]
last_yr_end = career["Season"].iloc[-1].split("-")[1]
career_yrs  = f"{first_yr} – 20{last_yr_end}" if int(last_yr_end) < 50 else f"{first_yr} – 19{last_yr_end}"
teams       = list(dict.fromkeys(career["Team"]))   # preserve order, dedup

best_season_idx = career["Barrett Score"].idxmax()
best_season     = career.loc[best_season_idx]

career_avg_score = career["Barrett Score"].mean()
career_avg_pts   = career["PTS"].mean()
career_avg_ast   = career["AST"].mean()
career_avg_reb   = career["REB"].mean()
total_games      = int(career["GP"].sum())

st.markdown(f"### {selected}")
st.caption(f"**{career_yrs}** · {n_seasons} seasons · {total_games:,} games · "
           f"Teams: {' → '.join(teams)}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Career Avg Barrett",   f"{career_avg_score:.1f}")
c2.metric("Career PPG",           f"{career_avg_pts:.1f}")
c3.metric("Career APG",           f"{career_avg_ast:.1f}")
c4.metric("Career RPG",           f"{career_avg_reb:.1f}")
c5.metric("Peak Season",
          f"{best_season['Barrett Score']:.1f}",
          f"{best_season['Season']}")

st.divider()

# ── Career arc chart ───────────────────────────────────────────────────────────
st.subheader("Career arc — Barrett Score by season")
fig = go.Figure()

# Color points by Barrett Score (red→gold→green)
def _val_color(v, vmin, vmax):
    if vmax <= vmin: return "#f1c40f"
    t = (v - vmin) / (vmax - vmin)
    if t < 0.5:
        # red to gold
        r1, g1, b1 = 0xe7, 0x4c, 0x3c
        r2, g2, b2 = 0xf1, 0xc4, 0x0f
        f = t * 2
    else:
        r1, g1, b1 = 0xf1, 0xc4, 0x0f
        r2, g2, b2 = 0x2e, 0xcc, 0x71
        f = (t - 0.5) * 2
    r = int(r1 + (r2 - r1) * f)
    g = int(g1 + (g2 - g1) * f)
    b = int(b1 + (b2 - b1) * f)
    return f"rgb({r},{g},{b})"

vmin, vmax = career["Barrett Score"].min(), career["Barrett Score"].max()
dot_colors = [_val_color(v, vmin, vmax) for v in career["Barrett Score"]]

fig.add_trace(go.Scatter(
    x=career["Season"], y=career["Barrett Score"],
    mode="lines+markers",
    line=dict(color="rgba(241, 196, 15, 0.6)", width=2.5),
    marker=dict(size=10, color=dot_colors,
                line=dict(color="#14142a", width=1.5)),
    text=career["Team"],
    customdata=career[["PTS", "AST", "REB", "Score Rank", "Total Players"]].values,
    hovertemplate=(
        "<b>%{x}</b> · %{text}<br>"
        "Barrett Score: %{y:.2f}<br>"
        "PTS %{customdata[0]:.1f} · AST %{customdata[1]:.1f} · REB %{customdata[2]:.1f}<br>"
        "Rank %{customdata[3]} / %{customdata[4]} that season"
        "<extra></extra>"
    ),
    showlegend=False,
))

# Mark peak season
fig.add_trace(go.Scatter(
    x=[best_season["Season"]], y=[best_season["Barrett Score"]],
    mode="markers",
    marker=dict(size=18, symbol="star", color="white",
                line=dict(width=1.5, color="#1a1a2e")),
    name="Peak season",
    hoverinfo="skip",
    showlegend=False,
))

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.15)",
    font_color="white",
    height=400,
    margin=dict(l=50, r=30, t=20, b=50),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="", type="category"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Barrett Score",
               tickformat=".1f"),
    hovermode="closest",
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.caption("★ = peak career season · dot color encodes the score (red = lowest, "
           "gold = mid, green = highest of this player's career)")

st.divider()

# ── Season-by-season table ─────────────────────────────────────────────────────
st.subheader("Season by season")

tbl = career[[
    "Season", "Team", "GP", "MPG", "PTS", "AST", "REB", "STL", "BLK", "TOV",
    "TS%", "Barrett Score", "Score Rank", "Total Players", "Salary",
]].copy()
tbl["Salary $M"] = (tbl["Salary"] / 1_000_000).round(2)
tbl = tbl.drop(columns=["Salary"])
tbl["Rank"] = tbl.apply(lambda r: f"{int(r['Score Rank'])}/{int(r['Total Players'])}", axis=1)
tbl = tbl.drop(columns=["Score Rank", "Total Players"])

# Highlight peak season row
def highlight_peak(row):
    if row["Season"] == best_season["Season"]:
        return ["background-color: rgba(241, 196, 15, 0.18); font-weight: 600"] * len(row)
    return [""] * len(row)

styled = (
    tbl.style
    .apply(highlight_peak, axis=1)
    .format({
        "MPG": "{:.1f}", "PTS": "{:.1f}", "AST": "{:.1f}", "REB": "{:.1f}",
        "STL": "{:.2f}", "BLK": "{:.2f}", "TOV": "{:.2f}",
        "TS%": "{:.1f}%", "Barrett Score": "{:.2f}", "Salary $M": "${:.2f}M",
    })
)

st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    height=min(700, max(120, len(tbl) * 35 + 40)),
    column_config={
        "Salary $M":     st.column_config.TextColumn("Salary",     help="Salary that season ($M). Some pre-2000 rookie scale and minimum contracts may show $0."),
        "Barrett Score": st.column_config.NumberColumn(format="%.2f"),
        "Rank":          st.column_config.TextColumn(help="Score rank that season out of all players who hit the minutes threshold."),
        "TS%":           st.column_config.TextColumn("TS%", help="True Shooting %."),
    },
)
st.caption(f"Highlighted row = peak season ({best_season['Season']}). "
           "Use the Legacy page for cross-player comparisons.")
