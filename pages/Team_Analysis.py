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
    fetch_bref_positions,
)

st.set_page_config(page_title="Barrett Score — Team Analysis", layout="wide", page_icon="🏀")

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

if st.button("← Home"):
    st.switch_page("app.py")

st.title("Barrett Score — Team Analysis")

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
# Team Analysis content
# ══════════════════════════════════════════════════════════════════════════════

st.caption(
    "Which front offices are getting the most value? "
    "Payroll efficiency = how much a team is over- or under-paying relative to what their "
    "players' Barrett Scores say they deserve. Negative = underpaying (efficient). "
    "Positive = overpaying (inefficient)."
)

team_grp = df.groupby("Team").agg(
    players        = ("Player",           "count"),
    actual_payroll = ("salary",           "sum"),
    proj_payroll   = ("projected_salary", "sum"),
    net_delta      = ("value_diff",       "sum"),
    avg_score      = ("barrett_score",    "mean"),
).reset_index()

team_grp["actual_payroll"] /= 1e6
team_grp["proj_payroll"]   /= 1e6
team_grp["net_delta"]      /= 1e6
team_grp["avg_score"]       = team_grp["avg_score"].round(2)

team_grp = team_grp.sort_values("net_delta")

fig_teams = px.bar(
    team_grp,
    x="Team", y="net_delta",
    color="net_delta",
    color_continuous_scale="RdYlGn_r",
    color_continuous_midpoint=0,
    labels={"net_delta": "Net Δ ($M)", "Team": ""},
    title="Payroll Efficiency by Team  (negative = underpaying, positive = overpaying)",
    height=420,
    text=team_grp["net_delta"].apply(lambda v: f"${v:+.1f}M"),
)
fig_teams.update_traces(textposition="outside", textfont_size=10)
fig_teams.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.15)",
    font_color="white",
    coloraxis_showscale=False,
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", categoryorder="total ascending"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickprefix="$", ticksuffix="M"),
    margin=dict(t=50, b=20),
)
st.plotly_chart(fig_teams, use_container_width=True, config={"displayModeBar": False})

best_player  = df.loc[df.groupby("Team")["value_diff"].idxmin(), ["Team", "Player"]].set_index("Team")["Player"]
worst_player = df.loc[df.groupby("Team")["value_diff"].idxmax(), ["Team", "Player"]].set_index("Team")["Player"]

team_tbl = team_grp.copy()
team_tbl["Best Value"]  = team_tbl["Team"].map(best_player)
team_tbl["Worst Value"] = team_tbl["Team"].map(worst_player)
team_tbl.columns = ["Team", "Players", "Actual $M", "Proj. $M",
                     "Net Δ $M", "Avg Score", "Best Value", "Worst Value"]
team_tbl.insert(0, "#", range(1, len(team_tbl) + 1))


def color_net(val):
    try:
        n = float(val)
    except (ValueError, TypeError):
        return ""
    if n < -20: return "color: #2ecc71; font-weight: bold"
    if n < 0:   return "color: #a8e6a8"
    if n > 20:  return "color: #e74c3c; font-weight: bold"
    if n > 0:   return "color: #f1a8a8"
    return ""


st.dataframe(
    team_tbl.style.map(color_net, subset=["Net Δ $M"]),
    column_config={
        "Actual $M":  st.column_config.NumberColumn(format="$%.1fM",
            help="Sum of all qualifying players' actual salaries."),
        "Proj. $M":   st.column_config.NumberColumn(format="$%.1fM",
            help="Sum of what those players would earn paid by Barrett Score rank."),
        "Net Δ $M":   st.column_config.NumberColumn(format="$%.1fM",
            help="Actual − Projected. Negative (green) = team is getting value. Positive (red) = overpaying."),
        "Avg Score":  st.column_config.NumberColumn(format="%.2f",
            help="Average Barrett Score across qualifying players on the team."),
        "Best Value": st.column_config.TextColumn(help="Most underpaid player (lowest Actual − Projected)."),
        "Worst Value":st.column_config.TextColumn(help="Most overpaid player (highest Actual − Projected)."),
    },
    use_container_width=True,
    hide_index=True,
    height=min(600, len(team_tbl) * 35 + 40),
)
st.caption(
    f"Based on **{len(df)}** players with ≥ {min_threshold} total minutes. "
    "Only players in the rankings pool are included — rookies on minimum deals may not appear."
)

st.divider()
drill_team = st.selectbox("Drill into a team", [""] + sorted(df["Team"].unique().tolist()),
                          key="team_drill")
if drill_team:
    team_players = df[df["Team"] == drill_team].copy()
    team_players["salary"]           /= 1e6
    team_players["projected_salary"] /= 1e6
    team_players["value_diff"]       /= 1e6
    team_players = team_players.sort_values("value_diff")
    team_players = team_players[["Player", "barrett_score", "score_rank",
                                  "salary", "projected_salary", "value_diff"]].copy()
    team_players.columns = ["Player", "Barrett Score", "Score Rank",
                              "Salary $M", "Proj. $M", "Δ $M"]
    team_players.insert(0, "#", range(1, len(team_players) + 1))
    st.dataframe(
        team_players.style.map(color_net, subset=["Δ $M"]),
        column_config={
            "Barrett Score": st.column_config.NumberColumn(format="%.2f"),
            "Salary $M":     st.column_config.NumberColumn(format="$%.2fM"),
            "Proj. $M":      st.column_config.NumberColumn(format="$%.2fM"),
            "Δ $M":          st.column_config.NumberColumn(format="$%.2fM"),
        },
        use_container_width=True,
        hide_index=True,
    )
