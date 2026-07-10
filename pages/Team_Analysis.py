import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import html

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from utils import (
    COMMON_CSS, SEASONS, DEFAULT_MIN_THRESHOLD,
    normalize, season_to_espn_year,
    build_ranked_projected,
    fetch_bref_positions, render_nav, render_page_chrome,
    theme_fig, html_table, render_playoff_toggle,
    render_barrett_score_explainer, _bootstrap_warm,
    PRE_1990_SALARY_NOTE,
    render_rail, face_img, TEAM_HEX, hex_rgba,
)

st.set_page_config(page_title="Team Analysis", page_icon="static/favicon.svg", layout="wide")

render_page_chrome()
_bootstrap_warm()
render_nav("Team Analysis")

# Playoff toggle sits on the title row, right-aligned (in-page, not the nav bar)
playoff_mode = bool(st.session_state.get("playoff_mode", False))
_tcol, _pcol = st.columns([5, 1], vertical_alignment="center")
with _tcol:
    st.title("Team Analysis (Playoffs)" if playoff_mode else "Team Analysis")
with _pcol:
    render_playoff_toggle()

st.caption(
    "Which front offices are getting the most value? "
    "Payroll efficiency = how much a team is over- or under-paying relative to what their "
    "players' Barrett Scores say they deserve. Negative = underpaying (efficient). "
    "Positive = overpaying (inefficient)."
)


# ── Season selector ────────────────────────────────────────────────────────────
ctrl_l, ctrl_mid, ctrl_r = st.columns([1, 1, 1])
with ctrl_l:
    season = st.selectbox("Season", SEASONS, index=0)
default_threshold = 100 if playoff_mode else DEFAULT_MIN_THRESHOLD
slider_max = 600 if playoff_mode else 1500
slider_step = 25 if playoff_mode else 50
with ctrl_r:
    min_threshold = st.slider(
        "Min total minutes", min_value=0, max_value=slider_max,
        value=default_threshold, step=slider_step,
        help="Hides players below this threshold. Ranks are always computed on the full pool.",
    )

# ── Data loading ───────────────────────────────────────────────────────────────
# build_ranked_projected is @st.cache_resource (no copy on hit) — must copy before mutating
df = build_ranked_projected(season, playoffs=playoff_mode)
if df.empty:
    st.warning(
        f"No {'playoff' if playoff_mode else 'regular season'} data for {season} yet. "
        "Try a previous season or toggle Playoff mode off."
    )
    st.stop()
df = df[df["total_min"] >= min_threshold].copy()

_bref_positions = fetch_bref_positions(season_to_espn_year(season), cache_v=3)
import team_suitors as _ts
_pos2k = _ts.load_player_positions()
# Curated 2K position (primary + secondary, e.g. "PG/SG"), BBRef coarse fallback.
df["position"] = df["Player"].map(
    lambda n: _ts.resolve_position(n, _bref_positions.get(normalize(n), ""), _pos2k))

# Warn when salary coverage is too sparse to make team-level totals meaningful
# (pre-1996 BBRef team pages miss most players, leaving many at $0).
_is_pre_1990 = int(season.split("-")[0]) < 1990
if _is_pre_1990:
    st.warning(PRE_1990_SALARY_NOTE, icon="📜")
_salary_coverage = (df["salary"] > 0).mean() if len(df) else 0.0
if _salary_coverage < 0.5 and not _is_pre_1990:
    st.warning(
        f"⚠️ Salary coverage for {season} is {_salary_coverage*100:.0f}%. "
        "Many players have no salary data on file. Team payroll totals below "
        "are unreliable for this season. Use the Rankings page for player-level analysis."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Team Analysis content
# ══════════════════════════════════════════════════════════════════════════════

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

render_rail("League view", "Payroll Efficiency by Team",
            count=f"{len(team_grp)} teams",
            meta="negative = underpaying · positive = overpaying")

fig_teams = px.bar(
    team_grp,
    x="Team", y="net_delta",
    color="net_delta",
    color_continuous_scale="RdYlGn_r",
    color_continuous_midpoint=0,
    labels={"net_delta": "Net Δ ($M)", "Team": ""},
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
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickprefix="$", ticksuffix="M", tickformat=".1f"),
    margin=dict(t=20, b=20),
)
st.plotly_chart(theme_fig(fig_teams), use_container_width=True, config={"displayModeBar": False})

best_player  = df.loc[df.groupby("Team")["value_diff"].idxmin(), ["Team", "Player"]].set_index("Team")["Player"]
worst_player = df.loc[df.groupby("Team")["value_diff"].idxmax(), ["Team", "Player"]].set_index("Team")["Player"]

team_tbl = team_grp.copy()
team_tbl["Best Value"]  = team_tbl["Team"].map(best_player)
team_tbl["Worst Value"] = team_tbl["Team"].map(worst_player)
team_tbl.columns = ["Team", "Players", "Actual $M", "Proj. $M",
                     "Net Δ $M", "Avg Score", "Best Value", "Worst Value"]
team_tbl.insert(0, "#", range(1, len(team_tbl) + 1))


def _sty_net(v, _row):
    try:
        n = float(v)
    except (ValueError, TypeError):
        return ""
    if n < -20: return "color:var(--value-good);font-weight:700"
    if n < 0:   return "color:var(--value-good-s)"
    if n > 20:  return "color:var(--value-bad);font-weight:700"
    if n > 0:   return "color:var(--value-bad-s)"
    return ""


render_rail("The ledger", "Team-by-Team Value",
            count=f"{len(team_tbl)} teams",
            meta=f"{season} playoffs" if playoff_mode else season)

# Avg Score clusters tightly at team level: scale bars 15-100 across visible min-max.
_as_lo = float(team_tbl["Avg Score"].min())
_as_rng = (float(team_tbl["Avg Score"].max()) - _as_lo) or 1.0


def _face_cell(v):
    n = str(v)
    return (f'<span class="hv-mini-wrap">{face_img(n, "hv-mini-face")}</span>'
            f'{html.escape(n)}')


html_table(
    team_tbl,
    formatters={
        "Team": lambda v: (f'<span class="tdot tdot-{html.escape(str(v), quote=True)}"></span>'
                           f'{html.escape(str(v))}'),
        "Actual $M": lambda v: f"${v:.1f}M",
        "Proj. $M":  lambda v: f"${v:.1f}M",
        "Net Δ $M":  lambda v: f"${v:.1f}M",
        "Avg Score": lambda v: f"{v:.2f}",
        "Best Value": _face_cell,
        "Worst Value": _face_cell,
    },
    raw={"Team", "Best Value", "Worst Value"},
    styles={
        "Net Δ $M": _sty_net,
        "Avg Score": lambda v, _r: (
            f"background:linear-gradient(90deg,var(--bar-tint) "
            f"{max(2, min(100, float(v) / ((_as_lo + _as_rng) or 1) * 100)):.0f}%,transparent 0)"
            if v == v else ""),
    },
    aligns={"#": "right", "Players": "right", "Actual $M": "right",
            "Proj. $M": "right", "Net Δ $M": "right", "Avg Score": "right"},
    numeric={"#", "Players", "Actual $M", "Proj. $M", "Net Δ $M", "Avg Score"},
    helps={
        "Actual $M": "Sum of all qualifying players' actual salaries.",
        "Proj. $M": "Sum of what those players would earn paid by Barrett Score rank.",
        "Net Δ $M": "Actual − Projected. Negative (green) = getting value; positive (red) = overpaying.",
        "Avg Score": "Average Barrett Score across qualifying players on the team.",
        "Best Value": "Most underpaid player (lowest Actual − Projected).",
        "Worst Value": "Most overpaid player (highest Actual − Projected).",
    },
    height=min(640, len(team_tbl) * 38 + 46),
)
st.caption(
    f"Based on **{len(df)}** players with ≥ {min_threshold} total minutes. "
    "Only players in the rankings pool are included; rookies on minimum deals may not appear."
)

render_rail("Drill-down", "Team Roster Detail")
drill_team = st.selectbox("Pick a Team", [""] + sorted(df["Team"].unique().tolist()),
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

    # Team-accent panel: left border wash + rail bar recolored to the picked team.
    _thx = TEAM_HEX.get(drill_team, "")
    if _thx:
        st.markdown(
            "<style>"
            f".st-key-hv_team_panel{{border-left:3px solid {hex_rgba(_thx, 0.55)};"
            "padding-left:1.1rem;}"
            f".st-key-hv_team_panel .hv-rail::before{{background:{_thx};}}"
            "</style>",
            unsafe_allow_html=True,
        )
    _ring = f' style="box-shadow:inset 0 0 0 2px {_thx}"' if _thx else ""
    _bs_lo = float(team_players["Barrett Score"].min())
    _bs_rng = (float(team_players["Barrett Score"].max()) - _bs_lo) or 1.0

    with st.container(key="hv_team_panel"):
        render_rail("The roster", f"{drill_team} Value Board",
                    count=f"{len(team_players)} players",
                    meta="sorted by value surplus")
        html_table(
            team_players,
            formatters={
                "Player": lambda v: (
                    f'<span class="hv-mini-wrap"{_ring}>{face_img(str(v), "hv-mini-face")}</span>'
                    f'{html.escape(str(v))}'),
                "Barrett Score": lambda v: f"{v:.2f}",
                "Salary $M":     lambda v: f"${v:.2f}M",
                "Proj. $M":      lambda v: f"${v:.2f}M",
                "Δ $M":          lambda v: f"${v:.2f}M",
            },
            raw={"Player"},
            styles={
                "Δ $M": _sty_net,
                "Barrett Score": lambda v, _r: (
                    f"background:linear-gradient(90deg,var(--bar-tint) "
                    f"{max(2, min(100, float(v) / ((_bs_lo + _bs_rng) or 1) * 100)):.0f}%,transparent 0)"
                    if v == v else ""),
            },
            aligns={"#": "right", "Barrett Score": "right", "Score Rank": "right",
                    "Salary $M": "right", "Proj. $M": "right", "Δ $M": "right"},
            numeric={"#", "Barrett Score", "Score Rank", "Salary $M", "Proj. $M", "Δ $M"},
            height=min(640, len(team_players) * 41 + 48),
        )


from utils import render_footer
render_footer()
