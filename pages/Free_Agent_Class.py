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
    build_ranked_projected,
    fetch_bref_positions, fetch_next_year_contracts, fetch_rookie_scale_players,
    _fmt_salary, fmt_next_contract,
    color_next_contract, style_rookie_salary, color_value_diff, render_nav, render_page_chrome,
    theme_fig, html_table,
    render_barrett_score_explainer, _bootstrap_warm,
)

st.set_page_config(page_title="Free Agent Class", page_icon="static/favicon.svg", layout="wide")

render_page_chrome()
_bootstrap_warm()
render_nav("Current Free Agents")

st.title("Free Agent Class")

st.caption(
    "Every player whose contract situation makes them available this offseason: "
    "UFAs, RFAs (team holds right of first refusal), player options (they may opt out), "
    "and team options (team may decline). Ranked by Barrett Score."
)

render_barrett_score_explainer()

# ── Season selector ────────────────────────────────────────────────────────────
# Free agency data (next-year contracts, options) is only reliable for the
# current season — Spotrac's URL has no historical year, so older seasons
# would mix today's free-agency status with stale stat data. Limit to the
# current + immediately prior season so the page always makes sense.
_FA_SEASONS = SEASONS[:2]
ctrl_l, ctrl_mid, ctrl_r = st.columns([1, 1, 1])
with ctrl_l:
    season = st.selectbox("Season", _FA_SEASONS, index=0)
with ctrl_r:
    min_threshold = st.slider(
        "Min total minutes", min_value=0, max_value=1500,
        value=DEFAULT_MIN_THRESHOLD, step=50,
        help="Hides players below this threshold. Ranks are always computed on the full pool.",
    )

# ── Data loading ───────────────────────────────────────────────────────────────
# build_ranked_projected is @st.cache_resource (no copy on hit) — must copy before mutating
df = build_ranked_projected(season)
df = df[df["total_min"] >= min_threshold].copy()

_bref_positions = fetch_bref_positions(season_to_espn_year(season), cache_v=3)
import team_suitors as _ts
_pos2k = _ts.load_player_positions()
# Curated 2K position (primary + secondary, e.g. "PG/SG"), BBRef coarse fallback.
df["position"] = df["Player"].map(
    lambda n: _ts.resolve_position(n, _bref_positions.get(normalize(n), ""), _pos2k))

_next_contracts = fetch_next_year_contracts(season_to_espn_year(season), cache_v=7)
_rookie_scale   = fetch_rookie_scale_players(season)

def _fmt_next_contract_local(player_name: str) -> str:
    return fmt_next_contract(player_name, _next_contracts)

df["next_contract"] = df["Player"].apply(_fmt_next_contract_local)

def _style_rookie_salary(row):
    return style_rookie_salary(row, _rookie_scale)

# ══════════════════════════════════════════════════════════════════════════════
# Free Agent Class content
# ══════════════════════════════════════════════════════════════════════════════

def _fa_status(row) -> str | None:
    nc   = row["next_contract"]
    name = row["Player"]
    if nc == "RFA":
        return "RFA"
    if nc == "—":
        if normalize(name) in _rookie_scale:
            return "RFA"
        return "UFA"
    if " PO" in nc:
        return "Player Option"
    if " TO" in nc:
        return "Team Option"
    return None

fa_df = df.copy()
fa_df["Status"] = fa_df.apply(_fa_status, axis=1)
fa_df = fa_df[fa_df["Status"].notna()].copy()

n_ufa = (fa_df["Status"] == "UFA").sum()
n_rfa = (fa_df["Status"] == "RFA").sum()
n_po  = (fa_df["Status"] == "Player Option").sum()
n_to  = (fa_df["Status"] == "Team Option").sum()

# Summary stat cards — colour-coded to the table's status language (UFA slate ·
# RFA green · PO blue · TO orange · Total teal) so the page has a visual anchor
# instead of a flat native-metric row. Hover shows the explainer.
_fa_stats = [
    ("Total Free Agents", len(fa_df),  "var(--accent-teal)", "Everyone available this offseason"),
    ("Unrestricted · UFA", int(n_ufa), "var(--fg-3)",        "No strings, free to sign with any team"),
    ("Restricted · RFA",   int(n_rfa), "var(--value-good)",  "Team holds right of first refusal on any offer sheet"),
    ("Player Options",     int(n_po),  "var(--blue)",        "Player can opt out and hit the market"),
    ("Team Options",       int(n_to),  "var(--orange)",      "Team may decline, making the player available"),
]
_fa_cards = ""
for _lab, _val, _c, _tip in _fa_stats:
    _fa_cards += (
        f'<div class="fa-stat" style="--c:{_c};" title="{_tip}">'
        f'<div class="fa-stat-num">{_val}</div>'
        f'<div class="fa-stat-lab">{_lab}</div></div>'
    )
st.markdown(
    "<style>"
    ".fa-stats{display:flex;gap:0.7rem;flex-wrap:wrap;margin:1.5rem 0 0.3rem;}"
    ".fa-stat{flex:1 1 0;min-width:118px;background:var(--panel-solid);"
    "border:1px solid var(--panel-line);border-top:3px solid var(--c);"
    "border-radius:10px;padding:0.85rem 0.6rem 0.75rem;text-align:center;"
    "box-shadow:var(--shadow-card);transition:transform .12s ease;}"
    ".fa-stat:hover{transform:translateY(-2px);}"
    ".fa-stat-num{font-size:2rem;font-weight:800;line-height:1;color:var(--c);}"
    ".fa-stat-lab{font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;"
    "color:var(--fg-4);margin-top:0.45rem;font-weight:600;}"
    f"</style><div class='fa-stats'>{_fa_cards}</div>",
    unsafe_allow_html=True,
)

st.divider()

fa_col_a, fa_col_b, fa_col_c, fa_col_d = st.columns([2, 1, 1, 1])
with fa_col_a:
    fa_search = st.text_input("Filter by name", "", key="fa_search")
with fa_col_b:
    fa_status_filter = st.selectbox(
        "Status", ["All", "UFA", "RFA", "Player Option", "Team Option"], key="fa_status"
    )
with fa_col_c:
    fa_pos_filter = st.selectbox(
        "Position", ["All", "PG", "SG", "SF", "PF", "C"], key="fa_pos"
    )
with fa_col_d:
    fa_team_opts = ["All"] + sorted(fa_df["Team"].unique().tolist())
    fa_team_filter = st.selectbox("Team", fa_team_opts, key="fa_team")

fa_display = fa_df.copy()
if fa_search:
    fa_display = fa_display[fa_display["Player"].str.contains(fa_search, case=False)]
if fa_status_filter != "All":
    fa_display = fa_display[fa_display["Status"] == fa_status_filter]
if fa_pos_filter != "All":
    fa_display = fa_display[fa_display["position"].str.contains(fa_pos_filter, regex=False, na=False)]
if fa_team_filter != "All":
    fa_display = fa_display[fa_display["Team"] == fa_team_filter]

fa_display = fa_display.sort_values("barrett_score", ascending=False).reset_index(drop=True)

fa_fmt = fa_display[[
    "Player", "Team", "position", "Status",
    "barrett_score", "salary", "projected_salary", "value_diff", "next_contract",
]].copy()

fa_fmt["salary"]           = fa_fmt["salary"] / 1_000_000
fa_fmt["projected_salary"] = fa_fmt["projected_salary"] / 1_000_000
fa_fmt["value_diff"]       = fa_fmt["value_diff"] / 1_000_000

fa_fmt.columns = [
    "Player", "Team", "Pos", "Status",
    "Barrett Score", "Salary", "Proj. Value", "Δ Market", "Next $",
]
fa_fmt.insert(0, "#", range(1, len(fa_fmt) + 1))


# Token-based cell styles for the themed HTML table (follows light/dark; the
# legacy color_* helpers return hardcoded hex for the remaining native grids).
def _sty_status(v, _row):
    return {
        "UFA":           "color:var(--fg-3)",
        "RFA":           "color:var(--value-good);font-weight:700",
        "Player Option": "color:var(--blue);font-weight:700",
        "Team Option":   "color:var(--orange);font-weight:700",
    }.get(v, "")

def _sty_delta(v, _row):
    try:
        n = float(v)
    except (TypeError, ValueError):
        return ""
    if n > 20:  return "color:var(--value-bad);font-weight:700"
    if n > 5:   return "color:var(--value-bad-s)"
    if n < -20: return "color:var(--value-good);font-weight:700"
    if n < -5:  return "color:var(--value-good-s)"
    return ""

def _sty_next(v, _row):
    s = str(v)
    if s == "—":   return "color:var(--fg-6)"
    if " TO" in s: return "color:var(--orange);font-weight:700"
    if " PO" in s: return "color:var(--blue);font-weight:700"
    return ""

def _sty_salary(_v, row):
    if normalize(str(row.get("Player", ""))) in _rookie_scale:
        return "color:var(--purple);font-weight:600"
    return ""

html_table(
    fa_fmt,
    formatters={
        "Barrett Score": lambda v: f"{v:.2f}",
        "Salary":        lambda v: f"${v:.2f}M",
        "Proj. Value":   lambda v: f"${v:.2f}M",
        "Δ Market":      lambda v: f"${v:.2f}M",
    },
    styles={
        "Status":   _sty_status,
        "Next $":   _sty_next,
        "Δ Market": _sty_delta,
        "Salary":   _sty_salary,
    },
    aligns={
        "#": "right", "Barrett Score": "right", "Salary": "right",
        "Proj. Value": "right", "Δ Market": "right",
    },
    numeric={"#", "Barrett Score", "Salary", "Proj. Value", "Δ Market"},
    helps={
        "Barrett Score": "Base Score × Availability Multiplier. Higher = more valuable.",
        "Salary": "Current season salary. Purple = rookie-scale contract (1st-round pick, yrs 1–4).",
        "Proj. Value": "What this player would earn if paid by their Barrett Score rank, a market-rate anchor.",
        "Δ Market": "Actual − Projected. Negative (green) = underpaid; positive (red) = overpaid.",
        "Next $": "Next-year option salary; UFAs have no set figure. Blue = player option, orange = team option.",
        "Status": "UFA = unrestricted · RFA = restricted (right of first refusal) · PO/TO = player/team option.",
    },
    height=min(820, max(220, len(fa_fmt) * 38 + 46)),
)

fa_dl_col, fa_cap_col = st.columns([1, 5])
with fa_dl_col:
    st.download_button(
        "Export CSV",
        data=fa_fmt.to_csv(index=False),
        file_name=f"barrett_score_free_agents_{season}.csv",
        mime="text/csv",
        key="fa_csv",
    )
with fa_cap_col:
    st.caption(
        f"**{len(fa_display)}** free agents shown · "
        "**Proj. Value** = salary of the player at the same Barrett Score rank in the current pool, "
        "a market-rate anchor for what this player should cost. "
        "**Δ Market**: green = underpaid (will demand raise) · red = overpaid (value risk)."
    )

if not fa_display.empty:
    st.divider()
    st.subheader("Position breakdown")
    pos_status = (
        fa_display.groupby(["position", "Status"])
        .size()
        .reset_index(name="count")
    )
    pos_status = pos_status[pos_status["position"] != ""]
    if not pos_status.empty:
        fig_fa = px.bar(
            pos_status,
            x="position", y="count",
            color="Status",
            color_discrete_map={
                "UFA":           "#aaaaaa",
                "Player Option": "#3498db",
                "Team Option":   "#f39c12",
            },
            barmode="stack",
            labels={"position": "", "count": "Players", "Status": ""},
            height=320,
            category_orders={"position": ["Guard", "Forward", "Center"]},
            text_auto="d",
        )
        fig_fa.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.15)",
            font_color="white",
            margin=dict(t=20, b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickformat="d"),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1),
        )
        st.plotly_chart(theme_fig(fig_fa), use_container_width=True, config={"displayModeBar": False})


from utils import render_footer
render_footer()
