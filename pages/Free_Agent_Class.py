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
    _fmt_salary, fmt_next_contract,
    color_next_contract, style_rookie_salary, color_value_diff,
)

st.set_page_config(page_title="Barrett Score — Free Agent Class", layout="wide", page_icon="🏀")

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

st.title("Barrett Score — Free Agent Class")

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

st.caption(
    "Every player whose contract situation makes them available this offseason — "
    "UFAs, RFAs (team holds right of first refusal), player options (they may opt out), "
    "and team options (team may decline). Ranked by Barrett Score — a GM's draft board."
)

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

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Free Agents", len(fa_df))
m2.metric("Unrestricted (UFA)", n_ufa)
m3.metric("Restricted (RFA)",   n_rfa, help="Team holds right of first refusal on any offer sheet")
m4.metric("Player Options",     n_po,  help="Player can opt out and hit the market")
m5.metric("Team Options",       n_to,  help="Team may decline, making player available")

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
        "Position", ["All", "Guard", "Forward", "Center"], key="fa_pos"
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
    fa_display = fa_display[fa_display["position"] == fa_pos_filter]
if fa_team_filter != "All":
    fa_display = fa_display[fa_display["Team"] == fa_team_filter]

fa_display = fa_display.sort_values("barrett_score", ascending=False).reset_index(drop=True)

fa_fmt = fa_display[[
    "Player", "Team", "position", "Status",
    "barrett_score", "salary", "projected_salary", "value_diff", "next_contract",
]].copy()

fa_fmt["salary"]           = [_fmt_salary(p, s) for p, s in zip(fa_fmt["Player"].values, fa_fmt["salary"].values)]
fa_fmt["projected_salary"] = fa_fmt["projected_salary"] / 1_000_000
fa_fmt["value_diff"]       = fa_fmt["value_diff"] / 1_000_000

fa_fmt.columns = [
    "Player", "Team", "Pos", "Status",
    "Barrett Score", "Salary", "Proj. Value", "Δ Market", "Next $",
]
fa_fmt.insert(0, "#", range(1, len(fa_fmt) + 1))


def color_fa_status(val):
    if val == "UFA":
        return "color: #aaaaaa"
    if val == "RFA":
        return "color: #2ecc71; font-weight: bold"
    if val == "Player Option":
        return "color: #3498db; font-weight: bold"
    if val == "Team Option":
        return "color: #f39c12; font-weight: bold"
    return ""


fa_style = (
    fa_fmt.style
    .map(color_fa_status,    subset=["Status"])
    .map(color_next_contract, subset=["Next $"])
    .apply(_style_rookie_salary, axis=1)
    .map(color_value_diff,    subset=["Δ Market"])
)

st.dataframe(
    fa_style,
    column_config={
        "Barrett Score": st.column_config.NumberColumn(format="%.2f",
            help="Base Score × Availability Multiplier. Higher = more valuable."),
        "Salary":        st.column_config.TextColumn(width="medium",
            help="Current season salary. Purple = rookie scale contract (1st-round pick, yrs 1–4)."),
        "Proj. Value":   st.column_config.NumberColumn(format="$%.2fM",
            help="What this player would earn if paid by their Barrett Score rank. "
                 "Useful baseline for contract negotiations."),
        "Δ Market":      st.column_config.NumberColumn(format="$%.2fM",
            help="Actual − Projected. Negative (green) = currently underpaid — "
                 "expect them to command a raise. Positive (red) = overpaid relative to production."),
        "Next $":        st.column_config.TextColumn("Next $",
            help="Option value or — for UFAs. Blue (PO) = player option. Orange (TO) = team option.",
            width="medium"),
        "Status":        st.column_config.TextColumn(
            help="UFA = unrestricted free agent. RFA = restricted (team has right of first refusal on offer sheets). "
                 "Player Option = player controls opt-out. Team Option = team controls whether to keep player."),
        "Pos":           st.column_config.TextColumn("Pos", width="small"),
    },
    use_container_width=True,
    hide_index=True,
    height=min(800, max(200, len(fa_fmt) * 35 + 40)),
)

fa_dl_col, fa_cap_col = st.columns([1, 5])
with fa_dl_col:
    st.download_button(
        "⬇ Export CSV",
        data=fa_fmt.to_csv(index=False),
        file_name=f"barrett_score_free_agents_{season}.csv",
        mime="text/csv",
        key="fa_csv",
    )
with fa_cap_col:
    st.caption(
        f"**{len(fa_display)}** free agents shown · "
        "**Proj. Value** = salary of the player at the same Barrett Score rank in the current pool — "
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
            text_auto=True,
        )
        fig_fa.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.15)",
            font_color="white",
            margin=dict(t=20, b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1),
        )
        st.plotly_chart(fig_fa, use_container_width=True, config={"displayModeBar": False})
