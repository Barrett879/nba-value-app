import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    COMMON_CSS, SEASONS, DEFAULT_MIN_THRESHOLD, SEASON_GAMES_LOOKUP,
    normalize, season_to_espn_year,
    build_ranked_projected, build_all_seasons_combined,
    fetch_bref_positions, fetch_next_year_contracts, fetch_rookie_scale_players,
    fetch_dlebron, fetch_career_trend, fetch_player_season_splits,
    fetch_monthly_scores, build_splits_data,
    _fmt_salary, fmt_next_contract,
    color_rank_diff, color_value_diff, color_next_contract, style_rookie_salary,
    render_nav, render_playoff_toggle, _bootstrap_warm,
)
import threading

st.set_page_config(page_title="Barrett Score — Rankings", layout="wide")

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

render_nav("Current Rankings")

if st.session_state.get("playoff_mode", False):
    st.title("Barrett Score — Playoff Rankings")
    st.caption("All scores computed from playoff games only. Salaries reflect the regular-season contract for that year.")
else:
    st.title("Barrett Score — NBA Contract Value Rankings")
    st.caption("A stat-driven ranking of every NBA player's contract value — who's underpaid, who's overpaid, and who's available.")

with st.expander("How is this calculated?"):
    st.markdown(
        "The Barrett Score's confidential formula combines scoring, playmaking, rebounding, defense, and efficiency "
        "into a single number. Then, it adjusts for how often they're actually on the floor. "
        "Salaries are then ranked against scores to find who's overpaid, underpaid, or worth exactly what they're making."
    )

# ── Season selector + playoff toggle ──────────────────────────────────────────
# playoff_mode is a session_state-backed sticky flag shared with Search,
# Legacy, Trades, and Team Analysis via render_playoff_toggle().
ctrl_l, ctrl_mid, ctrl_r = st.columns([1, 1, 1])
with ctrl_l:
    season = st.selectbox("Season", SEASONS, index=0)
with ctrl_mid:
    playoff_mode = render_playoff_toggle()

# Default minimum-minutes drops in playoff mode — playoff GP is 4-28 games,
# so a 500-min threshold would hide most of the field.
default_min_threshold = 100 if playoff_mode else DEFAULT_MIN_THRESHOLD
slider_max = 600 if playoff_mode else 1500
with ctrl_r:
    min_threshold = st.slider(
        "Min total minutes", min_value=0, max_value=slider_max,
        value=default_min_threshold, step=25 if playoff_mode else 50,
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
if df.empty:
    st.warning(
        f"No players cleared the {min_threshold}-minute threshold for "
        f"{season} {'playoffs' if playoff_mode else 'regular season'}. "
        "Lower the slider above to see more rows."
    )
    st.stop()

salary_lookup = tuple(
    (normalize(row["Player"]), row["salary"])
    for _, row in df.iterrows()
)

_bref_positions = fetch_bref_positions(season_to_espn_year(season), cache_v=3)
df["position"] = df["Player"].map(
    lambda n: _bref_positions.get(normalize(n), "")
)

_next_contracts = fetch_next_year_contracts(season_to_espn_year(season), cache_v=7)
_rookie_scale   = fetch_rookie_scale_players(season)

# ── Background cache warming (all seasons, bounded thread pool) ───────────────
_bootstrap_warm()

def _fmt_next_contract_local(player_name: str) -> str:
    return fmt_next_contract(player_name, _next_contracts)

df["next_contract"] = df["Player"].apply(_fmt_next_contract_local)

season_games = int(df["GP"].max())
splits_mpg_lookup = df.set_index("Player")["MPG"]

st.caption(
    f"**{len(df)}** players ranked · "
    f"**{(df['value_diff'] < -5_000_000).sum()}** underpaid (earning \\$5M+ below projection) · "
    f"**{(df['value_diff'] > 5_000_000).sum()}** overpaid (earning \\$5M+ above projection)"
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Hero callout cards
# ══════════════════════════════════════════════════════════════════════════════
_best_row      = df.loc[df["barrett_score"].idxmax()]
# Salary-based hero cards only meaningful when salary data exists.
# Pre-1996 has sparse coverage so $0-salary rows aren't actually "underpaid".
_value_pool = df[df["salary"] > 0]
if not _value_pool.empty:
    _steal_row     = _value_pool.loc[_value_pool["value_diff"].idxmin()]
    _overpaid_row  = _value_pool.loc[_value_pool["value_diff"].idxmax()]
else:
    _steal_row    = _best_row
    _overpaid_row = _best_row

# Most Improved: compare current season to the previous one
_season_idx   = SEASONS.index(season)
_prev_season  = SEASONS[_season_idx + 1] if _season_idx + 1 < len(SEASONS) else None
_improved_row = None
_improved_delta = None
_prev_df = None
if _prev_season:
    try:
        _prev_df  = build_ranked_projected(_prev_season, playoffs=playoff_mode)
        _prev_threshold = 100 if playoff_mode else DEFAULT_MIN_THRESHOLD
        _prev_df  = _prev_df[_prev_df["total_min"] >= _prev_threshold].copy()
        _merged   = df[["Player", "Team", "barrett_score"]].merge(
            _prev_df[["Player", "barrett_score"]].rename(columns={"barrett_score": "prev_score"}),
            on="Player", how="inner",
        )
        _merged["delta"] = _merged["barrett_score"] - _merged["prev_score"]
        _improved_row   = _merged.loc[_merged["delta"].idxmax()]
        _improved_delta = float(_improved_row["delta"])
    except Exception:
        pass

st.markdown("""
<style>
.hero-card {
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    height: 100%;
}
.hero-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em; opacity: .65; margin-bottom: .25rem; }
.hero-name  { font-size: 1.25rem; font-weight: 800; line-height: 1.2; }
.hero-sub   { font-size: 0.82rem; margin-top: .35rem; opacity: .75; }
</style>
""", unsafe_allow_html=True)

h1, h2, h3, h4 = st.columns(4, gap="medium")
with h1:
    st.markdown(f"""
    <div class="hero-card" style="background:#1a2e1a; border:1px solid #2ecc71;">
        <div class="hero-label">Best Player Right Now</div>
        <div class="hero-name">{_best_row['Player']}</div>
        <div class="hero-sub">{_best_row['Team']} · Score {_best_row['barrett_score']:.1f}</div>
    </div>""", unsafe_allow_html=True)
with h2:
    steal_diff = abs(_steal_row['value_diff'] / 1e6)
    st.markdown(f"""
    <div class="hero-card" style="background:#1a2a1a; border:1px solid #27ae60;">
        <div class="hero-label">Biggest Steal</div>
        <div class="hero-name">{_steal_row['Player']}</div>
        <div class="hero-sub">{_steal_row['Team']} · ${steal_diff:.1f}M below market value</div>
    </div>""", unsafe_allow_html=True)
with h3:
    over_diff = _overpaid_row['value_diff'] / 1e6
    st.markdown(f"""
    <div class="hero-card" style="background:#2e1a1a; border:1px solid #e74c3c;">
        <div class="hero-label">Most Overpaid</div>
        <div class="hero-name">{_overpaid_row['Player']}</div>
        <div class="hero-sub">{_overpaid_row['Team']} · ${over_diff:.1f}M above market value</div>
    </div>""", unsafe_allow_html=True)
with h4:
    if _improved_row is not None:
        _sign = "+" if _improved_delta >= 0 else ""
        st.markdown(f"""
        <div class="hero-card" style="background:#1a1a2e; border:1px solid #4cc9f0;">
            <div class="hero-label">Most Improved</div>
            <div class="hero-name">{_improved_row['Player']}</div>
            <div class="hero-sub">{_improved_row['Team']} · {_sign}{_improved_delta:.1f} pts vs last season</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="hero-card" style="background:#1a1a2e; border:1px solid #4cc9f0;">
            <div class="hero-label">Most Improved</div>
            <div class="hero-name">—</div>
            <div class="hero-sub">No prior season to compare</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Top 10 — current season bar chart
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Top 10 Players — Barrett Score")
st.caption(f"Current {season} Barrett Score with change vs prior season.")

_top10 = df.nsmallest(10, "score_rank")[["Player", "barrett_score"]].reset_index(drop=True)

# Attach prior-season delta if available (reuse _prev_df from Most Improved calc)
if _prev_df is not None:
    _top10 = _top10.merge(
        _prev_df[["Player", "barrett_score"]].rename(columns={"barrett_score": "prev_score"}),
        on="Player", how="left",
    )
    _top10["delta"] = _top10["barrett_score"] - _top10["prev_score"]
else:
    _top10["delta"] = float("nan")

# Sort descending so the best player is at the top of the horizontal chart
_top10 = _top10.sort_values("barrett_score", ascending=True)  # ascending=True → top at top in h-bar

def _delta_label(row):
    if pd.isna(row["delta"]):
        return f"{row['barrett_score']:.1f}"
    sign = "▲" if row["delta"] >= 0 else "▼"
    color = "#2ecc71" if row["delta"] >= 0 else "#e74c3c"
    return f"{row['barrett_score']:.1f}  <span style='color:{color};font-size:0.8em'>{sign} {abs(row['delta']):.1f}</span>"

# Bar colors: red for #1, fading to muted for #10
_bar_colors = [
    f"rgba(230,57,70,{0.5 + 0.05*i})" for i in range(len(_top10))
]

_fig_bar = go.Figure()
_fig_bar.add_trace(go.Bar(
    x=_top10["barrett_score"],
    y=_top10["Player"],
    orientation="h",
    marker=dict(
        color=_bar_colors,
        line=dict(width=0),
    ),
    text=[
        (f"▲ +{d:.1f}" if d >= 0 else f"▼ {d:.1f}") if not pd.isna(d) else ""
        for d in _top10["delta"]
    ],
    textposition="outside",
    textfont=dict(
        size=11,
        color=[
            ("#2ecc71" if (not pd.isna(d) and d >= 0) else "#e74c3c") if not pd.isna(d) else "#888"
            for d in _top10["delta"]
        ],
    ),
    customdata=_top10["delta"].apply(
        lambda d: f"{d:+.1f}" if not pd.isna(d) else "—"
    ).values,
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Barrett Score: %{x:.1f}<br>"
        "vs last season: %{customdata[0]}"
        "<extra></extra>"
    ),
))

_score_max = _top10["barrett_score"].max()
_fig_bar.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    height=360,
    margin=dict(l=10, r=80, t=10, b=30),
    showlegend=False,
    xaxis=dict(
        range=[0, _score_max * 1.18],
        gridcolor="rgba(255,255,255,0.06)",
        showticklabels=True,
        tickformat=".1f",
        title="",
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0)",
        title="",
        tickfont=dict(size=12),
    ),
    hovermode="closest",
    bargap=0.25,
)

st.plotly_chart(_fig_bar, use_container_width=True, config={"displayModeBar": False})
st.caption("▲ / ▼ = change in Barrett Score vs prior season")

st.divider()

# ── Rookie-scale style helper (closes over _rookie_scale) ──────────────────────
def _style_rookie_salary(row):
    return style_rookie_salary(row, _rookie_scale)

# ══════════════════════════════════════════════════════════════════════════════
# Rankings content
# ══════════════════════════════════════════════════════════════════════════════

player_id_map_full = dict(zip(df["Player"], df["PLAYER_ID"].astype(int)))
dlebron_lookup = fetch_dlebron(season)
_gp_sum = float(df["GP"].sum())
league_avg_ts = float((df["ts_pct"] * df["GP"]).sum() / _gp_sum) if _gp_sum > 0 else 0.0
if not dlebron_lookup:
    st.info("D-LEBRON (BBall Index's RAPM-based defensive metric) only goes back to 2009-10. "
            "For this season, defensive contribution is estimated from box-score stats — "
            "BLK, STL, DREB, and PF — centered on the season's league average. "
            "Calibrated to roughly match D-LEBRON's scale, but noisier (no on/off, no role-adjustment, "
            "no luck-adjustment).")


def render_splits_panel(player_name, season):
    if player_name not in player_id_map_full:
        return
    pid = player_id_map_full[player_name]
    d_leb = float(dlebron_lookup.get(pid, 0.0))
    splits = fetch_player_season_splits(pid, season, d_leb, league_avg_ts, season_games)
    if splits.empty:
        st.info("No per-team split data available.")
        return

    # For each row, override derived scores with the authoritative values from df
    # (df uses LeagueDashPlayerStats PerGame; splits use PlayerCareerStats totals/GP
    # which can differ slightly). For non-traded players this makes the panel match
    # the rankings table exactly. For traded players the TOT row gets the df values.
    df_row = df[df["Player"] == player_name]
    is_traded = len(splits) > 1  # multiple team stints

    rows_out = []
    row_styles = []
    for i, (_, r) in enumerate(splits.iterrows()):
        is_tot = r["Team"] == "TOT"

        # For single-team players or the TOT row, override derived scores with
        # the authoritative values from df (LeagueDashPlayerStats PerGame) so the
        # panel matches the rankings table exactly. df doesn't carry raw per-game
        # stats (PTS/AST/etc.) so those still come from the splits endpoint.
        use_main = (not is_traded) or is_tot
        if use_main and not df_row.empty:
            main = df_row.iloc[0]
            r = r.copy()
            r["ts_pct"]         = main["ts_pct"]
            r["efficiency_adj"] = main["efficiency_adj"]
            # Use the pace-adjusted base_score so Base × Avail = Barrett Score
            # arithmetic is internally consistent with the canonical column.
            r["base_score"]     = main.get("base_score_pace", main["base_score"])
            r["avail_mult"]     = main["avail_mult"]
            r["barrett_score"]  = main["barrett_score"]
            r["MPG"]            = main["MPG"]

        ts_str = f"{r['ts_pct']*100:.1f}%" if not pd.isna(r["ts_pct"]) else "—"
        rows_out.append({
            "#": i + 1, "Team": r["Team"],
            "GP": str(int(r["GP"])), "MPG": f"{r['MPG']:.2f}", "Total MIN": str(int(r["total_min"])),
            "PTS": f"{r['PTS']:.2f}", "AST": f"{r['AST']:.2f}",
            "OREB": f"{r['OREB']:.2f}", "DREB": f"{r['DREB']:.2f}",
            "BLK": f"{r['BLK']:.2f}", "STL": f"{r['STL']:.2f}",
            "TOV": f"{r['TOV']:.2f}", "PF": f"{r['PF']:.2f}",
            "D-LEBRON": f"{r['d_lebron']:.2f}", "TS%": ts_str,
            "Eff. Adj": f"{r['efficiency_adj']:.2f}",
            "Base Score": f"{r['base_score']:.2f}",
            "Avail ×": f"{r['avail_mult']:.3f}",
            "Barrett Score": f"{r['barrett_score']:.2f}",
        })
        row_styles.append("tot_stat" if is_tot else "stat")

    fmt = pd.DataFrame(rows_out)

    def style_rows(row):
        s = row_styles[row.name]
        if s == "tot_stat":
            return ["font-weight: bold; background-color: #2a2a2a"] * len(row)
        return [""] * len(row)

    st.dataframe(
        fmt.style.apply(style_rows, axis=1),
        column_config={
            "GP":           st.column_config.TextColumn(help="Games played with this team during the season."),
            "MPG":          st.column_config.TextColumn(help="Minutes per game with this team."),
            "Total MIN":    st.column_config.TextColumn(help="Total minutes played with this team."),
            "PTS":          st.column_config.NumberColumn(help="Points per game."),
            "AST":          st.column_config.NumberColumn(help="Assists per game."),
            "OREB":         st.column_config.NumberColumn(help="Offensive rebounds per game."),
            "DREB":         st.column_config.NumberColumn(help="Defensive rebounds per game."),
            "BLK":          st.column_config.NumberColumn(help="Blocks per game."),
            "STL":          st.column_config.NumberColumn(help="Steals per game."),
            "TOV":          st.column_config.NumberColumn(help="Turnovers per game."),
            "PF":           st.column_config.NumberColumn(help="Personal fouls per game."),
            "D-LEBRON":     st.column_config.NumberColumn(help="Defensive LEBRON — estimated points prevented per game vs average. Full-season metric, same across all stints."),
            "TS%":          st.column_config.TextColumn(help="True Shooting % — scoring efficiency across 2s, 3s, and free throws. PTS / (2 × (FGA + 0.44 × FTA)). League avg ~57%."),
            "Eff. Adj":     st.column_config.NumberColumn(help="Efficiency adjustment added to Base Score. clamp(0.15 × (TS% − League Avg TS%) × 100, −4, +4). Rewards efficient scorers, penalises inefficient ones."),
            "Base Score":   st.column_config.NumberColumn(help="PTS + AST×2 + OREB÷2 + DREB÷3 + BLK÷2 + STL÷1.5 − TOV÷1.5 − PF÷3 + D-LEBRON×2 + Eff. Adj. Raw per-game value before the availability multiplier."),
            "Avail ×":      st.column_config.NumberColumn(help="Availability multiplier (0.30–1.00). Rewards health and heavy minutes. 0.30 + 0.70 × √(min(Total MIN / (season games × 30.5), 1)). For traded players, season games is replaced by team games during that stint."),
            "Barrett Score":st.column_config.NumberColumn(help="Base Score × Availability Multiplier. The final contract value rating."),
        },
        use_container_width=True,
        hide_index=True,
    )


# ── Compare players (multiselect) ─────────────────────────────────────────
_all_seasons = build_all_seasons_combined()
# GP-weighted career avg so a 17-game cameo season doesn't drag legends down
# to role-player level in the dropdown ordering.
_career_avg_rnk = (
    _all_seasons.groupby("Player")
    .apply(lambda g: pd.Series({
        "avg_score": (g["barrett_score"] * g["GP"]).sum() / g["GP"].sum()
                     if g["GP"].sum() > 0 else g["barrett_score"].mean(),
    }))
    .reset_index()
    .sort_values("avg_score", ascending=False)
)
_current_players = set(df["Player"].unique())
all_player_names = (
    _career_avg_rnk[_career_avg_rnk["Player"].isin(_current_players)]["Player"].tolist()
)
compare_selected = st.multiselect(
    "Compare players",
    options=all_player_names,
    max_selections=10,
    placeholder="Select up to 10 players to compare…",
    key="rankings_multiselect",
)
st.session_state["rankings_selected"] = compare_selected

panel_placeholder = st.empty()

# ── Search / sort / team filters ──────────────────────────────────────────
col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
with col_a:
    search = st.text_input("Filter by player name", "")
with col_b:
    view = st.selectbox("Sort by", [
        "Barrett Score (best first)", "Salary (highest first)",
        "Most Underpaid", "Most Overpaid",
    ])
with col_c:
    team_options = ["All"] + sorted(df["Team"].unique().tolist())
    team_filter = st.selectbox("Team", team_options)
with col_d:
    pos_options = ["All", "Guard", "Forward", "Center"]
    pos_filter = st.selectbox("Position", pos_options)

display = df.copy()
if search:
    display = display[display["Player"].str.contains(search, case=False)]
if team_filter != "All":
    display = display[display["Team"] == team_filter]
if pos_filter != "All":
    display = display[display["position"] == pos_filter]

sort_map = {
    "Barrett Score (best first)": ("score_rank", True),
    "Salary (highest first)":     ("salary_rank", True),
    "Most Underpaid":             ("value_diff", True),
    "Most Overpaid":              ("value_diff", False),
}
sort_col, sort_asc = sort_map[view]
display = display.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

tog_a, tog_b, tog_c, tog_rest = st.columns([1, 1, 1, 5])
with tog_a:
    advanced = st.toggle("Advanced view", value=False)
with tog_b:
    if playoff_mode:
        show_splits = False
        st.caption("Splits view disabled in playoff mode")
    else:
        show_splits = st.toggle("Splits View", value=False,
                                help="Per-team stints ranked together. Team-switchers appear as separate rows.")
with tog_c:
    show_graph_mode = st.toggle("Graph mode", value=False,
                            help="Plot any two metrics against each other. Default view is "
                                 "Actual vs Projected salary; change axes to explore other relationships.")

# ── Graph mode (flexible scatter; defaults match the old Salary view) ─────────
if show_graph_mode:
    st.markdown("### Graph mode")
    st.caption("Plot any two metrics against each other. Pick X, Y, and color, then hover any dot for player detail.")

    # Drop rows with no salary data — they cluster at $0 and obscure salary axes.
    _n_no_salary = (display["salary"] <= 0).sum()
    gd = display[display["salary"] > 0].copy()
    gd["Salary $M"]        = (gd["salary"] / 1e6).round(2)
    gd["Proj. Salary $M"]  = (gd["projected_salary"] / 1e6).round(2)
    gd["Δ Market $M"]      = (gd["value_diff"] / 1e6).round(2)
    gd["Barrett Score"]    = gd["barrett_score"].round(2)
    gd["TS%"]              = (gd["ts_pct"] * 100).round(1)
    gd["D-LEBRON"]         = gd["d_lebron"].round(2)
    gd["Score Rank"]       = gd["score_rank"]
    gd["Salary Rank"]      = gd["salary_rank"]

    if _n_no_salary > 0:
        st.caption(
            f"⚠️ Hiding {_n_no_salary} player(s) with no salary data on file "
            "(common for older seasons). Their ranks remain accurate elsewhere."
        )

    NUMERIC_AXES = [
        "Barrett Score", "Salary $M", "Proj. Salary $M", "Δ Market $M",
        "GP", "MPG", "D-LEBRON", "TS%", "Score Rank", "Salary Rank",
    ]
    COLOR_NUMERIC     = ["Δ Market $M", "Barrett Score", "D-LEBRON", "TS%", "Salary $M"]
    COLOR_CATEGORICAL = ["position", "Team"]
    COLOR_AXES        = COLOR_NUMERIC + COLOR_CATEGORICAL

    g_col_a, g_col_b, g_col_c, g_col_d = st.columns([1, 1, 1, 2])
    with g_col_a:
        x_axis = st.selectbox("X axis", NUMERIC_AXES,
                              index=NUMERIC_AXES.index("Proj. Salary $M"),
                              key="graph_x")
    with g_col_b:
        y_axis = st.selectbox("Y axis", NUMERIC_AXES,
                              index=NUMERIC_AXES.index("Salary $M"),
                              key="graph_y")
    with g_col_c:
        color_axis = st.selectbox("Color by", COLOR_AXES, index=0, key="graph_color")
    with g_col_d:
        graph_search = st.text_input("Highlight player", "", key="graph_highlight",
                                      placeholder="Type a name to highlight on the chart…")

    is_color_numeric = color_axis in COLOR_NUMERIC

    scatter_kwargs = dict(
        x=x_axis, y=y_axis, color=color_axis,
        hover_name="Player",
        hover_data={
            "Team": True, "Score Rank": True,
            "Barrett Score": ":.2f",
            "Salary $M": ":.2f",
            "Δ Market $M": ":.2f",
        },
        height=520,
    )
    if is_color_numeric:
        scatter_kwargs["color_continuous_scale"] = (
            "RdYlGn_r" if color_axis == "Δ Market $M" else "Viridis"
        )
        if color_axis == "Δ Market $M":
            scatter_kwargs["color_continuous_midpoint"] = 0

    fig_scatter = px.scatter(gd, **scatter_kwargs)

    # Diagonal "fairly paid" reference only when both axes are salary
    is_salary_view = ({x_axis, y_axis} == {"Proj. Salary $M", "Salary $M"})
    if is_salary_view:
        axis_max = max(0.01, float(gd[[x_axis, y_axis]].max().max()) * 1.05)
        fig_scatter.add_shape(
            type="line", x0=0, y0=0, x1=axis_max, y1=axis_max,
            line=dict(color="rgba(255,255,255,0.5)", width=2, dash="dash"),
        )
        fig_scatter.add_annotation(
            x=axis_max * 0.72, y=axis_max * 0.82,
            text="Fairly paid", showarrow=False,
            font=dict(color="rgba(255,255,255,0.5)", size=11),
            textangle=-40,
        )
        fig_scatter.update_xaxes(range=[0, axis_max])
        fig_scatter.update_yaxes(range=[0, axis_max])

    if graph_search:
        hi = gd[gd["Player"].str.contains(graph_search, case=False)]
        if not hi.empty:
            fig_scatter.add_traces(px.scatter(
                hi, x=x_axis, y=y_axis, text="Player",
            ).update_traces(
                marker=dict(size=14, color="yellow", line=dict(color="black", width=1)),
                textposition="top center",
                name="Highlighted",
            ).data)

    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
        font_color="white",
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    # Salary axes get $/M formatting
    if "$M" in x_axis:
        fig_scatter.update_xaxes(tickprefix="$", ticksuffix="M", tickformat=".1f")
    if "$M" in y_axis:
        fig_scatter.update_yaxes(tickprefix="$", ticksuffix="M", tickformat=".1f")
    if color_axis == "Δ Market $M":
        fig_scatter.update_layout(coloraxis_colorbar=dict(title="Δ ($M)", tickformat=".1f"))

    st.plotly_chart(fig_scatter, use_container_width=True,
                    config={"displayModeBar": True, "displaylogo": False})

    # ── Companion table — sorted by value diff ────────────────────────────────
    tbl_col_a, tbl_col_b = st.columns([3, 1])
    with tbl_col_b:
        sort_mode = st.selectbox("Table sort",
                                  ["Most Overpaid", "Most Underpaid"], key="graph_sort")

    sc_tbl = gd[["Player", "Team", "Score Rank", "Barrett Score",
                 "Proj. Salary $M", "Salary $M", "Δ Market $M"]].copy()
    sc_tbl = sc_tbl.sort_values("Δ Market $M",
        ascending=(sort_mode == "Most Underpaid")).reset_index(drop=True)
    sc_tbl.insert(0, "#", range(1, len(sc_tbl) + 1))

    def _delta_color(val):
        try: n = float(val)
        except (ValueError, TypeError): return ""
        if n > 20:  return "color: #e74c3c; font-weight: bold"
        if n > 5:   return "color: #f1a8a8"
        if n < -20: return "color: #2ecc71; font-weight: bold"
        if n < -5:  return "color: #a8e6a8"
        return ""

    st.dataframe(
        sc_tbl.style.map(_delta_color, subset=["Δ Market $M"]),
        column_config={
            "Proj. Salary $M": st.column_config.NumberColumn(format="$%.2fM"),
            "Salary $M":       st.column_config.NumberColumn(format="$%.2fM"),
            "Δ Market $M":     st.column_config.NumberColumn(format="$%.2fM"),
            "Barrett Score":   st.column_config.NumberColumn(format="%.2f"),
        },
        use_container_width=True, hide_index=True, height=500,
    )
    st.caption(
        "**Δ Market** = Actual − Projected. Positive (red) = overpaid. Negative (green) = underpaid. "
        "Switch the X/Y dropdowns above to plot any two metrics — e.g. MPG vs Barrett Score, "
        "TS% vs Δ Market, or D-LEBRON vs Salary."
    )
    st.divider()

if show_splits:
    splits_df = build_splits_data(season, salary_lookup)

if show_splits and splits_df is not None:
    # ── Splits table ──────────────────────────────────────────────────────
    if search:
        matched = splits_df[splits_df["Player"].str.contains(search, case=False)]["Player"].unique()
        sdisplay = splits_df[splits_df["Player"].isin(matched)].copy()
    else:
        sdisplay = splits_df[splits_df["total_min"] >= min_threshold].copy()
    if team_filter != "All":
        team_players = sdisplay[sdisplay["Team"] == team_filter]["Player"].unique()
        sdisplay = sdisplay[
            (sdisplay["Team"] == team_filter) |
            ((sdisplay["Player"].isin(team_players)) & (sdisplay["Team"] == "TOT"))
        ]
    if pos_filter != "All":
        pos_players = set(df[df["position"] == pos_filter]["Player"])
        sdisplay = sdisplay[sdisplay["Player"].isin(pos_players)]

    # Use base_score_pace as the displayed "Base Score" so Base × Avail
    # = Barrett Score arithmetic is internally consistent with the
    # canonical (pace-adjusted) Barrett Score column.
    df_with_pace_alias = df.copy()
    if "base_score_pace" in df_with_pace_alias.columns:
        df_with_pace_alias["base_score"] = df_with_pace_alias["base_score_pace"]
    season_scores = df_with_pace_alias.set_index("Player")[["base_score", "avail_mult", "barrett_score"]]
    proj_lookup   = df.set_index("Player")[["salary", "projected_salary", "value_diff",
                                            "score_rank", "salary_rank", "rank_diff",
                                            "d_lebron", "ts_pct"]]
    traded_players = set(splits_df[splits_df["Team"] == "TOT"]["Player"])

    sdisplay = sdisplay[["Player", "Team", "GP", "total_min",
                          "base_score", "avail_mult", "barrett_score"]].copy()
    sdisplay["MPG"] = (sdisplay["total_min"] / sdisplay["GP"]).round(2)

    use_season = ~sdisplay["Player"].isin(traded_players) | (sdisplay["Team"] == "TOT")
    for col in ["base_score", "avail_mult", "barrett_score"]:
        sdisplay.loc[use_season, col] = sdisplay.loc[use_season, "Player"].map(season_scores[col])

    team_games = splits_df[splits_df["Team"] != "TOT"].groupby("Team")["GP"].max()
    MINS_PER_GAME_CAP = 2500 / 82
    stint_mask = sdisplay["Player"].isin(traded_players) & (sdisplay["Team"] != "TOT")
    if stint_mask.any():
        # v5 availability formula: 0.30 floor, sqrt of total_min/cap only.
        sdisplay.loc[stint_mask, "avail_mult"] = sdisplay[stint_mask].apply(
            lambda r: 0.30 + 0.70 * math.sqrt(
                min(r["total_min"] / (team_games.get(r["Team"], season_games) * MINS_PER_GAME_CAP), 1.0)
            ), axis=1
        )
        # Multiplying by base_score (which is base_score_pace from the alias
        # above) so the result matches the canonical pace-adjusted Barrett.
        sdisplay.loc[stint_mask, "barrett_score"] = (
            sdisplay.loc[stint_mask, "base_score"] * sdisplay.loc[stint_mask, "avail_mult"]
        )

    sdisplay = sdisplay[["Player", "Team", "GP", "MPG",
                          "base_score", "avail_mult", "barrett_score"]].join(
                          proj_lookup, on="Player", how="left")

    s_sort_map = {
        "Barrett Score (best first)": ("barrett_score", False),
        "Salary (highest first)":     ("salary", False),
        "Most Underpaid":             ("value_diff", True),
        "Most Overpaid":              ("value_diff", False),
    }
    s_col, s_asc = s_sort_map[view]
    sdisplay = sdisplay.sort_values(s_col, ascending=s_asc).reset_index(drop=True)

    def highlight_tot_row(row):
        if row["Team"] == "TOT":
            return ["font-weight: bold; background-color: #1e2a1e"] * len(row)
        if row["Player"] in traded_players:
            return ["background-color: #1a1f2e; color: #a0b0d0"] * len(row)
        return [""] * len(row)

    _nc_lookup = df.set_index("Player")["next_contract"]
    sdisplay["Next $"] = sdisplay["Player"].map(_nc_lookup).fillna("—")

    if advanced:
        sfmt = sdisplay[["Player", "Team", "GP", "MPG",
                          "base_score", "avail_mult", "barrett_score",
                          "score_rank", "salary", "projected_salary", "value_diff",
                          "salary_rank", "rank_diff", "d_lebron", "ts_pct", "Next $"]].copy()
        sfmt["salary"]           = sfmt["salary"] / 1_000_000
        sfmt["projected_salary"] = sfmt["projected_salary"] / 1_000_000
        sfmt["value_diff"]       = sfmt["value_diff"] / 1_000_000
        sfmt["ts_pct"]           = sfmt["ts_pct"] * 100
        sfmt.columns = ["Player", "Team", "GP", "MPG",
                        "Base Score", "Avail ×", "Barrett Score",
                        "Score Rank", "Salary", "Proj. Salary", "Δ Market",
                        "Salary Rank", "Rank Diff", "D-LEBRON", "TS%", "Next $"]
        sfmt.insert(0, "#", range(1, len(sfmt) + 1))
        s_style = sfmt.style.map(color_value_diff, subset=["Δ Market"]) \
                            .map(color_rank_diff, subset=["Rank Diff"]) \
                            .map(color_next_contract, subset=["Next $"]) \
                            .apply(_style_rookie_salary, axis=1) \
                            .apply(highlight_tot_row, axis=1)
        s_col_config = {
            "Next $":        st.column_config.TextColumn("Next $",
                help="Next season salary. White = guaranteed. Orange (TO) = team option. Blue (PO) = player option. Gray — = UFA.",
                width="medium"),
            "MPG":           st.column_config.NumberColumn(format="%.2f"),
            "Base Score":    st.column_config.NumberColumn(format="%.2f"),
            "Avail ×":       st.column_config.NumberColumn(format="%.3f"),
            "Barrett Score": st.column_config.NumberColumn(format="%.2f"),
            "Salary":        st.column_config.NumberColumn(format="$%.2fM",
                help="Player's actual salary this season. Purple = rookie scale contract (first-round pick, years 1–4)."),
            "Proj. Salary":  st.column_config.NumberColumn(format="$%.2fM"),
            "Δ Market":      st.column_config.NumberColumn(format="$%.2fM"),
            "D-LEBRON":      st.column_config.NumberColumn(format="%.2f"),
            "TS%":           st.column_config.NumberColumn(format="%.1f%%"),
        }
    else:
        sfmt = sdisplay[["Player", "Team", "barrett_score", "salary",
                          "projected_salary", "value_diff", "Next $"]].copy()
        sfmt["salary"]           = sfmt["salary"] / 1_000_000
        sfmt["projected_salary"] = sfmt["projected_salary"] / 1_000_000
        sfmt["value_diff"]       = sfmt["value_diff"] / 1_000_000
        sfmt.columns = ["Player", "Team", "Barrett Score", "Salary", "Proj. Salary", "Δ Market", "Next $"]
        sfmt.insert(0, "#", range(1, len(sfmt) + 1))
        s_style = sfmt.style.map(color_value_diff, subset=["Δ Market"]) \
                            .map(color_next_contract, subset=["Next $"]) \
                            .apply(_style_rookie_salary, axis=1) \
                            .apply(highlight_tot_row, axis=1)
        s_col_config = {
            "Next $":        st.column_config.TextColumn("Next $",
                help="Next season salary. White = guaranteed. Orange (TO) = team option. Blue (PO) = player option. Gray — = UFA.",
                width="medium"),
            "Barrett Score": st.column_config.NumberColumn(format="%.2f"),
            "Salary":        st.column_config.NumberColumn(format="$%.2fM",
                help="Player's actual salary this season. Purple = rookie scale contract (first-round pick, years 1–4)."),
            "Proj. Salary":  st.column_config.NumberColumn(format="$%.2fM"),
            "Δ Market":      st.column_config.NumberColumn(format="$%.2fM"),
        }

    st.dataframe(s_style, column_config=s_col_config,
                 use_container_width=True, hide_index=True, height=600)
    dl_col, cap_col = st.columns([1, 5])
    with dl_col:
        st.download_button(
            "Export CSV",
            data=sfmt.to_csv(index=False),
            file_name=f"barrett_score_splits_{season}.csv",
            mime="text/csv",
            key="splits_csv",
        )
    with cap_col:
        st.caption(
            f"{len(sdisplay)} rows shown. **TOT** = full season combined. "
            "Players who switched teams mid-season appear as separate stints. "
            "**Δ Market** = Actual − Projected. "
            "Purple salary = rookie scale. "
            "**Next $**: white = guaranteed · orange (TO) = team option · blue (PO) = player option · gray = UFA."
        )
else:
    # ── Rankings table ────────────────────────────────────────────────────
    # Use pace-adjusted base_score for the display so Base × Avail = Barrett
    # arithmetic is consistent. Falls back to raw base_score if the pace
    # column doesn't exist (older parquets pre-v6).
    _base_col = "base_score_pace" if "base_score_pace" in display.columns else "base_score"
    display_fmt = display[[
        "Player", "Team", "GP",
        _base_col, "avail_mult", "barrett_score",
        "score_rank", "salary", "projected_salary", "value_diff", "salary_rank", "rank_diff",
        "d_lebron", "ts_pct",
    ]].copy()
    if _base_col != "base_score":
        display_fmt = display_fmt.rename(columns={_base_col: "base_score"})
    display_fmt["MPG"] = display_fmt["Player"].map(splits_mpg_lookup)
    display_fmt = display_fmt[["Player", "Team", "GP", "MPG",
                               "base_score", "avail_mult", "barrett_score",
                               "score_rank", "salary", "projected_salary", "value_diff",
                               "salary_rank", "rank_diff", "d_lebron", "ts_pct"]]
    display_fmt["salary"]           = display_fmt["salary"] / 1_000_000
    display_fmt["projected_salary"] = display_fmt["projected_salary"] / 1_000_000
    display_fmt["value_diff"]       = display_fmt["value_diff"] / 1_000_000
    display_fmt["ts_pct"]           = display_fmt["ts_pct"] * 100
    display_fmt.columns = [
        "Player", "Team", "GP", "MPG",
        "Base Score", "Avail ×", "Barrett Score",
        "Score Rank", "Salary", "Proj. Salary", "Δ Market", "Salary Rank", "Rank Diff",
        "D-LEBRON", "TS%",
    ]
    display_fmt.insert(0, "#", range(1, len(display_fmt) + 1))
    display_fmt["Next $"] = display["next_contract"].values
    if not advanced:
        display_fmt = display_fmt[["#", "Player", "Team", "Barrett Score", "Salary", "Proj. Salary", "Δ Market", "Next $"]]

    style = display_fmt.style
    if "Rank Diff" in display_fmt.columns:
        style = style.map(color_rank_diff, subset=["Rank Diff"])
    if "Δ Market" in display_fmt.columns:
        style = style.map(color_value_diff, subset=["Δ Market"])
    if "Next $" in display_fmt.columns:
        style = style.map(color_next_contract, subset=["Next $"])
    style = style.apply(_style_rookie_salary, axis=1)

    col_config = {
        "Next $":        st.column_config.TextColumn("Next $",
            help="Next season salary. White = guaranteed. Orange (TO) = team option. Blue (PO) = player option. Gray — = unrestricted free agent.",
            width="medium"),
        "Barrett Score": st.column_config.NumberColumn(format="%.2f",
            help="Base Score × Availability Multiplier."),
        "Salary":        st.column_config.NumberColumn(format="$%.2fM",
            help="Player's actual salary this season. Purple = rookie scale contract (first-round pick, years 1–4)."),
        "Proj. Salary":  st.column_config.NumberColumn(format="$%.2fM",
            help="Salary earned by whoever holds the same rank by pay."),
        "Δ Market":      st.column_config.NumberColumn(format="$%.2fM",
            help="Actual − Projected. Positive (red) = overpaid. Negative (green) = underpaid."),
    }
    if advanced:
        col_config.update({
            "GP":         st.column_config.NumberColumn(format="%d", help="Games played this season."),
            "MPG":        st.column_config.NumberColumn(format="%.2f", help="Minutes per game."),
            "Base Score": st.column_config.NumberColumn(format="%.2f",
                help="PTS + AST×2 + OREB÷2 + DREB÷3 + BLK÷2 + STL÷1.5 − TOV÷1.5 − PF÷3 + D-LEBRON×2 + Eff. Adj."),
            "Avail ×":    st.column_config.NumberColumn(format="%.3f",
                help="0.30 + 0.70 × √(min(Total MIN / 2500, 1)). Range 0.30–1.00."),
            "Score Rank": st.column_config.NumberColumn(help="Rank by Barrett Score."),
            "Salary Rank":st.column_config.NumberColumn(help="Rank by actual salary."),
            "Rank Diff":  st.column_config.NumberColumn(help="Salary Rank − Score Rank. Positive = underpaid."),
            "D-LEBRON":   st.column_config.NumberColumn(format="%.2f",
                help="Points prevented per game vs average. From bball-index.com."),
            "TS%":        st.column_config.NumberColumn(format="%.1f%%",
                help="True Shooting %. League avg ~57%."),
        })

    table_height = min(600, max(100, len(display_fmt) * 35 + 40))
    st.dataframe(style, column_config=col_config,
                 use_container_width=True, hide_index=True, height=table_height)
    dl_col_r, cap_col_r = st.columns([1, 5])
    with dl_col_r:
        st.download_button(
            "Export CSV",
            data=display_fmt.to_csv(index=False),
            file_name=f"barrett_score_{season}.csv",
            mime="text/csv",
            key="rankings_csv",
        )
    with cap_col_r:
        if advanced:
            st.caption("**Rank Diff** = Salary Rank − Score Rank. **Δ Market** = Actual − Projected (red = overpaid, green = underpaid). "
                       "Purple salary = rookie scale contract (1st-round pick, yrs 1–4). "
                       "**Next $**: white = guaranteed · orange (TO) = team option · blue (PO) = player option · gray = UFA.")
        else:
            st.caption("**Proj. Salary** = what this player would earn paid by Barrett Score rank. **Δ Market** = Actual − Projected. "
                       "Purple salary = rookie scale contract (1st-round pick, yrs 1–4). "
                       "**Next $**: white = guaranteed · orange (TO) = team option · blue (PO) = player option · gray = UFA.")

# ── Fill panel placeholder (above multiselect) ────────────────────────────
new_selected = st.session_state.get("rankings_selected", [])
if new_selected:
    with panel_placeholder.container():

        all_trends = []
        for name in new_selected:
            if name in player_id_map_full:
                t = fetch_career_trend(
                    player_id_map_full[name],
                    num_seasons=20,
                    playoffs=playoff_mode,
                )
                if not t.empty:
                    t = t.copy()
                    t["Player"] = name
                    all_trends.append(t)

        if all_trends:
            trend_df = pd.concat(all_trends, ignore_index=True)

            available_seasons = sorted(trend_df["Season"].unique().tolist())
            sel_a, sel_b = st.columns(2)
            with sel_a:
                start_season = st.selectbox("From", available_seasons,
                                            index=max(0, len(available_seasons) - 5),
                                            key="trend_start")
            with sel_b:
                end_season = st.selectbox("To", available_seasons,
                                          index=len(available_seasons) - 1,
                                          key="trend_end")
            start_i = available_seasons.index(start_season)
            end_i   = available_seasons.index(end_season)
            if start_i > end_i:
                start_i, end_i = end_i, start_i
            selected_seasons = available_seasons[start_i : end_i + 1]
            trend_df = trend_df[trend_df["Season"].isin(selected_seasons)]
            trend_df = trend_df.sort_values("Season")
            n_seasons = len(selected_seasons)
            fig_trend = px.line(
                trend_df, x="Season", y="barrett_score",
                color="Player", markers=True,
                labels={"barrett_score": "Barrett Score", "Season": ""},
                title=f"Barrett Score — {n_seasons}-Season Trend",
                height=340,
                category_orders={"Season": selected_seasons},
            )
            cur = trend_df[trend_df["Season"] == season]
            if not cur.empty:
                fig_trend.add_scatter(
                    x=cur["Season"], y=cur["barrett_score"],
                    mode="markers",
                    marker=dict(size=14, symbol="star", color="white",
                                line=dict(width=1, color="black")),
                    showlegend=False,
                    hoverinfo="skip",
                )
            fig_trend.update_traces(line=dict(width=2.5), marker=dict(size=8),
                                    selector=dict(mode="lines+markers"))
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.15)",
                font_color="white",
                margin=dict(l=50, r=50, t=40, b=80),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="",
                           type="category",
                           categoryorder="array",
                           categoryarray=selected_seasons),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Barrett Score", tickformat=".1f"),
                legend=dict(
                    orientation="h",
                    x=0.5, xanchor="center",
                    y=-0.2, yanchor="top",
                    title="",
                ),
            )
            st.plotly_chart(fig_trend, use_container_width=True,
                            config={"displayModeBar": False})
            no_dlebron = [s for s in selected_seasons if s < "2009-10"]
            caption = "★ = current season"
            if no_dlebron:
                caption += (f"  ·  ⚠️ D-LEBRON unavailable for "
                            f"{', '.join(no_dlebron)} — defense estimated from box-score stats "
                            "(BLK, STL, DREB, PF) on the same scale")
            st.caption(caption)

        # ── Monthly cumulative trend (single-player only) ─────────────────────
        # Disabled in playoff mode — month-over-month is a regular-season concept
        # (playoffs are a single ~2-month window).
        if len(new_selected) == 1 and not playoff_mode:
            _solo = new_selected[0]
            if _solo in player_id_map_full:
                _solo_pid    = player_id_map_full[_solo]
                _solo_dleb   = float(dlebron_lookup.get(_solo_pid, 0.0))
                _monthly     = fetch_monthly_scores(_solo_pid, season, _solo_dleb, league_avg_ts)
                if not _monthly.empty:
                    st.markdown(f"#### {_solo} — {season} Monthly Cumulative Score")
                    st.caption("Each point is the cumulative season-to-date Barrett Score through end of that month.")

                    _season_score = float(df.loc[df["Player"] == _solo, "barrett_score"].iloc[0]) \
                        if _solo in df["Player"].values else None

                    _mfig = go.Figure()

                    # Shaded area under the line
                    _mfig.add_trace(go.Scatter(
                        x=_monthly["Month"], y=_monthly["barrett_score"],
                        mode="lines",
                        line=dict(color="rgba(230,57,70,0)"),
                        fill="tozeroy",
                        fillcolor="rgba(230,57,70,0.08)",
                        showlegend=False, hoverinfo="skip",
                    ))

                    # Main line
                    _mfig.add_trace(go.Scatter(
                        x=_monthly["Month"], y=_monthly["barrett_score"],
                        mode="lines+markers",
                        line=dict(color="#e63946", width=2.5),
                        marker=dict(size=8, color="#e63946",
                                    line=dict(width=1.5, color="white")),
                        customdata=_monthly[["GP", "team_GP", "avail_mult", "base_score"]].values,
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Barrett Score: <b>%{y:.1f}</b><br>"
                            "GP: %{customdata[0]} / %{customdata[1]} team games<br>"
                            "Avail ×: %{customdata[2]:.3f}<br>"
                            "Base Score: %{customdata[3]:.2f}"
                            "<extra></extra>"
                        ),
                        name="Barrett Score",
                        showlegend=False,
                    ))

                    # Dashed reference line for full-season score
                    if _season_score is not None:
                        _mfig.add_hline(
                            y=_season_score,
                            line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dot"),
                            annotation_text=f"Season avg {_season_score:.1f}",
                            annotation_position="top right",
                            annotation_font=dict(color="rgba(255,255,255,0.5)", size=11),
                        )

                    _mfig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.12)",
                        font_color="white",
                        height=300,
                        margin=dict(l=50, r=30, t=20, b=40),
                        xaxis=dict(
                            gridcolor="rgba(255,255,255,0.06)",
                            type="category",
                            categoryorder="array",
                            categoryarray=_monthly["Month"].tolist(),
                            title="",
                        ),
                        yaxis=dict(
                            gridcolor="rgba(255,255,255,0.06)",
                            title="Barrett Score",
                            tickformat=".1f",
                        ),
                        hovermode="x unified",
                    )
                    st.plotly_chart(_mfig, use_container_width=True,
                                    config={"displayModeBar": False})

        for name in new_selected:
            title_col, btn_col = st.columns([20, 1])
            with title_col:
                st.subheader(f"{name} — {season}")
            with btn_col:
                if st.button("✕", key=f"x_{name}", help="Remove"):
                    updated = [n for n in new_selected if n != name]
                    st.session_state["rankings_selected"] = updated
                    st.session_state["rankings_multiselect"] = updated
                    st.rerun()
            render_splits_panel(name, season)
        st.caption("**TOT** = full season combined (bold).")
        st.divider()
