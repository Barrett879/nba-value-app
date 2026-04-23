import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    COMMON_CSS, SEASONS, DEFAULT_MIN_THRESHOLD,
    normalize,
    build_all_seasons_combined, fetch_draft_classes,
    render_nav, _bootstrap_warm,
)

st.set_page_config(page_title="Barrett Score — Legacy", layout="wide", page_icon="🏀")

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
render_nav("🏛️ Legacy")

st.title("Barrett Score — Legacy")
st.caption(
    "The historical record. Every player, every season since 2006–07 — "
    "ranked, compared, and put in context."
)

# ── Load combined data ─────────────────────────────────────────────────────────
with st.spinner("Loading historical data across all seasons…"):
    all_df = build_all_seasons_combined()

if all_df.empty:
    st.error("Historical data unavailable. Please try again shortly.")
    st.stop()

# Season sort helper (ascending chronological order for charts)
def _season_year(s: str) -> int:
    return int(s.split("-")[0])

all_df["_season_year"] = all_df["Season"].apply(_season_year)
SEASONS_CHRON = sorted(all_df["Season"].unique(), key=_season_year)

# Pre-compute season-over-season delta per player (used by multiple sections)
_arc = (
    all_df.sort_values("_season_year")[["Player", "Season", "_season_year", "barrett_score"]]
    .copy()
)
_arc["prev_score"] = (
    _arc.groupby("Player")["barrett_score"].shift(1)
)
_arc["yoy_delta"] = _arc["barrett_score"] - _arc["prev_score"]


# ══════════════════════════════════════════════════════════════════════════════
# Hero cards — always visible at top
# ══════════════════════════════════════════════════════════════════════════════
_goat_row     = all_df.loc[all_df["barrett_score"].idxmax()]

# Most consistent: players with ≥5 qualifying seasons, ranked by avg score
_career_avg = (
    all_df.groupby("Player")
    .agg(avg_score=("barrett_score", "mean"), seasons=("Season", "nunique"))
    .reset_index()
)
_consistent = _career_avg[_career_avg["seasons"] >= 5].sort_values("avg_score", ascending=False)
_consistent_name  = _consistent.iloc[0]["Player"]  if not _consistent.empty else "—"
_consistent_avg   = _consistent.iloc[0]["avg_score"] if not _consistent.empty else 0
_consistent_seas  = int(_consistent.iloc[0]["seasons"]) if not _consistent.empty else 0

# Most undervalued single season
_steal_row = all_df.loc[all_df["value_diff"].idxmin()]

st.markdown("""
<style>
.legacy-hero {
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

lh1, lh2, lh3 = st.columns(3, gap="medium")
with lh1:
    st.markdown(f"""
    <div class="legacy-hero" style="background:#2a1a0a; border:1px solid #f39c12;">
        <div class="hero-label">Greatest Single Season</div>
        <div class="hero-name">{_goat_row['Player']}</div>
        <div class="hero-sub">{_goat_row['Season']} · {_goat_row['Team']} · Score {_goat_row['barrett_score']:.1f}</div>
    </div>""", unsafe_allow_html=True)
with lh2:
    st.markdown(f"""
    <div class="legacy-hero" style="background:#0a1a2a; border:1px solid #3498db;">
        <div class="hero-label">Most Consistent Career (≥5 seasons)</div>
        <div class="hero-name">{_consistent_name}</div>
        <div class="hero-sub">Avg {_consistent_avg:.1f} across {_consistent_seas} seasons</div>
    </div>""", unsafe_allow_html=True)
with lh3:
    steal_diff = abs(_steal_row["value_diff"] / 1e6)
    st.markdown(f"""
    <div class="legacy-hero" style="background:#0a2a0a; border:1px solid #2ecc71;">
        <div class="hero-label">Most Underpaid Season Ever</div>
        <div class="hero-name">{_steal_row['Player']}</div>
        <div class="hero-sub">{_steal_row['Season']} · {_steal_row['Team']} · ${steal_diff:.1f}M below market</div>
    </div>""", unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_rank, tab_arc, tab_era, tab_team, tab_long, tab_rec, tab_draft = st.tabs([
    "🏅 All-Time Rankings",
    "📈 Career Arc",
    "🗓️ Era Leaderboards",
    "🏟️ Team Legacy",
    "💎 Sustained Excellence",
    "📉 Records",
    "🎓 Draft Class",
])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: All-Time Rankings
# ─────────────────────────────────────────────────────────────────────────────
with tab_rank:
    st.subheader("Best Single-Season Barrett Scores — All Time")
    st.caption(
        "Every qualifying player-season since 2006–07, ranked by Barrett Score. "
        "Switch the sort to surface the most underpaid seasons in history."
    )

    rc1, rc2, rc3, rc4 = st.columns([2, 1, 1, 1])
    with rc1:
        rank_search = st.text_input("Filter by name", "", key="rank_search")
    with rc2:
        rank_season = st.selectbox("Season", ["All"] + list(reversed(SEASONS_CHRON)), key="rank_season")
    with rc3:
        rank_team_opts = ["All"] + sorted(all_df["Team"].unique().tolist())
        rank_team = st.selectbox("Team", rank_team_opts, key="rank_team")
    with rc4:
        rank_sort = st.selectbox(
            "Sort by",
            ["Barrett Score ↓", "Most Underpaid", "Most Overpaid"],
            key="rank_sort",
        )

    rank_display = all_df.copy()
    if rank_search:
        rank_display = rank_display[rank_display["Player"].str.contains(rank_search, case=False)]
    if rank_season != "All":
        rank_display = rank_display[rank_display["Season"] == rank_season]
    if rank_team != "All":
        rank_display = rank_display[rank_display["Team"] == rank_team]

    if rank_sort == "Barrett Score ↓":
        rank_display = rank_display.sort_values("barrett_score", ascending=False)
    elif rank_sort == "Most Underpaid":
        rank_display = rank_display.sort_values("value_diff", ascending=True)
    else:
        rank_display = rank_display.sort_values("value_diff", ascending=False)

    rank_display = rank_display.reset_index(drop=True)

    rank_tbl = rank_display[[
        "Player", "Season", "Team",
        "barrett_score", "score_rank",
        "salary", "value_diff",
    ]].copy()
    rank_tbl["salary"]     = rank_tbl["salary"]     / 1e6
    rank_tbl["value_diff"] = rank_tbl["value_diff"] / 1e6
    rank_tbl.columns = ["Player", "Season", "Team", "Barrett Score", "Season Rank", "Salary $M", "Δ Market $M"]
    rank_tbl.insert(0, "#", range(1, len(rank_tbl) + 1))

    def _color_delta_legacy(val):
        try:
            n = float(val)
        except (ValueError, TypeError):
            return ""
        if n < -20: return "color: #2ecc71; font-weight: bold"
        if n < -5:  return "color: #a8e6a8"
        if n > 20:  return "color: #e74c3c; font-weight: bold"
        if n > 5:   return "color: #f1a8a8"
        return ""

    st.dataframe(
        rank_tbl.style.map(_color_delta_legacy, subset=["Δ Market $M"]),
        column_config={
            "Barrett Score":  st.column_config.NumberColumn(format="%.2f"),
            "Season Rank":    st.column_config.NumberColumn(help="Rank within that season's pool."),
            "Salary $M":      st.column_config.NumberColumn(format="$%.2fM"),
            "Δ Market $M":    st.column_config.NumberColumn(
                format="$%.2fM",
                help="Actual − Projected within that season. Green = underpaid. Red = overpaid."),
        },
        use_container_width=True,
        hide_index=True,
        height=min(700, len(rank_tbl) * 35 + 40),
    )
    st.caption(f"**{len(rank_tbl):,}** player-seasons shown")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Career Arc
# ─────────────────────────────────────────────────────────────────────────────
with tab_arc:
    st.subheader("Career Arc")
    st.caption(
        "How did a player's Barrett Score evolve over their career? "
        "Only seasons where they met the minutes threshold (≥500) are shown."
    )

    arc_search = st.text_input(
        "Search player", "", key="arc_search",
        placeholder="e.g. LeBron James, Steph Curry, Nikola Jokic…"
    )

    if arc_search:
        arc_matches = all_df[all_df["Player"].str.contains(arc_search, case=False)]["Player"].unique()
        if len(arc_matches) == 0:
            st.info("No player found. Try a different spelling.")
        else:
            # If multiple matches, let user pick
            if len(arc_matches) > 1:
                arc_player = st.selectbox("Select player", sorted(arc_matches), key="arc_player_pick")
            else:
                arc_player = arc_matches[0]

            arc_df = (
                all_df[all_df["Player"] == arc_player]
                .sort_values("_season_year")[["Season", "barrett_score", "Team", "score_rank", "salary"]]
                .reset_index(drop=True)
            )

            if arc_df.empty:
                st.info("No qualifying seasons found.")
            else:
                peak_idx  = arc_df["barrett_score"].idxmax()
                peak_row  = arc_df.loc[peak_idx]
                avg_score = arc_df["barrett_score"].mean()

                # Summary metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Seasons Qualified", len(arc_df))
                mc2.metric("Peak Score", f"{peak_row['barrett_score']:.1f}", f"({peak_row['Season']})")
                mc3.metric("Career Average", f"{avg_score:.1f}")
                mc4.metric("Peak Season Rank", f"#{peak_row['score_rank']}")

                fig_arc = go.Figure()

                # Shaded area under the line
                fig_arc.add_trace(go.Scatter(
                    x=arc_df["Season"].tolist() + arc_df["Season"].tolist()[::-1],
                    y=arc_df["barrett_score"].tolist() + [0] * len(arc_df),
                    fill="toself",
                    fillcolor="rgba(241,196,15,0.08)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))

                # Main line
                fig_arc.add_trace(go.Scatter(
                    x=arc_df["Season"],
                    y=arc_df["barrett_score"],
                    mode="lines+markers",
                    line=dict(color="#f1c40f", width=3),
                    marker=dict(size=9, color="#f1c40f", line=dict(color="#fff", width=1.5)),
                    name=arc_player,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Score: %{y:.2f}<br>"
                        "<extra></extra>"
                    ),
                ))

                # Peak annotation
                fig_arc.add_annotation(
                    x=peak_row["Season"],
                    y=peak_row["barrett_score"],
                    text=f"  Peak: {peak_row['barrett_score']:.1f}",
                    showarrow=False,
                    font=dict(color="#f1c40f", size=11),
                    xanchor="left",
                    yanchor="bottom",
                )

                # Career average dashed line
                fig_arc.add_hline(
                    y=avg_score,
                    line_dash="dash",
                    line_color="rgba(255,255,255,0.3)",
                    annotation_text=f"Career avg {avg_score:.1f}",
                    annotation_position="right",
                    annotation_font_color="rgba(255,255,255,0.5)",
                )

                fig_arc.update_layout(
                    title=f"{arc_player} — Barrett Score by Season",
                    xaxis_title="",
                    yaxis_title="Barrett Score",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.15)",
                    font_color="white",
                    height=420,
                    margin=dict(t=50, b=20, r=120),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickangle=-35),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    showlegend=False,
                )
                st.plotly_chart(fig_arc, use_container_width=True, config={"displayModeBar": False})

                # Season-by-season table
                arc_tbl = arc_df.copy()
                arc_tbl["salary"] = arc_tbl["salary"] / 1e6
                arc_tbl.columns = ["Season", "Barrett Score", "Team", "Season Rank", "Salary $M"]
                st.dataframe(
                    arc_tbl.style.highlight_max(subset=["Barrett Score"], color="#4a3500", axis=0),
                    column_config={
                        "Barrett Score": st.column_config.NumberColumn(format="%.2f"),
                        "Salary $M":     st.column_config.NumberColumn(format="$%.2fM"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("👆 Type a player name above to see their career arc.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Era Leaderboards
# ─────────────────────────────────────────────────────────────────────────────
with tab_era:
    st.subheader("Era Leaderboards")
    st.caption("Average Barrett Score within each era. Minimum 2 qualifying seasons required.")

    ERAS = {
        "Pre-Analytics\n(2006–2012)":  [s for s in SEASONS_CHRON if _season_year(s) <= 2011],
        "Pace & Space\n(2012–2018)":   [s for s in SEASONS_CHRON if 2011 < _season_year(s) <= 2017],
        "Modern\n(2018–present)":      [s for s in SEASONS_CHRON if _season_year(s) > 2017],
    }

    era_cols = st.columns(3, gap="medium")
    for col, (era_name, era_seasons) in zip(era_cols, ERAS.items()):
        with col:
            era_df = all_df[all_df["Season"].isin(era_seasons)]
            era_stats = (
                era_df.groupby("Player")
                .agg(avg_score=("barrett_score", "mean"),
                     peak_score=("barrett_score", "max"),
                     seasons=("Season", "nunique"))
                .reset_index()
            )
            era_stats = era_stats[era_stats["seasons"] >= 2].sort_values("avg_score", ascending=False)
            era_stats["avg_score"]  = era_stats["avg_score"].round(2)
            era_stats["peak_score"] = era_stats["peak_score"].round(2)
            era_stats.insert(0, "#", range(1, len(era_stats) + 1))
            era_stats.columns = ["#", "Player", "Avg Score", "Peak Score", "Seasons"]

            era_label = era_name.replace("\n", " ")
            st.markdown(f"**{era_label}**")
            st.dataframe(
                era_stats.head(15),
                column_config={
                    "Avg Score":  st.column_config.NumberColumn(format="%.2f"),
                    "Peak Score": st.column_config.NumberColumn(format="%.2f"),
                },
                use_container_width=True,
                hide_index=True,
                height=560,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Team Legacy — Mount Rushmore
# ─────────────────────────────────────────────────────────────────────────────
with tab_team:
    st.subheader("Team Legacy — Mount Rushmore")
    st.caption(
        "The four best single-season Barrett Scores in franchise history. "
        "Note: team abbreviations reflect the season they were played (e.g. NJN = Brooklyn)."
    )

    teams_list = sorted(all_df["Team"].unique().tolist())
    rush_team  = st.selectbox("Select franchise", teams_list, key="rush_team")

    rush_df = (
        all_df[all_df["Team"] == rush_team]
        .sort_values("barrett_score", ascending=False)
        .head(4)
        .reset_index(drop=True)
    )

    if rush_df.empty:
        st.info("No data for this team.")
    else:
        rush_df["label"] = rush_df.apply(
            lambda r: f"{r['Player']}<br><span style='font-size:11px'>{r['Season']}</span>", axis=1
        )
        rush_df_sorted = rush_df.sort_values("barrett_score", ascending=True)

        fig_rush = go.Figure(go.Bar(
            x=rush_df_sorted["barrett_score"],
            y=rush_df_sorted.apply(lambda r: f"{r['Player']}  ({r['Season']})", axis=1),
            orientation="h",
            marker=dict(
                color=rush_df_sorted["barrett_score"],
                colorscale=[[0, "#b8860b"], [0.5, "#daa520"], [1, "#f1c40f"]],
                showscale=False,
                line=dict(color="rgba(0,0,0,0.3)", width=1),
            ),
            text=rush_df_sorted["barrett_score"].apply(lambda v: f"{v:.1f}"),
            textposition="outside",
            textfont=dict(color="white", size=13),
        ))
        fig_rush.update_layout(
            title=f"{rush_team} — All-Time Top 4 Barrett Score Seasons",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.15)",
            font_color="white",
            height=300,
            margin=dict(t=50, b=20, l=20, r=60),
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Barrett Score"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_rush, use_container_width=True, config={"displayModeBar": False})

        rush_tbl = rush_df[["Player", "Season", "barrett_score", "score_rank", "salary", "value_diff"]].copy()
        rush_tbl["salary"]     /= 1e6
        rush_tbl["value_diff"] /= 1e6
        rush_tbl.columns = ["Player", "Season", "Barrett Score", "League Rank That Season", "Salary $M", "Δ Market $M"]
        rush_tbl.insert(0, "#", range(1, len(rush_tbl) + 1))

        st.dataframe(
            rush_tbl.style.map(_color_delta_legacy, subset=["Δ Market $M"]),
            column_config={
                "Barrett Score":            st.column_config.NumberColumn(format="%.2f"),
                "League Rank That Season":  st.column_config.NumberColumn(help="Their score rank among all players that year."),
                "Salary $M":                st.column_config.NumberColumn(format="$%.2fM"),
                "Δ Market $M":              st.column_config.NumberColumn(format="$%.2fM"),
            },
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.markdown("**Full franchise leaderboard**")
        full_rush = (
            all_df[all_df["Team"] == rush_team]
            .sort_values("barrett_score", ascending=False)
            .reset_index(drop=True)
        )
        full_rush_tbl = full_rush[["Player", "Season", "barrett_score", "score_rank"]].copy()
        full_rush_tbl.columns = ["Player", "Season", "Barrett Score", "League Rank"]
        full_rush_tbl.insert(0, "#", range(1, len(full_rush_tbl) + 1))
        st.dataframe(
            full_rush_tbl,
            column_config={"Barrett Score": st.column_config.NumberColumn(format="%.2f")},
            use_container_width=True,
            hide_index=True,
            height=400,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5: Sustained Excellence (Longevity)
# ─────────────────────────────────────────────────────────────────────────────
with tab_long:
    st.subheader("Sustained Excellence")
    st.caption(
        "Who held elite Barrett Scores the longest? "
        "Average score across all qualifying seasons (≥500 min). "
        "Adjust the minimum season requirement below."
    )

    min_seas = st.slider("Minimum qualifying seasons", 2, 10, 5, key="long_min_seas")

    long_df = (
        all_df.groupby("Player")
        .agg(
            avg_score   = ("barrett_score", "mean"),
            peak_score  = ("barrett_score", "max"),
            min_score   = ("barrett_score", "min"),
            seasons     = ("Season", "nunique"),
            peak_season = ("Season", lambda s: all_df.loc[s.index[all_df.loc[s.index, "barrett_score"].argmax()], "Season"]),
            teams       = ("Team", lambda t: ", ".join(sorted(t.unique()))),
        )
        .reset_index()
    )
    long_df = long_df[long_df["seasons"] >= min_seas].sort_values("avg_score", ascending=False).reset_index(drop=True)
    long_df["avg_score"]  = long_df["avg_score"].round(2)
    long_df["peak_score"] = long_df["peak_score"].round(2)
    long_df["min_score"]  = long_df["min_score"].round(2)
    long_df.insert(0, "#", range(1, len(long_df) + 1))
    long_df.columns = ["#", "Player", "Avg Score", "Peak Score", "Floor Score", "Seasons", "Peak Season", "Teams"]

    # Scatter: avg score vs seasons (longevity map)
    fig_long = px.scatter(
        long_df,
        x="Seasons",
        y="Avg Score",
        hover_name="Player",
        hover_data={"Peak Score": ":.2f", "Peak Season": True},
        color="Avg Score",
        color_continuous_scale="YlOrRd",
        size="Avg Score",
        size_max=20,
        labels={"Seasons": "Qualifying Seasons", "Avg Score": "Average Barrett Score"},
        height=400,
    )
    fig_long.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
        font_color="white",
        coloraxis_showscale=False,
        margin=dict(t=20, b=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
    # Label top 10
    top_long = long_df.head(10)
    for _, row in top_long.iterrows():
        fig_long.add_annotation(
            x=row["Seasons"], y=row["Avg Score"],
            text=f"  {row['Player'].split()[-1]}",
            showarrow=False,
            font=dict(color="rgba(255,255,255,0.7)", size=10),
            xanchor="left",
        )
    st.plotly_chart(fig_long, use_container_width=True, config={"displayModeBar": False})

    st.dataframe(
        long_df,
        column_config={
            "Avg Score":   st.column_config.NumberColumn(format="%.2f",
                help="Average Barrett Score across all qualifying seasons."),
            "Peak Score":  st.column_config.NumberColumn(format="%.2f"),
            "Floor Score": st.column_config.NumberColumn(format="%.2f",
                help="Their lowest qualifying season score — the floor of their value."),
        },
        use_container_width=True,
        hide_index=True,
        height=min(600, len(long_df) * 35 + 40),
    )
    st.caption(
        f"**{len(long_df)}** players with ≥ {min_seas} qualifying seasons shown. "
        "Bubble size = average score."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6: Records (Most Undervalued + The Fall)
# ─────────────────────────────────────────────────────────────────────────────
with tab_rec:
    rec_sub1, rec_sub2 = st.tabs(["💰 Most Undervalued Ever", "📉 The Fall"])

    with rec_sub1:
        st.subheader("Most Undervalued Seasons in NBA History")
        st.caption(
            "Biggest gaps between what a player earned and what their Barrett Score rank deserved. "
            "These are the GMs who got away with something."
        )

        underval = (
            all_df.sort_values("value_diff", ascending=True)
            .reset_index(drop=True)
            .head(50)
        )
        uv_tbl = underval[[
            "Player", "Season", "Team",
            "barrett_score", "score_rank",
            "salary", "projected_salary", "value_diff",
        ]].copy()
        uv_tbl["salary"]           /= 1e6
        uv_tbl["projected_salary"] /= 1e6
        uv_tbl["value_diff"]       /= 1e6
        uv_tbl.columns = ["Player", "Season", "Team", "Barrett Score", "Score Rank", "Actual $M", "Proj. $M", "Δ Market $M"]
        uv_tbl.insert(0, "#", range(1, len(uv_tbl) + 1))

        st.dataframe(
            uv_tbl.style.map(_color_delta_legacy, subset=["Δ Market $M"]),
            column_config={
                "Barrett Score": st.column_config.NumberColumn(format="%.2f"),
                "Actual $M":     st.column_config.NumberColumn(format="$%.2fM"),
                "Proj. $M":      st.column_config.NumberColumn(format="$%.2fM"),
                "Δ Market $M":   st.column_config.NumberColumn(format="$%.2fM",
                    help="Negative = underpaid. The more negative, the bigger the steal for the team."),
            },
            use_container_width=True,
            hide_index=True,
            height=500,
        )

    with rec_sub2:
        st.subheader("The Fall — Biggest Single-Season Score Drops")
        st.caption(
            "Players whose Barrett Score fell the most from one season to the next — "
            "injuries, age, system changes, or contract-year motivation."
        )

        fall_df = _arc.dropna(subset=["yoy_delta"]).copy()
        fall_df = fall_df.sort_values("yoy_delta", ascending=True).reset_index(drop=True)

        fall_tbl = fall_df[[
            "Player", "Season", "barrett_score", "prev_score", "yoy_delta"
        ]].head(50).copy()
        fall_tbl["barrett_score"] = fall_tbl["barrett_score"].round(2)
        fall_tbl["prev_score"]    = fall_tbl["prev_score"].round(2)
        fall_tbl["yoy_delta"]     = fall_tbl["yoy_delta"].round(2)
        fall_tbl.columns = ["Player", "Season", "Score That Year", "Score Prev. Season", "Δ Score"]
        fall_tbl.insert(0, "#", range(1, len(fall_tbl) + 1))

        def _color_fall(val):
            try:
                n = float(val)
            except (ValueError, TypeError):
                return ""
            if n < -10: return "color: #e74c3c; font-weight: bold"
            if n < -4:  return "color: #f1a8a8"
            return ""

        st.dataframe(
            fall_tbl.style.map(_color_fall, subset=["Δ Score"]),
            column_config={
                "Score That Year":    st.column_config.NumberColumn(format="%.2f"),
                "Score Prev. Season": st.column_config.NumberColumn(format="%.2f"),
                "Δ Score":            st.column_config.NumberColumn(format="%.2f",
                    help="Negative = score dropped vs prior season. Larger drop = bigger fall."),
            },
            use_container_width=True,
            hide_index=True,
            height=500,
        )

        # Mini chart: biggest single fall
        if not fall_df.empty:
            worst_fall = fall_df.iloc[0]
            wf_player  = worst_fall["Player"]
            wf_arc = (
                all_df[all_df["Player"] == wf_player]
                .sort_values("_season_year")[["Season", "barrett_score"]]
            )
            if len(wf_arc) >= 2:
                fig_fall = px.line(
                    wf_arc, x="Season", y="barrett_score",
                    markers=True,
                    title=f"Biggest Fall: {wf_player}",
                    labels={"barrett_score": "Barrett Score", "Season": ""},
                    height=280,
                )
                fig_fall.update_traces(line_color="#e74c3c", line_width=2.5,
                                       marker=dict(size=8, color="#e74c3c"))
                fig_fall.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.15)",
                    font_color="white",
                    margin=dict(t=40, b=10),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickangle=-30),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                )
                st.plotly_chart(fig_fall, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# Tab 7: Draft Class
# ─────────────────────────────────────────────────────────────────────────────
with tab_draft:
    st.subheader("Draft Class Legacy")
    st.caption(
        "How did a draft class pan out? Select a draft year to see every player from "
        "that class who appears in our data, with their Barrett Score arcs side by side."
    )

    with st.spinner("Loading draft history…"):
        draft_df = fetch_draft_classes()

    if draft_df.empty:
        st.warning("Draft history data unavailable. Please try again later.")
    else:
        # Available draft years that overlap with our season data
        avail_years = sorted(
            [y for y in draft_df["draft_year"].unique() if 2003 <= y <= 2023],
            reverse=True,
        )
        draft_year_sel = st.selectbox(
            "Draft Year", avail_years, key="draft_year_sel",
            help="Players drafted this year who have ≥1 qualifying season in our data.",
        )

        class_players = draft_df[draft_df["draft_year"] == draft_year_sel]["player_norm"].tolist()

        # Match to our combined dataset by normalized name
        class_arc = all_df[
            all_df["Player"].apply(normalize).isin(class_players)
        ].copy()

        if class_arc.empty:
            st.info(f"No players from the {draft_year_sel} draft class found in our data yet.")
        else:
            # Show pick info where available
            draft_info = (
                draft_df[draft_df["draft_year"] == draft_year_sel]
                [["Player", "player_norm", "ROUND_NUMBER", "ROUND_PICK", "OVERALL_PICK"]]
                .copy()
            )
            # Merge peak score per player
            class_peaks = (
                class_arc.groupby("Player")
                .agg(peak_score=("barrett_score", "max"), seasons=("Season", "nunique"))
                .reset_index()
            )
            class_peaks["player_norm"] = class_peaks["Player"].apply(normalize)
            class_summary = class_peaks.merge(draft_info, on="player_norm", how="left")
            class_summary = class_summary.sort_values("peak_score", ascending=False).reset_index(drop=True)

            mc1, mc2 = st.columns([1, 2])
            with mc1:
                st.markdown(f"**{draft_year_sel} Class — {len(class_summary)} players in our data**")
                sum_tbl = class_summary[["Player", "OVERALL_PICK", "peak_score", "seasons"]].copy()
                sum_tbl.columns = ["Player", "Pick #", "Peak Score", "Seasons"]
                sum_tbl["Peak Score"] = sum_tbl["Peak Score"].round(2)
                st.dataframe(
                    sum_tbl,
                    column_config={
                        "Peak Score": st.column_config.NumberColumn(format="%.2f"),
                        "Pick #":     st.column_config.NumberColumn(format="%d"),
                    },
                    use_container_width=True,
                    hide_index=True,
                    height=min(500, len(sum_tbl) * 35 + 40),
                )

            with mc2:
                # Multi-line arc chart — top 10 by peak score
                top_class = class_summary.head(10)["Player"].tolist()
                arc_data  = class_arc[class_arc["Player"].isin(top_class)].sort_values("_season_year")

                fig_class = px.line(
                    arc_data,
                    x="Season", y="barrett_score",
                    color="Player",
                    markers=True,
                    labels={"barrett_score": "Barrett Score", "Season": ""},
                    title=f"{draft_year_sel} Draft Class — Career Arcs (Top 10 by peak)",
                    height=460,
                )
                fig_class.update_traces(line_width=2, marker_size=6)
                fig_class.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.15)",
                    font_color="white",
                    legend=dict(orientation="v", x=1.01, y=1, font_size=10),
                    margin=dict(t=50, b=20, r=140),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickangle=-30),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                )
                st.plotly_chart(fig_class, use_container_width=True, config={"displayModeBar": False})
