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
    fetch_player_career_all_seasons,
    render_nav, render_page_chrome, html_table, stat_cards,
    theme_fig, render_playoff_toggle, render_barrett_score_explainer, _bootstrap_warm,
    tier_color, gradient_points,
)


# ── Token-based cell styles for themed html_tables (follow light/dark) ─────────
def _hl_delta(v, _row):
    try:
        n = float(v)
    except (TypeError, ValueError):
        return ""
    if n > 20:  return "color:var(--value-bad);font-weight:700"
    if n > 5:   return "color:var(--value-bad-s)"
    if n < -20: return "color:var(--value-good);font-weight:700"
    if n < -5:  return "color:var(--value-good-s)"
    return ""

def _hl_fall(v, _row):
    try:
        n = float(v)
    except (TypeError, ValueError):
        return ""
    if n < -10: return "color:var(--value-bad);font-weight:700"
    if n < -4:  return "color:var(--value-bad-s)"
    return ""

st.set_page_config(page_title="Legacy", page_icon="static/favicon.svg", layout="wide")

render_page_chrome()
_bootstrap_warm()
render_nav("Legacy")

# Playoff toggle sits on the title row, right-aligned (in-page, not the nav bar)
playoff_mode = bool(st.session_state.get("playoff_mode", False))
_tcol, _pcol = st.columns([5, 1], vertical_alignment="center")
with _tcol:
    st.title("Playoff Legacy" if playoff_mode else "Legacy")
with _pcol:
    render_playoff_toggle()
if playoff_mode:
    st.caption(
        "The all-time playoff record. Every player-postseason from 1973–74 forward, "
        "ranked, compared, weighted by depth-of-run."
    )
else:
    st.caption(
        "The historical record. Every player, every season from 1973–74 to today, "
        "ranked, compared, and put in context."
    )

render_barrett_score_explainer()

# ── Load combined data ─────────────────────────────────────────────────────────
# build_all_seasons_combined is @st.cache_resource (no copy on hit) — must copy before mutating
# Playoff mode uses a much lower min_threshold since playoff GP is 4-28 games.
combined_threshold = 100 if playoff_mode else DEFAULT_MIN_THRESHOLD
with st.spinner(
    f"Loading {'playoff' if playoff_mode else 'historical'} data across all seasons…"
):
    all_df = build_all_seasons_combined(
        min_threshold=combined_threshold, playoffs=playoff_mode,
    ).copy()

if all_df.empty:
    if playoff_mode:
        st.warning(
            "No playoff data on disk yet. Toggle Playoff mode off to view "
            "regular-season legacy, or wait for the playoff seed to populate."
        )
    else:
        st.error("Historical data unavailable. Please try again shortly.")
    st.stop()

# Season sort helper (ascending chronological order for charts)
def _season_year(s: str) -> int:
    return int(s.split("-")[0])

all_df["_season_year"] = all_df["Season"].apply(_season_year)
SEASONS_CHRON = sorted(all_df["Season"].unique(), key=_season_year)

# Pre-compute season-over-season delta per player (used by "The Fall").
_arc = (
    all_df.sort_values("_season_year")[["Player", "Season", "_season_year", "barrett_score"]]
    .copy()
)
_arc["prev_score"] = _arc.groupby("Player")["barrett_score"].shift(1)
_arc["_prev_year"] = _arc.groupby("Player")["_season_year"].shift(1)
_arc["yoy_delta"] = _arc["barrett_score"] - _arc["prev_score"]
# Only a TRUE season-over-season change. Null out gaps — a missed/non-qualifying
# season or a retirement — so "The Fall" doesn't label a post-injury or
# post-baseball return (e.g. Jordan 1994-95 vs his 1992-93 season) a
# single-season collapse, contradicting its own "from one season to the next".
_arc.loc[_arc["_season_year"] != _arc["_prev_year"] + 1, ["prev_score", "yoy_delta"]] = float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Hero cards — always visible at top
# ══════════════════════════════════════════════════════════════════════════════
_goat_row     = all_df.loc[all_df["barrett_score"].idxmax()]

# Most consistent: players with ≥5 qualifying seasons, ranked by avg score.
# GP-weighted so a 17-game cameo doesn't get equal weight as an 82-game peak
# (which would let injury-shortened seasons drag legends below role players).
_career_avg = (
    all_df.groupby("Player")
    .apply(lambda g: pd.Series({
        "avg_score": (g["barrett_score"] * g["GP"]).sum() / g["GP"].sum()
                     if g["GP"].sum() > 0 else g["barrett_score"].mean(),
        "seasons":   g["Season"].nunique(),
    }))
    .reset_index()
)
_consistent = _career_avg[_career_avg["seasons"] >= 5].sort_values("avg_score", ascending=False)
_consistent_name  = _consistent.iloc[0]["Player"]  if not _consistent.empty else "—"
_consistent_avg   = _consistent.iloc[0]["avg_score"] if not _consistent.empty else 0
_consistent_seas  = int(_consistent.iloc[0]["seasons"]) if not _consistent.empty else 0

# Most undervalued single season — only count rows with real salary data
# (pre-1996 has sparse coverage, so $0-salary rows aren't actually "underpaid",
# they just don't have data and would dominate the ranking with fake gaps).
_value_pool = all_df[all_df["salary"] > 0]
if not _value_pool.empty:
    _steal_row = _value_pool.loc[_value_pool["value_diff"].idxmin()]
else:
    _steal_row = all_df.iloc[0]  # fallback; shouldn't happen in practice

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
    <div class="legacy-hero" style="background:var(--tint-bad); border:1px solid var(--orange);">
        <div class="hero-label">Greatest Single Season</div>
        <div class="hero-name">{_goat_row['Player']}</div>
        <div class="hero-sub">{_goat_row['Season']} · {_goat_row['Team']} · Score {_goat_row['barrett_score']:.1f}</div>
    </div>""", unsafe_allow_html=True)
with lh2:
    st.markdown(f"""
    <div class="legacy-hero" style="background:var(--panel-2); border:1px solid var(--blue);">
        <div class="hero-label">Most Consistent Career (≥5 seasons)</div>
        <div class="hero-name">{_consistent_name}</div>
        <div class="hero-sub">Avg {_consistent_avg:.1f} across {_consistent_seas} seasons</div>
    </div>""", unsafe_allow_html=True)
with lh3:
    steal_diff = abs(_steal_row["value_diff"] / 1e6)
    st.markdown(f"""
    <div class="legacy-hero" style="background:var(--tint-good); border:1px solid var(--value-good);">
        <div class="hero-label">Most Underpaid Season Ever</div>
        <div class="hero-name">{_steal_row['Player']}</div>
        <div class="hero-sub">{_steal_row['Season']} · {_steal_row['Team']} · ${steal_diff:.1f}M below market</div>
    </div>""", unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_rank, tab_arc, tab_era, tab_team, tab_long, tab_rec, tab_draft = st.tabs([
    "All-Time Rankings",
    "Career Arc",
    "Era Leaderboards",
    "Team Legacy",
    "Sustained Excellence",
    "Records",
    "Draft Class",
])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: All-Time Rankings
# ─────────────────────────────────────────────────────────────────────────────
with tab_rank:
    st.subheader("Best Single-Season Barrett Scores · All Time")
    st.caption(
        "Every qualifying player-season from 1973–74 to today, ranked by Barrett Score. "
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
        # Only consider rows with real salary data — pre-1996 $0-salary rows
        # would otherwise dominate with artificial value_diff gaps.
        rank_display = rank_display[rank_display["salary"] > 0].sort_values("value_diff", ascending=True)
    else:
        rank_display = rank_display[rank_display["salary"] > 0].sort_values("value_diff", ascending=False)

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

    html_table(
        rank_tbl,
        formatters={
            "Barrett Score": lambda v: f"{v:.2f}", "Season Rank": lambda v: str(int(v)),
            "Salary $M": lambda v: f"${v:.2f}M", "Δ Market $M": lambda v: f"${v:.2f}M",
        },
        styles={"Δ Market $M": _hl_delta},
        aligns={c: "right" for c in ["#", "Barrett Score", "Season Rank", "Salary $M", "Δ Market $M"]},
        numeric={"#", "Barrett Score", "Season Rank", "Salary $M", "Δ Market $M"},
        helps={
            "Season Rank": "Rank within that season's pool.",
            "Δ Market $M": "Actual − Projected within that season. Green = underpaid; red = overpaid.",
        },
        height=min(700, len(rank_tbl) * 38 + 46),
    )
    st.caption(f"**{len(rank_tbl):,}** player-seasons shown")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Career Arc
# ─────────────────────────────────────────────────────────────────────────────
with tab_arc:
    st.subheader("Career Arc")
    st.caption(
        "How did a player's Barrett Score evolve over their career? "
        "Every season they appear in the data is shown, including injury years."
    )

    _arc_options = (
        _career_avg.sort_values("avg_score", ascending=False)["Player"].tolist()
    )
    arc_player = st.selectbox(
        "Select player",
        options=_arc_options,
        index=None,
        placeholder="Search by name, sorted by career avg Barrett Score…",
        key="arc_player_select",
    )

    if arc_player:
        # Load every season this player appeared in — no minutes threshold
        with st.spinner(f"Loading full career for {arc_player}…"):
            _career_raw = fetch_player_career_all_seasons(arc_player, playoffs=playoff_mode)

            if _career_raw.empty:
                arc_df = pd.DataFrame()
            else:
                arc_df = (
                    _career_raw
                    .sort_values("_season_year")
                    [["Season", "_season_year", "barrett_score", "Team", "score_rank", "salary", "total_min", "GP"]]
                    .reset_index(drop=True)
                )

            if arc_df.empty:
                st.info("No seasons found for this player.")
            else:
                peak_idx  = arc_df["barrett_score"].idxmax()
                peak_row  = arc_df.loc[peak_idx]
                avg_score = arc_df["barrett_score"].mean()

                # Summary metrics
                stat_cards([
                    ("Seasons in Data", len(arc_df), "var(--fg-3)"),
                    (f"Peak Score · {peak_row['Season']}", f"{peak_row['barrett_score']:.1f}", "var(--gold)"),
                    ("Career Average", f"{avg_score:.1f}", "var(--accent-teal)"),
                    ("Peak Season Rank", f"#{int(peak_row['score_rank'])}", "var(--accent-red)"),
                ])

                fig_arc = go.Figure()

                # Nodes coloured by absolute Barrett-Score tier (0–10 … 50+)
                seasons   = arc_df["Season"].tolist()
                x_idx     = list(range(len(seasons)))
                y_vals    = arc_df["barrett_score"].tolist()
                vmin, vmax = min(y_vals), max(y_vals)
                dot_colors = [tier_color(v) for v in y_vals]

                # Theme-aware accents (the chart renders server-side, so it can't
                # read CSS vars — pick light/dark values here so the dashed avg
                # line + area don't vanish against a white canvas in light mode).
                _dark = st.session_state.get("theme_dark", False)
                _area_fill = "rgba(255,255,255,0.05)" if _dark else "rgba(20,22,40,0.05)"
                _avg_line  = "rgba(255,255,255,0.30)" if _dark else "rgba(20,22,40,0.28)"
                _avg_font  = "rgba(255,255,255,0.55)" if _dark else "rgba(20,22,40,0.55)"

                # Shaded area under the curve (down to the 0 baseline)
                fig_arc.add_trace(go.Scatter(
                    x=x_idx + x_idx[::-1],
                    y=y_vals + [0] * len(y_vals),
                    fill="toself",
                    fillcolor=_area_fill,
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))

                # Gradient connecting "line" (fades between the dot colours)
                if len(x_idx) >= 2:
                    gx, gy, gc = gradient_points(x_idx, y_vals, dot_colors)
                    fig_arc.add_trace(go.Scatter(
                        x=gx, y=gy, mode="markers",
                        marker=dict(size=4, color=gc, line=dict(width=0)),
                        hoverinfo="skip", showlegend=False,
                    ))

                # Season nodes (carry the hover)
                fig_arc.add_trace(go.Scatter(
                    x=x_idx, y=y_vals,
                    mode="markers",
                    marker=dict(size=9, color=dot_colors,
                                line=dict(color="#14142a", width=1.5)),
                    customdata=seasons,
                    hovertemplate=(
                        "<b>%{customdata}</b><br>"
                        "Score: %{y:.2f}<br>"
                        "<extra></extra>"
                    ),
                    name=arc_player,
                ))

                # Peak annotation
                _peak_x = seasons.index(peak_row["Season"]) if peak_row["Season"] in seasons else 0
                fig_arc.add_annotation(
                    x=_peak_x,
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
                    line_color=_avg_line,
                    annotation_text=f"Career avg {avg_score:.1f}",
                    annotation_position="right",
                    annotation_font_color=_avg_font,
                )

                n_seasons = len(arc_df)
                fig_arc.update_layout(
                    title=f"{arc_player} · Barrett Score by Season",
                    xaxis_title="",
                    yaxis_title="Barrett Score",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.15)",
                    font_color="white",
                    height=420,
                    margin=dict(t=50, b=60, r=120),
                    xaxis=dict(
                        range=[-0.5, n_seasons - 0.5],            # show every season
                        tickmode="array",
                        tickvals=x_idx,
                        ticktext=seasons,                         # preserve chron order
                        gridcolor="rgba(255,255,255,0.07)",
                        tickangle=-40,
                        tickfont=dict(size=11),
                    ),
                    # Baseline 0 → this player's peak (headroom for the label).
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickformat=".1f",
                               range=[min(0, vmin), vmax * 1.10]),
                    showlegend=False,
                )
                st.plotly_chart(theme_fig(fig_arc), use_container_width=True, config={"displayModeBar": False})
                st.caption("Color = Barrett Score tier in 10s "
                           "(red 0–10 · orange 10–20 · amber 20–30 · lime 30–40 · green 40–50 · teal 50+)")

                # Season-by-season table
                arc_tbl = arc_df[["Season", "barrett_score", "Team", "score_rank", "GP", "total_min", "salary"]].copy()
                arc_tbl["salary"] = arc_tbl["salary"] / 1e6
                arc_tbl.columns = ["Season", "Barrett Score", "Team", "Season Rank", "GP", "Total Min", "Salary $M"]
                _arc_peak = arc_tbl["Barrett Score"].max()
                html_table(
                    arc_tbl,
                    formatters={
                        "Barrett Score": lambda v: f"{v:.2f}", "Salary $M": lambda v: f"${v:.2f}M",
                        "Total Min": lambda v: str(int(v)), "GP": lambda v: str(int(v)),
                        "Season Rank": lambda v: str(int(v)),
                    },
                    aligns={c: "right" for c in ["Barrett Score", "Season Rank", "GP", "Total Min", "Salary $M"]},
                    numeric={"Barrett Score", "Season Rank", "GP", "Total Min", "Salary $M"},
                    helps={"Total Min": "Total minutes played that season.", "GP": "Games played."},
                    row_style=lambda rd: ("background:rgba(241,196,15,0.18);font-weight:600"
                                          if rd.get("Barrett Score") == _arc_peak else ""),
                    height=min(640, len(arc_tbl) * 38 + 46),
                )
    else:
        st.info("Type a player name above to see their career arc.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Era Leaderboards
# ─────────────────────────────────────────────────────────────────────────────
with tab_era:
    st.subheader("Era Leaderboards")
    st.caption("Average Barrett Score within each era (GP-weighted), minimum 2 qualifying seasons. "
               "The cards show each era's top 10 at a glance, pick an era below for the full table.")

    ERAS = {
        "Disco Era\n(1973–1979)":       [s for s in SEASONS_CHRON if _season_year(s) <= 1978],
        "Showtime / Bird\n(1979–1991)": [s for s in SEASONS_CHRON if 1978 < _season_year(s) <= 1990],
        "Jordan Era\n(1991–1998)":      [s for s in SEASONS_CHRON if 1990 < _season_year(s) <= 1997],
        "Post-Jordan\n(1998–2005)":     [s for s in SEASONS_CHRON if 1997 < _season_year(s) <= 2004],
        "Pre-Analytics\n(2005–2012)":   [s for s in SEASONS_CHRON if 2004 < _season_year(s) <= 2011],
        "Pace & Space\n(2012–2018)":    [s for s in SEASONS_CHRON if 2011 < _season_year(s) <= 2017],
        "Modern\n(2018–present)":       [s for s in SEASONS_CHRON if _season_year(s) > 2017],
    }
    # Drop empty eras (e.g. seasons not seeded yet)
    ERAS = {k: v for k, v in ERAS.items() if v}

    # Compute each era's leaderboard once — reused by both the cards and the
    # pick-an-era detail table below.
    def _era_leaderboard(era_seasons):
        edf = all_df[all_df["Season"].isin(era_seasons)]
        stats = (
            edf.groupby("Player")
            .apply(lambda g: pd.Series({
                "avg_score":  (g["barrett_score"] * g["GP"]).sum() / g["GP"].sum()
                              if g["GP"].sum() > 0 else g["barrett_score"].mean(),
                "peak_score": g["barrett_score"].max(),
                "seasons":    g["Season"].nunique(),
            }))
            .reset_index()
        )
        stats = stats[stats["seasons"] >= 2].sort_values("avg_score", ascending=False)
        stats["avg_score"]  = stats["avg_score"].round(2)
        stats["peak_score"] = stats["peak_score"].round(2)
        stats.insert(0, "#", range(1, len(stats) + 1))
        stats.columns = ["#", "Player", "Avg Score", "Peak Score", "Seasons"]
        return stats

    _era_boards = {name: _era_leaderboard(seasons) for name, seasons in ERAS.items()}

    def _abbrev(full_name: str) -> str:
        """'Michael Jordan' -> 'M. Jordan' so names fit the narrow cards."""
        parts = full_name.split(" ", 1)
        return f"{parts[0][0]}. {parts[1]}" if len(parts) == 2 and parts[0] else full_name

    # ── Compact cards: top 10 per era, avg score inline (no horizontal scroll) ──
    st.markdown(
        """
        <style>
        .era-card { background:var(--panel-solid); border:1px solid var(--panel-line);
            border-radius:10px; padding:0.7rem 0.55rem 0.6rem; box-shadow:var(--shadow-card);
            height:100%; }
        .era-title { font-weight:700; font-size:0.82rem; line-height:1.15; min-height:2.4rem;
            display:flex; align-items:flex-end; color:var(--fg-1);
            border-bottom:1px solid var(--hairline); padding-bottom:0.4rem; margin-bottom:0.4rem; }
        .era-row { display:flex; align-items:baseline; gap:0.4rem; padding:0.2rem 0.1rem;
            font-size:0.8rem; border-bottom:1px solid var(--hairline-soft); }
        .era-row:last-child { border-bottom:none; }
        .era-rank { color:var(--fg-5); width:1.15rem; text-align:right; font-size:0.72rem;
            flex-shrink:0; font-variant-numeric:tabular-nums; }
        .era-name { color:var(--fg-2); flex:1; white-space:nowrap; overflow:hidden;
            text-overflow:ellipsis; }
        .era-score { color:var(--fg-1); font-weight:700; flex-shrink:0;
            font-variant-numeric:tabular-nums; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    era_cols = st.columns(len(_era_boards), gap="small")
    for col, (era_name, board) in zip(era_cols, _era_boards.items()):
        with col:
            era_label = era_name.replace("\n", " ")
            rows = ""
            for _, r in board.head(10).iterrows():
                rows += (
                    f'<div class="era-row" title="{r["Player"]} · avg {r["Avg Score"]:.1f} · '
                    f'peak {r["Peak Score"]:.1f} · {int(r["Seasons"])} seasons">'
                    f'<span class="era-rank">{int(r["#"])}</span>'
                    f'<span class="era-name">{_abbrev(r["Player"])}</span>'
                    f'<span class="era-score">{r["Avg Score"]:.1f}</span></div>'
                )
            if not rows:
                rows = '<div class="era-row" style="color:var(--fg-5);">No qualifying players</div>'
            st.markdown(
                f'<div class="era-card"><div class="era-title">{era_label}</div>{rows}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Pick an era → full-width detail table (every stat, deeper) ──
    _era_keys = list(_era_boards.keys())
    _default_era = next((k for k in _era_keys if k.startswith("Jordan")), _era_keys[0])
    _pick = st.selectbox(
        "Full leaderboard for an era",
        _era_keys,
        index=_era_keys.index(_default_era),
        format_func=lambda s: s.replace("\n", " "),
    )
    _pick_board = _era_boards[_pick]
    _pick_label = _pick.replace("\n", " ")
    st.markdown(
        f"<div style='margin:0.2rem 0 0.4rem; color:var(--fg-3); font-size:0.85rem;'>"
        f"<b style='color:var(--fg-1);'>{_pick_label}</b> · "
        f"{len(_pick_board)} qualifying players (≥ 2 seasons)</div>",
        unsafe_allow_html=True,
    )
    if _pick_board.empty:
        st.info("No players qualified for this era yet.")
    else:
        html_table(
            _pick_board.head(50),
            formatters={
                "Avg Score": lambda v: f"{v:.2f}", "Peak Score": lambda v: f"{v:.2f}",
                "Seasons": lambda v: str(int(v)),
            },
            aligns={c: "right" for c in ["#", "Avg Score", "Peak Score", "Seasons"]},
            numeric={"#", "Avg Score", "Peak Score", "Seasons"},
            helps={
                "Avg Score": "GP-weighted average Barrett Score across the era.",
                "Peak Score": "Best single-season Barrett Score within the era.",
                "Seasons": "Qualifying seasons played in the era.",
            },
            height=min(640, len(_pick_board.head(50)) * 38 + 46),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Team Legacy — Mount Rushmore
# ─────────────────────────────────────────────────────────────────────────────
with tab_team:
    st.subheader("Team Legacy · Mount Rushmore")
    st.caption(
        "The four best single-season Barrett Scores in franchise history. "
        "Note: team abbreviations reflect the season they were played (e.g. NJN = Brooklyn)."
    )

    teams_list = sorted(all_df["Team"].unique().tolist())
    rush_team  = st.selectbox("Select franchise", teams_list, key="rush_team")

    rush_df = (
        all_df[all_df["Team"] == rush_team]
        .sort_values("barrett_score", ascending=False)
        .drop_duplicates("Player")   # one face per player — the 4 greatest PLAYERS,
        .head(4)                     # not the 4 best seasons of a single star
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
            title=f"{rush_team} · All-Time Top 4 Barrett Score Seasons",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.15)",
            font_color="white",
            height=300,
            margin=dict(t=50, b=20, l=20, r=60),
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Barrett Score", tickformat=".1f"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(theme_fig(fig_rush), use_container_width=True, config={"displayModeBar": False})

        rush_tbl = rush_df[["Player", "Season", "barrett_score", "score_rank", "salary", "value_diff"]].copy()
        rush_tbl["salary"]     /= 1e6
        rush_tbl["value_diff"] /= 1e6
        rush_tbl.columns = ["Player", "Season", "Barrett Score", "League Rank That Season", "Salary $M", "Δ Market $M"]
        rush_tbl.insert(0, "#", range(1, len(rush_tbl) + 1))

        html_table(
            rush_tbl,
            formatters={
                "Barrett Score": lambda v: f"{v:.2f}",
                "League Rank That Season": lambda v: str(int(v)),
                "Salary $M": lambda v: f"${v:.2f}M", "Δ Market $M": lambda v: f"${v:.2f}M",
            },
            styles={"Δ Market $M": _hl_delta},
            aligns={c: "right" for c in ["#", "Barrett Score", "League Rank That Season", "Salary $M", "Δ Market $M"]},
            numeric={"#", "Barrett Score", "League Rank That Season", "Salary $M", "Δ Market $M"},
            helps={"League Rank That Season": "Their score rank among all players that year."},
            height=min(560, len(rush_tbl) * 38 + 46),
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
        html_table(
            full_rush_tbl,
            formatters={"Barrett Score": lambda v: f"{v:.2f}", "League Rank": lambda v: str(int(v))},
            aligns={c: "right" for c in ["#", "Barrett Score", "League Rank"]},
            numeric={"#", "Barrett Score", "League Rank"},
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
        .apply(lambda g: pd.Series({
            "avg_score":   (g["barrett_score"] * g["GP"]).sum() / g["GP"].sum()
                           if g["GP"].sum() > 0 else g["barrett_score"].mean(),
            "peak_score":  g["barrett_score"].max(),
            "min_score":   g["barrett_score"].min(),
            "seasons":     g["Season"].nunique(),
            "peak_season": g.loc[g["barrett_score"].idxmax(), "Season"],
            "teams":       ", ".join(sorted(g["Team"].unique())),
        }))
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
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickformat="d"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickformat=".1f"),
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
    st.plotly_chart(theme_fig(fig_long), use_container_width=True, config={"displayModeBar": False})

    html_table(
        long_df,
        formatters={
            "Avg Score": lambda v: f"{v:.2f}", "Peak Score": lambda v: f"{v:.2f}",
            "Floor Score": lambda v: f"{v:.2f}", "Seasons": lambda v: str(int(v)),
        },
        aligns={c: "right" for c in ["#", "Avg Score", "Peak Score", "Floor Score", "Seasons"]},
        numeric={"#", "Avg Score", "Peak Score", "Floor Score", "Seasons"},
        helps={
            "Avg Score": "Average Barrett Score across all qualifying seasons.",
            "Floor Score": "Their lowest qualifying season score: the floor of their value.",
        },
        height=min(600, len(long_df) * 38 + 46),
    )
    st.caption(
        f"**{len(long_df)}** players with ≥ {min_seas} qualifying seasons shown. "
        "Bubble size = average score."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6: Records (Most Undervalued + The Fall)
# ─────────────────────────────────────────────────────────────────────────────
with tab_rec:
    rec_sub1, rec_sub2 = st.tabs(["Most Undervalued Ever", "The Fall"])

    with rec_sub1:
        st.subheader("Most Undervalued Seasons in NBA History")
        st.caption(
            "Biggest gaps between what a player earned and what their Barrett Score rank deserved. "
            "These are the GMs who got away with something."
        )

        # Only include rows with real salary data — pre-1996 has sparse
        # coverage so $0-salary rows aren't true "underpaid" cases.
        underval = (
            all_df[all_df["salary"] > 0]
            .sort_values("value_diff", ascending=True)
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

        html_table(
            uv_tbl,
            formatters={
                "Barrett Score": lambda v: f"{v:.2f}", "Score Rank": lambda v: str(int(v)),
                "Actual $M": lambda v: f"${v:.2f}M", "Proj. $M": lambda v: f"${v:.2f}M",
                "Δ Market $M": lambda v: f"${v:.2f}M",
            },
            styles={"Δ Market $M": _hl_delta},
            aligns={c: "right" for c in ["#", "Barrett Score", "Score Rank", "Actual $M", "Proj. $M", "Δ Market $M"]},
            numeric={"#", "Barrett Score", "Score Rank", "Actual $M", "Proj. $M", "Δ Market $M"},
            helps={"Δ Market $M": "Negative = underpaid. The more negative, the bigger the steal."},
            height=500,
        )

    with rec_sub2:
        st.subheader("The Fall · Biggest Single-Season Score Drops")
        st.caption(
            "Players whose Barrett Score fell the most from one season to the next, "
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

        html_table(
            fall_tbl,
            formatters={
                "Score That Year": lambda v: f"{v:.2f}", "Score Prev. Season": lambda v: f"{v:.2f}",
                "Δ Score": lambda v: f"{v:.2f}",
            },
            styles={"Δ Score": _hl_fall},
            aligns={c: "right" for c in ["#", "Score That Year", "Score Prev. Season", "Δ Score"]},
            numeric={"#", "Score That Year", "Score Prev. Season", "Δ Score"},
            helps={"Δ Score": "Negative = score dropped vs prior season. Larger drop = bigger fall."},
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
                    xaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickangle=-30, type="category"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickformat=".1f"),
                )
                st.plotly_chart(theme_fig(fig_fall), use_container_width=True, config={"displayModeBar": False})


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
        # Available draft years overlapping our season data (now 1973+).
        # Earliest drafts (Kareem '69, Walton '74) still have partial career
        # coverage since we start at 1973-74.
        avail_years = sorted(
            [y for y in draft_df["draft_year"].unique() if 1969 <= y <= 2024],
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
            # Drop draft_info's own "Player" column before merging to avoid Player_x / Player_y
            _draft_info_merge = draft_info.drop(columns=["Player"], errors="ignore")
            class_summary = class_peaks.merge(_draft_info_merge, on="player_norm", how="left")
            class_summary = class_summary.sort_values("peak_score", ascending=False).reset_index(drop=True)

            mc1, mc2 = st.columns([1, 2])
            with mc1:
                st.markdown(f"**{draft_year_sel} Class · {len(class_summary)} players in our data**")
                sum_tbl = class_summary[["Player", "OVERALL_PICK", "peak_score", "seasons"]].copy()
                sum_tbl.columns = ["Player", "Pick #", "Peak Score", "Seasons"]
                sum_tbl["Peak Score"] = sum_tbl["Peak Score"].round(2)
                html_table(
                    sum_tbl,
                    formatters={
                        "Pick #": lambda v: str(int(v)), "Peak Score": lambda v: f"{v:.2f}",
                        "Seasons": lambda v: str(int(v)),
                    },
                    aligns={c: "right" for c in ["Pick #", "Peak Score", "Seasons"]},
                    numeric={"Pick #", "Peak Score", "Seasons"},
                    height=min(500, len(sum_tbl) * 38 + 46),
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
                    title=f"{draft_year_sel} Draft Class · Career Arcs (Top 10 by peak)",
                    height=460,
                )
                fig_class.update_traces(line_width=2, marker_size=6)
                fig_class.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.15)",
                    font_color="white",
                    legend=dict(orientation="v", x=1.01, y=1, font_size=10),
                    margin=dict(t=50, b=20, r=140),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickangle=-30, type="category"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickformat=".1f"),
                )
                st.plotly_chart(theme_fig(fig_class), use_container_width=True, config={"displayModeBar": False})


from utils import render_footer
render_footer()
