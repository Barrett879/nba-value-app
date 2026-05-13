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
    render_nav, render_playoff_toggle, _bootstrap_warm,
    PRE_1990_SALARY_NOTE,
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

# Playoff mode lives in the top nav bar (sticky across pages via session_state)
playoff_mode = bool(st.session_state.get("playoff_mode", False))
if playoff_mode:
    st.title("Search Player — Playoff Mode")
    st.caption("Career arcs and per-season stats from postseason data only. Salaries reflect regular-season contracts.")
else:
    st.title("Search Player")
    st.caption("Find any player who's appeared in the league — career arcs, season-by-season stats, peak years. Add up to 10 players to compare careers head-to-head.")

# ── Search box ─────────────────────────────────────────────────────────────────
all_names = get_all_player_names()
if not all_names:
    st.error("Player database not yet loaded. Try again in a moment.")
    st.stop()

# Pre-select if we arrived here from the home-page search bar
_default = []
_handed_off = st.session_state.pop("search_player", None)
if _handed_off and _handed_off in all_names:
    _default = [_handed_off]

selected = st.multiselect(
    "Type a player name…  (add up to 10 to compare)",
    options=all_names,
    default=_default,
    max_selections=10,
    placeholder="Try LeBron James, Michael Jordan, Nikola Jokić…",
    key="player_search_multiselect",
)

if not selected:
    st.info(
        f"**{len(all_names):,} players** indexed across "
        f"every season we have data for. Names are sorted by career-average "
        "Barrett Score, so the legends rise to the top. "
        "Add a second player to overlay career arcs side by side."
    )
    st.stop()


# ── Era-adjustment toggle ────────────────────────────────────────────────────
# Pace-adjusted is now the canonical Barrett Score across the whole site
# (Rankings, Legacy, Trades, all-time lists). Toggle off to see the original
# unadjusted scoring for diagnostic / nostalgic purposes.
era_mode = st.radio(
    "Score mode",
    options=["Era-Adjusted (default)", "Raw / un-adjusted"],
    horizontal=True,
    index=0,
    help=(
        "Era-Adjusted scales volume stats (PTS, AST, REB, BLK, STL, TOV, PF) "
        "by pace, normalizing high-pace eras down and dead-ball eras up. "
        "This is the canonical Barrett Score everywhere on the site. "
        "Toggle to Raw to see the un-adjusted version for that season."
    ),
    key="search_era_mode",
)
SCORE_COL   = "Barrett Score" if era_mode.startswith("Era") else "Barrett (Raw)"
SCORE_LABEL = "Barrett Score" if era_mode.startswith("Era") else "Raw Barrett"


# ── Helper: load + cache one player's full career ─────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _load_career(name: str, playoffs: bool = False) -> pd.DataFrame:
    """Wraps fetch_player_full_career so each player is cached independently
    per (name, mode). Playoff and regular-season careers each get their
    own cache entry."""
    return fetch_player_full_career(name, playoffs=playoffs)


# ── Color palette for multi-player overlays ──────────────────────────────────
# 10 distinct colors so up to 10 selected players each get their own line.
# Hand-picked for legibility on a dark background; alternates warm/cool tones.
_PALETTE = [
    "#f1c40f",  # gold
    "#e74c3c",  # red
    "#3498db",  # blue
    "#2ecc71",  # green
    "#9b59b6",  # purple
    "#e67e22",  # orange
    "#1abc9c",  # teal
    "#ec407a",  # pink
    "#7ec8e8",  # sky
    "#c8d75e",  # lime
]


# ── Career-average helper (games-weighted, like real stat sites) ─────────────
# Simple per-season .mean() weights every season equally — so a 17-game
# 1994-95 MJ cameo contributes the same as an 82-game peak season, dragging
# career PPG below the canonical 30.1. GP-weighted means match BBRef numbers.
def _gp_weighted(career: pd.DataFrame, col: str) -> float:
    gp = career["GP"]
    total_gp = gp.sum()
    if total_gp <= 0:
        return float(career[col].mean()) if len(career) else 0.0
    return float((career[col] * gp).sum() / total_gp)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-PLAYER VIEW — full career detail
# ══════════════════════════════════════════════════════════════════════════════
if len(selected) == 1:
    player_name = selected[0]
    with st.spinner(f"Loading {player_name}'s career…"):
        career = _load_career(player_name, playoffs=playoff_mode)

    if career.empty:
        st.warning(f"No career data found for {player_name}.")
        st.stop()

    # ── Header summary ─────────────────────────────────────────────────────────
    n_seasons   = len(career)
    first_yr    = career["Season"].iloc[0].split("-")[0]
    last_yr_end = career["Season"].iloc[-1].split("-")[1]
    career_yrs  = f"{first_yr} – 20{last_yr_end}" if int(last_yr_end) < 50 else f"{first_yr} – 19{last_yr_end}"
    teams       = list(dict.fromkeys(career["Team"]))   # preserve order, dedup

    # Pre-1990 salary disclaimer if any of this player's seasons fall in that era
    if any(int(s.split("-")[0]) < 1990 for s in career["Season"]):
        st.warning(PRE_1990_SALARY_NOTE, icon="📜")

    best_season_idx = career[SCORE_COL].idxmax()
    best_season     = career.loc[best_season_idx]

    career_avg_score = _gp_weighted(career, SCORE_COL)
    career_avg_pts   = _gp_weighted(career, "PTS")
    career_avg_ast   = _gp_weighted(career, "AST")
    career_avg_reb   = _gp_weighted(career, "REB")
    total_games      = int(career["GP"].sum())

    st.markdown(f"### {player_name}")
    st.caption(f"**{career_yrs}** · {n_seasons} seasons · {total_games:,} games · "
               f"Teams: {' → '.join(teams)}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Career Avg {SCORE_LABEL}", f"{career_avg_score:.1f}")
    c2.metric("Career PPG",                f"{career_avg_pts:.1f}")
    c3.metric("Career APG",                f"{career_avg_ast:.1f}")
    c4.metric("Career RPG",                f"{career_avg_reb:.1f}")
    c5.metric("Peak Season",
              f"{best_season[SCORE_COL]:.1f}",
              f"{best_season['Season']}")

    st.divider()

    # ── Career arc chart ───────────────────────────────────────────────────────
    st.subheader(f"Career arc — {SCORE_LABEL} by season")
    fig = go.Figure()

    # Color points by Barrett Score (red→gold→green)
    def _val_color(v, vmin, vmax):
        if vmax <= vmin: return "#f1c40f"
        t = (v - vmin) / (vmax - vmin)
        if t < 0.5:
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

    vmin, vmax = career[SCORE_COL].min(), career[SCORE_COL].max()
    dot_colors = [_val_color(v, vmin, vmax) for v in career[SCORE_COL]]

    fig.add_trace(go.Scatter(
        x=career["Season"], y=career[SCORE_COL],
        mode="lines+markers",
        line=dict(color="rgba(241, 196, 15, 0.6)", width=2.5),
        marker=dict(size=10, color=dot_colors,
                    line=dict(color="#14142a", width=1.5)),
        text=career["Team"],
        customdata=career[["PTS", "AST", "REB", "Score Rank", "Total Players"]].values,
        hovertemplate=(
            "<b>%{x}</b> · %{text}<br>"
            f"{SCORE_LABEL}: " "%{y:.2f}<br>"
            "PTS %{customdata[0]:.1f} · AST %{customdata[1]:.1f} · REB %{customdata[2]:.1f}<br>"
            "Rank %{customdata[3]} / %{customdata[4]} that season"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # Mark peak season
    fig.add_trace(go.Scatter(
        x=[best_season["Season"]], y=[best_season[SCORE_COL]],
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
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title=SCORE_LABEL,
                   tickformat=".1f"),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("★ = peak career season · dot color encodes the score (red = lowest, "
               "gold = mid, green = highest of this player's career)")

    st.divider()

    # ── Season-by-season table ─────────────────────────────────────────────────
    st.subheader("Season by season")

    tbl = career[[
        "Season", "Team", "GP", "MPG", "PTS", "AST", "REB", "STL", "BLK", "TOV",
        "TS%", "Barrett Score", "Barrett (Raw)", "Score Rank", "Total Players", "Salary",
    ]].copy()
    tbl["Salary $M"] = (tbl["Salary"] / 1_000_000).round(2)
    tbl = tbl.drop(columns=["Salary"])
    tbl["Rank"] = tbl.apply(lambda r: f"{int(r['Score Rank'])}/{int(r['Total Players'])}", axis=1)
    tbl = tbl.drop(columns=["Score Rank", "Total Players"])

    # Highlight peak season row (using whichever score column is active)
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
            "TS%": "{:.1f}%", "Barrett Score": "{:.2f}",
            "Barrett (Raw)": "{:.2f}", "Salary $M": "${:.2f}M",
        })
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(700, max(120, len(tbl) * 35 + 40)),
        column_config={
            "GP":              st.column_config.NumberColumn(format="%d", help="Games played that season."),
            "Salary $M":       st.column_config.TextColumn("Salary",     help="Salary that season ($M). Some pre-2000 rookie scale and minimum contracts may show $0."),
            "Barrett Score":   st.column_config.NumberColumn(format="%.2f", help="Era-adjusted via pace — the canonical Barrett Score across the site."),
            "Barrett (Raw)":   st.column_config.NumberColumn(format="%.2f", help="Un-adjusted version for that season — preserved for reference and the Score-mode toggle."),
            "Rank":            st.column_config.TextColumn(help="Score rank that season — based on the canonical (era-adjusted) Barrett Score."),
            "TS%":             st.column_config.TextColumn("TS%", help="True Shooting %."),
        },
    )
    st.caption(f"Highlighted row = peak season ({best_season['Season']}). "
               "Use the Legacy page for cross-player comparisons.")


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON VIEW — 2+ players overlaid
# ══════════════════════════════════════════════════════════════════════════════
else:
    careers: dict[str, pd.DataFrame] = {}
    with st.spinner(f"Loading {len(selected)} careers…"):
        for name in selected:
            c = _load_career(name, playoffs=playoff_mode)
            if not c.empty:
                careers[name] = c

    if not careers:
        st.warning("No career data found for any of the selected players.")
        st.stop()

    # Drop any selected player whose career came back empty (rare)
    valid_selected = [n for n in selected if n in careers]
    if len(valid_selected) < len(selected):
        missing = [n for n in selected if n not in careers]
        st.caption(f"⚠️ No data found for: {', '.join(missing)}")

    # ── Mode selector ─────────────────────────────────────────────────────────
    st.markdown(
        f"### Comparing {len(valid_selected)} players: "
        + " · ".join(f"<span style='color:{_PALETTE[i]}'>{n}</span>"
                     for i, n in enumerate(valid_selected)),
        unsafe_allow_html=True,
    )

    align_mode = st.radio(
        "Align by",
        options=["Career year (Year 1 = rookie season)", "Actual season (1973-74 → today)"],
        horizontal=True,
        help="Career year aligns peaks for direct comparison. Actual season shows era context.",
        key="search_align_mode",
    )
    use_career_year = align_mode.startswith("Career year")

    st.divider()

    # ── Summary metrics — one row per player ──────────────────────────────────
    st.markdown("#### Career averages")
    rows = []
    for name in valid_selected:
        c = careers[name]
        peak_idx = c[SCORE_COL].idxmax()
        peak    = c.loc[peak_idx]
        n_seasons = len(c)
        first_yr  = c["Season"].iloc[0].split("-")[0]
        last_yr_e = c["Season"].iloc[-1].split("-")[1]
        career_yrs = (
            f"{first_yr} – 20{last_yr_e}" if int(last_yr_e) < 50
            else f"{first_yr} – 19{last_yr_e}"
        )
        rows.append({
            "Player":          name,
            "Career":          career_yrs,
            "Seasons":         n_seasons,
            "Games":           int(c["GP"].sum()),
            f"Avg {SCORE_LABEL}":  _gp_weighted(c, SCORE_COL),
            f"Peak {SCORE_LABEL}": float(peak[SCORE_COL]),
            "Peak Season":     peak["Season"],
            "PPG":             _gp_weighted(c, "PTS"),
            "APG":             _gp_weighted(c, "AST"),
            "RPG":             _gp_weighted(c, "REB"),
        })
    summary = pd.DataFrame(rows)
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            f"Avg {SCORE_LABEL}":  st.column_config.NumberColumn(format="%.2f"),
            f"Peak {SCORE_LABEL}": st.column_config.NumberColumn(format="%.2f"),
            "PPG":          st.column_config.NumberColumn(format="%.1f"),
            "APG":          st.column_config.NumberColumn(format="%.1f"),
            "RPG":          st.column_config.NumberColumn(format="%.1f"),
            "Games":        st.column_config.NumberColumn(format="%d"),
        },
    )

    st.divider()

    # ── Overlaid career arc chart ─────────────────────────────────────────────
    st.subheader(f"{SCORE_LABEL} — career arcs overlaid")

    fig = go.Figure()
    all_x_vals: list = []

    for i, name in enumerate(valid_selected):
        c = careers[name].copy()
        color = _PALETTE[i % len(_PALETTE)]

        if use_career_year:
            c["x"] = list(range(1, len(c) + 1))
        else:
            c["x"] = c["Season"]

        all_x_vals.extend(c["x"].tolist())

        fig.add_trace(go.Scatter(
            x=c["x"], y=c[SCORE_COL],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2.6),
            marker=dict(size=8, color=color, line=dict(color="#14142a", width=1.2)),
            text=c["Season"] + " · " + c["Team"],
            customdata=c[["PTS", "AST", "REB", "Score Rank", "Total Players"]].values,
            hovertemplate=(
                f"<b>{name}</b><br>"
                "%{text}<br>"
                f"{SCORE_LABEL}: " "%{y:.2f}<br>"
                "PTS %{customdata[0]:.1f} · AST %{customdata[1]:.1f} · REB %{customdata[2]:.1f}<br>"
                "Rank %{customdata[3]} / %{customdata[4]} that season"
                "<extra></extra>"
            ),
        ))

        # Star at each player's peak
        peak_idx = c[SCORE_COL].idxmax()
        peak     = c.loc[peak_idx]
        fig.add_trace(go.Scatter(
            x=[peak["x"]], y=[peak[SCORE_COL]],
            mode="markers",
            marker=dict(size=16, symbol="star", color=color,
                        line=dict(width=1.5, color="white")),
            hoverinfo="skip",
            showlegend=False,
        ))

    if use_career_year:
        x_title = "Career year (Year 1 = first NBA season in our data)"
        x_type  = "linear"
        # Lock the x-axis to 1 → longest career across the selected players
        # (shorter careers' lines just stop earlier rather than each chart
        # auto-fitting to its own range — keeps the canvas constant).
        max_career_year = max(
            (len(c) for c in careers.values() if not c.empty),
            default=1,
        )
        x_kwargs = dict(
            dtick=1 if max_career_year <= 25 else 2,
            range=[0.5, max_career_year + 0.5],
        )
    else:
        x_title = ""
        x_type  = "category"
        # Sort seasons chronologically across all players
        sorted_seasons = sorted(
            set(all_x_vals),
            key=lambda s: int(s.split("-")[0]) if isinstance(s, str) else s,
        )
        x_kwargs = dict(categoryorder="array", categoryarray=sorted_seasons)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
        font_color="white",
        height=460,
        margin=dict(l=50, r=30, t=20, b=70),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=x_title,
                   type=x_type, **x_kwargs),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title=SCORE_LABEL,
                   tickformat=".1f"),
        hovermode="closest",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18, yanchor="top",
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "★ = each player's peak season. "
        "Hover any point for the actual season + raw stats."
    )

    st.divider()

    # ── Side-by-side season tables ────────────────────────────────────────────
    st.subheader("Season by season")
    cols = st.columns(len(valid_selected))
    for col, name in zip(cols, valid_selected):
        with col:
            color = _PALETTE[valid_selected.index(name) % len(_PALETTE)]
            st.markdown(
                f"<div style='border-left:3px solid {color}; padding-left:0.5rem; "
                f"font-weight:700; margin-bottom:0.4rem;'>{name}</div>",
                unsafe_allow_html=True,
            )
            c = careers[name]
            tbl = c[["Season", "Team", "GP", "PTS", "AST", "REB", SCORE_COL]].copy()
            peak_season = c.loc[c[SCORE_COL].idxmax(), "Season"]

            def _hl(row, peak_s=peak_season):
                if row["Season"] == peak_s:
                    return ["background-color: rgba(241, 196, 15, 0.18); "
                            "font-weight: 600"] * len(row)
                return [""] * len(row)

            styled = (
                tbl.style
                .apply(_hl, axis=1)
                .format({
                    "PTS": "{:.1f}", "AST": "{:.1f}", "REB": "{:.1f}",
                    SCORE_COL: "{:.2f}",
                })
            )
            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                height=min(600, max(120, len(tbl) * 35 + 40)),
                column_config={
                    "GP":     st.column_config.NumberColumn(format="%d"),
                    SCORE_COL: st.column_config.NumberColumn(format="%.2f"),
                },
            )
    st.caption("Highlighted row = each player's peak season.")
