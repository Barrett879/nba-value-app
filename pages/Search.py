import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from utils import (
    COMMON_CSS,
    normalize,
    get_all_player_names, fetch_player_full_career,
    fetch_season_component_distribution,
    render_nav, render_playoff_toggle, render_barrett_score_explainer, _bootstrap_warm,
    PRE_1990_SALARY_NOTE,
)
from urllib.parse import quote

st.set_page_config(page_title="Search Player", layout="wide")
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
    st.title("Search Player (Playoff Mode)")
    st.caption("Career arcs and per-season stats from postseason data only. Salaries reflect regular-season contracts.")
else:
    st.title("Search Player")
    st.caption("Find any player who's appeared in the league: career arcs, season-by-season stats, peak years. Add up to 10 players to compare careers head-to-head.")

render_barrett_score_explainer()

# ── Search box ─────────────────────────────────────────────────────────────────
all_names = get_all_player_names()
if not all_names:
    st.error("Player database not yet loaded. Try again in a moment.")
    st.stop()

# Pre-select if we arrived here from one of three sources, in priority order:
#   1. ?player=Name (or ?player=A&player=B...) query-string deep link — used
#      for sharing direct URLs (e.g. /Search?player=Jok%C4%87)
#   2. session_state hand-off from the home-page search bar
#   3. nothing — empty state
_default: list[str] = []

_qs_players = st.query_params.get_all("player") if hasattr(st.query_params, "get_all") else (
    [st.query_params["player"]] if "player" in st.query_params else []
)

# Case-insensitive resolve so /Search?player=jokic still finds "Nikola Jokić".
def _resolve_qs_name(qs_name: str) -> str | None:
    if qs_name in all_names:
        return qs_name
    target = normalize(qs_name)
    for full in all_names:
        if normalize(full) == target:
            return full
    return None

for _qsp in _qs_players[:10]:
    _resolved = _resolve_qs_name(_qsp)
    if _resolved and _resolved not in _default:
        _default.append(_resolved)

# Fall back to home-page hand-off if no query string match.
if not _default:
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

# Mirror the current selection back into the URL so the address bar
# always reflects what's on screen. Triggers a no-op rerun if a user
# pasted a URL, then immediately changed the selection — that's fine.
_current_qs = st.query_params.get_all("player") if hasattr(st.query_params, "get_all") else (
    [st.query_params["player"]] if "player" in st.query_params else []
)
if list(selected) != list(_current_qs):
    if selected:
        st.query_params["player"] = selected
    elif "player" in st.query_params:
        del st.query_params["player"]

if not selected:
    st.info(
        f"**{len(all_names):,} players** indexed across "
        f"every season we have data for. Names are sorted by career-average "
        "Barrett Score, so the legends rise to the top. "
        "Add a second player to overlay career arcs side by side."
    )
    st.stop()


# ── Shareable link ───────────────────────────────────────────────────────────
# Rendered via components.html (iframe) instead of st.markdown so the inline
# onclick handler isn't stripped by Streamlit's HTML sanitizer. We also build
# the absolute URL inside the iframe's JS from window.parent.location.origin
# so it works on both the deployed Render URL and on localhost.
_share_path = "/Search?" + "&".join(f"player={quote(p)}" for p in selected)

# Two fallbacks for clipboard access in case navigator.clipboard isn't
# available (Safari private mode, older browsers, or iframe permission
# quirks): try navigator.clipboard first, then execCommand('copy') on a
# temporary textarea. Either way the user sees the "✓ Copied" flash.
_share_widget = f"""
<div style="display:flex; align-items:center; gap:0.6rem;
            margin: 0; flex-wrap:wrap; font-family: 'Source Sans Pro',
            -apple-system, BlinkMacSystemFont, sans-serif;">
  <span style="font-size:0.78rem; color:rgba(250,250,250,0.55);
               letter-spacing:0.02em; text-transform:uppercase;
               font-weight:600;">Share this view</span>
  <code id="share-url" style="background:rgba(255,255,255,0.04);
                              border:1px solid rgba(255,255,255,0.08);
                              border-radius:6px; padding:0.25rem 0.6rem;
                              font-size:0.82rem; color:#cdcdd5;
                              max-width:520px; overflow:hidden;
                              text-overflow:ellipsis; white-space:nowrap;">
    {_share_path}
  </code>
  <button id="share-btn" type="button"
          style="background:#e63946; color:white; border:none;
                 border-radius:6px; padding:0.3rem 0.85rem; font-size:0.8rem;
                 font-weight:600; cursor:pointer; transition:opacity 0.15s;
                 font-family:inherit;">
    Copy link
  </button>
</div>
<script>
  (function() {{
    const sharePath = {repr(_share_path)};
    const btn = document.getElementById("share-btn");
    if (!btn) return;

    // Resolve the absolute URL using the *parent* window's origin so the
    // link points at hoopsvalue.onrender.com (or wherever the app lives),
    // not the iframe's component-asset URL.
    function getFullUrl() {{
      try {{
        return window.parent.location.origin + sharePath;
      }} catch (e) {{
        return window.location.origin + sharePath;
      }}
    }}

    function flashCopied() {{
      btn.innerText = "✓ Copied";
      btn.style.background = "#2ecc71";
      setTimeout(() => {{
        btn.innerText = "Copy link";
        btn.style.background = "#e63946";
      }}, 1500);
    }}

    function fallbackCopy(text) {{
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      try {{ document.execCommand("copy"); flashCopied(); }}
      catch (e) {{ console.error(e); }}
      document.body.removeChild(ta);
    }}

    btn.addEventListener("click", function() {{
      const url = getFullUrl();
      if (navigator.clipboard && window.isSecureContext) {{
        navigator.clipboard.writeText(url).then(flashCopied)
          .catch(() => fallbackCopy(url));
      }} else {{
        fallbackCopy(url);
      }}
    }});

    btn.addEventListener("mouseover", () => btn.style.opacity = "0.85");
    btn.addEventListener("mouseout",  () => btn.style.opacity = "1");
  }})();
</script>
"""
components.html(_share_widget, height=58)


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
    st.subheader(f"Career arc · {SCORE_LABEL} by season")
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

    # ── Score Breakdown — what's driving the Barrett Score this season? ─────
    # Decomposes the score into its 6 inputs (scoring / playmaking / rebounding
    # / defense / efficiency / availability) so users can see *why* a player
    # ranks where they do, rather than just trusting the final number.
    st.subheader("Score Breakdown")

    _peak_season = best_season["Season"]
    _bd_seasons = career["Season"].tolist()
    _bd_default_idx = _bd_seasons.index(_peak_season) if _peak_season in _bd_seasons else len(_bd_seasons) - 1
    _bd_col_l, _bd_col_r = st.columns([1, 3])
    with _bd_col_l:
        _bd_season = st.selectbox(
            "Season",
            _bd_seasons,
            index=_bd_default_idx,
            key=f"breakdown_season_{player_name}",
            help="What was driving the Barrett Score this season? Defaults to the peak.",
        )

    _bd_row = career[career["Season"] == _bd_season].iloc[0]

    # Pull the full league-wide distribution for this season so we can rank
    # this player's component values against everyone else — a 95th-percentile
    # scorer in 2007-08 means "Kobe was a top-5% scorer that year."
    _dist = fetch_season_component_distribution(_bd_season, playoffs=playoff_mode)

    _components = [
        ("Scoring",      "#e63946", "PTS"),
        ("Playmaking",   "#f39c12", "AST × 1.5 − TOV / 1.5"),
        ("Rebounding",   "#16d4c1", "OREB / 2 + DREB / 3"),
        ("Defense",      "#3498db", "BLK / 2 + STL / 1.5 − PF / 3 + D-LEBRON × 2"),
        ("Efficiency",   "#9b59b6", "TS% vs league avg"),
        ("Availability", "#2ecc71", "GP × MPG vs 82-game cap"),
    ]

    if _dist.empty:
        st.info(
            f"No league-wide component data on disk for {_bd_season} yet — "
            "the percentile bars need that. Try a more recent season."
        )
    else:
        # Resolve this player's row in the distribution by PLAYER_ID first
        # (most reliable) and fall back to normalized name.
        _this_row = _dist[_dist["Player"].apply(normalize) == normalize(player_name)]
        if _this_row.empty:
            st.info("This player isn't in the season's ranking pool (minute-threshold).")
        else:
            _this = _this_row.iloc[0]
            _rows = []
            for label, color, formula in _components:
                vals = _dist[label].astype(float).values
                my_val = float(_this[label])
                # Percentile rank: % of league at or below this value. Higher = better.
                # Uses average rank for ties so duplicates don't artificially deflate.
                pct = (vals < my_val).sum() + 0.5 * (vals == my_val).sum()
                pct = (pct / len(vals)) * 100 if len(vals) else 0
                _rows.append({
                    "label":   label,
                    "color":   color,
                    "pct":     pct,
                    "value":   my_val,
                    "formula": formula,
                })
            _rows.sort(key=lambda r: r["pct"], reverse=True)

            def _fmt_value(label: str, value: float) -> str:
                if label == "Availability":
                    return f"{value * 100:.0f}%"
                if label == "Efficiency":
                    return f"{value:+.1f} pts"
                return f"{value:.1f}"

            _labels  = [r["label"] for r in _rows]
            _pcts    = [r["pct"] for r in _rows]
            _colors  = [r["color"] for r in _rows]
            _values  = [r["value"] for r in _rows]
            _formulas = [r["formula"] for r in _rows]
            _texts   = [
                f"<b>{r['pct']:.0f}th</b> · {_fmt_value(r['label'], r['value'])}"
                for r in _rows
            ]

            _bd_fig = go.Figure()
            # Background track for each row so 100% is visually anchored.
            _bd_fig.add_trace(go.Bar(
                x=[100] * len(_labels),
                y=_labels,
                orientation="h",
                marker=dict(color="rgba(255,255,255,0.04)",
                            line=dict(color="rgba(255,255,255,0.06)", width=1)),
                hoverinfo="skip",
                showlegend=False,
            ))
            # Foreground = actual percentile bar.
            _bd_fig.add_trace(go.Bar(
                x=_pcts,
                y=_labels,
                orientation="h",
                marker=dict(color=_colors, line=dict(color="rgba(0,0,0,0)", width=0)),
                text=_texts,
                textposition="outside",
                textfont=dict(color="#fff", size=13),
                customdata=list(zip(_values, _formulas)),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Percentile: <b>%{x:.0f}th</b><br>"
                    "Value: %{customdata[0]:.2f}<br>"
                    "<i>%{customdata[1]}</i><extra></extra>"
                ),
                showlegend=False,
            ))
            _bd_fig.update_layout(
                height=320,
                barmode="overlay",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cdcdd5"),
                margin=dict(l=10, r=80, t=10, b=30),
                xaxis=dict(
                    range=[0, 118],
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=["0", "25th", "50th", "75th", "100th"],
                    gridcolor="rgba(255,255,255,0.06)",
                    zerolinecolor="rgba(255,255,255,0.15)",
                    title="Percentile vs. all qualifying players this season",
                ),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
            )
            st.plotly_chart(_bd_fig, use_container_width=True,
                            config={"displayModeBar": False})

            # Summary strip: Base × Avail = Barrett Score for transparency.
            _bs = float(_bd_row["Barrett Score"])
            _av = float(_bd_row.get("Avail", 1.0))
            _base_score = _bs / _av if _av > 0 else _bs
            _avail_color = ("#2ecc71" if _av >= 0.85
                            else ("#f39c12" if _av >= 0.7 else "#e63946"))

            _eq_html = f"""
            <div style="display:flex; flex-wrap:wrap; align-items:center; gap:0.5rem;
                        margin-top:0.5rem; padding:0.85rem 1rem;
                        background:rgba(255,255,255,0.03);
                        border:1px solid rgba(255,255,255,0.08); border-radius:8px;">
                <span style="color:#cdcdd5; font-size:0.9rem;">
                    <b style="color:#fff; font-size:1.05rem;">{_base_score:.1f}</b>
                    <span style="color:#888; font-size:0.78rem; text-transform:uppercase;
                                 letter-spacing:0.04em; margin-left:0.3rem;">Base Score</span>
                </span>
                <span style="color:#666;">×</span>
                <span style="color:{_avail_color}; font-size:0.9rem;">
                    <b style="font-size:1.05rem;">{_av * 100:.0f}%</b>
                    <span style="color:#888; font-size:0.78rem; text-transform:uppercase;
                                 letter-spacing:0.04em; margin-left:0.3rem;">Availability</span>
                </span>
                <span style="color:#666;">=</span>
                <span style="color:#fff; font-size:0.9rem;">
                    <b style="font-size:1.15rem;">{_bs:.1f}</b>
                    <span style="color:#888; font-size:0.78rem; text-transform:uppercase;
                                 letter-spacing:0.04em; margin-left:0.3rem;">Barrett Score</span>
                </span>
                <span style="margin-left:auto; color:#888; font-size:0.75rem;">
                    {_bd_season} · {int(_bd_row['GP'])} GP · {_bd_row['MPG']:.1f} MPG
                </span>
            </div>
            """
            st.markdown(_eq_html, unsafe_allow_html=True)
            st.caption(
                "Each bar shows where this player ranked among all qualifying players "
                f"in {_bd_season} — 90th percentile = better than 90% of the league. "
                "Hover any bar for the underlying formula and raw value."
            )

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
            "Barrett Score":   st.column_config.NumberColumn(format="%.2f", help="Era-adjusted via pace. The canonical Barrett Score across the site."),
            "Barrett (Raw)":   st.column_config.NumberColumn(format="%.2f", help="Un-adjusted version for that season, preserved for reference and the Score-mode toggle."),
            "Rank":            st.column_config.TextColumn(help="Score rank that season, based on the canonical (era-adjusted) Barrett Score."),
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
    st.subheader(f"{SCORE_LABEL} · career arcs overlaid")

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
