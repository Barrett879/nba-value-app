import html
import math
import sys
from urllib.parse import quote
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    COMMON_CSS, SEASONS, DEFAULT_MIN_THRESHOLD, SEASON_GAMES_LOOKUP, season_is_live,
    live_min_threshold,
    normalize, season_to_espn_year,
    build_ranked_projected, build_all_seasons_combined,
    fetch_bref_positions, fetch_next_year_contracts, fetch_rookie_scale_players,
    NameIndex, _dc_path, team_bar_color,
    fetch_league_stats, fetch_bref_player_stats,
    fetch_dlebron, fetch_defense_blend, fetch_defense_blend_meta, fetch_career_trend, fetch_player_season_splits,
    fetch_monthly_scores, build_splits_data,
    _fmt_salary, fmt_next_contract,
    html_table,
    team_cell,
    render_nav, render_page_chrome, render_rail, face_img, TEAM_HEX,
    theme_fig, render_playoff_toggle, render_barrett_score_explainer, _bootstrap_warm,
    PRE_1990_SALARY_NOTE,
)
import threading

st.set_page_config(page_title="Rankings", page_icon="static/favicon.svg", layout="wide")

render_page_chrome()
render_nav("Current Rankings")

_pf_on = bool(st.session_state.get("playoff_mode", False))
_tcol, _pcol = st.columns([5, 1], vertical_alignment="center")
with _tcol:
    st.title("Playoff Rankings" if _pf_on else "NBA Contract Value Rankings")
with _pcol:
    render_playoff_toggle()
if _pf_on:
    st.caption("All scores computed from playoff games only. Salaries reflect the regular-season contract for that year.")
else:
    st.caption("Every qualifying player this season, ranked by Barrett Score. Green = underpaid · Red = overpaid")

st.markdown("""
<style>
  /* Breathing room above and below the "What is the Barrett Score?"
     expander. Targets the element-container that wraps the expander so
     vertical-block gap collapse doesn't eat the spacing. Caption / title
     spacing rules live in COMMON_CSS (utils.py) and apply sitewide. */
  [data-testid="stExpander"] {
      margin-top: 2rem !important;
      margin-bottom: 2rem !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Season selector ──────────────────────────────────────────────────────────
# playoff_mode lives in the top nav bar (rendered by render_nav above) and is
# read here from session_state — synced sitewide.
playoff_mode = bool(st.session_state.get("playoff_mode", False))
ctrl_l, ctrl_mid, ctrl_r = st.columns([1, 1, 1])
with ctrl_l:
    season = st.selectbox("Season", SEASONS, index=0)

# ── Data loading ───────────────────────────────────────────────────────────────
# Loaded BEFORE the minutes slider is drawn, because the slider's default has to
# scale to how much of the season has actually been played. This page shows ONE
# season's own numbers and never falls back to an earlier one, unlike the home
# board, so a young season has to be usable on its own terms.
# build_ranked_projected is @st.cache_resource (no copy on hit) — must copy before mutating
_df_all = build_ranked_projected(season, playoffs=playoff_mode)
if _df_all.empty:
    st.warning(
        f"No {'playoff' if playoff_mode else 'regular season'} data for {season} yet. "
        "Try a previous season or toggle Playoff mode off."
    )
    st.stop()

# Default minimum-minutes drops in playoff mode — playoff GP is 4-28 games,
# so a 500-min threshold would hide most of the field.
#
# It also has to scale for a season IN PROGRESS. 500 minutes is six a game over
# 82, which nobody can reach in October: one game into 2026-27 the entire field
# topped out at 40 total minutes, so the fixed default filtered out all 61
# players and the page stopped with "no players cleared the threshold". Holding
# the same six-minutes-a-game bar against games actually PLAYED keeps the page
# meaningful from opening night, and the slider is still free to override.
# Shared with Team Analysis via utils.live_min_threshold. This scaling was
# written here first and nowhere else, which is precisely why Team Analysis sat
# empty with "salary coverage 0%" until 2026-10-22.
_season_gp = int(_df_all["GP"].max() or 0)
default_min_threshold, slider_max, _step = live_min_threshold(
    season, _season_gp, playoff_mode)
with ctrl_r:
    min_threshold = st.slider(
        "Min total minutes", min_value=0, max_value=slider_max,
        value=default_min_threshold, step=_step,
        help=("Hides players with fewer total minutes than this. The default is "
              "about six minutes a game over the games played so far, so it "
              f"scales as the season goes ({_season_gp} "
              f"game{'' if _season_gp == 1 else 's'} in, that is "
              f"{default_min_threshold}). Ranks never change: they are always "
              "computed on the full pool, this only trims the list. Drag to 0 "
              "to show everyone."
              if season_is_live(season) and 0 < _season_gp < 82 else
              "Hides players with fewer total minutes than this (500 is roughly "
              "six minutes a game over a full season). Ranks never change: they "
              "are always computed on the full pool, this only trims the list. "
              "Drag to 0 to show everyone."),
    )
df = _df_all[_df_all["total_min"] >= min_threshold].copy()
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
import team_suitors as _ts
_pos2k = _ts.load_player_positions()
# Curated 2K position (primary + secondary, e.g. "PG/SG") — the same source the
# Contract Predictor matches comps on — with BBRef coarse as the fallback.
df["position"] = df["Player"].map(
    lambda n: _ts.resolve_position(n, _bref_positions.get(normalize(n), ""), _pos2k))

_next_contracts = fetch_next_year_contracts(season_to_espn_year(season), cache_v=7)
_rookie_scale   = fetch_rookie_scale_players(season)

# ── Background cache warming (all seasons, bounded thread pool) ───────────────
_bootstrap_warm()

def _fmt_next_contract_local(player_name: str) -> str:
    return fmt_next_contract(player_name, _next_contracts)

df["next_contract"] = df["Player"].apply(_fmt_next_contract_local)

season_games = int(df["GP"].max())
splits_mpg_lookup = df.set_index("Player")["MPG"]

_is_pre_1990 = int(season.split("-")[0]) < 1990
if _is_pre_1990:
    st.warning(PRE_1990_SALARY_NOTE, icon="📜")

_n_under = int((df['value_diff'] < -5_000_000).sum())
_n_over  = int((df['value_diff'] >  5_000_000).sum())
render_rail(
    "The headlines", "Season Standouts",
    count=f"{len(df)} players",
    meta=f"{_n_under} underpaid · {_n_over} overpaid by $5M+ vs projection",
)

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
def _attach_prev_score(current_df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
    """Two-tier prev-season score attachment that's defensive against every
    failure mode we've hit so far:
      1. PRIMARY: merge on PLAYER_ID coerced to nullable Int64 (handles
         int64-vs-string dtype mismatches across cache versions).
      2. FALLBACK: for any row that PLAYER_ID didn't match, look up by
         normalize(Player), catches cases where one parquet has a
         negative-hash ID (BBRef path) and the other has the real NBA
         Stats ID, or any dtype weirdness that slips through coercion.

    Returns current_df with a 'prev_score' column attached (NaN for any
    player who genuinely didn't exist in prev_df, which is honest).
    """
    out = current_df.copy()
    out["prev_score"] = float("nan")

    if "PLAYER_ID" in current_df.columns and "PLAYER_ID" in prev_df.columns:
        c_ids = pd.to_numeric(out["PLAYER_ID"], errors="coerce").astype("Int64")
        p_ids = pd.to_numeric(prev_df["PLAYER_ID"], errors="coerce").astype("Int64")
        prev_by_id = (
            prev_df.assign(_pid=p_ids)
                   .dropna(subset=["_pid"])
                   .drop_duplicates(subset=["_pid"], keep="first")
                   .set_index("_pid")["barrett_score"]
                   .to_dict()
        )
        out["prev_score"] = c_ids.map(prev_by_id).astype("Float64").astype(float)

    # Name fallback for any unmatched players
    missing = out["prev_score"].isna()
    if missing.any():
        prev_by_name = {
            normalize(n): s
            for n, s in zip(prev_df["Player"], prev_df["barrett_score"])
            if pd.notna(s)
        }
        out.loc[missing, "prev_score"] = out.loc[missing, "Player"].apply(
            lambda n: prev_by_name.get(normalize(n), float("nan"))
        )
    return out


if _prev_season:
    try:
        _prev_df  = build_ranked_projected(_prev_season, playoffs=playoff_mode)
        _prev_threshold = 100 if playoff_mode else DEFAULT_MIN_THRESHOLD
        _prev_df  = _prev_df[_prev_df["total_min"] >= _prev_threshold].copy()

        _df_for_merge = df[["Player", "Team", "barrett_score", "PLAYER_ID"]].copy()
        _merged = _attach_prev_score(_df_for_merge, _prev_df)
        _merged = _merged.dropna(subset=["prev_score"])
        _merged["delta"] = _merged["barrett_score"] - _merged["prev_score"]
        if not _merged.empty:
            _improved_row   = _merged.loc[_merged["delta"].idxmax()]
            _improved_delta = float(_improved_row["delta"])
    except Exception:
        pass

st.markdown("""
<style>
/* Standout cards: headshot + kicker/name/big value/sub, team-color left rail
   (mirrors the homepage front-page cards). */
[data-testid="stHorizontalBlock"]:has(.hero-card){margin-top:0.9rem;}
    .hero-card{position:relative;display:flex;align-items:center;gap:.8rem;
    background:var(--panel-solid);border:1px solid var(--panel-line);
    border-left:4px solid var(--team,var(--accent-teal));border-radius:12px;
    padding:.8rem 1rem;box-shadow:var(--shadow-card);min-height:104px;height:100%;}
img.hero-face{width:52px;height:52px;border-radius:50%;object-fit:cover;
    object-position:center 12%;background:var(--panel-2);
    border:2px solid var(--team,var(--panel-line));flex:0 0 auto;}
.hero-card .k{font-size:.64rem;font-weight:800;letter-spacing:.1em;
    text-transform:uppercase;color:var(--fg-4);}
.hero-card .nm{font-size:.95rem;font-weight:800;color:var(--fg-1);line-height:1.15;
    margin:.1rem 0;}
.hero-card .v{font-size:1.35rem;font-weight:800;font-variant-numeric:tabular-nums;
    line-height:1.1;}
.hero-card .v.good{color:var(--value-good);}
.hero-card .v.bad{color:var(--value-bad);}
.hero-card .v.teal{color:var(--accent-teal);}
.hero-card .v.sky{color:var(--sky);}
.hero-card .sub{font-size:.72rem;color:var(--fg-3);}
</style>
""", unsafe_allow_html=True)


def _hero_card(kicker: str, name, team, value_cls: str, value: str, sub: str) -> str:
    """One standout card. Every data-derived string is escaped; the headshot
    ('' when unknown, 404s auto-hide) and team rail come from the shared kit."""
    _hx = TEAM_HEX.get(str(team), "")
    _style = f' style="--team:{_hx}"' if _hx else ""
    return (f'<div class="hero-card"{_style}>'
            f'{face_img(str(name), "hero-face fp-face")}'
            f'<span style="min-width:0">'
            f'<span class="k" style="display:block">{html.escape(kicker)}</span>'
            f'<span class="nm" style="display:block">{html.escape(str(name))}</span>'
            f'<span class="v {value_cls}" style="display:block">{html.escape(value)}</span>'
            f'<span class="sub" style="display:block">{html.escape(sub)}</span>'
            f'</span></div>')


h1, h2, h3, h4 = st.columns(4, gap="medium")
with h1:
    _best_sal = float(_best_row["salary"] or 0) / 1e6
    _best_sub = (f"{_best_row['Team']} · ${_best_sal:.1f}M salary" if _best_sal > 0
                 else f"{_best_row['Team']} · League #1 by Barrett Score")
    st.markdown(_hero_card(
        "Best player right now", _best_row["Player"], _best_row["Team"],
        "teal", f"{_best_row['barrett_score']:.2f}", _best_sub,
    ), unsafe_allow_html=True)
with h2:
    steal_diff = abs(_steal_row['value_diff'] / 1e6)
    st.markdown(_hero_card(
        "Biggest steal", _steal_row["Player"], _steal_row["Team"],
        "good", f"-${steal_diff:.1f}M",
        f"paid ${_steal_row['salary'] / 1e6:.1f}M · worth ${_steal_row['projected_salary'] / 1e6:.1f}M",
    ), unsafe_allow_html=True)
with h3:
    over_diff = _overpaid_row['value_diff'] / 1e6
    st.markdown(_hero_card(
        "Most overpaid", _overpaid_row["Player"], _overpaid_row["Team"],
        "bad", f"+${over_diff:.1f}M",
        f"paid ${_overpaid_row['salary'] / 1e6:.1f}M · worth ${_overpaid_row['projected_salary'] / 1e6:.1f}M",
    ), unsafe_allow_html=True)
with h4:
    if _improved_row is not None:
        _sign = "+" if _improved_delta >= 0 else ""
        st.markdown(_hero_card(
            "Most improved", _improved_row["Player"], _improved_row["Team"],
            "sky", f"{_sign}{_improved_delta:.1f} pts",
            f"{_improved_row['prev_score']:.1f} → {_improved_row['barrett_score']:.1f} vs {_prev_season}",
        ), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="hero-card">
            <span style="min-width:0">
                <span class="k" style="display:block">Most improved</span>
                <span class="nm" style="display:block">—</span>
                <span class="sub" style="display:block">No prior season to compare</span>
            </span>
        </div>""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Top 10 — current season bar chart
# ══════════════════════════════════════════════════════════════════════════════
render_rail("The leaders", "Top 10 Players", meta=season)

_top10 = df.nsmallest(10, "score_rank")[["Player", "Team", "barrett_score", "PLAYER_ID"]].reset_index(drop=True)

# Same two-tier attachment as the Most-Improved hero card — PLAYER_ID
# (dtype-coerced) primary, name-normalize fallback. Anyone who's still
# missing after that genuinely didn't qualify last season (e.g. rookies).
if _prev_df is not None:
    _top10 = _attach_prev_score(_top10, _prev_df)
    _top10["delta"] = _top10["barrett_score"] - _top10["prev_score"]
else:
    _top10["delta"] = float("nan")

# Themed leaderboard rows (site kit style), best first: rank · headshot ·
# name+team (links to the player hub) · teal score bar · score · delta chip.
# Teal is the site's Barrett Score color; deltas are never color-alone (each
# carries a triangle glyph and a signed number).
_top10 = _top10.sort_values("barrett_score", ascending=False)
_score_max = float(_top10["barrett_score"].max()) or 1.0

_t10_rows = []
for _rk, (_, _r) in enumerate(_top10.iterrows(), start=1):
    _nm = str(_r["Player"])
    _tm = str(_r["Team"])
    _sc = float(_r["barrett_score"])
    _d = _r["delta"]
    if pd.isna(_d):
        _chip = '<span class="t10-d t10-na" title="Did not qualify last season">&ndash;</span>'
        _tip = f"{_sc:.1f} Barrett Score"
    else:
        _up = _d >= 0
        _chip = (f'<span class="t10-d {"t10-up" if _up else "t10-dn"}">'
                 f'{"&#9650;" if _up else "&#9660;"} {_d:+.1f}</span>')
        _tip = f"{_sc:.1f} Barrett Score, {_d:+.1f} vs prior season"
    _t10_rows.append(
        f'<a class="t10-row" href="/?player={quote(_nm)}" target="_top" title="{html.escape(_tip, quote=True)}">'
        f'<span class="t10-rk">{_rk}</span>'
        f'{face_img(_nm, "t10-face")}'
        f'<span class="t10-who"><span class="t10-name">{html.escape(_nm)}</span>'
        f'<span class="t10-team">{html.escape(_tm)}</span></span>'
        f'<span class="t10-track"><span class="t10-bar" style="width:{_sc / _score_max * 100:.1f}%"></span></span>'
        f'<span class="t10-score">{_sc:.1f}</span>'
        f'{_chip}</a>')

st.markdown("""
<style>
/* bottom padding oversized to outrun Streamlit 1.51's -16px stMarkdownContainer
   margin, netting ~20px of visible air above the caption */
.t10-wrap{display:flex;flex-direction:column;margin:.2rem 0 0;padding-bottom:2.3rem;}
a.t10-row{display:flex;align-items:center;gap:.6rem;padding:.34rem .4rem;
    border-radius:8px;text-decoration:none;}
a.t10-row:hover{background:var(--panel-hover);}
a.t10-row:hover .t10-name{color:var(--accent-teal);}
.t10-rk{flex:0 0 1.2rem;text-align:right;font-size:.74rem;font-weight:700;
    color:var(--fg-5);font-variant-numeric:tabular-nums;}
.t10-face{flex:0 0 30px;width:30px;height:30px;border-radius:50%;object-fit:cover;
    object-position:top;background:var(--panel-2);}
.t10-who{flex:0 0 200px;min-width:0;display:flex;align-items:baseline;gap:.45rem;overflow:hidden;}
.t10-name{font-size:.86rem;font-weight:700;color:var(--fg-1);white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis;}
.t10-team{font-size:.68rem;font-weight:600;color:var(--fg-5);}
.t10-track{flex:1;height:9px;border-radius:0 4px 4px 0;background:var(--panel-2);
    overflow:hidden;}
.t10-bar{display:block;height:100%;background:var(--accent-teal);border-radius:0 4px 4px 0;}
.t10-score{flex:0 0 2.6rem;text-align:right;font-size:.88rem;font-weight:800;
    color:var(--accent-teal);font-variant-numeric:tabular-nums;}
.t10-d{flex:0 0 3.4rem;text-align:right;font-size:.72rem;font-weight:700;
    font-variant-numeric:tabular-nums;white-space:nowrap;}
.t10-up{color:var(--value-good);} .t10-dn{color:var(--value-bad);} .t10-na{color:var(--fg-5);}
@media (max-width:640px){.t10-who{flex-basis:120px;}.t10-team{display:none;}}
</style>
<div class="t10-wrap">""" + "".join(_t10_rows) + "</div>", unsafe_allow_html=True)
st.caption("Bar = current Barrett Score · ▲ / ▼ = change vs prior season")

st.divider()

# ── Token-based cell styles for the themed html_table (follow light/dark) ──────
def _hsty_rankdiff(v, _row):
    try:
        n = float(v)
    except (TypeError, ValueError):
        return ""
    if n > 20:  return "color:var(--value-good);font-weight:700"
    if n > 0:   return "color:var(--value-good-s)"
    if n < -20: return "color:var(--value-bad);font-weight:700"
    if n < 0:   return "color:var(--value-bad-s)"
    return ""

def _hsty_delta(v, _row):
    try:
        n = float(v)
    except (TypeError, ValueError):
        return ""
    if n > 20:  return "color:var(--value-bad);font-weight:700"
    if n > 5:   return "color:var(--value-bad-s)"
    if n < -20: return "color:var(--value-good);font-weight:700"
    if n < -5:  return "color:var(--value-good-s)"
    return ""

def _fmt_next_chip(v):
    """Next-contract cell: money + a status chip (TO/PO/RFA), muted sentinel.
    Output is trusted HTML — the column must be listed in raw=."""
    s = str(v)
    if s == "—":
        return '<span style="color:var(--fg-6)">—</span>'
    if s == "RFA":
        return '<span class="hv-chip rfa">RFA</span>'
    if s.endswith(" TO"):
        return f'{html.escape(s[:-3])} <span class="hv-chip to">TO</span>'
    if s.endswith(" PO"):
        return f'{html.escape(s[:-3])} <span class="hv-chip po">PO</span>'
    return html.escape(s)

def _fmt_team_dot(v):
    """Team cell: team-color dot + abbrev, linked to that team's value page when
    one exists (raw column; unknown teams fall back to plain text)."""
    return team_cell(v)

def _score_bar_style(max_val):
    """In-cell Barrett Score bar scaled against the visible table's max."""
    try:
        _mx = float(max_val)
    except (TypeError, ValueError):
        _mx = 0.0
    if _mx != _mx:  # NaN
        _mx = 0.0
    def _sty(v, _row):
        try:
            n = float(v)
        except (TypeError, ValueError):
            return ""
        if n != n or _mx <= 0:
            return ""
        pct = max(0.0, min(n / _mx, 1.0)) * 100
        return f"background:linear-gradient(90deg,var(--bar-tint) {pct:.0f}%,transparent 0)"
    return _sty

def _hsty_salary(_v, row):
    if normalize(str(row.get("Player", ""))) in _rookie_scale:
        return "color:var(--purple);font-weight:600"
    return ""

def _fmt_salary_rs(v, row):
    """Salary cell: rookie-scale contracts get a tiny RS superscript so the
    purple color is not the only signal (colorblind-safe). Output is trusted
    HTML — the column must be listed in raw=."""
    s = f"${v:.2f}M"
    if normalize(str(row.get("Player", ""))) in _rookie_scale:
        return (s + '<sup style="color:var(--fg-6);font-size:0.62em;'
                    'margin-left:1px" title="Rookie scale contract">RS</sup>')
    return s

# ══════════════════════════════════════════════════════════════════════════════
# Rankings content
# ══════════════════════════════════════════════════════════════════════════════

player_id_map_full = dict(zip(df["Player"], df["PLAYER_ID"].astype(int)))
dlebron_lookup = fetch_dlebron(season)
_gp_sum = float(df["GP"].sum())
league_avg_ts = float((df["ts_pct"] * df["GP"]).sum() / _gp_sum) if _gp_sum > 0 else 0.0
if not dlebron_lookup:
    # The copy has to follow the metric. This page said "no on/off, no
    # role-adjustment" for twelve minutes after the blend went live, which was
    # the exact opposite of what it had started computing.
    _blend = fetch_defense_blend(season)
    if _blend:
        _meta = fetch_defense_blend_meta(season)
        _ends = (_meta.get("scaled_to") or {}).get("endpoints") or []
        _scale = (f"The best defender this season sits at {_ends[1]:+.0f} and the worst at "
                  f"{_ends[0]:+.0f}, and the score counts it twice, so defence swings "
                  f"{abs(_ends[0]) * 2 + abs(_ends[1]) * 2:.0f} points across the league. "
                  "Past seasons use BBall Index's own narrower scale, so defensive values "
                  "are not comparable across seasons.") if len(_ends) == 2 else ""
        st.info("D-LEBRON (BBall Index's RAPM-based defensive metric) is not published for a "
                "season in progress. Defensive contribution here is a blend of two on/off "
                "measures: a role-weighted composite of deterrence and event production, and a "
                "career-possession-weighted version that reads an established defender off what "
                f"opponents stop doing rather than off his counting stats. {_scale} "
                f"It covers {len(_blend)} players; anyone below the possession floor keeps the "
                "box-score estimate. Early in a season it is thin: a handful of games moves it "
                "a long way.")
    else:
        st.info("D-LEBRON (BBall Index's RAPM-based defensive metric) only goes back to 2009-10. "
                "For this season, defensive contribution is estimated from box-score stats "
                "(BLK, STL, DREB, and PF) centered on the season's league average. "
                "Calibrated to roughly match D-LEBRON's scale, but noisier (no on/off, no "
                "role-adjustment, no luck-adjustment).")


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
            "Defense": f"{r['d_lebron']:.2f}", "TS%": ts_str,
            "Eff. Adj": f"{r['efficiency_adj']:.2f}",
            "Base Score": f"{r['base_score']:.2f}",
            "Avail ×": f"{r['avail_mult']:.3f}",
            "Barrett Score": f"{r['barrett_score']:.2f}",
        })
        row_styles.append("tot_stat" if is_tot else "stat")

    fmt = pd.DataFrame(rows_out)

    _split_num = ["GP", "MPG", "Total MIN", "PTS", "AST", "OREB", "DREB", "BLK",
                  "STL", "TOV", "PF", "Defense", "TS%", "Eff. Adj", "Base Score",
                  "Avail ×", "Barrett Score"]
    html_table(
        fmt,
        formatters={"Team": _fmt_team_dot},
        raw={"Team"},
        aligns={c: "right" for c in _split_num},
        helps={
            "Defense": "Defensive impact in points prevented per game. Higher is better, and the score counts it 1.75 times this season.",
            "TS%": "True Shooting %. League avg ~57%.",
            "Eff. Adj": "Efficiency adjustment added to Base Score.",
            "Base Score": "Raw per-game value before the availability multiplier.",
            "Avail ×": "Availability multiplier (0.30–1.00).",
            "Barrett Score": "Base Score × Availability Multiplier.",
        },
        row_style=lambda rd: ("background:var(--panel-2);font-weight:700"
                              if rd.get("Team") == "TOT" else ""),
        height=min(420, max(120, len(fmt) * 38 + 46)),
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
# Also from _df_all: the same defect as the Team filter twenty lines below.
# Options drawn from the filtered frame drop players as the slider rises, and a
# keyed multiselect holding a player who is no longer an option loses him
# without a word.
_current_players = set(_df_all["Player"].unique())
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
    # From _df_all, the pool BEFORE the minutes slider, not from df after it.
    # Built from df, a team whose players all fell below the threshold vanished
    # from the list mid-session: raising the slider with LAL selected dropped
    # LAL out of the options, and an unkeyed selectbox whose options changed is
    # a NEW widget to Streamlit, so the selection silently reset to "All" and
    # the team could not even be typed back in. The option list has to be a
    # property of the season, not of the current filter.
    team_options = ["All"] + sorted(_df_all["Team"].dropna().unique().tolist())
    team_filter = st.selectbox("Team", team_options, key="rankings_team")
with col_d:
    pos_options = ["All", "PG", "SG", "SF", "PF", "C"]
    pos_filter = st.selectbox("Position", pos_options)

# ── Per-game line for the expandable row panel ────────────────────────────
# build_raw carries no PTS/AST/REB -- it only keeps what the score needs -- so
# the panel reads the box feed directly. Joined through NameIndex rather than a
# bare normalize() because the two frames come from different sources and
# disagree on suffixes and transliterations; a bare key silently shows an empty
# panel for exactly the players whose names are awkward.
@st.cache_data(ttl=3600, show_spinner=False)
def _per_game_lookup(season_key):
    """Per-game line for the row panel, from the COMMITTED artifact first.

    The live fetch is the fallback, not the default: on Render it reaches
    stats.nba.com, which is blocked there, and the live season has no committed
    league_stats parquet behind it. The first version of this shipped with the
    live fetch as the only source, so the panel rendered for all 353 players
    locally and for NONE in production, silently, because an empty lookup just
    skips the panel. Regenerate with scripts/build_per_game.py.
    """
    art = _dc_path(f"per_game_{season_key.replace('-', '_')}_v3.json")
    if art.exists():
        try:
            _d = json.loads(art.read_text())
            return _d.get("values") or {}
        except Exception:
            pass
    src = fetch_league_stats(season_key)
    if src is None or getattr(src, "empty", True) or "PTS" not in getattr(src, "columns", []):
        src = fetch_bref_player_stats(season_key)
    if src is None or getattr(src, "empty", True):
        return {}
    keep = [c for c in ("PTS", "REB", "OREB", "DREB", "AST", "STL", "BLK",
                        "TOV", "PF", "MIN", "GP", "FG_PCT", "FG3_PCT", "FT_PCT")
            if c in src.columns]
    return {str(n): {c: r[c] for c in keep}
            for n, r in zip(src["PLAYER_NAME"], src.to_dict("records"))}


@st.cache_data(ttl=3600, show_spinner=False)
def _per_game_meta(season_key):
    """The artifact's _meta: the bar reference values and the ranked pool.

    Its own cached function, NOT a side effect of _per_game_lookup. The first
    version set a module-level dict from inside that cached function, which
    does not run on a cache hit, and the dict was reassigned to {} further down
    the file anyway -- so the references were empty on every path and every bar
    rendered at zero width. A cached function that mutates a global is a
    function whose work silently stops happening.
    """
    art = _dc_path(f"per_game_{season_key.replace('-', '_')}_v3.json")
    if art.exists():
        try:
            return json.loads(art.read_text()).get("_meta") or {}
        except Exception:
            pass
    return {}


_pg = _per_game_lookup(season)
_pg_meta = _per_game_meta(season)
_bar_ref = _pg_meta.get("bar_scale") or {}
_bar_pool = int(_pg_meta.get("ranked_pool") or 0)
# Defence is signed, so its bar shows position within the season's own range
# rather than a proportion. Read from the blend artifact so moving the scale
# moves the bar, the same way the notice above derives its endpoints.
_def_ends = tuple((fetch_defense_blend_meta(season).get("scaled_to") or {})
                  .get("endpoints") or (-4.0, 6.0))
# TS% and Defence come from the board frame keyed by player, NOT from the row
# dict. The row dict only carries the columns the CURRENT view renders, so in
# simple view those two cells do not exist and both bars silently vanished --
# visible in Barrett's screenshot as a panel that stops after PFPG.
_ts_by_player = dict(zip(df["Player"], df["ts_pct"])) if "ts_pct" in df else {}
_def_by_player = dict(zip(df["Player"], df["d_lebron"])) if "d_lebron" in df else {}
_pg_idx = NameIndex(_pg) if _pg else None

# Bars, with the fill in the team's colour. Rank is drawn rather than stated
# because a bar survives a two-game sample that a percentage does not: 84.6%
# from the field reads as a bug, "4th in the league" reads as a fact.
# Two deliberate rows of five, in Barrett's order. The production line first,
# the cost-and-context line second.
_PANEL_ROW1 = [("PTS", "PPG", "PTS_RANK"), ("AST", "APG", "AST_RANK"),
               ("REB", "RPG", "REB_RANK"), ("STL", "SPG", "STL_RANK"),
               ("BLK", "BPG", "BLK_RANK")]
_PANEL_ROW2 = [("MIN", "MPG", "MIN_RANK"), ("TOV", "TPG", None),
               ("PF", "PFPG", None), ("__TS", "TS%", None),
               ("__DEF", "DEF", None)]
_PANEL_LOWER_IS_BETTER = {"TOV", "PF"}


def _ord(v):
    v = int(v)
    suf = "th" if 10 <= v % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(v % 10, "th")
    return f"{v}{suf}"


def _row_panel(rd):
    """The stat line that drops down when a row is clicked."""
    if _pg_idx is None:
        return ""
    import html as _h
    rec = _pg_idx.get(str(rd.get("Player", "")))
    if not rec:
        return ""
    pool = _bar_pool
    col = team_bar_color(str(rd.get("Team", "")))

    def _one(key, label, rank_key):
        """One bar. Width is value against a high REFERENCE value, not rank
        percentile: percentile saturates where it matters, since everyone in
        the league's top 25 of 388 drew at 94-100%."""
        if key == "__TS":
            v = _ts_by_player.get(str(rd.get("Player", "")))
            if v is None or v != v:
                return ""
            v = float(v) * 100.0
            txt, width = f"{v:.1f}%", min(100.0, max(3.0, v / 65.0 * 100.0))
            return (f'<span class="bar"><i>{label}</i><b>{txt}</b>'
                    f'<span class="trk"><span class="fil" style="width:{width:.0f}%">'
                    f'</span></span><u></u></span>')
        if key == "__DEF":
            v = _def_by_player.get(str(rd.get("Player", "")))
            if v is None or v != v:
                return ""
            # Signed, so position in its range rather than a proportion of it.
            lo, hi = _def_ends
            width = min(100.0, max(3.0, (float(v) - lo) / (hi - lo) * 100.0))
            return (f'<span class="bar"><i>{label}</i><b>{float(v):.2f}</b>'
                    f'<span class="trk"><span class="fil" style="width:{width:.0f}%">'
                    f'</span></span><u></u></span>')
        v = rec.get(key)
        if v is None or (isinstance(v, float) and v != v):
            return ""
        r = rec.get(rank_key) if rank_key else None
        has_rank = pool > 0 and r is not None and r == r and int(r) > 0
        ref = _bar_ref.get(key)
        try:
            width = min(100.0, max(3.0, float(v) / float(ref) * 100.0)) if ref else 0.0
        except (TypeError, ValueError, ZeroDivisionError):
            width = 0.0
        # Turnovers and fouls are the only stats here where MORE is worse, so
        # filling them like the rest made 4.5 fouls look like an achievement.
        # Inverted, every bar on the panel now means the same thing: longer is
        # better. The number beside it is still the raw count.
        if key in _PANEL_LOWER_IS_BETTER and width:
            width = min(100.0, max(3.0, 100.0 - width))
        return (f'<span class="bar"><i>{label}</i><b>{float(v):.1f}</b>'
                f'<span class="trk"><span class="fil" style="width:{width:.0f}%">'
                f'</span></span><u>{_ord(r) if has_rank else ""}</u></span>')

    r1 = "".join(_one(*b) for b in _PANEL_ROW1)
    r2 = "".join(_one(*b) for b in _PANEL_ROW2)
    if not r1 and not r2:
        return ""
    bars = [f'<div class="pbrow">{r1}</div>', f'<div class="pbrow">{r2}</div>']
    return f'<div class="pbars" style="--tc:{col}">{"".join(bars)}</div>'


display = df.copy()
if search:
    display = display[display["Player"].str.contains(search, case=False, regex=False)]
if team_filter != "All":
    display = display[display["Team"] == team_filter]
if pos_filter != "All":
    display = display[display["position"].str.contains(pos_filter, regex=False, na=False)]

sort_map = {
    "Barrett Score (best first)": ("score_rank", True),
    "Salary (highest first)":     ("salary_rank", True),
    "Most Underpaid":             ("value_diff", True),
    "Most Overpaid":              ("value_diff", False),
}
sort_col, sort_asc = sort_map[view]
display = display.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

# Now that the team list is the whole league rather than whoever survived the
# slider, a team with nobody above the threshold is reachable. Say so, instead
# of rendering an empty table that reads like a loading failure.
if display.empty:
    _bits = [b for b in (f"{team_filter}" if team_filter != "All" else "",
                         f"{pos_filter}" if pos_filter != "All" else "",
                         f'matching "{search}"' if search else "") if b]
    st.warning(
        f"No {' '.join(_bits) or 'player'} cleared the {min_threshold}-minute "
        f"threshold. Lower it with the slider above, or widen the filters."
    )

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
    render_rail("Explore", "Graph Mode", meta=season)
    st.caption("Plot any two metrics against each other. Pick X, Y, and color, then hover any dot for player detail.")

    # Drop rows with no salary data — they cluster at $0 and obscure salary axes.
    _n_no_salary = (display["salary"] <= 0).sum()
    gd = display[display["salary"] > 0].copy()
    gd["Salary $M"]        = (gd["salary"] / 1e6).round(2)
    gd["Proj. Salary $M"]  = (gd["projected_salary"] / 1e6).round(2)
    gd["Δ Market $M"]      = (gd["value_diff"] / 1e6).round(2)
    gd["Barrett Score"]    = gd["barrett_score"].round(2)
    gd["TS%"]              = (gd["ts_pct"] * 100).round(1)
    gd["Defense"]          = gd["d_lebron"].round(2)
    gd["Score Rank"]       = gd["score_rank"]
    gd["Salary Rank"]      = gd["salary_rank"]

    if _n_no_salary > 0:
        st.caption(
            f"⚠️ Hiding {_n_no_salary} player(s) with no salary data on file "
            "(common for older seasons). Their ranks remain accurate elsewhere."
        )

    NUMERIC_AXES = [
        "Barrett Score", "Salary $M", "Proj. Salary $M", "Δ Market $M",
        "GP", "MPG", "Defense", "TS%", "Score Rank", "Salary Rank",
    ]
    COLOR_NUMERIC     = ["Δ Market $M", "Barrett Score", "Defense", "TS%", "Salary $M"]
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
        hi = gd[gd["Player"].str.contains(graph_search, case=False, regex=False)]
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

    st.plotly_chart(theme_fig(fig_scatter), use_container_width=True,
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

    _g_bar = _score_bar_style(sc_tbl["Barrett Score"].max())
    html_table(
        sc_tbl,
        formatters={
            "Team":            _fmt_team_dot,
            "Score Rank":      lambda v: str(int(v)),
            "Barrett Score":   lambda v: f"{v:.2f}",
            "Proj. Salary $M": lambda v: f"${v:.2f}M",
            "Salary $M":       lambda v: f"${v:.2f}M",
            "Δ Market $M":     lambda v: f"${v:.2f}M",
        },
        raw={"Team"},
        styles={"Δ Market $M": _hsty_delta, "Barrett Score": _g_bar},
        aligns={c: "right" for c in ["#", "Score Rank", "Barrett Score",
                "Proj. Salary $M", "Salary $M", "Δ Market $M"]},
        numeric={"#", "Score Rank", "Barrett Score", "Proj. Salary $M",
                 "Salary $M", "Δ Market $M"},
        height=500,
    )
    st.caption(
        "**Δ Market** = Actual − Projected. Positive (red) = overpaid. Negative (green) = underpaid. "
        "Switch the X/Y dropdowns above to plot any two metrics: e.g. MPG vs Barrett Score, "
        "TS% vs Δ Market, or D-LEBRON vs Salary."
    )
    st.divider()

if show_splits:
    splits_df = build_splits_data(season, salary_lookup)

if show_splits and splits_df is not None:
    # ── Splits table ──────────────────────────────────────────────────────
    if search:
        matched = splits_df[splits_df["Player"].str.contains(search, case=False, regex=False)]["Player"].unique()
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
        pos_players = set(df[df["position"].str.contains(pos_filter, regex=False, na=False)]["Player"])
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

    _nc_lookup = df.set_index("Player")["next_contract"]
    sdisplay["Next $"] = sdisplay["Player"].map(_nc_lookup).fillna("—")

    render_rail("The board", f"{season} Splits", count=f"{len(sdisplay)} rows")

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
                        "Salary Rank", "Rank Diff", "Defense", "TS%", "Next $"]
        sfmt.insert(0, "#", range(1, len(sfmt) + 1))
    else:
        sfmt = sdisplay[["Player", "Team", "barrett_score", "salary",
                          "projected_salary", "value_diff", "Next $"]].copy()
        sfmt["salary"]           = sfmt["salary"] / 1_000_000
        sfmt["projected_salary"] = sfmt["projected_salary"] / 1_000_000
        sfmt["value_diff"]       = sfmt["value_diff"] / 1_000_000
        sfmt.columns = ["Player", "Team", "Barrett Score", "Salary", "Proj. Salary", "Δ Market", "Next $"]
        sfmt.insert(0, "#", range(1, len(sfmt) + 1))

    _s_num = {"#", "GP", "MPG", "Base Score", "Avail ×", "Barrett Score",
              "Score Rank", "Salary", "Proj. Salary", "Δ Market", "Salary Rank",
              "Rank Diff", "Defense", "TS%"}
    _s_bar = _score_bar_style(sfmt["Barrett Score"].max())
    html_table(
        sfmt,
        formatters={
            "Team": _fmt_team_dot, "Next $": _fmt_next_chip,
            "GP": lambda v: str(int(v)), "MPG": lambda v: f"{v:.2f}",
            "Base Score": lambda v: f"{v:.2f}", "Avail ×": lambda v: f"{v:.3f}",
            "Barrett Score": lambda v: f"{v:.2f}", "Score Rank": lambda v: str(int(v)),
            "Salary": _fmt_salary_rs, "Proj. Salary": lambda v: f"${v:.2f}M",
            "Δ Market": lambda v: f"${v:.2f}M", "Salary Rank": lambda v: str(int(v)),
            "Rank Diff": lambda v: f"{int(v):+d}", "Defense": lambda v: f"{v:.2f}",
            "TS%": lambda v: f"{v:.1f}%",
        },
        raw={"Team", "Next $", "Salary"},
        styles={"Rank Diff": _hsty_rankdiff, "Δ Market": _hsty_delta,
                "Barrett Score": _s_bar, "Salary": _hsty_salary},
        aligns={c: "right" for c in _s_num},
        numeric=_s_num,
        row_style=lambda rd: ("background:var(--panel-2);font-weight:700"
                              if rd.get("Team") == "TOT" else ""),
        height=600,
    )
    dl_col, cap_col = st.columns([1, 9])
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
            "Purple salary with the small RS mark = rookie scale. "
            "**Next $**: plain = guaranteed · TO chip = team option · PO chip = player option · gray = UFA."
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
        "Defense", "TS%",
    ]
    display_fmt.insert(0, "#", range(1, len(display_fmt) + 1))
    display_fmt["Next $"] = display["next_contract"].values
    if not advanced:
        display_fmt = display_fmt[["#", "Player", "Team", "Barrett Score", "Salary", "Proj. Salary", "Δ Market", "Next $"]]

    render_rail("The board",
                f"{season} {'Playoff ' if playoff_mode else ''}Rankings",
                count=f"{len(display_fmt)} players")

    # Superset of formatters/styles — html_table only applies those whose column
    # is actually present, so basic + advanced modes both work from one spec.
    _r_num = {"#", "GP", "MPG", "Base Score", "Avail ×", "Barrett Score",
              "Score Rank", "Salary", "Proj. Salary", "Δ Market", "Salary Rank",
              "Rank Diff", "Defense", "TS%"}
    _r_bar = _score_bar_style(display_fmt["Barrett Score"].max())
    html_table(
        display_fmt,
        formatters={
            "Team":          _fmt_team_dot,
            "Next $":        _fmt_next_chip,
            "GP":            lambda v: str(int(v)),
            "MPG":           lambda v: f"{v:.2f}",
            "Base Score":    lambda v: f"{v:.2f}",
            "Avail ×":       lambda v: f"{v:.3f}",
            "Barrett Score": lambda v: f"{v:.2f}",
            "Score Rank":    lambda v: str(int(v)),
            "Salary":        _fmt_salary_rs,
            "Proj. Salary":  lambda v: f"${v:.2f}M",
            "Δ Market":      lambda v: f"${v:.2f}M",
            "Salary Rank":   lambda v: str(int(v)),
            "Rank Diff":     lambda v: f"{int(v):+d}",
            "Defense":      lambda v: f"{v:.2f}",
            "TS%":           lambda v: f"{v:.1f}%",
        },
        raw={"Team", "Next $", "Salary"},
        styles={
            "Rank Diff":     _hsty_rankdiff,
            "Δ Market":      _hsty_delta,
            "Barrett Score": _r_bar,
            "Salary":        _hsty_salary,
        },
        aligns={c: "right" for c in _r_num},
        numeric=_r_num,
        helps={
            "Next $": "Next season salary. Plain = guaranteed · TO chip = team option · PO chip = player option · gray = UFA.",
            "Barrett Score": "Base Score × Availability Multiplier.",
            "Salary": "Player's actual salary this season. Purple with the RS mark = rookie-scale contract.",
            "Proj. Salary": "Salary earned by whoever holds the same rank by pay.",
            "Δ Market": "Actual − Projected. Positive (red) = overpaid; negative (green) = underpaid.",
            "Rank Diff": "Salary Rank − Score Rank. Positive (green) = underpaid.",
            "Defense": "Defensive impact in points prevented per game. Higher is better, and the score counts it 1.75 times this season.",
            "TS%": "True Shooting %. League avg ~57%.",
        },
        height=min(620, max(140, len(display_fmt) * 38 + 46)),
        detail=_row_panel,
        detail_col="Player",
    )
    dl_col_r, cap_col_r = st.columns([1, 9])
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
                       "Purple salary with the small RS mark = rookie scale contract (1st-round pick, yrs 1–4). "
                       "**Next $**: plain = guaranteed · TO chip = team option · PO chip = player option · gray = UFA.")
        else:
            st.caption("**Proj. Salary** = what this player would earn paid by Barrett Score rank. **Δ Market** = Actual − Projected. "
                       "Purple salary with the small RS mark = rookie scale contract (1st-round pick, yrs 1–4). "
                       "**Next $**: plain = guaranteed · TO chip = team option · PO chip = player option · gray = UFA.")

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
                title=f"Barrett Score · {n_seasons}-Season Trend",
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
            # Baseline at 0 → tallest score in the group (with a little headroom).
            _trend_max = trend_df["barrett_score"].max() if not trend_df.empty else 1
            _trend_min = trend_df["barrett_score"].min() if not trend_df.empty else 0
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.15)",
                font_color="white",
                margin=dict(l=50, r=50, t=40, b=80),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="",
                           type="category",
                           categoryorder="array",
                           categoryarray=selected_seasons),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Barrett Score",
                           tickformat=".1f", range=[min(0, _trend_min), _trend_max * 1.08]),
                legend=dict(
                    orientation="h",
                    x=0.5, xanchor="center",
                    y=-0.2, yanchor="top",
                    title="",
                ),
            )
            st.plotly_chart(theme_fig(fig_trend), use_container_width=True,
                            config={"displayModeBar": False})
            no_dlebron = [s for s in selected_seasons if s < "2009-10"]
            caption = "★ = current season"
            if no_dlebron:
                caption += (f"  ·  ⚠️ D-LEBRON unavailable for "
                            f"{', '.join(no_dlebron)}, defense estimated from box-score stats "
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
                    render_rail("Month by month", f"{_solo} Cumulative Score", meta=season)
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
                    st.plotly_chart(theme_fig(_mfig), use_container_width=True,
                                    config={"displayModeBar": False})

        for name in new_selected:
            title_col, btn_col = st.columns([20, 1], vertical_alignment="bottom")
            with title_col:
                render_rail("Season splits", name, meta=season)
            with btn_col:
                if st.button("✕", key=f"x_{name}", help="Remove"):
                    updated = [n for n in new_selected if n != name]
                    st.session_state["rankings_selected"] = updated
                    st.session_state["rankings_multiselect"] = updated
                    st.rerun()
            render_splits_panel(name, season)
        st.caption("**TOT** = full season combined (bold).")
        st.divider()


from utils import render_footer
render_footer()
