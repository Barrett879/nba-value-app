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
    _fmt_salary, fmt_next_contract, classify_fa_status,
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
    # Shared classifier — cross-checks the contract-end scraper for players the
    # salary feed omits (e.g. Austin Reaves' player option). Only the current
    # season has reliable contract data, so skip the cross-check otherwise.
    return classify_fa_status(row["Player"], row["next_contract"], _rookie_scale,
                              season, cross_check=(season == SEASONS[0]))

fa_df = df.copy()
fa_df["Status"] = fa_df.apply(_fa_status, axis=1)
fa_df = fa_df[fa_df["Status"].notna()].copy()

n_ufa = (fa_df["Status"] == "UFA").sum()
n_rfa = (fa_df["Status"] == "RFA").sum()
n_po  = (fa_df["Status"] == "Player Option").sum()
n_to  = (fa_df["Status"] == "Team Option").sum()

# ── Real 2026 signings: join the live accuracy tracker so this same list shows
# each player's actual deal + how the model's projection did, inline. As deals
# get reported (data/real_signings_2026.csv → build_accuracy_tracker.py), the
# "Signed" / "vs Model" columns fill in for the players who've come off the board.
import json as _json
from pathlib import Path as _Path
try:
    _acc = _json.loads((_Path(__file__).parent.parent / "cache" / "accuracy_tracker_v1.json").read_text())
except Exception:
    _acc = None
_signed = {normalize(s["player"]): s
           for s in (_acc or {}).get("signings", []) if s.get("model_M") is not None}
_scorecard = (_acc or {}).get("scorecard") or {}

# Actual 2026 option DECISIONS (opt in/out) for players still in the FA list. Most
# resolutions are derived (a signed Player-Option player necessarily opted out); this
# file only covers players who resolved their option WITHOUT a tracked new deal — i.e.
# opted IN (staying put) or opted OUT and still on the market. data/option_decisions_2026.csv
import csv as _csv
_decisions = {}   # normalized name -> (decision, option figure $M or None)
try:
    with open(_Path(__file__).parent.parent / "data" / "option_decisions_2026.csv") as _fh:
        for _r in _csv.DictReader(l for l in _fh if l.strip() and not l.lstrip().startswith("#")):
            if _r.get("player"):
                try:
                    _fig = float(_r.get("figure_M") or 0) or None
                except ValueError:
                    _fig = None
                _decisions[normalize(_r["player"])] = ((_r.get("decision") or "").strip(), _fig)
except Exception:
    _decisions = {}

_OUTCOME_LABEL = {"po_in": "PO Opt In", "po_out": "PO Opt Out",
                  "to_in": "TO Picked Up", "to_out": "TO Declined"}

def _fa_outcome(name: str, status: str, next_contract_str: str) -> str:
    """What actually happened to this free agent: their option decision and/or a
    signing. Falls back to the pending option figure when nothing has resolved yet."""
    n = normalize(name)
    signed = n in _signed
    dec, fig = _decisions.get(n, (None, None))
    if not dec and signed:                       # signed players reveal their option call
        if status == "Player Option": dec = "po_out"
        elif status == "Team Option": dec = "to_out"
    parts = []
    if dec in _OUTCOME_LABEL:
        label = _OUTCOME_LABEL[dec]
        # opting IN / a team picking up an option = the player stays at that salary,
        # so it's their next-year figure — surface it (else just the decision).
        if dec in ("po_in", "to_in") and fig:
            label += f" · ${fig:.1f}M"
        parts.append(label)
    if signed:
        parts.append("Signed")
    return " · ".join(parts) if parts else next_contract_str

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
        "Status", ["All", "Signed", "UFA", "RFA", "Player Option", "Team Option"], key="fa_status"
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
if fa_status_filter == "Signed":
    fa_display = fa_display[fa_display["Player"].map(normalize).isin(_signed)]
elif fa_status_filter != "All":
    fa_display = fa_display[fa_display["Status"] == fa_status_filter]
if fa_pos_filter != "All":
    fa_display = fa_display[fa_display["position"].str.contains(fa_pos_filter, regex=False, na=False)]
if fa_team_filter != "All":
    fa_display = fa_display[fa_display["Team"] == fa_team_filter]

fa_display = fa_display.sort_values("barrett_score", ascending=False).reset_index(drop=True)

# Outcome = what actually happened to this FA's contract situation (opt in/out + signed),
# replacing the raw next-year figure; falls back to the pending option figure when undecided.
fa_display["outcome"] = fa_display.apply(
    lambda r: _fa_outcome(r["Player"], r["Status"], r["next_contract"]), axis=1)

fa_fmt = fa_display[[
    "Player", "Team", "position", "Status",
    "barrett_score", "salary", "projected_salary", "value_diff", "outcome",
]].copy()

fa_fmt["salary"]           = fa_fmt["salary"] / 1_000_000
fa_fmt["projected_salary"] = fa_fmt["projected_salary"] / 1_000_000
fa_fmt["value_diff"]       = fa_fmt["value_diff"] / 1_000_000

fa_fmt.columns = [
    "Player", "Team", "Pos", "Status",
    "Barrett Score", "Salary", "Proj. Value", "Δ Market", "Outcome",
]
fa_fmt.insert(0, "#", range(1, len(fa_fmt) + 1))

# Real-signing columns: actual first-year salary + how the model's projection did,
# inline for any player who has already signed ("—" while still on the board).
_np_norm = fa_fmt["Player"].map(normalize)
fa_fmt["Signed"]   = _np_norm.map(lambda p: f"${_signed[p]['actual_M']:.1f}M" if p in _signed else "—")
fa_fmt["vs Model"] = _np_norm.map(lambda p: f"{_signed[p]['delta_M']:+.1f}M" if p in _signed else "—")


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

def _sty_outcome(v, _row):
    s = str(v)
    if s == "—":            return "color:var(--fg-6)"
    if "Signed" in s:       return "color:var(--accent-teal);font-weight:700"
    if "Opt In" in s or "Picked Up" in s:  return "color:var(--blue);font-weight:700"      # stayed put
    if "Opt Out" in s or "Declined" in s:  return "color:var(--orange);font-weight:700"    # hit the market
    if " TO" in s:          return "color:var(--orange);font-weight:700"   # pending team option
    if " PO" in s:          return "color:var(--blue);font-weight:700"     # pending player option
    return ""

def _sty_salary(_v, row):
    if normalize(str(row.get("Player", ""))) in _rookie_scale:
        return "color:var(--purple);font-weight:600"
    return ""

def _sty_signed(v, _row):
    # Actual signed deal: highlight teal when present, mute the "still on the board" dash.
    return "color:var(--fg-6)" if str(v) == "—" else "color:var(--accent-teal);font-weight:700"

def _sty_vs_model(v, _row):
    # Model projection minus actual: green within $4M (a hit), red beyond it.
    s = str(v)
    if s == "—":
        return "color:var(--fg-6)"
    try:
        n = float(s.replace("M", "").replace("+", ""))
    except ValueError:
        return ""
    return ("color:var(--value-good);font-weight:700" if abs(n) <= 4
            else "color:var(--value-bad);font-weight:700")

if _scorecard:
    st.caption(
        f"Tracking **{_scorecard['n']}** real 2026 signings in this list. The model's projection "
        f"landed within \\$4M on **{_scorecard['within_4M']}%** of them "
        f"(median miss \\${_scorecard['median_err_M']}M). Set the **Status** filter to **Signed** to "
        "see just those; the **Signed** and **vs Model** columns fill in as deals are reported."
    )

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
        "Outcome":  _sty_outcome,
        "Δ Market": _sty_delta,
        "Salary":   _sty_salary,
        "Signed":   _sty_signed,
        "vs Model": _sty_vs_model,
    },
    aligns={
        "#": "right", "Barrett Score": "right", "Salary": "right",
        "Proj. Value": "right", "Δ Market": "right",
        "Signed": "right", "vs Model": "right",
    },
    numeric={"#", "Barrett Score", "Salary", "Proj. Value", "Δ Market"},
    helps={
        "Barrett Score": "Base Score × Availability Multiplier. Higher = more valuable.",
        "Salary": "Current season salary. Purple = rookie-scale contract (1st-round pick, yrs 1–4).",
        "Proj. Value": "What this player would earn if paid by their Barrett Score rank, a market-rate anchor.",
        "Δ Market": "Actual − Projected. Negative (green) = underpaid; positive (red) = overpaid.",
        "Outcome": "What happened to this free agent: PO Opt In / Opt Out (player option), TO Picked Up / Declined (team option), and/or Signed. Falls back to the pending option figure if undecided.",
        "Status": "UFA = unrestricted · RFA = restricted (right of first refusal) · PO/TO = player/team option.",
        "Signed": "Actual first-year salary of the deal this player signed (real reported 2026 signings). “—” = still on the board.",
        "vs Model": "Our Contract Predictor projection minus the actual first-year salary. Positive = we projected high. Green = within $4M of the real deal.",
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
    # Collapse compound positions (e.g. "PG/SG", "SF/PF") to the primary/first one
    # so the chart has clean PG/SG/SF/PF/C buckets instead of every slash combo.
    _primary = fa_display["position"].astype(str).str.split("/").str[0].str.strip()
    pos_status = (
        pd.DataFrame({"position": _primary.values, "Status": fa_display["Status"].values})
        .groupby(["position", "Status"])
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
            category_orders={"position": ["PG", "SG", "SF", "PF", "C"]},
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
