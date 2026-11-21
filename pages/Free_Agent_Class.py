import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import datetime
import html
import logging
import re
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from utils import (
    COMMON_CSS, SEASONS, DEFAULT_MIN_THRESHOLD,
    normalize, season_to_espn_year,
    build_ranked_projected,
    fetch_bref_positions, fetch_next_year_contracts, fetch_rookie_scale_players,
    _fmt_salary, fmt_next_contract, classify_fa_status, unavailable_players, NameIndex,
    accuracy_tracker, signed_through_season, tracked_signings,
    position_layers,
    color_next_contract, style_rookie_salary, color_value_diff, render_nav, render_page_chrome,
    theme_fig, html_table, render_rail,
    render_barrett_score_explainer, _bootstrap_warm,
    verified_as_of as _utils_verified_as_of
)

logger = logging.getLogger(__name__)

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


# ── Season selector ────────────────────────────────────────────────────────────
# Free agency (next-year contracts, options, signings) only makes sense for the
# CURRENT season — Spotrac's URL has no historical year, so an older season would
# mix today's FA status with stale stat data. So the picker is just the current
# season; it advances on its own (e.g. to 2026-27) once that becomes SEASONS[0].
_FA_SEASONS = SEASONS[:1]
# The class year follows the scored season, so it advances with SEASONS[0]
# instead of leaving the board headed with an offseason that has closed.
_FA_YEAR = int(SEASONS[0][:4]) + 1
ctrl_l, ctrl_mid, ctrl_r = st.columns([1, 1, 1])
with ctrl_l:
    season = st.selectbox("Season", _FA_SEASONS, index=0)
with ctrl_mid:
    st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)  # align with inputs
    include_rookie_opts = st.checkbox(
        "Include rookie-scale options",
        value=False,
        help="Rookie-scale year-3/4 team options are auto-exercised, so these players (Wembanyama, "
             "Amen Thompson, Keyonte George, …) aren't really free agents. Toggle on to list them anyway, "
             "for fun.",
    )
with ctrl_r:
    min_threshold = st.slider(
        "Min total minutes", min_value=0, max_value=1500,
        value=0, step=50,
        help="Default 0 shows the ENTIRE free-agent class (injured/low-minute FAs like Zach "
             "Collins included). Raise it to focus on rotation players. Ranks are always "
             "computed on the full pool.",
    )

# ── Data loading ───────────────────────────────────────────────────────────────
# build_ranked_projected is @st.cache_resource (no copy on hit) — must copy before mutating
df = build_ranked_projected(season)
df = df[df["total_min"] >= min_threshold].copy()

_bref_positions = fetch_bref_positions(season_to_espn_year(season), cache_v=3)
import team_suitors as _ts
_pos2k = _ts.load_player_positions()
# Both layers describe the CURRENT roster, so they are taken through the one
# gate in utils.position_layers rather than applied to whatever season the
# picker is on. On a historical board they do not just read anachronistically,
# they name the WRONG PERSON: NameIndex is suffix-insensitive by design, so
# 1983-84 Rob Williams took Robert Williams III's "C" and 1997-98 Jaren Jackson
# took his son's editorial "PF/C".
#   position_overrides.csv  EDITORIAL, outranks every scraped position
#   master_roster pos       the last resort before group_flex INVENTS one:
#                           group_flex("") returns SF/PF, so a player no
#                           source knew was shown at small forward with
#                           nothing saying so (60 players on 2026-11-13,
#                           four centres and six point guards among them)
_posov_idx, _roster_pos = position_layers(season)
# Curated 2K position (primary + secondary, e.g. "PG/SG"), BBRef coarse fallback.
df["position"] = df["Player"].map(
    lambda n: _posov_idx.get(n) or _ts.resolve_position(
        n, _bref_positions.get(normalize(n), ""), _pos2k, _roster_pos))

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

# ── Real 2026 signings + option decisions, loaded UP-FRONT so the FA list can keep
# tracked signings visible (the salary feed sometimes reports a player's option figure
# as a plain salary, which would otherwise classify them as "under contract" and drop
# them). The same data also powers the Outcome column and the Show filter below.
# It no longer powers a Signed or vs Model column: both went on 2026-11-20.
#
# EVERY tracked signing, not just the ones the model could score. Until 2026-11-21
# this dict was filtered to "model_M is not None", which is the ACCURACY TRACKER's
# membership test and not a test of whether the man signed. 20 of the 131 signings
# carry no projection -- 11 two-ways and 9 others the pool never covered -- and the
# filter made all 20 invisible to the three things that ask "did he sign": the
# Outcome column, Show -> Signed, and Show -> Available. Ten were rows on the board,
# every one of them on a 2026-27 roster, and the page listed them as still on the
# market. The filter existed only to keep a None out of the Signed and vs Model
# columns' arithmetic; those columns went on 2026-11-20 and took its reason with
# them. The accuracy sentence near the table uses _scorecard, which carries its
# own denominator, so nothing on this page needs the scored subset. DO NOT
# re-add the filter.
#
# THIS DICT HAS THREE CONSUMERS, AND ONE OF THEM READS MEMBERSHIP, NOT A FIELD.
# _signing_status and _signed_through_board_year read "type" and "years", which
# every unscored row carries. _fa_outcome reads `n in _signed` and infers an
# option decision from it, so widening the dict widened that inference too --
# which is how Mo Bamba came to be published as having had a team option
# declined that nobody has decided. Gated at _fa_outcome; see the comment there.
# A fourth consumer that reads membership needs the same question asked again.
#
# It lives in utils.tracked_signings because it had been built THREE times:
# here, in tests/test_fa_board_membership.py and in app.py, and the first two
# filtered while app.py did not. A test that rebuilds it tests its own copy.
import json as _json          # still used by _load_pcv and the player hub below
import csv as _csv
from pathlib import Path as _Path
# ONE read of the tracker for both halves. Until 2026-11-21 the scorecard came
# from a fresh read here and the signings from utils' process-lifetime cache, so
# the caption could print two numbers from two versions of one file.
_acc = accuracy_tracker()
_signed = tracked_signings()
_scorecard = (_acc or {}).get("scorecard") or {}

# Annotations belong to the class the BOARD is showing, not to whichever class
# happened to be current when this was written. The board is titled with
# _FA_YEAR, so until 2026-10-22 a board headed "2027 Free Agent Board" carried
# every option decision from 2026, a class that had already closed.
#
# Only the annotations move. data/option_decisions_2026.csv stays the closed
# historical record and still sets 2026-27 salaries through app.py's own
# loader, which is deliberately NOT repointed: the accuracy tracker built from
# the 2026 files outranks every other salary source, and a gap there renders a
# signed player as an unsigned free agent.
_decisions = {}   # normalized name -> (decision, option figure $M or None)
_dec_file = f"option_decisions_{_FA_YEAR}.csv"
try:
    with open(_Path(__file__).parent.parent / "data" / _dec_file) as _fh:
        for _r in _csv.DictReader(l for l in _fh if l.strip() and not l.lstrip().startswith("#")):
            if _r.get("player"):
                try:
                    _fig = float(_r.get("figure_M") or 0) or None
                except ValueError:
                    logger.warning(
                        "%s: unparseable figure_M %r for player %s", _dec_file,
                        _r.get("figure_M"), _r["player"])
                    _fig = None
                _decisions[normalize(_r["player"])] = ((_r.get("decision") or "").strip(), _fig)
except Exception:
    _decisions = {}


def _offseason_as_of() -> str:
    """When the OFFSEASON records were last verified, via utils.verified_as_of.

    This caption claims "signings and option decisions", which are pre-opener
    facts and final once the season starts. master_roster was dropped from the
    group on 2026-11-19 because it keeps moving all season, so letting it drive
    this date asserted a re-check of signings and options that nobody did; the
    roster's own freshness is published on the home page, so it still cannot go
    stale unnoticed. The scan itself used to be a third copy of app.py's.
    """
    return _utils_verified_as_of("offseason")


def _signing_status(name: str) -> str:
    """FA status to show for a tracked 2026 signing the salary feed would otherwise drop."""
    n = normalize(name)
    dec = _decisions.get(n, (None, None))[0]
    if dec in ("po_in", "po_out"): return "Player Option"
    if dec in ("to_in", "to_out"): return "Team Option"
    return "RFA" if (_signed.get(n, {}).get("type") == "rfa") else "UFA"


def _fa_status(row) -> str | None:
    # Shared classifier — cross-checks the contract-end scraper for players the
    # salary feed omits (e.g. Austin Reaves' player option). Only the current
    # season has reliable contract data, so skip the cross-check otherwise.
    return classify_fa_status(row["Player"], row["next_contract"], _rookie_scale,
                              season, cross_check=(season == SEASONS[0]),
                              include_rookie_options=include_rookie_opts)

# ── Is a tracked signing STILL under contract in the board year? ─────────────
# The rescue below exists for a real case: the salary feed sometimes reports an
# option year as a plain guaranteed salary, the classifier then returns None,
# and a player who genuinely reaches the market would vanish off the board.
#
# But it was written when the board year and the tracked signings were the same
# offseason. A season later they are not, and it re-added EVERY tracked signing
# the classifier had correctly excluded -- 35 of 111 on 2026-11-20 -- labelling
# them UFA because that is _signing_status's default. Quentin Grimes signed
# 4yr/$60M with the Lakers in July 2026 and was listed as a UFA on the 2027
# board, from PHI, four months into a contract that runs to 2029-30.
#
# So ask the question the rescue was always assuming: is this player signed
# THROUGH the board year? data/contract_years_2026_27.csv answers it per season
# (G guaranteed, N non-guaranteed, T/P/M option, absent = contract ended), and
# an option year is NOT under contract for this purpose, which is precisely the
# case the rescue protects.
def _signed_through_board_year(name: str) -> bool:
    """Thin wrapper over utils.signed_through_season so the board and its tests
    share one implementation. The fallback term is the tracked signing's own
    length, which is all we know when the contract file has no row for him."""
    try:
        yrs = (_signed.get(normalize(name)) or {}).get("years")
    except Exception:
        yrs = None
    return signed_through_season(name, _FA_YEAR, yrs)


fa_df = df.copy()
fa_df["Status"] = fa_df.apply(_fa_status, axis=1)
# A tracked signing who genuinely reaches the market in the board year is kept
# even when the feed misreads his option figure as a plain salary.
_miss = (fa_df["Status"].isna()
         & fa_df["Player"].map(lambda p: normalize(p) in _signed)
         & ~fa_df["Player"].map(_signed_through_board_year))
if _miss.any():
    fa_df.loc[_miss, "Status"] = fa_df.loc[_miss, "Player"].map(_signing_status)
fa_df = fa_df[fa_df["Status"].notna()].copy()

# This is a FREE AGENT board: drop players who never actually reached the market
# because they opted INTO a player option (po_in) or had a team option PICKED UP
# (to_in). Everyone who genuinely became a free agent stays -- UFAs, RFAs, PO
# opt-outs, TO declines -- whether still available or since re-signed.
_stayed = fa_df["Player"].map(
    lambda p: _decisions.get(normalize(p), (None, None))[0] in ("po_in", "to_in"))
fa_df = fa_df[~_stayed].copy()

n_ufa = (fa_df["Status"] == "UFA").sum()
n_rfa = (fa_df["Status"] == "RFA").sum()
n_po  = (fa_df["Status"] == "Player Option").sum()
n_to  = (fa_df["Status"] == "Team Option").sum()

_OUTCOME_LABEL = {"po_in": "PO Opt In", "po_out": "PO Opt Out",
                  "to_in": "TO Picked Up", "to_out": "TO Declined"}

# "A signed player reveals his option call" is only true on a board showing the
# SAME class the signings belong to. On the 2026 board it held: a player with a
# 2026-27 option who signed a new 2026 deal must have opted out to get there.
#
# On a 2027 board it is backwards. The 2026 signing IS the contract that CONTAINS
# the 2027-28 option, so it says nothing about a decision nobody has made yet --
# and data/option_decisions_2027.csv says so in its own header. Ungated, this
# published "PO Opt Out" or "TO Declined" on 21 live rows, 17 of which
# data/contract_years_2026_27.csv shows holding a live 2027-28 option; every one
# of the 17 is a two-year 2026 deal whose SECOND YEAR IS THAT OPTION. The row
# then contradicts itself: Status "Player Option" beside Outcome "PO Opt Out".
#
# Third instance of one bug class in this file, all found within 36 hours: the
# _miss rescue re-added 2026 signings as 2027 UFAs, the model_M filter read "the
# model could score him" as "he signed", and this read "he signed in 2026" as
# "he declined his 2027 option". Each was correct when the board year and the
# tracked signings were the same offseason.
#
# Fail SAFE: no season in the tracker, or one that does not parse, means no
# inference. A recorded decision in option_decisions_{_FA_YEAR}.csv is unaffected
# -- that is sourced data, and it is what should light this column up in 2027.
try:
    _INFER_OPTION_CALL = int((_acc or {}).get("season")) == _FA_YEAR
except (TypeError, ValueError):
    _INFER_OPTION_CALL = False

# Retirement, joined through NameIndex because these are master_roster spellings
# ("C.J. McCollum") against the board's ("CJ McCollum"). This page is a record of
# the 2026 CLASS, so a retired player KEEPS his row -- he really did become a
# free agent this cycle -- but retirement is what happened to him, and he is not
# "still on the market".
_gone_idx = NameIndex(unavailable_players())


_GONE_LABEL = {"retired": "Retired", "g_league": "G League",
               "overseas": "Overseas", "nba_contract": "Under contract"}


def _gone_label(name: str) -> str | None:
    """What became of him, when it is not a signing: Retired / G League /
    Overseas. This page is a record of the 2026 CLASS, so he KEEPS his row —
    he really did reach free agency this cycle — but that is the outcome, and
    he is not "still on the market"."""
    r = _gone_idx.get(name)
    return _GONE_LABEL.get(r["reason"]) if r else None


def _fa_outcome(name: str, status: str, next_contract_str: str) -> str:
    """What actually happened to this free agent: their option decision and/or a
    signing. Falls back to the pending option figure when nothing has resolved yet."""
    n = normalize(name)
    signed = n in _signed
    dec, fig = _decisions.get(n, (None, None))
    if not dec and signed and _INFER_OPTION_CALL:   # only on the signings' own board
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
    # Last, and on its own if nothing else applies: it is the outcome, not a
    # footnote to one.
    _gl = _gone_label(name)
    if _gl:
        parts.append(_gl)
    return " · ".join(parts) if parts else next_contract_str


# Contract Predictor projection (model pcv) per FA, from the precomputed board cache
# (build_fa_board.py writes value_M = projected_contract_value).
#
# ONE figure on every row. Until 2026-11-20 a player in the accuracy tracker got
# his model_M instead, the projection made at the time of his 2026 signing, so
# that Predicted minus Signed equalled the vs-Model miss. That served the
# vs-Model column, which is gone, and it meant a board headed "2027 Free Agent
# Board" published a 2026-era projection on 61 of its 286 rows and today's on
# the other 225, with nothing marking which was which. They disagreed by up to
# $6.2M: Moritz Wagner read $3.8M where today's model says $10.0M.
def _load_pcv(rel: str) -> dict:
    out = {}
    try:
        d = _json.loads((_Path(__file__).parent.parent / rel).read_text())
        lst = d if isinstance(d, list) else next((v for v in d.values() if isinstance(v, list)), [])
        for p in lst:
            if isinstance(p, dict) and p.get("player") and p.get("value_M") is not None:
                out.setdefault(normalize(p["player"]), float(p["value_M"]))
    except Exception:
        pass
    return out

_pcv_by = {}
_max_norms: set = set()
try:
    # v2 filename (adds contract_end/max_pct/model stamp; players key unchanged):
    # Render's /data disk only seeds MISSING files, so the bump is what deploys it.
    _hub = _json.loads((_Path(__file__).parent.parent / "cache" / "player_hub_pcv_v2.json").read_text())
    _pcv_by = {n: p["pcv_M"] for n, p in _hub.get("players", {}).items() if p.get("pcv_M") is not None}
    _max_norms = {n for n, p in _hub.get("players", {}).items() if p.get("is_max")}
except Exception:
    pass
for _n, _v in _load_pcv("cache/fa_sim_v1.json").items():
    _pcv_by.setdefault(_n, _v)
for _n, _v in _load_pcv("cache/fa_extra_v1.json").items():
    _pcv_by.setdefault(_n, _v)

def _predicted_M(name: str):
    return _pcv_by.get(normalize(name))

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

# How many of the tracked signings are ON this board, not how many the tracker
# holds. Those differ now: the 2026 class was scored on the 2026 board, where
# every signing was a row, and on the 2027 board only the ones whose deal still
# leaves them available are. On 2026-11-20 the rail said "111 signings tracked"
# beside "286 free agents" when 61 of the 111 were on it.
#
# This counts SIGNINGS, so it rose on 2026-11-21 when _signed stopped being
# filtered to the scored subset. It is the same predicate as Show -> Signed, and
# the two must stay equal: the rail is checkable against the filter in one click.
_n_tracked = int(fa_df["Player"].map(lambda p: normalize(p) in _signed).sum())

render_rail(
    "The board", f"{_FA_YEAR} Free Agent Board",
    count=f"{len(fa_df)} free agents",
    meta=f"{_n_tracked} signings tracked" if _n_tracked else None,
)

fa_col_a, fa_col_show, fa_col_b, fa_col_c, fa_col_d = st.columns([2, 1, 1, 1, 1])
with fa_col_a:
    fa_search = st.text_input("Filter by name", "", key="fa_search")
with fa_col_show:
    fa_show = st.selectbox(
        "Show", ["All", "Available", "Signed"], key="fa_show",
        help="Available = still on the market (not signed, and not opted-in / team-exercised into "
             "staying). Signed = came off the board. Combines with Status, e.g. Available + UFA.",
    )
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
    # regex=False: this is raw user text. Left as a regex, a stray "(" or "["
    # raises re.error and takes the whole page down. na=False so a missing name
    # filters out instead of propagating NaN into the mask.
    fa_display = fa_display[
        fa_display["Player"].str.contains(fa_search, case=False, regex=False, na=False)
    ]
# Availability and Status are independent, so they stack (e.g. Available + UFA).
if fa_show == "Signed":
    fa_display = fa_display[fa_display["Player"].map(normalize).isin(_signed)]
elif fa_show == "Available":
    # Still free = hasn't signed AND hasn't opted in / been retained on an option.
    def _still_free(p):
        n = normalize(p)
        if n in _signed:
            return False
        if _gone_label(p):          # retired, in the G League, or playing abroad
            return False
        return _decisions.get(n, (None, None))[0] not in ("po_in", "to_in")
    fa_display = fa_display[fa_display["Player"].map(_still_free)]
if fa_status_filter != "All":
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

# Predicted = the Contract Predictor's model projection (pcv) for every FA.
# Raw numeric here; the gold MAX chip + $ formatting happen in the table formatter so
# click-to-sort stays on the model figure.
#
# The Signed and vs Model columns were removed on 2026-11-20 at Barrett's request.
# They are a 2026 retrospective and this board is the 2027 class: they carried a
# figure on 61 of 286 rows and a dash on the rest. Who already has a deal is still
# on the page, in the Outcome column and the Show filter. Bring them back when 2027
# deals start landing, against 2027 projections.
fa_fmt["Predicted"] = fa_fmt["Player"].map(_predicted_M)


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
    # Colored by OPTION TYPE to match the Status column's language: player options
    # blue, team options orange (whether exercised or declined); signed deals teal.
    s = str(v)
    if s == "—":            return "color:var(--fg-6)"
    if "Signed" in s:       return "color:var(--accent-teal);font-weight:700"
    if s.startswith("PO") or " PO" in s:  return "color:var(--blue);font-weight:700"
    if s.startswith("TO") or " TO" in s:  return "color:var(--orange);font-weight:700"
    return ""

def _sty_salary(_v, row):
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

def _fmt_predicted(v, row):
    # Model figure with the gold MAX chip prefix; the column stays numeric so
    # click-to-sort uses the raw dollar value.
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    chip = ('<span class="hv-chip max">MAX</span>'
            if normalize(str(row.get("Player", ""))) in _max_norms else "")
    return f"{chip}${float(v):.1f}M"

def _sty_predicted(v, _row):
    if v is None or (isinstance(v, float) and v != v):
        return "color:var(--fg-6)"
    return "color:var(--accent-teal)"

# In-cell score bars scaled 15-100 against the VISIBLE rows (FA scores cluster,
# so min-max keeps the bars legible).
_bs_vals = pd.to_numeric(fa_fmt["Barrett Score"], errors="coerce").dropna()
_bs_lo = float(_bs_vals.min()) if len(_bs_vals) else 0.0
_bs_hi = float(_bs_vals.max()) if len(_bs_vals) else 1.0
_bs_rng = (_bs_hi - _bs_lo) or 1.0

def _sty_score(v, _row):
    try:
        pct = max(2.0, min(100.0, float(v) / (_bs_hi or 1.0) * 100))
    except (TypeError, ValueError):
        return ""
    if pct != pct:  # NaN
        return ""
    return f"background:linear-gradient(90deg,var(--bar-tint) {pct:.0f}%,transparent 0)"

if _scorecard:
    # TWO DIFFERENT SETS, so two separate sentences. The percentage is scored over
    # the signings the model PROJECTED and _scorecard carries that denominator;
    # _n_tracked counts the signings on this board, scored or not. Joining them
    # with "N of those players" was true only while _signed was the scored subset,
    # and stopped being true on 2026-11-21.
    st.caption(
        f"The model's 2026 projections landed within \\$4M on "
        f"**{_scorecard['within_4M']}%** of the **{_scorecard['n']}** signings it "
        f"projected (median miss \\${_scorecard['median_err_M']}M). "
        f"**{_n_tracked}** players on this board signed a 2026 deal; set **Show** "
        "to **Signed** to see them, or **Available** for who's still on the market."
    )

_as_of = _offseason_as_of()
if _as_of:
    st.caption(f"Signings and option decisions verified as of {_as_of}.")

html_table(
    fa_fmt,
    formatters={
        "Team":          lambda v: (f'<span class="tdot tdot-{html.escape(str(v), quote=True)}"></span>'
                                    f'{html.escape(str(v))}'),
        "Barrett Score": lambda v: f"{v:.2f}",
        "Salary":        _fmt_salary_rs,
        "Proj. Value":   lambda v: f"${v:.2f}M",
        "Δ Market":      lambda v: f"${v:.2f}M",
        "Predicted":     _fmt_predicted,
    },
    raw={"Team", "Predicted", "Salary"},
    styles={
        "Status":        _sty_status,
        "Outcome":       _sty_outcome,
        "Δ Market":      _sty_delta,
        "Salary":        _sty_salary,
        "Barrett Score": _sty_score,
        "Predicted":     _sty_predicted,
    },
    aligns={
        "#": "right", "Barrett Score": "right", "Salary": "right",
        "Proj. Value": "right", "Δ Market": "right",
        "Predicted": "right",
    },
    numeric={"#", "Barrett Score", "Salary", "Proj. Value", "Δ Market", "Predicted"},
    helps={
        "Barrett Score": "Base Score × Availability Multiplier. Higher = more valuable.",
        "Salary": "Current season salary. Purple with the RS mark = rookie-scale contract (1st-round pick, yrs 1–4).",
        "Proj. Value": "What this player would earn if paid by their Barrett Score rank, a market-rate anchor.",
        "Δ Market": "Actual − Projected. Negative (green) = underpaid; positive (red) = overpaid.",
        "Outcome": "What happened to this free agent: PO Opt In / Opt Out (player option), TO Picked Up / Declined (team option), and/or Signed. Falls back to the pending option figure if undecided.",
        "Status": "UFA = unrestricted · RFA = restricted (right of first refusal) · PO/TO = player/team option.",
        "Predicted": "The Contract Predictor's model projection: what the model thinks this player signs for today. “—” = not precomputed for this player.",
    },
    height=min(820, max(220, len(fa_fmt) * 38 + 46)),
)

# Export CSV removed at Barrett's request (2026-11-01). The caption had a
# column of its own only to sit beside that button, so it goes back to full
# width rather than leaving an empty sixth on its left.
st.caption(
    f"**{len(fa_display)}** free agents shown · "
    "**Proj. Value** = salary of the player at the same Barrett Score rank in the current pool, "
    "a market-rate anchor for what this player should cost. "
    "**Δ Market**: green = underpaid (will demand raise) · red = overpaid (value risk). "
    "Purple salary with the small RS mark = rookie scale."
)

if not fa_display.empty:
    render_rail("The class", "Position breakdown",
                count=f"{len(fa_display)} players",
                meta="stacked by FA status")
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
