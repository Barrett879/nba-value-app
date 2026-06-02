"""Contract Predictor — predict a player's next contract.

Powered by a HistGradientBoostingRegressor (machine-learning model) trained on
1,900+ real contracts from the modern CBA era (2012-13 onward). Features
include trailing-weighted Barrett, prior salary, age, position, service years,
recent All-NBA selections, the rank-based projection, and advanced metrics
(usage rate, PIE, on/off net rating, true shooting). The model output is
post-processed with CBA max-contract cap and supermax floor rules.

Trained only on the modern era on purpose: the goal is predicting CURRENT
contracts, and pre-2012 deals come from a different financial regime (low
cap, old CBA). Trimming them measurably improves recent-season accuracy
(scripts/experiment_recency_window.py).

Accuracy, measured by expanding-window temporal cross-validation on recent
seasons (2021-2025) — train only on prior seasons, predict each subsequent
season the model has never seen. Graded on every real new contract
(minimums, buyouts, and market deals all count); the only exclusion is
rookie-scale step-ups, which aren't new signings — they're the CBA-mandated
next-year salary of a player's existing rookie deal.
  - 89% of predictions within 5% of the cap (~$8M)
  - 99% within 10% of cap (catastrophic misses under 2% of predictions)
  - Median |error|: ~2% of cap
Salary data is sanity-checked: mid-season buyout/waiver artifacts (a star's
prorated near-zero figure after a trade) and verified bad labels are
excluded, since they misrepresent the actual contract.

Every addition was gated on cross-validation, not a single split: advanced
stats earned their place (+1.1pp within-5% on paired CV, t=3.9), while a
two-stage classify-then-snap model and a stacked ensemble were within noise
and dropped. We ship only what the rigorous eval confirms.

See scripts/build_production_histgbm.py, confirm_advanced_features.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_searchbox import st_searchbox

from utils import (
    COMMON_CSS, SEASONS, normalize, season_to_espn_year,
    get_all_player_names, fetch_player_full_career,
    build_ranked_projected, fetch_league_stats, fetch_advanced_stats,
    fetch_bref_positions, fetch_player_positions_detailed, position_to_bucket,
    render_nav, render_page_chrome, render_barrett_score_explainer, _bootstrap_warm,
    html_table,
    # Calibration constants — single source of truth in utils
    SALARY_CAP_M, cap_dollars,
    CONTRACT_POSITION_MULTIPLIERS as POSITION_MULTIPLIERS,
    HEALTHY_SEASON_GP, NEW_CONTRACT_PCT, SUPERMAX_CAP_PCT,
    tiered_age_multiplier, durability_multiplier, playoff_bonus_multiplier,
    # Draft tier — used to keep comparables apples-to-apples (lottery picks
    # earn on pedigree; non-lottery developers don't).
    DRAFT_TIERS, DRAFT_TIER_ORDINAL,
    get_player_draft_info, build_draft_tier_lookup,
    # CBA / contract structure
    get_max_contract_eligibility,
    fetch_rookie_scale_players, fetch_next_year_contracts, fmt_next_contract,
    fetch_all_nba_selections, get_all_nba_in_window,
    # Contract end-year scraper — powers the "Current deal: $X through YYYY-YY"
    # context line under the hero (and nothing else after the forward-
    # projection revert).
    get_player_contract_info,
)


CURRENT_SEASON = SEASONS[0]            # latest real season — source of the player's stats
# A new contract signed "today" starts NEXT season, so we price the prediction
# in next-season cap dollars. Stats come from CURRENT_SEASON; the contract is
# valued at CONTRACT_SEASON's cap. Auto-advances with SEASONS.
_cs_start = int(CURRENT_SEASON[:4]) + 1
CONTRACT_SEASON = f"{_cs_start}-{(_cs_start + 1) % 100:02d}"

# Max-tier floor snaps eligible All-NBA stars to the EMPIRICAL eligible-star
# level (max − 3pp of cap), not the theoretical max — corrects a measured ~2pp
# top-tier overshoot (diag_residuals.py / test_floor_discount.py). The
# supermax floor stays at the full max (Designated Vets sign the full supermax).
MAX_FLOOR_DISCOUNT = 0.03


# ── Page boilerplate ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Contract Predictor", layout="wide")
render_page_chrome()
_bootstrap_warm()
render_nav("Contract Predictor")

st.title("Contract Predictor")
st.caption(
    f"Type a player's name to see what they'd command on a NEW contract signed "
    f"today, i.e. their {CONTRACT_SEASON} salary, at next season's projected cap."
)
# Spacer so the caption→searchbar gap matches the searchbar→info-box gap below
# (the caption otherwise sits flush against the search box).
st.markdown("<div style='height:21px'></div>", unsafe_allow_html=True)

# Methodology expanders live at the bottom of the page (after the prediction
# and comparables) so the page leads with the answer, not the methodology.


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fmt_money(v: float) -> str:
    if pd.isna(v) or v == 0:
        return "—"
    return f"${v / 1_000_000:.1f}M"


# Single-letter position display. The detailed BBRef scrape returns
# PG/SG/SF/PF/C; the legacy ESPN coarse scrape returns Guard/Forward/Center.
# When the detailed lookup misses a player (Sengun, some rookies) we fall
# back to coarse — but mixing "C" and "Center" in the same comps table
# looks broken. Normalize the coarse names to their single-letter equivalent.
_COARSE_TO_LETTER = {"Guard": "G", "Forward": "F", "Center": "C"}


def _pos_abbrev(pos: str) -> str:
    """Return the abbreviated position. PG/SG/SF/PF/C pass through; the
    coarse 'Guard'/'Forward'/'Center' fallback maps to G/F/C."""
    if not pos:
        return "—"
    return _COARSE_TO_LETTER.get(pos, pos)


def _fmt_draft(features: dict) -> str | None:
    """Short draft label for the metadata line.

    Drafted players: "Pick #3 (2018)" — the pick number already implies the
    tier (1-14 lottery, 15-30 first-round, etc.), so we drop the tier label.
    Undrafted players with a record: "Undrafted".
    Players with no draft data: None (suppressed entirely).
    """
    pick = features.get("draft_pick")
    year = features.get("draft_year")
    tier = features.get("draft_tier")
    if pick:
        year_str = f" ({year})" if year else ""
        return f"Pick #{pick}{year_str}"
    if tier == "Undrafted":
        return "Undrafted"
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_player_features(player_name: str, season: str = CURRENT_SEASON) -> dict | None:
    ranked = build_ranked_projected(season)
    if ranked.empty:
        return None
    name_norm = normalize(player_name)
    mask = ranked["Player"].apply(normalize) == name_norm
    if not mask.any():
        return None
    row = ranked[mask].iloc[0]

    raw = fetch_league_stats(season, "Regular Season")
    age = None
    if not raw.empty and "AGE" in raw.columns:
        age_lookup = dict(zip(raw["PLAYER_ID"], raw["AGE"]))
        age = age_lookup.get(int(row["PLAYER_ID"]))

    # Advanced stats (usage rate, PIE, on/off ratings, true shooting) — these
    # carry signal the box-score composite alone doesn't; paired CV confirmed
    # +1.12pp within-5% (t=3.9). Order MUST match _HISTGBM_ADV_COLS.
    adv_feats = {c: 0.0 for c in _HISTGBM_ADV_COLS}
    try:
        adv = fetch_advanced_stats(season, "Regular Season")
        if not adv.empty:
            arow = adv[adv["PLAYER_ID"] == int(row["PLAYER_ID"])]
            if not arow.empty:
                adv_feats = {c: float(arow.iloc[0].get(c, 0) or 0)
                             for c in _HISTGBM_ADV_COLS}
    except Exception:
        pass

    # Try the detailed BBRef per-game scrape first (PG/SG/SF/PF/C, better
    # coverage). Fall back to the older ESPN-salary scrape (G/F/C only),
    # then to Unknown.
    detailed_pos = "Unknown"
    try:
        detailed_lookup = fetch_player_positions_detailed(season, cache_v=3)
        detailed_pos = detailed_lookup.get(name_norm, "Unknown")
    except Exception:
        detailed_lookup = {}
    if detailed_pos == "Unknown":
        try:
            pos_lookup = fetch_bref_positions(season_to_espn_year(season), cache_v=3)
        except Exception:
            pos_lookup = {}
        coarse_fallback = pos_lookup.get(name_norm, "Unknown")
        # pos_bucket stays as the full "Guard"/"Forward"/"Center" name
        # because POSITION_MULTIPLIERS is keyed on those strings.
        # detailed_display normalizes to the single-letter abbreviation
        # to match the PG/SG/SF/PF/C style used everywhere else.
        pos_bucket = coarse_fallback
        detailed_display = _pos_abbrev(coarse_fallback)
    else:
        pos_bucket = position_to_bucket(detailed_pos)
        detailed_display = detailed_pos  # already PG/SG/SF/PF/C
    pos = pos_bucket  # use the 3-bucket for the multiplier (preserves fit)

    # ── Career-weighted RATE Score — the GM's view ──────────────────────────
    # Two important things going on here:
    #
    # 1. Career-weighted, not single-season. Real front offices project
    #    contracts off a body of work — not a half season. Weight the last
    #    three healthy seasons 50/30/20 (most recent first).
    #
    # 2. RATE score (no availability multiplier). The site-wide Barrett
    #    Score includes an availability multiplier that down-weights
    #    players who miss games. That's correct for ranking "who provided
    #    the most total value this season" — but WRONG for contract
    #    projection. GMs negotiate AAV based on rate stats (what you
    #    produce when on the floor); they treat durability as a SEPARATE
    #    concern handled via contract length + structure, not AAV.
    #
    #    Concretely: Curry played 41 games in 2025-26. His Barrett Score
    #    is ~26 (with availability multiplier ~0.83). His RATE score —
    #    what he produces per game when healthy — is ~31. GMs project
    #    his next AAV off ~31, not ~26.
    #
    # Rate score = Barrett Score / Avail multiplier. (Equivalent to
    # base_score_pace before the availability multiplier was applied.)
    career_barrett = None
    trailing_barrett = None  # same weighting but with availability — used for comp matching
    barrett_3yr_simple = None  # simple 3-yr mean (HistGBM feature)
    gp_3yr_simple      = None  # simple 3-yr GP mean (HistGBM feature)
    career_basis = "current season (no prior data)"
    try:
        career = fetch_player_full_career(player_name)
        if not career.empty:
            # Derive the rate score (un-availability-adjusted) from existing
            # career columns. Floor avail at 0.3 to avoid divide-by-zero
            # for any edge-case row that landed at zero availability.
            career = career.copy()
            career["Barrett Rate"] = (
                career["Barrett Score"]
                / career["Avail"].clip(lower=0.30)
            )

            # Apply the GP ≥ 40 filter (still useful — keeps tiny-sample
            # seasons out of the pool — but the avail deflation is no
            # longer compounding on top of it).
            healthy = career[career["GP"] >= HEALTHY_SEASON_GP]
            used_healthy_filter = len(healthy) >= 1
            pool = healthy if used_healthy_filter else career

            if not pool.empty:
                recent = pool.tail(3)
                weights_full = [0.20, 0.30, 0.50]
                weights = weights_full[-len(recent):]
                w_sum = sum(weights)
                career_barrett = float(
                    (recent["Barrett Rate"].values * weights).sum() / w_sum
                )
                # Same 50/30/20 weighting but on raw Barrett Score (with
                # availability). This is what `find_comparables` uses to
                # match against historical comps' walk-year scores —
                # apples-to-apples, since the walk year is also a single
                # availability-included season. Smooths out one-off bad
                # years (Vassell's injury 2025-26) without erasing the
                # availability signal the way the rate score does.
                trailing_barrett = float(
                    (recent["Barrett Score"].values * weights).sum() / w_sum
                )
                # Simple 3-yr means for the HistGBM model (uses unweighted
                # means as features, separate from the weighted ones above).
                barrett_3yr_simple = float(recent["Barrett Score"].mean())
                gp_3yr_simple      = float(recent["GP"].mean())
                seasons_used = list(recent["Season"].values)
                skipped = (
                    used_healthy_filter and len(career) > len(pool)
                )
                # Use &lt; instead of literal < so the string is safe to
                # interpolate into the math-line HTML. Streamlit's parser
                # otherwise treats `<40` as the start of an HTML tag and
                # corrupts everything after it.
                skip_note = (
                    " · low-GP seasons (&lt;40) skipped" if skipped else ""
                )
                career_basis = (
                    f"rate-score weighted avg of {len(recent)} healthy season"
                    f"{'s' if len(recent) > 1 else ''} "
                    f"({', '.join(seasons_used)}){skip_note}"
                )
    except Exception:
        pass

    # Fall back to current-season scores if no career history.
    if career_barrett is None:
        cur_avail = float(row.get("avail_mult", 1.0) or 1.0)
        cur_avail = max(cur_avail, 0.30)  # same clip as above
        career_barrett = float(row["barrett_score"]) / cur_avail
        career_basis = "current season rate only (rookie / first appearance)"
    if trailing_barrett is None:
        trailing_barrett = float(row["barrett_score"])

    # Effective rank: compare this player's RATE score to the rate scores
    # of every current-season player (also un-availability-adjusted) so
    # the rank-to-salary mapping is apples-to-apples. Using barrett_score
    # for both would mean comparing a healthy player to availability-
    # deflated ones, which inflates the rank artificially.
    if "avail_mult" in ranked.columns:
        ranked_avail = ranked["avail_mult"].clip(lower=0.30)
        cur_rate_scores = (ranked["barrett_score"] / ranked_avail).values
    else:
        cur_rate_scores = ranked["barrett_score"].values
    cur_rate_scores = np.sort(cur_rate_scores)[::-1]  # descending
    cur_salaries = ranked["salary"].sort_values(ascending=False).values
    effective_rank = int((cur_rate_scores > career_barrett).sum()) + 1
    capped_rank = min(effective_rank, len(cur_salaries)) - 1
    career_base_proj = float(cur_salaries[capped_rank])

    # Durability: separate from production rate. Curry's anomaly 41-GP
    # year shouldn't drag him into the chronic tier, but Embiid's
    # multi-season pattern of ~30 GP should heavily discount his
    # projected AAV.
    try:
        full_career = fetch_player_full_career(player_name)
        dur_mult, dur_tier, dur_avail = durability_multiplier(
            full_career, lookback_seasons=3,
        )
        trailing_gp_total = int(full_career.tail(3)["GP"].sum())
        trailing_gp_max   = min(len(full_career), 3) * 82
    except Exception:
        dur_mult, dur_tier, dur_avail = 1.0, "no career data", 0.0
        trailing_gp_total = 0
        trailing_gp_max = 0

    # Playoff bonus — proven playoff performers (Jokic, Tatum, SGA) earn
    # a premium at the negotiating table beyond what regular-season
    # production suggests. One-way: bonus only, no penalty for players on
    # tanking teams who can't get postseason reps.
    try:
        playoff_career = fetch_player_full_career(player_name, playoffs=True)
        playoff_mult, playoff_tier, playoff_barrett_val, playoff_gp = (
            playoff_bonus_multiplier(playoff_career, lookback_seasons=3)
        )
    except Exception:
        playoff_mult, playoff_tier, playoff_barrett_val, playoff_gp = (
            1.0, "No playoff data", 0.0, 0
        )

    # Draft tier — lottery picks signal earning ceiling beyond what production
    # rank implies; 2nd-round / undrafted developers don't get that bump.
    # Used by find_comparables to keep the comp pool apples-to-apples.
    draft_info = get_player_draft_info(player_name)

    # CBA max-contract eligibility — service years + team tenure + recent
    # All-NBA. Drives the cap/floor logic in predict_contract: caps the
    # projection at the player's CBA max %, and floors supermax-eligible
    # stars at their Designated max (35% or 30% of cap).
    try:
        elig = get_max_contract_eligibility(player_name, season)
    except Exception:
        elig = {
            "service_years": 0, "team_tenure": 0, "current_team": "",
            "recent_all_nba": [], "qualifying": False,
            "max_pct": 0.35, "supermax_tier": "Max 35%",
        }

    # Rookie scale lock — they CANNOT sign a new market deal until the
    # rookie deal expires. This is a hard CBA constraint.
    try:
        rookie_scale_set = fetch_rookie_scale_players(season)
        on_rookie_scale = name_norm in rookie_scale_set
    except Exception:
        on_rookie_scale = False

    # ── All-NBA count in trailing 3 seasons (HistGBM feature) ──────────────
    # 1+ recent All-NBA is a leading indicator of max-tier contracts —
    # the HistGBM uses it to upgrade projections for proven elites.
    try:
        recent_all_nba_list = get_all_nba_in_window(player_name, season, window_seasons=3)
        all_nba_3yr_count = len(recent_all_nba_list)
    except Exception:
        all_nba_3yr_count = 0

    # Contract end info — kept for display only. The prediction is "what
    # would this player sign for if going to their GM TODAY based on
    # current performance" — projecting forward 3-5 years with assumed
    # cap growth + assumed future All-NBA introduces too much speculation
    # to be useful. See get_player_contract_info for the underlying data.
    try:
        contract_info = get_player_contract_info(player_name)
    except Exception:
        contract_info = None

    return {
        "name":               row["Player"],
        "team":               row.get("Team", ""),
        "age":                age,
        "position":           pos,                # G/F/C — drives multiplier
        "position_detailed":  detailed_display,   # PG/SG/SF/PF/C — for display
        "barrett_score":      float(row["barrett_score"]),       # current season
        "career_barrett":     career_barrett,                    # 3-year weighted RATE (rank-mapping)
        "trailing_barrett":   trailing_barrett,                  # 3-year weighted with availability (comp matching)
        "career_basis":       career_basis,
        "score_rank":         int(row["score_rank"]),
        "effective_rank":     effective_rank,
        "career_base_proj":   career_base_proj,
        "durability_mult":    dur_mult,
        "durability_tier":    dur_tier,
        "durability_avail":   dur_avail,
        "trailing_gp_total":  trailing_gp_total,
        "trailing_gp_max":    trailing_gp_max,
        "playoff_mult":       playoff_mult,
        "playoff_tier":       playoff_tier,
        "playoff_barrett":    playoff_barrett_val,
        "playoff_gp":         playoff_gp,
        "draft_tier":         draft_info["draft_tier"],
        "draft_pick":         draft_info["draft_pick"],
        "draft_year":         draft_info["draft_year"],
        # CBA / contract structure
        "service_years":      elig["service_years"],
        "team_tenure":        elig["team_tenure"],
        "current_team":       elig["current_team"],
        "recent_all_nba":     elig["recent_all_nba"],
        # CBA eligibility AT TODAY's service/tenure — the "if signing
        # right now, what's your CBA max?" question.
        "supermax_eligible":  elig["qualifying"] and elig["supermax_tier"] in
                              ("Designated Vet (35%)", "Designated Rookie (30%)"),
        "max_pct":            elig["max_pct"],
        "supermax_tier":      elig["supermax_tier"],
        "on_rookie_scale":    on_rookie_scale,
        # Contract end info — kept for the UI "current deal: through X"
        # informational note. NOT used to project the prediction forward.
        "contract_end_season": (contract_info or {}).get("end_season"),
        "contract_last_year_type": (contract_info or {}).get("last_year_type"),
        "salary":             float(row.get("salary", 0) or 0),
        "projected_salary":   float(row.get("projected_salary", 0) or 0),
        "gp":                 int(row.get("GP", 0) or 0),
        "mpg":                float(row.get("MPG", 0) or 0),
        # HistGBM model inputs (single-source of truth for what the
        # model needs at prediction time).
        "barrett_3yr_simple": barrett_3yr_simple,
        "gp_3yr_simple":      gp_3yr_simple,
        "eff_adj":            float(row.get("efficiency_adj", 0) or 0),
        "d_lebron":           float(row.get("d_lebron", 0) or 0),
        "all_nba_3yr":        all_nba_3yr_count,
        "adv_feats":          adv_feats,   # USG/PIE/NET/TS/AST%/REB%
        "total_pool_size":    len(ranked),
    }


# ── HistGBM contract predictor ───────────────────────────────────────────────
# Single HistGradientBoostingRegressor trained on the modern CBA era only
# (2012-13 onward, ~1,900 contracts) — pre-2012 deals are a different financial
# regime and hurt current-season prediction (experiment_recency_window.py).
# Features: Barrett pruned set + advanced stats (usage/PIE/on-off/TS), the
# latter confirmed a real +1.12pp within-5% gain by paired CV (t=3.9).
# Temporal CV on recent seasons (2021-2025), graded on all real new contracts
# (rookie-scale step-ups + buyout-artifact/bad-label rows excluded): 89%
# within 5% of cap, 99% within 10% — incl. the All-NBA max-tier floor
# (apply_cba_postprocess), forward-validated at +1.07pp. A two-stage model +
# stacked ensemble were tested and came in within noise under CV.
# See scripts/build_production_histgbm.py and scripts/test_floor_forward.py.
_HISTGBM_PATH = Path(__file__).parent.parent / "models" / "contract_histgbm_v2.joblib"

# Advanced-stat feature order — MUST match build_production_histgbm.ADV_COLS.
_HISTGBM_ADV_COLS = ["USG_PCT", "PIE", "NET_RATING", "TS_PCT", "AST_PCT", "REB_PCT"]


@st.cache_resource(show_spinner=False)
def _load_histgbm():
    """Load the production HistGBM artifact once and cache it for the session.
    Returns the artifact dict {'model', 'feature_cols', ...} or None if the
    file is missing — in which case predict_contract falls back to the old
    rank-mapping formula."""
    try:
        import joblib
        if not _HISTGBM_PATH.exists():
            return None
        return joblib.load(_HISTGBM_PATH)
    except Exception:
        return None


def _histgbm_feature_vector(features: dict, target_season: str) -> np.ndarray | None:
    """Build the 28-feature input vector that the HistGBM expects, in the
    exact order the model was trained on (PRUNED_FEATURES + 8 derived + 6 advanced).
    Returns None if essential features are missing."""
    cap_dollars_val = SALARY_CAP_M.get(target_season, 154.6) * 1_000_000

    barrett        = float(features.get("career_barrett") or 0)
    barrett_single = float(features.get("barrett_score") or 0)
    # Fall back to trailing-weighted Barrett if we don't have a simple 3-yr.
    barrett_3yr    = float(features.get("barrett_3yr_simple")
                            or features.get("career_barrett") or barrett_single)
    score_rank     = float(features.get("score_rank") or 999)
    eff_adj        = float(features.get("eff_adj") or 0)
    d_lebron       = float(features.get("d_lebron") or 0)
    gp             = float(features.get("gp") or 0)
    gp_3yr         = float(features.get("gp_3yr_simple") or gp)
    age            = float(features.get("age") or 25)
    salary_prev_pct = float(features.get("salary") or 0) / cap_dollars_val
    career_base_proj_pct = float(features.get("career_base_proj") or 0) / cap_dollars_val
    years_in_league = float(features.get("service_years") or 0)
    all_nba_3yr    = float(features.get("all_nba_3yr") or 0)
    barrett_growth = (barrett_single / barrett_3yr) if barrett_3yr > 0 else 1.0

    pos_bucket = features.get("position", "Unknown")
    is_g = 1.0 if pos_bucket == "Guard"   else 0.0
    is_f = 1.0 if pos_bucket == "Forward" else 0.0

    # Advanced stats, appended in the exact training order (ADV_COLS).
    adv = features.get("adv_feats") or {}
    adv_vals = [float(adv.get(c, 0) or 0) for c in _HISTGBM_ADV_COLS]

    # Same order as build_production_histgbm.make_X_augmented:
    # PRUNED_FEATURES, then derived, then advanced stats.
    return np.array([[
        barrett, barrett_single, barrett_3yr,
        score_rank,
        eff_adj, d_lebron,
        gp, gp_3yr,
        age,
        salary_prev_pct, career_base_proj_pct,
        years_in_league,
        all_nba_3yr, barrett_growth,
        # Derived:
        age ** 2, barrett ** 2, np.log1p(score_rank),
        years_in_league ** 2,
        1.0 if years_in_league >= 7  else 0.0,
        1.0 if years_in_league >= 10 else 0.0,
        is_g, is_f,
        # Advanced (USG/PIE/NET/TS/AST%/REB%):
        *adv_vals,
    ]])


def _relative_band_dollars(predicted_dollars: float) -> float:
    """Confidence half-width as a % of the prediction — tier-aware. The model
    nails star/max deals (~10% typical relative error) and is noisier on mid-
    and minimum-level deals (~25–40%), so the band scales WITH the prediction
    rather than a flat ±X%-of-cap. (The old flat ±$6M band made a $3M projection
    read as "anywhere from $0 to $9M".) The %s mirror the median relative error
    by tier measured in the honest temporal CV — see the About expander."""
    pred_M = max(predicted_dollars, 0.0) / 1e6
    band_pct = float(np.interp(pred_M, [2.0, 8.0, 20.0, 45.0],
                                        [0.45, 0.33, 0.24, 0.12]))
    return predicted_dollars * band_pct


# NBA minimum salary scale by years of service, as a fraction of the cap (the
# published 2025-26 minimum table ÷ the 2025-26 cap of $154.6M). The CBA grows
# minimums with the cap, so the fraction is stable across seasons and auto-
# scales to any season's cap. Replaces the old flat 1.5%-of-cap floor, which
# only matched the ~2-yr minimum — it overstated the floor for rookies (a 1-yr
# min ≈ 1.3% of cap, not 1.5%) and understated it for vets (10+ yr ≈ 2.4%).
_MIN_SALARY_PCT_BY_SERVICE = (
    0.0082, 0.0133, 0.0149, 0.0154, 0.0159,   # 0–4 yrs
    0.0173, 0.0186, 0.0199, 0.0213, 0.0214,   # 5–9 yrs
)
_MIN_SALARY_PCT_10PLUS = 0.0235               # 10+ yrs


def min_salary_pct(service_years) -> float:
    """League-minimum salary as a fraction of the cap for a player with this
    many years of service (flat tail at 10+ yrs). Used as the model's lower
    clip and the displayed floor so a rookie isn't floored at a vet's minimum."""
    try:
        s = int(service_years or 0)
    except (TypeError, ValueError):
        s = 0
    if s < 0:
        s = 0
    return _MIN_SALARY_PCT_10PLUS if s >= 10 else _MIN_SALARY_PCT_BY_SERVICE[s]


def predict_contract_histgbm(features: dict, target_season: str = CONTRACT_SEASON,
                             stats_season: str = CURRENT_SEASON) -> dict | None:
    """Predict the contract a player would sign TODAY (a new deal starting
    `target_season`) from their `stats_season` production.

    Two different caps are in play, matching how the model was trained
    (prior-season stats → next-season contract as % of cap):
      - the feature vector normalizes the player's PRIOR salary by the
        stats-season cap (CURRENT_SEASON),
      - the predicted % of cap is converted to dollars at the CONTRACT-season
        cap (next season — where the new deal actually starts).
    Returns None if the HistGBM model isn't loadable (caller falls back).
    """
    artifact = _load_histgbm()
    if artifact is None:
        return None
    X = _histgbm_feature_vector(features, stats_season)   # prior salary ÷ stats-season cap
    if X is None:
        return None
    model = artifact["model"]
    # Floor at the player's service-scaled CBA minimum (a 1-yr min ≈ 1.3% of
    # cap, a 10+-yr min ≈ 2.4%) so the model can't emit a sub-minimum figure
    # for an established player whose trailing production cratered (the Clarkson
    # floor-glitch), nor a vet-minimum for a rookie; cap at the 35% absolute max.
    min_pct = min_salary_pct(features.get("service_years"))
    pred_pct = float(np.clip(model.predict(X)[0], min_pct, 0.35))

    # Convert to dollars at NEXT season's cap — the deal would start then.
    cap_dollars_val = SALARY_CAP_M.get(target_season, 165.0) * 1_000_000
    raw_predicted = pred_pct * cap_dollars_val
    base = raw_predicted  # for display compatibility with predict_contract

    # ── CBA cap-and-floor (structural rules the model can't see exactly) ────
    max_pct = float(features.get("max_pct", 0.35) or 0.35)
    cba_max_dollars = cap_dollars_val * max_pct
    supermax_eligible = bool(features.get("supermax_eligible", False))
    supermax_tier_label = features.get("supermax_tier", "")

    predicted = raw_predicted
    cba_cap_applied = False
    cba_floor_applied = False        # supermax floor → full max
    max_tier_floor_applied = False   # All-NBA near-max floor → max − 3pp

    if predicted > cba_max_dollars:
        predicted = cba_max_dollars
        cba_cap_applied = True

    target_age = features.get("age")
    in_prime = target_age is not None and target_age <= 32
    if supermax_eligible and in_prime and predicted < cba_max_dollars:
        predicted = cba_max_dollars
        cba_floor_applied = True

    # Max-tier floor: a recent-All-NBA star the model rates near-max (>=20% of
    # cap) and aged ≤33 gets lifted toward their max tier — but to the EMPIRICAL
    # eligible-star level (max − 3pp), not the theoretical max. The regressor
    # hedges below the max (stars got a spread of training outcomes); the floor
    # corrects that, while the 3pp discount corrects a measured ~2pp overshoot
    # from snapping discount-takers all the way to the max. Age gate spares
    # aging stars (Chris Paul 36). Forward-validated: All-NBA within-5% 82→86%.
    _floor_target = cba_max_dollars - MAX_FLOOR_DISCOUNT * cap_dollars_val
    if ((features.get("all_nba_3yr", 0) or 0) >= 1
            and raw_predicted >= 0.20 * cap_dollars_val
            and (target_age is None or target_age <= 33)
            and predicted < _floor_target):
        predicted = _floor_target
        max_tier_floor_applied = True

    band = _relative_band_dollars(predicted)

    return {
        "base":                 base,
        "age_mult":             1.0,  # baked into the model
        "age_tier":             "Model-internal",
        "pos_mult":             1.0,
        "pos_mult_raw":         1.0,
        "pos_mult_suppressed":  False,
        "durability_mult":      1.0,
        "durability_tier":      features.get("durability_tier", ""),
        "playoff_mult":         1.0,
        "playoff_tier":         features.get("playoff_tier", ""),
        "raw_predicted":        raw_predicted,
        "predicted":            predicted,
        "low":                  max(min_pct * cap_dollars_val, predicted - band),
        "min_floor_dollars":    min_pct * cap_dollars_val,
        "high":                 min(predicted + band, cba_max_dollars),
        "band":                 band,
        "cap":                  cap_dollars_val,
        "max_pct":              max_pct,
        "cba_max_dollars":      cba_max_dollars,
        "cba_cap_applied":      cba_cap_applied,
        "cba_floor_applied":    cba_floor_applied,
        "max_tier_floor_applied": max_tier_floor_applied,
        "supermax_eligible":    supermax_eligible,
        "supermax_tier_label":  supermax_tier_label,
        "model_used":           "HistGBM v2",
    }


def predict_contract(features: dict, target_season: str = CONTRACT_SEASON,
                     stats_season: str = CURRENT_SEASON) -> dict:
    # HistGBM model (machine-learning regression on ~1,900 modern-era
    # contracts, 2012+). Falls back to the legacy rank-mapping +
    # multipliers formula if the model artifact isn't available.
    hist = predict_contract_histgbm(features, target_season, stats_season)
    if hist is not None:
        return hist

    # Predicts "what would this player sign for TODAY based on current
    # performance, current cap, and current CBA eligibility." Doesn't
    # project forward to the actual signing year — too much speculation
    # (cap growth, future All-NBA, future tenure) makes a number 3-5
    # years out hard to interpret. Single defensible question: "if they
    # walked into their GM today and asked what they're worth, what
    # would the GM offer?"
    base = float(features["career_base_proj"])

    # Tiered age multiplier — uses CURRENT age.
    age_mult, age_tier = tiered_age_multiplier(
        age=features.get("age"),
        career_score=features.get("career_barrett", 0),
        current_rank=features.get("effective_rank"),
    )

    pos_mult = POSITION_MULTIPLIERS.get(features["position"], 1.0)

    cap_dollars_val = SALARY_CAP_M.get(target_season, 165.0) * 1_000_000
    supermax_threshold = cap_dollars_val * SUPERMAX_CAP_PCT

    # Position suppression at supermax tier:
    # Position multipliers were fit on mid-market signings (Centers get
    # systematically less than box score suggests). But supermax/max-
    # contract players sign at fixed CBA percentages regardless of
    # position. When base ≥ 28% of cap, drop the positional discount.
    pos_mult_applied = pos_mult
    if base >= supermax_threshold:
        pos_mult_applied = 1.0

    # Durability discount — based on past data.
    dur_mult = float(features.get("durability_mult", 1.0) or 1.0)
    dur_tier = features.get("durability_tier", "")

    # Playoff bonus — based on past playoff data.
    playoff_mult = float(features.get("playoff_mult", 1.0) or 1.0)
    playoff_tier = features.get("playoff_tier", "")

    raw_predicted = base * age_mult * pos_mult_applied * dur_mult * playoff_mult
    # Floor the legacy estimate at the player's service-scaled CBA minimum, to
    # match the HistGBM path (and keep min_floor_dollars below consistent).
    min_pct = min_salary_pct(features.get("service_years"))
    raw_predicted = max(raw_predicted, min_pct * cap_dollars_val)

    # ── CBA cap-and-floor adjustments (today's eligibility) ─────────────────
    max_pct = float(features.get("max_pct", 0.35) or 0.35)
    cba_max_dollars = cap_dollars_val * max_pct
    supermax_eligible = bool(features.get("supermax_eligible", False))
    supermax_tier_label = features.get("supermax_tier", "")

    predicted = raw_predicted
    cba_cap_applied = False
    cba_floor_applied = False        # supermax floor → full max
    max_tier_floor_applied = False   # All-NBA near-max floor → max − 3pp

    if predicted > cba_max_dollars:
        predicted = cba_max_dollars
        cba_cap_applied = True

    # Supermax floor — uses CURRENT age. Aging stars (Curry 38, LeBron 40)
    # who qualify technically but routinely take paycuts don't get the
    # floor lift.
    target_age = features.get("age")
    in_prime = target_age is not None and target_age <= 32
    if supermax_eligible and in_prime and predicted < cba_max_dollars:
        predicted = cba_max_dollars
        cba_floor_applied = True

    # Max-tier floor: a recent-All-NBA star the model rates near-max (>=20% of
    # cap) and aged ≤33 gets lifted toward their max tier — but to the EMPIRICAL
    # eligible-star level (max − 3pp), not the theoretical max. The regressor
    # hedges below the max (stars got a spread of training outcomes); the floor
    # corrects that, while the 3pp discount corrects a measured ~2pp overshoot
    # from snapping discount-takers all the way to the max. Age gate spares
    # aging stars (Chris Paul 36). Forward-validated: All-NBA within-5% 82→86%.
    _floor_target = cba_max_dollars - MAX_FLOOR_DISCOUNT * cap_dollars_val
    if ((features.get("all_nba_3yr", 0) or 0) >= 1
            and raw_predicted >= 0.20 * cap_dollars_val
            and (target_age is None or target_age <= 33)
            and predicted < _floor_target):
        predicted = _floor_target
        max_tier_floor_applied = True

    band = _relative_band_dollars(predicted)

    return {
        "base":                 base,
        "age_mult":             age_mult,
        "age_tier":             age_tier,
        "pos_mult":             pos_mult_applied,
        "pos_mult_raw":         pos_mult,
        "pos_mult_suppressed":  pos_mult_applied != pos_mult,
        "durability_mult":      dur_mult,
        "durability_tier":      dur_tier,
        "playoff_mult":         playoff_mult,
        "playoff_tier":         playoff_tier,
        "raw_predicted":        raw_predicted,
        "predicted":            predicted,
        "low":                  max(min_pct * cap_dollars_val, predicted - band),
        "min_floor_dollars":    min_pct * cap_dollars_val,
        "high":                 min(predicted + band, cba_max_dollars),
        "band":                 band,
        "cap":                  cap_dollars_val,
        "max_pct":              max_pct,
        "cba_max_dollars":      cba_max_dollars,
        "cba_cap_applied":      cba_cap_applied,
        "cba_floor_applied":    cba_floor_applied,
        "max_tier_floor_applied": max_tier_floor_applied,
        "supermax_eligible":    supermax_eligible,
        "supermax_tier_label":  supermax_tier_label,
    }


def detect_caveats(features: dict) -> list[str]:
    notes: list[str] = []
    age = features.get("age")
    salary = features.get("salary", 0)
    barrett = features.get("barrett_score", 0)

    # Rookie scale lock — driven by the actual NBA player index (first-round
    # picks within years 1-4 of their rookie scale). CBA-binding: they
    # cannot sign a new market deal until the rookie scale expires.
    if features.get("on_rookie_scale"):
        notes.append(
            "Currently on rookie scale (CBA-locked salary). The projection "
            "below is for their NEXT contract — i.e. their rookie-scale "
            "extension or first market deal."
        )

    # Supermax eligibility — surface the specific tier + dollar amount so
    # the user sees WHY the model floored their projection.
    if features.get("supermax_eligible"):
        recent = features.get("recent_all_nba", []) or []
        tier_label = features.get("supermax_tier", "")
        # Dollar amount for the supermax tier — at the CONTRACT-season cap
        # (the deal starts next season), matching what the prediction floors to.
        cap_M = SALARY_CAP_M.get(CONTRACT_SEASON, 165.0)
        max_pct = float(features.get("max_pct", 0.35) or 0.35)
        supermax_dollars_M = cap_M * max_pct
        notes.append(
            f"Supermax-eligible: {tier_label} ≈ ${supermax_dollars_M:.1f}M. "
            f"{len(recent)} All-NBA selection{'s' if len(recent) != 1 else ''} "
            f"in last 3 seasons + {features.get('team_tenure', 0)} years with "
            f"{features.get('current_team', 'current team')}. "
            f"Projection floored at this player's CBA max."
        )
    elif age and age >= 27 and barrett >= 28:
        # Production warrants supermax but missing CBA-binding criteria
        # (no All-NBA, not tenured, etc.). Only flag the speculative
        # "supermax-track if they hit All-NBA" path BEFORE this year's
        # All-NBA has been awarded. Once voting is closed and they
        # didn't make the team, the caveat is stale — the model handles
        # their pricing as a star-tier non-All-NBA producer.
        try:
            selections = fetch_all_nba_selections()
            all_nba_decided = any(
                s.get("season") == CURRENT_SEASON
                for player_sels in selections.values()
                for s in player_sels
            )
        except Exception:
            all_nba_decided = False
        if not all_nba_decided:
            notes.append(
                "Star-tier producer — supermax-track if they hit All-NBA "
                "this season AND stay with their current team. Projection "
                "uses standard max ceiling."
            )

    # Veteran end-of-career caveat removed — the prediction itself
    # (age multiplier + low rate score) already projects these players
    # into the vet-min range, no need to also caveat it.
    return notes


def explain_prediction(features: dict, prediction: dict,
                       market_median: float | None,
                       divergence: float) -> list[str]:
    """Plain-English bullets describing what drove the prediction.

    Each call returns 1-3 short sentences explaining:
      - What sets the dollar amount (CBA cap, supermax floor, normal projection)
      - How the market view compares (agreement, divergence, paycut filter)

    Tailored per player — the bullets for Luka (CBA-capped, lost supermax
    after trade) read differently than the bullets for Rui (model/market
    diverge because box score undersells role-player value).
    """
    bullets: list[str] = []
    name = features.get("name", "this player")
    final_M = prediction.get("predicted", 0) / 1e6
    raw_M = prediction.get("raw_predicted", prediction.get("predicted", 0)) / 1e6
    max_pct = prediction.get("max_pct", 0.35)
    max_pct_pct = int(round(max_pct * 100))
    cba_max_M = prediction.get("cba_max_dollars", 0) / 1e6
    supermax_tier = prediction.get("supermax_tier_label", "")
    svc = features.get("service_years", 0)
    tenure = features.get("team_tenure", 0)
    team = features.get("current_team", "current team")
    recent_nba = len(features.get("recent_all_nba", []) or [])
    contract_end = features.get("contract_end_season") or ""

    # ── Bullet 0: context line for players with multi-year contracts ──────
    # Honest framing — the prediction is "as of today" based on current
    # production. Doesn't try to project forward to the actual signing year.
    if contract_end and contract_end != CURRENT_SEASON:
        bullets.append(
            f"**Note:** {name}'s current contract runs through {contract_end}. "
            f"This projection answers \"what would a GM pay him *today* based "
            f"on his current performance?\" — not what he'll actually sign "
            f"for when the current deal ends."
        )

    # ── Bullet 1: what sets the dollar amount ──────────────────────────────
    if prediction.get("cba_floor_applied"):
        # Supermax-eligible: floored at Designated Vet/Rookie max.
        bullets.append(
            f"**${final_M:.1f}M = {max_pct_pct}% of cap (CBA-mandated max).** "
            f"{name} qualifies as **{supermax_tier}** with {recent_nba} recent "
            f"All-NBA selection{'s' if recent_nba != 1 else ''} + {tenure} years "
            f"on {team}. Raw production rate alone would have projected "
            f"${raw_M:.1f}M; the supermax floor lifts elite stars to their "
            f"eligible max."
        )
    elif prediction.get("max_tier_floor_applied"):
        # All-NBA near-max floor — lifted to ~max−3%, NOT the full/supermax max.
        bullets.append(
            f"**${final_M:.1f}M — lifted to the All-NBA max tier.** {name}'s "
            f"{recent_nba} recent All-NBA selection{'s' if recent_nba != 1 else ''} "
            f"mark him as max-caliber; raw production projected ${raw_M:.1f}M, but "
            f"the model hedges below the max for elite players, so we lift him to "
            f"the empirical eligible-star level (~3% under the {max_pct_pct}% max)."
        )
    elif prediction.get("cba_cap_applied"):
        # Production exceeds the player's CBA max (capped).
        if "Designated" in supermax_tier:
            bullets.append(
                f"**${final_M:.1f}M = {max_pct_pct}% of cap (CBA-mandated max).** "
                f"{name} qualifies as **{supermax_tier}** ({recent_nba} recent "
                f"All-NBA + {tenure} years on {team}). His raw production "
                f"projects ${raw_M:.1f}M, but the CBA caps him at this max."
            )
        else:
            # Standard max cap (no Designated tier).
            extra = ""
            if supermax_tier in ("Max 35%", "Max 30%", "Max 25%"):
                extra = " (no Designated Vet because he isn't tenured with his current team)"
            bullets.append(
                f"**${final_M:.1f}M = {max_pct_pct}% of cap.** "
                f"{name} has {svc} years of service → **{supermax_tier}** "
                f"under the CBA{extra}. Raw production would project "
                f"${raw_M:.1f}M; the max contract caps him here."
            )
    elif features.get("on_rookie_scale"):
        bullets.append(
            f"**Currently on rookie scale** (CBA-locked salary). The projection "
            f"of ${final_M:.1f}M is for his **next** deal — typically a rookie-"
            f"scale extension after year 4."
        )
    else:
        # Normal projection — base × multipliers, no CBA override.
        bullets.append(
            f"**${final_M:.1f}M** is built from his career rate score "
            f"({features.get('career_barrett', 0):.1f}, rank "
            f"#{features.get('effective_rank', 0)}), then adjusted for age, "
            f"position, durability, and recent playoff impact. See the math "
            f"breakdown below."
        )

    # ── Bullet 2: model vs market ──────────────────────────────────────────
    if market_median is not None:
        market_M = market_median / 1e6
        if divergence < 0.05:
            bullets.append(
                f"**Model and market agree** at ~${final_M:.1f}M — "
                f"comparable signings cluster right at this number."
            )
        elif divergence < 0.15:
            bullets.append(
                f"**Model and market are close** "
                f"(${final_M:.1f}M model, ${market_M:.1f}M market) — "
                f"strong agreement on this profile's value."
            )
        elif divergence < 0.30:
            higher = "market" if market_M > final_M else "model"
            bullets.append(
                f"**Model and market disagree somewhat** "
                f"(${final_M:.1f}M model vs ${market_M:.1f}M market). The "
                f"{higher} view is higher — use the range, not a point."
            )
        else:
            higher = "market" if market_M > final_M else "model"
            gap_pct = int(round(divergence * 100))
            bullets.append(
                f"**Big {gap_pct}% gap** between model (${final_M:.1f}M) and "
                f"market (${market_M:.1f}M). The {higher} captures something "
                f"the other doesn't — the box-score model sometimes underweights "
                f"intangibles like defense, fit, and locker-room value that GMs "
                f"actually pay for. Treat as a range."
            )

    return bullets


@st.cache_data(ttl=3600, show_spinner="Loading comparable signings…")
def load_historical_signings(n_recent_pairs: int = 3) -> pd.DataFrame:
    pairs = [(SEASONS[i + 1], SEASONS[i]) for i in range(n_recent_pairs)]
    rows: list[pd.DataFrame] = []
    # One bulk lookup outside the loop — fetch_draft_classes is cached for
    # the day, but build_draft_tier_lookup builds a dict once and we reuse.
    draft_lookup = build_draft_tier_lookup()
    for prev, curr in pairs:
        try:
            prev_df = build_ranked_projected(prev)
            curr_df = build_ranked_projected(curr)
            raw_prev = fetch_league_stats(prev, "Regular Season")
            # Use the better BBRef detailed positions, fall back to the older
            # ESPN coarse map when a player isn't in BBRef's table.
            detailed_lookup = fetch_player_positions_detailed(prev, cache_v=3)
            coarse_lookup = fetch_bref_positions(season_to_espn_year(prev), cache_v=3)
        except Exception:
            continue
        if prev_df.empty or curr_df.empty or raw_prev.empty:
            continue

        def _resolve_pos(n):
            d = detailed_lookup.get(normalize(n))
            if d:
                return position_to_bucket(d)
            return coarse_lookup.get(normalize(n), "Unknown")

        def _resolve_pos_detailed(n):
            d = detailed_lookup.get(normalize(n))
            if d:
                return d  # PG/SG/SF/PF/C
            # Fallback returns "Guard"/"Forward"/"Center" — convert to
            # single-letter so the comps Position column stays consistent
            # (Sengun was showing "Center" while others showed "C").
            return _pos_abbrev(coarse_lookup.get(normalize(n), "Unknown"))

        age_lookup = dict(zip(raw_prev["PLAYER_ID"], raw_prev.get("AGE", [])))
        curr_slim = curr_df[["PLAYER_ID", "salary"]].rename(
            columns={"salary": "salary_curr"})
        m = prev_df[prev_df["salary"] > 0].merge(curr_slim, on="PLAYER_ID", how="left")
        m = m[m["salary_curr"].notna() & (m["salary_curr"] > 0)]
        if m.empty:
            continue
        m["pct_change"] = (m["salary_curr"] - m["salary"]) / m["salary"]
        m = m[m["pct_change"].abs() >= NEW_CONTRACT_PCT]
        if m.empty:
            continue
        m["age"] = m["PLAYER_ID"].map(age_lookup)
        m["pos"] = m["Player"].map(_resolve_pos)
        m["pos_detailed"] = m["Player"].map(_resolve_pos_detailed)

        def _resolve_draft_tier(n):
            info = draft_lookup.get(normalize(n))
            return info["draft_tier"] if info else "Undrafted"

        def _resolve_draft_pick(n):
            info = draft_lookup.get(normalize(n))
            return info["draft_pick"] if info else None

        m["draft_tier"] = m["Player"].map(_resolve_draft_tier)
        m["draft_pick"] = m["Player"].map(_resolve_draft_pick)
        m["signed_in"] = curr
        m["prev_season"] = prev  # needed to compute career-weighted-at-signing
        rows.append(m[[
            "Player", "age", "pos", "pos_detailed", "barrett_score",
            "draft_tier", "draft_pick",
            "salary", "salary_curr", "signed_in", "prev_season",
        ]])

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["age", "barrett_score", "salary_curr"])

    # Exclude rookie-scale ladder steps (year-N → year-N+1 team-option
    # progressions, CBA-mandated). These aren't real new contracts but
    # they're often >25% YoY change because rookie scale escalates.
    #
    # Signal: both salaries still inside rookie scale band (<~15-18% of
    # the year's cap) AND player young enough to plausibly still be on
    # a rookie deal. Bumped the age cap from 23 → 25 to catch older
    # draftees: Desmond Bane was almost-22 when drafted, so his year-4
    # team option ($3.85M, 2023-24) lands at age 25 in NBA's Feb-1
    # convention. Same pattern for Tari Eason / any 22-23-year-old
    # draftee. Filter would have missed them all.
    #
    # Legitimate rookie EXTENSIONS (e.g. Sengun's $5.4M → $33.9M jump)
    # keep their signed_for ABOVE the rookie-scale band so they're
    # untouched by this filter.
    def _is_rookie_ladder(row) -> bool:
        cap_prev = SALARY_CAP_M.get(row["prev_season"], 154.6) * 1_000_000
        cap_curr = SALARY_CAP_M.get(row["signed_in"], 154.6) * 1_000_000
        if cap_prev <= 0 or cap_curr <= 0:
            return False
        salary_then_pct = float(row["salary"]) / cap_prev
        salary_curr_pct = float(row["salary_curr"]) / cap_curr
        age = row.get("age")
        if pd.isna(age) or age is None:
            return False
        return (
            salary_then_pct < 0.15
            and salary_curr_pct < 0.18
            and float(age) <= 25
        )

    mask = ~out.apply(_is_rookie_ladder, axis=1)
    out = out[mask].reset_index(drop=True)

    # Tag (don't drop) vet-min / training-camp signings (under 3% of cap,
    # ~$4.6M at the 2025-26 level): CBA minimum + bench-filler deals, not
    # market-rate contracts. find_comparables decides per-target whether to
    # use them. A below-replacement target (Broome-tier, trailing Barrett < 0)
    # SHOULD be priced against the minimum-salary market — dropping cheap deals
    # there floors his comp median at the ~$4.6M exclusion line and inflates the
    # projection. A mid/high target still drops them (score-far noise — the
    # original reason for the filter, e.g. Bates-Diop polluting Peyton Watson).
    def _is_vet_min(row) -> bool:
        cap_curr = SALARY_CAP_M.get(row["signed_in"], 154.6) * 1_000_000
        if cap_curr <= 0:
            return False
        return float(row["salary_curr"]) < cap_curr * 0.03

    out["is_vet_min"] = out.apply(_is_vet_min, axis=1)
    out = out.reset_index(drop=True)

    # Precompute career-weighted Barrett (50/30/20 of last 3 healthy seasons,
    # with availability — same metric as the target's trailing_barrett) for
    # every comp in the pool. Stored on the cached frame so we don't pay
    # the per-player fetch on every page load.
    #
    # Used by find_comparables for the min(sign_yr_gap, career_gap)
    # distance: a comp counts as close if EITHER their walk year OR their
    # 3-yr trajectory matched the target's trailing form. Pulls in
    # breakout-walk-year comps via their career, and trajectory-matched
    # comps via their walk year.
    out["career_weighted_barrett"] = out.apply(
        lambda r: _career_weighted_barrett_at(
            r["Player"], r["prev_season"], float(r["barrett_score"])
        ),
        axis=1,
    )
    return out


def _career_weighted_barrett_at(player_name: str, up_to_season: str,
                                fallback_score: float) -> float:
    """Weighted avg of a player's last 3 healthy seasons (GP ≥ 40) BEFORE
    signing. Same 50/30/20 weighting we use for the live player.
    Skipping injury years keeps the comparables apples-to-apples — we don't
    want to match Zach LaVine's healthy-Score against a comparable's
    injury-deflated walk year."""
    try:
        career = fetch_player_full_career(player_name)
        if career.empty:
            return fallback_score
        # Include only seasons up to and including the walk year.
        up_to = career[career["Season"] <= up_to_season]
        if up_to.empty:
            return fallback_score
        # Prefer healthy seasons (GP ≥ 40), fall back to all seasons if
        # the player has no healthy data on file.
        healthy = up_to[up_to["GP"] >= HEALTHY_SEASON_GP]
        pool = healthy if not healthy.empty else up_to
        recent = pool.tail(3)
        weights_full = [0.20, 0.30, 0.50]
        weights = weights_full[-len(recent):]
        w_sum = sum(weights)
        return float((recent["Barrett Score"].values * weights).sum() / w_sum)
    except Exception:
        return fallback_score


def _tier_penalty_weight(age) -> float:
    """How much should draft tier matter for this player?

    Draft pedigree is a strong predictor of *first-contract* value: lottery
    picks get paid for being lottery picks (Jalen Green, Jordan Poole) even
    when production rank says otherwise; 2nd-round developers (Rollins,
    Dinwiddie) have to earn it on the floor first. But by age ~30, the
    player has been priced by the market for years — their next deal is
    driven by production, age, durability, and role, not by where they
    were drafted a decade ago.

    Returns a scalar in [0, 1] that multiplies the tier-distance penalty:
      age ≤ 27  → 1.00  (full penalty, developer market)
      age = 28  → 0.75
      age = 29  → 0.50
      age = 30  → 0.25
      age ≥ 31  → 0.00  (veteran market — pedigree irrelevant)
    """
    if age is None:
        # Unknown age: be conservative, apply full penalty rather than
        # accidentally over-matching across cohorts.
        return 1.0
    a = float(age)
    if a <= 27:
        return 1.0
    if a >= 31:
        return 0.0
    return (31 - a) / 4.0


# Comp-pool quality gate. Elite / outlier players (Jokić, Giannis, Wemby) have
# few true peers; rather than pad the table to a fixed count with far-off
# matches — which pulls below-tier centers (Zubac, Brook Lopez) into Jokić's
# pool and corrupts the market median — keep ONLY comps whose sign-year score
# is within a band of the target. This band is a HARD cap the market view
# cannot bypass (no min-count fallback): if nobody falls in the band, we show
# no comps and suppress the market opinion rather than invent a non-peer —
# e.g. a -4.0 player must never be priced off +3.7 rotation centers.
COMP_SCORE_TOL_PCT = 0.25   # keep comps within ±25% of the target's current score…
COMP_SCORE_TOL_MIN = 5.0    # …or ±5 raw Barrett points, whichever band is wider
# Minimum-caliber targets (current Barrett under this) are priced against the
# minimum-salary market too — their real comps ARE minimum deals, so vet-min
# signings stay in their pool instead of being filtered out (which would leave
# only $5-10M rotation comps and inflate the market median). The rotation tier
# sits at ~9+ Barrett, so 5.0 catches fringe / replacement players (Thanasis
# 1.1, Amari 1.1, Jae'Sean Tate 3.1 — all were projecting $7-11M off non-peer
# comps) while leaving every Barrett ≥ 5 player's pool exactly as it was.
# Verified: scrubs drop to ~$2-4M market; rotation / mid / stars unchanged.
VETMIN_COMP_TARGET_MAX = 5.0


def find_comparables(features: dict, history: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """Match historical signings on **current-season Barrett** + age + position.

    The target is matched on its sign-year (current) Barrett — the same snapshot
    the comps carry, so it's apples-to-apples. We do NOT match on the trailing
    (last-3-healthy-years) average: it reaches back to a declined/aged player's
    prime (Kyle Lowry, 40, current 1.0 but trailing 17.7 → Chris Paul comps,
    $12M market). Walk-forward OOS (2021-25, minimums included) settled it —
    matching on current form cuts declined-player market error 11.6M→4.5M and
    minimum-signing error 6.6M→4.8M, with no loss on stable players and (unlike
    min(current, trailing), which re-breaks them) no under-rating of breakouts.
    See scripts/experiment_match_score.py.

    Score-distance weight is age-scaled (1.0 + tier_weight):
      - Young developers (≤27): 2.0× score — prevents Zion-style stretch
        matches where a far-off score sneaks in via perfect age/position/
        tier alignment. Forces the pool to favor score-close comps.
      - Veterans (≥31): 1.0× score — keeps age relatively important so
        aging vets match other aging vets (Harden's paycut cohort), not
        young production-similar stars (Brunson/Booker).

    Distance = |comp_sign_yr_score − target_current_score| × (1 + tier_weight)
             + |age_diff| × 1.5
             + position_penalty (broad G/F/C bucket — PG/SG are pooled)
             + tier_penalty (faded by age — see _tier_penalty_weight)
    """
    if history.empty:
        return history

    target_position = features["position"]
    target_age = features["age"] if features["age"] else 27
    # Match on CURRENT-season form, NOT the trailing (last-3-healthy-years)
    # average. Trailing reaches back to a declined/aged player's prime — Kyle
    # Lowry (40, current 1.0) read as trailing 17.7 and matched to Chris Paul,
    # market $12M. Walk-forward OOS (2021-25, minimums included) decided this:
    # matching on current form cuts declined-player market error 11.6M->4.5M and
    # minimum-signing error 6.6M->4.8M, with no loss on stable players AND
    # (unlike min(current,trailing), which re-breaks them) no under-rating of
    # breakouts. It's also apples-to-apples: comps are matched on THEIR sign-year
    # score. See scripts/experiment_match_score.py.
    target_barrett = float(features.get("barrett_score") or 0.0)
    target_tier = features.get("draft_tier", "Undrafted")
    target_tier_idx = DRAFT_TIER_ORDINAL.get(target_tier, 4)
    tier_weight = _tier_penalty_weight(features.get("age"))

    # Exclude the target player from their own comp pool. Otherwise Vassell's
    # prior rookie extension ($29.3M, 2024-25) shows up in his own table —
    # which both looks weird and gives them an artificial perfect-match
    # comparable that biases the median.
    target_norm = normalize(features.get("name", ""))
    history = history[history["Player"].apply(normalize) != target_norm]
    if history.empty:
        return history

    # Target-aware minimum-salary inclusion (see VETMIN_COMP_TARGET_MAX). Keep
    # vet-min comps ONLY for below-replacement targets — their real market IS
    # the minimum. Everyone with a valid Barrett ≥ 0 drops them, reproducing
    # the old pool exactly (zero change for rotation+ players). Missing/NaN
    # Barrett defaults to dropping (the safe, prior behavior).
    _keep_vetmin = (
        target_barrett is not None
        and not pd.isna(target_barrett)
        and float(target_barrett) < VETMIN_COMP_TARGET_MAX
    )
    if "is_vet_min" in history.columns and not _keep_vetmin:
        history = history[~history["is_vet_min"]]
        if history.empty:
            return history

    # Match against same position bucket (Guard / Forward / Center —
    # already broad; PG and SG are both "Guard"). Fall back to all
    # positions only when same-bucket pool is too small.
    same_pos = history[history["pos"] == target_position].copy()
    if len(same_pos) < n:
        same_pos = history.copy()

    comp_tier_idx = same_pos["draft_tier"].map(
        lambda t: DRAFT_TIER_ORDINAL.get(t, 4)
    )
    tier_penalty = (comp_tier_idx - target_tier_idx).abs() * 4 * tier_weight
    pos_penalty = (same_pos["pos"] != target_position).astype(float) * 20

    # Score weight scales with age (uses the same tier_w curve). For young
    # developers (≤27), score is 2x — prevents Zion-style stretch matches
    # where a far-off score sneaks past via perfect age/position/tier
    # alignment. For veterans (≥31), score reverts to 1x because their
    # market is age-driven (Harden/Curry should match aging-vet paycut
    # cohort, not young production-similar stars) — so keeping age
    # relatively important.
    score_weight = 1.0 + tier_weight  # 2.0 young → 1.0 old

    same_pos["distance"] = (
        (same_pos["barrett_score"] - target_barrett).abs() * score_weight
        + (same_pos["age"] - target_age).abs() * 1.5
        + pos_penalty
        + tier_penalty
    )

    # One row per player: keep each player's closest-matching signing so a
    # single player can't fill multiple comp slots or double-count in the
    # market median (e.g. Sabonis appearing twice in Jokić's pool).
    same_pos = (same_pos.sort_values("distance")
                        .drop_duplicates(subset="Player", keep="first"))

    # HARD score gate (the market view cannot bypass it). Restrict to comps
    # whose sign-year score is within a hybrid band of the target's trailing
    # score — COMP_SCORE_TOL_PCT of it, or COMP_SCORE_TOL_MIN raw Barrett points,
    # whichever is wider — so it scales with stars without collapsing for low
    # scorers. Gate FIRST, then take the n closest by composite distance among
    # the in-band peers (so a score-close-but-older comp isn't lost to the
    # composite top-n). No min-count fallback: the old code padded to a fixed
    # count with the closest names regardless of distance, which priced outliers
    # off non-peers (e.g. Broome -4.0 vs +3.7 centers, a 7-point swing). An
    # empty result is allowed — downstream it renders "no close comparables" and
    # suppresses the market opinion, so the projection stands on the model.
    tol = max(abs(target_barrett) * COMP_SCORE_TOL_PCT, COMP_SCORE_TOL_MIN)
    in_band = same_pos[(same_pos["barrett_score"] - target_barrett).abs() <= tol]
    return in_band.nsmallest(n, "distance").copy()


def _signing_cap(signed_in_season: str) -> float:
    """Salary cap in dollars for a comparable's signing year."""
    return SALARY_CAP_M.get(signed_in_season, 154.6) * 1_000_000


def _classify_context(row) -> str:
    """Classify what KIND of signing this was — context the model can't see.
    Returns one of: 'Supermax', 'Free-agent raise', 'Rookie extension',
    'Paycut to stay', 'Standard new deal'."""
    cap = _signing_cap(str(row["signed_in"]))
    salary_then = float(row["salary"])
    salary_signed = float(row["salary_curr"])
    age = float(row["age"]) if not pd.isna(row["age"]) else 27
    pct_change = (salary_signed - salary_then) / salary_then if salary_then > 0 else 0

    # Supermax-tier: signing >= 28% of cap (35% supermax tier or near-max deals).
    if salary_signed >= cap * SUPERMAX_CAP_PCT:
        return "Supermax"
    # Paycut: lost real money to stay or re-sign.
    if pct_change <= -0.10:
        return "Paycut"
    # Rookie extension: very low prior salary, young player, big jump.
    # Age threshold bumped from 24 → 26 to catch older draftees whose
    # rookie-deal year-5 starts at 25-26 (Bane, Tari Eason, etc.). Their
    # extension still qualifies as "Rookie extension" context — first
    # non-rookie deal off the scale.
    if salary_then < cap * 0.10 and age <= 26 and pct_change >= 0.50:
        return "Rookie extension"
    # Standard meaningful raise (typical free-agent / vet new deal).
    if pct_change >= 0.15:
        return "Free-agent raise"
    return "New deal"


_CONTEXT_BADGE_COLOR = {
    "Supermax":          "#9b59b6",
    "Free-agent raise":  "#2ecc71",
    "Rookie extension":  "#16d4c1",
    "Paycut":            "#e74c3c",
    "New deal":          "#999999",
}


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median: smallest value v such that the cumulative weight up
    to and including v is ≥ half the total weight. Used for the market
    view so the closest comparables count more than the farthest ones."""
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cumw = np.cumsum(w_sorted)
    threshold = cumw[-1] / 2.0
    idx = int(np.searchsorted(cumw, threshold))
    return float(v_sorted[min(idx, len(v_sorted) - 1)])


def _weighted_quantile(values: np.ndarray, weights: np.ndarray,
                       q: float) -> float:
    """Weighted q-th quantile. Generalization of _weighted_median for
    arbitrary quantiles (used for the middle-50% IQR display)."""
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cumw = np.cumsum(w_sorted)
    threshold = cumw[-1] * q
    idx = int(np.searchsorted(cumw, threshold))
    return float(v_sorted[min(idx, len(v_sorted) - 1)])


def _inverse_distance_weights(distances: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Convert match distances to weights via 1 / (distance + eps).
    Closer comparables get bigger weights; the +eps prevents division by
    zero and keeps the weight spread reasonable (a perfect match doesn't
    dominate every other comparable)."""
    return 1.0 / (np.asarray(distances, dtype=float) + eps)


def _comp_salaries_in_contract_dollars(comps: pd.DataFrame) -> np.ndarray:
    """Comparable signed salaries scaled to CONTRACT_SEASON (next-season) cap
    dollars, so the market median is apples-to-apples with the model's figure.
    Each comp signed in a different year at a different cap; a deal worth 20% of
    the 2024-25 cap is expressed as 20% of the 2026-27 cap."""
    contract_cap = SALARY_CAP_M.get(CONTRACT_SEASON, 165.0)
    factor = comps["signed_in"].map(
        lambda s: contract_cap / SALARY_CAP_M.get(s, 154.6)).astype(float).values
    return comps["salary_curr"].astype(float).values * factor


def _scouting_take(features: dict, comps: pd.DataFrame) -> dict:
    """Build the 'Scouting take' summary: top-3 names, weighted median deal,
    weighted IQR range, X-factor narrative.

    Uses inverse-distance weights so the closest comparables count more
    than the farthest — a tighter, more market-grounded second opinion."""
    if comps.empty:
        return {}

    salaries = _comp_salaries_in_contract_dollars(comps)  # 2026-27 dollars
    # If `distance` column is missing for some reason, fall back to equal
    # weighting so we never crash.
    if "distance" in comps.columns:
        weights = _inverse_distance_weights(comps["distance"].astype(float).values)
    else:
        weights = np.ones_like(salaries)

    median = _weighted_median(salaries, weights)
    q25    = _weighted_quantile(salaries, weights, 0.25)
    q75    = _weighted_quantile(salaries, weights, 0.75)

    top3 = comps.head(3)["Player"].tolist()

    # X-factor: largest meaningful divergence on either age or career Barrett.
    target_age = features["age"] or 27
    target_bar = features["career_barrett"]
    comp_age_med = float(comps["age"].median())
    comp_bar_med = float(comps["career_weighted_barrett"].median())
    age_diff = target_age - comp_age_med
    bar_diff = target_bar - comp_bar_med

    x_factor_parts: list[str] = []
    if abs(age_diff) >= 2:
        if age_diff < 0:
            x_factor_parts.append(
                f"younger than comps ({int(target_age)} vs {int(comp_age_med)}) "
                "→ projects higher"
            )
        else:
            x_factor_parts.append(
                f"older than comps ({int(target_age)} vs {int(comp_age_med)}) "
                "→ projects lower"
            )
    if abs(bar_diff) >= 3:
        if bar_diff > 0:
            x_factor_parts.append(
                f"higher score than comps ({target_bar:.1f} vs {comp_bar_med:.1f}) "
                "→ projects higher"
            )
        else:
            x_factor_parts.append(
                f"lower score than comps ({target_bar:.1f} vs {comp_bar_med:.1f}) "
                "→ projects lower"
            )

    if not x_factor_parts:
        x_factor = "Profile is right in line with comparable signings."
    else:
        x_factor = " · ".join(p.capitalize() for p in x_factor_parts) + "."

    return {
        "top3":     top3,
        "median":   median,
        "q25":      q25,
        "q75":      q75,
        "x_factor": x_factor,
    }


# ── Player picker ────────────────────────────────────────────────────────────
all_names = get_all_player_names()
if not all_names:
    st.error("Player database not yet loaded. Try again in a moment.")
    st.stop()

current_ranked = build_ranked_projected(CURRENT_SEASON)
current_names = (
    set(current_ranked["Player"].tolist())
    if not current_ranked.empty else set()
)
active_names = [n for n in all_names if n in current_names]

# Map each player → current-season Barrett score for ranking. The
# searchbox dropdown sorts by Barrett descending (Jokic / SGA / Giannis
# at the top instead of alphabetical "Aaron Holiday") so users can browse
# the best players first. Falls back to 0 for players without a current
# Barrett (shouldn't happen for active_names but defensive).
_barrett_lookup: dict[str, float] = {}
if not current_ranked.empty:
    _barrett_lookup = dict(zip(
        current_ranked["Player"],
        current_ranked["barrett_score"].fillna(0.0),
    ))
active_names = sorted(active_names, key=lambda n: -_barrett_lookup.get(n, 0.0))

# Free-agent subset for the "Free agents" search toggle — SAME definition as the
# FA Class page: a player is a FA if their next-year contract is RFA, an expiring
# UFA ("—"), or carries a player/team option (PO/TO). A normal next-year salary
# means they're under contract -> not a FA. Falls back to the full roster if the
# contract data can't be fetched, so the toggle never empties the search.
try:
    _fa_next = fetch_next_year_contracts(season_to_espn_year(CURRENT_SEASON), cache_v=7)
    def _is_free_agent(_n: str) -> bool:
        s = fmt_next_contract(_n, _fa_next)
        return s == "RFA" or s == "—" or " PO" in s or " TO" in s
    _fa_names = [n for n in active_names if _is_free_agent(n)] or active_names
except Exception:
    _fa_names = active_names

_PICKER_KEY = "contract_predictor_player"

# Resolve initial value from URL ?player= param (deep-link support).
# st_searchbox manages its own state internally via the `key`; we only
# need to seed the default on first render.
_init_player = None
if "player" in st.query_params:
    qp = st.query_params["player"]
    _init_player = next(
        (n for n in active_names if normalize(n) == normalize(qp)),
        None,
    )


def _current_player() -> str | None:
    """Player currently selected in the searchbox (survives reruns via the
    component's own session state), so the empty/centered list can orbit them.
    Falls back to the URL deep-link seed before the first interaction."""
    state = st.session_state.get(_PICKER_KEY)
    if isinstance(state, dict) and state.get("result"):
        return state["result"]
    return _init_player


def _search_players(query: str) -> list[str]:
    """Filter active-season players by case- and accent-insensitive substring.

    Empty query (or the box still holding the selected player's full name on
    focus) → the FULL roster (sorted by Barrett desc) so the dropdown stays
    scrollable. An injected script auto-scrolls it to center the selected
    player on open (see the scroll-centering helper below the searchbox).

    With a real query → match by quality (exact → prefix → substring), then
    Barrett score descending within each match tier. Capped at 10.
    """
    # "Free agents" toggle restricts the searchable pool to FAs only.
    pool = _fa_names if st.session_state.get("fa_filter") else active_names
    cur = _current_player()
    if not query or not query.strip():
        return pool
    q = normalize(query)
    # Box untouched since selection (still the player's full name on focus) →
    # the full scrollable roster, not just the lone exact match.
    if cur and q == normalize(cur):
        return pool
    scored: list[tuple[int, float, str]] = []
    for n in pool:
        nn = normalize(n)
        if nn == q:
            quality = 3
        elif nn.startswith(q):
            quality = 2
        elif q in nn:
            quality = 1
        else:
            continue
        scored.append((quality, _barrett_lookup.get(n, 0.0), n))
    # Sort: quality DESC, then Barrett DESC, then name as final tiebreak.
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [n for _, _, n in scored[:10]]


# st_searchbox is a real text input (Cmd+A, cursor positioning, character
# selection all work) backed by a custom dropdown of matches. Replaces the
# old st.selectbox which displayed selected values as static label text.
#   - edit_after_submit="option": after selecting a player, the input
#     updates to show the selected player's FULL name (not the partial
#     search term the user typed). Still editable in place — Cmd+A to
#     replace, etc.
#   - default_options: shown before any typing — the component does NOT
#     call _search_players with an empty string, so we explicitly seed it
#     with the alphabetical roster so users can browse without typing.
#   - rerun_on_update=True: refreshes the page when a new player is
#     picked so the contract details update.
# Softer, rounder control + menu so it reads as a search bar, not a form box.
# The component renders in an iframe locked to the light Streamlit config, so in
# dark mode it'd be a glaring white box — make it theme-aware: dark fill + light
# text + a dark wrapper so the (light) iframe body doesn't show in the rounded
# corners.
if st.session_state.get("theme_dark", False):
    _SEARCHBOX_STYLE = {
        "wrapper": {"backgroundColor": "#0a0a14"},
        "searchbox": {
            "control": {"borderRadius": 24, "padding": "3px 12px",
                        "backgroundColor": "#16181f", "border": "1px solid #2c2c40"},
            "menuList": {"borderRadius": 16, "backgroundColor": "#16181f"},
            "input": {"color": "#e8e8f0"},
            "singleValue": {"color": "#e8e8f0"},
            "placeholder": {"color": "#8a8a9a"},
            "option": {"color": "#dcdce6", "backgroundColor": "#16181f",
                       "highlightColor": "#23233a"},
        },
        "dropdown": {"fill": "#9aa0aa"},
        "clear": {"fill": "#9aa0aa"},
    }
else:
    _SEARCHBOX_STYLE = {
        "searchbox": {
            "control": {"borderRadius": 24, "padding": "3px 12px"},
            "menuList": {"borderRadius": 16},
        },
    }
# THE white strip: a browser draws an iframe's native backing surface using its
# `color-scheme`. The searchbox iframe is color-scheme:normal → the browser
# paints that surface WHITE, which shows as a white line at the iframe's edge in
# dark mode (invisible to every getComputedStyle bg/border/shadow probe because
# it's the UA paint layer, not a styled box). Forcing color-scheme:dark makes
# the browser paint it dark. Belt-and-suspenders: also set a dark background.
_sb_dark = st.session_state.get("theme_dark", False)
_sb_iframe_bg = "#0a0a14" if _sb_dark else "#ffffff"
_sb_scheme = "dark" if _sb_dark else "light"
st.markdown(
    "<style>iframe[title='streamlit_searchbox.searchbox']"
    f"{{background:{_sb_iframe_bg} !important;border:none !important;"
    f"color-scheme:{_sb_scheme} !important;}}</style>",
    unsafe_allow_html=True,
)
# In dark mode the react-select dropdown 'menu' container defaults to WHITE and
# the component exposes no override for it (only menuList/option/control). It
# also mounts only when the dropdown opens. So reach INTO the searchbox iframe
# from a 0-height helper iframe and inject a <style> that darkens the menu;
# a MutationObserver re-applies it each time the menu remounts.
if st.session_state.get("theme_dark", False):
    import streamlit.components.v1 as _c
    _c.html(
        """
        <script>
        (function () {
          var pdoc = window.parent.document;
          function styleSearchbox() {
            var f = pdoc.querySelector("iframe[title='streamlit_searchbox.searchbox']");
            if (!f) return false;
            var idoc = f.contentDocument; if (!idoc) return false;
            if (idoc.getElementById('hv-sb-dark')) return true;
            // Force the iframe document's colour-scheme dark so the BROWSER paints
            // its native backing surface dark (the source of the white seam),
            // not just the styled boxes.
            try { idoc.documentElement.style.colorScheme = 'dark'; } catch (e) {}
            var s = idoc.createElement('style'); s.id = 'hv-sb-dark';
            s.textContent =
              // The iframe BODY is the locked light config colour (#f4f6f8); it's
              // taller than the input, so the light shows as a strip between the
              // input and the floating menu. Paint html/body to the page bg.
              "html{color-scheme:dark!important;}"
              + "html,body{background:#0a0a14!important;}"
              + "div[class$='-menu'],div[class*='-menu ']{background:#16181f!important;"
              + "border:1px solid #2c2c40!important;border-radius:16px!important;"
              + "box-shadow:0 8px 24px rgba(0,0,0,.45)!important;overflow:hidden!important;}"
              // Any stray inline-white element (e.g. a loading bar) → dark too.
              + "div[style*='background: white'],div[style*='background:#fff']"
              + "{background:#16181f!important;}";
            idoc.head.appendChild(s);
            return true;
          }
          var n = 0, t = setInterval(function () {
            if (styleSearchbox() || ++n > 40) clearInterval(t);
          }, 150);
        })();
        </script>
        """,
        height=0,
    )
# Auto-scroll the open dropdown so the selected player sits CENTERED in the
# (scrollable) full roster instead of pinned to the top. Reaches into the
# searchbox iframe on input focus and scrollIntoView({block:'center'}) the
# matching option. Both themes; harmless while typing (the typed text won't
# match a full name). Only injected when a player is already selected.
if _init_player:
    import json as _json
    import streamlit.components.v1 as _sc
    _sc.html(
        """
        <script>
        (function () {
          var pdoc = window.parent.document;
          var TARGET = __HV_TARGET__;
          function center(tries) {
            tries = tries || 0;
            var f = pdoc.querySelector("iframe[title='streamlit_searchbox.searchbox']");
            if (f && f.contentDocument) {
              var opts = f.contentDocument.querySelectorAll("[class*='option']");
              for (var i = 0; i < opts.length; i++) {
                if (opts[i].textContent.trim() === TARGET) {
                  opts[i].scrollIntoView({ block: 'center' });
                  return;
                }
              }
            }
            if (tries < 30) setTimeout(function () { center(tries + 1); }, 80);
          }
          var n = 0, t = setInterval(function () {
            var f = pdoc.querySelector("iframe[title='streamlit_searchbox.searchbox']");
            if (f && f.contentDocument && !f.contentDocument.__hvScrollHook) {
              f.contentDocument.__hvScrollHook = true;
              f.contentDocument.addEventListener('focusin', function (e) {
                if (e.target && e.target.tagName === 'INPUT') center(0);
              });
            }
            if (++n > 40) clearInterval(t);
          }, 150);
        })();
        </script>
        """.replace("__HV_TARGET__", _json.dumps(_init_player)),
        height=0,
    )
# "Free agents" filter at the right end of the search bar. Streamlit has the
# toggle's new value in session_state from the start of the rerun, so read it up
# front. st_searchbox caches its dropdown options when its key is first created
# and ignores later default_options changes — so when the toggle FLIPS, drop the
# searchbox's cached state to force it to re-seed from the new (filtered) pool.
_fa_on = bool(st.session_state.get("fa_filter"))
if st.session_state.get("_fa_prev") != _fa_on:
    st.session_state["_fa_prev"] = _fa_on
    st.session_state.pop(_PICKER_KEY, None)
_sb_col, _fa_col = st.columns([9, 2], vertical_alignment="center")
with _fa_col:
    st.toggle("Free agents", key="fa_filter",
              help="Search only free agents — UFA, RFA, and player/team options")
with _sb_col:
    selected = st_searchbox(
        search_function=_search_players,
        placeholder="Type a player name…",
        default=_init_player,
        # Seed the box TEXT with the selected player so the name stays put after
        # selection / on reload (not just the placeholder).
        default_searchterm=(_init_player or ""),
        # Scrollable pool (FA-only when the toggle is on), Barrett desc; the
        # injected script auto-scrolls the open dropdown to center the pick.
        default_options=(_fa_names if _fa_on else active_names),
        edit_after_submit="option",
        rerun_on_update=True,
        style_overrides=_SEARCHBOX_STYLE,
        key=_PICKER_KEY,
    )

# Mirror selection into the URL for deep-linking. Skip if unchanged to
# avoid re-triggering the URL → state seed in a loop.
if selected:
    if st.query_params.get("player") != selected:
        st.query_params["player"] = selected
elif "player" in st.query_params:
    del st.query_params["player"]

if not selected:
    st.info(
        f"**{len(active_names):,} active players** available. Try a star to see "
        "a supermax-eligible note, a veteran for the age discount, or a young "
        "rising player for the rookie-scale caveat."
    )
    st.stop()

# ── Compute prediction ───────────────────────────────────────────────────────
# Wrapped in a spinner: get_player_features + predict_contract are cached with
# show_spinner=False, so on a COLD cache the page would otherwise sit blank for
# the whole data fetch (reads as "broken"). The spinner makes it clear it's
# working while the projection is built.
with st.spinner(f"Projecting {selected}'s next contract…"):
    features = get_player_features(selected, CURRENT_SEASON)
    if features is None:
        st.warning(f"Couldn't find {selected} in {CURRENT_SEASON} data.")
        st.stop()
    prediction = predict_contract(features)  # stats: CURRENT_SEASON → contract: CONTRACT_SEASON
    caveats = detect_caveats(features)

# Compute comparables here too — we need their median for the hero card
# so the "Model vs. Market" framing is visible up top, not buried below.
_history = load_historical_signings(n_recent_pairs=3)
_comps = (
    find_comparables(features, _history, n=6)
    if not _history.empty else pd.DataFrame()
)

# Tag context up front so we can filter the market-median pool. We'll
# carry this through to the table render below (no double computation).
if not _comps.empty:
    _comps = _comps.copy()
    _comps["context"] = _comps.apply(_classify_context, axis=1)

# ── Market view logic ───────────────────────────────────────────────────────
# Three cases, decided by what the queried player is and what the
# comparables pool looks like:
#
#   1. Standard:  distance-weighted median of all 6 comps. Used for
#                 most players (non-supermax tier, or supermax tier
#                 with enough non-Paycut comps).
#
#   2. Filtered:  supermax-tier player WHERE ≥3 non-Paycut comps exist.
#                 Drop Paycut comps and take the weighted median of the
#                 remaining "stayed at tier" cohort. Notes the filter
#                 to the user.
#
#   3. Suppressed: supermax-tier player WHERE <3 non-Paycut comps exist
#                  (Curry-tier outliers — almost nobody has stayed at
#                  supermax this late in their career). The pool is
#                  dominated by Paycuts that don't represent this
#                  player's market. Show "Limited comparable data"
#                  instead of a misleading $3M-tier number; anchor
#                  the honest range on current salary instead.
_market_used_comps = _comps
_market_filter_applied = False
_market_suppressed = False
if not _comps.empty:
    _player_cur_sal_pct = (
        float(features.get("salary", 0) or 0)
        / (SALARY_CAP_M.get(CURRENT_SEASON, 154.6) * 1_000_000)
    )
    if _player_cur_sal_pct >= SUPERMAX_CAP_PCT:
        _non_paycut = _comps[_comps["context"] != "Paycut"]
        if len(_non_paycut) >= 3:
            # Case 2: enough non-Paycut data to compute a meaningful median.
            _market_used_comps = _non_paycut
            _market_filter_applied = True
        else:
            # Case 3: too few good comps. Don't show the broken median.
            _market_suppressed = True

# Market view uses distance-weighted median so the closest comparables
# count more than the farthest.
if _market_suppressed:
    _market_median = None
elif not _market_used_comps.empty:
    _salaries = _comp_salaries_in_contract_dollars(_market_used_comps)  # 2026-27 dollars
    if "distance" in _market_used_comps.columns:
        _weights = _inverse_distance_weights(
            _market_used_comps["distance"].astype(float).values
        )
    else:
        _weights = np.ones_like(_salaries)
    _market_median = _weighted_median(_salaries, _weights)
else:
    _market_median = None

def _confidence_bar_html(model_M, low_M, high_M, secondary_M=None,
                         secondary_color="#16d4c1", secondary_label="market",
                         scale_min_M=None, scale_max_M=None,
                         tertiary_M=None, tertiary_color="#9aa0aa",
                         tertiary_label="current"):
    """Premium-analytics range bar: a shaded confidence band, a bright Model
    marker, an optional secondary (anchor) dot, and an optional tertiary hollow
    ring (the player's current salary — shows the raise).

    The band passed in already folds in the market second opinion (it spans
    model ∪ market), so model-vs-market divergence reads as band width — there
    is no separate market marker on the line.

    When scale_min_M / scale_max_M are supplied, the TRACK is fixed to that
    span — the veteran-minimum floor up to the player's OWN capped max — so a
    maxed-out player's marker pegs the right edge and everyone else reads as a
    fraction of their own ceiling; the endpoints are labelled min / max. Without
    a scale it auto-fits the player's own band."""
    fixed = scale_min_M is not None and scale_max_M is not None
    if fixed:
        bar_lo, span = scale_min_M, max(scale_max_M - scale_min_M, 0.1)
    else:
        pts = [low_M, high_M, model_M] + ([secondary_M] if secondary_M is not None else [])
        lo, hi = min(pts), max(pts)
        pad = max((hi - lo) * 0.16, 1.0)
        bar_lo, span = lo - pad, max((hi + pad) - (lo - pad), 0.1)
    # Clamp to [0, 100] so a prediction at the floor pegs the LEFT edge (flush
    # "minimum") and one at the max pegs the right edge.
    p = lambda v: max(0.0, min(100.0, (v - bar_lo) / span * 100))
    bl, br, mp = p(low_M), p(high_M), p(model_M)
    # The model's $ value is labelled directly under the white marker (see
    # `val`), so it's no longer carried in the legend.
    legend_parts = []
    sec = ""
    if secondary_M is not None:
        sp = p(secondary_M)
        sec = (f'<div style="position:absolute; left:{sp}%; top:50%; width:12px; height:12px;'
               f' background:{secondary_color}; border:2px solid var(--panel-solid); border-radius:50%;'
               f' transform:translate(-50%,-50%); z-index:2;"></div>')
        legend_parts.append(f'<span style="color:{secondary_color};">●</span> {secondary_label}')
    # Tertiary marker: the player's current salary, drawn as a hollow ring so
    # it reads as "where they are now" vs the model/market markers (the raise).
    ter = ""
    if tertiary_M is not None:
        tp = p(tertiary_M)
        ter = (f'<div style="position:absolute; left:{tp}%; top:50%; width:11px; height:11px;'
               f' background:var(--panel-solid); border:2px solid {tertiary_color}; border-radius:50%;'
               f' transform:translate(-50%,-50%); z-index:1;"></div>')
        legend_parts.append(f'<span style="color:{tertiary_color};">○</span> {tertiary_label}')
    legend = '  ·  '.join(legend_parts)
    # Endpoint labels: league min/max when the scale is fixed (tagged so they
    # aren't misread as the player's own range), else the player's own band.
    if fixed:
        lab_lo = f'${scale_min_M:.1f}M <span style="color:var(--fg-6);">min</span>'
        lab_hi = f'<span style="color:var(--fg-6);">max</span> ${scale_max_M:.1f}M'
    else:
        lab_lo, lab_hi = f'${low_M:.1f}M', f'${high_M:.1f}M'
    # Built as a single newline-free string. A standalone {sec} line (empty
    # when there's no secondary marker) would be a whitespace-only line, which
    # CommonMark treats as a blank line that TERMINATES the surrounding HTML
    # block — Streamlit would then render the rest of the hero as literal text.
    # Concatenation keeps the whole bar on one line and sidesteps that bug.
    # Value label sitting just under the white model marker (kept inside the
    # track even when the marker pegs an edge).
    if mp >= 85:
        _vx = f'right:{max(100 - mp, 0):.1f}%;'
    elif mp <= 15:
        _vx = f'left:{mp:.1f}%;'
    else:
        _vx = f'left:{mp:.1f}%; transform:translateX(-50%);'
    # Edge-aware marker transform: flush LEFT at the floor, flush RIGHT at the
    # max, centered in between — so a minimum projection visibly pegs the edge.
    if mp <= 0.5:
        _mtx = 'transform:translateX(0);'
    elif mp >= 99.5:
        _mtx = 'transform:translateX(-100%);'
    else:
        _mtx = 'transform:translateX(-50%);'
    val = (
        f'<div style="position:absolute; {_vx} top:14px; white-space:nowrap; '
        f'font-size:0.82rem; font-weight:800; color:var(--fg-1); z-index:4;">'
        f'${model_M:.1f}M</div>'
    )
    track = (
        f'<div style="position:relative; height:8px; border-radius:5px; '
        f'background:var(--hairline);">'
        f'<div style="position:absolute; left:{bl:.1f}%; width:{max(br-bl,1):.1f}%; '
        f'top:0; bottom:0; border-radius:5px; '
        f'background:linear-gradient(90deg, rgba(22,212,193,0.22), rgba(22,212,193,0.5));"></div>'
        f'{ter}{sec}'
        f'<div style="position:absolute; left:{mp:.1f}%; top:-5px; width:3px; height:18px; '
        f'background:var(--fg-1); border-radius:2px; {_mtx} '
        f'box-shadow:0 0 10px rgba(255,255,255,0.75); z-index:3;"></div>'
        f'{val}'
        f'</div>'
    )
    labels = (
        f'<div style="display:flex; justify-content:space-between; align-items:center; '
        f'font-size:0.7rem; color:var(--fg-5); margin-top:1.6rem;">'
        f'<span>{lab_lo}</span>'
        f'<span style="color:var(--fg-4);">{legend}</span>'
        f'<span>{lab_hi}</span>'
        f'</div>'
    )
    return f'<div style="margin-top:1.15rem;">{track}{labels}</div>'


# ── Big number header: dominant Model $ + confidence bar + market secondary ───
predicted_M = prediction["predicted"] / 1_000_000
low_M  = prediction["low"]  / 1_000_000
high_M = prediction["high"] / 1_000_000
_model_only_M = predicted_M   # keep the pure model output for the explainer

# Blend toward the market ONLY in the mid-tier, where it earns its keep. The
# model is noisy on mid-tier role players (the comp median is the stronger
# signal there), but it NAILS the extremes — minimums (service floor) and stars
# (CBA max) — where blending toward noisier comps only adds error. Walk-forward
# OOS (scripts/experiment_blend_value.py): gating the blend to a model
# projection of ~$7-25M cuts overall median error $2.7M -> $1.9M vs always-on,
# killing the minimum/star over-projections while keeping the mid-tier market
# consensus. Max-capped / supermax-floor players stay exempt (their number is a
# CBA rule, not a noisy estimate); cap the market weight at 0.65.
_BLEND_TIER_LO_M, _BLEND_TIER_HI_M = 7.0, 25.0
# Service-scaled league-minimum floor (model dollars → millions): the bar's
# left edge, the blended-band floor, and the "≈ League min" test all use it.
_min_floor_M = prediction.get(
    "min_floor_dollars", 0.015 * SALARY_CAP_M.get(CONTRACT_SEASON, 165.0) * 1e6) / 1e6
_blended_toward_market = False
if (_market_median is not None
        and not prediction.get("cba_cap_applied")
        and not prediction.get("cba_floor_applied")
        and _BLEND_TIER_LO_M <= predicted_M <= _BLEND_TIER_HI_M):
    _mkt_M = _market_median / 1_000_000
    _hi = max(predicted_M, _mkt_M)
    _gap = abs(predicted_M - _mkt_M) / _hi if _hi > 0 else 0.0
    if _gap > 0.25 and _mkt_M > 0:
        # Once they diverge past 25%, give the market real weight: 0.35 at the
        # 25% threshold, ramping to 0.65 by a 60%+ gap. (A 47% Joe-style gap →
        # ~0.5 weight, pulling $18.8M + $12.8M to ~$15.8M.)
        _w_mkt = min(0.65, 0.35 + 0.30 * (_gap - 0.25) / 0.35)
        _blended_M = (1 - _w_mkt) * predicted_M + _w_mkt * _mkt_M
        # Never let a low market blend a projection BELOW the player's CBA
        # minimum (e.g. a fringe vet whose min comps median under his vet-min).
        _blended_M = max(_blended_M, _min_floor_M)
        if abs(_blended_M - predicted_M) > 0.05:
            _blended_toward_market = True
            # Recompute the band around the blended number using the same
            # tier-aware half-width the model uses (45%→12% by size).
            _bw = float(np.interp(_blended_M, [2.0, 8.0, 20.0, 45.0],
                                              [0.45, 0.33, 0.24, 0.12]))
            predicted_M = _blended_M
            low_M  = max(_min_floor_M, _blended_M * (1 - _bw))
            high_M = _blended_M * (1 + _bw)

# Confidence-bar scale: from the player's service-scaled league-min floor up to THIS
# player's OWN capped max — their 25/30/35% tier (prediction["cba_max_dollars"]).
# Anchoring the right edge to the player's personal ceiling means a maxed-out
# player (e.g. Luka, capped at the 30% max) fills the bar to the right edge,
# while everyone else reads as a fraction of their own max. Falls back to the
# 35% supermax ceiling if the player's max tier is unknown.
_scale_cap_M  = SALARY_CAP_M.get(CONTRACT_SEASON, 165.0)
_scale_min_M  = _min_floor_M   # service-scaled league minimum (see above)
_scale_max_M  = (prediction.get("cba_max_dollars") or 0.35 * _scale_cap_M * 1e6) / 1e6
# Current (stats-season) salary, drawn on the bar as a hollow ring so the gap to
# the model marker reads as the projected raise. None when no salary is on file.
_prev_sal_M     = float(features.get("salary") or 0) / 1e6
_prev_sal_M     = _prev_sal_M if _prev_sal_M > 0 else None
_prev_sal_label = f"{CURRENT_SEASON[2:]} Salary"   # "2025-26" -> "25-26 Salary"

# Initialize divergence so downstream code (confidence label, "Why this
# prediction" explainer) can reference it whether or not market data exists.
divergence = 0.0

# Was the projection floored at the league minimum (service-scaled — see
# _min_floor_M above)? Drives the "League minimum" tag beside the number and
# suppresses the misleading "+X% vs current deal" (rookie min → vet min is a
# CBA step-up, not a market raise).
_proj_is_min = bool(predicted_M <= _min_floor_M * 1.03)

# Compact CBA-status caption shown under the Model dollar (both hero
# variants — with market and without). Gives a one-line "why" at a
# glance; the full explanation lives in the About expander.
_supermax_label = prediction.get("supermax_tier_label", "")
if _proj_is_min:
    _model_caption = "League minimum"
elif prediction.get("cba_cap_applied"):
    _model_caption = f"Capped at max ({_supermax_label})"
elif prediction.get("cba_floor_applied"):
    _model_caption = f"Supermax floor ({_supermax_label})"
elif prediction.get("max_tier_floor_applied"):
    _model_caption = "All-NBA near-max"
elif features.get("on_rookie_scale"):
    _model_caption = "Currently on rookie scale"
elif _blended_toward_market:
    _model_caption = "Blended w/ market"
else:
    _model_caption = ""

# Inline caption chip shown beside the hero number (premium layout).
_caption_chip = (
    f'<span style="display:inline-block; margin-left:0.65rem; padding:0.18rem 0.65rem;'
    f' border-radius:999px; background:rgba(240,179,91,0.14); color:var(--amber);'
    f' font-size:0.72rem; font-weight:600; vertical-align:middle;'
    f' white-space:nowrap;">{_model_caption}</span>'
    if _model_caption else ''
)

# Was the player ALREADY earning a max-level salary? NBA max deals carry 8%
# annual raises, so a player re-upping at the max shows a year-1 figure BELOW a
# raised later year of his current max — a fake "pay cut". Detect max-to-max:
# his current salary ≥ 90% of his own max-tier dollars at THIS season's cap.
_cur_cap_M = SALARY_CAP_M.get(CURRENT_SEASON, 154.6)
_max_pct = float(prediction.get("max_pct", 0.30) or 0.30)
_prev_was_max = bool(
    _prev_sal_M and _max_pct and _prev_sal_M >= 0.90 * _max_pct * _cur_cap_M
)
_proj_is_max = bool(prediction.get("cba_cap_applied") or prediction.get("cba_floor_applied"))

# Raise-vs-current callout — projected $ vs the current/last deal. Green if a
# raise, red if a pay cut. BUT when both the current deal and the projection are
# the max, the year-over-year delta is just the 8% raise mechanics, not a real
# value change — show a neutral "Max ↔ Max" chip instead of a misleading cut.
_raise_html = ""
if _proj_is_min:
    # League minimum — the tag beside the number already says it; suppress the
    # misleading "+X% vs current deal" (rookie min → vet min is a CBA step-up,
    # not a market raise) by leaving this callout empty.
    _raise_html = ""
elif _prev_was_max and _proj_is_max:
    _raise_html = (
        '<div style="text-align:right; line-height:1.2;">'
        '<div style="display:inline-block; padding:0.28rem 0.7rem; border-radius:999px;'
        ' background:rgba(22,212,193,0.12); border:1px solid rgba(22,212,193,0.30);'
        ' color:var(--accent-teal); font-size:0.9rem; font-weight:800;'
        ' white-space:nowrap;">Max&nbsp;↔&nbsp;Max</div>'
        '<div style="font-size:0.72rem; color:var(--fg-4); margin-top:0.3rem;'
        ' max-width:15rem;">Last Season was Max Deal</div></div>'
    )
elif _prev_sal_M and _prev_sal_M > 0:
    _delta_M = predicted_M - _prev_sal_M
    _pct = (predicted_M / _prev_sal_M - 1) * 100
    _up = _delta_M >= 0
    _raise_html = (
        f'<div style="text-align:right; line-height:1.15;">'
        f'<div style="font-size:1.25rem; font-weight:800;'
        f' color:{"var(--value-good)" if _up else "var(--value-bad)"};">'
        f'{"▲" if _up else "▼"} {"+" if _up else "−"}${abs(_delta_M):.1f}M'
        f'<span style="font-size:0.7rem; color:var(--fg-4); font-weight:600;">/yr</span>'
        f'</div>'
        f'<div style="font-size:0.72rem; color:var(--fg-4); margin-top:0.15rem;">'
        f'{"+" if _up else ""}{_pct:.0f}% vs current deal (${_prev_sal_M:.1f}M)'
        f'</div></div>'
    )

# Score chip rendered next to the player name in the hero title. Pre-built
# here so the f-string template doesn't have any conditional logic that
# could trigger the markdown blank-line bug.
_score_chip_html = (
    f'<span style="display:inline-block; margin-left:0.7rem; '
    f'padding:0.18rem 0.6rem; border-radius:999px; '
    f'background:rgba(22,212,193,0.10); '
    f'border:1px solid rgba(22,212,193,0.30); '
    f'font-size:0.72rem; font-weight:700; color:var(--accent-teal); '
    f'letter-spacing:0.04em; vertical-align:4px;">'
    f'Score {features["barrett_score"]:.1f} (#{features["score_rank"]})'
    f'</span>'
)

# Informational "current deal" / "previous salary" context line.
# Three cases:
#   1. Multi-year contract with years remaining → "Current deal: $X through YYYY-YY (option)"
#   2. Final year of multi-year deal or no contract on file → "Previous: $X (last season) · FA next summer"
#   3. No current salary on file → empty (very rare — e.g. unsigned rookie)
_contract_end = features.get("contract_end_season") or ""
_last_type = features.get("contract_last_year_type") or ""
_cur_sal_dollars = float(features.get("salary") or 0)
_signing_html = ""
if _contract_end and _contract_end != CURRENT_SEASON:
    # Case 1: multi-year deal with years remaining.
    _type_blurb = {
        "player_option":  " (player option)",
        "team_option":    " (team option)",
        "et_option":      " (early termination option)",
    }.get(_last_type, "")
    _sal_str = f'{_fmt_money(_cur_sal_dollars)} ' if _cur_sal_dollars > 0 else ''
    _signing_html = (
        f'<div style="margin-top:0.35rem; font-size:0.74rem; color:var(--fg-4);">'
        f'Current deal: {_sal_str}through {_contract_end}{_type_blurb}'
        f'</div>'
    )
elif _cur_sal_dollars > 0:
    # Case 2: free agent this offseason (or final year of expiring deal).
    # Show their current/previous salary so users have a baseline.
    _signing_html = (
        f'<div style="margin-top:0.35rem; font-size:0.74rem; color:var(--fg-4);">'
        f'Previous: {_fmt_money(_cur_sal_dollars)} ({CURRENT_SEASON}) · '
        f'free agent next summer'
        f'</div>'
    )

if _market_median is not None:
    market_M = _market_median / 1_000_000
    # Fold the market into the confidence band: equal to the model band when
    # model and market agree, widened to span both when they diverge — so the
    # band width *is* the model-vs-market uncertainty. No separate market marker.
    _band_lo_M = min(low_M, market_M)
    _band_hi_M = max(high_M, market_M)
    # Honest range = min(model, market) → max(model_high, market)
    honest_low_M  = min(predicted_M, market_M)
    honest_high_M = max(predicted_M, market_M)
    # Note when model vs market diverge by >40% — flag it for the user.
    divergence = (
        abs(predicted_M - market_M) / max(predicted_M, market_M)
        if max(predicted_M, market_M) > 0 else 0
    )
    diverge_note = (
        f'<div style="margin-top:0.7rem; padding-top:0.6rem; '
        f'border-top:1px solid var(--hairline); '
        f'font-size:0.78rem; color:var(--orange);">'
        f'⚠ Model and market diverge by {divergence*100:.0f}%. '
        f'Use as a range.'
        f'</div>'
        if divergence >= 0.40 else ''
    )
    # Compact player metadata line. Player name + Score appear in the
    # hero title above. Current salary moves to the signing-window line
    # below (combined with the contract end year). So the meta line is
    # just the biographic context: team · season · age · position · draft.
    _meta_bits = []
    if features.get("team"): _meta_bits.append(features["team"])
    _meta_bits.append(CURRENT_SEASON)
    if features.get("age"): _meta_bits.append(f"Age {int(features['age'])}")
    _meta_bits.append(str(features.get("position_detailed", features["position"])))
    _draft_label = _fmt_draft(features)
    if _draft_label:
        _meta_bits.append(_draft_label)
    _player_meta_line = " · ".join(_meta_bits)
    _header_html = f"""
    <div style="background:linear-gradient(135deg, rgba(230,57,70,0.07) 0%, rgba(22,212,193,0.06) 100%);
                border:1px solid var(--hairline); border-radius:16px;
                padding:1.5rem 1.9rem 1.7rem; margin: 0.5rem 0 1.3rem 0;">
      <div style="font-size:1.5rem; color:var(--fg-1); font-weight:800; line-height:1.2;">
        {features["name"]}{_score_chip_html}
      </div>
      <div style="font-size:0.7rem; color:var(--fg-4); text-transform:uppercase;
                  letter-spacing:0.12em; font-weight:600; margin-top:0.45rem;">
        Projected {CONTRACT_SEASON} contract
      </div>
      <div style="font-size:0.78rem; color:var(--fg-3); margin-top:0.15rem;">
        {_player_meta_line}
      </div>{_signing_html}
      <div style="margin-top:1.05rem; display:flex; align-items:flex-end; justify-content:space-between; flex-wrap:wrap; gap:0.8rem;">
        <div style="line-height:0.9;"><span style="font-size:3.4rem; font-weight:800; color:var(--fg-1); letter-spacing:-0.02em;">${predicted_M:.1f}M</span><span style="font-size:0.95rem; color:var(--fg-4); font-weight:600;"> /yr</span>{_caption_chip}</div>{_raise_html}
      </div>
      {_confidence_bar_html(predicted_M, _band_lo_M, _band_hi_M, scale_min_M=_scale_min_M, scale_max_M=_scale_max_M, tertiary_M=_prev_sal_M, tertiary_label=_prev_sal_label)}
      <div style="margin-top:0.95rem; font-size:0.82rem; color:var(--fg-4);">
        Market second opinion
        <span style="color:var(--fg-6);">(median of comparable signings)</span>:
        &nbsp;<b style="color:var(--accent-teal); font-size:0.95rem;">${market_M:.1f}M</b>
      </div>{diverge_note}
    </div>
    """
else:
    # No comparables available — fall back to the model-only display.
    # Player name + Score appear in the hero title; meta line has the
    # biographic context only (matches the market-data branch above).
    _meta_bits = []
    if features.get("team"): _meta_bits.append(features["team"])
    _meta_bits.append(CURRENT_SEASON)
    if features.get("age"): _meta_bits.append(f"Age {int(features['age'])}")
    _meta_bits.append(str(features.get("position_detailed", features["position"])))
    _draft_label = _fmt_draft(features)
    if _draft_label:
        _meta_bits.append(_draft_label)
    _player_meta_line = " · ".join(_meta_bits)

    # Two sub-cases for the no-market display:
    #   - Suppressed: queried player is supermax-tier but the comparables
    #     pool is too thin/biased to produce a meaningful Market number
    #     (Curry-tier — the "stayed at supermax into late 30s" cohort
    #     barely exists in our 3-year window). Anchor the range on
    #     current salary instead.
    #   - Generic: no comparables on disk at all (rookies, niche profiles).
    if _market_suppressed and float(features.get("salary", 0) or 0) > 0:
        cur_sal = float(features["salary"])
        cur_sal_M = cur_sal / 1_000_000
        _explainer = (
            'Market view unavailable — the queried player is currently '
            'supermax-tier, but the historical comparables pool for that '
            'cohort is too sparse (5+ Paycut signings, no "stayed at tier" '
            'comps). Anchoring the range on current salary instead.'
        )
        _header_html = f"""
        <div style="background:linear-gradient(135deg, rgba(230,57,70,0.07) 0%, rgba(22,212,193,0.06) 100%);
                    border:1px solid var(--hairline); border-radius:16px;
                    padding:1.5rem 1.9rem 1.7rem; margin: 0.5rem 0 1.3rem 0;">
          <div style="font-size:1.5rem; color:var(--fg-1); font-weight:800; line-height:1.2;">
            {features["name"]}{_score_chip_html}
          </div>
          <div style="font-size:0.7rem; color:var(--fg-4); text-transform:uppercase;
                      letter-spacing:0.12em; font-weight:600; margin-top:0.45rem;">
            Projected {CONTRACT_SEASON} contract
          </div>
          <div style="font-size:0.78rem; color:var(--fg-3); margin-top:0.15rem;">
            {_player_meta_line}
          </div>{_signing_html}
          <div style="margin-top:1.05rem; display:flex; align-items:flex-end; justify-content:space-between; flex-wrap:wrap; gap:0.8rem;">
            <div style="line-height:0.9;"><span style="font-size:3.4rem; font-weight:800; color:var(--fg-1); letter-spacing:-0.02em;">${predicted_M:.1f}M</span><span style="font-size:0.95rem; color:var(--fg-4); font-weight:600;"> /yr</span>{_caption_chip}</div>{_raise_html}
          </div>
          {_confidence_bar_html(predicted_M, low_M, high_M, cur_sal_M, "#16d4c1", "current", scale_min_M=_scale_min_M, scale_max_M=_scale_max_M)}
          <div style="margin-top:0.95rem; font-size:0.82rem; color:var(--fg-4);">
            Current salary anchor:
            &nbsp;<b style="color:var(--accent-teal); font-size:0.95rem;">${cur_sal_M:.1f}M</b>
          </div>
          <div style="margin-top:0.7rem; padding-top:0.6rem;
                      border-top:1px solid var(--hairline);
                      font-size:0.78rem; color:var(--orange);">
            ⚠ {_explainer}
          </div>
        </div>
        """
    else:
        _header_html = f"""
        <div style="background:linear-gradient(135deg, rgba(230,57,70,0.07) 0%, rgba(22,212,193,0.06) 100%);
                    border:1px solid var(--hairline); border-radius:16px;
                    padding:1.5rem 1.9rem 1.7rem; margin: 0.5rem 0 1.3rem 0;">
          <div style="font-size:1.5rem; color:var(--fg-1); font-weight:800; line-height:1.2;">
            {features["name"]}{_score_chip_html}
          </div>
          <div style="font-size:0.7rem; color:var(--fg-4); text-transform:uppercase;
                      letter-spacing:0.12em; font-weight:600; margin-top:0.45rem;">
            Projected {CONTRACT_SEASON} contract
          </div>
          <div style="font-size:0.78rem; color:var(--fg-3); margin-top:0.15rem;">
            {_player_meta_line}
          </div>{_signing_html}
          <div style="margin-top:1.05rem; display:flex; align-items:flex-end; justify-content:space-between; flex-wrap:wrap; gap:0.8rem;">
            <div style="line-height:0.9;"><span style="font-size:3.4rem; font-weight:800; color:var(--fg-1); letter-spacing:-0.02em;">${predicted_M:.1f}M</span><span style="font-size:0.95rem; color:var(--fg-4); font-weight:600;"> /yr</span>{_caption_chip}</div>{_raise_html}
          </div>
          {_confidence_bar_html(predicted_M, low_M, high_M, scale_min_M=_scale_min_M, scale_max_M=_scale_max_M, tertiary_M=_prev_sal_M, tertiary_label=_prev_sal_label)}
          <div style="margin-top:0.95rem; font-size:0.78rem; color:var(--fg-4);">
            Model prediction only — no comparable signings on file.
          </div>
        </div>
        """
st.markdown(_header_html, unsafe_allow_html=True)

# ── Structural caveats — compact chip-style instead of full-width banners ────
# Playoff multiplier display moved to the About expander (visible in the
# math line). Removed from the main view so the hero stays clean — the
# multiplier already affects the dollar amount; redundant to show it
# again on the front.
#
# Supermax-eligible caveat is special-cased: the chip just says
# "Supermax-eligible" with the full description as a tooltip (title attr)
# you see on hover. Other caveats render as full-text yellow chips.
if caveats:
    _chip_base = (
        'display:inline-block; background:rgba(240,179,91,0.12); '
        'border:1px solid rgba(240,179,91,0.30); border-radius:999px; '
        'padding:0.32rem 0.85rem; margin: 0 0.45rem 0.45rem 0; '
        'font-size:0.78rem; font-weight:600; color:var(--amber);'
    )

    def _caveat_chip(note: str) -> str:
        if note.startswith("Supermax-eligible:"):
            # Split short label and full detail for hover tooltip.
            detail = note[len("Supermax-eligible: "):].strip()
            # Escape the quote in the detail so it survives in the
            # HTML title attribute (no inner double quotes).
            safe_detail = detail.replace('"', '&quot;')
            return (
                f'<div title="{safe_detail}" '
                f'style="{_chip_base} cursor:help;">⚠ Supermax-eligible</div>'
            )
        return f'<div style="{_chip_base}">⚠ {note}</div>'

    _caveat_chips_html = "".join(_caveat_chip(n) for n in caveats)
    st.markdown(
        f'<div style="margin: 0.4rem 0 1rem 0;">{_caveat_chips_html}</div>',
        unsafe_allow_html=True,
    )

# Math breakdown lives inside the "About this prediction" expander below —
# keeps the main view clean (Model / Market / Honest range) while still
# letting curious users see the per-player calculation.

# "Why this prediction" plain-English bullets live inside the About
# expander below. The main view stays clean — the CBA-status caption
# under the Model dollar gives a one-line summary; the expander has
# the full reasoning for curious users.
_explain_bullets = explain_prediction(
    features, prediction, _market_median, divergence,
)

# ── Comparables ──────────────────────────────────────────────────────────────
st.subheader("Comparable signings")
st.caption(
    "Closest matches on trailing-weighted Barrett (last 3 healthy years, "
    "50/30/20) + age + position."
)

# Reuse the comparables we already loaded at the top for the hero card.
history = _history
if history.empty:
    st.info("No historical comparables on disk yet.")
else:
    comps = _comps
    if comps.empty:
        st.info("No close comparables found.")
    else:
        # ── Scouting take card ──────────────────────────────────────────────
        take = _scouting_take(features, comps)
        if take:
            top3_str = ", ".join(take["top3"])
            scouting_html = f"""
            <div style="background:linear-gradient(135deg, rgba(22,212,193,0.07) 0%, rgba(22,212,193,0.03) 100%);
                        border:1px solid rgba(22,212,193,0.22);
                        border-radius:14px; padding:1.2rem 1.5rem; margin:0.4rem 0 1.1rem 0;">
              <div style="font-size:0.7rem; color:var(--accent-teal); letter-spacing:0.12em;
                          text-transform:uppercase; font-weight:700; margin-bottom:0.85rem;">
                Scouting take · market view
              </div>
              <div style="display:flex; flex-wrap:wrap; gap:2rem; margin-bottom:0.7rem;
                          align-items:flex-end;">
                <div>
                  <div style="font-size:0.66rem; color:var(--fg-4);
                              text-transform:uppercase; letter-spacing:0.08em; font-weight:600;">
                    Median new contract
                  </div>
                  <div style="font-size:1.7rem; color:var(--fg-1); font-weight:800;
                              line-height:1.1; margin-top:0.15rem;">
                    ${take['median']/1e6:.1f}M
                  </div>
                </div>
                <div>
                  <div style="font-size:0.66rem; color:var(--fg-4);
                              text-transform:uppercase; letter-spacing:0.08em; font-weight:600;">
                    Middle 50%
                  </div>
                  <div style="font-size:1.7rem; color:var(--fg-2); font-weight:700;
                              line-height:1.1; margin-top:0.15rem;">
                    ${take['q25']/1e6:.1f}<span style="color:var(--fg-6); font-weight:500;">–</span>${take['q75']/1e6:.1f}M
                  </div>
                </div>
                <div style="flex:1; min-width:240px;">
                  <div style="font-size:0.66rem; color:var(--fg-4);
                              text-transform:uppercase; letter-spacing:0.08em; font-weight:600;">
                    Closest 3 comps
                  </div>
                  <div style="font-size:0.95rem; color:var(--fg-2); margin-top:0.3rem;
                              line-height:1.4;">{top3_str}</div>
                </div>
              </div>
              <div style="font-size:0.84rem; color:var(--fg-4);
                          border-top:1px solid var(--hairline);
                          padding-top:0.6rem; margin-top:0.2rem;">
                <b style="color:var(--accent-teal);">Note</b>&nbsp; {take['x_factor']}
              </div>
            </div>
            """
            st.markdown(scouting_html, unsafe_allow_html=True)

        # ── Comparables table with Context column ──────────────────────────
        # `comps` already has the context column attached up at the
        # market-median computation site — reuse it instead of recomputing.
        comps_with_ctx = (
            comps.copy() if "context" in comps.columns
            else comps.assign(context=comps.apply(_classify_context, axis=1))
        )

        _pos_col = comps_with_ctx.get("pos_detailed", comps_with_ctx["pos"]).fillna(
            comps_with_ctx["pos"])
        # Final normalize: any remaining "Guard"/"Forward"/"Center" → G/F/C.
        # Belt-and-suspenders — the resolvers in load_historical_signings
        # should already produce single-letter, but if anything slips
        # through (cached row, edge case) we still display consistently.
        _pos_col = _pos_col.map(_pos_abbrev)

        # Build a draft column. Use tier alone (compact) — full pick number
        # would crowd the table. Fall back to "—" when unknown.
        def _comp_draft_label(t, p):
            if not isinstance(t, str) or not t:
                return "—"
            if t == "Undrafted":
                return "Undrafted"
            try:
                p_int = int(p) if p is not None and not pd.isna(p) else None
            except (TypeError, ValueError):
                p_int = None
            return f"{t} (#{p_int})" if p_int else t

        _draft_col = [
            _comp_draft_label(t, p)
            for t, p in zip(
                comps_with_ctx.get("draft_tier", pd.Series([], dtype=str)),
                comps_with_ctx.get("draft_pick", pd.Series([], dtype="object")),
            )
        ]

        comp_disp = pd.DataFrame({
            "Player":         comps_with_ctx["Player"].values,
            "Signed in":      comps_with_ctx["signed_in"].values,
            "Age then":       comps_with_ctx["age"].astype(int).values,
            "Position":       _pos_col.values,
            "Draft":          _draft_col,
            "Context":        comps_with_ctx["context"].values,
            "Career Score":   comps_with_ctx["career_weighted_barrett"].round(1).values,
            "Sign-yr Score":  comps_with_ctx["barrett_score"].round(1).values,
            "Signed for":     [_fmt_money(v) for v in comps_with_ctx["salary_curr"]],
        })
        html_table(
            comp_disp,
            formatters={
                "Age then":      lambda v: str(int(v)),
                "Career Score":  lambda v: f"{v:.1f}",
                "Sign-yr Score": lambda v: f"{v:.1f}",
            },
            aligns={"Age then": "right", "Career Score": "right",
                    "Sign-yr Score": "right", "Signed for": "right"},
            numeric={"Age then", "Career Score", "Sign-yr Score"},
            helps={
                "Career Score": "Trailing-weighted Barrett (last 3 healthy years, 50/30/20).",
                "Sign-yr Score": "Barrett Score in the season they signed.",
                "Signed for": "Actual deal in its signing year (not cap-adjusted).",
                "Context": "Supermax · Free-agent raise · Rookie extension · Paycut.",
            },
            height=min(440, 60 + len(comp_disp) * 38),
        )

        st.caption(
            f"\"Signed for\" shows each comp's ACTUAL deal in its signing year; the "
            f"market median above cap-adjusts them to {CONTRACT_SEASON} dollars so it's "
            f"comparable to the model. "
            "Context tags: **Supermax** ≥28% cap · **Free-agent raise** ≥15% bump · "
            "**Rookie extension** first non-rookie deal · **Paycut** took less to stay. "
            "Draft tiers: Lottery (1-14) · Mid-1st (15-22) · Late-1st (23-30) · 2nd (31-60) · Undrafted."
        )

# ── Likely suitors (experimental) ───────────────────────────────────────────
# Hidden until data/team_landscape_2026.csv has real cap data entered. "Need" is
# computed LIVE — the player's Barrett vs each team's depth chart at his position
# (build_rosters from current_ranked); affordability comes from the CSV. Fully
# wrapped in try/except so a data/import hiccup can never break the page.
_ts_about_note = None   # suitors methodology blurb → rendered in the About expander
try:
    import team_suitors as _ts
    _ts_land = _ts.load_team_landscape()
    if _ts.landscape_is_filled(_ts_land):
        # current_ranked has Player/Team/barrett_score but NO position column —
        # join the season's detailed positions (keyed by normalized name) so the
        # roster can be bucketed by position.
        _cr = current_ranked.copy()
        try:
            _posmap = fetch_player_positions_detailed(CURRENT_SEASON, cache_v=3) or {}
        except Exception:
            _posmap = {}
        # Positions: NBA 2K26 primary/secondary (+ user overrides), with the BBRef
        # single position as the fallback for anyone 2K doesn't list. This is what
        # lets a SG/SF wing compete at both spots and a pure PG stay PG-only.
        _pos2k = _ts.load_player_positions()
        _cr["pos"] = _cr["Player"].map(
            lambda _p: _ts.resolve_position(_p, _posmap.get(normalize(_p), ""), _pos2k))
        # his current team — captured BEFORE we drop him — so it can appear as a
        # Bird-rights re-signer (a team can always re-sign its own free agent)
        _self_norm = normalize(features.get("name", ""))
        _self_rows = _cr[_cr["Player"].map(normalize) == _self_norm]
        _self_team = str(_self_rows.iloc[0]["Team"]) if not _self_rows.empty else None
        # restricted FA (his team can match any offer) if he's finishing a rookie-scale deal
        _is_rfa = bool(features.get("on_rookie_scale"))
        # skill-fit layer: does a team need a guy who SHOOTS / REBOUNDS / PASSES / DEFENDS?
        _skill_fit = None
        try:
            _box = fetch_league_stats(CURRENT_SEASON, "Regular Season")
            _adv = fetch_advanced_stats(CURRENT_SEASON, "Regular Season")
            _team_sk = _ts.build_team_skills(_box, _adv)
            _self_pid = (int(_self_rows.iloc[0]["PLAYER_ID"])
                         if (not _self_rows.empty and "PLAYER_ID" in _self_rows.columns) else None)
            if _self_pid is not None:
                _skill_fit = _ts.skill_fit_scores(
                    _ts.player_skills(_self_pid, _box, _adv), _team_sk)
        except Exception:
            _skill_fit = None
        # drop the player himself so his own team isn't shown "upgrading over" him
        _cr = _cr[_cr["Player"].map(normalize) != _self_norm]
        _ts_rost = _ts.build_rosters(_cr)
        _ts_pos = _ts.resolve_position(
            features.get("name", ""),
            features.get("position_detailed") or features.get("position") or "",
            _pos2k)
        _ts_suitors = (
            _ts.rank_suitors(predicted_M, float(features["barrett_score"]),
                             _ts_pos, _ts_rost, _ts_land, n=6,
                             incumbent_team=_self_team,
                             age=features.get("age"), is_rfa=_is_rfa,
                             skill_fit=_skill_fit)
            if not _ts_rost.empty else []
        )
        if _ts_suitors:
            _ts_rows = "".join(
                f'<div style="display:flex; align-items:baseline; gap:0.7rem; padding:0.45rem 0; '
                f'border-top:1px solid var(--hairline);">'
                f'<span style="font-weight:800; color:var(--accent-teal); width:2.6rem;">{_s["team"]}</span>'
                f'<span style="font-weight:800; color:var(--fg-2); width:3.6rem; '
                f'white-space:nowrap;">${_s["offer_M"]:.0f}M</span>'
                f'<span style="color:var(--fg-2); font-size:0.85rem;">{_s["reason"]}</span>'
                f'<span style="margin-left:auto; color:var(--fg-5); font-size:0.76rem; '
                f'white-space:nowrap;">{_s["tool"]}</span></div>'
                for _s in _ts_suitors
            )
            st.markdown(
                '<div style="background:linear-gradient(135deg, rgba(230,57,70,0.07) 0%, '
                'rgba(22,212,193,0.06) 100%); border:1px solid var(--hairline); '
                'border-radius:14px; padding:1.1rem 1.4rem; margin:0.4rem 0 0.2rem;">'
                '<div style="font-size:0.7rem; color:var(--accent-teal); letter-spacing:0.12em; '
                'text-transform:uppercase; font-weight:700; margin-bottom:0.5rem;">'
                f'Likely suitors — projected offers (model value ${predicted_M:.0f}M)</div>{_ts_rows}</div>',
                unsafe_allow_html=True,
            )
            # Explanatory blurb is rendered in the About expander below (keeps
            # the suitors card itself clean) — see _ts_about_note.
            _asof = str(getattr(_ts_land, "attrs", {}).get("as_of", "")).strip()
            _fa_label = ("restricted FA — his team can match any offer"
                         if _is_rfa else "unrestricted FA")
            _ts_about_note = (
                f"Experimental — {_fa_label}. Each team's number is the model's value "
                "scaled by fit (starter vs depth) and capped by their cap room / exception "
                "(or Bird rights). Ranked by a model trained on 1,800+ historical signings "
                "(incumbency, cap, position, team timeline × his age & value) — it places the "
                "actual signing team in this shortlist ~6 of 10 times; skill fit is shown as a label."
                + (f" Cap data: {_asof}." if _asof else "")
            )
except Exception:
    pass

# ── Methodology footer (collapsed — info-after-action) ──────────────────────
st.divider()
with st.expander("About this prediction"):
    # Concise model summary up top — the quick "what is this" before the
    # player-specific reasoning and the deep methodology below.
    st.markdown(
        "A machine-learning model (HistGBM) trained on 1,900+ modern-era "
        "contracts (2012+), built on the Barrett Score plus age, position, "
        "service years, All-NBA history, and advanced metrics (usage, PIE, "
        "on/off rating). Validated by temporal cross-validation on five "
        "seasons of real signings (2021-25). In dollar terms it's most "
        "accurate for role and rotation players — typically within $2-4M — "
        "and looser for stars, where it tends to under-shoot by ~$5M because "
        "max-salary rules, cap space, and deal timing drive those contracts "
        "more than on-court production. Read star and max figures as a "
        "ballpark, not a quote."
    )

    # Likely-suitors methodology — only when that (experimental) section
    # actually rendered above for this player.
    if _ts_about_note:
        st.markdown(f"### Likely suitors\n\n{_ts_about_note}")

    # ── Plain-English explanation (player-specific) ─────────────────────────
    # Tailored bullets describing what drove the dollar amount and how
    # model vs market compare. Sits at the top of the expander so curious
    # users see the "why" before the math.
    if _explain_bullets:
        _explain_md = "\n\n".join(_explain_bullets)
        st.markdown(
            f"### Why this prediction\n\n{_explain_md}\n\n---"
        )

    # ── Per-player math breakdown ───────────────────────────────────────────
    # Same one-line equation that used to live in the main view. Moved here
    # so the predicted-contract hero is the focal point of the page; the
    # math is for curious users who want to see how the number was built.
    base_M = prediction["base"] / 1_000_000
    _pos_factor_note = (
        f" (suppressed from ×{prediction['pos_mult_raw']:.2f} — base ≥28% of cap)"
        if prediction.get("pos_mult_suppressed") else ""
    )
    # Annotate the age multiplier with the tier label so the user sees WHY
    # Harden's 36-year-old multiplier is gentler than an average 36yo's.
    _age_factor_note = (
        f" · {prediction['age_tier']} tier"
        if prediction.get("age_tier") and prediction["age_tier"] not in
           ("Prime (≤28)", "Unknown") else ""
    )
    # Durability multiplier only shown in the math line when it's actually
    # moving the number — Healthy → ×1.00 means no change, no need to clutter.
    _dur_mult = prediction.get("durability_mult", 1.0) or 1.0
    _dur_tier = prediction.get("durability_tier", "") or ""
    _show_durability = _dur_mult != 1.0
    _dur_html_fragment = (
        f'&nbsp;<span style="color:var(--fg-6);">×</span>&nbsp;'
        f'<b>×{_dur_mult:.2f}</b>'
        f'<span style="color:var(--fg-5);"> (durability: {_dur_tier} · '
        f'{features.get("trailing_gp_total", 0)}/{features.get("trailing_gp_max", 246)} '
        f'GP over last 3 yrs)</span>'
        if _show_durability else ""
    )
    # Playoff multiplier — only shown when ≠ 1.00 (i.e., player earned a bonus).
    _playoff_mult = prediction.get("playoff_mult", 1.0) or 1.0
    _playoff_tier = prediction.get("playoff_tier", "") or ""
    _playoff_gp_target = features.get("playoff_gp", 0)
    _playoff_barrett_target = features.get("playoff_barrett", 0.0)
    _show_playoff = _playoff_mult != 1.0
    _playoff_html_fragment = (
        f'&nbsp;<span style="color:var(--fg-6);">×</span>&nbsp;'
        f'<b>×{_playoff_mult:.2f}</b>'
        f'<span style="color:var(--fg-5);"> (playoff: {_playoff_tier} · '
        f'Barrett {_playoff_barrett_target:.1f} on {_playoff_gp_target} GP last postseason)</span>'
        if _show_playoff else ""
    )

    # Build the math line as a single-line HTML string. Multi-line f-strings
    # of HTML have caused rendering bugs (Streamlit's markdown parser sees a
    # blank line — produced when _dur_html_fragment is empty — and switches
    # into code-block mode for everything after it). Single-line construction
    # avoids the issue entirely.
    _age_label = int(features['age']) if features['age'] else '?'
    _pos_label = features.get('position_detailed', features['position'])

    # CBA cap/floor adjustment — show a final "→ adjusted to $X" step when
    # the raw model was overridden by CBA rules. Math equation shows the
    # raw model result before the override; the → arrow shows the override.
    _raw_predicted_M = (prediction.get("raw_predicted", predicted_M * 1e6)) / 1e6
    _cba_max_M = prediction.get("cba_max_dollars", 0) / 1e6
    _cba_cap_applied = prediction.get("cba_cap_applied", False)
    _cba_floor_applied = prediction.get("cba_floor_applied", False)
    _max_tier_floor_applied = prediction.get("max_tier_floor_applied", False)
    _supermax_tier_label = prediction.get("supermax_tier_label", "")
    if _cba_cap_applied:
        _cba_fragment = (
            f' &nbsp;<span style="color:var(--fg-6);">→</span>&nbsp; '
            f'<b style="color:var(--accent-red);">capped at ${_cba_max_M:.1f}M</b>'
            f' <span style="color:var(--fg-5);">(CBA max: {_supermax_tier_label})</span>'
        )
    elif _cba_floor_applied:
        _cba_fragment = (
            f' &nbsp;<span style="color:var(--fg-6);">→</span>&nbsp; '
            f'<b style="color:var(--accent-teal);">floored at ${_cba_max_M:.1f}M</b>'
            f' <span style="color:var(--fg-5);">(supermax: {_supermax_tier_label})</span>'
        )
    elif _max_tier_floor_applied:
        _cba_fragment = (
            f' &nbsp;<span style="color:var(--fg-6);">→</span>&nbsp; '
            f'<b style="color:var(--accent-teal);">lifted to ${predicted_M:.1f}M</b>'
            f' <span style="color:var(--fg-5);">(All-NBA near-max)</span>'
        )
    else:
        _cba_fragment = ""

    # When CBA applies, the equation should end at the RAW model result so
    # the user can see what the box-score-driven model said before override.
    # When CBA doesn't apply, the equation ends at the final prediction.
    _equation_end_M = _raw_predicted_M if (_cba_cap_applied or _cba_floor_applied) else predicted_M

    if prediction.get("model_used") == "HistGBM v2":
        # ML-model variant: no multiplicative breakdown — show the model's
        # raw output plus inputs that drove it, then any CBA override.
        _ml_inputs = (
            f'career rate {features["career_barrett"]:.1f} · rank #{features["effective_rank"]}'
        )
        if features.get("all_nba_3yr"):
            _ml_inputs += f' · All-NBA last 3yr: {features["all_nba_3yr"]}'
        _ml_inputs += f' · age {features["age"] or "?"} · {features["service_years"]} yrs service'
        _math_line = (
            '<span style="color:var(--fg-4); font-size:0.7rem; letter-spacing:0.08em;'
            ' text-transform:uppercase; margin-right:0.5rem;">Model</span>'
            f'<b style="color:var(--fg-1);">${_raw_predicted_M:.1f}M</b>'
            f' <span style="color:var(--fg-5);">(HistGBM ML output from {_ml_inputs})</span>'
            f' &nbsp;<span style="color:var(--fg-6);">→</span>&nbsp; '
            f'<b style="color:var(--accent-teal);">${_equation_end_M:.1f}M</b>'
            f' <span style="color:var(--fg-6);">±${prediction["band"]/1_000_000:.1f}M</span>'
            f'{_cba_fragment}'
        )
    else:
        _math_line = (
            '<span style="color:var(--fg-4); font-size:0.7rem; letter-spacing:0.08em;'
            ' text-transform:uppercase; margin-right:0.5rem;">Math</span>'
            f'<b style="color:var(--fg-1);">${base_M:.1f}M</b>'
            f' <span style="color:var(--fg-5);">(career rate score '
            f'{features["career_barrett"]:.1f} → rank #{features["effective_rank"]})'
            f'</span> &nbsp;<span style="color:var(--fg-6);">×</span>&nbsp; '
            f'<b>×{prediction["age_mult"]:.2f}</b>'
            f' <span style="color:var(--fg-5);">(age {_age_label}{_age_factor_note})</span>'
            f' &nbsp;<span style="color:var(--fg-6);">×</span>&nbsp; '
            f'<b>×{prediction["pos_mult"]:.2f}</b>'
            f' <span style="color:var(--fg-5);">({_pos_label}{_pos_factor_note})</span>'
            f'{_dur_html_fragment}'
            f'{_playoff_html_fragment}'
            f' &nbsp;<span style="color:var(--fg-6);">=</span>&nbsp; '
            f'<b style="color:var(--accent-teal);">${_equation_end_M:.1f}M</b>'
            f' <span style="color:var(--fg-6);">±${prediction["band"]/1_000_000:.1f}M</span>'
            f'{_cba_fragment}'
        )
    _breakdown_html = (
        '<div style="background:var(--hairline-soft);'
        ' border:1px solid var(--hairline);'
        ' border-radius:10px; padding:0.85rem 1.1rem; margin: 0 0 1rem 0;'
        ' font-size:0.95rem; color:var(--fg-2); line-height:1.55;">'
        f'{_math_line}'
        '<div style="font-size:0.72rem; color:var(--fg-5); margin-top:0.35rem;">'
        f'Base uses {features["career_basis"]}.'
        '</div>'
        '</div>'
    )
    st.markdown(_breakdown_html, unsafe_allow_html=True)

    render_barrett_score_explainer()
    st.markdown(
        """
        ### How the next contract is predicted
        Layers stack on top of the Barrett Score, then CBA rules constrain
        the final number:

        1. **Career-weighted Rate Score** — uses a weighted average of
           the player's last 3 **healthy seasons** (GP ≥ 40), with 50/30/20
           weighting (most recent first). The **rate score** is the
           Barrett Score with the availability multiplier divided out —
           i.e. what the player produces per game when on the floor,
           regardless of how many games they played. GMs negotiate AAV
           based on rate stats; durability is handled separately via
           contract length and structure. Without this, a 41-game Curry
           season looks like Rotation-tier production when on rate he's
           still Elite.
        2. **Base projection** — what the player at that career-weighted rank
           would earn based on the current season's salary distribution.
        3. **Age multiplier** — fit on 2014-22 real new contracts. A 33yo
           signs for ~28% less than a 27yo at the same Barrett Score.
        4. **Position multiplier** — Centers are systematically overprojected
           by the box-score-heavy Barrett Score (rebounds aren't paid like
           points). **Suppressed at the supermax tier** (base ≥28% of cap)
           since max-contract players sign at fixed CBA percentages
           regardless of position.
        5. **Durability multiplier** — trailing-3-year availability tier
           (Healthy / Mild / Moderate / Chronic / Severe). Embiid's chronic
           injury history applies ~0.78× even though his rate is elite.
        6. **Playoff bonus** — based on the player's **most recent
           qualifying playoff appearance** (≥4 GP filters out cameos).
           GMs negotiate off the freshest playoff impression, not a
           multi-year average — Bruce Brown after BKN, Wiggins after the
           GSW title, Rui after the LAL WCF run all got paid off ONE
           playoff run. Tiers: Elite ≥31 Barrett (×1.15), Strong ≥24
           (×1.10), Solid ≥16 (×1.05). One-way bonus — no penalty for
           lottery-team players who can't earn postseason reps.
        7. **CBA max cap** — derived from years of NBA service. 0-6 yrs:
           25% of cap. 7-9 yrs: 30%. 10+ yrs: 35%. Caps the projection
           because no player can legally earn more than their max.
        8. **Supermax floor** — for players with recent All-NBA selections
           AND tenure with their current team (Designated Vet at 35%,
           Designated Rookie at 30%). Elite stars in their prime almost
           universally take the max they're offered. Floor disabled for
           aging vets (age >33+) who routinely take paycuts.

        **Confidence band:** ±$5.9M (≈3.6% of the 2026-27 cap).

        ### What's in the model
        - Production (Barrett Score, healthy-season trailing average)
        - Age (with tier-aware decline curve)
        - Position (G/F/C bucket)
        - Durability (last 3 yrs GP)
        - **Playoff performance** (trailing-3-postseason Barrett tier bonus)
        - Draft pedigree (lottery / mid-1st / late-1st / 2nd / undrafted)
        - **All-NBA selections** (scraped from BBRef awards page)
        - **NBA service years** (derived from career data)
        - **Years with current team** (Bird-rights proxy via consecutive
          seasons on the same team — derived from career data)
        - **Rookie-scale lock** (uses our existing rookie-scale roster)
        - **CBA max-contract tiers** (25% / 30% / 35% based on service)
        - **Designated Rookie / Designated Vet (supermax) eligibility**
        - **Advanced metrics** (usage rate, PIE, on/off net rating, true
          shooting) — possession- and impact-level signal the box score
          alone misses.

        ### What's not in the model yet
        - **Detailed Bird rights** (we approximate via team tenure; the
          real CBA distinguishes Early-Bird / Non-Bird / Full Bird).
        - **Team-by-team cap space** (affects which team can offer, less
          so league-wide AAV).
        - **Agent identity / negotiating leverage** (public but hard to
          quantify).
        - **Off-court marketability** (jersey sales, brand value).
        - **Future production** — we project from recent past; nobody
          knows next year's box score.

        The prediction is from a HistGradientBoosting machine-learning model
        (sklearn) trained on ~1,900 real contracts from the modern CBA era
        (2012-13 onward). The model learned from features including trailing-
        weighted Barrett Score, prior salary, age, position, service years,
        recent All-NBA selections, the rank-based projection, and advanced
        metrics (usage rate, PIE, on/off net rating, true shooting) — then
        post-processed with CBA max-contract cap and supermax floor rules.

        **Why only 2012+?** The goal is predicting *current* contracts.
        Pre-2012 deals come from a different financial regime (lower cap, the
        old CBA), and including them measurably *hurts* recent-season accuracy.
        Trimming to the modern era is a deliberate recency choice, validated by
        a training-window search (scripts/experiment_recency_window.py).

        **Validation — expanding-window temporal cross-validation on recent
        seasons (2021-2025).** The honest way to measure a forecasting model:
        train only on prior seasons, predict each subsequent season the model
        has never seen. Graded on **every real new contract** — minimum
        signings and market deals all count. Exclusions are only: rookie-scale
        step-ups (Luka's locked year-4 raise — not a signing) and salary-data
        errors (mid-season buyout/waiver artifacts like a star's prorated
        near-zero figure after a trade, and a handful of verified bad labels),
        which misrepresent the actual contract.

        Graded on every real new contract, in **actual dollars** (not
        share-of-cap):
        - **Minimum / role (under $7M):** median miss ~$1.2M — right within
          $3M about three times in four
        - **Rotation ($7–15M):** median miss ~$3.8M
        - **Mid-tier ($15–25M):** median miss ~$5.5M
        - **Stars ($25M+):** median miss ~$5–9M, and the model systematically
          *under*-projects them by ~$5M

        Headline "within 5% of cap" figures (89% of predictions; 99% within
        10%) are share-of-cap, which flatters cheap deals — a $7M miss is
        "within 5% of cap" whether the player makes $50M or $3M. The honest
        takeaway: trust the dollar figure most for role and rotation players;
        for stars, read it as a ballpark, not a quote.

        We tested four ways to tighten the star end — isotonic de-biasing, a
        leak-free market-comp feature, max-floor retuning, and quantile-loss
        prediction intervals — and none survived honest out-of-sample grading,
        so none shipped. The star-contract residual is genuinely irreducible
        from box-score inputs: those deals are set in negotiations, not on the
        floor. (The market *blend* on the bar is a separate display-layer
        guardrail for visible outliers.)

        Every feature was gated on cross-validation, not a single split. The
        advanced metrics earned their place (+1.1pp within-5% on paired CV,
        t=3.9). A two-stage classify-the-regime-then-snap model and a 4-learner
        stacked ensemble were tested and came in within noise — so they were
        dropped. We ship only what the rigorous evaluation confirms.

        **What the number means.** It's a *new-contract* projection: the model
        prices this season's production and reports it in NEXT season's cap
        dollars — i.e. "what this player would command if he signed a fresh deal
        today." It is a market-value estimate, so it ignores one-off circumstances
        of a player's *current* deal (a buyout, a hometown discount); a bought-out
        player is valued on his game, not his bargain.

        The remaining misses are almost all young breakouts landing their
        first max extension off a tiny prior salary (Porter, Simons, Suggs) —
        the genuinely hard call no production model nails ahead of time.
        """
    )
