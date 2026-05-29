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
season the model has never seen. Graded on MARKET contracts only: we exclude
CBA-minimum signings, buyouts, and rookie-scale locks, which are fixed/
situational, not negotiated valuations the model is meant to predict.
  - 81% of predictions within 5% of the cap (~$8M)
  - 97% within 10% of cap (catastrophic misses under 3% of predictions)
  - Median |error|: ~2% of cap
(Including the easy minimum signings would read ~87% within 5%, but that
pads the stat — the 81% on real negotiated deals is the honest number.)

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
    # Calibration constants — single source of truth in utils
    SALARY_CAP_M, cap_dollars,
    CONTRACT_POSITION_MULTIPLIERS as POSITION_MULTIPLIERS,
    CONFIDENCE_BAND_PCT_OF_CAP,
    HEALTHY_SEASON_GP, NEW_CONTRACT_PCT, SUPERMAX_CAP_PCT,
    tiered_age_multiplier, durability_multiplier, playoff_bonus_multiplier,
    # Draft tier — used to keep comparables apples-to-apples (lottery picks
    # earn on pedigree; non-lottery developers don't).
    DRAFT_TIERS, DRAFT_TIER_ORDINAL,
    get_player_draft_info, build_draft_tier_lookup,
    # CBA / contract structure
    get_max_contract_eligibility,
    fetch_rookie_scale_players,
    fetch_all_nba_selections, get_all_nba_in_window,
    # Contract end-year scraper — powers the "Current deal: $X through YYYY-YY"
    # context line under the hero (and nothing else after the forward-
    # projection revert).
    get_player_contract_info,
)


CURRENT_SEASON = SEASONS[0]


# ── Page boilerplate ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Contract Predictor", layout="wide")
render_page_chrome()
_bootstrap_warm()
render_nav("Contract Predictor")

st.title("Contract Predictor")
st.caption(
    "Type a player's name to see their projected next contract. A machine-"
    "learning model (HistGBM) trained on 1,900+ modern-era contracts (2012+), "
    "built on the Barrett Score plus age, position, service years, All-NBA "
    "history, and advanced metrics (usage, PIE, on/off rating). Validated by "
    "temporal cross-validation on real market contracts: 81% of predictions "
    "within 5% of the cap, 97% within 10%."
)

# Methodology expanders live at the bottom of the page (after the prediction
# and comparables) so the page leads with the answer, not the methodology.


# ── Helpers ──────────────────────────────────────────────────────────────────
# _age_bucket is imported from utils as `age_bucket` (aliased). Single source
# of truth so the bucket boundaries stay in sync with the analyzer scripts.


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
# Temporal CV on recent seasons (2021-2025), graded on market contracts only
# (excl. minimums/buyouts/rookie-locks): 81% within 5% of cap, 97% within 10%.
# A two-stage model and a stacked ensemble were tested and came in within
# noise under CV — the simple regressor wins, so we ship it.
# See scripts/build_production_histgbm.py and scripts/confirm_advanced_features.py.
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
    """Build the 22-feature input vector that the HistGBM expects, in the
    exact order the model was trained on (PRUNED_FEATURES + 8 derived).
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


def predict_contract_histgbm(features: dict, target_season: str = CURRENT_SEASON
                              ) -> dict | None:
    """HistGBM v2 prediction with CBA cap/floor post-processing.
    Returns the same dict format as predict_contract. Returns None if the
    HistGBM model isn't loadable — caller should fall back to predict_contract.
    """
    artifact = _load_histgbm()
    if artifact is None:
        return None
    X = _histgbm_feature_vector(features, target_season)
    if X is None:
        return None
    model = artifact["model"]
    pred_pct = float(np.clip(model.predict(X)[0], 0.001, 0.45))

    cap_dollars_val = SALARY_CAP_M.get(target_season, 154.6) * 1_000_000
    raw_predicted = pred_pct * cap_dollars_val
    base = raw_predicted  # for display compatibility with predict_contract

    # ── CBA cap-and-floor (structural rules the model can't see exactly) ────
    max_pct = float(features.get("max_pct", 0.35) or 0.35)
    cba_max_dollars = cap_dollars_val * max_pct
    supermax_eligible = bool(features.get("supermax_eligible", False))
    supermax_tier_label = features.get("supermax_tier", "")

    predicted = raw_predicted
    cba_cap_applied = False
    cba_floor_applied = False

    if predicted > cba_max_dollars:
        predicted = cba_max_dollars
        cba_cap_applied = True

    target_age = features.get("age")
    in_prime = target_age is not None and target_age <= 32
    if supermax_eligible and in_prime and predicted < cba_max_dollars:
        predicted = cba_max_dollars
        cba_floor_applied = True

    band = cap_dollars_val * CONFIDENCE_BAND_PCT_OF_CAP

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
        "low":                  max(0, predicted - band),
        "high":                 predicted + band,
        "band":                 band,
        "cap":                  cap_dollars_val,
        "max_pct":              max_pct,
        "cba_max_dollars":      cba_max_dollars,
        "cba_cap_applied":      cba_cap_applied,
        "cba_floor_applied":    cba_floor_applied,
        "supermax_eligible":    supermax_eligible,
        "supermax_tier_label":  supermax_tier_label,
        "model_used":           "HistGBM v2",
    }


def predict_contract(features: dict, target_season: str = CURRENT_SEASON) -> dict:
    # HistGBM model (machine-learning regression on ~1,900 modern-era
    # contracts, 2012+). Falls back to the legacy rank-mapping +
    # multipliers formula if the model artifact isn't available.
    hist = predict_contract_histgbm(features, target_season)
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

    cap_dollars_val = SALARY_CAP_M.get(target_season, 154.6) * 1_000_000
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

    # ── CBA cap-and-floor adjustments (today's eligibility) ─────────────────
    max_pct = float(features.get("max_pct", 0.35) or 0.35)
    cba_max_dollars = cap_dollars_val * max_pct
    supermax_eligible = bool(features.get("supermax_eligible", False))
    supermax_tier_label = features.get("supermax_tier", "")

    predicted = raw_predicted
    cba_cap_applied = False
    cba_floor_applied = False

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

    band = cap_dollars_val * CONFIDENCE_BAND_PCT_OF_CAP

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
        "low":                  max(0, predicted - band),
        "high":                 predicted + band,
        "band":                 band,
        "cap":                  cap_dollars_val,
        "max_pct":              max_pct,
        "cba_max_dollars":      cba_max_dollars,
        "cba_cap_applied":      cba_cap_applied,
        "cba_floor_applied":    cba_floor_applied,
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
        # Compute the dollar amount for the supermax tier.
        cap_M = SALARY_CAP_M.get(CURRENT_SEASON, 154.6)
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

    # Filter out vet-min / training-camp signings (under 3% of cap, ~$4.6M
    # at the 2025-26 level). These aren't market-rate deals — they're
    # CBA-mandated minimum-salary signings + bench-filler contracts. They
    # showed up as noisy outliers in the comp pool (e.g. Keita Bates-Diop
    # at $2.4M for a young role-player target like Peyton Watson).
    # Players in vet-min cohorts are still predicted correctly by the
    # model's career-rate base + "veteran end-of-career" caveat — we
    # just don't pollute mid-tier comp pools with their signings.
    def _is_vet_min(row) -> bool:
        cap_curr = SALARY_CAP_M.get(row["signed_in"], 154.6) * 1_000_000
        if cap_curr <= 0:
            return False
        return float(row["salary_curr"]) < cap_curr * 0.03

    out = out[~out.apply(_is_vet_min, axis=1)].reset_index(drop=True)

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


def find_comparables(features: dict, history: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """Match historical signings on **trailing-weighted Barrett** + age + position.

    The target's trailing_barrett is a 50/30/20-weighted average of the
    player's last 3 healthy sign-year Barrett Scores (with availability) —
    same weighting as career_barrett, but raw Barrett instead of rate.

    Score-distance weight is age-scaled (1.0 + tier_weight):
      - Young developers (≤27): 2.0× score — prevents Zion-style stretch
        matches where a far-off score sneaks in via perfect age/position/
        tier alignment. Forces the pool to favor score-close comps.
      - Veterans (≥31): 1.0× score — keeps age relatively important so
        aging vets match other aging vets (Harden's paycut cohort), not
        young production-similar stars (Brunson/Booker).

    Tried min(sign-yr, career-weighted) for the score term but it brought
    back paycut comps via their pre-collapse career scores (Lonzo career
    19.5 matched Vassell trailing 17.2 even though Lonzo signed for $10M
    after collapsing to sign-yr 9). Sign-yr-only is the right discriminator
    because that's the snapshot the market actually priced.

    Distance = |comp_walk_yr − target_trailing| × (1 + tier_weight)
             + |age_diff| × 1.5
             + position_penalty (broad G/F/C bucket — PG/SG are pooled)
             + tier_penalty (faded by age — see _tier_penalty_weight)
    """
    if history.empty:
        return history

    target_position = features["position"]
    target_age = features["age"] if features["age"] else 27
    # Trailing-weighted Barrett with availability — smoothed sign-yr.
    target_barrett = features.get("trailing_barrett", features["barrett_score"])
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

    return same_pos.nsmallest(n, "distance").copy()


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


def _scouting_take(features: dict, comps: pd.DataFrame) -> dict:
    """Build the 'Scouting take' summary: top-3 names, weighted median deal,
    weighted IQR range, X-factor narrative.

    Uses inverse-distance weights so the closest comparables count more
    than the farthest — a tighter, more market-grounded second opinion."""
    if comps.empty:
        return {}

    salaries = comps["salary_curr"].astype(float).values
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


def _search_players(query: str) -> list[str]:
    """Filter active-season players by case- and accent-insensitive substring.

    Empty query → return the full active-season roster (already sorted by
    Barrett score descending — Jokic / SGA / Giannis at the top).

    With a query → match by quality (exact → prefix → substring), then
    Barrett score descending within each match tier. Capped at 10.
    """
    if not query or not query.strip():
        # Already pre-sorted by Barrett desc.
        return active_names
    q = normalize(query)
    scored: list[tuple[int, float, str]] = []
    for n in active_names:
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
selected = st_searchbox(
    search_function=_search_players,
    placeholder="Type a player name…",
    default=_init_player,
    default_options=active_names,  # already sorted by Barrett desc
    edit_after_submit="option",
    rerun_on_update=True,
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
features = get_player_features(selected, CURRENT_SEASON)
if features is None:
    st.warning(f"Couldn't find {selected} in {CURRENT_SEASON} data.")
    st.stop()

prediction = predict_contract(features, CURRENT_SEASON)
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
    _salaries = _market_used_comps["salary_curr"].astype(float).values
    if "distance" in _market_used_comps.columns:
        _weights = _inverse_distance_weights(
            _market_used_comps["distance"].astype(float).values
        )
    else:
        _weights = np.ones_like(_salaries)
    _market_median = _weighted_median(_salaries, _weights)
else:
    _market_median = None

# ── Big number header: Model + Market side-by-side ───────────────────────────
predicted_M = prediction["predicted"] / 1_000_000
low_M  = prediction["low"]  / 1_000_000
high_M = prediction["high"] / 1_000_000

# Initialize divergence so downstream code (confidence label, "Why this
# prediction" explainer) can reference it whether or not market data exists.
divergence = 0.0

# Compact CBA-status caption shown under the Model dollar (both hero
# variants — with market and without). Gives a one-line "why" at a
# glance; the full explanation lives in the About expander.
_supermax_label = prediction.get("supermax_tier_label", "")
if prediction.get("cba_cap_applied"):
    _model_caption = f"Capped at max ({_supermax_label})"
elif prediction.get("cba_floor_applied"):
    _model_caption = f"Supermax floor ({_supermax_label})"
elif features.get("on_rookie_scale"):
    _model_caption = "Currently on rookie scale"
else:
    _model_caption = ""

# Pre-build the caption HTML as a single string. Two design notes:
#   1. Inline conditional in the f-string template caused Streamlit's
#      markdown parser to flip into code-block mode whenever the caption
#      was empty (blank-line bug). Building here keeps the interpolation
#      site a single string.
#   2. position:absolute so the caption doesn't affect the Model column's
#      height. Without this, the caption pushed Market and Range dollars
#      up to align with Model's caption bottom (flex-end alignment) —
#      making the three dollar amounts sit on different vertical levels.
_model_caption_html = (
    f'<div style="position:absolute; top:100%; left:0; '
    f'white-space:nowrap; font-size:0.7rem; color:#16d4c1; '
    f'margin-top:0.25rem; font-weight:600;">{_model_caption}</div>'
    if _model_caption else ''
)

# Score chip rendered next to the player name in the hero title. Pre-built
# here so the f-string template doesn't have any conditional logic that
# could trigger the markdown blank-line bug.
_score_chip_html = (
    f'<span style="display:inline-block; margin-left:0.7rem; '
    f'padding:0.18rem 0.6rem; border-radius:999px; '
    f'background:rgba(22,212,193,0.10); '
    f'border:1px solid rgba(22,212,193,0.30); '
    f'font-size:0.72rem; font-weight:700; color:#16d4c1; '
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
        f'<div style="margin-top:0.35rem; font-size:0.74rem; color:#888;">'
        f'Current deal: {_sal_str}through {_contract_end}{_type_blurb}'
        f'</div>'
    )
elif _cur_sal_dollars > 0:
    # Case 2: free agent this offseason (or final year of expiring deal).
    # Show their current/previous salary so users have a baseline.
    _signing_html = (
        f'<div style="margin-top:0.35rem; font-size:0.74rem; color:#888;">'
        f'Previous: {_fmt_money(_cur_sal_dollars)} ({CURRENT_SEASON}) · '
        f'free agent next summer'
        f'</div>'
    )

if _market_median is not None:
    market_M = _market_median / 1_000_000
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
        f'border-top:1px solid rgba(255,255,255,0.08); '
        f'font-size:0.78rem; color:#f39c12;">'
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
    <div style="background:linear-gradient(135deg, rgba(230,57,70,0.10) 0%, rgba(22,212,193,0.08) 100%);
                border:1px solid rgba(255,255,255,0.12); border-radius:14px;
                padding:1.4rem 1.8rem 1.8rem 1.8rem; margin: 0.5rem 0 1.2rem 0;">
      <div style="font-size:1.5rem; color:#fff; font-weight:800; line-height:1.2;">
        {features["name"]}{_score_chip_html}
      </div>
      <div style="font-size:0.72rem; color:#888; text-transform:uppercase;
                  letter-spacing:0.1em; font-weight:600; margin-top:0.4rem;">
        Predicted next contract
      </div>
      <div style="font-size:0.78rem; color:#aaa; margin-top:0.15rem;">
        {_player_meta_line}
      </div>{_signing_html}

      <div style="display:flex; gap:1.6rem; margin-top:0.85rem;
                  flex-wrap:wrap; align-items:flex-end;">
        <div style="position:relative;">
          <div style="font-size:0.65rem; color:#888;
                      text-transform:uppercase; letter-spacing:0.08em;">
            Model
          </div>
          <div style="font-size:2.2rem; font-weight:800; color:#fff;
                      line-height:1;">${predicted_M:.1f}M</div>{_model_caption_html}
        </div>
        <div style="font-size:1.4rem; color:#444; padding-bottom:0.4rem;">|</div>
        <div>
          <div style="font-size:0.65rem; color:#888;
                      text-transform:uppercase; letter-spacing:0.08em;">
            Market (comps median)
          </div>
          <div style="font-size:2.2rem; font-weight:800; color:#16d4c1;
                      line-height:1;">
            ${market_M:.1f}M
          </div>
        </div>
        <div style="margin-left:auto; text-align:right;">
          <div style="font-size:0.65rem; color:#888;
                      text-transform:uppercase; letter-spacing:0.08em;">
            Range
          </div>
          <div style="font-size:1.3rem; color:#fff; font-weight:700;
                      line-height:1.1;">
            ${honest_low_M:.1f}M – ${honest_high_M:.1f}M
          </div>
        </div>
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
        # Range expands to cover both the model prediction and current
        # salary — the player will likely land somewhere in this band.
        anchor_low_M  = min(predicted_M, cur_sal_M)
        anchor_high_M = max(predicted_M, cur_sal_M)
        _explainer = (
            'Market view unavailable — the queried player is currently '
            'supermax-tier, but the historical comparables pool for that '
            'cohort is too sparse (5+ Paycut signings, no "stayed at tier" '
            'comps). Anchoring the range on current salary instead.'
        )
        _header_html = f"""
        <div style="background:linear-gradient(135deg, rgba(230,57,70,0.10) 0%, rgba(22,212,193,0.08) 100%);
                    border:1px solid rgba(255,255,255,0.12); border-radius:14px;
                    padding:1.4rem 1.8rem; margin: 0.5rem 0 1.2rem 0;">
          <div style="font-size:1.5rem; color:#fff; font-weight:800; line-height:1.2;">
            {features["name"]}{_score_chip_html}
          </div>
          <div style="font-size:0.72rem; color:#888; text-transform:uppercase;
                      letter-spacing:0.1em; font-weight:600; margin-top:0.4rem;">
            Predicted next contract
          </div>
          <div style="font-size:0.78rem; color:#aaa; margin-top:0.15rem;">
            {_player_meta_line}
          </div>{_signing_html}
          <div style="display:flex; gap:1.6rem; margin-top:0.85rem;
                      flex-wrap:wrap; align-items:flex-end;">
            <div style="position:relative;">
              <div style="font-size:0.65rem; color:#888;
                          text-transform:uppercase; letter-spacing:0.08em;">
                Model
              </div>
              <div style="font-size:2.2rem; font-weight:800; color:#fff;
                          line-height:1;">${predicted_M:.1f}M</div>{_model_caption_html}
            </div>
            <div style="font-size:1.4rem; color:#444; padding-bottom:0.4rem;">|</div>
            <div>
              <div style="font-size:0.65rem; color:#888;
                          text-transform:uppercase; letter-spacing:0.08em;">
                Current salary (anchor)
              </div>
              <div style="font-size:2.2rem; font-weight:800; color:#16d4c1;
                          line-height:1;">${cur_sal_M:.1f}M</div>
            </div>
            <div style="margin-left:auto; text-align:right;">
              <div style="font-size:0.65rem; color:#888;
                          text-transform:uppercase; letter-spacing:0.08em;">
                Range
              </div>
              <div style="font-size:1.3rem; color:#fff; font-weight:700;
                          line-height:1.1;">
                ${anchor_low_M:.1f}M – ${anchor_high_M:.1f}M
              </div>
            </div>
          </div>
          <div style="margin-top:0.7rem; padding-top:0.6rem;
                      border-top:1px solid rgba(255,255,255,0.08);
                      font-size:0.78rem; color:#f39c12;">
            ⚠ {_explainer}
          </div>
        </div>
        """
    else:
        _header_html = f"""
        <div style="background:linear-gradient(135deg, rgba(230,57,70,0.10) 0%, rgba(22,212,193,0.08) 100%);
                    border:1px solid rgba(255,255,255,0.12); border-radius:14px;
                    padding:1.4rem 1.8rem; margin: 0.5rem 0 1.2rem 0;">
          <div style="font-size:1.5rem; color:#fff; font-weight:800; line-height:1.2;">
            {features["name"]}{_score_chip_html}
          </div>
          <div style="font-size:0.72rem; color:#888; text-transform:uppercase;
                      letter-spacing:0.1em; font-weight:600; margin-top:0.4rem;">
            Predicted next contract
          </div>
          <div style="font-size:0.78rem; color:#aaa; margin-top:0.15rem;">
            {_player_meta_line}
          </div>{_signing_html}
          <div style="display:flex; align-items:baseline; gap:1rem;
                      margin-top:0.7rem; flex-wrap:wrap;">
            <div style="font-size:2.8rem; font-weight:800; color:#fff; line-height:1;">
              ${predicted_M:.1f}M
            </div>
            <div style="color:#aaa; font-size:0.9rem;">
              ±${prediction['band']/1_000_000:.1f}M band ·
              range <b style="color:#cdcdd5;">${low_M:.1f}M</b> –
              <b style="color:#cdcdd5;">${high_M:.1f}M</b>
            </div>
          </div>
          <div style="margin-top:0.35rem; font-size:0.75rem; color:#888;">
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
    def _caveat_chip(note: str) -> str:
        if note.startswith("Supermax-eligible:"):
            # Split short label and full detail for hover tooltip.
            detail = note[len("Supermax-eligible: "):].strip()
            # Escape the quote in the detail so it survives in the
            # HTML title attribute (no inner double quotes).
            safe_detail = detail.replace('"', '&quot;')
            return (
                f'<div title="{safe_detail}" '
                f'style="display:inline-block; cursor:help; '
                f'background:rgba(243,156,18,0.10); '
                f'border:1px solid rgba(243,156,18,0.30); border-radius:6px; '
                f'padding:0.3rem 0.7rem; margin: 0 0.4rem 0.4rem 0; '
                f'font-size:0.8rem; color:#f1c40f;">⚠ Supermax-eligible</div>'
            )
        return (
            f'<div style="display:inline-block; '
            f'background:rgba(243,156,18,0.10); '
            f'border:1px solid rgba(243,156,18,0.30); border-radius:6px; '
            f'padding:0.3rem 0.7rem; margin: 0 0.4rem 0.4rem 0; '
            f'font-size:0.8rem; color:#f1c40f;">⚠ {note}</div>'
        )

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
            <div style="background:rgba(22,212,193,0.06);
                        border:1px solid rgba(22,212,193,0.25);
                        border-radius:10px; padding:1rem 1.3rem; margin:0.4rem 0 1rem 0;">
              <div style="font-size:0.74rem; color:#16d4c1; letter-spacing:0.08em;
                          text-transform:uppercase; font-weight:700; margin-bottom:0.5rem;">
                📋 Scouting take
              </div>
              <div style="display:flex; flex-wrap:wrap; gap:1.5rem; margin-bottom:0.6rem;">
                <div>
                  <div style="font-size:0.72rem; color:#888;
                              text-transform:uppercase; letter-spacing:0.05em;">
                    Median new contract
                  </div>
                  <div style="font-size:1.6rem; color:#fff; font-weight:700;">
                    ${take['median']/1e6:.1f}M
                  </div>
                </div>
                <div>
                  <div style="font-size:0.72rem; color:#888;
                              text-transform:uppercase; letter-spacing:0.05em;">
                    Middle 50%
                  </div>
                  <div style="font-size:1.6rem; color:#cdcdd5; font-weight:600;">
                    ${take['q25']/1e6:.1f}M – ${take['q75']/1e6:.1f}M
                  </div>
                </div>
                <div style="flex:1; min-width:260px;">
                  <div style="font-size:0.72rem; color:#888;
                              text-transform:uppercase; letter-spacing:0.05em;">
                    Closest 3 comps
                  </div>
                  <div style="font-size:0.95rem; color:#cdcdd5;">{top3_str}</div>
                </div>
              </div>
              <div style="font-size:0.85rem; color:#aaa;
                          border-top:1px solid rgba(255,255,255,0.08);
                          padding-top:0.5rem;">
                <b style="color:#cdcdd5;">Note:</b> {take['x_factor']}
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
        st.dataframe(comp_disp, use_container_width=True, hide_index=True,
                     height=min(400, 60 + len(comp_disp) * 35))

        st.caption(
            "Context tags: **Supermax** ≥28% cap · **Free-agent raise** ≥15% bump · "
            "**Rookie extension** first non-rookie deal · **Paycut** took less to stay. "
            "Draft tiers: Lottery (1-14) · Mid-1st (15-22) · Late-1st (23-30) · 2nd (31-60) · Undrafted."
        )

# ── Methodology footer (collapsed — info-after-action) ──────────────────────
st.divider()
with st.expander("About this prediction"):
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
        f'&nbsp;<span style="color:#666;">×</span>&nbsp;'
        f'<b>×{_dur_mult:.2f}</b>'
        f'<span style="color:#777;"> (durability: {_dur_tier} · '
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
        f'&nbsp;<span style="color:#666;">×</span>&nbsp;'
        f'<b>×{_playoff_mult:.2f}</b>'
        f'<span style="color:#777;"> (playoff: {_playoff_tier} · '
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
    _supermax_tier_label = prediction.get("supermax_tier_label", "")
    if _cba_cap_applied:
        _cba_fragment = (
            f' &nbsp;<span style="color:#666;">→</span>&nbsp; '
            f'<b style="color:#e63946;">capped at ${_cba_max_M:.1f}M</b>'
            f' <span style="color:#777;">(CBA max: {_supermax_tier_label})</span>'
        )
    elif _cba_floor_applied:
        _cba_fragment = (
            f' &nbsp;<span style="color:#666;">→</span>&nbsp; '
            f'<b style="color:#16d4c1;">floored at ${_cba_max_M:.1f}M</b>'
            f' <span style="color:#777;">(supermax: {_supermax_tier_label})</span>'
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
            '<span style="color:#888; font-size:0.7rem; letter-spacing:0.08em;'
            ' text-transform:uppercase; margin-right:0.5rem;">Model</span>'
            f'<b style="color:#fff;">${_raw_predicted_M:.1f}M</b>'
            f' <span style="color:#777;">(HistGBM ML output from {_ml_inputs})</span>'
            f' &nbsp;<span style="color:#666;">→</span>&nbsp; '
            f'<b style="color:#16d4c1;">${_equation_end_M:.1f}M</b>'
            f' <span style="color:#666;">±${prediction["band"]/1_000_000:.1f}M</span>'
            f'{_cba_fragment}'
        )
    else:
        _math_line = (
            '<span style="color:#888; font-size:0.7rem; letter-spacing:0.08em;'
            ' text-transform:uppercase; margin-right:0.5rem;">Math</span>'
            f'<b style="color:#fff;">${base_M:.1f}M</b>'
            f' <span style="color:#777;">(career rate score '
            f'{features["career_barrett"]:.1f} → rank #{features["effective_rank"]})'
            f'</span> &nbsp;<span style="color:#666;">×</span>&nbsp; '
            f'<b>×{prediction["age_mult"]:.2f}</b>'
            f' <span style="color:#777;">(age {_age_label}{_age_factor_note})</span>'
            f' &nbsp;<span style="color:#666;">×</span>&nbsp; '
            f'<b>×{prediction["pos_mult"]:.2f}</b>'
            f' <span style="color:#777;">({_pos_label}{_pos_factor_note})</span>'
            f'{_dur_html_fragment}'
            f'{_playoff_html_fragment}'
            f' &nbsp;<span style="color:#666;">=</span>&nbsp; '
            f'<b style="color:#16d4c1;">${_equation_end_M:.1f}M</b>'
            f' <span style="color:#666;">±${prediction["band"]/1_000_000:.1f}M</span>'
            f'{_cba_fragment}'
        )
    _breakdown_html = (
        '<div style="background:rgba(255,255,255,0.03);'
        ' border:1px solid rgba(255,255,255,0.08);'
        ' border-radius:10px; padding:0.85rem 1.1rem; margin: 0 0 1rem 0;'
        ' font-size:0.95rem; color:#cdcdd5; line-height:1.55;">'
        f'{_math_line}'
        '<div style="font-size:0.72rem; color:#777; margin-top:0.35rem;">'
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

        **Confidence band:** ±$5.5M reflects out-of-sample median error.

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
        has never seen. Graded on **market contracts only** — we exclude
        CBA-minimum signings, buyouts, and rookie-scale locks (Luka's locked
        year-4 step-up, etc.), which are fixed or situational, not negotiated
        valuations the model is meant to predict.

        - **81% of predictions within 5% of the cap** (~$8M)
        - **97% within 10% of cap** — catastrophic misses under 3%
        - Median |error|: ~2% of cap, ~$3M in 2025-26 dollars

        (Counting the easy minimum signings would read ~87% within 5%, but
        that pads the number — 81% on real negotiated deals is the honest one.)

        Every feature was gated on cross-validation, not a single split. The
        advanced metrics earned their place (+1.1pp within-5% on paired CV,
        t=3.9). A two-stage classify-the-regime-then-snap model and a 4-learner
        stacked ensemble were tested and came in within noise — so they were
        dropped. We ship only what the rigorous evaluation confirms.

        The remaining misses are almost all young breakouts landing their
        first max extension off a tiny prior salary (Porter, Simons, Suggs) —
        the genuinely hard call no production model nails ahead of time.
        """
    )
