"""Contract Predictor — predict a player's next contract.

Takes a player's current production (Barrett Score), applies age + position
calibration multipliers learned from 2014-22 historical contracts, and returns
a dollar projection with a confidence band and a list of comparable signings.

Out-of-sample accuracy: ~79% within 5% of cap on 1,406 real new contracts
since 2015 (modern era). Median error 1.8% of cap (~$2.7M in 2025-26 dollars).

The full validation suite — across 4,500 signings going back to 1985 — comes
in at 74% within 5% of cap; the modern-era number is the more defensible
claim because it matches the cap-era the model was tuned for.
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

from utils import (
    COMMON_CSS, SEASONS, normalize, season_to_espn_year,
    get_all_player_names, fetch_player_full_career,
    build_ranked_projected, fetch_league_stats,
    fetch_bref_positions, fetch_player_positions_detailed, position_to_bucket,
    render_nav, render_page_chrome, render_barrett_score_explainer, _bootstrap_warm,
    # Calibration constants — single source of truth in utils
    SALARY_CAP_M, cap_dollars,
    CONTRACT_POSITION_MULTIPLIERS as POSITION_MULTIPLIERS,
    CONFIDENCE_BAND_PCT_OF_CAP,
    HEALTHY_SEASON_GP, NEW_CONTRACT_PCT, SUPERMAX_CAP_PCT,
    tiered_age_multiplier, durability_multiplier,
    # Draft tier — used to keep comparables apples-to-apples (lottery picks
    # earn on pedigree; non-lottery developers don't).
    DRAFT_TIERS, DRAFT_TIER_ORDINAL,
    get_player_draft_info, build_draft_tier_lookup,
    # CBA / contract structure
    get_max_contract_eligibility,
    fetch_rookie_scale_players,
    fetch_all_nba_selections,
)


CURRENT_SEASON = SEASONS[0]


# ── Page boilerplate ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Contract Predictor", layout="wide")
render_page_chrome()
_bootstrap_warm()
render_nav("Contract Predictor")

st.title("Contract Predictor")
st.caption(
    "Type a player's name to see their projected next contract. Based on the "
    "Barrett Score, adjusted for age and position. Out-of-sample accuracy: "
    "79% within 5% of cap on 1,406 real new contracts since 2015 (modern era)."
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


def _fmt_draft(features: dict) -> str | None:
    """Short draft label for the metadata line, e.g. 'Lottery (#7, 2021)' or
    'Undrafted'. Returns None when we have no useful info to display."""
    tier = features.get("draft_tier")
    pick = features.get("draft_pick")
    year = features.get("draft_year")
    if not tier:
        return None
    if tier == "Undrafted" and not pick:
        # Drop entirely if there's nothing useful — avoids clutter for
        # players where the draft API just didn't return a record.
        return None
    if pick:
        suffix = f" (#{pick}{', ' + str(year) if year else ''})"
        return f"{tier}{suffix}"
    return tier


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

    # Try the detailed BBRef per-game scrape first (PG/SG/SF/PF/C, better
    # coverage). Fall back to the older ESPN-salary scrape (G/F/C only),
    # then to Unknown.
    detailed_pos = "Unknown"
    try:
        detailed_lookup = fetch_player_positions_detailed(season, cache_v=2)
        detailed_pos = detailed_lookup.get(name_norm, "Unknown")
    except Exception:
        detailed_lookup = {}
    if detailed_pos == "Unknown":
        try:
            pos_lookup = fetch_bref_positions(season_to_espn_year(season), cache_v=3)
        except Exception:
            pos_lookup = {}
        coarse_fallback = pos_lookup.get(name_norm, "Unknown")
        # Old function returns "Guard"/"Forward"/"Center" already.
        pos_bucket = coarse_fallback
        detailed_display = coarse_fallback
    else:
        pos_bucket = position_to_bucket(detailed_pos)
        detailed_display = detailed_pos
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
        "draft_tier":         draft_info["draft_tier"],
        "draft_pick":         draft_info["draft_pick"],
        "draft_year":         draft_info["draft_year"],
        # CBA / contract structure
        "service_years":      elig["service_years"],
        "team_tenure":        elig["team_tenure"],
        "current_team":       elig["current_team"],
        "recent_all_nba":     elig["recent_all_nba"],
        "supermax_eligible":  elig["qualifying"] and elig["supermax_tier"] in
                              ("Designated Vet (35%)", "Designated Rookie (30%)"),
        "max_pct":            elig["max_pct"],
        "supermax_tier":      elig["supermax_tier"],
        "on_rookie_scale":    on_rookie_scale,
        "salary":             float(row.get("salary", 0) or 0),
        "projected_salary":   float(row.get("projected_salary", 0) or 0),
        "gp":                 int(row.get("GP", 0) or 0),
        "mpg":                float(row.get("MPG", 0) or 0),
        "total_pool_size":    len(ranked),
    }


def predict_contract(features: dict, target_season: str = CURRENT_SEASON) -> dict:
    # Use the career-weighted projection as the base — much more stable
    # for players in the middle of an injury/comeback season than the raw
    # current-season Barrett rank.
    base = features["career_base_proj"]

    # Tiered age multiplier (replaces the old bucket-based version).
    # Function of (age, career_score, current_rank) — captures the real
    # NBA pattern that elite producers don't follow the average aging
    # curve. See utils.tiered_age_multiplier for the three tiers and
    # their decline rates.
    age_mult, age_tier = tiered_age_multiplier(
        age=features.get("age"),
        career_score=features.get("career_barrett", 0),
        current_rank=features.get("effective_rank"),
    )

    pos_mult = POSITION_MULTIPLIERS.get(features["position"], 1.0)

    cap_M = SALARY_CAP_M.get(target_season, 154.6)
    cap_dollars_val = cap_M * 1_000_000
    supermax_threshold = cap_dollars_val * SUPERMAX_CAP_PCT

    # Position suppression at supermax tier:
    # Position multipliers were fit on mid-market signings (Centers get
    # systematically less than box score suggests). But supermax/max-
    # contract players sign at fixed CBA percentages regardless of
    # position. When base ≥ 28% of cap, drop the positional discount.
    pos_mult_applied = pos_mult
    if base >= supermax_threshold:
        pos_mult_applied = 1.0

    # Durability discount — comes from get_player_features which already
    # computed it from the full career. Healthy players (avail ≥ 85%
    # over 3 years): ×1.00, no penalty. Chronic injury cases (Embiid):
    # heavy discount even when rate score is elite.
    dur_mult = float(features.get("durability_mult", 1.0) or 1.0)
    dur_tier = features.get("durability_tier", "")

    raw_predicted = base * age_mult * pos_mult_applied * dur_mult

    # ── CBA cap-and-floor adjustments ───────────────────────────────────────
    # 1. Cap at player's CBA max %. A 25% max player (≤6 yrs service)
    #    cannot legally sign for more than 25% of cap, no matter what the
    #    model says. Hard ceiling — this is CBA-binding.
    # 2. Floor at supermax threshold IF player is Designated Vet / Designated
    #    Rookie eligible. Elite players who qualify almost universally take
    #    the max they're offered.
    # 3. Rookie-scale lock: if currently on rookie scale, they CAN'T sign
    #    a new market deal until the rookie deal expires. The model is
    #    still projecting their NEXT contract, which is a real signal, so
    #    we don't override the projection — but we surface this as a caveat.
    max_pct = float(features.get("max_pct", 0.35) or 0.35)
    cba_max_dollars = cap_dollars_val * max_pct
    supermax_eligible = bool(features.get("supermax_eligible", False))
    supermax_tier_label = features.get("supermax_tier", "")

    predicted = raw_predicted
    cba_cap_applied = False
    cba_floor_applied = False

    # Apply max cap.
    if predicted > cba_max_dollars:
        predicted = cba_max_dollars
        cba_cap_applied = True

    # Supermax floor — only for designated-eligible players IN THEIR PRIME.
    # Floor at their eligible max (35% Vet or 30% Rookie). Without this, an
    # All-NBA player whose Barrett rate doesn't quite peg the model at 35%
    # would be under-projected (Jokic / SGA / Wemby pattern).
    #
    # Age cutoff (≤32): older "Designated Vet" players (Curry at 38, LeBron
    # at 40) technically qualify but routinely take paycuts. The floor would
    # over-project their next contract by 50%+. Let the model's age multiplier
    # guide the projection for aging vets — supermax becomes a ceiling for
    # them, not a floor.
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

    # Rookie scale lock — now driven by the actual rookie-scale set, not
    # a salary heuristic. CBA-binding: they cannot sign a new market deal
    # until the rookie scale expires.
    if features.get("on_rookie_scale"):
        notes.append(
            "Currently on rookie scale (CBA-locked salary). The projection "
            "below is for their NEXT contract — i.e. their rookie-scale "
            "extension or first market deal."
        )
    elif salary > 0 and salary < 12_000_000 and age and age <= 23:
        # Fallback heuristic for players not in our rookie-scale set.
        notes.append(
            "Possibly on rookie scale — locked salary by CBA until contract "
            "expires (usually year 4). The market price below forecasts their "
            "next contract, not their current one."
        )

    # Supermax eligibility — surface the specific tier so the user sees
    # WHY the model floored their projection.
    if features.get("supermax_eligible"):
        recent = features.get("recent_all_nba", []) or []
        tier_label = features.get("supermax_tier", "")
        notes.append(
            f"Supermax-eligible: {tier_label}. "
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

    if age and age >= 33 and barrett < 20:
        notes.append(
            "Veteran end-of-career zone — may sign for the minimum "
            "(~$2-3M) regardless of production if rosters are full."
        )
    return notes


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
            detailed_lookup = fetch_player_positions_detailed(prev, cache_v=2)
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
                return d
            return coarse_lookup.get(normalize(n), "Unknown")

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
                f"younger than median comp ({int(target_age)} vs {int(comp_age_med)}) "
                "— upside lean"
            )
        else:
            x_factor_parts.append(
                f"older than median comp ({int(target_age)} vs {int(comp_age_med)}) "
                "— downward lean"
            )
    if abs(bar_diff) >= 3:
        if bar_diff > 0:
            x_factor_parts.append(
                f"higher career Score than median comp ({target_bar:.1f} vs {comp_bar_med:.1f}) "
                "— premium lean"
            )
        else:
            x_factor_parts.append(
                f"lower career Score than median comp ({target_bar:.1f} vs {comp_bar_med:.1f}) "
                "— discount lean"
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

_PICKER_KEY = "contract_predictor_player"

# Seed widget state from URL on first load only — before the widget
# renders. After that, the widget's own state is the source of truth
# and the on_change callback mirrors it back into the URL.
if _PICKER_KEY not in st.session_state and "player" in st.query_params:
    qp = st.query_params["player"]
    qp_resolved = next(
        (n for n in active_names if normalize(n) == normalize(qp)),
        None,
    )
    if qp_resolved:
        st.session_state[_PICKER_KEY] = qp_resolved


def _on_player_change():
    """Sync the widget's current value back into ?player= when it changes.
    Fires only on actual selection changes (not on every rerun), which
    avoids the re-render loop the old `st.query_params[...] = selected`
    one-liner triggered."""
    sel = st.session_state.get(_PICKER_KEY)
    if sel:
        st.query_params["player"] = sel
    elif "player" in st.query_params:
        del st.query_params["player"]


selected = st.selectbox(
    "Player",
    options=active_names,
    index=None,  # placeholder shows when nothing's selected
    placeholder="Type a name…",
    label_visibility="collapsed",
    key=_PICKER_KEY,
    on_change=_on_player_change,
)

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
    # If we filtered out Paycut comparables (because the player is
    # currently a supermax-tier earner whose true comparables don't
    # include the "former star, took less to keep playing" cohort),
    # surface that to the user so the higher market median isn't
    # mysterious.
    _filter_note = (
        f'<div style="margin-top:0.5rem; font-size:0.74rem; color:#888;">'
        f'Market view excludes Paycut comparables for currently '
        f'supermax-tier players — the "stayed at tier" cohort is the '
        f'right peer group, not "former star took less."'
        f'</div>'
        if _market_filter_applied else ''
    )
    diverge_note = diverge_note + _filter_note
    # Compact player metadata line — replaces the standalone Player Snapshot
    # section below by inlining age / position / current salary / current Barrett.
    _meta_bits = [features["name"]]
    if features.get("team"): _meta_bits.append(features["team"])
    _meta_bits.append(CURRENT_SEASON)
    if features.get("age"): _meta_bits.append(f"Age {int(features['age'])}")
    _meta_bits.append(str(features.get("position_detailed", features["position"])))
    _draft_label = _fmt_draft(features)
    if _draft_label:
        _meta_bits.append(_draft_label)
    _meta_bits.append(f"Barrett {features['barrett_score']:.1f} (#{features['score_rank']})")
    if features.get("salary", 0) > 0:
        _meta_bits.append(f"Currently {_fmt_money(features['salary'])}")
    _player_meta_line = " · ".join(_meta_bits)

    _header_html = f"""
    <div style="background:linear-gradient(135deg, rgba(230,57,70,0.10) 0%, rgba(22,212,193,0.08) 100%);
                border:1px solid rgba(255,255,255,0.12); border-radius:14px;
                padding:1.4rem 1.8rem; margin: 0.5rem 0 1.2rem 0;">
      <div style="font-size:0.72rem; color:#888; text-transform:uppercase;
                  letter-spacing:0.1em; font-weight:600;">
        Predicted next contract
      </div>
      <div style="font-size:0.78rem; color:#aaa; margin-top:0.15rem;">
        {_player_meta_line}
      </div>

      <div style="display:flex; gap:1.6rem; margin-top:0.85rem;
                  flex-wrap:wrap; align-items:flex-end;">
        <div>
          <div style="font-size:0.65rem; color:#888;
                      text-transform:uppercase; letter-spacing:0.08em;">
            Model
          </div>
          <div style="font-size:2.2rem; font-weight:800; color:#fff;
                      line-height:1;">
            ${predicted_M:.1f}M
          </div>
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
    _meta_bits = [features["name"]]
    if features.get("team"): _meta_bits.append(features["team"])
    _meta_bits.append(CURRENT_SEASON)
    if features.get("age"): _meta_bits.append(f"Age {int(features['age'])}")
    _meta_bits.append(str(features.get("position_detailed", features["position"])))
    _draft_label = _fmt_draft(features)
    if _draft_label:
        _meta_bits.append(_draft_label)
    _meta_bits.append(f"Barrett {features['barrett_score']:.1f} (#{features['score_rank']})")
    if features.get("salary", 0) > 0:
        _meta_bits.append(f"Currently {_fmt_money(features['salary'])}")
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
          <div style="font-size:0.72rem; color:#888; text-transform:uppercase;
                      letter-spacing:0.1em; font-weight:600;">
            Predicted next contract
          </div>
          <div style="font-size:0.78rem; color:#aaa; margin-top:0.15rem;">
            {_player_meta_line}
          </div>
          <div style="display:flex; gap:1.6rem; margin-top:0.85rem;
                      flex-wrap:wrap; align-items:flex-end;">
            <div>
              <div style="font-size:0.65rem; color:#888;
                          text-transform:uppercase; letter-spacing:0.08em;">
                Model
              </div>
              <div style="font-size:2.2rem; font-weight:800; color:#fff;
                          line-height:1;">${predicted_M:.1f}M</div>
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
          <div style="font-size:0.72rem; color:#888; text-transform:uppercase;
                      letter-spacing:0.1em; font-weight:600;">
            Predicted next contract
          </div>
          <div style="font-size:0.78rem; color:#aaa; margin-top:0.15rem;">
            {_player_meta_line}
          </div>
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
if caveats:
    _caveat_chips_html = "".join(
        f'<div style="display:inline-block; background:rgba(243,156,18,0.10); '
        f'border:1px solid rgba(243,156,18,0.30); border-radius:6px; '
        f'padding:0.3rem 0.7rem; margin: 0 0.4rem 0.4rem 0; '
        f'font-size:0.8rem; color:#f1c40f;">⚠ {note}</div>'
        for note in caveats
    )
    st.markdown(
        f'<div style="margin: -0.4rem 0 1rem 0;">{_caveat_chips_html}</div>',
        unsafe_allow_html=True,
    )

# Math breakdown lives inside the "About this prediction" expander below —
# keeps the main view clean (Model / Market / Honest range) while still
# letting curious users see the per-player calculation.

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
                <b style="color:#cdcdd5;">X factor:</b> {take['x_factor']}
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
        6. **CBA max cap** — derived from years of NBA service. 0-6 yrs:
           25% of cap. 7-9 yrs: 30%. 10+ yrs: 35%. Caps the projection
           because no player can legally earn more than their max.
        7. **Supermax floor** — for players with recent All-NBA selections
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
        - Draft pedigree (lottery / mid-1st / late-1st / 2nd / undrafted)
        - **All-NBA selections** (scraped from BBRef awards page)
        - **NBA service years** (derived from career data)
        - **Years with current team** (Bird-rights proxy via consecutive
          seasons on the same team — derived from career data)
        - **Rookie-scale lock** (uses our existing rookie-scale roster)
        - **CBA max-contract tiers** (25% / 30% / 35% based on service)
        - **Designated Rookie / Designated Vet (supermax) eligibility**

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

        Validated out-of-sample on 1,406 actual new contracts signed since
        2015: median error 1.8% of cap (~$2.7M), 79% of predictions land
        within 5% of cap (~$8M), 94% within 10%. The biggest remaining
        misses are veteran-minimum signings and one-off paycut deals where
        market value doesn't apply.
        """
    )
