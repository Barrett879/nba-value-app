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
    CONTRACT_AGE_MULTIPLIERS as AGE_MULTIPLIERS,
    CONTRACT_POSITION_MULTIPLIERS as POSITION_MULTIPLIERS,
    CONFIDENCE_BAND_PCT_OF_CAP,
    HEALTHY_SEASON_GP, NEW_CONTRACT_PCT, SUPERMAX_CAP_PCT,
    age_bucket as _age_bucket,
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

    # ── Career-weighted Barrett Score — the GM's view ────────────────────────
    # Real front offices project contracts off a body of work, not a half
    # season. We pull the player's full per-season career and weight the
    # last three completed seasons (50/30/20, most recent first), skipping
    # the current season since it's still in progress (and injuries /
    # ramp-up artificially deflate availability-adjusted scores).
    career_barrett = None
    career_basis = "current season (no prior data)"
    try:
        career = fetch_player_full_career(player_name)
        if not career.empty:
            # Apply the same GP ≥ 40 filter to ALL seasons including the
            # current one. Late in the season this naturally includes the
            # current year (Luka with 70+ games gets picked up); early in
            # the season it excludes the in-progress year (only 10-20
            # games played, availability multiplier deflates the score).
            # Historical injury years (e.g. Zach LaVine's 25-GP 2023-24)
            # also drop out the same way.
            healthy = career[career["GP"] >= HEALTHY_SEASON_GP]
            used_healthy_filter = len(healthy) >= 1
            pool = healthy if used_healthy_filter else career

            if not pool.empty:
                # Take the most recent up to 3 healthy seasons (50/30/20
                # weighting, most recent gets the highest weight).
                recent = pool.tail(3)
                weights_full = [0.20, 0.30, 0.50]
                weights = weights_full[-len(recent):]
                w_sum = sum(weights)
                career_barrett = float(
                    (recent["Barrett Score"].values * weights).sum() / w_sum
                )
                seasons_used = list(recent["Season"].values)
                skipped = (
                    used_healthy_filter and len(career) > len(pool)
                )
                skip_note = (
                    " · low-GP seasons (<40) skipped" if skipped else ""
                )
                career_basis = (
                    f"weighted avg of {len(recent)} healthy season"
                    f"{'s' if len(recent) > 1 else ''} "
                    f"({', '.join(seasons_used)}){skip_note}"
                )
    except Exception:
        pass

    # Fall back to current-season Barrett if no career history.
    if career_barrett is None:
        career_barrett = float(row["barrett_score"])
        career_basis = "current season only (rookie / first appearance)"

    # Find what salary rank this career-weighted score would command in the
    # current season's pool — gives a stable "GM-view" base projection.
    cur_scores = ranked["barrett_score"].sort_values(ascending=False).values
    cur_salaries = ranked["salary"].sort_values(ascending=False).values
    # Career-weighted rank = number of current players with a higher score
    effective_rank = int((cur_scores > career_barrett).sum()) + 1
    capped_rank = min(effective_rank, len(cur_salaries)) - 1
    career_base_proj = float(cur_salaries[capped_rank])

    return {
        "name":               row["Player"],
        "team":               row.get("Team", ""),
        "age":                age,
        "position":           pos,                # G/F/C — drives multiplier
        "position_detailed":  detailed_display,   # PG/SG/SF/PF/C — for display
        "barrett_score":      float(row["barrett_score"]),       # current season
        "career_barrett":     career_barrett,                    # 3-year weighted
        "career_basis":       career_basis,
        "score_rank":         int(row["score_rank"]),
        "effective_rank":     effective_rank,
        "career_base_proj":   career_base_proj,
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
    age_mult = AGE_MULTIPLIERS.get(_age_bucket(features["age"]), 1.0)
    pos_mult = POSITION_MULTIPLIERS.get(features["position"], 1.0)

    cap_M = SALARY_CAP_M.get(target_season, 154.6)
    cap_dollars_val = cap_M * 1_000_000
    supermax_threshold = cap_dollars_val * SUPERMAX_CAP_PCT

    # Supermax-tier suppression: the position multipliers were fit on
    # mid-market signings, where Centers especially get systematically
    # less than their box-score rank suggests. But supermax/max-contract
    # players sign at fixed CBA percentages of the cap regardless of
    # position — Jokić doesn't get a "Center discount" on his deal.
    pos_mult_applied = pos_mult
    if base >= supermax_threshold:
        pos_mult_applied = 1.0  # no positional discount at the top tier

    # Tier-aware age multiplier:
    # The standard age multipliers (0.57 at 35+, 0.72 at 32-34) are fit on
    # the AVERAGE aging player, who takes a paycut to keep playing. But
    # stars currently earning at supermax tier are a structurally
    # different cohort — they sign for max% (CBA-capped, not market-
    # discounted). Treating Curry like the average 38yo PG predicts $25M
    # for a guy currently making $60M; that's wrong by ~50%.
    #
    # Heuristic: if the player is currently earning ≥28% of cap (supermax
    # tier) AND their career Score still places them in the elite pool,
    # floor the age multiplier at 0.85. They might decline 15% but not
    # 45%.
    current_salary = float(features.get("salary", 0) or 0)
    current_salary_pct = current_salary / cap_dollars_val if cap_dollars_val > 0 else 0
    age_mult_applied = age_mult
    age_mult_floored = False
    if (
        current_salary_pct >= SUPERMAX_CAP_PCT
        and base >= supermax_threshold
        and age_mult < 0.85
    ):
        age_mult_applied = 0.85
        age_mult_floored = True

    predicted = base * age_mult_applied * pos_mult_applied
    band = cap_dollars_val * CONFIDENCE_BAND_PCT_OF_CAP

    return {
        "base":                base,
        "age_mult":            age_mult_applied,    # the one actually used
        "age_mult_raw":        age_mult,            # the unfloored value
        "age_mult_floored":    age_mult_floored,
        "pos_mult":            pos_mult_applied,
        "pos_mult_raw":        pos_mult,
        "pos_mult_suppressed": pos_mult_applied != pos_mult,
        "predicted":           predicted,
        "low":                 max(0, predicted - band),
        "high":                predicted + band,
        "band":                band,
        "cap":                 cap_dollars_val,
    }


def detect_caveats(features: dict) -> list[str]:
    notes: list[str] = []
    age = features.get("age")
    salary = features.get("salary", 0)
    barrett = features.get("barrett_score", 0)

    if salary > 0 and salary < 12_000_000 and age and age <= 23:
        notes.append(
            "Possibly on rookie scale — locked salary by CBA until contract "
            "expires (usually year 4). The market price below forecasts their "
            "next contract, not their current one."
        )
    if age and age >= 27 and barrett >= 28:
        notes.append(
            "Star-tier producer — if All-NBA-eligible, may sign a supermax "
            "(35% of cap, ~$54M in 2025-26), which would exceed this projection."
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
        m["signed_in"] = curr
        m["prev_season"] = prev  # needed to compute career-weighted-at-signing
        rows.append(m[[
            "Player", "age", "pos", "pos_detailed", "barrett_score",
            "salary", "salary_curr", "signed_in", "prev_season",
        ]])

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["age", "barrett_score", "salary_curr"])

    # Exclude rookie-scale ladder steps (year-N → year-N+1 team-option
    # progressions, CBA-mandated). These aren't real new contracts but
    # they're often >25% YoY change because rookie scale escalates.
    # Signal: both salaries still inside rookie scale band (<~15-18% of
    # the year's cap) AND player young (≤23). Legitimate rookie
    # extensions (e.g. Sengun's $5.4M → $33.9M jump) keep their
    # signed_for ABOVE the band, so they're untouched.
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
            and float(age) <= 23
        )

    mask = ~out.apply(_is_rookie_ladder, axis=1)
    out = out[mask].reset_index(drop=True)
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


def find_comparables(features: dict, history: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """Two-pass match:
       1. Coarse: same position, top-30 closest by walk-year Barrett + age
       2. Refine: compute career-weighted Barrett for those 30, re-sort
          using career-weighted vs. career-weighted (apples to apples)
    """
    if history.empty:
        return history

    target_position = features["position"]
    target_age = features["age"] if features["age"] else 27
    target_barrett = features["career_barrett"]

    # ── Coarse pass: prefer same position, take top-30 by quick distance ─────
    same_pos = history[history["pos"] == target_position].copy()
    if len(same_pos) < n:
        # Not enough same-position comparables — open up to all positions
        # with a soft penalty so same-position still wins ties.
        same_pos = history.copy()
    same_pos["coarse_dist"] = (
        (same_pos["barrett_score"] - target_barrett).abs()
        + (same_pos["age"] - target_age).abs() * 1.5
        + (same_pos["pos"] != target_position).astype(float) * 10
    )
    top30 = same_pos.nsmallest(min(30, len(same_pos)), "coarse_dist").copy()

    # ── Refine pass: career-weighted-at-signing, apples to apples ────────────
    top30["career_weighted_barrett"] = top30.apply(
        lambda r: _career_weighted_barrett_at(
            r["Player"], r["prev_season"], float(r["barrett_score"])
        ),
        axis=1,
    )

    pos_match = (top30["pos"] == target_position).astype(int)
    pos_penalty = (1 - pos_match) * 20

    top30["distance"] = (
        (top30["career_weighted_barrett"] - target_barrett).abs()
        + (top30["age"] - target_age).abs() * 1.5
        + pos_penalty
    )
    return top30.nsmallest(n, "distance")


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
    if salary_then < cap * 0.10 and age <= 24 and pct_change >= 0.50:
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
        f'Use the range as the honest answer — single number would overclaim.'
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
            Honest range
          </div>
          <div style="font-size:1.3rem; color:#fff; font-weight:700;
                      line-height:1.1;">
            ${honest_low_M:.1f}M – ${honest_high_M:.1f}M
          </div>
        </div>
      </div>
      {diverge_note}
    </div>
    """
else:
    # No comparables available — fall back to the model-only display.
    _meta_bits = [features["name"]]
    if features.get("team"): _meta_bits.append(features["team"])
    _meta_bits.append(CURRENT_SEASON)
    if features.get("age"): _meta_bits.append(f"Age {int(features['age'])}")
    _meta_bits.append(str(features.get("position_detailed", features["position"])))
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
                Honest range
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

# ── Breakdown — single horizontal line ───────────────────────────────────────
# Was a 4-column grid taking ~3 vertical inches; now a one-line equation with
# tiny detail annotations underneath. Same info, way less screen real estate.
base_M = prediction["base"] / 1_000_000
_pos_factor_note = (
    f" (suppressed from ×{prediction['pos_mult_raw']:.2f} — base ≥28% of cap)"
    if prediction.get("pos_mult_suppressed") else ""
)
_age_factor_note = (
    f" (floored from ×{prediction['age_mult_raw']:.2f} — currently supermax-tier)"
    if prediction.get("age_mult_floored") else ""
)
_breakdown_one_line = f"""
<div style="background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:10px; padding:0.85rem 1.1rem; margin: 0 0 0.7rem 0;
            font-size:0.95rem; color:#cdcdd5; line-height:1.55;">
  <span style="color:#888; font-size:0.7rem; letter-spacing:0.08em;
               text-transform:uppercase; margin-right:0.5rem;">Math</span>
  <b style="color:#fff;">${base_M:.1f}M</b>
  <span style="color:#777;">(career Score {features['career_barrett']:.1f} → rank #{features['effective_rank']})</span>
  &nbsp;<span style="color:#666;">×</span>&nbsp;
  <b>×{prediction['age_mult']:.2f}</b>
  <span style="color:#777;">(age {int(features['age']) if features['age'] else '?'}{_age_factor_note})</span>
  &nbsp;<span style="color:#666;">×</span>&nbsp;
  <b>×{prediction['pos_mult']:.2f}</b>
  <span style="color:#777;">({features.get('position_detailed', features['position'])}{_pos_factor_note})</span>
  &nbsp;<span style="color:#666;">=</span>&nbsp;
  <b style="color:#16d4c1;">${predicted_M:.1f}M</b>
  <span style="color:#666;">±${prediction['band']/1_000_000:.1f}M</span>
  <div style="font-size:0.72rem; color:#777; margin-top:0.35rem;">
    Base uses {features['career_basis']}.
  </div>
</div>
"""
st.markdown(_breakdown_one_line, unsafe_allow_html=True)

# ── Comparables ──────────────────────────────────────────────────────────────
st.subheader("Comparable signings")
st.caption(
    "Closest career-weighted Score + age + position matches. Real signings — "
    "the model's market sanity check."
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
        comp_disp = pd.DataFrame({
            "Player":         comps_with_ctx["Player"].values,
            "Signed in":      comps_with_ctx["signed_in"].values,
            "Age then":       comps_with_ctx["age"].astype(int).values,
            "Position":       _pos_col.values,
            "Context":        comps_with_ctx["context"].values,
            "Career Score":   comps_with_ctx["career_weighted_barrett"].round(1).values,
            "Sign-yr Score":  comps_with_ctx["barrett_score"].round(1).values,
            "Signed for":     [_fmt_money(v) for v in comps_with_ctx["salary_curr"]],
        })
        st.dataframe(comp_disp, use_container_width=True, hide_index=True,
                     height=min(400, 60 + len(comp_disp) * 35))

        st.caption(
            "Context tags: **Supermax** ≥28% cap · **Free-agent raise** ≥15% bump · "
            "**Rookie extension** first non-rookie deal · **Paycut** took less to stay."
        )

# ── Methodology footer (collapsed — info-after-action) ──────────────────────
st.divider()
with st.expander("About this prediction"):
    render_barrett_score_explainer()
    st.markdown(
        """
        ### How the next contract is predicted
        Four layers stack on top of the Barrett Score:

        1. **Career-weighted Barrett Score** — uses a weighted average of
           the player's last 3 **healthy seasons** (GP ≥ 40), with 50/30/20
           weighting (most recent first). The current season is included
           once it has enough games to be representative; injury years
           (historical or in-progress mid-season) are skipped. Matches
           how front offices evaluate bodies of work.
        2. **Base projection** — what the player at that career-weighted rank
           would earn based on the current season's salary distribution.
        3. **Age multiplier** — fit on 2014-22 real new contracts. A 33yo
           signs for ~28% less than a 27yo at the same Barrett Score.
        4. **Position multiplier** — Centers are systematically overprojected
           by the box-score-heavy Barrett Score (rebounds aren't paid like
           points). **Suppressed at the supermax tier** (base ≥28% of cap)
           since max-contract players sign at fixed CBA percentages
           regardless of position.

        **Confidence band:** ±$5.5M reflects out-of-sample median error.

        ### Limitations
        The model uses Barrett Score, age, and position. It **can't see**
        contract structure (Bird rights, supermax eligibility, rookie scale
        lock), team cap space, agent leverage, or off-court factors.
        Validated out-of-sample on 1,406 actual new contracts signed since
        2015: median error 1.8% of cap (~$2.7M), 79% of predictions land
        within 5% of cap (~$8M), 94% within 10%. The misses are usually
        supermax extensions, rookie-scale contracts, or veteran-minimum
        signings — situations the box score can't predict from production
        alone.
        """
    )
