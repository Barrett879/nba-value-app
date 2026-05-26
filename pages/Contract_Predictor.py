"""Contract Predictor — predict a player's next contract.

Takes a player's current production (Barrett Score), applies age + position
calibration multipliers learned from 2014-22 historical contracts, and returns
a dollar projection with a confidence band and a list of comparable signings.

Out-of-sample accuracy: ~80% within 5% of cap on 435 real new contracts since
2022. Median error 1.8% of cap (~$2.7M in 2025-26 dollars).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils import (
    COMMON_CSS, SEASONS, normalize, season_to_espn_year,
    get_all_player_names, fetch_player_full_career,
    build_ranked_projected, fetch_league_stats,
    fetch_bref_positions, fetch_player_positions_detailed, position_to_bucket,
    render_nav, render_barrett_score_explainer, _bootstrap_warm,
)


# ── Calibration parameters (from out-of-sample test, 2014-22 training) ───────
# Age multipliers — front offices heavily discount age.
AGE_MULTIPLIERS = {
    "≤22":   0.890,
    "23-25": 0.971,
    "26-28": 1.000,
    "29-31": 1.000,
    "32-34": 0.723,
    "35+":   0.574,
}

# Position multipliers — Centers systematically overprojected by the
# box-score-weighted Barrett Score (rebounds aren't paid like points).
POSITION_MULTIPLIERS = {
    "Guard":   0.971,
    "Forward": 0.949,
    "Center":  0.810,
    "Unknown": 0.960,
}

SALARY_CAP_M = {
    "2015-16": 70.0,  "2016-17": 94.1,  "2017-18": 99.1,  "2018-19": 101.9,
    "2019-20": 109.1, "2020-21": 109.1, "2021-22": 112.4, "2022-23": 123.7,
    "2023-24": 136.0, "2024-25": 140.6, "2025-26": 154.6,
}

CURRENT_SEASON = SEASONS[0]
CONFIDENCE_BAND_PCT_OF_CAP = 3.6 / 100  # ≈ 2 × median |err| out-of-sample


# ── Page boilerplate ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Contract Predictor", layout="wide")
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
render_nav("Contract Predictor")

st.title("Contract Predictor")
st.caption(
    "Type a player's name to see their projected next contract. Based on the "
    "Barrett Score, adjusted for age and position. Out-of-sample accuracy: "
    "80% within 5% of cap on 435 real new contracts since 2022."
)

# Methodology expanders live at the bottom of the page (after the prediction
# and comparables) so the page leads with the answer, not the methodology.


# ── Helpers ──────────────────────────────────────────────────────────────────
def _age_bucket(age) -> str:
    if pd.isna(age):
        return "UNK"
    age = int(age)
    if age <= 22: return "≤22"
    if age <= 25: return "23-25"
    if age <= 28: return "26-28"
    if age <= 31: return "29-31"
    if age <= 34: return "32-34"
    return "35+"


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
            # Drop the in-progress current season — partial-season data
            # gets penalized by the availability multiplier mid-year.
            completed = career[career["Season"] != season].copy()
            if not completed.empty:
                # Take the last (most recent) up to 3 completed seasons.
                recent = completed.tail(3)
                weights_full = [0.20, 0.30, 0.50]
                weights = weights_full[-len(recent):]
                w_sum = sum(weights)
                career_barrett = float(
                    (recent["Barrett Score"].values * weights).sum() / w_sum
                )
                seasons_used = list(recent["Season"].values)
                career_basis = (
                    f"weighted avg of {len(recent)} prior season"
                    f"{'s' if len(recent) > 1 else ''} "
                    f"({', '.join(seasons_used)})"
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

    # Supermax-tier suppression: the position multipliers were fit on
    # mid-market signings, where Centers especially get systematically
    # less than their box-score rank suggests. But supermax/max-contract
    # players sign at fixed CBA percentages of the cap regardless of
    # position — Jokić doesn't get a "Center discount" on his deal.
    # When the base projection is already ≥28% of cap (supermax / near-max
    # tier), unwind the position multiplier so we don't fake-discount
    # the league's best players.
    cap_M = SALARY_CAP_M.get(target_season, 154.6)
    supermax_threshold = cap_M * 1_000_000 * 0.28
    pos_mult_applied = pos_mult
    if base >= supermax_threshold:
        pos_mult_applied = 1.0  # no positional discount at the top tier

    predicted = base * age_mult * pos_mult_applied

    cap_M = SALARY_CAP_M.get(target_season, 154.6)
    band = cap_M * 1_000_000 * CONFIDENCE_BAND_PCT_OF_CAP

    return {
        "base":              base,
        "age_mult":          age_mult,
        "pos_mult":          pos_mult_applied,    # the multiplier actually used
        "pos_mult_raw":      pos_mult,            # the unsuppressed value, for transparency
        "pos_mult_suppressed": pos_mult_applied != pos_mult,
        "predicted":         predicted,
        "low":               max(0, predicted - band),
        "high":              predicted + band,
        "band":              band,
        "cap":               cap_M * 1_000_000,
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
        m = m[m["pct_change"].abs() >= 0.25]
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
    return out


def _career_weighted_barrett_at(player_name: str, up_to_season: str,
                                fallback_score: float) -> float:
    """Weighted avg of a player's last 3 completed seasons BEFORE signing.
    Mirrors the same 50/30/20 weighting we use for the live player.
    Falls back to the walk-year score if no career data."""
    try:
        career = fetch_player_full_career(player_name)
        if career.empty:
            return fallback_score
        # Sort by season chronologically — fetch_player_full_career already
        # returns oldest→newest. Include seasons up to and including the
        # season they played before signing.
        up_to = career[career["Season"] <= up_to_season]
        if up_to.empty:
            return fallback_score
        recent = up_to.tail(3)
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
    if salary_signed >= cap * 0.28:
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


def _scouting_take(features: dict, comps: pd.DataFrame) -> dict:
    """Build the 'Scouting take' summary: top-3 names, median deal, IQR
    range, X-factor narrative."""
    if comps.empty:
        return {}

    salaries = comps["salary_curr"].astype(float).values
    median = float(pd.Series(salaries).median())
    q25 = float(pd.Series(salaries).quantile(0.25))
    q75 = float(pd.Series(salaries).quantile(0.75))

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
                f"higher career Barrett than median comp ({target_bar:.1f} vs {comp_bar_med:.1f}) "
                "— premium lean"
            )
        else:
            x_factor_parts.append(
                f"lower career Barrett than median comp ({target_bar:.1f} vs {comp_bar_med:.1f}) "
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

_default_idx = None
if "player" in st.query_params:
    qp = st.query_params["player"]
    qp_resolved = next(
        (n for n in active_names if normalize(n) == normalize(qp)),
        None,
    )
    if qp_resolved:
        _default_idx = active_names.index(qp_resolved)

selected = st.selectbox(
    "Player",
    options=[""] + active_names,
    index=0 if _default_idx is None else _default_idx + 1,
    placeholder="Type a name…",
    label_visibility="collapsed",
)

if not selected:
    st.info(
        f"**{len(active_names):,} active players** available. Try a star to see "
        "a supermax-eligible note, a veteran for the age discount, or a young "
        "rising player for the rookie-scale caveat."
    )
    st.stop()

st.query_params["player"] = selected

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
_market_median = (
    float(_comps["salary_curr"].median()) if not _comps.empty else None
)

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
_breakdown_one_line = f"""
<div style="background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:10px; padding:0.85rem 1.1rem; margin: 0 0 0.7rem 0;
            font-size:0.95rem; color:#cdcdd5; line-height:1.55;">
  <span style="color:#888; font-size:0.7rem; letter-spacing:0.08em;
               text-transform:uppercase; margin-right:0.5rem;">Math</span>
  <b style="color:#fff;">${base_M:.1f}M</b>
  <span style="color:#777;">(career Barrett {features['career_barrett']:.1f} → rank #{features['effective_rank']})</span>
  &nbsp;<span style="color:#666;">×</span>&nbsp;
  <b>×{prediction['age_mult']:.2f}</b>
  <span style="color:#777;">(age {int(features['age']) if features['age'] else '?'})</span>
  &nbsp;<span style="color:#666;">×</span>&nbsp;
  <b>×{prediction['pos_mult']:.2f}</b>
  <span style="color:#777;">({features.get('position_detailed', features['position'])}{_pos_factor_note})</span>
  &nbsp;<span style="color:#666;">=</span>&nbsp;
  <b style="color:#16d4c1;">${predicted_M:.1f}M</b>
  <span style="color:#666;">±${prediction['band']/1_000_000:.1f}M</span>
  <div style="font-size:0.72rem; color:#777; margin-top:0.35rem;">
    Base uses {features['career_basis']} (not the in-progress current season).
  </div>
</div>
"""
st.markdown(_breakdown_one_line, unsafe_allow_html=True)

# ── Comparables ──────────────────────────────────────────────────────────────
st.subheader("Comparable signings")
st.caption(
    "Closest career-weighted Barrett + age + position matches. Real signings — "
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
        comps_with_ctx = comps.copy()
        comps_with_ctx["context"] = comps_with_ctx.apply(_classify_context, axis=1)

        _pos_col = comps_with_ctx.get("pos_detailed", comps_with_ctx["pos"]).fillna(
            comps_with_ctx["pos"])
        comp_disp = pd.DataFrame({
            "Player":         comps_with_ctx["Player"].values,
            "Signed in":      comps_with_ctx["signed_in"].values,
            "Age then":       comps_with_ctx["age"].astype(int).values,
            "Position":       _pos_col.values,
            "Context":        comps_with_ctx["context"].values,
            "Career Barrett": comps_with_ctx["career_weighted_barrett"].round(1).values,
            "Walk-yr Barrett": comps_with_ctx["barrett_score"].round(1).values,
            "Salary then":    [_fmt_money(v) for v in comps_with_ctx["salary"]],
            "Signed for":     [_fmt_money(v) for v in comps_with_ctx["salary_curr"]],
            "Δ":              [f"{((c - p)/p)*100:+.0f}%" if p > 0 else "—"
                              for p, c in zip(comps_with_ctx["salary"],
                                              comps_with_ctx["salary_curr"])],
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

        1. **Career-weighted Barrett Score** — instead of just the in-progress
           current season (which gets deflated by partial games + the
           availability multiplier), the projection uses a weighted average
           of the player's last 3 *completed* seasons (50/30/20 weighting).
           Matches how front offices evaluate bodies of work.
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
        Median out-of-sample error is 1.8% of cap (~$2.7M); 80% of
        predictions land within $8M of actual. The 20% that don't are
        usually supermax extensions, rookie-scale contracts, or
        veteran-minimum signings — situations the box score can't predict.
        """
    )
