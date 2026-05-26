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
    get_all_player_names,
    build_ranked_projected, fetch_league_stats, fetch_bref_positions,
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

render_barrett_score_explainer()


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

    try:
        pos_lookup = fetch_bref_positions(season_to_espn_year(season), cache_v=3)
    except Exception:
        pos_lookup = {}
    pos = pos_lookup.get(name_norm, "Unknown")

    return {
        "name":            row["Player"],
        "team":            row.get("Team", ""),
        "age":             age,
        "position":        pos,
        "barrett_score":   float(row["barrett_score"]),
        "score_rank":      int(row["score_rank"]),
        "salary":          float(row.get("salary", 0) or 0),
        "projected_salary": float(row.get("projected_salary", 0) or 0),
        "gp":              int(row.get("GP", 0) or 0),
        "mpg":             float(row.get("MPG", 0) or 0),
        "total_pool_size": len(ranked),
    }


def predict_contract(features: dict, target_season: str = CURRENT_SEASON) -> dict:
    base = features["projected_salary"]
    age_mult = AGE_MULTIPLIERS.get(_age_bucket(features["age"]), 1.0)
    pos_mult = POSITION_MULTIPLIERS.get(features["position"], 1.0)
    predicted = base * age_mult * pos_mult

    cap_M = SALARY_CAP_M.get(target_season, 154.6)
    band = cap_M * 1_000_000 * CONFIDENCE_BAND_PCT_OF_CAP

    return {
        "base":        base,
        "age_mult":    age_mult,
        "pos_mult":    pos_mult,
        "predicted":   predicted,
        "low":         max(0, predicted - band),
        "high":        predicted + band,
        "band":        band,
        "cap":         cap_M * 1_000_000,
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
            pos_lookup = fetch_bref_positions(season_to_espn_year(prev), cache_v=3)
        except Exception:
            continue
        if prev_df.empty or curr_df.empty or raw_prev.empty:
            continue

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
        m["pos"] = m["Player"].map(lambda n: pos_lookup.get(normalize(n), "Unknown"))
        m["signed_in"] = curr
        rows.append(m[[
            "Player", "age", "pos", "barrett_score",
            "salary", "salary_curr", "signed_in",
        ]])

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["age", "barrett_score", "salary_curr"])
    return out


def find_comparables(features: dict, history: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    if history.empty:
        return history

    history = history.copy()
    pos_match = (history["pos"] == features["position"]).astype(int)
    pos_penalty = (1 - pos_match) * 20  # 20-Barrett-point penalty for wrong position

    age = features["age"] if features["age"] else 27
    barrett = features["barrett_score"]
    barrett_diff = (history["barrett_score"] - barrett).abs()
    age_diff = (history["age"] - age).abs() * 1.5

    history["distance"] = barrett_diff + age_diff + pos_penalty
    return history.nsmallest(n, "distance")


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

# ── Big number header ────────────────────────────────────────────────────────
predicted_M = prediction["predicted"] / 1_000_000
low_M  = prediction["low"]  / 1_000_000
high_M = prediction["high"] / 1_000_000

_header_html = f"""
<div style="background:linear-gradient(135deg, rgba(230,57,70,0.10) 0%, rgba(22,212,193,0.08) 100%);
            border:1px solid rgba(255,255,255,0.12); border-radius:14px;
            padding:1.8rem 2rem; margin: 0.5rem 0 1.5rem 0;">
  <div style="display:flex; align-items:baseline; flex-wrap:wrap; gap:1rem;">
    <div style="font-size:0.78rem; color:#888; text-transform:uppercase;
                letter-spacing:0.1em; font-weight:600;">
      Predicted next contract
    </div>
    <div style="margin-left:auto; font-size:0.78rem; color:#888;">
      {features['name']} · {features['team']} · {CURRENT_SEASON}
    </div>
  </div>
  <div style="display:flex; align-items:baseline; gap:1.2rem;
              margin-top:0.6rem; flex-wrap:wrap;">
    <div style="font-size:3.2rem; font-weight:800; color:#fff; line-height:1;">
      ${predicted_M:.1f}M
    </div>
    <div style="color:#aaa; font-size:1rem;">
      per year · range
      <b style="color:#cdcdd5;">${low_M:.1f}M</b> –
      <b style="color:#cdcdd5;">${high_M:.1f}M</b>
    </div>
  </div>
</div>
"""
st.markdown(_header_html, unsafe_allow_html=True)

# ── Structural caveats ────────────────────────────────────────────────────────
if caveats:
    for note in caveats:
        st.info(f"⚠ {note}")

# ── Breakdown ────────────────────────────────────────────────────────────────
st.subheader("Why this number?")
base_M = prediction["base"] / 1_000_000
breakdown_html = f"""
<div style="display:grid; grid-template-columns: auto auto auto auto; gap:0.6rem;
            align-items:center; margin: 0.5rem 0 1rem 0;
            background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:10px; padding:1rem 1.4rem;">

  <div style="font-size:0.74rem; color:#888; letter-spacing:0.05em;
              text-transform:uppercase;">Base (Barrett rank)</div>
  <div style="font-size:0.74rem; color:#888; letter-spacing:0.05em;
              text-transform:uppercase;">× Age factor</div>
  <div style="font-size:0.74rem; color:#888; letter-spacing:0.05em;
              text-transform:uppercase;">× Position factor</div>
  <div style="font-size:0.74rem; color:#888; letter-spacing:0.05em;
              text-transform:uppercase;">= Predicted</div>

  <div style="font-size:1.4rem; color:#fff; font-weight:600;">${base_M:.1f}M</div>
  <div style="font-size:1.4rem; color:#cdcdd5;">×{prediction['age_mult']:.2f}</div>
  <div style="font-size:1.4rem; color:#cdcdd5;">×{prediction['pos_mult']:.2f}</div>
  <div style="font-size:1.4rem; color:#16d4c1; font-weight:700;">${predicted_M:.1f}M</div>

  <div style="color:#666; font-size:0.78rem;">Rank #{features['score_rank']} of {features['total_pool_size']}</div>
  <div style="color:#666; font-size:0.78rem;">Age {int(features['age']) if features['age'] else '?'} · "{_age_bucket(features['age'])}"</div>
  <div style="color:#666; font-size:0.78rem;">{features['position']}</div>
  <div style="color:#666; font-size:0.78rem;">±${prediction['band']/1_000_000:.1f}M band</div>
</div>
"""
st.markdown(breakdown_html, unsafe_allow_html=True)

# ── Player snapshot ──────────────────────────────────────────────────────────
st.subheader("Player snapshot")
snap_cols = st.columns(5)
with snap_cols[0]:
    st.metric("Age", int(features["age"]) if features["age"] else "—")
with snap_cols[1]:
    st.metric("Position", features["position"])
with snap_cols[2]:
    st.metric("Barrett Score", f"{features['barrett_score']:.1f}")
with snap_cols[3]:
    st.metric("Score Rank", f"#{features['score_rank']}")
with snap_cols[4]:
    st.metric("Current Salary", _fmt_money(features["salary"]))

# ── Comparables ──────────────────────────────────────────────────────────────
st.subheader("Comparable signings (last 3 seasons)")
st.caption(
    "Players with the closest (Barrett Score, age, position) profile who actually "
    "signed new contracts in recent seasons. The most useful sanity check on the "
    "predicted dollar amount."
)

history = load_historical_signings(n_recent_pairs=3)
if history.empty:
    st.info("No historical comparables on disk yet.")
else:
    comps = find_comparables(features, history, n=6)
    if comps.empty:
        st.info("No close comparables found.")
    else:
        comp_disp = pd.DataFrame({
            "Player":      comps["Player"].values,
            "Signed in":   comps["signed_in"].values,
            "Age then":    comps["age"].astype(int).values,
            "Position":    comps["pos"].values,
            "Barrett":     comps["barrett_score"].round(1).values,
            "Salary then": [_fmt_money(v) for v in comps["salary"]],
            "Signed for":  [_fmt_money(v) for v in comps["salary_curr"]],
            "Δ":           [f"{((c - p)/p)*100:+.0f}%" if p > 0 else "—"
                            for p, c in zip(comps["salary"], comps["salary_curr"])],
        })
        st.dataframe(comp_disp, use_container_width=True, hide_index=True,
                     height=min(400, 60 + len(comp_disp) * 35))

        actuals = comps["salary_curr"].values
        st.caption(
            f"**Comparables-implied range:** "
            f"${min(actuals)/1e6:.1f}M – ${max(actuals)/1e6:.1f}M, "
            f"median ${pd.Series(actuals).median()/1e6:.1f}M. "
            "Use this as a second opinion on the model's prediction above."
        )

# ── Methodology footer ───────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Limitations** · The model uses Barrett Score (production), age, and "
    "position — it can't see contract structure (Bird rights, supermax "
    "eligibility, rookie scale lock), team cap space, agent leverage, or "
    "off-court factors. Median out-of-sample error is 1.8% of cap (~$2.7M in "
    "current dollars); 80% of predictions land within $8M of actual; the 20% "
    "that don't are usually supermax extensions, rookie-scale contracts, or "
    "veteran-minimum signings — situations the box score can't predict."
)
