"""Track Record — how well did last season's Barrett Score calls hold up?

For each of last season's biggest "underpaid" and "overpaid" calls (by
value_diff), we look up that player's salary this season and check whether
the call directionally panned out: underpaid players should command raises,
overpaid players should get released, traded down, or take pay cuts.

This is the most defensible marketing pitch for the site — public receipts
on a quantitative claim. If we're directionally right ~70% of the time,
say so with the actual numbers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from utils import (
    COMMON_CSS, SEASONS,
    build_ranked_projected,
    render_nav, render_barrett_score_explainer, _bootstrap_warm,
)

st.set_page_config(page_title="Track Record", layout="wide")
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
render_nav("Track Record")

st.title("Track Record")
st.caption(
    "Did last season's Barrett Score calls hold up? Every underpaid prediction "
    "from last year, checked against this year's actual salary. Public receipts "
    "on a quantitative claim."
)

render_barrett_score_explainer()

# ── Season pair ──────────────────────────────────────────────────────────────
# Default to current season vs. previous, but let users pick any pair so they
# can grade calls from previous years too.
_curr_default = SEASONS[0]
_prev_default = SEASONS[1] if len(SEASONS) > 1 else SEASONS[0]

_col_l, _col_r = st.columns([1, 1])
with _col_l:
    prev_season = st.selectbox(
        "We said (last season)",
        SEASONS[1:],
        index=0,
        help="The season whose Barrett-Score-based projections we're grading.",
    )
with _col_r:
    # Curr season must be after prev. SEASONS is newest→oldest, so curr must
    # be at a *lower* index than prev.
    _prev_idx = SEASONS.index(prev_season)
    _curr_options = SEASONS[:_prev_idx]
    if not _curr_options:
        _curr_options = [SEASONS[0]]
    curr_season = st.selectbox(
        "What actually happened (this season)",
        _curr_options,
        index=0,
        help="The season whose actual salaries we use to verify the call.",
    )

# ── Load both seasons' ranked data ────────────────────────────────────────────
prev_df = build_ranked_projected(prev_season).copy()
curr_df = build_ranked_projected(curr_season).copy()

if prev_df.empty:
    st.warning(f"No ranked data on disk for {prev_season} yet. Pick an earlier season pair.")
    st.stop()
if curr_df.empty:
    st.warning(f"No ranked data on disk for {curr_season} yet. Pick an earlier season pair.")
    st.stop()

# Sanity: need salary data on both sides.
prev_df = prev_df[prev_df["salary"] > 0].copy()
if prev_df.empty:
    st.warning(f"Salary coverage for {prev_season} is too sparse to grade calls.")
    st.stop()

# Join by PLAYER_ID (immune to name diacritic / casing differences).
join_cols = ["PLAYER_ID", "Player", "salary", "projected_salary", "value_diff", "barrett_score"]
prev_slim = prev_df[join_cols].rename(columns={
    "salary":           "salary_prev",
    "projected_salary": "proj_prev",
    "value_diff":       "value_diff_prev",
    "barrett_score":    "barrett_prev",
})
curr_slim = curr_df[["PLAYER_ID", "salary", "barrett_score"]].rename(columns={
    "salary":        "salary_curr",
    "barrett_score": "barrett_curr",
})

merged = prev_slim.merge(curr_slim, on="PLAYER_ID", how="left")
merged["salary_change"]   = merged["salary_curr"].fillna(0) - merged["salary_prev"]
merged["pct_change"]      = (merged["salary_change"] / merged["salary_prev"]).where(
    merged["salary_prev"] > 0
)
merged["still_in_league"] = merged["salary_curr"].fillna(0) > 0


# ── Verdict helpers ───────────────────────────────────────────────────────────
def _verdict_for_underpaid(row) -> str:
    """We said: due for a raise. Did they get one?"""
    if not row["still_in_league"]:
        return "Left league"
    pct = row["pct_change"]
    if pd.isna(pct):
        return "—"
    if pct >= 0.15:
        return "✓ Got raise"
    if pct <= -0.10:
        return "✗ Paycut"
    return "↔ Flat"


def _verdict_for_overpaid(row) -> str:
    """We said: value risk. Did the contract correct?"""
    if not row["still_in_league"]:
        return "✓ Left league"
    pct = row["pct_change"]
    if pd.isna(pct):
        return "—"
    if pct <= -0.10:
        return "✓ Paycut"
    if pct >= 0.15:
        return "✗ Got raise"
    return "↔ Flat"


def _verdict_score(verdict: str) -> int | None:
    """Map verdict text to a numeric correctness flag for accuracy %.
    1 = correct call, 0 = wrong call, None = not gradeable (flat / left league).
    """
    if verdict.startswith("✓"):
        return 1
    if verdict.startswith("✗"):
        return 0
    return None  # ↔ Flat, Left league (for underpaid), — etc. don't count


# Top N most-underpaid and most-overpaid from last season.
TOP_N = 20
underpaid_calls = merged.nsmallest(TOP_N, "value_diff_prev").copy()
underpaid_calls["verdict"] = underpaid_calls.apply(_verdict_for_underpaid, axis=1)
underpaid_calls["correct"] = underpaid_calls["verdict"].apply(_verdict_score)

overpaid_calls = merged.nlargest(TOP_N, "value_diff_prev").copy()
overpaid_calls["verdict"] = overpaid_calls.apply(_verdict_for_overpaid, axis=1)
overpaid_calls["correct"] = overpaid_calls["verdict"].apply(_verdict_score)

# ── Accuracy summary cards ────────────────────────────────────────────────────
_under_graded = underpaid_calls["correct"].dropna()
_over_graded  = overpaid_calls["correct"].dropna()
_under_acc = _under_graded.mean() * 100 if len(_under_graded) else None
_over_acc  = _over_graded.mean()  * 100 if len(_over_graded)  else None

def _acc_card(label: str, accuracy: float | None, graded_n: int, total_n: int, color: str) -> str:
    if accuracy is None:
        body = '<div class="acc-num" style="color:#888;">—</div>'
        sub  = "Not enough gradeable outcomes yet"
    else:
        body = f'<div class="acc-num" style="color:{color};">{accuracy:.0f}%</div>'
        sub  = f"{int(_under_graded.sum() if 'Underpaid' in label else _over_graded.sum())}/{graded_n} correct · {total_n - graded_n} non-gradeable"
    return f"""
    <div style="background:rgba(255,255,255,0.03); border:1px solid {color}40;
                border-radius:10px; padding:1.2rem 1.5rem; text-align:center;">
        <div style="font-size:0.72rem; color:#888; letter-spacing:0.08em;
                    text-transform:uppercase; font-weight:600;
                    margin-bottom:0.4rem;">{label}</div>
        {body}
        <div style="font-size:0.78rem; color:#999; margin-top:0.4rem;">{sub}</div>
    </div>
    """

_acc_l, _acc_r = st.columns(2)
with _acc_l:
    st.markdown(_acc_card(
        f"Underpaid calls ({prev_season})",
        _under_acc, len(_under_graded), TOP_N, "#2ecc71",
    ), unsafe_allow_html=True)
with _acc_r:
    st.markdown(_acc_card(
        f"Overpaid calls ({prev_season})",
        _over_acc, len(_over_graded), TOP_N, "#e63946",
    ), unsafe_allow_html=True)

st.caption(
    "A call is **correct** when the player's salary moved in the predicted direction "
    "by at least 10–15% (raise for underpaid, paycut or release for overpaid). "
    "Players still on a guaranteed multi-year contract often show ↔ Flat and aren't "
    "counted toward accuracy — those calls haven't had their chance yet."
)

st.divider()


# ── Render side-by-side tables ────────────────────────────────────────────────
def _fmt_money(v) -> str:
    if pd.isna(v) or v == 0:
        return "—"
    return f"${v / 1_000_000:.1f}M"


def _fmt_pct(v) -> str:
    if pd.isna(v):
        return "—"
    return f"{v * 100:+.0f}%"


def _build_display(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Player":    df["Player"].values,
        "Salary then":     df["salary_prev"].apply(_fmt_money).values,
        "Salary now":      df["salary_curr"].apply(_fmt_money).values,
        "Δ":               df["pct_change"].apply(_fmt_pct).values,
        "Verdict":         df["verdict"].values,
    })
    return out


_tbl_l, _tbl_r = st.columns(2)
with _tbl_l:
    st.subheader(f"We said: Underpaid · top {TOP_N} ({prev_season})")
    st.caption("Players whose Barrett Score said they deserved a much bigger contract.")
    _disp = _build_display(underpaid_calls)
    st.dataframe(
        _disp, use_container_width=True, hide_index=True,
        height=min(700, 60 + len(_disp) * 35),
    )

with _tbl_r:
    st.subheader(f"We said: Overpaid · top {TOP_N} ({prev_season})")
    st.caption("Players whose contract was outsized relative to their Barrett Score.")
    _disp = _build_display(overpaid_calls)
    st.dataframe(
        _disp, use_container_width=True, hide_index=True,
        height=min(700, 60 + len(_disp) * 35),
    )

st.divider()
st.caption(
    f"Sourced from cached rankings on disk. {prev_season} projection compared to "
    f"{curr_season} actual salary. Players who retired, signed overseas, or are on "
    "two-way deals show 'Left league' for the underpaid table (counts as non-gradeable) "
    "and '✓ Left league' for the overpaid table (counts as a correct call — the market "
    "agreed they weren't worth NBA money)."
)
