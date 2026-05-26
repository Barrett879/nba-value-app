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
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
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


# ── Helpers shared by both tabs ──────────────────────────────────────────────
def _fmt_money(v) -> str:
    if pd.isna(v) or v == 0:
        return "—"
    return f"${v / 1_000_000:.1f}M"


def _fmt_pct(v) -> str:
    if pd.isna(v):
        return "—"
    return f"{v * 100:+.0f}%"


def _stat_card(label: str, value: str, sub: str, color: str) -> str:
    return f"""
    <div style="background:rgba(255,255,255,0.03); border:1px solid {color}40;
                border-radius:10px; padding:1.2rem 1.5rem; text-align:center;">
        <div style="font-size:0.72rem; color:#888; letter-spacing:0.08em;
                    text-transform:uppercase; font-weight:600;
                    margin-bottom:0.4rem;">{label}</div>
        <div style="font-size:2.4rem; font-weight:700; color:{color};
                    line-height:1;">{value}</div>
        <div style="font-size:0.78rem; color:#999; margin-top:0.4rem;">{sub}</div>
    </div>
    """


tab_direction, tab_dollars = st.tabs([
    "Directional accuracy",
    "Dollar-amount accuracy",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Directional accuracy: did we call the *direction* (raise vs. paycut)?
# Easy mode — calling Jokić "underpaid" is obvious; he was going to get a raise
# no matter what. The hard test is in Tab 2.
# ══════════════════════════════════════════════════════════════════════════════
with tab_direction:
    _under_graded = underpaid_calls["correct"].dropna()
    _over_graded  = overpaid_calls["correct"].dropna()
    _under_acc = _under_graded.mean() * 100 if len(_under_graded) else None
    _over_acc  = _over_graded.mean()  * 100 if len(_over_graded)  else None

    _acc_l, _acc_r = st.columns(2)
    with _acc_l:
        if _under_acc is None:
            body = "—"
            sub  = "Not enough gradeable outcomes yet"
        else:
            body = f"{_under_acc:.0f}%"
            sub  = f"{int(_under_graded.sum())}/{len(_under_graded)} correct · {TOP_N - len(_under_graded)} non-gradeable"
        st.markdown(_stat_card(
            f"Underpaid calls ({prev_season})", body, sub, "#2ecc71",
        ), unsafe_allow_html=True)
    with _acc_r:
        if _over_acc is None:
            body = "—"
            sub  = "Not enough gradeable outcomes yet"
        else:
            body = f"{_over_acc:.0f}%"
            sub  = f"{int(_over_graded.sum())}/{len(_over_graded)} correct · {TOP_N - len(_over_graded)} non-gradeable"
        st.markdown(_stat_card(
            f"Overpaid calls ({prev_season})", body, sub, "#e63946",
        ), unsafe_allow_html=True)

    st.caption(
        "A call is **correct** when the player's salary moved in the predicted direction "
        "by at least 10–15% (raise for underpaid, paycut or release for overpaid). "
        "Players still on a guaranteed multi-year contract often show ↔ Flat and aren't "
        "counted toward accuracy — those calls haven't had their chance yet."
    )

    st.divider()

    def _build_display(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({
            "Player":      df["Player"].values,
            "Salary then": df["salary_prev"].apply(_fmt_money).values,
            "Salary now":  df["salary_curr"].apply(_fmt_money).values,
            "Δ":           df["pct_change"].apply(_fmt_pct).values,
            "Verdict":     df["verdict"].values,
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

    st.caption(
        f"Sourced from cached rankings on disk. {prev_season} projection compared to "
        f"{curr_season} actual salary. Players who retired, signed overseas, or are on "
        "two-way deals show 'Left league' for the underpaid table (counts as non-gradeable) "
        "and '✓ Left league' for the overpaid table (counts as a correct call — the market "
        "agreed they weren't worth NBA money)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Dollar-amount accuracy: how close was the projected $ to actual $?
# Predicting that a star is underpaid is easy. Predicting *how much* their new
# contract should be is the real model test.
# ══════════════════════════════════════════════════════════════════════════════
with tab_dollars:
    st.markdown(
        "Predicting that a great underpaid player will eventually get paid is easy. "
        "The real test: how close was the **projected dollar amount** to what the "
        "market actually paid?"
    )

    # Pool of testable predictions: anyone who was in the prev pool AND is still
    # in the league this year with a real salary.
    dollar_pool = merged[
        (merged["proj_prev"].fillna(0) > 0)
        & (merged["salary_curr"].fillna(0) > 0)
    ].copy()
    dollar_pool["abs_error"]  = (dollar_pool["salary_curr"] - dollar_pool["proj_prev"]).abs()
    dollar_pool["signed_err"] = dollar_pool["salary_curr"] - dollar_pool["proj_prev"]
    dollar_pool["pct_err"]    = (dollar_pool["signed_err"] / dollar_pool["salary_curr"]).where(
        dollar_pool["salary_curr"] > 0
    )

    # Optional filter: only players whose contract materially changed YoY (new
    # deal / extension / paycut). Multi-year guaranteed contracts that didn't
    # shift are easy to "predict" since the salary is locked.
    _only_changed = st.checkbox(
        f"Only show players whose {curr_season} salary differs from {prev_season} by ≥15% "
        "(filters out locked multi-year deals, leaving the actual market predictions)",
        value=True,
    )
    if _only_changed:
        dollar_pool = dollar_pool[dollar_pool["pct_change"].abs() >= 0.15].copy()

    if dollar_pool.empty:
        st.warning(
            "No qualifying predictions to grade with the current filter. "
            "Uncheck the filter above or pick an earlier season pair."
        )
        st.stop()

    # ── Summary stat cards ───────────────────────────────────────────────────
    _median_err = float(dollar_pool["abs_error"].median())
    _within_5  = float((dollar_pool["abs_error"] <= 5_000_000).mean() * 100)
    _within_10 = float((dollar_pool["abs_error"] <= 10_000_000).mean() * 100)
    _bias_med  = float(dollar_pool["signed_err"].median())
    _bias_dir  = (
        f"+${_bias_med / 1_000_000:.1f}M (market pays more)" if _bias_med > 0
        else f"−${abs(_bias_med) / 1_000_000:.1f}M (model overshoots)"
    )

    _c1, _c2, _c3, _c4 = st.columns(4)
    with _c1:
        st.markdown(_stat_card(
            "Median |error|",
            f"${_median_err / 1_000_000:.1f}M",
            f"{len(dollar_pool)} predictions in pool",
            "#3498db",
        ), unsafe_allow_html=True)
    with _c2:
        st.markdown(_stat_card(
            "Within $5M",
            f"{_within_5:.0f}%",
            "Hit within $5M of actual",
            "#2ecc71",
        ), unsafe_allow_html=True)
    with _c3:
        st.markdown(_stat_card(
            "Within $10M",
            f"{_within_10:.0f}%",
            "Hit within $10M of actual",
            "#16d4c1",
        ), unsafe_allow_html=True)
    with _c4:
        st.markdown(_stat_card(
            "Median bias",
            _bias_dir.split(" ")[0],
            _bias_dir.split(" ", 1)[1].strip("()"),
            "#f39c12" if abs(_bias_med) > 2_000_000 else "#888",
        ), unsafe_allow_html=True)

    st.caption(
        "**Median |error|** = typical $ gap between projection and actual. "
        "**Within $5M / $10M** = share of predictions that landed inside that band. "
        "**Median bias** ≠ 0 usually means the salary cap rose between seasons "
        "(positive bias = market paid more than the model projected because the cap grew)."
    )

    st.divider()

    # ── Scatter: projection vs. actual ───────────────────────────────────────
    st.subheader("Projection vs. actual")
    st.caption(
        f"Each dot = one player. X-axis: what {prev_season}'s Barrett Score projected. "
        f"Y-axis: what they actually earn in {curr_season}. Dashed line = perfect prediction. "
        "Points above the line = we underprojected; points below = overprojected."
    )

    _x = dollar_pool["proj_prev"].values / 1_000_000
    _y = dollar_pool["salary_curr"].values / 1_000_000
    _err = dollar_pool["abs_error"].values / 1_000_000
    _names = dollar_pool["Player"].values

    _axis_max = float(max(_x.max(), _y.max())) * 1.08

    _scatter = go.Figure()
    # Perfect-prediction diagonal.
    _scatter.add_trace(go.Scatter(
        x=[0, _axis_max], y=[0, _axis_max],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"),
        hoverinfo="skip", showlegend=False,
    ))
    # ±$5M error band as a shaded zone — visual goalposts for "close enough".
    _scatter.add_trace(go.Scatter(
        x=[0, _axis_max, _axis_max, 0, 0],
        y=[5, _axis_max + 5, _axis_max - 5, -5, 5],
        fill="toself", fillcolor="rgba(46, 204, 113, 0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=False,
    ))
    # Actual predictions.
    _scatter.add_trace(go.Scatter(
        x=_x, y=_y,
        mode="markers",
        marker=dict(
            size=9,
            color=_err,
            colorscale=[(0, "#2ecc71"), (0.5, "#f39c12"), (1, "#e63946")],
            cmin=0, cmax=max(15, float(_err.max())),
            colorbar=dict(
                title=dict(text="|Error| ($M)", side="right"),
                thickness=12, len=0.7, x=1.02,
            ),
            line=dict(color="rgba(255,255,255,0.4)", width=0.5),
        ),
        customdata=list(zip(
            _names.tolist(),
            _x.tolist(),
            _y.tolist(),
            _err.tolist(),
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Projected: $%{customdata[1]:.1f}M<br>"
            "Actual: $%{customdata[2]:.1f}M<br>"
            "Error: $%{customdata[3]:.1f}M<extra></extra>"
        ),
        showlegend=False,
    ))
    _scatter.update_layout(
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cdcdd5"),
        margin=dict(l=60, r=110, t=20, b=50),
        xaxis=dict(
            title=f"Projected (from {prev_season}) — $M",
            range=[0, _axis_max],
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.15)",
        ),
        yaxis=dict(
            title=f"Actual ({curr_season}) — $M",
            range=[0, _axis_max],
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.15)",
            scaleanchor="x", scaleratio=1,
        ),
    )
    st.plotly_chart(_scatter, use_container_width=True,
                    config={"displayModeBar": False})

    st.divider()

    # ── Best & worst predictions ─────────────────────────────────────────────
    _best  = dollar_pool.nsmallest(15, "abs_error").copy()
    _worst = dollar_pool.nlargest(15, "abs_error").copy()

    def _build_dollar_display(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "Player":     df["Player"].values,
            "Projected":  df["proj_prev"].apply(_fmt_money).values,
            "Actual":     df["salary_curr"].apply(_fmt_money).values,
            "Error":      df["signed_err"].apply(
                lambda v: ("+" if v >= 0 else "−") + f"${abs(v)/1_000_000:.1f}M"
                if not pd.isna(v) else "—"
            ).values,
            "|Error|":    df["abs_error"].apply(_fmt_money).values,
        })

    _bl, _br = st.columns(2)
    with _bl:
        st.subheader("Sharpest predictions")
        st.caption("Smallest $ gap between projection and actual. The model called it.")
        st.dataframe(
            _build_dollar_display(_best),
            use_container_width=True, hide_index=True,
            height=min(700, 60 + len(_best) * 35),
        )
    with _br:
        st.subheader("Biggest misses")
        st.caption("Largest gaps. Where the model got the dollar amount most wrong.")
        st.dataframe(
            _build_dollar_display(_worst),
            use_container_width=True, hide_index=True,
            height=min(700, 60 + len(_worst) * 35),
        )

    st.caption(
        f"All comparisons use {prev_season} projected salary vs. {curr_season} actual salary. "
        "Salary cap grew between seasons, so a perfectly accurate model would still show "
        "a small positive median bias. The Median bias card above quantifies that drift."
    )
