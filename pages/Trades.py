import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from utils import (
    COMMON_CSS, SEASONS,
    HISTORICAL_TRADES,
    get_all_player_names, trade_side_summary,
    render_nav, render_playoff_toggle, _bootstrap_warm,
)

st.set_page_config(page_title="Barrett Score — Trades", layout="wide")
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
render_nav("Trades")

_t_l, _t_r = st.columns([3, 1])
with _t_l:
    if st.session_state.get("playoff_mode", False):
        st.title("Trade Comparison — Playoff Mode")
        st.caption(
            "Stack any two sides of a trade using PLAYOFF Barrett Scores. Salaries stay "
            "regular-season (one annual contract). Useful for trades that swung a "
            "championship — what each side actually delivered when it mattered."
        )
    else:
        st.title("Trade Comparison")
        st.caption(
            "Stack any two sides of a trade against each other. Barrett Score totals + salary "
            "tell you who came out ahead — at the time of the deal, and what actually happened the year after."
        )
with _t_r:
    playoff_mode = render_playoff_toggle()

# ── Mode toggle ────────────────────────────────────────────────────────────────
mode = st.radio(
    "",
    options=["Historical trade", "Build your own"],
    horizontal=True,
    label_visibility="collapsed",
    key="trade_mode",
)

st.divider()

# ── Helpers ────────────────────────────────────────────────────────────────────
def _render_side_summary(label: str, color: str, summary: dict, season: str):
    st.markdown(f"#### {label}")
    if summary["found"]:
        rows = summary["rows"][["Player", "Team", "barrett_score", "salary"]].copy()
        rows.columns = ["Player", "Team", "Barrett Score", "Salary"]
        rows["Salary"] = (rows["Salary"] / 1_000_000).round(2)
        rows = rows.sort_values("Barrett Score", ascending=False).reset_index(drop=True)
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            height=min(400, 40 + 35 * len(rows)),
            column_config={
                "Barrett Score": st.column_config.NumberColumn(format="%.2f"),
                "Salary":        st.column_config.NumberColumn("Salary", format="$%.2fM"),
            },
        )
    else:
        st.info("No matched players in this season's data.")

    if summary["missing"]:
        st.caption(
            f"Not found in {season}: " + ", ".join(summary["missing"]) +
            " (rookies under min-minutes threshold, traded mid-season, or pre-1996 with thin coverage)"
        )

    c1, c2 = st.columns(2)
    c1.metric("Total Barrett",  f"{summary['barrett_total']:.1f}")
    c2.metric("Total Salary",   f"${summary['salary_total']/1_000_000:.1f}M")


def _render_winner_banner(a_total: float, b_total: float,
                           a_label: str, b_label: str, context: str):
    delta = a_total - b_total
    if abs(delta) < 0.5:
        st.markdown(
            f'<div style="background:#2a2a44; border-left:4px solid #888; padding:0.9rem 1.1rem; '
            f'border-radius:6px; color:#fff;"><b>{context}</b><br>'
            f'Roughly even — within 0.5 Barrett points.</div>',
            unsafe_allow_html=True,
        )
    elif delta > 0:
        st.markdown(
            f'<div style="background:#1a3a1a; border-left:4px solid #2ecc71; padding:0.9rem 1.1rem; '
            f'border-radius:6px; color:#fff;"><b>{context}</b><br>'
            f'<b style="color:#2ecc71;">{a_label}</b> wins by '
            f'<b>{abs(delta):.1f}</b> Barrett points.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:#1a3a1a; border-left:4px solid #2ecc71; padding:0.9rem 1.1rem; '
            f'border-radius:6px; color:#fff;"><b>{context}</b><br>'
            f'<b style="color:#2ecc71;">{b_label}</b> wins by '
            f'<b>{abs(delta):.1f}</b> Barrett points.</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Historical trade mode
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Historical trade":
    trade_names = [t["name"] for t in HISTORICAL_TRADES]
    pick = st.selectbox("Pick a famous trade", options=trade_names, key="hist_trade_pick")
    trade = next(t for t in HISTORICAL_TRADES if t["name"] == pick)

    season_at  = trade["season"]
    season_aft = trade["year_after"]

    st.caption(trade["notes"])

    # Trade-year evaluation
    st.markdown(f"### At time of trade — {season_at}")
    sum_a_at = trade_side_summary(tuple(trade["side_a"]), season_at, playoffs=playoff_mode)
    sum_b_at = trade_side_summary(tuple(trade["side_b"]), season_at, playoffs=playoff_mode)

    cA, cB = st.columns(2)
    with cA:
        _render_side_summary(
            f"{trade['side_a_team']} receives",
            "#e63946",
            sum_a_at,
            season_at,
        )
    with cB:
        _render_side_summary(
            f"{trade['side_b_team']} receives",
            "#3498db",
            sum_b_at,
            season_at,
        )

    _render_winner_banner(
        sum_a_at["barrett_total"], sum_b_at["barrett_total"],
        trade["side_a_team"], trade["side_b_team"],
        f"At-trade evaluation ({season_at})",
    )

    st.divider()

    # Year-after evaluation
    st.markdown(f"### One season later — {season_aft}")
    st.caption(
        "How each side actually performed the year after the trade. Players who left in free agency or "
        "got hurt show up missing here."
    )

    sum_a_aft = trade_side_summary(tuple(trade["side_a"]), season_aft, playoffs=playoff_mode)
    sum_b_aft = trade_side_summary(tuple(trade["side_b"]), season_aft, playoffs=playoff_mode)

    cA2, cB2 = st.columns(2)
    with cA2:
        _render_side_summary(
            f"{trade['side_a_team']} side",
            "#e63946",
            sum_a_aft,
            season_aft,
        )
    with cB2:
        _render_side_summary(
            f"{trade['side_b_team']} side",
            "#3498db",
            sum_b_aft,
            season_aft,
        )

    _render_winner_banner(
        sum_a_aft["barrett_total"], sum_b_aft["barrett_total"],
        trade["side_a_team"], trade["side_b_team"],
        f"Year-after reality ({season_aft})",
    )

# ══════════════════════════════════════════════════════════════════════════════
# Build-your-own trade mode
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.caption(
        "Pick the season, then drag any number of players into each side. The math runs as you go."
    )

    season = st.selectbox("Season", options=SEASONS, index=0, key="byo_season")

    # Build the player roster from current-season stats + all-time list
    all_names = get_all_player_names()

    st.markdown("")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### Team A receives")
        side_a = st.multiselect(
            "Players going to Team A",
            options=all_names,
            placeholder="Type a name to add…",
            key="byo_side_a",
            label_visibility="collapsed",
        )
    with cB:
        st.markdown("#### Team B receives")
        side_b = st.multiselect(
            "Players going to Team B",
            options=all_names,
            placeholder="Type a name to add…",
            key="byo_side_b",
            label_visibility="collapsed",
        )

    st.divider()

    overlap = set(side_a) & set(side_b)
    if overlap:
        st.warning(f"Same player on both sides: {', '.join(overlap)}. Remove from one side to compare.")

    if side_a or side_b:
        sum_a = trade_side_summary(tuple(side_a), season, playoffs=playoff_mode)
        sum_b = trade_side_summary(tuple(side_b), season, playoffs=playoff_mode)

        c1, c2 = st.columns(2)
        with c1:
            _render_side_summary("Team A side", "#e63946", sum_a, season)
        with c2:
            _render_side_summary("Team B side", "#3498db", sum_b, season)

        if side_a and side_b:
            _render_winner_banner(
                sum_a["barrett_total"], sum_b["barrett_total"],
                "Team A", "Team B",
                f"Trade evaluation ({season})",
            )
    else:
        st.info("Add at least one player to each side to see the comparison.")
