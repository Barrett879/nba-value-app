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

# Playoff mode lives in the top nav bar (sticky across pages via session_state)
playoff_mode = bool(st.session_state.get("playoff_mode", False))
if playoff_mode:
    st.title("Trade Comparison — Playoff Mode")
    st.caption(
        "Player-side breakdowns use PLAYOFF Barrett Scores. Salaries always reflect "
        "regular-season contracts. The verdicts and key-points stay grounded in actual "
        "outcomes (championships, Finals, who actually delivered)."
    )
else:
    st.title("Trade Comparison")
    st.caption(
        "Stack any two sides of a trade and compare the player breakdowns. "
        "For famous trades, the verdict is grounded in what actually happened — "
        "championships, Finals runs, what the picks became — not just summed Barrett Scores."
    )

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
def _render_side_summary(label: str, color: str, summary: dict, season: str,
                          picks_note: str | None = None):
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

    if picks_note and picks_note != "—":
        st.caption(f"**Plus:** {picks_note}")


def _render_verdict(trade: dict):
    """Editorial verdict box, grounded in actual outcomes (titles, Finals, etc).
    Replaces the previous misleading 'X wins by N Barrett points' auto-banner.
    """
    winner = trade.get("winner", "wash")
    verdict = trade.get("verdict", "")
    points  = trade.get("key_points", [])

    if winner == "side_a":
        bg, border = "#1a3a1a", "#2ecc71"
        winner_label = trade["side_a_team"]
    elif winner == "side_b":
        bg, border = "#1a3a1a", "#2ecc71"
        winner_label = trade["side_b_team"]
    else:
        bg, border = "#2a2a44", "#888"
        winner_label = "Roughly even"

    points_html = "".join(
        f'<li style="margin-bottom:0.25rem;">{p}</li>' for p in points
    )

    st.markdown(
        f'<div style="background:{bg}; border-left:4px solid {border}; '
        f'padding:1rem 1.2rem; border-radius:6px; color:#fff;">'
        f'<div style="font-size:0.75rem; opacity:0.7; text-transform:uppercase; letter-spacing:0.08em;">Verdict</div>'
        f'<div style="font-size:1.15rem; font-weight:700; margin:0.2rem 0 0.5rem 0;">{verdict}</div>'
        f'<ul style="margin:0.4rem 0 0 1.2rem; padding:0; font-size:0.88rem; opacity:0.92;">{points_html}</ul>'
        f'</div>',
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

    # Editorial verdict up top — the core of the page now
    _render_verdict(trade)
    st.markdown("")

    # Trade-year player breakdown
    st.markdown(f"### Player breakdown — {season_at}")
    st.caption(
        "Barrett Scores from the season the trade landed in. "
        "Picks / future assets listed below each side where applicable."
    )
    sum_a_at = trade_side_summary(tuple(trade["side_a"]), season_at, playoffs=playoff_mode)
    sum_b_at = trade_side_summary(tuple(trade["side_b"]), season_at, playoffs=playoff_mode)

    cA, cB = st.columns(2)
    with cA:
        _render_side_summary(
            f"{trade['side_a_team']} receives",
            "#e63946",
            sum_a_at,
            season_at,
            picks_note=trade.get("side_a_picks"),
        )
    with cB:
        _render_side_summary(
            f"{trade['side_b_team']} receives",
            "#3498db",
            sum_b_at,
            season_at,
            picks_note=trade.get("side_b_picks"),
        )

    st.divider()

    # Year-after view (no auto-verdict — just the data)
    st.markdown(f"### One season later — {season_aft}")
    st.caption(
        "Same lookups against the next season's data. Players who left in free agency "
        "or got hurt show up missing here. Useful for tracking how the deal aged on day-1, "
        "but the editorial verdict above accounts for the full multi-year outcome."
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

# ══════════════════════════════════════════════════════════════════════════════
# Build-your-own trade mode
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.caption(
        "Pick the season, then drag any number of players into each side. "
        "We show each side's Barrett totals and salaries — no automated 'winner' label, "
        "because trades hinge on picks, fit, and multi-year outcomes that a single "
        "Barrett sum can't capture. You judge."
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
            st.info(
                "💡 No auto-verdict here — Barrett Score totals favour whichever side has "
                "more star talent, but real trade outcomes are decided by picks, fit, "
                "championships, and what assets become 3-5 years later. Look at the "
                "Historical trade tab for examples of how those factors actually played out."
            )
    else:
        st.info("Add at least one player to each side to see the comparison.")
