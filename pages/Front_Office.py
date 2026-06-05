import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
from utils import (
    render_nav, render_page_chrome, _bootstrap_warm,
    stat_cards, html_table,
)

st.set_page_config(page_title="Front Office", layout="wide")

render_page_chrome()
_bootstrap_warm()
render_nav("Front Office")

st.title("Front Office")
st.caption(
    "The Contract Predictor, flipped to the team's side of the table. Pick a club and "
    "see the free agents it should chase this offseason — who to re-sign, who to pursue, "
    "the contract it would realistically offer, and why. Same engine as Likely Suitors, "
    "run from the front office's chair."
)

# ── Load the pre-computed board (built offline by scripts/build_fa_board.py) ─────
_BOARD = Path(__file__).parent.parent / "cache" / "fa_board_v1.json"
if not _BOARD.exists():
    st.warning(
        "The Front Office board hasn't been generated yet. Run "
        "`python scripts/build_fa_board.py` to build it.", icon="🏗️")
    st.stop()
DATA = json.loads(_BOARD.read_text())
TEAMS = DATA["teams"]

# ── Team picker (full names → abbreviation) ─────────────────────────────────────
_name_to_abbr = {b["name"]: ab for ab, b in TEAMS.items()}
_names = sorted(_name_to_abbr)
_default = _name_to_abbr.get("Los Angeles Lakers", _names[0])
_default_name = next(n for n, a in _name_to_abbr.items() if a == _default)

pick = st.selectbox("Team", _names, index=_names.index(_default_name))
B = TEAMS[_name_to_abbr[pick]]

# ── Header: cap tools, timeline, needs ──────────────────────────────────────────
_TL_COLOR = {
    "title":     "var(--value-good)",
    "playoff":   "var(--blue)",
    "retooling": "var(--orange)",
    "rebuild":   "var(--fg-3)",
}
needs = ", ".join(B["needs"]) if B["needs"] else "Roster set"
thin = ", ".join(B.get("thin", []))
stat_cards([
    ("Projected Cap Room", f"${B['cap_room_M']}M", "var(--accent-teal)", "if its own FAs are renounced"),
    ("Mid-Level Exception", f"${B['exception_M']}M", "var(--blue)", "the over-the-cap tool"),
    ("Timeline", B["timeline"].title(), _TL_COLOR.get(B["timeline"], "var(--fg-2)"),
     "how the model weighs fit"),
    ("Positions of Need", needs, "var(--orange)", (f"thin at {thin}" if thin else "no starter on the books")),
])
st.caption(
    "**Cap room is the theoretical max** — the space a team would have only if it renounced its own "
    "free agents. In practice most contenders re-sign their own (Bird rights, which don't use cap room) "
    "and shop with the mid-level exception. Offers below are bounded by whichever tool actually applies."
)

_STATUS_COLOR = {
    "UFA": "var(--fg-3)", "RFA": "var(--value-good)",
    "Player Option": "var(--blue)", "Team Option": "var(--orange)",
}


def _sty_status(v, _row):
    return f"color:{_STATUS_COLOR.get(str(v), 'var(--fg-2)')};font-weight:600"


def _sty_offer(v, _row):
    return "color:var(--accent-teal);font-weight:700"


st.divider()

# ── Pursue (external targets) ───────────────────────────────────────────────────
st.subheader(f"Who {B['name'].split()[-1]} should pursue")
st.caption(
    "External free agents ranked by how keenly a team of this timeline would chase them, "
    "gated for affordability — no banking on a star taking a massive paycut (an aging vet "
    "on minimum money to chase a ring is the one exception)."
)
if B["pursue"]:
    pur = pd.DataFrame(B["pursue"])
    pur = pur[["name", "pos", "from", "status", "value_M", "offer_M", "tool", "why"]]
    pur.columns = ["Target", "Pos", "From", "Status", "Market Value", "Their Offer", "Tool", "Fit"]
    pur.insert(0, "#", range(1, len(pur) + 1))
    html_table(
        pur,
        formatters={"Market Value": lambda v: f"${v:.0f}M", "Their Offer": lambda v: f"${v:.0f}M"},
        styles={"Status": _sty_status, "Their Offer": _sty_offer},
        aligns={"#": "right", "Market Value": "right", "Their Offer": "right"},
        numeric={"#", "Market Value", "Their Offer"},
        helps={
            "Market Value": "What the player projects to earn on the open market (Contract Predictor).",
            "Their Offer": "What this team would realistically offer, capped by the tool it has available.",
            "Tool": "Cap room, the mid-level exception, or a veteran-minimum slot.",
            "Fit": "Why he fits — fills a need, upgrades a starter, or rotation/minimum depth.",
        },
        height=min(720, len(pur) * 38 + 46),
    )
else:
    st.info("No realistic external targets — this team is capped out with a full rotation.")

st.divider()

# ── Re-sign your own ────────────────────────────────────────────────────────────
st.subheader("Re-sign their own free agents")
st.caption("Players already on the roster who can be kept via Bird rights — no cap room required.")
if B["resign"]:
    res = pd.DataFrame(B["resign"])
    res = res[["name", "pos", "status", "value_M", "offer_M"]]
    res.columns = ["Player", "Pos", "Status", "Market Value", "Re-sign Cost"]
    res.insert(0, "#", range(1, len(res) + 1))
    html_table(
        res,
        formatters={"Market Value": lambda v: f"${v:.0f}M", "Re-sign Cost": lambda v: f"${v:.0f}M"},
        styles={"Status": _sty_status, "Re-sign Cost": _sty_offer},
        aligns={"#": "right", "Market Value": "right", "Re-sign Cost": "right"},
        numeric={"#", "Market Value", "Re-sign Cost"},
        helps={"Re-sign Cost": "What it would cost to keep him — his projected market value via Bird rights."},
        height=min(520, len(res) * 38 + 46),
    )
else:
    st.info("No notable free agents of their own to re-sign.")

# ── Method ──────────────────────────────────────────────────────────────────────
with st.expander("How these boards are built"):
    st.markdown(
        f"""
- **Candidate pool** — every free agent this offseason (UFA, RFA, player/team option) with enough
  minutes to rank: **{DATA['n_free_agents']}** players for **{DATA['season']}**.
- **Each player's market value** comes from the Contract Predictor model, so the numbers match the
  player-facing page exactly.
- **Roster fit** — where he'd slot into this team's depth chart at his position (start, upgrade, or depth),
  using the same curated positions as the rest of the site.
- **Affordability gate** — a team can't realistically land a player it would massively underpay; he'll
  get closer to his value elsewhere. The lone exception is an aging vet taking minimum/exception money
  to chase a ring.
- **Timeline fit** — a contender chases win-now production and ring-chasing vets; a rebuilder chases youth
  and passes on aging stars. Derived from ~900 real free-agent signings (2013–2025).
- This is the inverse of the **Likely Suitors** list on each player's Contract Predictor page — same engine,
  read from the team's side instead of the player's.
        """
    )
