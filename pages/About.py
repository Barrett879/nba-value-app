import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import streamlit as st
from utils import (
    render_nav, render_page_chrome, render_rail, render_footer, _bootstrap_warm,
)


def _accuracy_phrase() -> str:
    """Live "N% of M deals within $4M" phrase from the accuracy tracker, so the
    About copy never drifts from the homepage figure."""
    try:
        sc = (json.loads((Path(__file__).parent.parent / "cache"
                          / "accuracy_tracker_v1.json").read_text()).get("scorecard") or {})
        if sc.get("n"):
            return f"about {sc['within_4M']:.0f}% of the {sc['n']} tracked deals so far come in"
    except Exception:
        pass
    return "most tracked deals so far come in"

st.set_page_config(page_title="About", page_icon="static/favicon.svg", layout="wide")

render_page_chrome()
_bootstrap_warm()
render_nav("")  # footer-only page, no active top-nav tab

st.title("About HoopsValue")
st.markdown(
    "HoopsValue is an independent NBA analytics project measuring what every "
    "player is really worth: their on-court value against what they are paid, "
    "for every season since 1973. It is not affiliated with, endorsed by, or "
    "sponsored by the National Basketball Association or any team."
)

# ── The Barrett Score ──────────────────────────────────────────────────────────
render_rail("The metric", "The Barrett Score")
st.markdown(
    "The Barrett Score is the site's player-value metric. It combines production, "
    "efficiency, and availability, scoring, playmaking, rebounding, and defense, "
    "weighed by how efficiently a player produces and how many games he actually "
    "plays, into a single number. Every score is then compared against real NBA "
    "contracts to show who is underpaid, overpaid, or paid roughly in line with "
    "their on-court value."
)

# ── Contract predictions and accuracy ──────────────────────────────────────────
st.markdown('<a id="accuracy"></a>', unsafe_allow_html=True)
render_rail("The model", "Contract predictions and accuracy")
st.markdown(
    "Contract predictions come from a gradient-boosted model trained on 1,900+ "
    "real NBA contracts from the modern salary-cap era (2012 on), then blended "
    "toward comparable players so each estimate stays anchored to the market. We "
    "test it the fair way, on deals it never saw during training, using "
    "expanding-window temporal cross-validation, and we score it against real "
    f"2026 signings as they land. So far {_accuracy_phrase()} within "
    "\\$4M of the actual contract, a figure that updates as the offseason tracker "
    "grows."
)

# ── Data sources ───────────────────────────────────────────────────────────────
render_rail("The inputs", "Data sources")
st.markdown(
    "Player stats come from NBA.com through the nba_api, with historical numbers "
    "from Basketball-Reference and salary data from ESPN. Offseason contracts, "
    "options, and signings are hand-verified against Spotrac and beat reporting. "
    "Where the data is still moving during the offseason, the pages carry a "
    "freshness caption noting when it was last checked."
)

# ── Contact ────────────────────────────────────────────────────────────────────
render_rail("Get in touch", "Contact")
st.markdown(
    "Questions, corrections, and feedback are welcome. Reach us at "
    "[contact@hoopsvalue.com](mailto:contact@hoopsvalue.com) or on X at "
    "[@HoopsValue](https://x.com/HoopsValue)."
)

render_footer()
