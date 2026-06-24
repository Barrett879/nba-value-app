import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
from utils import (
    COMMON_CSS, render_page_chrome, render_nav, _bootstrap_warm,
    html_table, stat_cards,
)

st.set_page_config(page_title="Accuracy Tracker · HoopsValue", page_icon="static/favicon.svg", layout="wide")
render_page_chrome()
_bootstrap_warm()
render_nav("Accuracy")
st.markdown(COMMON_CSS, unsafe_allow_html=True)

st.title("Accuracy Tracker")
st.caption(
    "How the model's projections are holding up against the **actual** 2026 free-agency "
    "signings. The projection is the model's pure guess — real deals are logged separately "
    "and never fed back in, so this is an honest scorecard."
)

CACHE = Path(__file__).parent.parent / "cache" / "accuracy_tracker_v1.json"
try:
    data = json.loads(CACHE.read_text())
except Exception:
    st.warning("Accuracy data isn't built yet — run `scripts/build_accuracy_tracker.py`.")
    st.stop()

signings = [s for s in data.get("signings", []) if s.get("model_M") is not None]
if not signings:
    st.info("No signings logged yet. Add rows to `data/real_signings_2026.csv` and rebuild.")
    st.stop()

sc = data.get("scorecard") or {}
if sc:
    good = "var(--value-good)"
    stat_cards([
        ("Signings tracked", sc["n"], "var(--accent-teal)"),
        ("Within $4M", f"{sc['within_4M']}%", good if sc["within_4M"] >= 60 else "var(--amber)"),
        ("Within 5% of cap", f"{sc['within_5cap']}%", good if sc["within_5cap"] >= 80 else "var(--amber)"),
        ("Median miss", f"${sc['median_err_M']}M", "var(--accent-teal)"),
        ("Bias", f"${sc['bias_M']:+}M", "var(--accent-teal)", "+ projects high"),
    ])

df = pd.DataFrame([{
    "Player":     s["player"],
    "Team":       s.get("team") or "—",
    "Deal":       s["deal"],
    "Actual yr1": s["actual_M"],
    "Model":      s["model_M"],
    "Range":      f"${s['low_M']:.0f}–{s['high_M']:.0f}M",
    "Miss":       s["delta_M"],
    "Within $4M": "Hit" if s.get("in4") else "Miss",
} for s in signings])

html_table(
    df,
    formatters={
        "Actual yr1": lambda v: f"${v:.1f}M",
        "Model":      lambda v: f"${v:.1f}M",
        "Miss":       lambda v: f"{v:+.1f}M",
    },
    aligns={"Actual yr1": "right", "Model": "right", "Range": "right",
            "Miss": "right", "Within $4M": "center"},
    numeric=["Actual yr1", "Model", "Miss"],
    styles={
        "Miss":       lambda v, r: "color:var(--value-good)" if abs(v) <= 4 else "color:var(--value-bad)",
        "Within $4M": lambda v, r: "color:var(--value-good); font-weight:700" if v == "Hit" else "color:var(--value-bad); font-weight:700",
    },
    height=460,
)

st.caption(
    "“Within \\$4M” = the model's projected first-year salary landed within \\$4M of the real "
    "deal. The projection is exactly what the Contract Predictor shows; actual deals come "
    "from `data/real_signings_2026.csv`. Rebuilt by `scripts/build_accuracy_tracker.py`."
)
