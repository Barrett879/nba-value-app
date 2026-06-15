"""Free Agency Simulation — one projected landing spot for every 2026 free agent.

Runs the player-side suitor engine (the same one behind Likely Suitors) over the
whole free-agent pool and gives each player a single best-guess destination. Two
modes, toggled here:
  - Most likely : the validated top-1-optimal call (re-signs dominate, ~51% top-1).
  - Realistic   : softens the re-sign pull for unrestricted FAs so a believable
                  share change teams; a light market-clear keeps any one team from
                  hoarding signings. Lower per-pick accuracy by design.

A league-wide market-clearing engine was prototyped and backtested first
(scripts/backtest_fa_sim.py); it lost ~7 points of accuracy versus the
independent ranking, so the independent best-guess is what ships here.

Data is pre-built into cache/fa_sim_v1.json by scripts/build_fa_board.py.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")
import html as _h

import streamlit as st

from utils import render_nav, render_page_chrome, render_footer, _bootstrap_warm, stat_cards

st.set_page_config(page_title="Free Agency Simulation", page_icon="static/favicon.svg", layout="wide")
render_page_chrome()
_bootstrap_warm()
render_nav("Free Agency Simulation")

st.title("Free Agency Simulation")

_SIM = Path(__file__).parent.parent / "cache" / "fa_sim_v1.json"
if not _SIM.exists():
    st.warning("The simulation hasn't been generated yet. Run "
               "`python scripts/build_fa_board.py` to build it.", icon="🏗️")
    render_footer(); st.stop()

DATA = json.loads(_SIM.read_text())
PLAYERS = DATA.get("players", [])
ACC = DATA.get("accuracy", {})
CONTRACT = DATA.get("contract_season", "2026-27")

_CONF_COLOR = {"High": "var(--value-good)", "Medium": "var(--orange)", "Low": "var(--fg-4)"}
_STATUS_COLOR = {"UFA": "var(--blue)", "RFA": "var(--accent-teal)",
                 "Player Option": "var(--purple)", "Team Option": "var(--purple)"}

st.caption(
    f"Every projected free agent for the {CONTRACT} offseason, with one best-guess "
    "destination apiece. Same engine as a player's Likely Suitors, run across the "
    "whole pool. Exact landing spots are genuinely hard to call (real signings turn "
    "on sign-and-trades, agent ties, and fit a model can't see), so treat this as an "
    "educated projection, strongest on re-signings."
)

# ── Mode toggle ──────────────────────────────────────────────────────────────
_mode_label = st.radio(
    "Projection mode",
    ["Realistic", "Most likely"],
    horizontal=True, key="fa_sim_mode",
    help=("Realistic: a believable mix of stays and moves (lower per-pick accuracy). "
          "Most likely: the accuracy-optimal call, which re-signs most free agents."))
MODE = "realistic" if _mode_label == "Realistic" else "likely"


def pick(p):
    return p.get(MODE, {})


# ── Summary cards (mode-aware) ───────────────────────────────────────────────
n = len(PLAYERS)
moves = sum(1 for p in PLAYERS if not pick(p).get("is_resign"))
if MODE == "likely":
    a = ACC.get("likely", {})
    acc_v = f"~{a.get('top1_recent', 51)}%"
    acc_sub = "exact-team, validated on 1,821 signings"
else:
    acc_v = "lower"
    acc_sub = "trades accuracy for a believable spread"
stat_cards([
    ("Free Agents", str(n), "var(--accent-teal)", "projected for the offseason"),
    ("Projected to Move", str(moves), "var(--orange)", f"{n - moves} re-sign / stay"),
    ("Re-sign Rate", f"{(n - moves) / n * 100:.0f}%" if n else "—", "var(--value-good)",
     "league norm is ~half"),
    ("Top-1 Accuracy", acc_v, "var(--blue)", acc_sub),
])

# ── Controls ─────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    view = st.radio("View", ["By player", "By team"], horizontal=True, key="fa_sim_view")
with c2:
    conf_f = st.multiselect("Confidence", ["High", "Medium", "Low"], default=[], key="fa_sim_conf")
with c3:
    query = st.text_input("Search player", "", key="fa_sim_q").strip().lower()


def _passes(p):
    if conf_f and pick(p).get("confidence") not in conf_f:
        return False
    if query and query not in p["player"].lower():
        return False
    return True


def _chip(text, color):
    return (f"<span style='display:inline-block;font-size:0.68rem;font-weight:700;"
            f"padding:0.08rem 0.45rem;border-radius:999px;background:color-mix(in srgb,{color} 16%,transparent);"
            f"color:{color};white-space:nowrap'>{_h.escape(text)}</span>")


def _dest_cell(p):
    pk = pick(p)
    if pk.get("is_resign"):
        return (f"<b>{_h.escape(p['incumbent'])}</b> "
                f"<span style='color:var(--fg-5);font-size:0.74rem'>re-sign</span>")
    return (f"<span style='color:var(--fg-5)'>{_h.escape(p['incumbent'])}</span> "
            f"<span style='color:var(--fg-5)'>&rarr;</span> "
            f"<b style='color:var(--accent-teal)'>{_h.escape(pk.get('predicted',''))}</b>")


_TH = ("text-align:left;font-size:0.7rem;letter-spacing:0.03em;text-transform:uppercase;"
       "color:var(--fg-5);font-weight:700;padding:0.5rem 0.6rem;border-bottom:1px solid var(--panel-line)")
_TD = "padding:0.55rem 0.6rem;border-bottom:1px solid var(--hairline-soft);font-size:0.9rem;vertical-align:middle"


def _player_table(rows):
    h = [f"<div style='overflow-x:auto'><table style='width:100%;border-collapse:collapse'>",
         "<thead><tr>",
         f"<th style='{_TH}'>Player</th><th style='{_TH}'>Pos</th>",
         f"<th style='{_TH};text-align:right'>Proj $</th><th style='{_TH}'>Status</th>",
         f"<th style='{_TH}'>Projected Destination</th><th style='{_TH}'>Confidence</th>",
         "</tr></thead><tbody>"]
    for p in rows:
        pk = pick(p)
        conf = pk.get("confidence", "Low")
        st_c = _STATUS_COLOR.get(p["status"], "var(--fg-4)")
        h.append(
            "<tr>"
            f"<td style='{_TD}'><b>{_h.escape(p['player'])}</b>"
            f"<div style='font-size:0.72rem;color:var(--fg-5)'>age {p['age']} &middot; {_h.escape(p['incumbent_name'])}</div></td>"
            f"<td style='{_TD};color:var(--fg-4)'>{_h.escape(p['pos'])}</td>"
            f"<td style='{_TD};text-align:right;font-weight:700'>${p['value_M']:.0f}M</td>"
            f"<td style='{_TD}'>{_chip(p['status'], st_c)}</td>"
            f"<td style='{_TD}'>{_dest_cell(p)}</td>"
            f"<td style='{_TD}'>{_chip(conf, _CONF_COLOR.get(conf, 'var(--fg-4)'))}</td>"
            "</tr>")
    h.append("</tbody></table></div>")
    st.markdown("".join(h), unsafe_allow_html=True)


def _team_view(rows):
    # Group by projected destination; show who each team keeps vs adds.
    by = {}
    for p in rows:
        by.setdefault(pick(p)["predicted"], []).append(p)
    for tm in sorted(by, key=lambda t: -len(by[t])):
        grp = sorted(by[tm], key=lambda p: -p["value_M"])
        name = pick(grp[0])["predicted_name"] if grp else tm
        adds = [p for p in grp if not pick(p)["is_resign"]]
        keeps = [p for p in grp if pick(p)["is_resign"]]
        with st.expander(f"{name}  —  {len(adds)} add, {len(keeps)} re-sign", expanded=False):
            for p in grp:
                pk = pick(p)
                tag = (_chip("re-sign", "var(--value-good)") if pk["is_resign"]
                       else _chip(f"from {p['incumbent']}", "var(--accent-teal)"))
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"padding:0.3rem 0;border-bottom:1px solid var(--hairline-soft)'>"
                    f"<span><b>{_h.escape(p['player'])}</b> "
                    f"<span style='color:var(--fg-5);font-size:0.78rem'>{_h.escape(p['pos'])}</span></span>"
                    f"<span>${p['value_M']:.0f}M &nbsp; {tag} &nbsp; "
                    f"{_chip(pk['confidence'], _CONF_COLOR.get(pk['confidence'], 'var(--fg-4)'))}</span></div>",
                    unsafe_allow_html=True)


rows = [p for p in PLAYERS if _passes(p)]
st.markdown(f"<div style='color:var(--fg-5);font-size:0.8rem;margin:0.3rem 0 0.6rem'>"
            f"Showing {len(rows)} of {n} free agents.</div>", unsafe_allow_html=True)

if view == "By player":
    _player_table(rows)
else:
    _team_view(rows)

with st.expander("How this works + accuracy"):
    st.markdown(
        "- **Engine.** For every free agent, the model scores all 30 teams on offer "
        "(projected contract, scaled by fit and capped by the team's cap tools), team "
        "need at his position, and how a club of that timeline values his age/role, then "
        "blends in a destination model trained on 1,810 historical signings. The top team "
        "is his projected landing spot.\n"
        "- **Most likely** is the accuracy-optimal call: re-signing is the single strongest "
        "signal (about half of all FAs re-sign), so this mode keeps most players home. "
        "Measured **~51% top-1 / ~59% top-5** on recent seasons; exact landing spots have a "
        "hard noise ceiling.\n"
        "- **Realistic** softens the re-sign pull for unrestricted free agents (restricted "
        "FAs and option-decliners stay, as they do in reality) and runs a light cap check so "
        "no single team signs everyone. It reads more like a real offseason but its individual "
        "team-change calls are educated guesses, not high-confidence.\n"
        "- **Confidence.** High = a re-signing. Medium = a clear starter-level fit on real "
        "money. Low = a depth/role move or an ambiguous market — expect these to miss often.\n"
        "- A full **market-clearing simulation** was prototyped and backtested first; it lost "
        "~7 points of accuracy (real signings clear through sign-and-trades and cap holds a "
        "clean model can't capture), so the independent best-guess is what runs here."
    )

render_footer()
