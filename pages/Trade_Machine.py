"""Trade Machine: build a 2-4 team trade, get a live legality verdict with the
exact CBA rule broken in plain English, plus the HoopsValue value grade.

All request-path data is precomputed JSON (team_rosters/team_pages, pick
ledger, CBA config) -- no model work, per the site doctrine. The rules engine
is trade_rules.py (verified 2023-CBA spec; see docs/plan_trade_machine.md).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import html
import json

import streamlit as st

from utils import normalize, render_nav, render_page_chrome, render_rail, _bootstrap_warm
from trade_rules import (CBAConfig, Trade, TradePick, TradePlayer, TradeTeam,
                         validate)

st.set_page_config(page_title="Trade Machine", page_icon="static/favicon.svg", layout="wide")
render_page_chrome()
_bootstrap_warm()
render_nav("Trade Machine")

_ROOT = Path(__file__).parent.parent
_TODAY = "2026-07-27"


@st.cache_data(show_spinner=False)
def _data():
    teams = json.loads((_ROOT / "cache" / "team_pages.json").read_text())["teams"]
    ledger = json.loads((_ROOT / "cache" / "pick_ledger.json").read_text())["teams"]
    signings = {}
    with (_ROOT / "data" / "real_signings_2026.csv").open() as f:
        for r in csv.DictReader(x for x in f if not x.lstrip().startswith("#")):
            if r.get("player") and r.get("signed_date"):
                d = r["signed_date"].strip()
                signings[normalize(r["player"])] = {
                    "date": d + "-01" if len(d) == 7 else d,
                    "type": (r.get("type") or "").strip().lower(),
                }
    return teams, ledger, signings


TEAMS, LEDGER, SIGNINGS = _data()
CFG = CBAConfig.load("2026-27")

st.title("Trade Machine")
st.caption(
    "Build a trade between two to four teams and get an instant verdict under "
    "the 2023 CBA: salary matching, apron rules, hard caps, pick rules, and "
    "trade timing, each explained in plain English. Rosters and salaries are "
    "the real 2026-27 books; projected (unsigned) players are not tradable."
)

st.markdown("""
<style>
.tmv-banner{border-radius:12px;padding:.8rem 1.1rem;font-weight:800;font-size:1.05rem;
    margin:.4rem 0 .8rem;border:1px solid;}
.tmv-legal{color:var(--accent-teal);border-color:var(--accent-teal);background:var(--tint-good);}
.tmv-illegal{color:var(--value-bad);border-color:var(--value-bad);background:var(--tint-bad);}
.tmv-vio{border-left:3px solid var(--value-bad);padding:.45rem .8rem;margin:.3rem 0;
    background:var(--panel-solid);border-radius:0 8px 8px 0;font-size:.88rem;}
.tmv-vio .cite{color:var(--fg-5);font-size:.72rem;}
.tmv-note{border-left:3px solid var(--amber);padding:.45rem .8rem;margin:.3rem 0;
    background:var(--panel-solid);border-radius:0 8px 8px 0;font-size:.85rem;color:var(--fg-2);}
.tmv-card{background:var(--panel-solid);border:1px solid var(--panel-line);border-radius:12px;
    padding:.7rem .9rem;margin:.25rem 0;}
.tmv-card .h{font-size:.8rem;font-weight:800;margin-bottom:.3rem;}
.tmv-card .row{display:flex;justify-content:space-between;font-size:.82rem;padding:.12rem 0;
    font-variant-numeric:tabular-nums;}
.tmv-card .lab{color:var(--fg-4);}
.tmv-good{color:var(--value-good);font-weight:700;}
.tmv-bad{color:var(--value-bad);font-weight:700;}
.tmv-neu{color:var(--fg-2);font-weight:600;}
</style>
""", unsafe_allow_html=True)

_abbrs = sorted(TEAMS)
_sel = st.multiselect(
    "Teams in the trade (2 to 4)", _abbrs, default=["LAL", "BKN"],
    max_selections=4, format_func=lambda a: f"{a} - {TEAMS[a]['name']}",
    key="tm_teams")

if len(_sel) < 2:
    st.info("Pick at least two teams to start building a trade.")
    st.stop()

_key_ns = "_".join(_sel)  # fresh widgets when the team set changes


def _tradable(abbr):
    """Real contracts only: no projected moves, no unsigned draftee rows."""
    return [p for p in TEAMS[abbr]["players"]
            if p.get("real", True) and p["role"] != "Draft pick"
            and p.get("salary") is not None]


def _tradable_picks(abbr):
    out = []
    for p in LEDGER.get(abbr, {}).get("controls", []):
        low = (p.get("notes") or "").lower()
        if "frozen" in low or "untradeable" in low or "untradable" in low:
            continue                      # second-apron frozen pick
        if p.get("swap_with") and p["origin"] == abbr:
            continue                      # swap-encumbered own pick: not in v1
        out.append(p)
    return out


cols = st.columns(len(_sel))
_legs = {}
for i, abbr in enumerate(_sel):
    t = TEAMS[abbr]
    with cols[i]:
        st.markdown(f"**{t['name']}**")
        st.caption(f"2026-27 payroll \\${t['payroll']:.1f}M")
        roster = _tradable(abbr)
        names = [p["n"] for p in roster]
        sal = {p["n"]: p["salary"] for p in roster}
        out_players = st.multiselect(
            "Sends", names, key=f"out_{abbr}_{_key_ns}",
            # format_func output is plain text (not markdown): no $ escaping
            format_func=lambda n, _s=sal: f"{n} (${_s[n]:.1f}M)")
        picks = _tradable_picks(abbr)
        pick_lbls = [
            f"{p['year']} {'own' if p['origin'] == abbr else p['origin']} 1st"
            + (" (protected)" if p["protection"] else "")
            for p in picks]
        out_pick_lbls = st.multiselect(
            "Sends picks", pick_lbls, key=f"pk_{abbr}_{_key_ns}")
        cash = st.number_input(
            "Sends cash (\\$M)", 0.0, float(CFG.cash_annual_limit), 0.0, 0.5,
            key=f"cash_{abbr}_{_key_ns}")
        others = [a for a in _sel if a != abbr]
        dest = st.selectbox(
            "To", others, key=f"dest_{abbr}_{_key_ns}",
            format_func=lambda a: TEAMS[a]["name"]) if len(others) > 1 else others[0]
        _legs[abbr] = {
            "players": [p for p in roster if p["n"] in out_players],
            "picks": [picks[pick_lbls.index(l)] for l in out_pick_lbls],
            "cash": cash, "dest": dest,
        }

# ── assemble the Trade for the engine ─────────────────────────────────────────
_tt = {}
for abbr in _sel:
    _tt[abbr] = TradeTeam(
        abbr=abbr, payroll=TEAMS[abbr]["payroll"],
        roster_size=TEAMS[abbr]["size"],
        owned_future_firsts=list(LEDGER.get(abbr, {}).get("covered", [])),
    )
for abbr, leg in _legs.items():
    for p in leg["players"]:
        sig = SIGNINGS.get(normalize(p["n"]))
        tp = TradePlayer(
            name=p["n"], salary=float(p["salary"]),
            signed_date=(sig or {}).get("date"),
            signed_via="bird_resign" if (sig or {}).get("type") == "resign" else "standard")
        _tt[abbr].out_players.append(tp)
        _tt[leg["dest"]].in_players.append(tp)
    for pk in leg["picks"]:
        tpk = TradePick(year=pk["year"], round=1, protection=pk["protection"],
                        swap=False)
        _tt[abbr].out_picks.append(tpk)
        _tt[leg["dest"]].in_picks.append(tpk)
    if leg["cash"] > 0:
        _tt[abbr].cash_out += leg["cash"]
        _tt[leg["dest"]].cash_in += leg["cash"]

_moved = any(t.out_players or t.out_picks or t.cash_out > 0 for t in _tt.values())
if not _moved:
    st.info("Add players, picks, or cash to a side to see the verdict.")
    st.stop()

verdict = validate(Trade(teams=list(_tt.values()), trade_date=_TODAY), CFG)

# ── verdict ───────────────────────────────────────────────────────────────────
render_rail("", "The Verdict")
if verdict.legal:
    st.markdown('<div class="tmv-banner tmv-legal">TRADE WORKS under the 2023 CBA</div>',
                unsafe_allow_html=True)
else:
    st.markdown(f'<div class="tmv-banner tmv-illegal">ILLEGAL TRADE: '
                f'{len(verdict.violations)} rule '
                f'{"violation" if len(verdict.violations) == 1 else "violations"}</div>',
                unsafe_allow_html=True)
    for v in verdict.violations:
        st.markdown(f'<div class="tmv-vio">{html.escape(v.message)}'
                    f'<div class="cite">{html.escape(v.rule_cite)}</div></div>',
                    unsafe_allow_html=True)
for x in verdict.notes:
    st.markdown(f'<div class="tmv-note">{html.escape(x.message)}</div>',
                unsafe_allow_html=True)

# ── cap impact + value grade per team ─────────────────────────────────────────
render_rail("", "Cap Impact and Value Grade")
_val = {p["n"]: p.get("value") for a in _sel for p in TEAMS[a]["players"]}
_bar = {p["n"]: p.get("barrett") for a in _sel for p in TEAMS[a]["players"]}
imp_cols = st.columns(len(_sel))
_deltas = {}
for i, abbr in enumerate(_sel):
    t = _tt[abbr]
    imp = next(x for x in verdict.impact if x.abbr == abbr)
    v_in = sum(_val.get(p.name) or 0 for p in t.in_players)
    v_out = sum(_val.get(p.name) or 0 for p in t.out_players)
    dv = v_in - v_out
    _deltas[abbr] = dv
    ds = t.salary_in - t.salary_out
    cls = "tmv-good" if dv > 1 else ("tmv-bad" if dv < -1 else "tmv-neu")
    apron_room = imp.room_to_apron1
    # &#36; not \$: these rows land inside a raw-HTML block, where markdown
    # does not strip backslash escapes (\$ renders literally).
    D = "&#36;"

    def _sgn(x):
        return f"{'+' if x >= 0 else '-'}{D}{abs(x):.1f}M"

    rows = [
        ("Payroll", f"{D}{imp.payroll_before:.1f}M &rarr; {D}{imp.payroll_after:.1f}M"),
        ("Salary delta", _sgn(ds)),
        ("Room to first apron", f"{'-' if apron_room < 0 else ''}{D}{abs(apron_room):.1f}M"),
        ("Roster", f"{imp.roster_after} players"),
        ("Market value delta", f'<span class="{cls}">{_sgn(dv)}</span>'),
    ]
    if imp.hard_cap_triggered:
        rows.append(("Hard cap", f"{D}{imp.hard_cap_triggered:.1f}M"))
    body = "".join(f'<div class="row"><span class="lab">{a}</span><span>{b}</span></div>'
                   for a, b in rows)
    with imp_cols[i]:
        st.markdown(f'<div class="tmv-card"><div class="h">{TEAMS[abbr]["name"]}</div>'
                    f"{body}</div>", unsafe_allow_html=True)

if verdict.legal and any(abs(d) > 1 for d in _deltas.values()):
    w = max(_deltas, key=_deltas.get)
    st.caption(f"By HoopsValue market value, {TEAMS[w]['name']} win this trade, "
               f"adding \\${_deltas[w]:.1f}M in surplus value.")

st.caption(
    "Legality uses the verified 2023 CBA rules: the salary matching formula, "
    "first and second apron limits, hard cap triggers, the Stepien rule on "
    "future firsts, and trade timing. Ambiguous edge cases resolve "
    "conservatively. Player values are the model's 2026-27 market values."
)
