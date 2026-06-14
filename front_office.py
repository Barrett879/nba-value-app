"""Front Office, the team-side view of the Contract Predictor.

Rendered as a mode of pages/Contract_Predictor.py (not a standalone page). The
board for all 30 teams is pre-computed by scripts/build_fa_board.py into
cache/fa_board_v1.json so this loads instantly, no live prediction.

The inverse of Likely Suitors: pick a team and see the free agents it should
chase this offseason (re-sign its own + pursue external), each with the contract
it would realistically offer and why.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from utils import stat_cards, html_table

_BOARD = Path(__file__).parent / "cache" / "fa_board_v1.json"

_TL_COLOR = {
    "title":     "var(--value-good)",
    "playoff":   "var(--blue)",
    "retooling": "var(--orange)",
    "rebuild":   "var(--fg-3)",
}
_STATUS_COLOR = {
    "UFA": "var(--fg-3)", "RFA": "var(--value-good)",
    "Player Option": "var(--blue)", "Team Option": "var(--orange)",
}


def _sty_status(v, _row):
    return f"color:{_STATUS_COLOR.get(str(v), 'var(--fg-2)')};font-weight:600"


def _sty_offer(v, _row):
    return "color:var(--accent-teal);font-weight:700"


_GRADE_COLOR = {"A+": "var(--value-good)", "A": "var(--accent-teal)",
                "A-": "var(--blue)", "B+": "var(--orange)", "B": "var(--fg-3)"}

_FIT_CSS = """
<style>
.hv-fits { display:flex; gap:0.7rem; flex-wrap:wrap; margin:0.2rem 0 1rem; padding-top:1rem; }
.hv-fit { flex:1 1 0; min-width:210px; background:var(--panel-solid);
    border:1px solid var(--panel-line); border-top:3px solid var(--c);
    border-radius:10px; padding:0.95rem 1rem 0.9rem; position:relative; box-shadow:var(--shadow-card); }
.hv-fit-grade { position:absolute; top:0.7rem; right:0.95rem; font-size:1.55rem; font-weight:800;
    line-height:1; color:var(--c); }
.hv-fit-name { font-size:1.1rem; font-weight:800; line-height:1.15; padding-right:2.6rem; }
.hv-fit-sub { font-size:0.74rem; font-weight:600; color:var(--fg-4); margin-top:0.15rem;
    text-transform:uppercase; letter-spacing:0.03em; }
.hv-fit-money { font-size:0.86rem; color:var(--fg-3); margin:0.5rem 0 0.45rem; }
.hv-fit-money b { color:var(--accent-teal); }
.hv-fit-why { font-size:0.9rem; color:var(--fg-2); line-height:1.36; }
</style>
"""


def _fit_card(f):
    import html as _h
    c = _GRADE_COLOR.get(f["grade"], "var(--accent-teal)")
    return (f"<div class='hv-fit' style='--c:{c}'>"
            f"<div class='hv-fit-grade'>{_h.escape(str(f['grade']))}</div>"
            f"<div class='hv-fit-name'>{_h.escape(str(f['name']))}</div>"
            f"<div class='hv-fit-sub'>{_h.escape(str(f['pos']))} &middot; from {_h.escape(str(f['from']))} "
            f"&middot; {_h.escape(str(f['status']))}</div>"
            f"<div class='hv-fit-money'>${f['value_M']:.0f}M market &rarr; <b>${f['offer_M']:.0f}M offer</b></div>"
            f"<div class='hv-fit-why'>{_h.escape(str(f['why']))}</div></div>")


def _cap_bar_html(committed, cap, tax, apron2, after):
    """A horizontal cap bar: current committed payroll (teal) plus the realistic
    plan's spend (hatched), against the salary cap / luxury tax / second-apron
    lines. The white marker is the projected payroll after the plan."""
    vals = [committed, cap, tax, apron2, after]
    lo, hi = min(vals) - 10, max(vals) + 10
    span = max(hi - lo, 1.0)
    p = lambda v: max(0.0, min(100.0, (v - lo) / span * 100))
    cw, aw = p(committed), p(after)
    seg_now = (f"<div style='position:absolute;left:0;width:{cw:.1f}%;top:0;bottom:0;border-radius:7px 0 0 7px;"
               f"background:linear-gradient(90deg,rgba(22,212,193,.32),rgba(22,212,193,.6));'></div>")
    seg_plan = (f"<div style='position:absolute;left:{cw:.1f}%;width:{max(aw - cw, 0.4):.1f}%;top:0;bottom:0;"
                f"background:repeating-linear-gradient(45deg,rgba(241,196,15,.45),rgba(241,196,15,.45) 5px,"
                f"transparent 5px,transparent 11px);'></div>") if after > committed + 0.5 else ""
    def tick(v, label, color):
        x = p(v)
        return (f"<div style='position:absolute;left:{x:.1f}%;top:-4px;bottom:18px;width:2px;background:{color};'></div>"
                f"<div style='position:absolute;left:{x:.1f}%;bottom:-1px;transform:translateX(-50%);font-size:0.6rem;"
                f"line-height:1.1;color:var(--fg-5);white-space:nowrap;text-align:center;'>{label}"
                f"<br><b style='color:var(--fg-3)'>${v:.0f}M</b></div>")
    ticks = (tick(cap, "Cap", "var(--hairline)") + tick(tax, "Tax", "var(--orange)")
             + tick(apron2, "2nd apron", "var(--value-bad)"))
    amark = (f"<div style='position:absolute;left:{aw:.1f}%;top:-7px;width:3px;height:28px;border-radius:2px;"
             f"background:var(--fg-1);transform:translateX(-50%);box-shadow:0 0 6px rgba(0,0,0,.35);'></div>")
    legend = ("<div style='display:flex;gap:1.1rem;font-size:0.68rem;color:var(--fg-4);margin-bottom:0.5rem;'>"
              "<span><span style='color:var(--accent-teal)'>&#9632;</span> committed payroll</span>"
              "<span><span style='color:var(--gold)'>&#9632;</span> this offseason (re-signs + signings)</span>"
              "<span><span style='color:var(--fg-1)'>&#9612;</span> after the offseason</span></div>")
    return (f"<div style='margin:0.2rem 0 2.8rem;'>{legend}"
            f"<div style='position:relative;height:14px;border-radius:7px;background:var(--hairline-soft);'>"
            f"{seg_now}{seg_plan}{ticks}{amark}</div></div>")


_PLAN_TOOL_COLOR = {"Re-sign": "var(--gold)", "Cap room": "var(--accent-teal)",
                    "Room exception": "var(--blue)", "Mid-level": "var(--blue)",
                    "Depth": "var(--fg-3)", "Minimum": "var(--fg-5)"}
_PLAN_CSS = """
<style>
.hv-plan { display:flex; gap:0.75rem 0.55rem; flex-wrap:wrap; margin:1.05rem 0 1.85rem; }
.hv-plan-chip { background:var(--panel-solid); border:1px solid var(--panel-line);
    border-left:3px solid var(--c); border-radius:9px; padding:0.7rem 0.9rem; min-width:148px; }
.hv-plan-name { font-weight:800; font-size:0.96rem; line-height:1.15; }
.hv-plan-sub { font-size:0.7rem; color:var(--fg-4); margin-top:0.15rem; }
.hv-plan-tool { font-size:0.82rem; font-weight:700; color:var(--c); margin-top:0.45rem; }
</style>
"""


def _plan_chip(m):
    import html as _h
    c = _PLAN_TOOL_COLOR.get(m["tool"], "var(--fg-4)")
    _src = f" &middot; from {_h.escape(str(m['from']))}" if m.get("from") else ""
    return (f"<div class='hv-plan-chip' style='--c:{c}'>"
            f"<div class='hv-plan-name'>{_h.escape(str(m['name']))}</div>"
            f"<div class='hv-plan-sub'>{_h.escape(str(m['pos']))}{_src}</div>"
            f"<div class='hv-plan-tool'>{_h.escape(str(m['tool']))} &middot; ${m['cost_M']:.0f}M</div></div>")


def render_front_office():
    """Render the team-side board into the current page (no page chrome/nav)."""
    if not _BOARD.exists():
        st.warning(
            "The Front Office board hasn't been generated yet. Run "
            "`python scripts/build_fa_board.py` to build it.", icon="🏗️")
        return
    DATA = json.loads(_BOARD.read_text())
    TEAMS = DATA["teams"]

    st.caption(
        "The contract predictor from the team's side of the table. Pick a club and see the free "
        "agents it should chase this offseason, who to re-sign, who to pursue, the contract it "
        "would realistically offer, and why. Same engine as a player's Likely Suitors, run from "
        "the front office's chair."
    )

    # ── Team picker (full names → abbreviation), reflected in the URL ────────────
    # Starts with NO team selected (a prompt, not an auto-pick). ?team=LAL deep-
    # links straight to a club, and once a team is chosen the selection is
    # mirrored back into the URL so a board is shareable/bookmarkable. Options
    # stay in STABLE sorted order (never reordered) so the selectbox keeps its
    # element identity; a ?team= deep-link seeds the widget once, then it owns it.
    _name_to_abbr = {b["name"]: ab for ab, b in TEAMS.items()}
    _abbr_to_name = {ab: b["name"] for ab, b in TEAMS.items()}
    _names = sorted(_name_to_abbr)
    if "fo_team" not in st.session_state:
        _url_name = _abbr_to_name.get((st.query_params.get("team") or "").upper())
        if _url_name:
            st.session_state["fo_team"] = _url_name
    pick = st.selectbox("Team", _names, index=None, placeholder="Select a team…", key="fo_team")
    if not pick:
        st.info("Pick a team to see its offseason board — who to re-sign, who to "
                "pursue, and the contract it would realistically offer.")
        return
    _ab = _name_to_abbr.get(pick)                       # mirror the pick into the URL
    if _ab and st.query_params.get("team") != _ab:
        st.query_params["team"] = _ab
    B = TEAMS[_name_to_abbr[pick]]
    _short = B["name"].split()[-1]

    # ── Header: cap tools, timeline, needs ──────────────────────────────────────
    needs = ", ".join(B["needs"]) if B["needs"] else "Roster set"
    thin = ", ".join(B.get("thin", []))
    stat_cards([
        ("Projected Cap Room", f"${B['cap_room_M']}M", "var(--accent-teal)", "if its own FAs are renounced"),
        ("Mid-Level Exception", f"${B['exception_M']}M", "var(--blue)", "the over-the-cap tool"),
        ("Timeline", B["timeline"].title(), _TL_COLOR.get(B["timeline"], "var(--fg-2)"),
         "how the model weighs fit"),
        ("Positions of Need", needs, "var(--orange)",
         (f"thin at {thin}" if thin else "no starter on the books")),
    ])
    st.caption(
        "**Cap room is the theoretical max**, the space a team would have only if it renounced its own "
        "free agents. In practice most contenders re-sign their own (Bird rights, which don't use cap room) "
        "and shop with the mid-level exception. Offers below are bounded by whichever tool actually applies."
    )

    # ── Cap bar: committed payroll + the plan, against cap / tax / second apron ──
    _rp = B.get("resign_plan") or {}
    _committed = B.get("committed_M") or _rp.get("committed_M") or 0
    _tax = B.get("tax_M") or _rp.get("tax_M") or 0
    _apron2 = B.get("apron2_M") or _rp.get("apron2_M") or 0
    _plan_cost = sum(m["cost_M"] for m in B.get("plan", []))       # re-signs + external, one plan
    _after = _committed + _plan_cost
    if _committed and _apron2:
        st.markdown(_cap_bar_html(_committed, DATA["cap_M"], _tax, _apron2, _after),
                    unsafe_allow_html=True)

    # ── Current roster, collapsed, ordered by Barrett Score ─────────────────────
    _roster = B.get("roster", [])
    if _roster:
        with st.expander(f"{_short} guaranteed roster ({len(_roster)} under contract, by Barrett Score)"):
            _rdf = pd.DataFrame(_roster)[["name", "pos", "barrett", "salary_M"]]
            _rdf.columns = ["Player", "Pos", "Barrett", "Salary"]
            _rdf.insert(0, "#", range(1, len(_rdf) + 1))
            html_table(
                _rdf,
                formatters={"Barrett": lambda v: f"{v:.1f}", "Salary": lambda v: f"${v:.1f}M"},
                aligns={"#": "right", "Barrett": "right", "Salary": "right"},
                numeric={"#", "Barrett", "Salary"},
                height=min(740, len(_rdf) * 35 + 44),
            )

    st.divider()

    # ── Best fits (the featured suggestion) ─────────────────────────────────────
    fits = B.get("best_fits", [])
    if fits:
        st.markdown(f"#### Best fits for the {_short}")
        st.caption(
            "Our top matches, roster need, the team's timeline, and value fused into one fit grade. "
            "The standouts on the board, not just the priciest names a team could sign.")
        st.markdown(_FIT_CSS, unsafe_allow_html=True)
        st.markdown(f"<div class='hv-fits'>{''.join(_fit_card(f) for f in fits)}</div>",
                    unsafe_allow_html=True)
        st.divider()

    # ── The realistic offseason: re-signs + external adds, then the full board ───
    st.subheader(f"{_short} offseason plan")
    _plan_moves = B.get("plan", [])
    _resigns = [m for m in _plan_moves if m.get("kind") == "resign"]
    _adds = [m for m in _plan_moves if m.get("kind") != "resign"]
    if _plan_moves:
        _parts = []
        if _resigns:
            _parts.append(f"**re-sign {len(_resigns)}** of their own worth keeping")
        if _adds:
            if any(m["tool"] == "Cap room" for m in _adds):
                _tool = "cap room plus minimums"
            elif any(m["tool"] in ("Mid-level", "Room exception") for m in _adds):
                _tool = "the mid-level plus minimums"
            else:
                _tool = "veteran minimums"
            _parts.append(f"**add {len(_adds)}** using {_tool}")
        st.markdown(
            f"What {_short} would realistically do this offseason: " + " and ".join(_parts) +
            ". Marginal role players walk — nobody is kept just to fill a spot.")
        st.markdown(_PLAN_CSS + f"<div class='hv-plan'>{''.join(_plan_chip(m) for m in _plan_moves)}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(
            f"Already past the second apron (the practical ceiling) on guaranteed money, so {_short} "
            "are capped out: their offseason is veteran minimums and exercising any cheap options.")
    st.caption(
        "The full board below ranks every external free agent by how keenly a team of this timeline "
        "would chase them — each scored independently (so the mid-level shows up on more than one). "
        "Gated for affordability; an aging vet on minimum money to chase a ring is the lone exception."
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
                "Market Value": "What the player projects to earn on the open market (the player-side predictor).",
                "Their Offer": "What this team would realistically offer, capped by the tool it has available.",
                "Tool": "Cap room, the mid-level exception, or a veteran-minimum slot.",
                "Fit": "Why he fits, fills a need, upgrades a starter, or rotation/minimum depth.",
            },
            height=min(720, len(pur) * 38 + 46),
        )
    else:
        st.info("No realistic external targets, this team is capped out with a full rotation.")

    st.divider()

    # ── Re-sign your own (cap-aware: who can they actually afford to keep?) ──────
    st.subheader("Re-sign their own free agents")
    _plan = B.get("resign_plan")
    if _plan:
        _over = _plan["all_in_M"] > _plan["apron2_M"]
        _tail = ("so they can't keep them all under the line — here's who fits and who gets "
                 "squeezed out:") if _over else "keeping them comfortably under the ceiling."
        st.caption((
            f"Committed payroll is **${_plan['committed_M']}M**. The luxury tax starts at "
            f"${_plan['tax_M']}M and the second apron (the practical ceiling) at "
            f"**${_plan['apron2_M']}M**. Re-signing everyone worth keeping (Bird rights, plus "
            f"exercising cheap options) would run **${_plan['all_in_M']}M**, {_tail} A fringe role "
            f"player on real money is flagged 'let walk' — fair value, but replaceable by a "
            f"minimum.").replace("$", "\\$"))
        _status_by = {x["name"]: x["status"] for x in B["resign"]}
        rp = pd.DataFrame(_plan["keeps"])
        rp["status"] = rp["name"].map(_status_by).fillna("UFA")

        def _verdict(r):
            if not r.get("worth", True):
                return "Let walk"
            return "Keep" if r["keep"] else "Can't afford"

        rp["verdict"] = rp.apply(_verdict, axis=1)
        rp = rp[["name", "pos", "status", "cost_M", "running_M", "verdict"]]
        rp.columns = ["Player", "Pos", "Status", "Keep $", "Running Payroll", "Verdict"]
        rp.insert(0, "#", range(1, len(rp) + 1))
        _ap2, _tax = _plan["apron2_M"], _plan["tax_M"]

        def _sty_run(v, _r):
            try:
                n = float(v)
            except (ValueError, TypeError):
                return ""
            if n >= _ap2:
                return "color:var(--value-bad);font-weight:700"
            return "color:var(--orange)" if n >= _tax else ""

        def _sty_verdict(v, _r):
            if v == "Keep":
                return "color:var(--value-good);font-weight:600"
            if v == "Can't afford":
                return "color:var(--value-bad);font-weight:700"
            return "color:var(--fg-4)"                  # let walk: replaceable, not unaffordable

        def _fmt_run(v):
            try:
                if v is None or pd.isna(v):
                    return "—"
            except (ValueError, TypeError):
                return "—"
            return f"${v:.0f}M"

        html_table(
            rp,
            formatters={"Keep $": lambda v: f"${v:.0f}M", "Running Payroll": _fmt_run},
            styles={"Status": _sty_status, "Running Payroll": _sty_run, "Verdict": _sty_verdict},
            aligns={"#": "right", "Keep $": "right", "Running Payroll": "right"},
            numeric={"#", "Keep $", "Running Payroll"},
            helps={"Running Payroll": "Cumulative payroll if you keep this player plus everyone worth keeping above him.",
                   "Verdict": "Keep = cost-effective and fits under the second apron; can't afford = the keep that tips the team over it; let walk = fair value but a replaceable role player, better spent on a minimum."},
            height=min(560, len(rp) * 38 + 46),
        )
    elif B["resign"]:
        st.caption("Players already on the roster who can be kept via Bird rights, no cap room required.")
        res = pd.DataFrame(B["resign"])[["name", "pos", "status", "value_M", "offer_M"]]
        res.columns = ["Player", "Pos", "Status", "Market Value", "Re-sign Cost"]
        res.insert(0, "#", range(1, len(res) + 1))
        html_table(
            res,
            formatters={"Market Value": lambda v: f"${v:.0f}M", "Re-sign Cost": lambda v: f"${v:.0f}M"},
            styles={"Status": _sty_status, "Re-sign Cost": _sty_offer},
            aligns={"#": "right", "Market Value": "right", "Re-sign Cost": "right"},
            numeric={"#", "Market Value", "Re-sign Cost"},
            height=min(520, len(res) * 38 + 46),
        )
    else:
        st.info("No notable free agents of their own to re-sign.")

    # ── Method ──────────────────────────────────────────────────────────────────
    with st.expander("How these boards are built"):
        st.markdown(
            f"""
- **Candidate pool**, every free agent this offseason (UFA, RFA, player/team option) with enough
  minutes to rank: **{DATA['n_free_agents']}** players for **{DATA['season']}**.
- **Each player's market value** comes from the same model as the player-side predictor, so the
  numbers match exactly.
- **Roster fit**, where he'd slot into this team's depth chart at his position (start, upgrade, or
  depth), using the same curated positions as the rest of the site.
- **Affordability gate**, a team can't realistically land a player it would massively underpay; he'll
  get closer to his value elsewhere. The lone exception is an aging vet taking minimum/exception money
  to chase a ring.
- **Timeline fit**, a contender chases win-now production and ring-chasing vets; a rebuilder chases youth
  and passes on aging stars. Derived from ~900 real free-agent signings (2013–2025).
- This is the inverse of the **Likely Suitors** list on the player side, same engine, read from the
  team's chair instead of the player's.
            """
        )
