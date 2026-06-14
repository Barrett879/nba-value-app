"""Pre-compute the Front Office board for all 30 teams and persist it to
cache/fa_board_v1.json so the page loads instantly (no live prediction).

The inverse of Likely Suitors: for each team, the free agents it should target
this offseason, split into re-signing its own (Bird rights) and pursuing
external players, each with the contract the team would realistically offer and
why, gated for affordability (no banking on a player taking a huge paycut) and
timeline fit (rebuilds pass on aging stars; that lives in team_suitors.desire).

Re-run whenever the current-season data or the model changes (same cadence as
the comp pool). Usage:  python -u scripts/build_fa_board.py
"""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from utils import (normalize, DEFAULT_MIN_THRESHOLD,  # noqa: E402
                   option_opt_in_prob, OPTION_OPT_IN_THRESHOLD,
                   classify_fa_status)
import team_suitors as ts  # noqa: E402

OUT = ROOT / "cache" / "fa_board_v1.json"
POSITIONS = ["PG", "SG", "SF", "PF", "C"]
STARTER = 15.0          # a league-average starter's Barrett Score (for needs)
MIN_MONEY = 7.0         # at/below this an offer is low-cost depth money, not a real bid
MIN_SALARY = 3.0        # at/below this it's a true veteran-minimum deal (vs BAE-level depth)

SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "cp", "__file__": str((ROOT / "pages" / "Contract_Predictor.py").resolve())}
exec(compile("".join(SRC[:cut]), "cp", "exec"), ns)

CUR = ns["CURRENT_SEASON"]
CONTRACT = ns.get("CONTRACT_SEASON", CUR)
gpf, pc, fmt_nc = ns["get_player_features"], ns["predict_contract"], ns["fmt_next_contract"]
# Market value = the SAME market-blended figure the Contract Predictor hero
# shows (model blended toward comps), not the raw model output — so a player's
# value is identical on his page and on every team's board.
pcv = ns["projected_contract_value"]

full = ns["build_ranked_projected"](CUR).copy()
pos2k = ts.load_player_positions()
full["pos"] = full["Player"].map(lambda n: ts.resolve_position(n, "", pos2k))
nc = ns["fetch_next_year_contracts"](ns["season_to_espn_year"](CUR), cache_v=7)
rookie = ns["fetch_rookie_scale_players"](CUR)
payroll = pd.DataFrame({"team": full["Team"].astype(str), "player": full["Player"].astype(str)})
CAP_M = ns["SALARY_CAP_M"].get(CONTRACT, 165.0)
CAP_TABLE = ts.compute_cap_space(payroll, nc, CAP_M)
LAND = ts.apply_real_cap(ts.load_team_landscape(), CAP_TABLE)
# League pay lines, derived from the cap like team_suitors: luxury tax ~1.215x,
# second apron ~1.344x. The 2nd apron is the practical ceiling for re-signings.
TAX_M, APRON2_M = round(CAP_M * 1.215, 1), round(CAP_M * ts._APRON2_RATIO, 1)
ROST = ts.build_rosters(full)


def fa_status(name):
    # Shared single-source classifier (utils.classify_fa_status): the next-year
    # salary feed, cross-checked against the contract-end scraper for option-
    # holders / signed players the feed omits (e.g. Reaves' PO, Will Richard's
    # rookie deal). Same logic the Free Agent Class page + home summary use.
    return classify_fa_status(name, fmt_nc(name, nc), rookie, CUR)


# ── Free-agent candidate pool: predict each one once ──────────────────────────
print("predicting free-agent pool ...")
qualified = full[full["total_min"] >= DEFAULT_MIN_THRESHOLD]
cands, opted_in = [], []
for _, r in qualified.iterrows():
    name = str(r["Player"])
    st_ = fa_status(name)
    if not st_:
        continue
    f = gpf(name, CUR)
    if not f:
        continue
    value_M = round(float(pcv(f)) / 1e6, 1)
    age = int(f.get("age") or 0)
    # Option-year salary (team OR player option) from the contracts feed — the
    # pre-set figure the option would pay next season, in $M.
    opt_M = float((nc.get(normalize(name)) or {}).get("salary") or 0) / 1e6
    # A player who will exercise his player option is staying — not a free agent,
    # so he shouldn't appear on anyone's board. Drop the likely opt-ins.
    if st_ == "Player Option" and option_opt_in_prob(opt_M, value_M, age) >= OPTION_OPT_IN_THRESHOLD:
        opted_in.append(f"{name} (${opt_M:.0f}M opt)")
        continue
    cands.append({
        "name": name, "status": st_,
        "value_M": value_M,
        "opt_M": round(opt_M, 1),
        "barrett": round(float(f["barrett_score"]), 1),
        "pos": ts.resolve_position(name, f.get("position_detailed") or "", pos2k),
        "age": age,
        "team": str(f.get("current_team") or r["Team"]),
    })
print(f"  {len(cands)} free agents  ({len(opted_in)} option-holders excluded as likely opt-ins)")
for x in opted_in:
    print(f"    opt-in: {x}")


def primary(pos):
    return ts._primary_position(pos)


def needs_for(roster):
    """Positions with no starter-level primary player (need) and thin spots."""
    need, thin = [], []
    for p in POSITIONS:
        here = sorted((float(b) for pp, b in zip(roster["pos"], roster["barrett"])
                       if primary(pp) == p), reverse=True)
        if not here or here[0] < STARTER:
            need.append(p)
        elif len(here) < 2 or here[1] < 8.0:
            thin.append(p)
    return need, thin


def why(x, exc):
    if x["is_inc"]:
        return ("Pick up team option" if x["status"] == "Team Option"
                else "Re-sign · Bird rights")
    if x["offer_M"] <= MIN_MONEY:
        return "Veteran-minimum depth"
    if x["slot"] == 0 and x["displaces"]:
        return f"Upgrade · would start over {x['displaces']}"
    if x["slot"] == 0:
        return f"Fills an opening at {primary(x['pos'])}"
    if x["displaces"]:
        return f"Rotation depth behind {x['displaces']}"
    return "Rotation depth"


def _grade(s):
    return "A+" if s >= 82 else "A" if s >= 74 else "A-" if s >= 66 else "B+" if s >= 58 else "B"


def best_fits_for(rows, needs, thin):
    """The free agents who best MATCH this team, roster need + the team's positions
    of need + timeline desire + value, fused into one fit score. Deliberately distinct
    from the keenness-ranked board: it rewards genuine need-fillers and bargains over
    the most expensive name a team could sign, deduped to one pick per position."""
    scored = []
    for x in rows:
        if x["is_inc"]:
            continue
        slot = x["slot"]
        # Depth-chart slot: a starter-level fit dominates; deep depth is heavily discounted.
        need = (1.0 if slot == 0 and not x["displaces"] else 0.82 if slot == 0
                else 0.45 if slot == 1 else 0.15)
        val, off = x["value_M"], x["offer_M"]
        value_score = max(0.5, min(1.0, 0.5 + (val - off) / max(val, 1.0)))
        p = primary(x["pos"])
        # Bonus for actually plugging a position the team has no/thin starter at.
        pos_bonus = 0.12 if p in needs else 0.06 if p in thin else 0.0
        score = round(min(1.0, 0.45 * need + 0.30 * x["des"] + 0.25 * value_score + pos_bonus) * 100)
        if slot == 0 and not x["displaces"]:
            need_txt = f"Fills the opening at {p}"
        elif slot == 0:
            need_txt = f"Upgrades on {x['displaces']} at {p}"
        elif slot == 1:
            need_txt = f"Adds a needed second body at {p}"
        else:
            need_txt = f"Rotation depth at {p}"
        val_txt = (f"a bargain at ${off}M against his ${val}M market" if off < val * 0.9
                   else f"fair value at ${off}M")
        scored.append({"name": x["name"], "pos": x["pos"], "from": x["team"],
                       "status": x["status"], "value_M": val, "offer_M": off,
                       "fit": score, "grade": _grade(score), "ppos": p, "slot": slot,
                       "why": f"{need_txt}, {val_txt}."})
    # One pick per position so the three fits diversify (no two C upgrades for the same hole).
    scored.sort(key=lambda s: -s["fit"])
    seen, out = set(), []
    for s in scored:
        if s["ppos"] in seen:
            continue
        seen.add(s["ppos"])
        out.append(s)
    return out[:3]


def resign_plan(team, resign_rows):
    """Rank a team's own free agents by quality, run a cumulative payroll total
    from its committed salary, and flag who pushes it past the second apron (the
    practical ceiling), i.e. who it realistically can't afford to keep. Skipped
    when we don't have a plausible committed payroll for the team."""
    committed = float((CAP_TABLE.get(team) or {}).get("committed_M") or 0.0)
    if committed < 50.0 or not resign_rows:
        return None
    running = committed
    tax_r, apron2_r = round(TAX_M), round(APRON2_M)
    keeps = []
    for x in sorted(resign_rows, key=lambda r: -r["barrett"]):
        running += x["offer_M"]
        run_r = round(running)                       # the displayed number drives the verdict
        zone = "ok" if run_r < tax_r else "tax" if run_r < apron2_r else "over"
        keeps.append({"name": x["name"], "pos": x["pos"], "cost_M": x["offer_M"],
                      "running_M": run_r, "zone": zone, "keep": run_r < apron2_r})
    return {"committed_M": round(committed), "tax_M": tax_r,
            "apron2_M": apron2_r, "all_in_M": round(running), "keeps": keeps}


def offseason_plan(pursue_rows, cap_room, mle):
    """A REALISTIC external-signing haul (vs the full 20-deep pursue board where
    every target gets an independent offer). Walk the keenness-ranked targets and
    spend the team's ACTUAL tools, CBA-correctly:
      - a cap-space team spends its cap-room pool, then the ~$8M room exception;
      - an over-the-cap team spends ONE mid-level exception (~$15M, split-able);
      - either way, a couple of veteran minimums for depth.
    A team uses cap room OR the full mid-level, never both. Each tool is used
    once; stops at 5 moves (a real summer). A target the team can't afford with
    its remaining tools is skipped."""
    if cap_room >= 8:                                  # cap-space team
        cap_left, exc_left, exc_label = cap_room, 8.0, "Room exception"
    else:                                              # over the cap
        cap_left, exc_left, exc_label = 0.0, mle, "Mid-level"
    out, lowcost = [], 0
    for x in pursue_rows:
        offer = x["offer_M"]
        if cap_left + 1e-6 >= offer:                   # cap-space team, any size
            tool, cap_left = "Cap room", cap_left - offer
        elif offer > MIN_MONEY and exc_left + 1e-6 >= offer:
            tool, exc_left = exc_label, exc_left - offer
        elif offer <= MIN_SALARY and lowcost < 3:      # a true veteran minimum
            tool, lowcost = "Minimum", lowcost + 1
        elif offer <= MIN_MONEY and lowcost < 3:       # BAE-level depth
            tool, lowcost = "Depth", lowcost + 1
        else:
            continue
        out.append({"name": x["name"], "pos": x["pos"], "from": x["team"],
                    "cost_M": offer, "tool": tool})
        if len(out) >= 5:
            break
    return out


def board_for(team):
    row = LAND[LAND["team"].astype(str) == team]
    if row.empty:
        return None
    t = row.iloc[0]
    cap, exc = float(t["cap_space_M"]), float(t["exception_M"])
    tl = str(t.get("timeline", "")).strip().lower()
    roster = ROST[ROST["team"] == team]
    rows = []
    for c in cands:
        is_inc = c["team"] == team
        need = ts.roster_need(c["barrett"], c["pos"],
                              roster[roster["player"].map(normalize) != normalize(c["name"])])
        if need["slot"] >= ts.INTEREST_DEPTH + (2 if is_inc else 0):
            continue
        tool = max(cap, exc, c["value_M"] if is_inc else 0.0)
        fit = 1.0 if is_inc else ts._FIT_FACTOR.get(need["slot"], 0.45)
        offer = min(c["value_M"], tool, c["value_M"] * fit)
        # Keeping a team-option player means EXERCISING the option, not signing a
        # new (usually larger) Bird-rights deal — so his keep cost is the option
        # salary, capped at his market value for the rare above-market option
        # (you'd decline it and re-sign cheaper). Only on his own team; another
        # team can't take a player the option-holder controls.
        if is_inc and c["status"] == "Team Option" and c.get("opt_M"):
            offer = min(offer, c["opt_M"])
        if offer < 1.0:
            continue
        if not is_inc:
            disc = 1.0 - offer / c["value_M"] if c["value_M"] > 0 else 0.0
            ring = (offer <= max(exc, 6.0)) and c["age"] >= 32 and c["value_M"] <= 18.0
            if disc > 0.40 and not ring:
                continue
        des = 1.0 if is_inc else ts.desire_weight(tl, c["age"], c["value_M"])
        rows.append({**c, "offer_M": round(offer), "slot": need["slot"],
                     "displaces": need["displaces"], "is_inc": is_inc,
                     "keen": offer * des, "des": des,
                     "tool": ("Team option" if is_inc and c["status"] == "Team Option"
                              else "Bird rights" if is_inc
                              else "Cap room" if cap + 1e-6 >= offer
                              else "Mid-level exception" if exc >= 12
                              else "Veteran minimum")})
    rows.sort(key=lambda x: -x["keen"])
    need, thin = needs_for(roster)

    def pack(x):
        return {"name": x["name"], "pos": x["pos"], "status": x["status"],
                "from": x["team"], "value_M": x["value_M"], "offer_M": x["offer_M"],
                "barrett": x["barrett"], "age": x["age"], "tool": x["tool"],
                "why": why(x, exc)}

    _pursue = [x for x in rows if not x["is_inc"]]
    _committed = round(float((CAP_TABLE.get(team) or {}).get("committed_M") or 0.0))
    # Guaranteed roster, ordered by Barrett Score (for the collapsed roster view):
    # only players under contract for next season — exclude UFAs/RFAs and player/
    # team options (classify_fa_status returns non-None for any free agent/option;
    # None means he's signed and staying).
    _tf = full[full["Team"].astype(str) == team].sort_values("barrett_score", ascending=False)
    _roster = []
    for _, r in _tf.iterrows():
        _nm = str(r["Player"])
        if classify_fa_status(_nm, fmt_nc(_nm, nc), rookie, CUR) is not None:
            continue                                   # free agent / option -> not guaranteed
        _roster.append({"name": _nm, "pos": str(r["pos"]),
                        "barrett": round(float(r["barrett_score"] or 0), 1),
                        "salary_M": round(float(r.get("salary", 0) or 0) / 1e6, 1)})

    return {
        "team": team,
        "name": str(t.get("team_name", team)),
        "cap_room_M": round(cap), "exception_M": round(exc),
        "committed_M": _committed, "tax_M": round(TAX_M), "apron2_M": round(APRON2_M),
        "timeline": ts._TL_DISPLAY.get(tl, tl) or "—",
        "needs": need, "thin": thin,
        "roster": _roster,
        "best_fits": best_fits_for(rows, need, thin),
        "plan": offseason_plan(_pursue, cap, exc),
        "resign": [pack(x) for x in rows if x["is_inc"]],
        "resign_plan": resign_plan(team, [x for x in rows if x["is_inc"]]),
        "pursue": [pack(x) for x in _pursue][:20],
    }


teams = sorted(LAND["team"].astype(str).unique())
boards = {}
for tm in teams:
    b = board_for(tm)
    if b:
        boards[tm] = b
        print(f"  {tm:4} {b['timeline']:11} cap ${b['cap_room_M']:>3}M  "
              f"resign {len(b['resign']):>2}  pursue {len(b['pursue']):>2}  needs {','.join(b['needs']) or '—'}")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({
    "season": CUR, "contract_season": CONTRACT,
    "cap_M": ns["SALARY_CAP_M"].get(CONTRACT, 165.0),
    "n_free_agents": len(cands),
    "teams": boards,
}, indent=1))
print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(boards)} teams)")
