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
# first apron ~1.267x, second apron ~1.344x. A team OVER the 1st apron can only
# use the taxpayer mid-level (~$5.7M) and is hard-capped at the 2nd apron; a team
# OVER the 2nd apron has no mid-level at all (minimums only).
TAX_M = round(CAP_M * 1.215, 1)
APRON1_M = round(CAP_M * ts._APRON1_RATIO, 1)
APRON2_M = round(CAP_M * ts._APRON2_RATIO, 1)
TAXPAYER_MLE_M = 5.7    # the taxpayer mid-level (2026-27); vs the ~$15M non-taxpayer MLE
ROST = ts.build_rosters(full)

# Manual roster corrections (data/roster_corrections.csv): players the scraped
# contract feed still lists as rostered but who've actually been waived. Keyed
# by (team, normalized name) -> action. Only "waived" is used today; it drops
# the player from that team's guaranteed roster without touching his stats.
_ROSTER_FIX = {}
_rc_path = ROOT / "data" / "roster_corrections.csv"
if _rc_path.exists():
    _rc = pd.read_csv(_rc_path, comment="#")
    for _, _r in _rc.iterrows():
        _ROSTER_FIX[(str(_r["team"]).strip(), normalize(str(_r["player"])))] = \
            str(_r["action"]).strip().lower()

# 2026 draft picks (data/draft_picks_2026.csv): current ownership per pick, post
# May-2026 lottery (order set; draft ~June 24). DRAFT_PICKS[team] = list of
# {"overall","round","cost_M"}. First-round cost = 120%-of-scale cap hit (the
# standard rookie deal); second-rounders are modeled as two-way (no standard
# roster spot, ~no cap hit) and just shown for completeness.
_ROOKIE_SCALE_M = {1: 14.8, 2: 13.2, 3: 11.9, 4: 10.7, 5: 9.7, 6: 8.8, 7: 8.0,
                   8: 7.4, 9: 6.8, 10: 6.4, 11: 6.1, 12: 5.8, 13: 5.5, 14: 5.2,
                   15: 5.0, 16: 4.7, 17: 4.5, 18: 4.3, 19: 4.1, 20: 3.9, 21: 3.8,
                   22: 3.6, 23: 3.5, 24: 3.3, 25: 3.2, 26: 3.1, 27: 3.0, 28: 3.0,
                   29: 3.0, 30: 2.9}
DRAFT_PICKS = {}
_dp_path = ROOT / "data" / "draft_picks_2026.csv"
if _dp_path.exists():
    _dp = pd.read_csv(_dp_path, comment="#")
    for _, _r in _dp.iterrows():
        _ov, _rd, _tm = int(_r["overall"]), int(_r["round"]), str(_r["team"]).strip()
        DRAFT_PICKS.setdefault(_tm, []).append(
            {"overall": _ov, "round": _rd,
             "cost_M": _ROOKIE_SCALE_M.get(_ov, 0.0) if _rd == 1 else 0.0})


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


SECONDARY_W = 0.25      # a player's secondary position counts this much (vs 1.0 primary)


def pos_weights(pos_str):
    """A player's positional coverage: 1.0 at his primary spot, SECONDARY_W at
    each secondary one (a SG/SF is a full SG and a partial SF). Used so a player
    who can slide over partly covers that position for needs + roster balance."""
    out = {}
    for i, p in enumerate(str(pos_str).split("/")):
        p = p.strip()
        if p in POSITIONS:
            out[p] = max(out.get(p, 0.0), 1.0 if i == 0 else SECONDARY_W)
    return out


def needs_for(roster):
    """Positions with no starter-level player (need) and thin spots. A player
    counts fully at his primary position and partly (SECONDARY_W) at each
    secondary one, so a forward who can slide over partly covers that spot."""
    need, thin = [], []
    for p in POSITIONS:
        here = []
        for pp, b in zip(roster["pos"], roster["barrett"]):
            wt = pos_weights(pp).get(p)
            if wt:
                here.append(float(b) * wt)
        here.sort(reverse=True)
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


def _worth_resign(barrett, value_M, cost_M):
    """Is keeping this free agent cost-effective, or would the team just be
    re-signing him for the sake of it? Worth it when he's a genuine rotation
    contributor (Barrett >= 7), a clear bargain (market value well above the keep
    cost, e.g. a cheap team option), or cheap enough to be a no-risk flier
    (minimum money). A fair-priced fringe role player on real money walks — the
    team replaces him with a minimum rather than pay up to keep a replaceable
    piece."""
    return barrett >= 7.0 or cost_M <= MIN_SALARY or (value_M - cost_M) >= 5.0


def resign_plan(team, resign_rows, reserve=0.0):
    """Rank a team's own free agents by quality and decide who it actually keeps.
    Two gates: (1) is he WORTH re-signing (cost-effective, not just filling a
    spot — see `_worth_resign`); (2) does keeping him still fit under the second
    apron (the practical ceiling). A cumulative payroll total runs over the
    worth-keeping players in quality order, so the apron line falls where the
    team runs out of room. `reserve` (the rookie-scale money owed to the team's
    own first-round picks) is counted up front, so the apron line accounts for
    the picks too — the team lets its marginal keeper walk rather than blow past
    the 2nd apron. Skipped when we don't have a plausible committed payroll."""
    committed = float((CAP_TABLE.get(team) or {}).get("committed_M") or 0.0)
    if committed < 50.0 or not resign_rows:
        return None
    running = committed + reserve
    tax_r, apron2_r = round(TAX_M), round(APRON2_M)
    keeps = []
    for x in sorted(resign_rows, key=lambda r: -r["barrett"]):
        worth = _worth_resign(x["barrett"], x["value_M"], x["offer_M"])
        if not worth:                                # replaceable -> let him walk
            keeps.append({"name": x["name"], "pos": x["pos"], "cost_M": x["offer_M"],
                          "value_M": x["value_M"], "barrett": x["barrett"],
                          "running_M": None, "zone": "walk", "worth": False, "keep": False})
            continue
        running += x["offer_M"]
        run_r = round(running)                       # the displayed number drives the verdict
        afford = run_r < apron2_r
        zone = "ok" if run_r < tax_r else "tax" if run_r < apron2_r else "over"
        keeps.append({"name": x["name"], "pos": x["pos"], "cost_M": x["offer_M"],
                      "value_M": x["value_M"], "barrett": x["barrett"],
                      "running_M": run_r, "zone": zone, "worth": True,
                      "afford": afford, "keep": afford})
    return {"committed_M": round(committed), "reserve_M": round(reserve),
            "tax_M": tax_r, "apron2_M": apron2_r,
            "all_in_M": round(running), "keeps": keeps}


def offseason_plan(pursue_rows, cap_room, mle, apron_room, max_adds=5,
                   pos_counts=None, pos_cap=3, mle_label="Mid-level", floor=0):
    """A REALISTIC external-signing haul (vs the full 20-deep pursue board where
    every target gets an independent offer). Walk the keenness-ranked targets and
    spend the team's ACTUAL tools, CBA-correctly:
      - a cap-space team spends its cap-room pool, then the ~$8M room exception;
      - an over-the-cap team spends ONE mid-level exception (~$15M, split-able);
      - either way, a couple of veteran minimums for depth.
    A team uses cap room OR the full mid-level, never both. CRUCIALLY, the total
    is ALSO hard-capped by the second apron (`apron_room` = how much is left
    below it AFTER re-signing their own): a team that re-signs its core up to the
    apron can't pile a full mid-level on top — it's capped out, minimums only.

    Two passes. Pass 1 spends the real tools (cap room, mid-level) on the top
    targets, each bounded by `apron_room` (the space below the team's hard cap
    AFTER re-signs + picks). USING an exception hard-caps the team at that line.
    Pass 2 fills the roster with veteran minimums: the NBA's 14-man floor forces
    minimums up to `floor` adds even past the apron, BUT only if no hard cap was
    triggered — a club that used an exception is locked at its hard cap and can't
    exceed it for anything, while a club that signed nobody (or minimums only) can
    keep adding minimums over the apron (minimum deals are exempt). Past the floor
    the 15th man is added only if a minimum still fits. The big tools (mid-level,
    cap room) are FINITE pools, drawn down as used — once a pool or the apron room
    is gone, no more mid-sized deals.

    Positional balance: `pos_counts` is the projected roster's count by primary
    position (guaranteed + re-signs); a target is skipped if his primary spot is
    already `pos_cap`-deep, so the team fills its actual needs instead of, say,
    a fourth center."""
    if cap_room >= 8:                                  # cap-space team
        cap_left, exc_left, exc_label = cap_room, 8.0, "Room exception"
    else:                                              # over the cap: one mid-level
        cap_left, exc_left, exc_label = 0.0, mle, mle_label
    pos_counts = dict(pos_counts or {})
    out, spent, used_exc = [], 0.0, False              # spent = ALL add money; used_exc = an apron-restricted tool was used
    # Pass 1 — the real targets (cap room / mid-level signings). Each is bounded
    # by the apron room left below the team's hard cap. Using an exception (the
    # mid-level or the room exception) HARD-CAPS the team at that line.
    for x in pursue_rows:
        if len(out) >= max_adds:
            break
        offer = x["offer_M"]
        if offer <= MIN_SALARY:                        # minimums handled in pass 2
            continue
        _pos = primary(x["pos"])
        if pos_counts.get(_pos, 0) >= pos_cap:          # primary spot already deep -> skip
            continue
        if spent + offer > apron_room + 1e-6:          # the bigger tools are apron-bounded
            continue                                   #   -> would cross the hard cap
        if cap_left + 1e-6 >= offer:
            tool, cap_left, spent = "Cap room", cap_left - offer, spent + offer
        elif exc_left + 1e-6 >= offer:                 # the one exception, split across players
            tool, exc_left, spent, used_exc = exc_label, exc_left - offer, spent + offer, True
        else:
            continue                                   # no tool can fund this offer -> can't sign
        out.append({"name": x["name"], "pos": x["pos"], "from": x["team"],
                    "cost_M": offer, "tool": tool})
        pos_counts[_pos] = pos_counts.get(_pos, 0) + 1
    # Pass 2 — fill the roster with veteran minimums. The 14-man FLOOR forces
    # minimums even past the apron, BUT only when the team hasn't triggered a hard
    # cap: a club that used an exception is locked at that line and can't exceed it
    # for any reason, while a club that signed nobody (or minimums only) can keep
    # adding minimums over the apron (minimum deals are exempt). Past the floor the
    # 15th man is added only if a minimum still fits under the apron.
    for x in pursue_rows:
        if len(out) >= max_adds:
            break
        offer = x["offer_M"]
        if offer > MIN_SALARY:                          # big-tool targets already handled
            continue
        _pos = primary(x["pos"])
        if pos_counts.get(_pos, 0) >= pos_cap:
            continue
        forced = (not used_exc) and len(out) < floor    # reach 14 unless a hard cap is in force
        if not forced and spent + offer > apron_room + 1e-6:
            continue
        out.append({"name": x["name"], "pos": x["pos"], "from": x["team"],
                    "cost_M": offer, "tool": "Minimum"})
        pos_counts[_pos] = pos_counts.get(_pos, 0) + 1
        spent += offer
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
        if _ROSTER_FIX.get((team, normalize(_nm))) == "waived":
            continue                                   # waived since the feed snapshot
        if classify_fa_status(_nm, fmt_nc(_nm, nc), rookie, CUR) is not None:
            continue                                   # free agent / option -> not guaranteed
        # Salary = NEXT season (CONTRACT_SEASON, e.g. 2026-27) to match the
        # forward-looking roster + the 2026-27 cap bar — not the current-season
        # figure. (Dyson Daniels reads $25M on his extension, not last year's
        # $7.7M.) Fall back to the current salary if the next-year feed is missing
        # him (it omits some 2nd-rounders/two-ways).
        _ny = float((nc.get(normalize(_nm)) or {}).get("salary") or 0)
        _sal = _ny if _ny > 0 else float(r.get("salary", 0) or 0)
        _roster.append({"name": _nm, "pos": str(r["pos"]),
                        "barrett": round(float(r["barrett_score"] or 0), 1),
                        "salary_M": round(_sal / 1e6, 1)})

    # The whole offseason has to fit a 15-man standard roster. Open spots =
    # 15 minus the players already under contract. Re-signings come first (keep
    # your own), then external adds fill whatever's left — so the plan can't
    # carry the team past 15 (no "re-sign 5 + add 5" on top of 9 guaranteed = 19).
    ROSTER_MAX = 15
    # Draft picks are locked in — they take roster spots and rookie money before
    # any free-agent move. First-rounders occupy a standard roster spot at their
    # 120%-of-scale cap hit; second-rounders are modeled as two-way (no standard
    # spot, ~no cap hit) and just shown for completeness. Computed before the
    # re-sign plan so the rookie money is reserved against the 2nd-apron line.
    _team_picks = sorted(DRAFT_PICKS.get(team, []), key=lambda p: p["overall"])
    _first = [p for p in _team_picks if p["round"] == 1]
    _second = [p for p in _team_picks if p["round"] == 2]
    _pick_moves = ([{"name": f"2026 Pick #{p['overall']}", "pos": "1st round",
                     "cost_M": round(p["cost_M"]), "tool": "Draft pick", "kind": "pick",
                     "overall": p["overall"], "round": 1} for p in _first]
                   + [{"name": f"2026 Pick #{p['overall']}", "pos": "2nd round",
                       "cost_M": 0, "tool": "2nd-round pick", "kind": "pick",
                       "overall": p["overall"], "round": 2} for p in _second])
    _pick_cost = round(sum(p["cost_M"] for p in _first))
    # Re-sign plan reserves the pick money up front, so the marginal keeper walks
    # rather than push the team past the second apron.
    _rp = resign_plan(team, [x for x in rows if x["is_inc"]], reserve=_pick_cost)
    # Open standard spots = 15 minus players under contract minus the first-round
    # picks (the picks fill spots first). Re-signings then fill what's left, then
    # external adds — so the plan can't carry the team past 15.
    _open_spots = max(0, ROSTER_MAX - len(_roster) - len(_first))
    # A keeper can be worth it AND affordable yet still get squeezed out: once the
    # draft picks and the higher-value keepers have filled the 15 spots, there's no
    # room left. Mark those (Barrett order) so the re-sign board's verdict matches
    # the plan instead of showing "Keep" for someone the team can't actually fit.
    if _rp:
        _kept_n = 0
        for _k in _rp["keeps"]:
            if _k.get("afford") and _k.get("keep"):
                if _kept_n < _open_spots:
                    _kept_n += 1
                else:
                    _k["keep"], _k["zone"] = False, "noroom"
    # The re-signings ARE part of the realistic plan — the cost-effective keepers
    # (worth it AND fitting under the apron, per resign_plan's two gates), capped
    # at the open roster spots. Show them alongside the external signings.
    _keepers = [k for k in (_rp or {}).get("keeps", []) if k.get("keep")]
    _resign_moves = [{"name": k["name"], "pos": k["pos"], "cost_M": k["cost_M"],
                      "tool": "Re-sign", "kind": "resign"}
                     for k in _keepers[:_open_spots]]
    # Cap room is the THEORETICAL max (own FAs renounced). A team that re-signs
    # its own keepers via Bird rights is then over the cap and only has the
    # mid-level — so the external part of the plan gets cap room MINUS those
    # re-signings AND the rookie-scale money owed to its first-round picks.
    _resign_cost = round(sum(m["cost_M"] for m in _resign_moves))
    # Payroll already committed before any free-agent ADD (under contract +
    # re-signs + rookie-scale money for the picks). The apron TIER this lands in
    # sets both the over-the-cap tool and the hard ceiling on big-money signings:
    #   - cap-space team: cap room + room exception, hard-capped at the 1st apron
    #   - over the cap, under the 1st apron: full non-taxpayer mid-level (~$15M),
    #     hard-capped at the 1st apron (using the full MLE triggers that cap)
    #   - over the 1st apron: ONLY the taxpayer mid-level (~$5.7M), hard cap = 2nd apron
    #   - over the 2nd apron: NO mid-level at all — minimums only
    _base = _committed + _resign_cost + _pick_cost
    _real_cap_room = max(0.0, CAP_M - _base)
    if _real_cap_room >= 8:
        _mle, _mle_label, _hard_cap = exc, "Mid-level", APRON1_M
    elif _base >= APRON2_M:
        _mle, _mle_label, _hard_cap = 0.0, "Mid-level", APRON2_M
    elif _base >= APRON1_M:
        _mle, _mle_label, _hard_cap = TAXPAYER_MLE_M, "Taxpayer MLE", APRON2_M
    else:
        _mle, _mle_label, _hard_cap = exc, "Mid-level", APRON1_M
    _apron_room = max(0.0, round(_hard_cap) - _base)
    # The PROJECTED 2026-27 roster: who's under contract plus the keepers they
    # re-sign. Positions of need and roster balance are judged against THIS, not
    # the 2025-26 roster that still counts departing free agents. (A position-
    # weighted count: primary 1.0, secondary SECONDARY_W, per `pos_weights`.)
    _kept = _keepers[:_open_spots]
    _proj = ([{"pos": r["pos"], "barrett": r["barrett"]} for r in _roster]
             + [{"pos": k["pos"], "barrett": k["barrett"]} for k in _kept])
    need, thin = needs_for(pd.DataFrame(_proj)) if _proj else ([], [])
    # Roster balance for external adds: count the projected roster's depth by
    # PRIMARY position (so the plan won't stack a 4th of one position). Draft
    # picks are position-unknown, so they don't seed the counts.
    _pos_counts = {}
    for _m in _roster + _resign_moves:
        _pp = primary(_m["pos"])
        _pos_counts[_pp] = _pos_counts.get(_pp, 0) + 1
    _adds_left = max(0, _open_spots - len(_resign_moves))
    # NBA teams must carry at least 14. Players already locked = under contract +
    # first-round picks + re-signs; the plan must add enough minimums to reach 14
    # even if the team is capped out at the apron (it still has to field a roster).
    _locked = len(_roster) + len(_first) + len(_resign_moves)
    _floor = min(_adds_left, max(0, 14 - _locked))
    _external = offseason_plan(_pursue, _real_cap_room, _mle, _apron_room,
                               max_adds=_adds_left, pos_counts=_pos_counts,
                               mle_label=_mle_label, floor=_floor)
    for _m in _external:
        _m["kind"] = "external"
    _plan = _resign_moves + _external + _pick_moves

    return {
        "team": team,
        "name": str(t.get("team_name", team)),
        "cap_room_M": round(cap), "exception_M": round(exc),
        "committed_M": _committed, "resign_cost_M": _resign_cost,
        "tax_M": round(TAX_M), "apron1_M": round(APRON1_M), "apron2_M": round(APRON2_M),
        "timeline": ts._TL_DISPLAY.get(tl, tl) or "—",
        "needs": need, "thin": thin,
        "roster": _roster,
        "best_fits": best_fits_for(rows, need, thin),
        "plan": _plan,
        "resign": [pack(x) for x in rows if x["is_inc"]],
        "resign_plan": _rp,
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


# ── Free-agency simulation: one best-guess destination per free agent ─────────
# Runs the player-side suitor engine (rank_suitors) for EVERY free agent and
# takes his single most-likely landing spot. This is the INDEPENDENT prediction:
# a market-clearing pass was backtested (scripts/backtest_fa_sim.py) and lost ~7
# points of top-1 accuracy (real signings clear via sign-and-trades / cap holds /
# exceptions a clean cap model can't see, filtering out the true team ~36% of the
# time), so we keep the validated-best independent ranking. Measured accuracy:
# ~51% top-1 / ~59% top-5 on recent seasons (2021-25); re-signs are reliable,
# players who change teams are the noisy ~70%.
# Two modes, toggled on the page:
#   likely    - the validated top-1-optimal pick (incumbent weight 5.0). Re-signs
#               ~90% of FAs (the safe, accuracy-max call) but shows little movement.
#   realistic - softens the re-sign pull for UNRESTRICTED FAs only (RFAs/options
#               stay sticky — they really do mostly stay), so a believable share of
#               UFAs are projected to move and the board reads like free agency.
#               Lower raw top-1 (team changes are individually noisy) — see
#               scripts/backtest_fa_sim.py for the measured number.
print("\nsimulating free-agency destinations (likely + realistic) ...")
SIM_OUT = ROOT / "cache" / "fa_sim_v1.json"
MAX_REAL_ADDS = 4                                      # a team signs ~3-4 outside FAs in one summer
MIN_M = 2.3
CAP = ns["SALARY_CAP_M"].get(CONTRACT, 165.0)
_TEAMNAME = dict(zip(LAND["team"].astype(str), LAND["team_name"].astype(str)))
_FA_STATUS = {normalize(c["name"]): c["status"] for c in cands}

# ── Team Builder-derived REAL cap state (the key to money-coherent realism) ───
# Each team's actual room for OUTSIDE free agents, after the board's own re-signs
# and draft picks: real cap room if it has any, else one mid-level, capped at the
# apron it lands in. Built from the board (committed payroll + resign_cost + picks)
# so the realistic mode can't, say, have a capped-out Lakers add $40M while also
# keeping its stars. `LAND_REAL` feeds rank_suitors so external offers + tool
# labels reflect THIS, not the theoretical "renounce everyone" cap room in LAND.
_board_resigns = {normalize(m["name"]) for b in boards.values()
                  for m in b.get("plan", []) if m.get("kind") == "resign"}


def _real_state(b):
    pick_cost = sum(m["cost_M"] for m in b.get("plan", []) if m.get("kind") == "pick")
    base = b["committed_M"] + b["resign_cost_M"] + pick_cost
    ap1, ap2, exc = b["apron1_M"], b["apron2_M"], b["exception_M"]
    real_cap = max(0.0, CAP - base)
    if real_cap >= 8:
        cap_b, mle_b, hard = real_cap, exc, ap1        # cap room (+ room exception)
    elif base >= ap2:
        cap_b, mle_b, hard = 0.0, 0.0, ap2             # over the 2nd apron: minimums only
    elif base >= ap1:
        cap_b, mle_b, hard = 0.0, 6.0, ap2             # over the 1st apron: taxpayer MLE
    else:
        cap_b, mle_b, hard = 0.0, exc, ap1             # under the cap: full mid-level
    return cap_b, mle_b, max(0.0, round(hard) - base)


budget, _capcol, _exccol = {}, {}, {}
for tm, b in boards.items():
    cap_b, mle_b, aroom = _real_state(b)
    budget[tm] = {"cap": cap_b, "mle": mle_b, "mle_used": False, "apron_room": aroom}
    _capcol[tm], _exccol[tm] = cap_b, mle_b
LAND_REAL = LAND.copy()
LAND_REAL["cap_space_M"] = LAND_REAL["team"].astype(str).map(_capcol).fillna(LAND_REAL["cap_space_M"])
LAND_REAL["exception_M"] = LAND_REAL["team"].astype(str).map(_exccol).fillna(LAND_REAL["exception_M"])


def _conf(entry, c):
    if entry["is_incumbent"]:
        return "High"
    if entry["slot"] == 0 and c["value_M"] >= 10.0:    # clear starter-level target with real money
        return "Medium"
    return "Low"                                        # team changes / depth pieces are noisy


def _pick(entry, board, c):
    """Build a prediction dict from a chosen board entry; alts = the rest of the board."""
    if entry is None:                                  # no feasible suitor -> back on a minimum
        return {"predicted": c["team"], "predicted_name": _TEAMNAME.get(c["team"], c["team"]),
                "is_resign": True, "confidence": "Low", "offer_M": c["value_M"],
                "tool": "minimum", "reason": "no external market — likely back on a minimum",
                "alts": []}
    return {"predicted": entry["team"], "predicted_name": entry["team_name"],
            "is_resign": bool(entry["is_incumbent"]), "confidence": _conf(entry, c),
            "offer_M": entry["offer_M"], "tool": entry["tool"], "reason": entry["reason"],
            "alts": [{"team": d["team"], "team_name": d["team_name"],
                      "offer_M": d["offer_M"], "tool": d["tool"]}
                     for d in board if d is not entry][:2]}


# Pass 1 — score every FA two ways. "likely" board uses the production landscape
# (theoretical cap room) + GBM blend = the accuracy-max per-player pick. "realistic"
# board ranks OUTSIDE suitors against LAND_REAL (real post-re-sign cap room + real
# exceptions) with the GBM dropped, so external offers + tools are money-true.
prelim = []
for c in cands:
    args = dict(n=6, incumbent_team=c["team"], age=c["age"],
                is_rfa=(c["status"] == "RFA"), fa_status=_FA_STATUS)
    b_like = ts.rank_suitors(c["value_M"], c["barrett"], c["pos"], ROST, landscape=LAND, **args)
    b_real = ts.rank_suitors(c["value_M"], c["barrett"], c["pos"], ROST,
                             landscape=LAND_REAL, blend_dest=False, **args)
    prelim.append({"c": c, "b_like": b_like, "b_real": b_real,
                   "likely": _pick(b_like[0] if b_like else None, b_like, c)})

# Pass 2 — REALISTIC mode, made cap-coherent with the Team Builder. Two rules:
#   (1) Re-signs come straight from each team's Team Builder plan (the cost-
#       effective keepers it can actually fit under the apron). So if the board
#       only re-signs 5 Lakers, the sim shows 5 — not 7 over the 2nd apron.
#   (2) External adds are bounded by each team's REAL post-re-sign budget from
#       the board (committed payroll + those re-signs + draft picks -> real cap
#       room, or just the mid-level if over the cap, hard-capped at the apron) —
#       not the theoretical "renounce everyone" cap room. So a capped-out team
#       can't add $40M of outside players while keeping its own stars.
# (Likely mode is untouched: it's the per-player accuracy-max pick.)
# Roster spots are tracked so no team exceeds 15: players under contract + first-
# round picks occupy spots up front, then re-signs / external adds fill the rest.
ROSTER_MAX = 15
rost_used = {tm: len(b.get("roster", []))
             + sum(1 for m in b.get("plan", []) if m.get("kind") == "pick" and m.get("round") == 1)
             for tm, b in boards.items()}
adds = {}
real_by_name = {}
# Step A — board re-signs occupy their spots.
_resigned = set()
for item in prelim:
    c = item["c"]; nm = normalize(c["name"])
    if nm not in _board_resigns:
        continue
    inc = next((d for d in item["b_like"] if d["is_incumbent"]), None)
    rp = (_pick(inc, item["b_like"], c) if inc else {
        "predicted": c["team"], "predicted_name": _TEAMNAME.get(c["team"], c["team"]),
        "is_resign": True, "confidence": "High", "offer_M": c["value_M"],
        "reason": "re-signs (Team Builder plan)", "alts": []})
    rp["tool"] = "Bird rights"                          # re-sign their own, not via cap room
    real_by_name[c["name"]] = rp; _resigned.add(c["name"])
    rost_used[c["team"]] = rost_used.get(c["team"], 0) + 1
# Step B — external clear for everyone else: best OUTSIDE team that has both a
# roster spot and the real budget. Best-first so the stars get the open money.
_leftover = []
for item in sorted((x for x in prelim if x["c"]["name"] not in _resigned),
                   key=lambda x: -x["c"]["value_M"]):
    c = item["c"]; chosen = None
    for d in item["b_real"]:
        if d["is_incumbent"]:                          # board let him walk -> not a re-sign here
            continue
        t = d["team"]; o = d["offer_M"]; bud = budget.get(t)
        if not bud or rost_used.get(t, 99) >= ROSTER_MAX or adds.get(t, 0) >= MAX_REAL_ADDS:
            continue
        if bud["cap"] + 1e-6 >= o and bud["apron_room"] + 1e-6 >= o:        # cap room
            bud["cap"] -= o; bud["apron_room"] -= o; chosen = d; break
        if (not bud["mle_used"]) and bud["mle"] + 1e-6 >= o and bud["apron_room"] + 1e-6 >= o:  # one MLE
            bud["mle_used"] = True; bud["apron_room"] -= o; chosen = d; break
        if o <= MIN_M + 1e-6:                          # veteran minimum (apron-exempt)
            chosen = {**d, "tool": "minimum"}; break
    if chosen:
        t = chosen["team"]; adds[t] = adds.get(t, 0) + 1; rost_used[t] = rost_used.get(t, 0) + 1
        real_by_name[c["name"]] = _pick(chosen, item["b_real"], c)
    else:
        _leftover.append(item)
# Step C — fill rosters to the 14-man minimum with NAMED minimum deals (no generic
# "+N" gaps unless the notable-FA pool is genuinely exhausted). Three sub-passes:
ROSTER_MIN = 14


def _min_pick(team, c, reason):
    return {"predicted": team, "predicted_name": _TEAMNAME.get(team, team),
            "is_resign": team == c["team"], "confidence": "Low",
            "offer_M": round(min(c["value_M"], MIN_M), 1), "tool": "minimum",
            "reason": reason, "alts": []}


# C1 — a team's own leftover free agents come back on minimums where it still
# needs bodies (teams fill out with their own fringe guys before chasing others).
pool = []
for item in sorted(_leftover, key=lambda x: -x["c"]["value_M"]):
    c = item["c"]; t = c["team"]
    if rost_used.get(t, 99) < ROSTER_MIN:
        rost_used[t] = rost_used.get(t, 0) + 1
        real_by_name[c["name"]] = _min_pick(t, c, "back on a minimum")
    else:
        pool.append(item)
# C2 — remaining teams under 14 sign outside minimum bodies, preferring a player
# who fills a position of need, else the best available.
for tm in sorted((t for t in boards if rost_used.get(t, 0) < ROSTER_MIN),
                 key=lambda t: rost_used.get(t, 0)):
    need_pos = set(boards[tm].get("needs", []) + boards[tm].get("thin", []))
    while rost_used.get(tm, 0) < ROSTER_MIN and pool:
        ranked = sorted(pool, key=lambda x: -x["c"]["barrett"])
        fit = next((it for it in ranked if primary(it["c"]["pos"]) in need_pos), ranked[0])
        pool.remove(fit); rost_used[tm] = rost_used.get(tm, 0) + 1
        real_by_name[fit["c"]["name"]] = _min_pick(tm, fit["c"], "minimum deal to fill a roster need")
# C3 — anyone still unplaced: a 15th-man minimum at home if there's room, else unsigned.
for item in pool:
    c = item["c"]; t = c["team"]
    if rost_used.get(t, 99) < ROSTER_MAX:
        rost_used[t] = rost_used.get(t, 0) + 1
        real_by_name[c["name"]] = _min_pick(t, c, "back on a minimum")
    else:
        real_by_name[c["name"]] = {
            "predicted": None, "predicted_name": "Unsigned", "is_resign": False,
            "confidence": "Low", "offer_M": 0, "tool": "unsigned",
            "reason": "no roster spot — likely a minimum, two-way, or overseas deal", "alts": []}

sim_players = []
for item in prelim:
    c = item["c"]
    sim_players.append({
        "player": c["name"], "pos": c["pos"], "age": c["age"],
        "barrett": c["barrett"], "value_M": c["value_M"], "status": c["status"],
        "incumbent": c["team"], "incumbent_name": _TEAMNAME.get(c["team"], c["team"]),
        "likely": item["likely"], "realistic": real_by_name[c["name"]],
    })
sim_players.sort(key=lambda p: -p["value_M"])          # stars first


def _room_label(bud):
    if bud["cap"] >= 8:    return f"${round(bud['cap'])}M cap room"
    if bud["mle"] >= 15:   return "full mid-level"
    if bud["mle"] >= 6:    return "taxpayer mid-level"
    if bud["mle"] > 0:     return "mid-level"
    return "minimums only"


# Per-team cap context (Team Builder data) so the by-team view can SHOW that the
# projected signings actually fit: guaranteed payroll, the re-sign bill, and the
# real room left for outside FAs.
sim_teams = {tm: {"name": _TEAMNAME.get(tm, tm),
                  "committed_M": round(b["committed_M"]),
                  "resign_cost_M": round(b["resign_cost_M"]),
                  "room": _room_label(budget[tm]),
                  "under_contract": len(b.get("roster", [])),
                  "picks": sum(1 for m in b.get("plan", [])
                               if m.get("kind") == "pick" and m.get("round") == 1)}
             for tm, b in boards.items()}
SIM_OUT.write_text(json.dumps({
    "season": CUR, "contract_season": CONTRACT,
    "cap_M": ns["SALARY_CAP_M"].get(CONTRACT, 165.0),
    "n": len(sim_players),
    # Backtested on 1,821 historical signings (scripts/backtest_fa_sim.py).
    "accuracy": {
        "likely":    {"top1_recent": 51, "top5_recent": 59, "top1_all": 45},
        "realistic": {"top1_recent": None, "top5_recent": None, "top1_all": None},
        "note": "single best-guess via the suitor engine; re-signs reliable, team changes noisy",
    },
    "teams": sim_teams,
    "players": sim_players,
}, indent=1))
_rl = sum(1 for p in sim_players if p["likely"]["is_resign"])
_rr = sum(1 for p in sim_players if p["realistic"]["is_resign"])
_ufa = [p for p in sim_players if p["status"] == "UFA"]
_ufa_move = sum(1 for p in _ufa if not p["realistic"]["is_resign"])
print(f"  wrote {SIM_OUT.relative_to(ROOT)}  ({len(sim_players)} FAs)")
print(f"    likely:    {_rl} re-sign, {len(sim_players) - _rl} move")
print(f"    realistic: {_rr} re-sign, {len(sim_players) - _rr} move "
      f"(UFAs: {_ufa_move}/{len(_ufa)} move)")
