"""Pre-compute the Front Office board for all 30 teams and persist it to
cache/fa_board_v1.json so the page loads instantly (no live prediction).

The inverse of Likely Suitors: for each team, the free agents it should target
this offseason — split into re-signing its own (Bird rights) and pursuing
external players — each with the contract the team would realistically offer and
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
from utils import normalize, DEFAULT_MIN_THRESHOLD  # noqa: E402
import team_suitors as ts  # noqa: E402

OUT = ROOT / "cache" / "fa_board_v1.json"
POSITIONS = ["PG", "SG", "SF", "PF", "C"]
STARTER = 15.0          # a league-average starter's Barrett Score (for needs)
MIN_MONEY = 7.0         # at/below this an offer is veteran-minimum money, not a real bid

SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "cp", "__file__": str((ROOT / "pages" / "Contract_Predictor.py").resolve())}
exec(compile("".join(SRC[:cut]), "cp", "exec"), ns)

CUR = ns["CURRENT_SEASON"]
CONTRACT = ns.get("CONTRACT_SEASON", CUR)
gpf, pc, fmt_nc = ns["get_player_features"], ns["predict_contract"], ns["fmt_next_contract"]

full = ns["build_ranked_projected"](CUR).copy()
pos2k = ts.load_player_positions()
full["pos"] = full["Player"].map(lambda n: ts.resolve_position(n, "", pos2k))
nc = ns["fetch_next_year_contracts"](ns["season_to_espn_year"](CUR), cache_v=7)
rookie = ns["fetch_rookie_scale_players"](CUR)
payroll = pd.DataFrame({"team": full["Team"].astype(str), "player": full["Player"].astype(str)})
LAND = ts.apply_real_cap(ts.load_team_landscape(),
                         ts.compute_cap_space(payroll, nc, ns["SALARY_CAP_M"].get(CONTRACT, 165.0)))
ROST = ts.build_rosters(full)


def fa_status(name):
    s = fmt_nc(name, nc)
    if s == "RFA":
        return "RFA"
    if s == "—":
        return "RFA" if normalize(name) in rookie else "UFA"
    if " PO" in s:
        return "Player Option"
    if " TO" in s:
        return "Team Option"
    return None


# ── Free-agent candidate pool: predict each one once ──────────────────────────
print("predicting free-agent pool ...")
qualified = full[full["total_min"] >= DEFAULT_MIN_THRESHOLD]
cands = []
for _, r in qualified.iterrows():
    name = str(r["Player"])
    st_ = fa_status(name)
    if not st_:
        continue
    f = gpf(name, CUR)
    if not f:
        continue
    cands.append({
        "name": name, "status": st_,
        "value_M": round(float(pc(f)["predicted"]) / 1e6, 1),
        "barrett": round(float(f["barrett_score"]), 1),
        "pos": ts.resolve_position(name, f.get("position_detailed") or "", pos2k),
        "age": int(f.get("age") or 0),
        "team": str(f.get("current_team") or r["Team"]),
    })
print(f"  {len(cands)} free agents")


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
        return "Re-sign · Bird rights"
    if x["offer_M"] <= MIN_MONEY:
        return "Veteran-minimum depth"
    if x["slot"] == 0 and x["displaces"]:
        return f"Upgrade · would start over {x['displaces']}"
    if x["slot"] == 0:
        return f"Fills an opening at {primary(x['pos'])}"
    if x["displaces"]:
        return f"Rotation depth behind {x['displaces']}"
    return "Rotation depth"


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
                     "keen": offer * des,
                     "tool": ("Bird rights" if is_inc
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

    return {
        "team": team,
        "name": str(t.get("team_name", team)),
        "cap_room_M": round(cap), "exception_M": round(exc),
        "timeline": ts._TL_DISPLAY.get(tl, tl) or "—",
        "needs": need, "thin": thin,
        "resign": [pack(x) for x in rows if x["is_inc"]],
        "pursue": [pack(x) for x in rows if not x["is_inc"]][:18],
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
