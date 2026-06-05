"""Team-facing contract predictor (concept probe): pick a team -> the free agents
it should target, each with the contract the team would realistically offer and
why. The inverse of Likely Suitors, reusing the same engine (roster need + real
cap tools + timeline desire).

Usage:  python -u scripts/team_targets_probe.py LAL
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from utils import normalize, DEFAULT_MIN_THRESHOLD  # noqa: E402
import team_suitors as ts  # noqa: E402

TEAMS = [a.upper() for a in sys.argv[1:]] or ["LAL"]

SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "cp", "__file__": str((ROOT / "pages" / "Contract_Predictor.py").resolve())}
exec(compile("".join(SRC[:cut]), "cp", "exec"), ns)

CUR = ns["CURRENT_SEASON"]
CONTRACT = ns.get("CONTRACT_SEASON", CUR)
gpf, pc, fmt_nc = ns["get_player_features"], ns["predict_contract"], ns["fmt_next_contract"]
DEFAULT_MIN = DEFAULT_MIN_THRESHOLD

full = ns["build_ranked_projected"](CUR).copy()
pos2k = ts.load_player_positions()
full["pos"] = full["Player"].map(lambda n: ts.resolve_position(n, "", pos2k))
nc = ns["fetch_next_year_contracts"](ns["season_to_espn_year"](CUR), cache_v=7)
rookie = ns["fetch_rookie_scale_players"](CUR)
payroll = pd.DataFrame({"team": full["Team"].astype(str), "player": full["Player"].astype(str)})
LAND = ts.apply_real_cap(ts.load_team_landscape(),
                         ts.compute_cap_space(payroll, nc, ns["SALARY_CAP_M"].get(CONTRACT, 165.0)))
ROST = ts.build_rosters(full)


def _fa_status(name):
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


# ── Build the free-agent candidate pool (qualified players only) ──────────────
qualified = full[full["total_min"] >= DEFAULT_MIN]
cands = []
for _, r in qualified.iterrows():
    name = str(r["Player"])
    st_ = _fa_status(name)
    if not st_:
        continue
    f = gpf(name, CUR)
    if not f:
        continue
    cands.append({
        "name": name, "status": st_,
        "price_M": float(pc(f)["predicted"]) / 1e6,
        "barrett": float(f["barrett_score"]),
        "pos": ts.resolve_position(name, f.get("position_detailed") or "", pos2k),
        "age": f.get("age"),
        "team": str(f.get("current_team") or r["Team"]),
    })


def team_targets(team, n=12):
    row = LAND[LAND["team"].astype(str) == team]
    if row.empty:
        return None, []
    t = row.iloc[0]
    cap, exc = float(t["cap_space_M"]), float(t["exception_M"])
    tl = str(t.get("timeline", "")).strip().lower()
    roster = ROST[ROST["team"] == team]
    out = []
    for c in cands:
        is_inc = c["team"] == team
        need = ts.roster_need(c["barrett"], c["pos"],
                              roster[roster["player"].map(normalize) != normalize(c["name"])])
        if need["slot"] >= ts.INTEREST_DEPTH + (2 if is_inc else 0):
            continue
        tool = max(cap, exc, c["price_M"] if is_inc else 0.0)
        fit = 1.0 if is_inc else ts._FIT_FACTOR.get(need["slot"], 0.45)
        offer = min(c["price_M"], tool, c["price_M"] * fit)
        if offer < 1.0:
            continue
        # Affordability realism: a team can't realistically land a player it would
        # massively underpay — he'll get closer to his value elsewhere. The lone
        # exception is an aging vet taking minimum/exception money to chase a ring.
        if not is_inc:
            value = c["price_M"]
            discount = 1.0 - offer / value if value > 0 else 0.0
            age = float(c["age"]) if c["age"] is not None else 27.0
            ring_chase = (offer <= max(exc, 6.0)) and age >= 32 and value <= 18.0
            if discount > 0.40 and not ring_chase:
                continue
        des = 1.0 if is_inc else ts.desire_weight(tl, c["age"], c["price_M"])
        keen = offer * des
        out.append({**c, "offer_M": offer, "slot": need["slot"],
                    "displaces": need["displaces"], "is_inc": is_inc, "keen": keen,
                    "tool_label": ("Bird rights" if is_inc else
                                   "cap room" if cap + 1e-6 >= offer else
                                   "mid-level exception" if exc >= 15 else "exception")})
    out.sort(key=lambda x: -x["keen"])
    return t, out[:n]


print(f"FA candidate pool: {len(cands)} players\n")
for TEAM in TEAMS:
    t, tgts = team_targets(TEAM, n=16)
    if t is None:
        print(f"{TEAM}: not in landscape\n"); continue
    print(f"=== {TEAM} · cap room ${float(t['cap_space_M']):.0f}M · exception ${float(t['exception_M']):.0f}M "
          f"· timeline '{str(t.get('timeline','')).strip()}' ===")
    resign = [x for x in tgts if x["is_inc"]]
    pursue = [x for x in tgts if not x["is_inc"]]
    print("  -- re-sign your own --")
    for x in resign:
        print(f"    {x['name']:22}{x['pos']:7}{x['status']:16}${x['offer_M']:>5.0f}M")
    print("  -- pursue (external) --")
    for x in pursue:
        why = (f"start over {x['displaces']}" if x["slot"] == 0 and x["displaces"]
               else f"fill a need ({x['pos']})" if x["slot"] == 0
               else f"depth behind {x['displaces']}" if x["displaces"] else "depth")
        print(f"    {x['name']:22}{x['pos']:7}{x['status']:16}${x['offer_M']:>5.0f}M  {why}")
    print()
