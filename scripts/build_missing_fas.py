#!/usr/bin/env python3
"""Project the 2026 free agents the main FA pool never ranked.

The Front Office board's candidate pool (scripts/build_fa_board.py) is gated by
fa_status() + a minutes filter, so a tail of genuine 2026 free agents (no 2026-27
salary on the books AND not flagged by the pool) is dropped entirely — they show
up in neither the under-contract feed nor the simulation. That left ~87 players
off the spreadsheet, including real rotation names (Porzingis, Rozier, Lowry...).

This rebuilds JUST those players with the SAME engine the sim uses
(team_suitors.rank_suitors, independent best-guess) and writes an isolated
supplement, cache/fa_extra_v1.json, consumed only by the spreadsheet export. The
shipped board / sim JSON (and therefore the live app) are left untouched.

Usage:  python -u scripts/build_missing_fas.py
"""
import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from utils import normalize, classify_fa_status  # noqa: E402
import team_suitors as ts  # noqa: E402
import pickle

# Reuse the Contract_Predictor namespace exactly as the board builder does, so
# value / Barrett / positions are identical to the rest of the projection.
SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "cp", "__file__": str((ROOT / "pages" / "Contract_Predictor.py").resolve())}
exec(compile("".join(SRC[:cut]), "cp", "exec"), ns)

CUR = ns["CURRENT_SEASON"]
CONTRACT = ns.get("CONTRACT_SEASON", CUR)
gpf, pcv, fmt_nc = ns["get_player_features"], ns["projected_contract_value"], ns["fmt_next_contract"]

print("loading player universe + landscape ...")
full = ns["build_ranked_projected"](CUR).copy()
pos2k = ts.load_player_positions()
full["pos"] = full["Player"].map(lambda n: ts.resolve_position(n, "", pos2k))
nc = ns["fetch_next_year_contracts"](ns["season_to_espn_year"](CUR), cache_v=7)
rookie = ns["fetch_rookie_scale_players"](CUR)
payroll = pd.DataFrame({"team": full["Team"].astype(str), "player": full["Player"].astype(str)})
CAP_M = ns["SALARY_CAP_M"].get(CONTRACT, 165.0)
LAND = ts.apply_real_cap(ts.load_team_landscape(), ts.compute_cap_space(payroll, nc, CAP_M))
ROST = ts.build_rosters(full)
cey = pickle.load(open(ROOT / "cache" / "contract_end_years_v1.pkl", "rb"))
sim = json.loads((ROOT / "cache" / "fa_sim_v1.json").read_text())

ABBR_FIX = {"PHO": "PHX", "CHO": "CHA", "BRK": "BKN", "NOH": "NOP"}
_SUF = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$")
_strip = lambda n: _SUF.sub("", n)

# ── The gap: cey players on a roster, but in NEITHER the salary feed NOR the pool ──
# CRITICAL: "absent from the salary feed" does NOT mean "free agent" — the next-year
# feed omits 2nd-rounders / two-ways / some rookie deals. classify_fa_status is the
# single source of truth (it cross-checks the contract-end scraper), so a player it
# calls SIGNED (returns None) is dropped here, not mislabeled as a free agent. Same
# gate the board pool uses. (Without this, Will Richard's rookie deal et al. leak in.)
_disp = {normalize(p): p for p in full["Player"].astype(str)}
ncn = set(nc)
fapool = {normalize(p["player"]) for p in sim["players"]}
covered = ncn | fapool
covered |= {_strip(x) for x in covered}
missing, signed_skip = [], []
for n, info in cey.items():
    tm = (info or {}).get("current_team")
    if not tm or n in covered or _strip(n) in covered:
        continue
    name = _disp.get(n) or _disp.get(_strip(n)) or n.title()
    if not classify_fa_status(name, fmt_nc(name, nc), rookie, CUR):
        signed_skip.append(name)                       # actually under contract — feed just omits the salary
        continue
    missing.append((n, ABBR_FIX.get(tm, tm)))
print(f"  {len(missing)} genuine free agents | {len(signed_skip)} signed players skipped (feed omits salary)")

ROTATION_VALUE_M = 6.0   # at/above this market value -> a real rotation FA worth folding into a roster

listing, projected = [], []
for n, tm in missing:
    name = _disp.get(n) or _disp.get(_strip(n)) or n.title()
    f = gpf(name, CUR)
    pos = ts.resolve_position(name, (f or {}).get("position_detailed") or "", pos2k)
    if not f:
        listing.append({"player": name, "team": tm, "pos": pos,
                        "barrett": None, "value_M": None, "projected": False})
        continue
    value_M = round(float(pcv(f)) / 1e6, 1)
    barrett = round(float(f.get("barrett_score") or 0), 1)
    age = int(f.get("age") or 0)
    row = {"player": name, "team": tm, "pos": pos, "barrett": barrett, "value_M": value_M}
    if value_M < ROTATION_VALUE_M:
        listing.append({**row, "projected": False})
        continue
    # Independent best-guess destination — the same call the sim's "likely" board makes.
    board = ts.rank_suitors(value_M, barrett, pos, ROST, landscape=LAND, n=6,
                            incumbent_team=tm, age=age)
    if not board:
        listing.append({**row, "projected": False})
        continue
    top = board[0]
    dest = ABBR_FIX.get(str(top["team"]), str(top["team"]))
    rec = {**row, "projected": True,
           "dest_team": dest, "dest_name": top.get("team_name", dest),
           "offer_M": round(float(top["offer_M"]), 1),
           "is_resign": bool(top["is_incumbent"]),
           "tool": top.get("tool", ""), "status": "UFA"}
    projected.append(rec)
    listing.append({**row, "projected": True, "dest_team": dest,
                    "offer_M": rec["offer_M"], "is_resign": rec["is_resign"]})

listing.sort(key=lambda x: (x["team"], -(x["value_M"] or 0)))
projected.sort(key=lambda x: -x["offer_M"])
OUT = ROOT / "cache" / "fa_extra_v1.json"
OUT.write_text(json.dumps({"listing": listing, "projected": projected}, indent=1))
print(f"  {len(projected)} rotation FAs projected into rosters, {len(listing)} total listed")
for p in projected:
    arrow = "re-signs" if p["is_resign"] else "->"
    print(f"    {p['player']:24s} {p['team']} {arrow} {p['dest_team']:4s} ${p['offer_M']:.1f}M ({p['tool']})")
print(f"wrote {OUT}")
