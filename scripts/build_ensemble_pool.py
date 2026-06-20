#!/usr/bin/env python3
"""Pre-build a shared, pre-priced free-agent pool so an ENSEMBLE of projection
models can each run on identical inputs (no model reload per model).

Writes cache/fa_ensemble_pool.json:
  cands     : [{name, status, value_M, barrett, pos, age, team}]  (the priced FA pool)
  rosters   : records of the live per-player roster table (team/player/pos/barrett)
  landscape : records of the team landscape (team/team_name/cap_space_M/exception_M/timeline)
  teamnames : {abbr: full name}
  cap_M     : salary cap

Each ensemble model reads this + cache/fa_board_v1.json (team needs/committed) and
applies its own philosophy via team_suitors.rank_suitors. Re-run after the model or
current-season data changes.  Usage:  python -u scripts/build_ensemble_pool.py
"""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from utils import normalize, DEFAULT_MIN_THRESHOLD, option_opt_in_prob, OPTION_OPT_IN_THRESHOLD, classify_fa_status  # noqa: E402
import team_suitors as ts  # noqa: E402

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


import csv as _csv
_TEAM_CORR = {}
_tc = ROOT / "data" / "team_corrections.csv"
if _tc.exists():
    for _r in _csv.DictReader([l for l in _tc.read_text().splitlines() if not l.lstrip().startswith("#")]):
        if _r.get("player") and _r.get("team"):
            _TEAM_CORR[normalize(_r["player"])] = _r["team"].strip()
_ABBR = {"PHO": "PHX", "CHO": "CHA", "BRK": "BKN", "NOH": "NOP"}
def _team_of(name, stats_team, stale_team):                # season team of record (see build_fa_board)
    t = _TEAM_CORR.get(normalize(name)) or str(stats_team or "") or str(stale_team or "")
    return _ABBR.get(t, t)


def fa_status(name):
    return classify_fa_status(name, fmt_nc(name, nc), rookie, CUR)


_OPT_OUT = set()
_ovr_oo = ROOT / "data" / "fa_sim_overrides.csv"
if _ovr_oo.exists():
    for _r in _csv.DictReader(_ovr_oo.read_text().splitlines()):
        if (_r.get("action") or "").strip().lower() == "opt_out" and _r.get("player"):
            _OPT_OUT.add(normalize(_r["player"]))
print("pricing the free-agent pool ...")
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
    value_M = round(float(pcv(f)) / 1e6, 1)
    age = int(f.get("age") or 0)
    opt_M = float((nc.get(normalize(name)) or {}).get("salary") or 0) / 1e6
    _optout = normalize(name) in _OPT_OUT
    if st_ == "Player Option" and not _optout and option_opt_in_prob(opt_M, value_M, age) >= OPTION_OPT_IN_THRESHOLD:
        continue
    if st_ == "Team Option":                             # team's call, not a FA market - kept or dropped via corrections
        continue
    if _optout:
        st_ = "UFA"
    cands.append({"name": name, "status": st_, "value_M": value_M,
                  "barrett": round(float(f["barrett_score"]), 1),
                  "pos": ts.resolve_position(name, f.get("position_detailed") or "", pos2k),
                  "age": age, "team": str(f.get("current_team") or r["Team"])})

teamnames = dict(zip(LAND["team"].astype(str), LAND["team_name"].astype(str)))
out = {
    "cap_M": CAP_M,
    "cands": cands,
    "rosters": ROST.to_dict("records"),
    "landscape": LAND.to_dict("records"),
    "teamnames": teamnames,
}
OUT = ROOT / "cache" / "fa_ensemble_pool.json"
OUT.write_text(json.dumps(out))
print(f"  {len(cands)} free agents | {len(ROST)} roster rows | {len(LAND)} teams")
print(f"wrote {OUT}")
