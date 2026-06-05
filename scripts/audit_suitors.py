"""League-wide sanity sweep of the Likely Suitors engine. Runs rank_suitors for
every qualified player and flags violations of basic invariants, so nonsense
can't hide in the long tail.

Usage:  python -u scripts/audit_suitors.py
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

SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "cp", "__file__": str((ROOT / "pages" / "Contract_Predictor.py").resolve())}
exec(compile("".join(SRC[:cut]), "cp", "exec"), ns)

CUR = ns["CURRENT_SEASON"]
CONTRACT = ns.get("CONTRACT_SEASON", CUR)
gpf, pc = ns["get_player_features"], ns["predict_contract"]
DEFAULT_MIN = DEFAULT_MIN_THRESHOLD

full = ns["build_ranked_projected"](CUR).copy()
pos2k = ts.load_player_positions()
full["pos"] = full["Player"].map(lambda n: ts.resolve_position(n, "", pos2k))
nc = ns["fetch_next_year_contracts"](ns["season_to_espn_year"](CUR), cache_v=7)
payroll = pd.DataFrame({"team": full["Team"].astype(str), "player": full["Player"].astype(str)})
LAND = ts.apply_real_cap(ts.load_team_landscape(),
                         ts.compute_cap_space(payroll, nc, ns["SALARY_CAP_M"].get(CONTRACT, 165.0)))
ROST = ts.build_rosters(full)
tags = {"rfa": "RFA", "player_option": "player option", "team_option": "team option"}
status = {n: tags[(v or {}).get("type")] for n, v in nc.items() if (v or {}).get("type") in tags}

qualified = full[full["total_min"] >= DEFAULT_MIN].sort_values("score_rank")
flags = {k: [] for k in ["OFFER_OVER_VALUE", "DUP_TEAM", "EMPTY_MARKET",
                         "INCUMBENT_MISSING", "BAD_DISPLACE"]}
# Aging-star spot-check: their suitors should skew title/playoff, not rebuild.
WATCH = {"LeBron James", "Kevin Durant", "James Harden", "Chris Paul",
         "Stephen Curry", "Jimmy Butler"}
tl_of = dict(zip(LAND["team"].astype(str), LAND["timeline"].astype(str)))
watch_out = {}
n_done = n_with = 0
for rank, (_, row) in enumerate(qualified.iterrows(), 1):
    name = str(row["Player"])
    try:
        f = gpf(name, CUR)
        if not f:
            continue
        pm = float(pc(f)["predicted"]) / 1e6
        rost = ROST[ROST["player"].map(normalize) != normalize(name)]
        pos = ts.resolve_position(name, f.get("position_detailed") or "", pos2k)
        s = ts.rank_suitors(pm, float(f["barrett_score"]), pos, rost, LAND, n=6,
                            incumbent_team=f.get("current_team"), age=f.get("age"),
                            is_rfa=(status.get(normalize(name)) == "RFA"),
                            skill_fit=None, fa_status=status)
    except Exception as e:
        flags.setdefault("ERROR", []).append(f"{name}: {e}")
        continue
    n_done += 1
    bar = float(f["barrett_score"])
    sal = float(f.get("salary") or 0)
    teams = [x["team"] for x in s]
    if s:
        n_with += 1
    if name in WATCH:
        watch_out[name] = (int(f.get("age") or 0),
                           [(x["team"], tl_of.get(x["team"], "?"), round(x["offer_M"])) for x in s])
    for x in s:
        if x["offer_M"] > pm + 1.0:
            flags["OFFER_OVER_VALUE"].append(f"{name}: {x['team']} ${x['offer_M']:.0f}M > value ${pm:.0f}M")
        if (x.get("displaces_score") is not None and "start over" in x.get("reason", "")
                and float(x["displaces_score"]) >= bar):
            flags["BAD_DISPLACE"].append(f"{name} ({bar:.1f}) -> {x['reason']}")
    if len(teams) != len(set(teams)):
        flags["DUP_TEAM"].append(f"{name}: {teams}")
    if not s and sal > 0 and pm >= 5 and rank <= 250:
        flags["EMPTY_MARKET"].append(f"{name} (rank {rank}, ${pm:.0f}M, sal ${sal/1e6:.0f}M)")
    if rank <= 80 and f.get("current_team") and not any(x.get("is_incumbent") for x in s):
        flags["INCUMBENT_MISSING"].append(f"{name} (rank {rank}, {f.get('current_team')})")

print(f"swept {n_done} players · {n_with} have suitors · {n_done - n_with} empty\n")
for k, v in flags.items():
    print(f"=== {k}: {len(v)} ===")
    for line in v[:25]:
        print("  ", line)

print("\n=== aging-star suitor spot-check (age 32+ should skew title/playoff) ===")
for k in sorted(watch_out):
    age, brd = watch_out[k]
    print(f"  {k} (age {age}): " + ", ".join(f"{t}[{tl}] ${o}M" for t, tl, o in brd))
