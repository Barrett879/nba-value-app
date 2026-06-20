#!/usr/bin/env python3
"""Run an ENSEMBLE of free-agency projection models, each with a distinct
philosophy, on the shared pre-priced pool (cache/fa_ensemble_pool.json), and write
a per-player comparison to cache/fa_ensemble_v1.json.

Four lenses (a deliberate movement gradient from "most likely" to "max churn"):
  Consensus    - the validated independent best-guess (board top-1, ~90% stay)
  Follow Money - a player is lured out only by a STRONG outside bid (>=85% of value)
  Win-Now      - starter-level FAs gravitate to big markets / contenders, soft cap
  Player Mkt   - every unrestricted FA takes his best OUTSIDE offer (a mover's market)

NOTE (real finding): the incumbent-weight knob barely moves the top pick, and a naive
"highest bidder" model collapses onto Consensus too, because the suitor engine gives
the incumbent a full-value Bird-rights offer (so for ~90% of players he is BOTH the
top-ranked suitor AND the highest bidder). Movement therefore has to be modeled
structurally, not by re-weighting. Run after build_ensemble_pool.
Usage:  python -u scripts/build_fa_ensemble.py
"""
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import team_suitors as ts  # noqa: E402

d = json.loads((ROOT / "cache" / "fa_ensemble_pool.json").read_text())
ROST = pd.DataFrame(d["rosters"])
LAND = pd.DataFrame(d["landscape"])
TEAMNAME = d["teamnames"]
timeline = {r["team"]: str(r.get("timeline", "")).lower() for r in d["landscape"]}
MARKET = {"LAL", "LAC", "NYK", "BKN", "CHI", "GSW", "MIA", "PHI", "BOS", "DAL"}
CONTEND = ("title", "contender", "playoff")
cands = d["cands"]
CAP_ADDS = 5                       # soft cap: one team can add at most this many FAs in a model
MIN_M = 2.3

MODELS = ["Consensus", "Follow the Money", "Win-Now / Big Market", "Player's Market"]
LURE = 0.85          # Follow the Money: a player leaves only if an OUTSIDE team bids >= this fraction of his value


def board_for(c, **args):
    return ts.rank_suitors(c["value_M"], c["barrett"], c["pos"], ROST, landscape=LAND, n=10,
                           incumbent_team=c["team"], age=c["age"], is_rfa=(c["status"] == "RFA"), **args)


def run(model):
    out, added = {}, defaultdict(int)
    # aggressive models place best-first so the soft team cap binds sensibly
    order = sorted(cands, key=lambda c: -c["value_M"]) if model in ("Follow the Money", "Win-Now / Big Market") else cands
    for c in order:
        nm, inc = c["name"], c["team"]
        args = {"blend_dest": False} if model in ("Follow the Money", "Player's Market") else {}
        b = board_for(c, **args)
        if not b:
            out[nm] = (inc, round(min(c["value_M"], MIN_M), 1)); continue
        if model == "Consensus":
            e = b[0]
        elif model == "Follow the Money":
            # leave only for a strong outside bid (a team that really pays up); RFAs stay
            outside = [x for x in b if not x["is_incumbent"] and added[x["team"]] < CAP_ADDS]
            best = max(outside, key=lambda x: x["offer_M"]) if outside else None
            e = best if (best and c["status"] != "RFA" and best["offer_M"] >= LURE * c["value_M"]) else b[0]
        elif model == "Win-Now / Big Market":
            if c["value_M"] >= 8:
                cs = [x for x in b if (x["team"] in MARKET or timeline.get(x["team"]) in CONTEND)
                      and not x["is_incumbent"] and added[x["team"]] < CAP_ADDS]
                e = cs[0] if cs else b[0]
            else:
                e = b[0]
        else:  # Player's Market
            e = next((x for x in b if not x["is_incumbent"]), b[0]) if c["status"] == "UFA" else b[0]
        if not e["is_incumbent"]:
            added[e["team"]] += 1
        out[nm] = (e["team"], round(float(e["offer_M"]), 1))
    return out


maps = {m: run(m) for m in MODELS}
print("model           re-sign%   movers")
for m in MODELS:
    stay = sum(1 for c in cands if maps[m][c["name"]][0] == c["team"])
    print(f"  {m:22s} {round(100*stay/len(cands),1):>5}%   {len(cands)-stay}")

players = []
for c in cands:
    nm = c["name"]
    teams = {m: maps[m][nm][0] for m in MODELS}
    # Anchor on the VALIDATED Consensus projection; agreement = how many lenses confirm
    # it (4 = even the aggressive scenarios keep him there = rock-solid; 1 = only the
    # base model holds, every lens moves him = volatile). NOT a modal vote, which could
    # override the accurate base model with where the stress-test lenses happen to cluster.
    base = teams["Consensus"]
    agree = sum(1 for m in MODELS if teams[m] == base)
    players.append({
        "player": nm, "incumbent": c["team"], "pos": c["pos"], "value_M": c["value_M"],
        "status": c["status"],
        "picks": {m: {"team": teams[m], "offer_M": maps[m][nm][1]} for m in MODELS},
        "consensus": base, "agreement": agree,
    })
players.sort(key=lambda p: -p["value_M"])
OUT = ROOT / "cache" / "fa_ensemble_v1.json"
OUT.write_text(json.dumps({"models": MODELS, "teamnames": TEAMNAME, "players": players}))
locks = sum(1 for p in players if p["agreement"] >= len(MODELS))
splits = sum(1 for p in players if p["agreement"] <= 2)
print(f"\n{len(players)} FAs | {locks} all-{len(MODELS)}-agree | {splits} genuine toss-ups (<=2 of {len(MODELS)})")
print(f"wrote {OUT}")
