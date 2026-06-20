#!/usr/bin/env python3
"""Dump each player's authoritative current team = where he actually played this
season (the stats feed's Team), with data/team_corrections.csv layered on top.

The contract-end scraper and the contract-features layer can carry a STALE team
(e.g. a mock-trade rumor put Rui Hachimura on POR though he never left LAL), which
breaks incumbent / Bird-rights logic. The season stats team is the reliable source;
this writes it to cache/current_teams_v1.json for the export + builds to trust.

Usage:  python -u scripts/dump_current_teams.py
"""
import csv
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from utils import normalize  # noqa: E402

SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "cp", "__file__": str((ROOT / "pages" / "Contract_Predictor.py").resolve())}
exec(compile("".join(SRC[:cut]), "cp", "exec"), ns)

full = ns["build_ranked_projected"](ns["CURRENT_SEASON"])
ABBR = {"PHO": "PHX", "CHO": "CHA", "BRK": "BKN", "NOH": "NOP"}
teams = {normalize(r["Player"]): ABBR.get(str(r["Team"]), str(r["Team"])) for _, r in full.iterrows()}

# manual overrides win (for the rare case even the stats team is wrong)
corr = ROOT / "data" / "team_corrections.csv"
n_corr = 0
if corr.exists():
    for r in csv.DictReader([l for l in corr.read_text().splitlines() if not l.lstrip().startswith("#")]):
        if r.get("player") and r.get("team"):
            teams[normalize(r["player"])] = ABBR.get(r["team"].strip(), r["team"].strip()); n_corr += 1

OUT = ROOT / "cache" / "current_teams_v1.json"
OUT.write_text(json.dumps(teams))
print(f"  {len(teams)} players | {n_corr} manual corrections applied")
print(f"wrote {OUT}")
