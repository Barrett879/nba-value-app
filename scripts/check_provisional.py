"""Do the hand-entered box scores add up, and are they still needed?

    /opt/anaconda3/bin/python3 -u scripts/check_provisional.py

Read-only. Prints PASS/FAIL per team, exits non-zero on any failure.

WHY THIS EXISTS. data/provisional_box_scores.csv is typed in by hand off a
screenshot, which is the least trustworthy input this repo has. A single
misread digit becomes a Barrett Score on the live site with nothing to
distinguish it from a real one.

The box score itself carries the answer key: every published table prints a
TEAM totals row, and the player column sums must equal it. That is arithmetic,
not judgement, so it catches a misread digit and a mis-aligned column the same
way. Rows with player=TEAM hold those published totals and are the thing the
player rows are checked against; they are never merged into the stats.

It also reports whether the real feed has caught up, because the whole file is
supposed to be temporary and nothing else will remind anyone.
"""
from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FP = ROOT / "data" / "provisional_box_scores.csv"
SUM_COLS = ["pts", "fga", "fta", "reb", "ast", "tov", "stl", "blk", "oreb", "dreb", "pf"]

results: list[tuple[str, str, str]] = []


def record(state, name, detail):
    results.append((state, name, detail))


if not FP.exists():
    print("\n  PROVISIONAL BOX SCORES    none on file, nothing to check\n")
    sys.exit(0)

with open(FP, newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(
        l for l in fh if l.strip() and not l.lstrip().startswith("#")))

by_team = collections.defaultdict(list)
for r in rows:
    by_team[(r.get("team") or "").strip()].append(r)


def num(r, k):
    try:
        return float(r.get(k) or 0)
    except ValueError:
        return float("nan")


for team, grp in sorted(by_team.items()):
    totals_row = [r for r in grp if (r.get("player") or "").strip().upper() == "TEAM"]
    players = [r for r in grp if (r.get("player") or "").strip().upper() != "TEAM"]
    if not totals_row:
        record("FAIL", team, "no TEAM totals row, so nothing can be checked against")
        continue
    t = totals_row[0]
    bad = []
    for c in SUM_COLS:
        got = sum(num(r, c) for r in players)
        want = num(t, c)
        if want != want:                       # NaN: not published for this column
            continue
        if abs(got - want) > 1e-6:
            bad.append(f"{c}: players sum to {got:g}, TEAM row says {want:g}")
    # reb must equal oreb+dreb on every single row, which catches a column shift
    shifted = [r["player"] for r in players
               if abs(num(r, "reb") - (num(r, "oreb") + num(r, "dreb"))) > 1e-6]
    if shifted:
        bad.append(f"reb != oreb+dreb for {shifted[:4]}")
    record("FAIL" if bad else "PASS", f"{team} ({len(players)} players)",
           "; ".join(bad) if bad else
           f"every column sums to the published TEAM row, and reb = oreb + dreb throughout")

# ── is it still needed? ──────────────────────────────────────────────────────
try:
    from utils import SEASONS, fetch_bref_player_stats
    import pandas as pd
    live = fetch_bref_player_stats(SEASONS[0])
    for team, grp in sorted(by_team.items()):
        gp_after = next((r.get("team_gp_after") for r in grp if r.get("team_gp_after")), None)
        have = live[live["TEAM_ABBREVIATION"] == team] if len(live) else live
        src = float(pd.to_numeric(have["GP"], errors="coerce").max()) if len(have) else 0.0
        if gp_after and src >= float(gp_after):
            record("WARN", f"{team} superseded",
                   f"the feed now reports GP {src:g} >= {gp_after}, so these rows are "
                   f"being ignored and should be deleted")
        else:
            record("PASS", f"{team} still needed",
                   f"feed reports GP {src:g}, provisional covers up to {gp_after}")
except Exception as e:                                               # noqa: BLE001
    record("WARN", "supersession check", f"could not query the feed: {e}")


print()
print(f"  PROVISIONAL BOX SCORES    {len(rows)} rows, {len(by_team)} team block(s)")
print()
width = max(len(n) for _, n, _ in results)
for state, name, detail in results:
    print(f"  [{state:4}] {name:{width}}  {detail}")
fails = [r for r in results if r[0] == "FAIL"]
warns = [r for r in results if r[0] == "WARN"]
print()
print(f"  {'FAIL: ' + str(len(fails)) + ' block(s) do not add up' if fails else 'PASS: every block matches its published totals' + (f', {len(warns)} warning(s)' if warns else '')}")
print()
sys.exit(1 if fails else 0)
