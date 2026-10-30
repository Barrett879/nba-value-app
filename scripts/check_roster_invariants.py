"""Roster legality and integrity checks for data/master_roster.csv.

    /opt/anaconda3/bin/python3 -u scripts/check_roster_invariants.py

Read-only. Prints PASS/FAIL per check, exits non-zero on any failure.

RUN THIS EVERY VERIFICATION ROUND. On 2026-10-19 Barrett asked whether the
roster updates were actually right, and the answer was no: four teams carried 16
standard contracts, which the CBA never allows, one carried 13, and three
carried four two-ways against a maximum of three. Nothing in the repo would have
told us.

The cause is worth remembering. That round processed the five waivers that
created DEAD MONEY and skipped roughly a hundred Exhibit 10 camp cuts on the
grounds that they were cap-neutral. They are cap-neutral. They are not
ROSTER-neutral, and conflating those two is what left non-players on team
sheets. A cut always changes the roster even when it changes no money.

A missing row is as bad as a wrong one. app.py treats absence from this file as
"not under contract, so show him as available", so a two-way the file never knew
about is offered on the board as a free agent. Kevin McCullar Jr. was doing
exactly that.

Counts are checked against the CBA limits, which do not move. Comparing against
a published league-wide count (Hoops Rumors keeps one) catches the rest, but
that needs a human with a browser, so it is not automated here.
"""
from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import NameIndex, normalize  # noqa: E402

# CBA: 15 standard maximum, 14 minimum during the regular season (a team may sit
# at 13 for up to two weeks, and 28 days across a season, so 13 is a WARN not a
# FAIL). Two-ways cap at 3 and do not count toward the 15.
MAX_STD, MIN_STD, HARD_MIN_STD, MAX_TW = 15, 14, 13, 3

results: list[tuple[str, str, str]] = []


def record(state: str, name: str, detail: str) -> None:
    results.append((state, name, detail))


def rows_of(path) -> list:
    with open(path) as fh:
        return list(csv.DictReader(
            l for l in fh if l.strip() and not l.lstrip().startswith("#")))


with open(ROOT / "data" / "master_roster.csv") as fh:
    rows = list(csv.DictReader(
        l for l in fh if l.strip() and not l.lstrip().startswith("#")))

std, tw, fa = collections.Counter(), collections.Counter(), collections.Counter()
for r in rows:
    k = (r.get("kind") or "").strip()
    if k == "standard":
        std[r["team"]] += 1
    elif k == "two_way":
        tw[r["team"]] += 1
    elif k == "free_agent":
        fa[r["team"]] += 1

teams = sorted(set(std) | set(tw) | set(fa))

# ── 1. field widths ──────────────────────────────────────────────────────────
bad_width = [r.get("player") for r in rows if len(r) != 7]
record("FAIL" if bad_width else "PASS", "row widths",
       f"malformed: {bad_width}" if bad_width else f"all {len(rows)} rows have 7 fields")

# ── 2. thirty teams ──────────────────────────────────────────────────────────
record("PASS" if len(teams) == 30 else "FAIL", "team count",
       f"{len(teams)} distinct teams" + ("" if len(teams) == 30 else " (expected 30)"))

# ── 3. standard contracts within the CBA limits ──────────────────────────────
over = {t: std[t] for t in teams if std[t] > MAX_STD}
under = {t: std[t] for t in teams if std[t] < HARD_MIN_STD}
at13 = {t: std[t] for t in teams if std[t] == HARD_MIN_STD}
if over:
    record("FAIL", f"standard <= {MAX_STD}",
           "OVER THE LIMIT, never legal: " + ", ".join(f"{t}={n}" for t, n in over.items()))
else:
    record("PASS", f"standard <= {MAX_STD}", "no team over the maximum")
if under:
    record("FAIL", f"standard >= {HARD_MIN_STD}",
           "below even the temporary floor: " + ", ".join(f"{t}={n}" for t, n in under.items()))
elif at13:
    record("WARN", f"standard >= {MIN_STD}",
           "at 13, legal only for two weeks: " + ", ".join(f"{t}={n}" for t, n in at13.items()))
else:
    record("PASS", f"standard >= {MIN_STD}", "every team at 14 or 15")

# ── 4. two-ways within the limit ─────────────────────────────────────────────
tw_over = {t: tw[t] for t in teams if tw[t] > MAX_TW}
record("FAIL" if tw_over else "PASS", f"two-way <= {MAX_TW}",
       ("OVER THE LIMIT: " + ", ".join(f"{t}={n}" for t, n in tw_over.items()))
       if tw_over else "no team over three two-ways")

# ── 5. a team with no two-ways at all is almost certainly missing rows ───────
# Not illegal, but every team normally fills all three slots, and 0 is what a
# wholesale gap looks like: NYK read 0 while carrying three.
tw_zero = [t for t in teams if tw[t] == 0]
record("WARN" if tw_zero else "PASS", "two-way coverage",
       (f"{tw_zero} carry none, which usually means rows are missing rather than empty slots")
       if tw_zero else "every team has at least one two-way on file")

# ── 6. nobody listed twice ───────────────────────────────────────────────────
seen = collections.Counter(normalize(r["player"]) for r in rows if r.get("player"))
dupes = {k: v for k, v in seen.items() if v > 1}
record("FAIL" if dupes else "PASS", "no duplicate players",
       f"{len(dupes)} duplicated: {list(dupes)[:6]}" if dupes else f"{len(seen)} distinct players")

# ── 7. a $0 free agent with no est value is DROPPED by build_redraft_pool ────
import re  # noqa: E402
missing_est = [r["player"] for r in rows
               if (r.get("kind") or "").strip() == "free_agent"
               and float(r.get("salary_M") or 0) <= 0
               and not re.search(r"est\s+value\s*\$([0-9.]+)M", r.get("notes") or "", re.I)]
record("WARN" if missing_est else "PASS", "free agents priced for the pool",
       (f"{len(missing_est)} have no 'est value' phrase, so build_redraft_pool drops them. "
        f"Fine for fringe players with no model value; add the model pcv for anyone who should "
        f"appear. e.g. {missing_est[:4]}")
       if missing_est else "every $0 free agent carries an est value")

# ── 8. a player cannot be under contract and a free agent at once ────────────
by_name: dict[str, set[str]] = collections.defaultdict(set)
for r in rows:
    by_name[normalize(r["player"])].add((r.get("kind") or "").strip())
both = [n for n, ks in by_name.items() if "free_agent" in ks and ks & {"standard", "two_way"}]
record("FAIL" if both else "PASS", "no player both signed and free",
       f"{both[:6]}" if both else "no contradictory rows")


# ── 8b. data/retired_players.csv ────────────────────────────────────────────
# Two ways this file goes wrong, both silent. A row that fails validation (no
# date, no source) is DROPPED by utils.retired_players, so the player goes on
# being offered as a free agent -- the exact failure the file exists to prevent.
# And a name nobody can resolve does nothing at all, which looks identical to
# the player not being retired.
from utils import retired_players, NameIndex, RETIRED_PATH  # noqa: E402
_ret = retired_players()
_raw = 0
if RETIRED_PATH.exists():
    with open(RETIRED_PATH) as fh:
        _raw = sum(1 for r in csv.DictReader(l for l in fh if l.strip()
                                             and not l.lstrip().startswith("#"))
                   if (r.get("player") or "").strip())
record("FAIL" if _raw != len(_ret) else "PASS", "retirements carry date + source",
       (f"{_raw - len(_ret)} of {_raw} row(s) dropped for a missing or malformed "
        f"date/source, so they do nothing") if _raw != len(_ret)
       else (f"all {_raw} row(s) valid: {', '.join(sorted(_ret))}" if _ret
             else "no retirements on file"))

_mr_idx = NameIndex({r["player"]: True for r in rows})
_unresolved = [n for n in _ret if not _mr_idx.get(n)]
record("WARN" if _unresolved else "PASS", "retirements resolve against a roster row",
       (f"{_unresolved} have no master_roster row. NOT an error by itself -- a "
        f"retired veteran is exactly the player who falls out of the roster "
        f"files, and the free-agent surfaces join on the BOARD's names, not "
        f"this one. Check the spelling matches the board.")
       if _unresolved else f"all {len(_ret)} resolve")

_ret_rostered = [n for n in _ret
                 if by_name[normalize(n)] & {"standard", "two_way"}]
record("FAIL" if _ret_rostered else "PASS", "nobody retired and rostered",
       f"{_ret_rostered}" if _ret_rostered else "no contradictory rows")


# ── 9. the hand-set override files actually resolve ──────────────────────────
# A misspelled name in either file does NOTHING, silently: the lookup misses and
# the player keeps his scraped position or his score-ranked slot. Nobody would
# notice except by staring at the depth chart and wondering why an instruction
# did not take.
SLOT_FIRST = {"PG", "SG", "SF", "PF", "C", "G", "F"}
roster_idx = NameIndex({r["player"]: r for r in rows if r.get("player")})
for fname, poscol in (("position_overrides.csv", "pos"), ("depth_order.csv", None)):
    fp = ROOT / "data" / fname
    if not fp.exists():
        record("WARN", f"{fname}", "absent")
        continue
    unresolved, badpos = [], []
    for r in rows_of(fp):
        nm = (r.get("player") or "").strip()
        if not nm:
            continue
        if roster_idx.get(nm) is None:
            unresolved.append(nm)
        if poscol:
            first = ((r.get(poscol) or "").split("/")[0] or "").strip().upper()
            if first not in SLOT_FIRST:
                badpos.append(f"{nm}={r.get(poscol)!r}")
            alt = ((r.get("alt_pos") or "").split("/")[0] or "").strip().upper()
            if (r.get("alt_pos") or "").strip() and alt not in SLOT_FIRST:
                badpos.append(f"{nm} alt={r.get('alt_pos')!r}")
            dep = (r.get("alt_when_out") or "").strip()
            if dep and roster_idx.get(dep) is None:
                unresolved.append(f"{dep} (alt_when_out for {nm})")
    detail = []
    if unresolved:
        detail.append("unresolved names: " + ", ".join(unresolved[:5]))
    if badpos:
        detail.append("unmappable positions: " + ", ".join(badpos[:5]))
    record("FAIL" if detail else "PASS", f"{fname} resolves",
           "; ".join(detail) if detail else
           f"every player and position in {fname} resolves against master_roster")


# ── report ───────────────────────────────────────────────────────────────────
print()
print(f"  ROSTER INVARIANTS    {sum(std.values())} standard, {sum(tw.values())} two-way, "
      f"{sum(fa.values())} free agents, {len(rows)} rows")
print()
width = max(len(n) for _, n, _ in results)
for state, name, detail in results:
    print(f"  [{state:4}] {name:{width}}  {detail}")

print()
print("  per team (standard + two-way):")
line = []
for t in teams:
    mark = "!" if (std[t] > MAX_STD or std[t] < MIN_STD or tw[t] > MAX_TW) else " "
    line.append(f"{t} {std[t]:>2}+{tw[t]}{mark}")
for i in range(0, len(line), 6):
    print("    " + "   ".join(line[i:i + 6]))

fails = [r for r in results if r[0] == "FAIL"]
warns = [r for r in results if r[0] == "WARN"]
print()
if fails:
    print(f"  FAIL: {len(fails)} invariant(s) broken. The roster is not in a legal state.")
else:
    print(f"  PASS: every hard invariant holds" + (f", {len(warns)} warning(s)" if warns else ""))
print()
sys.exit(1 if fails else 0)
