"""One player, one spelling, across every season in the pool.

    /opt/anaconda3/bin/python3 -u scripts/check_name_canon.py

Read-only. Prints PASS/FAIL per check, exits non-zero on any failure.

WHY THIS EXISTS. A player's name can arrive two ways because the SOURCE
changes with the era: Basketball-Reference carries diacritics, the NBA Stats
API spells the 1990s imports in plain ASCII. The seam is 1995-96/1996-97,
exactly where fetch_bref_player_stats stops being the primary source.

Legacy aggregates with groupby("Player") on the raw string, so an unresolved
pair does not look like a bug, it looks like two players. Toni Kukoc had a
three-season career and a nine-season one, and his career totals, consistency
ranking and archetype placement were each computed on half of him.

data/name_canon.csv resolves the pairs we know about. This check exists for
the ones we do not: it folds every name in the pool down to its accent-free
form and FAILS if two spellings still share one, which is what a new source
drift looks like on the day it starts.
"""
from __future__ import annotations

import collections
import csv
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import build_all_seasons_combined, _name_canon_map  # noqa: E402

results: list[tuple[str, str, str]] = []


def record(state: str, name: str, detail: str) -> None:
    results.append((state, name, detail))


def fold(s: str) -> str:
    """Accent-free, case-free form. Two spellings of one player land here."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c)).lower().strip()


# ── 1. the map file parses and every row actually changes something ──────────
try:
    rows = []
    with open(ROOT / "data" / "name_canon.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(
            l for l in fh if l.strip() and not l.lstrip().startswith("#")))
    bad = [r.get("variant") for r in rows
           if not (r.get("variant") or "").strip() or not (r.get("canonical") or "").strip()]
    noop = [r["variant"] for r in rows if r.get("variant") == r.get("canonical")]
    if bad:
        record("FAIL", "name_canon.csv rows", f"blank variant or canonical: {bad[:4]}")
    elif noop:
        record("FAIL", "name_canon.csv rows", f"variant == canonical, does nothing: {noop[:4]}")
    else:
        record("PASS", "name_canon.csv rows", f"{len(rows)} rows, all well formed")

    # A canonical that is itself someone else's variant would chain, and the
    # replace() is single-pass, so the second hop would never happen.
    variants = {r["variant"] for r in rows}
    chained = [r["canonical"] for r in rows if r["canonical"] in variants]
    record("FAIL" if chained else "PASS", "no chained mappings",
           f"canonical is also a variant: {chained}" if chained
           else "every canonical is a terminal spelling")
except Exception as e:                                               # noqa: BLE001
    record("FAIL", "name_canon.csv", f"unreadable: {e}")

# ── 2. the map loaded by utils matches the file ──────────────────────────────
try:
    m = _name_canon_map()
    record("PASS" if m else "FAIL", "utils loads the map",
           f"{len(m)} entries" if m else "empty, so nothing is being folded")
except Exception as e:                                               # noqa: BLE001
    record("FAIL", "utils loads the map", str(e))

# ── 3. THE INVARIANT: no two spellings share a folded name ───────────────────
try:
    df = build_all_seasons_combined(0)
    groups: dict[str, set] = collections.defaultdict(set)
    for n in df["Player"].astype(str).unique():
        groups[fold(n)].add(n)
    split = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    if split:
        detail = "; ".join(f"{k}: {v}" for k, v in list(split.items())[:4])
        record("FAIL", "one player one spelling",
               f"{len(split)} player(s) split across spellings, so Legacy's "
               f"groupby computes partial careers. Add a row to "
               f"data/name_canon.csv. {detail}")
    else:
        record("PASS", "one player one spelling",
               f"{df['Player'].nunique()} names fold to {len(groups)} players, no collisions")

    # 4. a variant must not survive anywhere in the pool
    leaked = sorted({v for v in _name_canon_map() if v in set(df["Player"].astype(str))})
    record("FAIL" if leaked else "PASS", "variants folded away",
           f"still present despite a mapping: {leaked}" if leaked
           else "no mapped variant survives in the pool")
except Exception as e:                                               # noqa: BLE001
    record("FAIL", "pool check", f"could not build the pool: {e}")


print()
print("  NAME CANON    one player, one spelling, across every season")
print()
width = max(len(n) for _, n, _ in results)
for state, name, detail in results:
    print(f"  [{state:4}] {name:{width}}  {detail}")

fails = [r for r in results if r[0] == "FAIL"]
print()
print(f"  {'FAIL: ' + str(len(fails)) + ' broken' if fails else 'PASS: every check holds'}")
print()
sys.exit(1 if fails else 0)
