"""Audit the player -> agent -> agency map (data/player_agents.csv + override).

A read-only sanity sweep to run before trusting the Agency Database page:
  1. coverage of the current active NBA roster
  2. agent rows whose player name does NOT match the app roster - usually a
     name-format mismatch to pin in data/player_agents_override.csv (or a
     non-NBA / international client, which is harmless)
  3. active players with no agent yet (coverage gaps, worst offenders by Barrett)
  4. agency-name spelling collisions ("Klutch Sports Management" vs "...Group")
     that would split one real agency into two
  5. agents-per-player distribution (multi-agent is expected, just informational)

Run:  python scripts/audit_agents.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agency_db import (  # noqa: E402
    load_player_agents, build_index, match, has_agent,
)

# Generic words stripped when testing whether two agency spellings are "the same".
_AGENCY_STOP = {
    "sports", "management", "entertainment", "group", "llc", "inc", "and",
    "the", "agency", "company", "co", "partners", "international", "global",
    "associates", "enterprises",
}


def agency_key(name: str) -> str:
    """A loose key so near-identical agency spellings collide on purpose."""
    s = re.sub(r"[&.,'/-]", " ", name.lower())
    toks = [t for t in s.split() if t and t not in _AGENCY_STOP]
    return " ".join(toks[:2])


def _roster() -> tuple[list, str]:
    """[(player_name, barrett_score), ...] for the current active roster, or []."""
    try:
        from utils import build_ranked_projected, SEASONS
        df = build_ranked_projected(SEASONS[0])
        if df is None or df.empty:
            return [], SEASONS[0]
        return [(str(p), float(b)) for p, b in
                zip(df["Player"], df["barrett_score"].fillna(0.0))], SEASONS[0]
    except Exception as e:                      # audit still useful without the roster
        print(f"  (could not build roster: {type(e).__name__} {e})")
        return [], "?"


def main() -> int:
    amap = load_player_agents()
    n = len(amap)
    agents = {a["agent"] for r in amap.values() for a in r["agents"]}
    agencies = {a["agency"] for r in amap.values() for a in r["agents"] if a["agency"]}
    print(f"player_agents: {n} players | {len(agents)} agents | {len(agencies)} agencies\n")

    roster, season = _roster()
    if roster:
        index = build_index(amap)               # exact same tiers the page uses
        have, miss, matched = [], [], set()
        for nm, b in roster:
            rec = match(nm, amap, index)
            if has_agent(rec):
                have.append((nm, b))
                matched.add(id(rec))
            else:
                miss.append((nm, b))
        pct = len(have) / len(roster) * 100 if roster else 0
        print(f"[1] COVERAGE ({season}): {len(have)}/{len(roster)} active players "
              f"have an agent ({pct:.1f}%)\n")

        # [2] agent rows never matched by an active player (non-NBA or a name the
        #     tiered matcher still cannot bridge, e.g. a nickname)
        unmatched = sorted(r["name"] for r in amap.values() if id(r) not in matched)
        print(f"[2] {len(unmatched)} agent rows match no active player "
              f"(non-NBA, or an unbridged nickname). First 25:")
        for nm in unmatched[:25]:
            print(f"      {nm}")
        print()

        # [3] active players with no agent, worst by Barrett
        miss_sorted = sorted(miss, key=lambda x: -x[1])
        print(f"[3] {len(miss)} active players with NO agent. Top 25 by Barrett:")
        for nm, b in miss_sorted[:25]:
            print(f"      {nm}  (Barrett {b:.1f})")
        print()
    else:
        print("[1-3] skipped (no roster available)\n")

    # [4] agency spelling collisions, across every agent of every player
    by_key: dict[str, set] = defaultdict(set)
    for rec in amap.values():
        for a in rec["agents"]:
            if a["agency"]:
                by_key[agency_key(a["agency"])].add(a["agency"])
    collisions = {k: v for k, v in by_key.items() if len(v) > 1}
    print(f"[4] {len(collisions)} agency-name collisions (same key, different spelling):")
    for k, v in sorted(collisions.items()):
        print(f"      {k!r}: {sorted(v)}")
    if not collisions:
        print("      none")
    print()

    # [5] agents-per-player distribution (multi-agent is the model, not a bug)
    dist = Counter(len(rec["agents"]) for rec in amap.values())
    print("[5] agents per player (multi-agent is expected):")
    for k in sorted(dist):
        print(f"      {k} agent(s): {dist[k]} players")
    top = [rec for rec in amap.values() if len(rec["agents"]) > 1]
    top.sort(key=lambda r: -len(r["agents"]))
    print("    most-represented:")
    for rec in top[:6]:
        print(f"      {rec['name']}: {', '.join(a['agent'] for a in rec['agents'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
