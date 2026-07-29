#!/usr/bin/env python3
"""Precompute the nav search index: cache/nav_search_v1.json.

The search box renders on every page, so it cannot afford model work on the
request path. This bakes everything a result row shows -- name, team, position,
Barrett Score, league rank, headshot id -- into one small file the nav just
reads.

    python scripts/build_nav_search.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils import (NameIndex, SEASONS, TEAM_HEX, _headshot_id_map,  # noqa: E402
                   build_ranked_projected, normalize)

OUT = ROOT / "cache" / "nav_search_v1.json"
SEASON = SEASONS[0]


def main() -> None:
    df = build_ranked_projected(SEASON)
    heads = NameIndex(_headshot_id_map())

    # position comes from the master roster first (hand-verified, and the same
    # label the Rosters page shows), then the projected board
    pos = NameIndex()
    try:
        board = json.loads((ROOT / "cache" / "team_pages.json").read_text())["teams"]
        for t in board.values():
            for p in t["players"]:
                if p.get("pos"):
                    pos.add(p["n"], p["pos"])
    except Exception:
        pass
    import csv
    try:
        lines = [l for l in (ROOT / "data" / "master_roster.csv").read_text().splitlines()
                 if not l.lstrip().startswith("#")]
        master = NameIndex()
        for r in csv.DictReader(lines):
            if r.get("player") and r.get("pos"):
                master.add(r["player"], r["pos"].strip())
    except Exception:
        master = NameIndex()

    rows = []
    ranked = df.sort_values("barrett_score", ascending=False).reset_index(drop=True)
    for i, r in ranked.iterrows():
        name = str(r["Player"])
        pid = heads.get(name)
        team = str(r.get("Team") or "")
        rows.append({
            "n": name,
            "t": team,
            "c": TEAM_HEX.get(team, ""),
            "p": (master.get(name) or pos.get(name) or ""),
            "s": round(float(r["barrett_score"]), 1),
            "r": i + 1,
            "h": pid or None,
        })
    OUT.write_text(json.dumps({"season": SEASON, "players": rows},
                              separators=(",", ":")))
    kb = round(OUT.stat().st_size / 1024)
    print(f"wrote {OUT.name}: {len(rows)} players, {kb}KB")
    for x in rows[:5]:
        print(f"  #{x['r']:<3} {x['n']:24} {x['t']:4} {x['p']:8} {x['s']}")


if __name__ == "__main__":
    main()
