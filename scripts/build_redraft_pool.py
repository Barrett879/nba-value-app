"""Build cache/redraft_pool_v1.json: every player in the league with his 2026-27
contract and the four numbers the Redraft board shows -- Barrett Score, PRA,
TS% and D-LEBRON.

The Redraft page strips team status from everyone and lets you refill rosters
from scratch, so this pool is deliberately the WHOLE league, not one team's
book. A player under contract carries his real 2026-27 salary; an unsigned free
agent has no contract to carry, so he is priced at the model's own estimate and
flagged `proj` -- the page marks those so a user never mistakes a projection for
a signed number.

Usage:  python -u scripts/build_redraft_pool.py
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils import (build_raw, fetch_league_stats, normalize, NameIndex,   # noqa: E402
                   _headshot_id_map, SEASONS)

VALUE_SEASON = "2025-26"
OUT = ROOT / "cache" / "redraft_pool_v1.json"


def _rows(path):
    lines = [l for l in path.read_text(encoding="utf-8").splitlines()
             if not l.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def main():
    master = _rows(ROOT / "data" / "master_roster.csv")

    # value-season numbers: the score, the defensive impact estimate, and the
    # shooting efficiency, all keyed name-tolerantly (feeds spell accents and
    # suffixes differently and an exact key silently drops the movers)
    stat = NameIndex()
    for r in build_raw(VALUE_SEASON).itertuples():
        stat.add(r.Player, {
            "bs": round(float(r.barrett_score), 1),
            "dleb": (None if r.d_lebron != r.d_lebron else round(float(r.d_lebron), 1)),
            "ts": (None if r.ts_pct != r.ts_pct else round(float(r.ts_pct) * 100, 1)),
            "gp": int(r.GP), "mpg": round(float(r.MPG), 1),
        })

    # PRA comes from the box-score feed, not the raw frame
    pra = NameIndex()
    try:
        for _, r in fetch_league_stats(VALUE_SEASON).iterrows():
            pra.add(str(r["PLAYER_NAME"]),
                    round(float(r["PTS"]) + float(r["REB"]) + float(r["AST"]), 1))
    except Exception as e:                      # never block the build
        print(f"  league stats unavailable ({e}); PRA will be blank", flush=True)

    heads = NameIndex(_headshot_id_map())

    pool, seen = [], set()
    for r in master:
        name = (r.get("player") or "").strip()
        kind = (r.get("kind") or "").strip()
        if not name or kind not in ("standard", "two_way", "free_agent"):
            continue
        key = normalize(name)
        if key in seen:                          # a name can appear once only
            continue
        seen.add(key)

        sal = float(r.get("salary_M") or 0)
        note = (r.get("notes") or "").strip()
        proj = False
        if sal <= 0:
            # unsigned: the roster file carries "est value $12.3M" for these
            m = re.search(r"\$([0-9.]+)M", note)
            if m:
                sal, proj = float(m.group(1)), True
            elif kind == "two_way":
                sal = 0.679                      # the two-way slot amount
            else:
                continue                         # no contract and no estimate

        s = stat.get(name) or {}
        pid = heads.get(name)
        pool.append({
            "n": name,
            "pos": (r.get("pos") or "").strip(),
            "t25": (r.get("team") or "").strip(),   # last team, shown as context only
            "sal": round(sal, 2),
            "proj": proj,
            "tw": kind == "two_way",
            "bs": s.get("bs"), "pra": pra.get(name),
            "ts": s.get("ts"), "dleb": s.get("dleb"),
            "gp": s.get("gp"), "mpg": s.get("mpg"),
            "pid": pid,
        })

    # best players first: the board opens on the names a drafter wants
    pool.sort(key=lambda p: (-(p["bs"] if p["bs"] is not None else -1), -p["sal"]))
    OUT.write_text(json.dumps({"season": "2026-27", "value_season": VALUE_SEASON,
                               "players": pool}, separators=(",", ":")))
    withstat = sum(1 for p in pool if p["bs"] is not None)
    withpra = sum(1 for p in pool if p["pra"] is not None)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(pool)} players, "
          f"{withstat} with a score, {withpra} with PRA, "
          f"{sum(1 for p in pool if p['proj'])} priced by model)", flush=True)


if __name__ == "__main__":
    main()
