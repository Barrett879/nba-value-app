"""Build cache/redraft_alltime_v1.json: every notable player-season since
1973-74, priced on one era-neutral scale.

The pricing is the whole idea. A player's contract is thrown away and replaced
by the salary belonging to HIS RANK. Rank the league by Barrett Score inside a
player's own season, take that ordinal, and pay him whatever the player at that
same ordinal on the 2025-26 salary ladder is paid. Michael Jordan was the best
player in 1990-91, so he is paid what the best-paid player of 2025-26 is paid;
so is LeBron in 2012-13; so is Jokic in 2025-26. Eras stop arguing about
inflation and a 1991 pick costs what it was worth.

Usage:  python -u scripts/build_redraft_alltime.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils import (all_seasons_path, NameIndex, _headshot_id_map)   # noqa: E402

LADDER_SEASON = "2025-26"
PER_SEASON = 40           # deep enough for a 30-team draft, small enough to ship
OUT = ROOT / "cache" / "redraft_alltime_v1.json"


def main():
    df = pd.read_parquet(all_seasons_path(500))
    df = df[df["score_rank"].notna() & df["barrett_score"].notna()]

    # the ladder: 2025-26 salaries, richest first. Rank r is paid ladder[r-1].
    cur = df[df["Season"] == LADDER_SEASON]
    ladder = sorted((float(s) for s in cur["salary"].dropna() if float(s) > 0), reverse=True)
    if not ladder:
        raise SystemExit(f"no {LADDER_SEASON} salaries to build the ladder from")
    print(f"  ladder: {len(ladder)} salaries, top ${ladder[0]/1e6:.1f}M, "
          f"floor ${ladder[-1]/1e6:.2f}M", flush=True)

    heads = NameIndex(_headshot_id_map())

    # Per-game PRA, read STRAIGHT off the cached league-stats parquets rather
    # than through a fetcher. Only 23 of the 53 seasons have a bref_stats cache,
    # so the fetch path went to Basketball-Reference for the other 30 and sat
    # there being rate-limited; these files cover every season and are local.
    pra_by_season, missing = {}, []
    for season in sorted(df["Season"].unique()):
        # Two caches, and the split is real: the NBA-API league-stats files
        # exist for every season but are EMPTY before 1996-97, which is why a
        # file-exists check was not enough -- 23 seasons read as zero rows and
        # silently lost their PRA. Basketball-Reference covers exactly those.
        box = None
        f = ROOT / "cache" / f"league_stats_{season.replace('-', '_')}.parquet"
        if f.exists():
            b = pd.read_parquet(f)
            if len(b):
                box = b
        if box is None:
            f2 = ROOT / "cache" / f"bref_stats_v3_{season}.parquet"
            if f2.exists():
                b = pd.read_parquet(f2)
                if len(b):
                    box = b
        if box is None:
            missing.append(season)
            continue
        # Two shapes live under this name: the NBA-API era carries REB, the
        # older Basketball-Reference era carries OREB and DREB and no REB at
        # all. Reading only REB silently dropped every season before 1996-97.
        cols = set(box.columns)
        if "REB" in cols:
            reb = lambda r: float(r["REB"])
        elif {"OREB", "DREB"} <= cols:
            reb = lambda r: float(r["OREB"]) + float(r["DREB"])
        else:
            missing.append(season); continue
        idx = NameIndex()
        for _, r in box.iterrows():
            try:
                idx.add(str(r["PLAYER_NAME"]),
                        round(float(r["PTS"]) + reb(r) + float(r["AST"]), 1))
            except Exception:
                continue
        pra_by_season[season] = idx
    if missing:
        print(f"  no box scores cached for {len(missing)} season(s): "
              f"{', '.join(missing[:6])}", flush=True)

    out = []
    for season, grp in df.groupby("Season"):
        grp = grp.nsmallest(PER_SEASON, "score_rank")
        pra = pra_by_season.get(season)
        for r in grp.itertuples():
            rank = int(r.score_rank)
            if rank > len(ladder):
                continue                       # deeper than the league pays
            name = str(r.Player)
            out.append({
                "n": name,
                "yr": season,
                "rank": rank,
                "sal": round(ladder[rank - 1] / 1e6, 2),
                "bs": round(float(r.barrett_score), 1),
                "pra": (pra.get(name) if pra else None),
                "ts": (None if r.ts_pct != r.ts_pct else round(float(r.ts_pct) * 100, 1)),
                "dleb": (None if r.d_lebron != r.d_lebron else round(float(r.d_lebron), 1)),
                "gp": int(r.GP), "mpg": round(float(r.MPG), 1),
                "pid": heads.get(name),
            })

    out.sort(key=lambda p: -p["bs"])
    OUT.write_text(json.dumps({"ladder_season": LADDER_SEASON, "players": out},
                              separators=(",", ":")))
    seasons = len({p["yr"] for p in out})
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(out)} player-seasons across "
          f"{seasons} seasons, {sum(1 for p in out if p['pra'] is not None)} with PRA, "
          f"{kb:.0f}KB)", flush=True)
    for p in out[:3]:
        print(f"    {p['n']} {p['yr']}  rank {p['rank']} -> ${p['sal']}M", flush=True)


if __name__ == "__main__":
    main()
