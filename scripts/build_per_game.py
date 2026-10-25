"""Ship the per-game line the Rankings row panel needs.

    /opt/anaconda3/bin/python3 -u scripts/build_per_game.py [season]

The panel cannot fetch this live. build_raw carries no PTS/AST/REB, and on
Render fetch_league_stats reaches stats.nba.com, which is blocked there, while
the live season has no committed league_stats parquet to fall back on. The
panel therefore rendered for all 353 players locally and for NONE in
production, silently, because an empty lookup just skips the panel.

So it is precomputed here and committed, like the defence blend. Same
consequence, stated so it is not forgotten: the panel is as fresh as the last
push, and build_defense_blend.py should be rerun alongside this one.

Written SORTED, so a rebuild with no new games is byte-identical and "has
anything changed?" stays answerable by diff.
"""
from __future__ import annotations
import json, logging, sys
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
import utils
logging.disable(logging.NOTSET)

FIELDS = ("PTS", "REB", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF",
          "MIN", "GP", "FG_PCT", "FG3_PCT", "FT_PCT",
          "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PLUS_MINUS",
          # League ranks ship with the feed, so the panel can draw where a
          # player sits instead of asking the reader to know what 8.5 rebounds
          # means. Free: no extra request, no second artifact to keep fresh.
          "PTS_RANK", "REB_RANK", "AST_RANK", "STL_RANK", "BLK_RANK",
          "MIN_RANK", "FG_PCT_RANK")
VERSION = "v2"  # v2 adds ranks and makes/attempts; v1 was the bare per-game line


def main():
    season = sys.argv[1] if len(sys.argv) > 1 else utils.SEASONS[0]
    src = utils.fetch_league_stats(season)
    which = "league_stats"
    if src is None or getattr(src, "empty", True) or "PTS" not in getattr(src, "columns", []):
        src = utils.fetch_bref_player_stats(season)
        which = "bref"
    if src is None or getattr(src, "empty", True):
        print("FAILED: no per-game source returned anything"); return 1

    # Fold in the provisional box scores, exactly as def_lib.players does.
    # Without this the panel reports FEWER games than the board: the first
    # build had Brandon Miller at 3 while every other surface said 5, because
    # the rest of the pipeline merges provisional games and this did not.
    # Two readings of "games played" on one page is worse than a stale panel.
    try:
        src = utils._merge_provisional_box_scores(src, season)
    except Exception as e:
        print(f"  WARNING: provisional merge skipped ({type(e).__name__})")

    keep = [c for c in FIELDS if c in src.columns]
    vals = {}
    for name, rec in zip(src["PLAYER_NAME"], src.to_dict("records")):
        row = {}
        for c in keep:
            v = rec.get(c)
            if v is None or (isinstance(v, float) and v != v):
                continue
            row[c] = round(float(v), 4)
        if row:
            vals[str(name)] = row

    # Reference values the panel's bars are drawn against. Rank percentile was
    # tried first and is useless at the top of the board: everyone in the top
    # 25 of 388 renders at 94-100% and the bar stops carrying information
    # exactly where people look. A bar against a high reference VALUE keeps its
    # range. The reference is the 95th percentile among rotation players, not
    # the maximum, so one 36-point night on two games cannot flatten everyone.
    rot = src[pd.to_numeric(src.get("MIN"), errors="coerce").fillna(0) >= 12] \
        if "MIN" in src.columns else src
    if len(rot) < 30:
        rot = src
    scale = {}
    for c in ("PTS", "REB", "AST", "STL", "BLK", "MIN"):
        if c not in src.columns:
            continue
        v = pd.to_numeric(rot[c], errors="coerce").dropna()
        if len(v):
            scale[c] = round(float(np.percentile(v, 95)), 3) or 1.0

    out = {"_meta": {"season": season, "generated": dt.date.today().isoformat(),
                     "source": which, "fields": keep, "players": len(vals),
                     "ranked_pool": int(len(src)),
                     "bar_scale": scale,
                     "bar_scale_note": ("95th percentile among players at 12+ minutes; "
                                        "bars are value/reference, capped at 100%"),
                     "caveat": "Not fetchable on Render; as fresh as the last push."},
           "values": {k: vals[k] for k in sorted(vals)}}
    path = ROOT / "cache" / f"per_game_{season.replace('-', '_')}_{VERSION}.json"
    path.write_text(json.dumps(out, indent=1))

    # every player the board will show must resolve, or the panel is silently
    # empty for exactly the awkward names
    raw = utils.build_raw(season)
    idx = utils.NameIndex(out["values"])
    miss = [p for p in raw["Player"] if idx.get(str(p)) is None]
    print(f"wrote {path}")
    print(f"  {len(vals)} players from {which}, fields {keep}")
    print(f"  board coverage: {len(raw) - len(miss)} of {len(raw)}"
          + (f"  MISSING: {miss[:8]}" if miss else "  (none missing)"))
    return 1 if miss else 0


if __name__ == "__main__":
    sys.exit(main())
