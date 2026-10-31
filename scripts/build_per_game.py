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
# The six ranks the panel draws are RECOMPUTED after the provisional merge (see
# main). FG_PCT_RANK is left exactly as the feed gave it: the merge updates FGA
# but not FGM, so the shooting percentages here are pre-merge and self-consistent
# with each other. Nothing displays them. Do not start.
# v3 adds the TOV and PF bar references. They were added to v2 IN PLACE, and
# that never reached production: /data seeds a repo cache only when the
# FILENAME is missing, so the first v2 (with no TOV/PF reference) stayed on the
# disk forever and those two bars rendered at zero width on the live site while
# working perfectly here. Contents change => filename changes. The check below
# now enforces it rather than relying on remembering.
VERSION = "v13"  # v8: 2026-10-27 slate (4 games)


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

    # Ranks must be recomputed AFTER the merge, never carried from the feed.
    # The feed's ranks describe the feed's numbers, and the merge moves those
    # numbers: Ty Jerome went 34.7 -> 29.8 when a quiet fourth game folded in
    # and kept 1st, while Anthony Edwards went 28.7 -> 33.3 and kept 6th. The
    # panel then showed "34.3 PPG, 2nd" above "29.8 PPG, 1st", which is how
    # Barrett found it. A rank is a statement ABOUT a value, so it cannot
    # outlive an edit to that value.
    #
    # Same pool the feed used (every player in the frame), so the numbers stay
    # comparable with what shipped before; only the staleness is removed.
    for _stat in ("PTS", "REB", "AST", "STL", "BLK", "MIN"):
        _rk = f"{_stat}_RANK"
        if _stat not in src.columns or _rk not in src.columns:
            continue
        _v = pd.to_numeric(src[_stat], errors="coerce")
        src[_rk] = _v.rank(ascending=False, method="min").astype("Int64")

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
    for c in ("PTS", "REB", "AST", "STL", "BLK", "MIN", "TOV", "PF"):
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

    # Refuse to change an existing artifact's SHAPE under the same name. New
    # numbers are fine and expected -- that is a daily rebuild. A different set
    # of fields or bar references is not: production would keep the old file
    # and the new bars would silently render empty, which is exactly what
    # happened to TOV and PF on 2026-10-25.
    if path.exists():
        try:
            prev = json.loads(path.read_text())
            pm = prev.get("_meta") or {}
            if (set(pm.get("fields") or []) != set(keep)
                    or set((pm.get("bar_scale") or {})) != set(scale)):
                print(f"REFUSING to write {path.name}: its SHAPE changed but the "
                      f"version did not.\n  fields added/removed: "
                      f"{set(keep) ^ set(pm.get('fields') or [])}\n  bar refs "
                      f"added/removed: {set(scale) ^ set(pm.get('bar_scale') or {})}\n"
                      f"  Bump VERSION in this script and in the reader in "
                      f"pages/Rankings.py, then rerun.")
                return 1
        except Exception:
            pass
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
