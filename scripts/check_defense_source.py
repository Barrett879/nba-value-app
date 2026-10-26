"""Is the board's defence actually the blend, or did it fall back to the proxy?

    /opt/anaconda3/bin/python3 -u scripts/check_defense_source.py [season]

The Defense column is whatever `d_lebron` holds, and that column has two
possible sources: the on/off blend artifact, or the box-score proxy used for
pre-2009 seasons. They are the same dtype, the same shape and the same order of
magnitude, so a board built off the wrong one looks completely normal.

On 2026-10-25 the board was rebuilt BEFORE build_defense_blend wrote the new
version. fetch_defense_blend found no file, returned {}, and all 362 players
fell through to the proxy. The site shipped it. 63 players moved by more than
two points, Kris Dunn by -4.5 and Cooper Flagg by -4.4, because the live season
weights defence 1.75x. Nothing failed, nothing logged, every row was present
and plausible.

So this compares the two directly: for every player in BOTH the board and the
blend, the board's value must equal the blend's. A disagreement means the board
was built against a different artifact than the one now on disk -- the wrong
version, a stale copy, or no copy at all.

Exit code 1 on failure.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
import pandas as pd
import utils
from utils import NameIndex
logging.disable(logging.NOTSET)

TOL = 0.005          # the board stores full precision; this is display rounding


def main(season: str) -> int:
    board_path = utils._rev_path(
        f"raw_{season.replace('-', '_')}_{utils.FORMULA_VERSION}.parquet")
    if not board_path.exists():
        print(f"no board cache at {board_path.name}")
        return 1
    board = pd.read_parquet(board_path)

    blend = utils.fetch_defense_blend(season)
    print(f"{board_path.name}  vs the blend artifact\n")
    if not blend:
        print("FAIL: there is NO blend artifact for this season, so every player on "
              "the board is on the box proxy.\n  Build it with "
              "scripts/build_defense_blend.py, then rebuild the board.")
        return 1

    ni = NameIndex(dict(blend))
    matched, bad = 0, []
    for _, r in board.iterrows():
        want = ni.get(str(r["Player"]))
        if want is None:
            continue                      # legitimately below the possession floor
        matched += 1
        got = float(r["d_lebron"])
        if abs(got - float(want)) > TOL:
            bad.append((str(r["Player"]), got, float(want)))

    lo, hi = float(board["d_lebron"].min()), float(board["d_lebron"].max())
    bl = list(blend.values())
    print(f"  board rows            {len(board)}")
    print(f"  blend players         {len(blend)}")
    print(f"  present in both       {matched}")
    print(f"  board d_lebron range  {lo:+.2f} .. {hi:+.2f}")
    print(f"  blend range           {min(bl):+.2f} .. {max(bl):+.2f}")

    if not matched:
        print("\nFAIL: not one board player matched the blend by name. That is a join "
              "failure, not a coverage gap.")
        return 1

    if bad:
        print(f"\n  {len(bad)} of {matched} disagree:")
        for name, got, want in sorted(bad, key=lambda x: -abs(x[1] - x[2]))[:15]:
            print(f"    {name:24s} board {got:+6.2f}   blend {want:+6.2f}")
        print(f"\nFAIL: the board was not built against this blend artifact. Rebuild "
              f"the board (DATA_REV bump if the current one is already live).")
        return 1

    print(f"\nPASS: all {matched} shared players match the blend exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else utils.SEASONS[0]))
