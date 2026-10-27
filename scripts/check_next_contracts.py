"""Is the Next $ column actually populated, or is it a column of dashes?

    /opt/anaconda3/bin/python3 -u scripts/check_next_contracts.py [season]

The sentinel is load-bearing here, which is what makes this failure invisible:
an absent player renders as the gray dash, and the legend defines that dash as
"UFA". So a feed that returns nothing does not look broken. It looks like a
league where everybody is a free agent, which is a sentence nobody reads.

That is how it shipped. The season flipped on 2026-10-20, the code began asking
for next_contracts_2027_v7.pkl, which had never been built, and for a week every
one of the 372 players on the board showed a dash. Nothing logged, because
"empty" and "all free agents" are the same bytes.

The underlying trap is seasonal: _derive_next_year_contracts calls
fetch_salaries(espn_year + 1), and ESPN only publishes a season's salaries once
that season has begun. For eleven months of the year the live derivation CANNOT
work, so the committed artifact is the only thing standing between this column
and a wall of dashes. Rebuild it with scripts/build_next_contracts.py.

Exit code 1 on failure.
"""
from __future__ import annotations
import logging, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
import pandas as pd
import utils
from utils import fmt_next_contract
logging.disable(logging.NOTSET)

MIN_FEED = 200          # fetch_next_year_contracts' own "this scrape failed" floor
MIN_COVER = 0.50        # of the board; 2026-27 sits near 0.79


def main(season: str) -> int:
    espn_year = utils.season_to_espn_year(season)
    path = utils._dc_path(f"next_contracts_{espn_year}_v7.pkl")
    feed = utils.fetch_next_year_contracts(espn_year, cache_v=7)

    print(f"{season}  ->  {path.name}")
    print(f"  file on disk      {'yes' if path.exists() else 'NO'}")
    print(f"  feed entries      {len(feed)}")

    board_path = utils._rev_path(
        f"raw_{season.replace('-', '_')}_{utils.FORMULA_VERSION}.parquet")
    if not board_path.exists():
        print(f"  no board cache at {board_path.name}")
        return 1
    board = pd.read_parquet(board_path)
    vals = [fmt_next_contract(p, feed) for p in board["Player"]]
    filled = sum(1 for v in vals if v != "—")
    cover = filled / len(board) if len(board) else 0.0

    from collections import Counter
    kinds = Counter("dash" if v == "—" else "RFA" if v == "RFA"
                    else "TO" if v.endswith(" TO") else "PO" if v.endswith(" PO")
                    else "money" for v in vals)
    print(f"  board rows        {len(board)}")
    print(f"  with a value      {filled}  ({cover:.0%})")
    print(f"  breakdown         {dict(kinds)}")

    if not path.exists():
        print(f"\nFAIL: {path.name} has never been built. The live derivation cannot "
              f"build it mid-season (ESPN has no {espn_year + 1} salaries yet), so the "
              f"column will stay empty until you run scripts/build_next_contracts.py.")
        return 1
    if len(feed) < MIN_FEED:
        print(f"\nFAIL: only {len(feed)} entries, under the {MIN_FEED} floor that "
              f"fetch_next_year_contracts itself treats as a failed scrape.")
        return 1
    if cover < MIN_COVER:
        print(f"\nFAIL: only {cover:.0%} of the board has a Next $ value, under the "
              f"{MIN_COVER:.0%} floor. A column of dashes reads as 'everyone is a "
              f"free agent', which is why nobody notices it.")
        return 1
    print(f"\nPASS: {filled} of {len(board)} board players carry a Next $ value.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else utils.SEASONS[0]))
