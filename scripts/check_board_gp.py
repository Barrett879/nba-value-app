"""Does any team's board show more games played than it has actually played?

    /opt/anaconda3/bin/python3 -u scripts/check_board_gp.py [season]

A board GP above the team's real game count means a game got counted twice.
That is not a rounding wobble: every per-game average for that team is divided
by the wrong denominator, so the whole team reads low (or, for a player whose
extra game was a good one, high) and nothing else on the page looks wrong.

Written after 2026-10-25, when the provisional merge folded already-counted
games back in for the three franchises whose BBRef code differs from their NBA
code -- Charlotte showed 5 games on a night it had played 3. Every internal
check passed, because the frame was perfectly consistent with itself. The only
thing that could catch it was comparing against a source that counts games a
different way, which is what this does: the board comes from the stat feed, the
truth comes from the schedule.
"""
from __future__ import annotations
import logging, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
logging.disable(logging.WARNING)
import pandas as pd
import utils
import def_lib as L
logging.disable(logging.NOTSET)


def main(season: str) -> int:
    path = utils._rev_path(
        f"raw_{season.replace('-', '_')}_{utils.FORMULA_VERSION}.parquet")
    if not path.exists():
        print(f"no board cache at {path.name}")
        return 1
    board = pd.read_parquet(path)
    played, _lineups, _missing = L.feed_gap(season)

    print(f"{path.name}  vs the {season} schedule\n")
    rows, bad = [], []
    for team, grp in board.groupby("Team"):
        gp = float(pd.to_numeric(grp["GP"], errors="coerce").max())
        truth = played.get(team)
        if truth is None:
            # A team the schedule has never heard of is a code that leaked
            # through untranslated -- exactly the phantom "BRK" row this bug
            # produced alongside the inflated "BKN" one.
            bad.append((team, gp, "not a team in the schedule feed"))
            rows.append((team, gp, "-", "UNKNOWN TEAM"))
            continue
        if gp > truth:
            bad.append((team, gp, f"schedule says {truth}"))
        rows.append((team, gp, truth, "ok" if gp <= truth else "OVER"))

    for team, gp, truth, verdict in sorted(rows, key=lambda r: (r[3] == "ok", r[0])):
        print(f"  {verdict:12s} {team:4s} board {gp:>4}   schedule {truth}")

    missing = sorted(set(played) - set(board["Team"].dropna().unique()))
    if missing:
        print(f"\nteams in the schedule with no board rows: {missing}")

    if bad:
        print(f"\nFAIL: {len(bad)} team(s) show more games than they have played.")
        return 1
    print("\nPASS: no team's board GP exceeds its real game count.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else utils.SEASONS[0]))
