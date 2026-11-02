"""The CBA max-contract window has to be anchored on a season that was PLAYED.

All-NBA teams are voted in May. A season still in progress therefore has no
selections at all, and a window anchored on it does not come back partial, it
comes back EMPTY -- which at the call site is indistinguishable from a player
who was never selected. Nothing raises, and the answer is simply one tier too
low.

That is what happened on 2026-10-20, when SEASONS[0] moved to the season being
played and stopped being the last season complete. Every designated-rookie and
designated-vet check in the app lost its most recent qualifying season that
morning: Duren, Holmgren, Jalen Johnson and Wembanyama each held a 2025-26
All-NBA selection that the window no longer looked at, so each read 25% instead
of 30%.

These tests pin the anchor, not the arithmetic around it. The first two are pure
logic and hold whatever the wall clock says; the last two go through the
production functions rather than reimplementing the window, so a future edit to
get_all_nba_in_window cannot pass them by testing a copy.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils  # noqa: E402


def test_anchor_is_never_a_live_season():
    """Whatever you hand it, the anchor names a season that has finished."""
    for season in utils.SEASONS[:8]:
        anchor = utils.last_completed_season(season)
        assert not utils.season_is_live(anchor), (
            f"last_completed_season({season!r}) returned {anchor!r}, which is "
            "still being played, so it has no All-NBA teams to read"
        )


def test_anchor_leaves_a_finished_season_alone():
    """It shifts the live season back and touches nothing else.

    Also idempotent: anchoring an anchor is a no-op. A backtest that asks about
    a historical season must get that season, not the one before it.
    """
    for season in utils.SEASONS[1:8]:
        if utils.season_is_live(season):
            continue
        assert utils.last_completed_season(season) == season
    once = utils.last_completed_season(utils.SEASONS[0])
    assert utils.last_completed_season(once) == once


def test_window_sees_the_last_completed_season():
    """Every player selected in the last completed season is found by the
    1-season window the eligibility check uses for "most recent season"."""
    anchor = utils.last_completed_season(utils.SEASONS[0])
    selections = utils.fetch_all_nba_selections()
    picked = [n for n, ss in selections.items()
              if any(s["season"] == anchor for s in ss)]
    if not picked:
        pytest.skip(f"no cached All-NBA selections for {anchor}")
    assert len(picked) == 15, f"{anchor}: {len(picked)} selections, expected 15"
    for name in picked:
        assert utils.get_all_nba_in_window(name, anchor, 1), (
            f"{name} was All-NBA in {anchor} but the immediate window missed it"
        )


def test_designated_rookie_bump_survives_the_rollover():
    """The case this file was written for.

    Jalen Duren made the 2025-26 All-NBA Third Team, which satisfies the 5th
    Year 30% Max Criteria for a deal beginning with his fifth season. His max is
    30% of the cap, not 25%, and it has to stay 30% while 2026-27 is unplayed.
    """
    anchor = utils.last_completed_season(utils.SEASONS[0])
    if anchor != "2025-26":
        pytest.skip(f"fixture is 2025-26 All-NBA; last completed is {anchor}")
    elig = utils.get_max_contract_eligibility("Jalen Duren", utils.SEASONS[0])
    assert elig["qualifying"] is True, "2025-26 All-NBA Third Team did not qualify"
    assert elig["max_pct"] == 0.30, elig["supermax_tier"]


if __name__ == "__main__":
    # Script mode, matching tests/test_trade_rules.py: a line per check, so a
    # probe reads as PASS or FAIL rather than as a traceback to interpret.
    rc = 0
    for label, fn in (
        ("anchor is never a live season", test_anchor_is_never_a_live_season),
        ("anchor leaves a finished season alone", test_anchor_leaves_a_finished_season_alone),
        ("window sees the last completed season", test_window_sees_the_last_completed_season),
        ("designated rookie bump survives the rollover",
         test_designated_rookie_bump_survives_the_rollover),
    ):
        try:
            fn()
            print(f"ok  {label}")
        except Exception as exc:                       # includes pytest's Skipped
            kind = "SKIP" if type(exc).__name__ == "Skipped" else "FAIL"
            print(f"{kind}  {label}: {exc}")
            if kind == "FAIL":
                rc = 1
    print("\nMAX-ELIGIBILITY WINDOW ANCHORED" if rc == 0 else
          "\nMAX-ELIGIBILITY WINDOW MISANCHORED")
    sys.exit(rc)
