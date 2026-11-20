"""A player carries the last season he ACTUALLY PLAYED, not "one season back".

Barrett, 2026-11-19: "use current season data unless they literally haven't
played a single minute. In that case, use the last season where they did play."

The first half already shipped (utils.BLEND_MIN_MINUTES = 0, set 2026-10-29).
The second half stopped at exactly one season back, so a rostered player who
missed a whole season carried no score anywhere on the site. Ben Simmons was
that player: 51 games and 22.0 MPG in 2024-25, nothing on the board.

The old limit existed for a real reason, written into the code it replaced: an
unbounded walk padded the board with men whose last game was years ago and who
are mostly out of the league. Measured against the committed frames on
2026-11-19, walking four seasons back UNGATED adds 271 rows to a 471-row board,
and 45 of the first 57 are players the site has no record of at all.

So the walk is GATED on tracked_players(), not shortened. Gated, the same walk
adds ELEVEN rows and terminates on its own. These tests pin the gate, the
bound, and the deliberate asymmetry: the FIRST season back is ungated, because
that is the shipped "last complete season" basis and the FA board needs those
rows (75 of the 128 it adds are free agents).
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.disable(logging.WARNING)
import utils  # noqa: E402
from utils import (BLEND_WALKBACK_SEASONS, OUT_OF_LEAGUE_REASONS, SEASONS,  # noqa: E402
                   NameIndex, out_of_league_players, tracked_players)
logging.disable(logging.NOTSET)


def _frame(names):
    """A minimal frame with everything apply_rankings/apply_projections need."""
    return pd.DataFrame({
        "Player": list(names),
        "total_min": [500.0] * len(names),
        "barrett_score": [float(10 + i) for i in range(len(names))],
        "salary": [float(1_000_000 * (i + 1)) for i in range(len(names))],
        "Team": ["XXX"] * len(names),
    })


@pytest.fixture
def blend(monkeypatch):
    """build_ranked_blended over synthetic seasons, production code path."""
    def run(by_season: dict, tracked=()):
        monkeypatch.setattr(utils, "build_raw",
                            lambda s, playoffs=False: _frame(by_season.get(s, [])))
        monkeypatch.setattr(utils, "_raw_disk_exists", lambda s, playoffs=False: True)
        monkeypatch.setattr(utils, "tracked_players", lambda: tuple(tracked))
        df = utils.build_ranked_blended(SEASONS[0])
        return {str(r.Player): str(r.score_season) for r in df.itertuples()}
    return run


# ── tracked_players: the gate itself ─────────────────────────────────────────

def test_tracked_players_is_not_empty():
    assert len(tracked_players()) > 300, (
        "the gate came back near-empty, which silently disables the walk-back")


def test_tracked_players_excludes_the_out_of_league():
    gone = out_of_league_players()
    if not gone:
        pytest.skip("no out-of-league players recorded")
    ni = NameIndex({n: True for n in tracked_players()})
    leaked = sorted(n for n in gone if ni.get(n) is not None)
    assert not leaked, (
        f"retired/overseas/G-League players passed the gate: {leaked[:10]}. "
        f"Reasons held out are {OUT_OF_LEAGUE_REASONS}.")


def test_tracked_players_keeps_free_agents():
    """Deliberately WIDER than the roster: the one-season-back step already
    carries 75 free agents, so gating the deeper seasons to rostered players
    only would have them answer a different question."""
    import csv
    fas = [r["player"].strip() for r in csv.DictReader(
        l for l in (ROOT / "data" / "master_roster.csv").read_text(
            encoding="utf-8").splitlines()
        if l.strip() and not l.lstrip().startswith("#"))
        if (r.get("kind") or "").strip() == "free_agent"]
    gone = NameIndex(out_of_league_players())
    signable = [n for n in fas if gone.get(n) is None]
    if not signable:
        pytest.skip("no signable free agents in master_roster")
    ni = NameIndex({n: True for n in tracked_players()})
    assert ni.get(signable[0]) is not None, (
        f"{signable[0]} is a signable free agent and must pass the gate")


# ── the walk ─────────────────────────────────────────────────────────────────

def test_a_tracked_player_two_seasons_back_is_picked_up(blend):
    """Ben Simmons' case, which the one-season limit missed entirely."""
    got = blend({SEASONS[0]: ["Current Guy"], SEASONS[2]: ["Missed A Year"]},
                tracked=["Missed A Year"])
    assert got.get("Missed A Year") == SEASONS[2], got


def test_an_untracked_player_two_seasons_back_is_not(blend):
    """The 45-of-57 strangers the old comment was right about."""
    got = blend({SEASONS[0]: ["Current Guy"], SEASONS[2]: ["Left The League"]},
                tracked=[])
    assert "Left The League" not in got, got


def test_the_first_season_back_is_not_gated(blend):
    """Shipped behaviour, kept on purpose."""
    got = blend({SEASONS[0]: ["Current Guy"], SEASONS[1]: ["Not Tracked"]},
                tracked=[])
    assert got.get("Not Tracked") == SEASONS[1], got


def test_the_walk_back_depth_is_four():
    """Pinned as a VALUE, not derived.

    The first version of the test below computed its target season from
    BLEND_WALKBACK_SEASONS itself, so raising the constant moved the
    goalposts and the test passed for any depth -- it survived the mutation
    run on 2026-11-19. Raising this number means reopening the padding
    measurement in build_ranked_blended, so it should cost a test edit.
    """
    assert BLEND_WALKBACK_SEASONS == 4, (
        "walk-back depth changed; re-measure what the extra season adds "
        "before accepting it, and update test_the_walk_is_bounded's index")


def test_the_walk_is_bounded(blend):
    """A player whose only season is FIVE back must not be reached.

    The index is a literal on purpose; see the test above.
    """
    assert BLEND_WALKBACK_SEASONS < 5, "this test no longer proves a bound"
    got = blend({SEASONS[0]: ["Current Guy"], SEASONS[5]: ["Ancient History"]},
                tracked=["Ancient History"])
    assert "Ancient History" not in got, (
        f"{SEASONS[5]} is five seasons back and outside the walk: {got}")


def test_the_last_season_played_wins_not_the_oldest(blend):
    got = blend({SEASONS[0]: ["Current Guy"],
                 SEASONS[2]: ["Comeback"], SEASONS[3]: ["Comeback"]},
                tracked=["Comeback"])
    assert got.get("Comeback") == SEASONS[2], (
        f"must take the most recent season he played, got {got}")


def test_a_current_season_player_is_never_overwritten_by_a_fallback(blend):
    got = blend({SEASONS[0]: ["Both Seasons"], SEASONS[2]: ["Both Seasons"]},
                tracked=["Both Seasons"])
    assert got.get("Both Seasons") == SEASONS[0], got


def test_a_missing_fallback_parquet_is_skipped_not_rebuilt(monkeypatch):
    """A missing historical parquet falls through to a live network rebuild.
    On Render /data need not hold all 54 of them, and this is the request
    path, so a deeper season with no file on disk must simply be skipped."""
    asked: list = []

    def fake_build_raw(s, playoffs=False):
        asked.append(s)
        return _frame(["Current Guy"] if s == SEASONS[0] else [])

    monkeypatch.setattr(utils, "build_raw", fake_build_raw)
    monkeypatch.setattr(utils, "_raw_disk_exists", lambda s, playoffs=False: False)
    monkeypatch.setattr(utils, "tracked_players", lambda: ("Whoever",))
    utils.build_ranked_blended(SEASONS[0])
    deep = [s for s in asked if s in SEASONS[2:]]
    assert not deep, f"opened a fallback season with no parquet on disk: {deep}"
    assert SEASONS[1] in asked, (
        "the FIRST season back must still be requested unconditionally, as it "
        "always was")
