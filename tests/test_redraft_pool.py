"""The Redraft board may only offer players who are actually in the NBA.

Written 2026-11-19, after scripts/build_redraft_pool.py shipped with NO filter
at all. Any master_roster free_agent row carrying an "est value" phrase was
priced and offered, so the board served 38 men who are not in the league:
Russell Westbrook, Chris Paul, Kyle Lowry and Garrett Temple all retired and
all draftable at about $3.9M, plus 20 overseas and 14 in the G League. They
carried `proj`, so the page said the PRICE was a projection while saying
nothing about the man being unavailable to any NBA team.

THE FILTER IS THE NARROW ONE AND THAT IS THE POINT. utils.unavailable_players
answers "could a team sign him today" and is a strictly LARGER set, because it
also excludes players under NBA contract -- who cannot be signed but are
exactly the people a redraft is for. Copying build_fa_board's filter would have
fixed the retired players and wrongly dropped Blake Wesley and Pete Nance.
Half these tests exist to keep the distinction from collapsing back.

Reads the COMMITTED artifact, so it runs in a clean checkout and catches the
pool going stale or being rebuilt by something without the filter.
"""
from __future__ import annotations
import ast
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.disable(logging.WARNING)
from utils import (NameIndex, OUT_OF_LEAGUE_REASONS,  # noqa: E402
                   UNAVAILABLE_REASONS, out_of_league_players,
                   unavailable_players)
logging.disable(logging.NOTSET)

POOL = ROOT / "cache" / "redraft_pool_v1.json"
BUILDER = ROOT / "scripts" / "build_redraft_pool.py"


def _pool():
    if not POOL.exists():
        pytest.skip(f"{POOL.name} not committed")
    return json.loads(POOL.read_text())["players"]


# ==========================================================================
# the artifact
# ==========================================================================
def test_no_pool_member_is_out_of_the_league():
    """A name here is a player a user can draft, so a man who has retired or
    left for Europe must not appear. Joined through NameIndex, never a bare
    key: the files spell suffixes and accents differently and an exact-key
    miss fails OPEN, leaving him on the board as a bargain."""
    players = _pool()
    gone = out_of_league_players()
    idx = NameIndex({n: n for n in gone})
    bad = sorted({p["n"] for p in players if idx.get(p["n"])})
    assert not bad, (
        f"{POOL.name} offers {len(bad)} player(s) who are not in the NBA: "
        f"{bad}. Each is in data/unavailable_players.csv for a reason in "
        f"{OUT_OF_LEAGUE_REASONS}. Rebuild with scripts/build_redraft_pool.py, "
        "which filters them; if the name got through the filter, the two files "
        "spell him differently and the NameIndex join needs looking at."
    )


def test_players_on_nba_contracts_are_NOT_excluded():
    """The complement of the test above, and the reason the filter is narrow.

    Everyone in unavailable_players is unsignable, but `nba_contract` means he
    is unsignable BECAUSE HE IS ALREADY IN THE LEAGUE. Using the larger set
    here would quietly shrink the pool by exactly those players, which looks
    like a working filter and is a different bug.
    """
    in_league = {n for n, v in unavailable_players().items()
                 if v.get("reason") == "nba_contract"}
    if not in_league:
        pytest.skip("no nba_contract rows on file today")
    idx = NameIndex({p["n"]: p["n"] for p in _pool()})
    # Not every such player need be in the pool (he may have no master_roster
    # row or no price), but the FILTER must not be what removed him.
    gone_idx = NameIndex({n: n for n in out_of_league_players()})
    wrongly_filtered = sorted(n for n in in_league if gone_idx.get(n))
    assert not wrongly_filtered, (
        f"out_of_league_players() excludes {wrongly_filtered}, who are in the "
        "league on NBA contracts. OUT_OF_LEAGUE_REASONS must not contain "
        "'nba_contract'."
    )


def test_the_pool_is_not_empty_and_is_plausibly_the_league():
    """A positive control. Every assertion above passes trivially on an empty
    pool, and a filter bug that dropped everyone would read as clean."""
    players = _pool()
    assert len(players) > 400, (
        f"{POOL.name} has only {len(players)} players; the league plus unsigned "
        "free agents ran about 554 on 2026-11-19, so this is a build failure, "
        "not a filter."
    )
    assert any(p.get("bs") is not None for p in players)


# ==========================================================================
# the reason split, which is the part that will be got wrong again
# ==========================================================================
def test_out_of_league_is_a_strict_subset_of_unavailable():
    assert set(OUT_OF_LEAGUE_REASONS) < set(UNAVAILABLE_REASONS)
    assert "nba_contract" not in OUT_OF_LEAGUE_REASONS
    assert set(out_of_league_players()) <= set(unavailable_players())


def test_a_new_unavailable_reason_defaults_to_out_of_league():
    """OUT_OF_LEAGUE_REASONS is DERIVED from UNAVAILABLE_REASONS minus the
    in-league exception, so adding a reason without classifying it excludes
    that player from the draft pool rather than silently offering him. Wrongly
    omitting a fringe player costs a row; wrongly offering a retired one is the
    bug this whole file is about."""
    unclassified = set(UNAVAILABLE_REASONS) - set(OUT_OF_LEAGUE_REASONS)
    assert unclassified == {"nba_contract"}, (
        f"these reasons are treated as IN the league: {sorted(unclassified)}. "
        "Only 'nba_contract' should be. If a new reason was added, decide "
        "deliberately which side it falls on."
    )


# ==========================================================================
# the builder, so the artifact cannot be regenerated without the filter
# ==========================================================================
def test_the_builder_uses_the_narrow_filter():
    src = BUILDER.read_text()
    tree = ast.parse(src)
    imported = {a.name for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module == "utils"
                for a in n.names}
    assert "out_of_league_players" in imported, (
        f"{BUILDER.name} does not import out_of_league_players, so it cannot be "
        "filtering the pool."
    )
    assert "unavailable_players" not in imported, (
        f"{BUILDER.name} imports unavailable_players, the LARGER 'cannot be "
        "signed' set. Here that also drops players who are in the league on "
        "NBA contracts, which is the whole distinction this filter exists to "
        "make."
    )


def test_the_builder_skips_rather_than_prices_them():
    """The skip has to come BEFORE the "est value" branch, or a retired player
    acquires a plausible-looking salary on his way to being dropped, and any
    future reordering silently puts him back on the board with it."""
    src = BUILDER.read_text()
    assert "gone_idx" in src and "continue" in src
    skip = src.index("gone_idx.get(name)")
    price = src.index("est\\s+value")
    assert skip < price, (
        "the out-of-league skip now runs AFTER the est-value pricing branch; "
        "it must come first."
    )
