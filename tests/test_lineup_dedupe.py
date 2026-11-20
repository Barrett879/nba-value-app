"""One real game must appear in the defensive sample exactly once.

ESPN dates a game by UTC and the NBA feed by local date, so a game tipping
after 8pm ET carries TWO different dates. `def_lib.games()` deduped an
ESPN-rebuilt lineup file against the real pbpstats one on
`(date, away@home)` -- read out of the lineup FILES, where those two dates
disagree -- and its own docstring stated the hole without noticing it: "if
pbpstats later processes the same matchup ON THE SAME DATE, the real file must
win."

Eleven games therefore kept both copies:

    games_2026-27.json          420 entries
    after the old dedupe        237
    ESPN completed               226      <- eleven too many

and 15 of 30 teams were handed a defensive sample LARGER than the number of
games they had played: HOU 20 against 16 (a 25% inflated sample), MIL 18/16,
SAS 18/16, BOS 17/15, DAL 17/15. `defense_blend` is 0.818 of `d_lebron` and
`barrett_score` is built on that, so it moved the headline number for half the
league. The blend logged "2-20 team games" every night.

THE FIX: key on the FEED ENTRY, not the lineup file. The backfill already
resolves the local date correctly -- an ESPN row carries `Date` and
`_matchup`, and those agree with the NBA row for the same game even when the
two files are a day apart.

THE FIX THAT WOULD HAVE BEEN WRONG: a plus-or-minus one day window. Teams play
each other on consecutive days, and this season has two such pairs. A window
deletes a real game. That is pinned below.

AND THE REASON IT SURVIVED: `feed_gap`'s `missing` is a ONE-WAY set
difference, so a feed holding MORE than ESPN could never trip it. The nightly
printed the surplus in its own headline every night and gated on nothing.
"""
from __future__ import annotations
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

logging.disable(logging.WARNING)
import def_lib as L  # noqa: E402
from utils import SEASONS  # noqa: E402
logging.disable(logging.NOTSET)

PB = ROOT / "cache" / "pbpstats"
SCHED = ROOT / "cache" / "team_games_2026-27.json"
NIGHTLY = ROOT / "scripts" / "nightly.py"
SEASON = SEASONS[0]


@pytest.fixture(scope="module")
def kept():
    if not (PB / f"games_{SEASON}.json").exists():
        pytest.skip("lineup feed absent")
    return L.games(SEASON)


def test_the_feed_holds_no_game_espn_does_not_report_completed(kept):
    """The direction nobody checked. A surplus is a double count, a preseason
    fixture, or the wrong season -- never benign."""
    surplus = L.feed_surplus(SEASON)
    assert not surplus, (
        f"{len(surplus)} game(s) in the lineup feed are not completed ESPN "
        f"games: {surplus[:8]}")


def test_no_team_has_more_lineup_games_than_it_has_played(kept):
    """The damage, measured where it lands. Houston had 20 against 16."""
    if not SCHED.exists():
        pytest.skip("schedule absent")
    sched = json.loads(SCHED.read_text(encoding="utf-8"))
    per: Counter = Counter()
    for g in kept:
        mu = L._game_key(g)[1]
        if "@" not in mu:
            continue
        a, h = mu.split("@", 1)
        per[a] += 1
        per[h] += 1
    over = {t: (n, sched[t]) for t, n in per.items()
            if isinstance(sched.get(t), int) and n > sched[t]}
    assert not over, f"defensive sample exceeds games played (feed, schedule): {over}"


def test_every_kept_game_is_distinct(kept):
    """The dedupe's actual job, stated directly."""
    keys = [L._game_key(g) for g in kept]
    dupes = [k for k, n in Counter(keys).items() if n > 1]
    assert not dupes, f"the same game is kept twice: {dupes}"


def test_the_dedupe_still_keeps_a_genuinely_espn_only_game():
    """A positive control. Shadowing EVERY rebuild would pass all three tests
    above and silently drop games the NBA feed never published -- which is the
    whole reason the backfill exists.

    Asked of CONSTRUCTED rows, not of the live feed. This test first shipped
    as `assert espn_only` against today's feed, and that is a TIME BOMB: a
    rebuild survives only while pbpstats is behind, so the day the feed caught
    up on all 226 games every rebuild was correctly shadowed, zero survived,
    and the control failed on a perfectly healthy dedupe. Nothing would have
    reported it either: HoopsValue has no CI and scripts/nightly.py does not
    run pytest, so the suite is red only for whoever next runs it by hand.
    Three rows answer the real question.
    """
    real = {"GameId": "0022600100", "Date": "2026-11-17",
            "AwayTeamAbbreviation": "CHA", "HomeTeamAbbreviation": "LAL"}
    shadowed = {"GameId": "E401909999", "Date": "2026-11-17",
                "_matchup": "CHA@LAL"}          # same game, ESPN's copy
    only = {"GameId": "E401909998", "Date": "2026-11-17",
            "_matchup": "UTA@SAC"}              # the feed never published it

    kept_ids = {str(g["GameId"]) for g in L._shadow_rebuilds([real, shadowed, only])}
    assert "0022600100" in kept_ids, "the real pbpstats file must be kept"
    assert "E401909999" not in kept_ids, (
        "an ESPN rebuild of a game pbpstats has now processed must lose to "
        "the real file, or the rebuild shadows better data forever")
    assert "E401909998" in kept_ids, (
        "a game only ESPN reports was dropped; shadowing every rebuild passes "
        "the surplus, per-team and distinctness tests while silently losing "
        "exactly the games the backfill exists to recover")


def test_the_live_feed_shadows_rebuilds_without_losing_games(kept):
    """The live-feed half, stated so it cannot time-bomb: whatever the feed's
    state, the kept set must equal the real files plus only those rebuilds
    with no real counterpart. On 2026-11-20 that is 226 real and 0 rebuilds;
    in October it was 182 real and 16 rebuilds. Both are correct."""
    real_keys = {L._game_key(g) for g in kept
                 if not str(g.get("GameId", "")).startswith("E")}
    leaked = [L._game_key(g) for g in kept
              if str(g.get("GameId", "")).startswith("E")
              and L._game_key(g) in real_keys]
    assert not leaked, f"a rebuild survived alongside its real file: {leaked}"


def test_consecutive_day_rematches_are_not_collapsed(kept):
    """THE TRAP in the obvious fix. Widening the key to a plus-or-minus one
    day window would delete a real game, because teams DO play each other on
    consecutive days. Both halves of each pair must survive."""
    by_matchup: dict[str, list[str]] = {}
    for g in kept:
        d, mu = L._game_key(g)
        by_matchup.setdefault(mu, []).append(d)
    pairs = {mu: sorted(ds) for mu, ds in by_matchup.items()
             if any(abs(int(x) - int(y)) == 1
                    for x in ds for y in ds if x < y and x.isdigit() and y.isdigit())}
    assert pairs, (
        "no consecutive-day rematch survives in the feed, so this test cannot "
        "prove a date-window fix would have destroyed one")
    for mu, ds in pairs.items():
        assert len(ds) == len(set(ds)), f"{mu} kept a duplicate date: {ds}"


def test_the_key_comes_from_the_feed_entry_not_the_lineup_file():
    """The lineup FILE's date is ESPN's UTC one on a late game. Reading it is
    what caused this, so the identity must not come from there."""
    src = (ROOT / "scripts" / "def_lib.py").read_text(encoding="utf-8")
    fn = src.split("def _game_key(", 1)[1].split("\ndef ", 1)[0]
    body = fn.split('"""', 2)[-1]
    assert "lineupopp" not in body and "away_team_abbreviation" not in body, (
        "_game_key reads the lineup file again; the feed entry is the source")
    assert '_matchup' in body and 'g.get("Date")' in body


def test_a_date_window_would_be_a_regression_not_a_fix(kept):
    """Pins the reasoning, not just the code: show that matching on matchup
    plus a one-day window really does merge games that are separate."""
    by_matchup: dict[str, list[str]] = {}
    for g in kept:
        d, mu = L._game_key(g)
        if d.isdigit():
            by_matchup.setdefault(mu, []).append(d)
    would_merge = [(mu, sorted(ds)) for mu, ds in by_matchup.items()
                   if any(abs(int(x) - int(y)) == 1
                          for x in ds for y in ds if x < y)]
    assert would_merge, (
        "no pair in this season is one day apart, so the window fix looks "
        "safe here; it is not safe in general and this test is the record")


def test_the_nightly_fails_on_a_surplus():
    """A check nothing calls is not a check. The remedy for a surplus is not
    automatic, so the nightly must STOP rather than ship a blend built on
    double-counted games."""
    src = NIGHTLY.read_text(encoding="utf-8")
    assert "feed_surplus" in src, "nightly.py never asks about a surplus"
    after = src.split("feed_surplus", 1)[1][:600]
    assert "return 1" in after, "the nightly notices a surplus and carries on"
