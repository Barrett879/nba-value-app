"""An option is a fact about ONE season, and the cache that held them outlived it.

Barrett, 2026-11-20, looking at the 2027 Free Agent Board: "this is 26-27 season,
why would austin reaves be here?" He re-signed 4yr/$184.76M in July 2026 and is
under contract to 2029-30. He was on the board because
cache/option_status_2026.json says player_option for him -- the option on his OLD
deal, for 2026-27, which he DECLINED in June 2026 in order to sign the new one. An
already-executed decision was being published as a pending one.

utils._load_option_status read that one hardcoded filename whatever season it was
asked about. The file's own metadata says {"season": "2025-26", "upcoming":
"2026-27"} and all 72 entries carry end_season 2026-27, so every decision in it had
closed. classify_fa_status consults it FIRST and returns on a hit, so the correct
feed behind it never got a say: 12 players who are on the books for 2027-28 were
published as option holders, Wembanyama at number one, signed to 2031-32.

These tests call utils directly. An earlier test file in this repo reimplemented its
gate and three of four mutations survived, so the rule here is that the production
function is the thing under test.
"""
from __future__ import annotations
import csv
import json
import logging
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.disable(logging.WARNING)
import utils  # noqa: E402
from utils import normalize  # noqa: E402
logging.disable(logging.NOTSET)

CUR = utils.SEASONS[0]
BOARD_YEAR = int(CUR[:4]) + 1
UPCOMING = f"{BOARD_YEAR}-{(BOARD_YEAR + 1) % 100:02d}"
CACHE = ROOT / "cache"
CY = ROOT / "data" / "contract_years_2026_27.csv"


def _rows(path: Path):
    return list(csv.DictReader(
        l for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.lstrip().startswith("#")))


@pytest.fixture(autouse=True)
def _clear_memo():
    """The loader memoizes per season; a stale memo would hide a broken gate."""
    utils._OPTION_STATUS_CACHE.clear()
    yield
    utils._OPTION_STATUS_CACHE.clear()


@pytest.fixture(scope="module")
def flags():
    out = {}
    for r in _rows(CY):
        if r.get("player"):
            out[normalize(r["player"])] = dict(
                x.split(":", 1) for x in (r.get("seasons") or "").split("|") if ":" in x)
    return out


FEED_NAME = f"next_contracts_{utils.season_to_espn_year(CUR)}_v7.pkl"


@pytest.fixture(scope="module")
def feed():
    """Through utils._rev_path, because that is what production reads.

    The first version of this file opened cache/<FEED_NAME> directly. That is the
    un-revisioned FLOOR; _rev_path prefers the DATA_REV copy, and on 2026-11-20
    the floor was seven days older than the served r49 file and disagreed with it
    about six option holders. Reading the floor made a measurement say the salary
    feed never mislabels an option, which is false, and nearly cost Al Horford his
    row.
    """
    p = utils._rev_path(FEED_NAME)
    if not p.exists():
        pytest.skip("next-year feed cache absent")
    return pickle.loads(p.read_bytes())


@pytest.fixture(scope="module")
def ends():
    p = CACHE / "contract_end_years_v2.pkl"
    if not p.exists():
        pytest.skip("contract-end cache absent")
    return pickle.loads(p.read_bytes())


@pytest.fixture(scope="module")
def rookies():
    logging.disable(logging.WARNING)
    try:
        return utils.fetch_rookie_scale_players(CUR)
    except Exception:
        pytest.skip("rookie-scale set unavailable")
    finally:
        logging.disable(logging.NOTSET)


def _status(name, feed, ends, rookies, include_rookie=False):
    """PRODUCTION CODE. Not a copy of the branch."""
    return utils.classify_fa_status(
        name, utils.fmt_next_contract(name, feed), rookies, CUR,
        cross_check=True, include_rookie_options=include_rookie,
        contract_end_map=ends)


def test_the_loader_refuses_a_cache_built_for_another_season():
    """THE BUG. cache/option_status_2026.json describes the 2026-27 offseason.
    This season's board decides the %s one, so that file must not be consulted.""" % UPCOMING
    stale = CACHE / "option_status_2026.json"
    if not stale.exists():
        pytest.skip("the stale 2026 cache is gone; this test has outlived its subject")
    meta = json.loads(stale.read_text())
    assert meta.get("upcoming") == "2026-27", (
        "option_status_2026.json no longer claims the 2026-27 offseason; reread "
        "this test, its premise has changed")

    got = utils._load_option_status(CUR)
    assert got == {}, (
        f"the loader handed back {len(got)} option entries for {UPCOMING} out of a "
        f"file that says it describes {meta.get('upcoming')}. Those decisions are "
        f"closed, and reusing them publishes players who are under contract as "
        f"free agents")


def test_the_loader_still_serves_the_season_it_was_built_for():
    """POSITIVE CONTROL, and the whole point: the gate is season-aware, not a mute.
    A fix that simply stopped reading the file would pass the test above."""
    stale = CACHE / "option_status_2026.json"
    if not stale.exists():
        pytest.skip("the stale 2026 cache is gone")
    n = len(json.loads(stale.read_text()).get("options") or {})
    assert n > 0, "the 2026 cache has no entries, so this control proves nothing"
    got = utils._load_option_status("2025-26")
    assert len(got) == n, (
        f"asked for the season this file DOES describe (2025-26 -> 2026-27 "
        f"offseason) and got {len(got)} of {n} entries. The gate is rejecting "
        f"everything rather than rejecting the wrong season")


def test_a_cache_whose_metadata_disagrees_with_its_filename_is_refused():
    """DEFENCE IN DEPTH. The filename is the first gate; a file could still be
    rebuilt under the right name carrying the wrong season inside. Uses a far
    future season so it cannot collide with real data, and removes it again."""
    probe_season = "2098-99"
    probe_year = 2099
    p = CACHE / f"option_status_{probe_year}.json"
    assert not p.exists(), f"{p.name} already exists; pick another probe season"
    p.write_text(json.dumps({
        "season": "2025-26", "upcoming": "2026-27",     # deliberately the WRONG season
        "options": {"somebody": {"type": "player_option", "end_season": "2026-27"}},
    }))
    try:
        got = utils._load_option_status(probe_season)
        assert got == {}, (
            "a file named for the right season but carrying another season's "
            "'upcoming' was trusted; only the filename is being checked")
        utils._OPTION_STATUS_CACHE.clear()
        p.write_text(json.dumps({
            "season": probe_season, "upcoming": "2099-00",
            "options": {"somebody": {"type": "player_option"}},
        }))
        got = utils._load_option_status(probe_season)
        assert "somebody" in got, (
            "a file whose metadata MATCHES its season was refused, so the check "
            "is rejecting on something other than the season")
    finally:
        p.unlink(missing_ok=True)


def test_nobody_guaranteed_for_the_upcoming_season_is_a_free_agent(flags, feed, ends, rookies):
    """THE INVARIANT BARRETT'S QUESTION MAPS TO, checked against a source the
    classifier does not read. data/contract_years_2026_27.csv is built from the
    Spotrac yearly pages. A G flag for the upcoming season is unambiguous: the money
    is guaranteed, so the player cannot be a free agent in that offseason."""
    gtd = [(n, f) for n, f in flags.items() if f.get(str(BOARD_YEAR)) == "G"]
    assert len(gtd) >= 100, (
        f"only {len(gtd)} players are recorded as guaranteed for {UPCOMING}; "
        f"contract_years looks broken and this test would pass vacuously")

    display = {normalize(r["player"]): r["player"] for r in _rows(CY) if r.get("player")}
    wrong = []
    for n, _f in gtd:
        st = _status(display[n], feed, ends, rookies)
        if st is not None:
            wrong.append(f"{display[n]} -> {st}")
    assert not wrong, (
        f"these have GUARANTEED money for {UPCOMING} and the classifier calls them "
        f"free agents: {wrong[:20]}")


def test_nobody_on_the_books_for_the_upcoming_season_is_a_free_agent(flags, feed, ends, rookies):
    """Non-guaranteed money is still money owed unless the player is waived, so an
    N flag means he is on the books and not a free agent by expiry.

    This used to pin two known exceptions by name. It no longer needs to: four
    players (Quenton Jackson, Dylan Cardwell, Johni Broome, Micah Peavy) carry
    2026:G|2027:N in contract_years while the contract-end scraper says their deal
    ends 2026-27, and the feed omits all four, so they reached the board as UFAs.
    utils.signed_through_season already treated G and N alike for the board's
    rescue gate; classify_fa_status now applies the same rule."""
    onbooks = [n for n, f in flags.items() if f.get(str(BOARD_YEAR)) in ("G", "N")]
    assert len(onbooks) >= 100, (
        f"only {len(onbooks)} players on the books for {UPCOMING}; contract_years "
        f"looks broken and this test would pass vacuously")
    nongtd = [n for n, f in flags.items() if f.get(str(BOARD_YEAR)) == "N"]
    assert len(nongtd) >= 5, (
        f"only {len(nongtd)} non-guaranteed {UPCOMING} seasons; the N half of this "
        f"test is not exercising anything")

    display = {normalize(r["player"]): r["player"] for r in _rows(CY) if r.get("player")}
    wrong = [f"{display[n]} ({flags[n].get(str(BOARD_YEAR))}) -> "
             f"{_status(display[n], feed, ends, rookies)}"
             for n in onbooks
             if n in display and _status(display[n], feed, ends, rookies) is not None]
    assert not wrong, (
        f"these are on the books for {UPCOMING} and classify as free agents: {wrong[:20]}")


def test_a_genuine_upcoming_option_is_still_an_option(flags, feed, ends, rookies):
    """THE REGRESSION THE CACHE EXISTED TO PREVENT. Gating a stale option source
    must not silently drop the option holders it was protecting. Rookie-scale
    option years are excluded: those are suppressed on purpose unless the board's
    'Include rookie-scale options' box is ticked."""
    display = {normalize(r["player"]): r["player"] for r in _rows(CY) if r.get("player")}
    holders = [n for n, f in flags.items()
               if f.get(str(BOARD_YEAR)) in ("P", "T") and n not in rookies]
    assert len(holders) >= 20, (
        f"only {len(holders)} non-rookie option holders found for {UPCOMING}; "
        f"this test is not exercising the regression")

    dropped = [display[n] for n in holders
               if n in display and _status(display[n], feed, ends, rookies) is None]
    assert not dropped, (
        f"{len(dropped)} of {len(holders)} players holding a real {UPCOMING} option "
        f"are classified as not-free-agents, so the option sources dropped the very "
        f"players they exist to protect: {dropped[:20]}")


def test_austin_reaves_and_victor_wembanyama_are_not_free_agents(feed, ends, rookies):
    """The report, pinned by name. Reaves re-signed in July 2026 through 2029-30;
    Wembanyama signed a max rookie extension starting 2027-28 and was number one on
    the board. Both were published as option holders off a closed 2026 decision."""
    for name, through in (("Austin Reaves", "2029-30"), ("Victor Wembanyama", "2031-32")):
        st = _status(name, feed, ends, rookies)
        assert st is None, (
            f"{name} is under contract to {through} and the classifier calls him "
            f"{st!r} for the {UPCOMING} offseason")


def test_the_loader_cannot_be_called_without_a_season():
    """Pins the signature. The bug was possible only because this function took no
    season and read one hardcoded filename, so a caller could not get it wrong --
    and could not get it right either."""
    with pytest.raises(TypeError):
        utils._load_option_status()


def test_every_caller_passes_a_season():
    """A new caller reintroducing the season-free read would restore the bug in
    exactly one line, and no other test would notice."""
    bad = []
    for p in sorted(ROOT.glob("*.py")) + sorted(ROOT.glob("pages/*.py")) + sorted(ROOT.glob("scripts/*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "_load_option_status()" in line:
                bad.append(f"{p.relative_to(ROOT)}:{i}")
    assert not bad, f"these call the loader with no season: {bad}"


def test_the_feed_fixture_reads_what_production_reads(feed):
    """GUARDS THE MISTAKE THIS FILE ALREADY MADE ONCE, and takes the `feed`
    fixture so that it is the FIXTURE under test rather than a second copy of its
    logic -- the first version of this guard recomputed _rev_path itself, so
    pointing the fixture back at the floor did not fail it.

    The floor is cache/<FEED_NAME>; production reads utils._rev_path(FEED_NAME),
    which prefers the DATA_REV copy. On 2026-11-20 the floor was seven days older
    than the served r49 file and disagreed about six option holders, which made a
    measurement claim the salary feed never mislabels an option. It does, and
    believing otherwise nearly cost Al Horford his row."""
    served = utils._rev_path(FEED_NAME)
    if not served.exists():
        pytest.skip("next-year feed cache absent")
    want = pickle.loads(served.read_bytes())
    assert feed == want, (
        f"the feed fixture does not hold what production serves. It has "
        f"{len(feed)} entries; {served.name} has {len(want)}. A fixture reading "
        f"the un-revisioned floor makes every measurement in this file describe a "
        f"file nobody serves.")


def test_an_option_the_feed_reads_as_guaranteed_is_still_an_option(flags, feed, ends, rookies):
    """THE REGRESSION THE STALE CACHE WAS MASKING, and the reason an option source
    still sits in front of the salary feed.

    Gating the stale cache alone dropped Al Horford: he holds a real 2027-28
    player option, the served feed reads him as 'guaranteed', and nothing else was
    left to say otherwise. Measured on the served feed there were SIX such
    players, which is why 'the feed labels options correctly now' was wrong."""
    display = {normalize(r["player"]): r["player"] for r in _rows(CY) if r.get("player")}
    misread = [n for n, f in flags.items()
               if f.get(str(BOARD_YEAR)) in ("P", "T", "M")
               and (feed.get(n) or {}).get("type") == "guaranteed"
               and n not in rookies]
    assert misread, (
        "the feed no longer reads any real option as guaranteed. If that is "
        "genuinely true the option sources may be removable, but verify it on the "
        "SERVED feed before believing it: reading the un-revisioned floor is how "
        "this was got wrong the first time")
    dropped = [display[n] for n in misread
               if n in display and _status(display[n], feed, ends, rookies) is None]
    assert not dropped, (
        f"the feed reads these players' real {UPCOMING} option as guaranteed and "
        f"nothing rescued them, so they are off the board: {dropped}")


def test_a_rookie_scale_option_year_is_suppressed_unless_asked_for(flags, feed, ends, rookies):
    """THE CHECKBOX. A rookie deal's year-3/4 team option is auto-exercised, so the
    player is not available: hidden by default, listed when the caller asks.

    This gate used to live inside the option-cache branch and so fired only for
    players in that cache. It therefore hid 17 players it should not have while
    never reaching the rookie options it exists for."""
    display = {normalize(r["player"]): r["player"] for r in _rows(CY) if r.get("player")}
    rookie_opts = [n for n in rookies
                   if flags.get(n, {}).get(str(BOARD_YEAR)) in ("P", "T", "M")
                   and n in display]
    assert len(rookie_opts) >= 10, (
        f"only {len(rookie_opts)} rookie-scale players hold a {UPCOMING} option; "
        f"this test is not exercising the checkbox")

    shown = [display[n] for n in rookie_opts
             if _status(display[n], feed, ends, rookies, include_rookie=False) is not None]
    assert not shown, (
        f"these hold an auto-exercised rookie-scale option and are on the board "
        f"with the box unticked: {shown[:20]}")

    hidden = [display[n] for n in rookie_opts
              if _status(display[n], feed, ends, rookies, include_rookie=True) is None]
    assert not hidden, (
        f"ticking 'Include rookie-scale options' did not bring these back, so the "
        f"box does nothing: {hidden[:20]}")


def test_an_expiring_rookie_contract_is_still_restricted_free_agency(flags, feed, ends, rookies):
    """THE OTHER DIRECTION, and the bug the old placement caused. A 2023 first
    rounder's rookie deal ENDS this season, so for this board he is a genuine RFA
    and the checkbox must not touch him. Anthony Black, Scoot Henderson and Gradey
    Dick were all being hidden with the box unticked."""
    display = {normalize(r["player"]): r["player"] for r in _rows(CY) if r.get("player")}
    pool = [n for n in rookies if n in display]
    rfa_off = {n for n in pool
               if _status(display[n], feed, ends, rookies, include_rookie=False) == "RFA"}
    rfa_on = {n for n in pool
              if _status(display[n], feed, ends, rookies, include_rookie=True) == "RFA"}
    assert len(rfa_off) >= 5, (
        f"only {len(rfa_off)} rookie-scale RFAs found; this test is not exercising "
        f"the distinction between an option year and an expiring rookie deal")
    assert rfa_off == rfa_on, (
        f"the rookie-scale checkbox is moving RFAs, which it must not: "
        f"{sorted(display[n] for n in rfa_off ^ rfa_on)[:20]}")


def test_the_rookie_gate_is_applied_after_the_sources_not_inside_one():
    """Pins the placement, because that WAS the bug: a gate inside one source
    branch applies to whichever players that source happens to know about."""
    src = (ROOT / "utils.py").read_text(encoding="utf-8")
    body = src.split("def _fa_status_from_sources", 1)[1].split("def classify_fa_status", 1)[0]
    assert "include_rookie_options" not in body, (
        "the rookie gate is back inside the resolver, so it applies only to the "
        "source that answered rather than to the answer")
    wrapper = src.split("def classify_fa_status", 1)[1].split("\ndef ", 1)[0]
    assert "include_rookie_options" in wrapper and "rookie_set" in wrapper, (
        "classify_fa_status no longer applies the rookie-scale gate")
