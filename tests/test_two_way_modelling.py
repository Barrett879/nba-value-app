"""Two-way contracts: all of them, and modelled as two-ways.

Written 2026-11-19, after Barrett asked why Arthur Kaluma was the only two-way
player on the Lakers' Trade Machine roster. He was, and the reason was that
cache/team_rosters_2627.json -- a hand-maintained file with no scheduled
builder, last written 2026-10-30 -- happened to carry his deal as a projected
re-signing and not Chris Manon's or Carson Cooper's. League-wide the Trade
Machine carried 18 of the league's 89 two-way players on its roster cards,
and 29 of 30 teams were wrong.

Two things were broken, and fixing either one alone made it worse:

  THE SET. master_roster is the only file in the repo that records two-way
  deals (the ESPN salary feed omits them). build_team_pages now emits ALL of
  them from that one authority, instead of keeping the 18 the spine happened
  to carry.

  THE MODELLING. trade_rules already excludes two-ways from salary matching
  and from the 15-man limit everywhere (CBA 1.8) -- salary_out, matching_out,
  salary_in, matchable_in, counted_out, the roster delta, and the Dec-15
  eligibility rule. None of it had ever fired, because templates/
  trade_machine.html hardcoded `two_way:false` for every player. And
  `roster_size` was fed the spine's `size`, which COUNTS two-ways, so TWELVE
  teams were already over the 15-man limit before any trade was proposed and
  tripped a phantom ROSTER_MAX on every deal. Setting the flag WITHOUT fixing
  the count made it worse, not better: LAL read 17 instead of 16.

So these tests pin both halves together, plus the template, because the JS is
a hand port of trade_rules.py with nothing else guarding it.

A SECOND ROUND, the same day, after an adversarial pre-push review found three
things the first round introduced or left open:

  THE NAME GAP. utils.NameIndex cannot bridge a HYPHEN against a SPACE: its key
  strips punctuation but keeps spaces, so "David Jones-Garcia" (the roster
  spine) and "David Jones Garcia" (master_roster, and the live NBA feed) share
  no key in EITHER direction. Both the skip and the collision guard missed, and
  the artifact published the same man on the Spurs card twice -- once as a
  standard "Re-sign" with no score -- which inflated San Antonio's standard
  count by one. build_team_pages now carries a letters-and-digits-only fallback
  key for identity only.

  THE UNBOUNDED SLOT. Making two-ways invisible to the 15-man count REMOVED the
  accident that was bounding them: with two_way hardcoded false, an incoming
  two-way counted as a standard contract and ROSTER_MAX caught it. Afterwards a
  team could be told "TRADE WORKS" while ending up with four, and 29 of 30
  teams carry exactly three. trade_rules and the JS port now check
  two_way_slots directly.

  THE PAYROLL BASELINE. Two-way money does not count against the cap, and the
  engine already excluded two-ways from salary_out/salary_in -- but their
  salaries stayed inside `payroll`, so sending one out credited nothing. With
  Indiana $0.6M under a hard cap and $2.0M of two-way money in its total, that
  is enough to call a legal trade illegal.
"""
from __future__ import annotations
import json
import logging
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.disable(logging.WARNING)
import trade_rules as tr  # noqa: E402
from utils import NameIndex, two_way_players  # noqa: E402
logging.disable(logging.NOTSET)

PAGES = ROOT / "cache" / "team_pages.json"
TMPL = ROOT / "templates" / "trade_machine.html"
CFG = tr.CBAConfig.load("2026-27")
SEASON = "2026-27"


@pytest.fixture(scope="module")
def teams():
    if not PAGES.exists():
        pytest.skip("cache/team_pages.json absent")
    return json.loads(PAGES.read_text())["teams"]


# ── the set ──────────────────────────────────────────────────────────────────

def test_every_master_roster_two_way_reaches_the_trade_machine(teams):
    """The defect Barrett saw: 19 of 89 present, the 19 arbitrary."""
    want = set(two_way_players())
    have = {p["n"] for t in teams.values() for p in t["players"] if p["two_way"]}
    missing = sorted(want - have)
    assert not missing, (
        f"{len(missing)} two-way player(s) on no Trade Machine roster: "
        f"{missing[:12]}. master_roster is the only file that records two-way "
        f"deals, so a player absent here cannot be traded on the site at all.")


def test_no_two_way_card_is_invented(teams):
    """The converse: nothing is a two-way unless master_roster says so."""
    ni = NameIndex({n: True for n in two_way_players()})
    bogus = sorted(p["n"] for t in teams.values() for p in t["players"]
                   if p["two_way"] and ni.get(p["n"]) is None)
    assert not bogus, f"flagged two-way but not in master_roster: {bogus}"


def test_two_way_slots_are_respected(teams):
    """Three per team (cba_config two_way_slots). The builder refuses past it,
    and that guard earned its place twice on the day it was written: once when
    a fallback row's stale Team put Drew Timme and Kobe Bufkin on the Lakers,
    and once when the spine still had Sean Pedulla on Houston after
    master_roster moved him to Sacramento."""
    over = {a: n for a, n in
            ((a, sum(1 for p in t["players"] if p["two_way"]))
             for a, t in teams.items()) if n > CFG.two_way_slots}
    assert not over, f"teams over {CFG.two_way_slots} two-way slots: {over}"


def test_nobody_is_on_two_rosters(teams):
    seen: dict[str, str] = {}
    dupes = []
    for a, t in teams.items():
        for p in t["players"]:
            if p["n"] in seen:
                dupes.append((p["n"], seen[p["n"]], a))
            seen[p["n"]] = a
    assert not dupes, f"players on two roster cards: {dupes}"


# ── the count ────────────────────────────────────────────────────────────────

def test_std_size_counts_standard_contracts_only(teams):
    """std_size is what roster_size is fed. It must be the standard count."""
    wrong = {a: (t["std_size"], sum(1 for p in t["players"] if not p["two_way"]))
             for a, t in teams.items()
             if t["std_size"] != sum(1 for p in t["players"] if not p["two_way"])}
    assert not wrong, f"std_size != non-two-way card count (got, want): {wrong}"


def test_std_size_is_not_just_size_minus_two_ways(teams):
    """`size` also counts players dropped as free agents. Subtracting only the
    two-ways left LAC reading 16 standard contracts, because waived Jamarion
    Sharp was still inside its 17. Counting is the fix; this pins it."""
    assert any(t["std_size"] != t["size"] - sum(1 for p in t["players"] if p["two_way"])
               for t in teams.values()), (
        "std_size looks like a subtraction from `size` again; it must be a "
        "direct count of the standard contracts actually emitted")


def test_no_team_is_over_the_roster_limit_before_a_trade(teams):
    """Twelve teams were: ten from counting two-ways as standard contracts,
    and two more (HOU, LAC) from counting players the builder had already
    dropped as free agents. A team already over the limit trips ROSTER_MAX on
    any trade it appears in."""
    over = {a: t["std_size"] for a, t in teams.items()
            if t["std_size"] > CFG.roster_max_inseason}
    assert not over, (
        f"teams over the {CFG.roster_max_inseason}-man limit with no trade "
        f"proposed: {over}")


# ── the modelling ────────────────────────────────────────────────────────────

def _swap(a_size, b_size, out_p, in_p):
    a = tr.TradeTeam(abbr="AAA", payroll=100.0, roster_size=a_size,
                     out_players=[out_p], in_players=[in_p])
    b = tr.TradeTeam(abbr="BBB", payroll=100.0, roster_size=b_size,
                     out_players=[in_p], in_players=[out_p])
    return tr.Trade(teams=[a, b], trade_date="2026-11-19")


def test_two_way_for_two_way_is_legal_on_a_full_roster():
    tw = lambda n: tr.TradePlayer(n, 0.678882, two_way=True)  # noqa: E731
    v, _ = tr.check_cash_and_roster(
        _swap(CFG.roster_max_inseason, CFG.roster_max_inseason,
              tw("Arthur Kaluma"), tw("Javon Small")), CFG)
    assert not v, (
        "a two-way-for-two-way swap must not touch the 15-man count: "
        f"{[x.code for x in v]}")


def test_a_standard_player_into_a_full_roster_is_still_caught():
    """The fix must not become a hole."""
    v, _ = tr.check_cash_and_roster(
        _swap(CFG.roster_max_inseason, CFG.roster_max_inseason,
              tr.TradePlayer("Arthur Kaluma", 0.678882, two_way=True),
              tr.TradePlayer("A Standard Guy", 5.0)), CFG)
    assert [x.code for x in v] == ["ROSTER_MAX"], [x.code for x in v]


def test_a_two_way_salary_does_not_match_salary():
    t = tr.TradeTeam(abbr="AAA", payroll=100.0, roster_size=15,
                     out_players=[tr.TradePlayer("tw", 0.678882, two_way=True)],
                     in_players=[tr.TradePlayer("tw2", 0.678882, two_way=True)])
    assert t.salary_out == 0.0
    assert t.salary_in == 0.0
    assert t.matching_out(CFG) == 0.0
    assert t.matchable_in() == 0.0
    assert t.counted_out() == []


# ── the JS port, which nothing else guards ───────────────────────────────────

def _js() -> str:
    return TMPL.read_text(encoding="utf-8")


def test_the_js_port_reads_the_two_way_flag():
    """It hardcoded `two_way:false` for every roster player, so every exclusion
    in trade_rules was dead code on the live page."""
    js = _js()
    assert "two_way:!!p.two_way" in js, (
        "templates/trade_machine.html no longer reads the per-player two_way "
        "flag when building a roster player; trade_rules' two-way handling is "
        "dead code again")
    # the sign-and-trade branch is a standard contract by definition and stays
    # false, so exactly one hardcoded false is expected.
    assert js.count("two_way:false") == 1, (
        f"expected exactly one deliberate `two_way:false` (the sign-and-trade "
        f"branch); found {js.count('two_way:false')}")


def test_the_js_port_feeds_std_size_to_the_engine():
    assert "std_size" in _js(), (
        "the JS must take roster_size from std_size, not `size`; `size` counts "
        "two-ways and put ten teams over the 15-man limit")
    assert not re.search(r"roster_size:DATA\.teams\[a\]\.size\b", _js()), (
        "roster_size is being fed `size` again")


def test_the_payload_carries_the_flag_the_js_now_reads():
    page = (ROOT / "pages" / "Trade_Machine.py").read_text(encoding="utf-8")
    assert '"two_way": bool(p.get("two_way"))' in page
    assert '"std_size"' in page


# ── second round: the name gap, the slot limit, the payroll baseline ─────────

def _hard(n):
    return re.sub(r"[^a-z0-9]", "", str(n).lower())


def test_no_player_is_duplicated_under_a_punctuation_insensitive_key(teams):
    """The blocker: NameIndex cannot bridge a hyphen against a space.

    test_nobody_is_on_two_rosters compares the raw `n` strings, so two
    spellings of one man read as two players and it passed while the artifact
    carried David Jones-Garcia AND David Jones Garcia on San Antonio.
    """
    seen, dupes = {}, []
    for abbr, t in teams.items():
        for p in t["players"]:
            k = _hard(p["n"])
            if k in seen:
                dupes.append((p["n"], seen[k], abbr))
            seen[k] = (abbr, p["n"])
    assert not dupes, (
        f"same player on two cards under different spellings: {dupes}")


def _master_roster_counts():
    import csv as _csv
    out = {}
    for r in _csv.DictReader(
            l for l in (ROOT / "data" / "master_roster.csv").read_text(
                encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")):
        kind = (r.get("kind") or "").strip()
        team = (r.get("team") or "").strip()
        if r.get("player") and team:
            out.setdefault(team, {})
            out[team][kind] = out[team].get(kind, 0) + 1
    return out


def test_std_size_never_exceeds_the_master_roster_standard_count(teams):
    """The tell that caught the duplicate.

    A ghost row inflates std_size above what master_roster actually says the
    team carries, and an inflated std_size is exactly the phantom ROSTER_MAX
    this change set out to delete. Running LOW is allowed: the spine lags
    master_roster and misses recent signings (CHI and OKC do today), which is
    conservative. Running HIGH means the artifact invented a contract.
    """
    mr = _master_roster_counts()
    high = {a: (t["std_size"], mr.get(a, {}).get("standard", 0))
            for a, t in teams.items()
            if t["std_size"] > mr.get(a, {}).get("standard", 0)}
    assert not high, (
        f"std_size exceeds master_roster's standard count (got, want): {high}")


def test_two_way_size_is_published_and_correct(teams):
    """The engine needs it; nothing else bounds the two-way count."""
    missing = [a for a, t in teams.items() if "two_way_size" not in t]
    assert not missing, f"teams with no two_way_size: {missing}"
    wrong = {a: (t["two_way_size"], sum(1 for p in t["players"] if p["two_way"]))
             for a, t in teams.items()
             if t["two_way_size"] != sum(1 for p in t["players"] if p["two_way"])}
    assert not wrong, f"two_way_size != two-way card count (got, want): {wrong}"
    over = {a: t["two_way_size"] for a, t in teams.items()
            if t["two_way_size"] > CFG.two_way_slots}
    assert not over, f"teams over {CFG.two_way_slots} two-way slots: {over}"


def test_a_fourth_two_way_is_refused():
    """The regression the first round introduced.

    With two_way hardcoded false, an incoming two-way counted as a standard
    contract and ROSTER_MAX bounded it by accident. Excluding two-ways from the
    15-man count removed that bound and nothing replaced it, so a team at three
    could take a fourth and the page rendered "TRADE WORKS".
    """
    full = tr.TradeTeam(
        abbr="AAA", payroll=100.0, roster_size=14,
        two_way_size=CFG.two_way_slots,
        out_players=[], in_players=[tr.TradePlayer("tw", 0.678882, two_way=True)])
    other = tr.TradeTeam(
        abbr="BBB", payroll=100.0, roster_size=15, two_way_size=3,
        out_players=[tr.TradePlayer("tw", 0.678882, two_way=True)], in_players=[])
    v, _ = tr.check_cash_and_roster(
        tr.Trade(teams=[full, other], trade_date="2026-11-19"), CFG)
    assert "TWO_WAY_SLOTS" in [x.code for x in v], (
        f"a {CFG.two_way_slots + 1}th two-way was allowed: {[x.code for x in v]}")


def test_a_one_for_one_two_way_swap_keeps_the_slot_count_legal():
    """The fix must not forbid the ordinary case."""
    tw = lambda n: tr.TradePlayer(n, 0.678882, two_way=True)  # noqa: E731
    a = tr.TradeTeam(abbr="AAA", payroll=100.0, roster_size=15,
                     two_way_size=CFG.two_way_slots,
                     out_players=[tw("out")], in_players=[tw("in")])
    b = tr.TradeTeam(abbr="BBB", payroll=100.0, roster_size=15,
                     two_way_size=CFG.two_way_slots,
                     out_players=[tw("in")], in_players=[tw("out")])
    v, _ = tr.check_cash_and_roster(
        tr.Trade(teams=[a, b], trade_date="2026-11-19"), CFG)
    assert not v, [x.code for x in v]


def test_the_js_port_checks_two_way_slots():
    js = _js()
    assert "TWO_WAY_SLOTS" in js, (
        "the JS port has no two-way slot check; trade_rules does, so the live "
        "page would say TRADE WORKS where the engine says it does not")
    assert "two_way_size" in js, "the JS does not read two_way_size"


def test_payroll_excludes_two_way_money(teams):
    """Two-way salary does not count against the cap, and the engine excludes
    two-ways from salary_out/salary_in, so leaving their money in the baseline
    meant sending one out credited nothing."""
    spine_path = ROOT / "cache" / "team_rosters_2627.json"
    if not spine_path.exists():
        pytest.skip("roster spine absent")
    spine = json.loads(spine_path.read_text())["teams"]
    ni = NameIndex({n: True for n in two_way_players()})
    for abbr, t in teams.items():
        sp = spine.get(abbr)
        if not sp:
            continue
        tw_money = sum(
            (p.get("salary") or 0) for p in sp["players"]
            if ni.get(p["n"]) is not None or _hard(p["n"]) in {
                _hard(x) for x in two_way_players()})
        assert abs(t["payroll"] - round(sp["total"] - tw_money, 1)) < 0.051, (
            f"{abbr}: payroll {t['payroll']} should be spine total "
            f"{sp['total']} minus {tw_money:.1f} of two-way money")


def test_the_two_way_chip_follows_master_roster_not_contract_years():
    """contract_years is scraped and goes stale. Cam Christie's stale LAC row
    is the $2.30M team-option deal his two-way replaced, so testing only for a
    MISSING row let the join hit and rendered "2NG" on a two-way card while the
    engine traded him as a two-way."""
    page = (ROOT / "pages" / "Trade_Machine.py").read_text(encoding="utf-8")
    assert 'if p.get("two_way"):' in page, (
        "the two-way chip must be set unconditionally from the two_way flag")
    assert '_yrs is None and p.get("two_way")' not in page, (
        "the chip is conditioned on a MISSING contract_years row again; a "
        "stale row on the same team will suppress it")


def test_the_two_way_scale_cannot_silently_become_zero():
    """`.get(season, 0)` would price all 89 two-ways at $0.0M the first season
    nobody adds a key, and check_season_surfaces guards SEASONS[0] while the
    builder reads the spine's contract_season -- keys that drift every July."""
    builder = (ROOT / "scripts" / "build_team_pages.py").read_text(encoding="utf-8")
    assert "TWO_WAY_SALARY.get(" not in builder, (
        "the two-way scale is read with a silent default again")
    spine_path = ROOT / "cache" / "team_rosters_2627.json"
    if spine_path.exists():
        season = json.loads(spine_path.read_text())["contract_season"]
        from utils import TWO_WAY_SALARY
        assert season in TWO_WAY_SALARY, (
            f"TWO_WAY_SALARY has no entry for the spine's contract_season "
            f"{season}; the builder will refuse")
