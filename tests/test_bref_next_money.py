"""Basketball Reference as the second source for next-season money.

Spotrac began answering 403 to this machine on 2026-11-17, on every path
including robots.txt, and a single source for the Next $ column is what turned
one block into a refresh that shipped nothing. scripts/bref_next_money.py is
the second source. These tests pin the two things about it that fail SILENTLY.

  * THE SEASON COLUMN IS DERIVED. The page names its season columns y1..y6
    positionally, so "y2" means 2027-28 only until the season flips. A
    hardcoded y2 would then quote 2028-29 money as next season's: every row
    would parse, every count would look right, and nothing would refuse. Half
    these tests exist to make that un-hardcodeable, and the label-shift test
    FAILS on a parser that pins the index.
  * REFUSAL BEATS A PLAUSIBLE NUMBER. A wrong salary is indistinguishable from
    a real one once it reaches the column, and an absent player renders as the
    gray sentinel the legend defines as "UFA". Both are invisible, so every
    shape this parser does not understand has to raise.

The fixtures here are synthetic and deliberately small. The live page is NOT
committed as a fixture: a cached payload cannot see a format change, and the
live shape is checked by the builder's own refusals on every nightly run.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location(
    "bref_next_money", ROOT / "scripts" / "bref_next_money.py")
B = importlib.util.module_from_spec(spec)
sys.modules["bref_next_money"] = B
spec.loader.exec_module(B)


# --------------------------------------------------------------------------
# A structurally faithful miniature of /contracts/players.html.
# `labels` is the season header, in page order, so a test can SHIFT which
# y<N> carries 2027-28 and catch a parser that pinned the index.
# --------------------------------------------------------------------------
def page(rows, labels=("2026-27", "2027-28", "2028-29"), table_id=B.TABLE_ID,
         with_tbody=True, repeat_header=False):
    ys = [f"y{i+1}" for i in range(len(labels))]
    head = ("<tr><th data-stat='ranker'>Rk</th>"
            "<th data-stat='player'>Player</th>"
            "<th data-stat='team_id'>Tm</th>"
            + "".join(f"<th data-stat='{y}'>{l}</th>" for y, l in zip(ys, labels))
            + "<th data-stat='remain_gtd'>Guaranteed</th></tr>")
    body = []
    if repeat_header:
        body.append(f"<tr class='thead'>{head}</tr>")
    for i, r in enumerate(rows, 1):
        cells = []
        for y in ys:
            raw, cls = r.get(y, ("", ["right", "iz"]))
            cells.append(f"<td class='{' '.join(cls)}' data-stat='{y}'>{raw}</td>")
        body.append(
            f"<tr><th data-stat='ranker'>{i}</th>"
            f"<td class='left' data-stat='player'>{r['player']}</td>"
            f"<td class='left' data-stat='team_id'>{r.get('team','BOS')}</td>"
            + "".join(cells)
            + f"<td data-stat='remain_gtd'>{r.get('gtd','$1,000,000')}</td></tr>")
    inner = "".join(body)
    guts = f"<tbody>{inner}</tbody>" if with_tbody else inner
    return (f"<html><body><table id='{table_id}'>"
            f"<thead>{head}</thead>{guts}</table></body></html>")


def money(raw, *extra):
    return (raw, ["right", *extra])


def filler(n, start=1, y="y2"):
    """Enough plausible rows to clear MIN_ROWS without being the subject.

    `y` names the column the money goes in. It is a PARAMETER because the
    shifted-header test moves 2027-28 to y3, and filler money welded to y2
    would land in 2026-27 there -- leaving one paid row, so the parser would
    refuse on MIN_ROWS and the test would pass for the wrong reason.
    """
    return [{"player": f"Filler Player {i}", "team": "BOS",
             y: money(f"${10 + i % 40},{100 + i:03d},000")}
            for i in range(start, start + n)]


# ==========================================================================
# the season column is DERIVED from the label, never from the position
# ==========================================================================
def test_resolves_the_requested_season_by_label():
    rows = [{"player": "Alpha One", "y1": money("$1,000,000"),
             "y2": money("$2,000,000"), "y3": money("$3,000,000")}] + filler(300)
    got = B.parse(page(rows), "2027-28")
    assert got["alpha one"][0]["salary"] == 2_000_000


def test_a_shifted_header_moves_the_column_with_it():
    """The SAME request, a page where 2027-28 has moved from y2 to y3.

    This is the season-flip shape. A parser that reads y2 returns $2,000,000
    here, which is 2026-27 money served as next season's -- plausible, wrong,
    and silent. Correct behaviour is $3,000,000.
    """
    rows = [{"player": "Alpha One", "y1": money("$1,000,000"),
             "y2": money("$2,000,000"), "y3": money("$3,000,000")}]
    # Filler in BOTH columns on purpose. With money only in y3, a parser
    # pinned to y2 finds one paid row and trips MIN_ROWS -- caught, but caught
    # as a refusal, when the failure being pinned here is a PLAUSIBLE WRONG
    # FIGURE that refuses nothing. Paying both columns leaves the pinned
    # parser 300 happy rows and one wrong salary, so the assertion is what
    # fires. Verified by mutation: hardcoding col="y2" fails on the assert
    # below, not on the floor.
    pad = [{**r, "y3": money("$7,000,000")} for r in filler(300)]
    html = page(rows + pad, labels=("2025-26", "2026-27", "2027-28"))
    got = B.parse(html, "2027-28")
    assert got["alpha one"][0]["salary"] == 3_000_000, (
        "2027-28 sat at y3 on this page; a pinned y2 returned 2026-27 money "
        "and nothing refused")
    assert got["filler player 1"][0]["salary"] == 7_000_000


def test_season_columns_maps_every_label():
    import bs4
    tbl = bs4.BeautifulSoup(page(filler(1), labels=("2030-31", "2031-32")),
                            "lxml").find("table")
    assert B.season_columns(tbl) == {"2030-31": "y1", "2031-32": "y2"}


def test_refuses_when_the_requested_season_is_absent():
    """Never falls back to an adjacent column. Quoting 2028-29 money as
    2027-28's is the exact failure build_next_contracts.py's season-literal
    comment warns about, and it cannot be noticed downstream."""
    with pytest.raises(B.SourceRefused) as e:
        B.parse(page(filler(300), labels=("2026-27", "2028-29")), "2027-28")
    assert "2027-28" in str(e.value)


# ==========================================================================
# option type comes from the cell CLASS, which is why this source was chosen
# ==========================================================================
@pytest.mark.parametrize("cls,expected", [
    ("salary-pl", "player_option"),
    ("salary-tm", "team_option"),
    (None, "guaranteed"),
])
def test_option_type_from_the_cell_class(cls, expected):
    r = {"player": "Opt Player",
         "y2": money("$9,000,000", *( [cls] if cls else [] ))}
    got = B.parse(page([r] + filler(300)), "2027-28")
    assert got["opt player"][0]["type"] == expected


def test_the_class_is_read_off_the_right_season_only():
    """A player option in 2028-29 must not make 2027-28 an option.

    Stephen Curry is exactly this on the live page: 2027-28 plain, 2028-29 a
    player option. Reading the row's classes instead of the cell's would turn
    guaranteed money into an option, which the board renders as a different
    contract.
    """
    r = {"player": "Curry Shape", "y1": money("$62,587,158"),
         "y2": money("$55,700,000"), "y3": money("$60,156,000", "salary-pl")}
    got = B.parse(page([r] + filler(300)), "2027-28")
    assert got["curry shape"][0]["type"] == "guaranteed"
    assert got["curry shape"][0]["salary"] == 55_700_000


# ==========================================================================
# absence is meaningful: an empty cell is a UFA, not a zero
# ==========================================================================
def test_an_empty_cell_yields_no_salary_and_a_named_flag():
    got = B.parse(page([{"player": "Expiring Deal", "y1": money("$5,000,000")}]
                       + filler(300)), "2027-28")
    row = got["expiring deal"][0]
    assert row["salary"] is None and row["flag"] == "empty"


def test_an_unrecognised_cell_is_flagged_never_guessed():
    got = B.parse(page([{"player": "Odd Cell", "y2": money("Two-Way")}]
                       + filler(300)), "2027-28")
    row = got["odd cell"][0]
    assert row["salary"] is None
    assert row["flag"].startswith("unparsed:") and "Two-Way" in row["flag"]


def test_money_must_match_the_whole_cell():
    """Anchored on purpose. The Spotrac path's hard-won bug was a greedy
    pattern eating back into the salary on a cell with the cap percentage
    welded on ("$28,006,89815.9%" -> $28,006). Here anything with extra
    characters is refused outright rather than silently truncated."""
    got = B.parse(page([{"player": "Glued Pct", "y2": money("$28,006,89815.9%")}]
                       + filler(300)), "2027-28")
    assert got["glued pct"][0]["salary"] is None


# ==========================================================================
# one player, several teams: every row survives the parse
# ==========================================================================
def test_a_split_contract_keeps_both_rows():
    """This page lists a waived-and-stretched player ONCE PER TEAM still
    carrying part of the deal, with identical year columns and a different
    Guaranteed total (Lillard on MIL and POR). Collapsing here would pick
    arbitrarily between two rows and hide that there were two."""
    rows = [{"player": "Split Deal", "team": "MIL",
             "y2": money("$36,620,603", "salary-pl"), "gtd": "$90,066,412"},
            {"player": "Split Deal", "team": "POR",
             "y2": money("$36,620,603", "salary-pl"), "gtd": "$13,398,800"}]
    got = B.parse(page(rows + filler(300)), "2027-28")
    assert len(got["split deal"]) == 2
    assert {r["team"] for r in got["split deal"]} == {"MIL", "POR"}
    assert {r["guaranteed"] for r in got["split deal"]} == {"$90,066,412",
                                                            "$13,398,800"}


def test_repeated_header_rows_are_not_players():
    """Both signals, because only one of them is guaranteed.

    page() emits the repeat header as a <tr> nested inside a <tr>, which lxml
    un-nests into a bare row carrying the header's cells and NO class="thead".
    That is the shape the class check cannot see, and it caught a real gap:
    the parser emitted a player literally named "Player". It is now skipped on
    the header's own label for that column instead.
    """
    got = B.parse(page(filler(300), repeat_header=True), "2027-28")
    assert "player" not in got, (
        'a repeated header reached the body as a player named "Player"')
    assert "rk" not in got
    assert "filler player 1" in got


def test_a_header_row_that_lost_its_class_is_still_skipped():
    """Stated directly, not as a side effect of how page() nests rows."""
    leak = ("<tr><th data-stat='ranker'>Rk</th>"
            "<th data-stat='player'>Player</th>"
            "<td data-stat='y2'>$9,999,999</td></tr>")
    html = page(filler(300)).replace("<tbody>", "<tbody>" + leak)
    got = B.parse(html, "2027-28")
    assert "player" not in got


# ==========================================================================
# refusals. every one of these would otherwise read as "no money next season"
# ==========================================================================
def test_refuses_a_missing_table():
    with pytest.raises(B.SourceRefused):
        B.parse(page(filler(300), table_id="renamed-table"), "2027-28")


def test_refuses_a_missing_tbody():
    with pytest.raises(B.SourceRefused):
        B.parse(page(filler(300), with_tbody=False), "2027-28")


def test_refuses_too_few_paid_rows():
    with pytest.raises(B.SourceRefused) as e:
        B.parse(page(filler(10)), "2027-28")
    assert str(B.MIN_ROWS) in str(e.value)


def test_refuses_an_absurd_salary():
    rows = [{"player": "Tiny Money", "y2": money("$32")}] + filler(300)
    with pytest.raises(B.SourceRefused) as e:
        B.parse(page(rows), "2027-28")
    assert "Tiny Money" in str(e.value)


def test_a_stretched_waiver_is_NOT_refused():
    """The floor that guards this parser is a backstop, not the league
    minimum. Didi Louzada is owed $268,032 in 2027-28 -- a stretched waiver,
    real money, far under the minimum. A $500,000 floor copied from the
    Spotrac path refused the entire live page over that one row, which is a
    source that never loads."""
    rows = [{"player": "Stretched Guy", "y1": money("$268,032"),
             "y2": money("$268,032"), "y3": money("$268,032"),
             "gtd": "$804,096"}] + filler(300)
    got = B.parse(page(rows), "2027-28")
    assert got["stretched guy"][0]["salary"] == 268_032


def test_the_positive_control_parses():
    """If this fails, every refusal test above proves nothing."""
    got = B.parse(page(filler(300)), "2027-28")
    assert len(got) == 300


# ==========================================================================
# the page checked against ITSELF: no second source required
# ==========================================================================
def test_a_guarantee_above_all_listed_money_is_impossible_and_refuses():
    """The only check here that needs no second source.

    Guaranteed is a TOTAL of the year columns, so the series can never sum to
    less than it. If a season column is ever added and this parser does not
    read it, the missing money shows up as a shortfall against the guarantee
    on many rows at once -- which is the difference between "a figure is
    wrong" (a second source can tell us) and "the parse is incomplete" (only
    this can).
    """
    rows = [{"player": f"Shortfall {i}", "y2": money("$5,000,000"),
             "gtd": "$99,000,000"} for i in range(B.MAX_SHORTFALLS + 1)]
    with pytest.raises(B.SourceRefused) as e:
        B.parse(page(rows + filler(300)), "2027-28")
    assert "arithmetically impossible" in str(e.value)


def test_a_single_shortfall_is_reported_not_refused(capsys):
    """Refusing the whole league over one source-side typo is how a source
    stops loading. Many shortfalls mean the parser; one means the page."""
    rows = [{"player": "Odd Guarantee", "y2": money("$5,000,000"),
             "gtd": "$99,000,000"}]
    got = B.parse(page(rows + filler(300)), "2027-28")
    assert "odd guarantee" in got
    assert "Odd Guarantee" in capsys.readouterr().out


def test_money_above_the_guarantee_is_normal_and_passes():
    """204 of 431 rows on the live page sum HIGHER than their guarantee,
    because option years and non-guaranteed years are real money that is not
    guaranteed. A check that flagged those would flag half the league."""
    rows = [{"player": "Optioned Guy", "y1": money("$10,000,000"),
             "y2": money("$11,000,000", "salary-pl"), "gtd": "$10,000,000"}]
    got = B.parse(page(rows + filler(300)), "2027-28")
    assert got["optioned guy"][0]["salary"] == 11_000_000


# ==========================================================================
# encoding. a mojibaked name is a player with no contract
# ==========================================================================
def test_diacritics_survive_the_parse():
    rows = [{"player": "Nikola Jokić", "y2": money("$62,841,702", "salary-pl")}]
    got = B.parse(page(rows + filler(300)), "2027-28")
    assert any("joki" in k for k in got), sorted(got)[:5]
    hit = [k for k in got if "joki" in k][0]
    assert got[hit][0]["display"] == "Nikola Jokić"
    assert got[hit][0]["type"] == "player_option"


def test_fetch_decodes_through_the_repo_helper():
    """Not requests' .text. requests falls back to latin-1 with no charset
    header, and a mojibaked name drops out of every NameIndex join."""
    import inspect
    src = inspect.getsource(B.fetch)
    assert "_html_text" in src and ".text" not in src.replace("_html_text", "")


# ==========================================================================
# the builder side: the source CHAIN and the policy that sits above the parse
# ==========================================================================
BN_SPEC = importlib.util.spec_from_file_location(
    "build_next_contracts", ROOT / "scripts" / "build_next_contracts.py")


def _bn():
    """build_next_contracts, imported fresh.

    Imported lazily and per-test: it pulls in utils, which is heavy, and these
    tests monkeypatch module constants on it.
    """
    mod = importlib.util.module_from_spec(BN_SPEC)
    sys.modules["build_next_contracts"] = mod
    BN_SPEC.loader.exec_module(mod)
    return mod


def test_every_source_declares_its_own_salary_floor():
    """The floor is a property of the PARSER, not of the money.

    Spotrac welds the cap percentage onto the figure, so a pattern that stops
    early returns a plausible fraction ($28,006 of $28M) and that path needs a
    floor above its own truncations. Basketball Reference's pattern is anchored
    to the whole cell and cannot truncate, so the same floor there refused the
    entire live page over a real $268,032 stretched waiver. One floor for both
    sources is the bug; this pins that they differ.
    """
    bn = _bn()
    floors = {label: floor for label, _, floor in bn.SOURCES}
    assert len(bn.SOURCES[0]) == 3, "a source is (label, fn, floor)"
    spotrac = [f for l, f in floors.items() if "Spotrac" in l][0]
    bref = [f for l, f in floors.items() if "Basketball Reference" in l][0]
    assert spotrac == 500_000
    assert bref < spotrac, (
        "Basketball Reference's floor must be below Spotrac's, or a real "
        "stretched waiver refuses the whole source")
    assert bref == B.MIN_PLAUSIBLE, "taken from the parser, not re-typed"


def test_spotrac_is_tried_before_basketball_reference():
    """Order is deliberate: contract_years_*.csv was compiled from Spotrac and
    is keyed to its spellings, so the chain must self-heal to Spotrac the day
    it stops blocking us, without anyone editing the file."""
    bn = _bn()
    labels = [l for l, _, _ in bn.SOURCES]
    assert "Spotrac" in labels[0] and "Basketball Reference" in labels[1]


def test_scrape_league_falls_through_a_dead_source():
    bn = _bn()
    calls = []

    def dead():
        calls.append("dead")
        return None

    def alive():
        calls.append("alive")
        return {"somebody": ("active", "BOS", "Somebody", 1.0, None)}, {"somebody": "team_option"}

    bn.SOURCES = (("dead one", dead, 500_000), ("live one", alive, 9_000))
    league, types, label, floor = bn.scrape_league()
    assert calls == ["dead", "alive"]
    assert label == "live one" and floor == 9_000
    assert types == {"somebody": "team_option"}


def test_scrape_league_reports_total_failure_distinguishably():
    """Not an empty league. An empty league means "nobody has money next
    season", which renders as a league of free agents and is the exact failure
    this file exists to prevent."""
    bn = _bn()
    bn.SOURCES = (("dead", lambda: None, 500_000),)
    league, types, label, floor = bn.scrape_league()
    assert league is None and label is None


def test_previous_rfa_orders_by_revision_number_not_mtime(tmp_path, monkeypatch):
    """r9 can be newer ON DISK than r40 after a carry-forward or an iCloud
    re-materialisation, so mtime ordering silently reads a nine-revision-old
    file. scripts/nightly.py's newest_revision carries the same warning."""
    bn = _bn()
    monkeypatch.setattr(bn.utils, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(bn.utils, "DATA_REV", "r99")
    bn.utils._pkl_save(tmp_path / "next_contracts_2027_v7_r40.pkl",
                       {"right guy": {"salary": 0.0, "type": "rfa"}})
    bn.utils._pkl_save(tmp_path / "next_contracts_2027_v7_r9.pkl",
                       {"wrong guy": {"salary": 0.0, "type": "rfa"}})
    # r9 touched LAST, so it is the newest by mtime and the oldest by revision
    import os, time as _t
    os.utime(tmp_path / "next_contracts_2027_v7_r9.pkl", (_t.time() + 60,) * 2)
    got = bn._previous_rfa(2027)
    assert set(got) == {"right guy"}, (
        "ordered by mtime, this reads r9 and carries the wrong holds forward")


def test_previous_rfa_excludes_the_revision_being_built(tmp_path, monkeypatch):
    """Reading the file this build is about to write makes the carry-forward a
    no-op that silently reproduces whatever is already there -- including a
    carried-forward copy from a previous failed night."""
    bn = _bn()
    monkeypatch.setattr(bn.utils, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(bn.utils, "DATA_REV", "r44")
    bn.utils._pkl_save(tmp_path / "next_contracts_2027_v7_r44.pkl",
                       {"mine": {"salary": 0.0, "type": "rfa"}})
    bn.utils._pkl_save(tmp_path / "next_contracts_2027_v7_r43.pkl",
                       {"theirs": {"salary": 0.0, "type": "rfa"}})
    assert set(bn._previous_rfa(2027)) == {"theirs"}


def test_previous_rfa_keeps_only_rfa_records(tmp_path, monkeypatch):
    bn = _bn()
    monkeypatch.setattr(bn.utils, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(bn.utils, "DATA_REV", "r99")
    bn.utils._pkl_save(tmp_path / "next_contracts_2027_v7_r5.pkl", {
        "held": {"salary": 0.0, "type": "rfa"},
        "paid": {"salary": 20_000_000.0, "type": "guaranteed"}})
    assert set(bn._previous_rfa(2027)) == {"held"}


def test_bref_team_codes_are_spoken_in_the_repo_dialect(monkeypatch):
    """Basketball Reference says BRK/CHO/PHO where this repo says BKN/CHA/PHX.
    Left raw, the builder's `moved` report announces that every Net, Hornet
    and Sun changed teams."""
    bn = _bn()
    monkeypatch.setattr(bn.bref_next_money, "league_money", lambda label: {
        "net guy": [{"display": "Net Guy", "team": "BRK", "salary": 5e6,
                     "type": "guaranteed", "flag": None, "guaranteed": "",
                     "years": {}}],
        "hornet guy": [{"display": "Hornet Guy", "team": "CHO", "salary": 5e6,
                        "type": "guaranteed", "flag": None, "guaranteed": "",
                        "years": {}}],
        "sun guy": [{"display": "Sun Guy", "team": "PHO", "salary": 5e6,
                     "type": "guaranteed", "flag": None, "guaranteed": "",
                     "years": {}}]})
    league, types = bn.bref_league()
    assert {v[1] for v in league.values()} == {"BKN", "CHA", "PHX"}


def test_bref_drops_every_player_who_appears_on_two_teams(monkeypatch, capsys):
    """The league page quotes a LEAGUE-WIDE TOTAL, so a multi-team row must
    never be written -- agreeing figures included.

    Proved on the team pages, not inferred: /contracts/POR.html has Damian
    Lillard at $14,104,000 for 2027-28 and /contracts/MIL.html at a flat
    $22,516,603, and the league page's $36,620,603 is the two added together
    to the dollar. team_payroll sums this artifact into the team that ROSTERS
    the player, so the aggregate charges Milwaukee's dead money to Portland
    and double-counts money data/dead_money_2026_27.csv already carries.

    An earlier version of this test asserted the agreeing pair was KEPT,
    which pinned the bug. Agreement between two rows of an aggregate says
    only that it is the same aggregate, which is what it is designed to be.
    """
    bn = _bn()
    base = {"type": "guaranteed", "flag": None, "guaranteed": "", "years": {}}
    monkeypatch.setattr(bn.bref_next_money, "league_money", lambda label: {
        "split guy": [{"display": "Split Guy", "team": "MIL",
                       "salary": 36_620_603.0, **base},
                      {"display": "Split Guy", "team": "POR",
                       "salary": 36_620_603.0, **base}],
        "single guy": [{"display": "Single Guy", "team": "BOS",
                        "salary": 20_000_000.0, **base}]})
    league, types = bn.bref_league()
    assert "single guy" in league
    assert "split guy" not in league, (
        "an aggregate figure must not reach the artifact even when both rows "
        "agree -- that agreement is the aggregate, not a corroboration")
    out = capsys.readouterr().out
    assert "Split Guy" in out and "LEAGUE-WIDE TOTAL" in out, (
        "a dropped player renders the gray UFA sentinel, so the drop has to "
        "be loud")


def test_too_many_split_rows_refuses_instead_of_dropping(monkeypatch):
    """Six on 2026-11-18. A count far above that means the page changed
    meaning, and silently dropping dozens turns them all into free agents."""
    bn = _bn()
    base = {"type": "guaranteed", "flag": None, "guaranteed": "", "years": {}}
    many = {f"split {i}": [{"display": f"Split {i}", "team": "MIL",
                            "salary": 1e7, **base},
                           {"display": f"Split {i}", "team": "POR",
                            "salary": 1e7, **base}]
            for i in range(bn.MAX_SPLIT_ROWS + 1)}
    monkeypatch.setattr(bn.bref_next_money, "league_money", lambda label: many)
    assert bn.bref_league() is None


def test_bref_marks_an_empty_cell_as_having_no_salary(monkeypatch):
    """So the builder's `salary is None` branch handles it and the player
    renders as the UFA sentinel -- which is the honest reading of a contract
    that ends this season."""
    bn = _bn()
    monkeypatch.setattr(bn.bref_next_money, "league_money", lambda label: {
        "expiring guy": [{"display": "Expiring Guy", "team": "BOS",
                          "salary": None, "type": "guaranteed",
                          "flag": "empty", "guaranteed": "", "years": {}}]})
    league, types = bn.bref_league()
    assert league["expiring guy"][3] is None
    assert "expiring guy" not in types, (
        "a player with no money must not contribute an option type")


def test_bref_refusal_is_a_dead_source_not_an_empty_one(monkeypatch):
    bn = _bn()

    def refuse(label):
        raise bn.bref_next_money.SourceRefused("layout changed")

    monkeypatch.setattr(bn.bref_next_money, "league_money", refuse)
    assert bn.bref_league() is None


# ==========================================================================
# a client block is a statement about the CLIENT, not about one page
# ==========================================================================
class _Resp:
    def __init__(self, code):
        self.status_code = code


def _http_error(code):
    import requests
    e = requests.HTTPError(f"{code} Client Error")
    e.response = _Resp(code)
    return e


@pytest.mark.parametrize("code", [401, 403, 429])
def test_a_client_block_stops_after_one_request(monkeypatch, code):
    """On 2026-11-18 Spotrac answered 403 thirty times in a row. Asking 29
    more times after the first costs twelve seconds every nightly for as long
    as the block lasts, and hammering a site that is already refusing us is
    how a block gets widened."""
    bn = _bn()
    calls = []

    def boom(abbr):
        calls.append(abbr)
        raise _http_error(code)

    monkeypatch.setattr(bn, "team_money", boom)
    monkeypatch.setattr(bn.time, "sleep", lambda s: None)
    assert bn.spotrac_league() is None
    assert len(calls) == 1, f"asked {len(calls)} pages after an HTTP {code}"


def test_the_first_request_still_goes_out_so_the_chain_self_heals(monkeypatch):
    """The short-circuit must not become "never ask Spotrac again". The first
    probe every run is the whole reason the chain recovers on its own the day
    the block lifts, with nobody editing this file."""
    bn = _bn()
    calls = []

    def ok(abbr):
        calls.append(abbr)
        return {bn.normalize(f"Player {abbr}"):
                ("active", abbr, f"Player {abbr}", 5_000_000.0, None)}

    monkeypatch.setattr(bn, "team_money", ok)
    monkeypatch.setattr(bn.time, "sleep", lambda s: None)
    got = bn.spotrac_league()
    assert got is not None, "a working Spotrac must still be used"
    assert len(calls) == 30, "all thirty pages when nothing is blocking"
    league, types = got
    assert len(league) == 30 and types == {}, (
        "Spotrac carries no option type; that is why the CSV exists")


def test_a_non_block_failure_does_not_short_circuit(monkeypatch):
    """A timeout on one team page is about that page. Only 401/403/429 say
    something about the client, so anything else keeps going and the build
    refuses at the end on the partial-scrape check instead."""
    bn = _bn()
    calls = []

    def flaky(abbr):
        calls.append(abbr)
        raise TimeoutError("slow")

    monkeypatch.setattr(bn, "team_money", flaky)
    monkeypatch.setattr(bn.time, "sleep", lambda s: None)
    assert bn.spotrac_league() is None
    assert len(calls) == 30, "a timeout is not a client block"


# ==========================================================================
# the alias pass was DEAD in every committed build
# ==========================================================================
def test_newest_board_falls_back_when_this_revision_has_no_board(tmp_path, monkeypatch, capsys):
    """The nightly runs this builder at step 3c, BEFORE step 4 writes the
    board for the revision being built. So _rev_path resolved to a file that
    did not exist, the pass printed "skipping", and stamps r40 through r44 all
    recorded aliases=0 -- seven board players and $67,674,436 rendering the
    gray UFA sentinel, every night.
    """
    bn = _bn()
    monkeypatch.setattr(bn.utils, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(bn.utils, "DATA_REV", "r45")
    stem = f"raw_2026_27_{bn.utils.FORMULA_VERSION}"
    (tmp_path / f"{stem}_r43.parquet").write_bytes(b"x")
    (tmp_path / f"{stem}_r44.parquet").write_bytes(b"x")
    got = bn.newest_board()
    assert got is not None, "r45 has no board yet; it must not give up"
    assert got.name == f"{stem}_r44.parquet", "the NEWEST earlier revision"
    assert "r44" in capsys.readouterr().out, "the substitution must be stated"


def test_newest_board_prefers_this_revision_when_it_does_exist(tmp_path, monkeypatch):
    bn = _bn()
    monkeypatch.setattr(bn.utils, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(bn.utils, "DATA_REV", "r44")
    stem = f"raw_2026_27_{bn.utils.FORMULA_VERSION}"
    (tmp_path / f"{stem}_r43.parquet").write_bytes(b"x")
    (tmp_path / f"{stem}_r44.parquet").write_bytes(b"x")
    assert bn.newest_board().name == f"{stem}_r44.parquet"


def test_newest_board_orders_by_revision_number_not_string(tmp_path, monkeypatch):
    """r9 sorts above r40 as a string. Both _previous_rfa and the nightly's
    newest_revision carry the same warning."""
    bn = _bn()
    monkeypatch.setattr(bn.utils, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(bn.utils, "DATA_REV", "r99")
    stem = f"raw_2026_27_{bn.utils.FORMULA_VERSION}"
    (tmp_path / f"{stem}_r9.parquet").write_bytes(b"x")
    (tmp_path / f"{stem}_r40.parquet").write_bytes(b"x")
    assert bn.newest_board().name == f"{stem}_r40.parquet"


def test_newest_board_returns_none_when_there_is_no_board_at_all(tmp_path, monkeypatch):
    """None, not a non-existent path, so the caller warns instead of
    half-reading."""
    bn = _bn()
    monkeypatch.setattr(bn.utils, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(bn.utils, "DATA_REV", "r45")
    assert bn.newest_board() is None
