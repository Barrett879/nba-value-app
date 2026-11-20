"""The hard-cap ledger has to be internally possible, not just sourced.

A hard cap is the one number in the CBA that admits no exception: a capped
team may not exceed it for any reason, Bird rights included. So a row of this
ledger is not just a claim about the world, it is a claim that must agree with
our own payroll. For four months it did not.

    GSW   cap apron1 ($209.015M)   committed salary $220.4M   $11.4M OVER

`check_hard_caps` tests `payroll_after() > hard_cap` whether or not the trade
made anything worse, so the Trade Machine refused EVERY Warriors trade and
blamed the user's trade for a baseline the user never touched.

THE ROOT CAUSE WAS A COINCIDENCE OF TWO CAP FIGURES. Melton re-signed for
$5.477M. The 2026-27 bi-annual exception is $5.477M -- exactly -- so the
mechanism was read off the amount, and the bi-annual caps at the FIRST apron.
The taxpayer midlevel is $6.064M and covers $5.477M just as well, and it caps
at the SECOND apron, which GSW clears by $1.2M. Hoops Rumors lists Golden
State in its second-apron group: "Used the taxpayer mid-level exception to
sign De'Anthony Melton." Two exceptions can pay the same salary; the amount
alone can never tell you which one was used.

WHAT ACTUALLY CAUGHT IT was not a source check, it was the arithmetic: a team
above its own hard cap is impossible, so the file was wrong. That assertion
lives in test_dead_money_baseline.py and now runs with no carve-out. What this
file adds is the structural half -- that the apron LEVEL agrees with the
mechanism the row names, and that no capped row ships on an unnamed source,
which is how this one got in. GSW was the only capped row whose notes never
said "confirmed by verifier".
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HARD_CAPS = ROOT / "data" / "hard_caps_2026_27.csv"
MASTER = ROOT / "data" / "master_roster.csv"
CBA = ROOT / "data" / "cba_config_2026_27.json"
PAGES = ROOT / "cache" / "team_pages.json"

# Article VII: the binding cap is the LOWEST apron any used mechanism
# triggers, which is why a row naming both (Miami, Toronto) is apron1.
APRON1_MECHS = {"ntmle", "snt", "bae", "expanded", "tpe"}
APRON2_MECHS = {"tmle", "agg", "cash"}


def _rows(path: Path):
    return list(csv.DictReader(
        l for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.lstrip().startswith("#")))


def _mechanisms(trigger: str) -> set[str]:
    """Which CBA mechanisms a trigger sentence names.

    THE TRAP: "non-taxpayer midlevel" CONTAINS "taxpayer midlevel". Searched
    naively, all eleven non-taxpayer-MLE rows read as taxpayer-MLE rows, every
    one of them flips from apron1 to apron2, and the test inverts into
    asserting the bug. The non-taxpayer phrase is masked out first.
    """
    t = trigger.lower()
    masked = t.replace("non-taxpayer", "@@")
    found = set()
    if "non-taxpayer" in t:
        found.add("ntmle")
    if "taxpayer midlevel" in masked:
        found.add("tmle")
    if "sign-and-trade" in t:
        found.add("snt")
    if "bi-annual" in t:
        found.add("bae")
    if "more than 100%" in t:
        found.add("expanded")
    if "trade exception" in t or "prior-year exception" in t:
        found.add("tpe")
    if "aggregat" in t:
        found.add("agg")
    if "cash" in t:
        found.add("cash")
    return found


@pytest.fixture(scope="module")
def ledger():
    return _rows(HARD_CAPS)


@pytest.fixture(scope="module")
def cfg():
    return json.loads(CBA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def capped(ledger):
    return [r for r in ledger
            if (r.get("cap") or "").strip() in ("apron1", "apron2")]


# ── positive controls: a scan that reads nothing reports everything clean ────

def test_the_ledger_lists_all_thirty_teams_exactly_once(ledger):
    """The file's own doctrine is that a status change is a row EDIT, never an
    addition, so all thirty teams are always present. This is also the control
    on the parse: a reader that silently got nothing would pass every
    assertion below it."""
    teams = [(r.get("team") or "").strip() for r in ledger]
    assert len(teams) == 30, f"expected 30 rows, parsed {len(teams)}"
    dupes = {t for t in teams if teams.count(t) > 1}
    assert not dupes, f"duplicated team rows: {dupes}"
    assert all(teams), "a row has no team"


def test_the_classifier_finds_every_mechanism_family(capped):
    """Control on `_mechanisms`. If it returned empty sets the consistency
    test below would pass vacuously on all thirty rows, so each family must
    actually be recognised somewhere in the live file."""
    seen: set[str] = set()
    for r in capped:
        seen |= _mechanisms(r["trigger"])
    for fam in ("ntmle", "tmle", "snt", "bae", "expanded", "tpe", "agg", "cash"):
        assert fam in seen, (
            f"no row in the ledger reads as {fam!r}, so the classifier cannot "
            f"be trusted to police it")


def test_non_taxpayer_midlevel_does_not_read_as_taxpayer_midlevel():
    """The substring trap, pinned directly. Both phrases must classify as
    exactly one family, and not each other's."""
    nt = _mechanisms("Signed X with the full non-taxpayer midlevel")
    tx = _mechanisms("Signed Y with the taxpayer midlevel")
    assert nt == {"ntmle"}, nt
    assert tx == {"tmle"}, tx
    assert "tmle" not in nt, (
        "non-taxpayer midlevel classified as the TAXPAYER midlevel, which "
        "would flip eleven apron1 rows to apron2")


# ── the structural invariant ────────────────────────────────────────────────

def test_the_apron_level_matches_the_mechanism_the_row_names(capped):
    """Binding cap = the LOWEST apron triggered. A row naming any first-apron
    mechanism must be apron1 even when it also names a second-apron one, which
    is why Miami (taxpayer MLE plus expanded matching) and Toronto
    (aggregation plus expanded matching) are both apron1. A row naming ONLY
    second-apron mechanisms must be apron2.

    This would not have caught the GSW row -- "bi-annual" and apron1 agree
    with each other perfectly, and the mechanism itself was the false part. It
    catches the next one: a level typed one line off from its mechanism.
    """
    wrong = []
    for r in capped:
        team, cap = r["team"].strip(), r["cap"].strip()
        mechs = _mechanisms(r["trigger"])
        if not mechs:
            wrong.append((team, cap, "no mechanism recognised in the trigger"))
        elif mechs & APRON1_MECHS and cap != "apron1":
            wrong.append((team, cap, f"names {sorted(mechs & APRON1_MECHS)}, "
                                     f"a first-apron trigger"))
        elif not (mechs & APRON1_MECHS) and cap != "apron2":
            wrong.append((team, cap, f"names only {sorted(mechs)}, "
                                     f"second-apron triggers"))
    assert not wrong, f"apron level disagrees with the stated mechanism: {wrong}"


def test_no_capped_row_ships_on_an_unnamed_source(capped):
    """Every capped row restricts a team for a whole league year, so each one
    owes a mechanism and a citable source. The GSW row's source was "Yossi
    Gozlan via critic agent" -- an inference step, not a publication -- and it
    was the row that was wrong."""
    thin = [(r["team"].strip(), (r.get("source") or "").strip())
            for r in capped
            if not (r.get("trigger") or "").strip()
            or not (r.get("source") or "").strip()
            or "critic agent" in (r.get("source") or "").lower()]
    assert not thin, (
        f"capped rows with no mechanism, no source, or a source that is an "
        f"agent rather than a publication: {thin}")


def test_the_trigger_carries_no_double_quote(capped):
    """It is injected raw into an HTML title attribute, and the template's
    esc() escapes angle brackets and ampersands but NOT quotes, so one double
    quote ends the attribute early."""
    bad = [r["team"].strip() for r in capped if '"' in r["trigger"]]
    assert not bad, f"a double quote in the tooltip text would break it: {bad}"


# ── the symptom, through production code ───────────────────────────────────

def test_the_engine_accepts_a_warriors_trade_at_the_published_baseline(ledger):
    """The user-visible defect, end to end and in the engine itself rather
    than restated: take the cap level THIS FILE publishes for GSW, the payroll
    the page publishes for GSW, and assert the engine does not refuse before
    the user has added anything to the trade.

    Reading the level from the ledger rather than naming apron2 here is the
    point -- it is what makes this test fail if the row is reverted, instead
    of quietly testing a constant.

    The second half is the positive control: under the first-apron cap the
    same team IS refused, which is what the page did for four months. Without
    it, an engine that had stopped checking hard caps at all would pass.
    """
    import trade_rules as tr
    C = tr.CBAConfig.load("2026-27")
    pages = json.loads(PAGES.read_text(encoding="utf-8"))
    g = pages["teams"].get("GSW")
    if not g:
        pytest.skip("team_pages has no GSW")
    committed = round(float(g["payroll"]) + float(g.get("dead") or 0.0), 1)

    row = [r for r in ledger if r["team"].strip() == "GSW"]
    assert row, "no GSW row"
    published = {"apron1": C.apron1, "apron2": C.apron2}.get(row[0]["cap"].strip())
    assert published is not None, (
        f"GSW carries cap {row[0]['cap']!r}, which is not an apron level")

    def codes(cap):
        t = tr.TradeTeam(abbr="GSW", payroll=committed, roster_size=15,
                         hard_cap=cap)
        v, _n = tr.check_hard_caps(
            tr.Trade(teams=[t], trade_date="2026-11-20"), C)
        return [x.code for x in v]

    assert "HARD_CAP_EXCEEDED" not in codes(published), (
        f"GSW is published at ${committed}M committed against the "
        f"${published}M cap this ledger gives it, and the engine still "
        f"refuses a trade the user has not touched")
    assert "HARD_CAP_EXCEEDED" in codes(C.apron1), (
        "the first-apron cap no longer refuses a $220M no-op trade, so this "
        "control proves nothing about the fix")


def test_golden_state_is_capped_at_the_second_apron_not_the_first(ledger):
    """Pins the verdict itself. Hoops Rumors lists GSW in its second-apron
    group for the taxpayer-MLE signing of Melton, and apron1 is arithmetically
    impossible at a $220.4M book."""
    g = [r for r in ledger if r["team"].strip() == "GSW"]
    assert g, "no GSW row"
    assert g[0]["cap"].strip() == "apron2", (
        f"GSW is back to {g[0]['cap']!r}; a first-apron cap cannot coexist "
        f"with its published payroll")
    assert "taxpayer midlevel" in g[0]["trigger"].lower()
    assert "bi-annual" not in g[0]["trigger"].lower()


def test_the_two_exception_figures_that_coincide_are_pinned(cfg):
    """THE REASONING, not just the row. The whole error was that Melton's
    salary equals the bi-annual exception to the cent, so the amount looked
    like proof of the mechanism. If these two figures ever stop coinciding the
    story in this file stops making sense and should be reread -- and while
    they do coincide, nobody should "re-fix" the row by matching the number
    again.
    """
    mel = [r for r in _rows(MASTER)
           if "Melton" in (r.get("player") or "")
           and (r.get("team") or "").strip() == "GSW"]
    assert mel, "Melton is no longer on the GSW master roster"
    salary = float(mel[0]["salary_M"])
    bae, tmle = float(cfg["bae"]), float(cfg["tmle"])

    assert abs(salary - round(bae, 2)) < 0.011, (
        f"Melton ${salary}M no longer matches the bi-annual ${bae}M; the "
        f"coincidence that caused this bug is gone, so reread the row")
    assert salary <= tmle + 1e-9, (
        f"${salary}M does not fit inside the taxpayer midlevel ${tmle}M, so "
        f"the mechanism now on the row cannot pay it")
    assert bae != tmle, "the two exceptions are distinct figures"
