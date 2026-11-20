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
import re
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

    TWO SUBSTRING TRAPS, both of which actually bit.

    1. "non-taxpayer midlevel" CONTAINS "taxpayer midlevel". Searched naively,
       all eleven non-taxpayer-MLE rows read as taxpayer-MLE rows, every one
       flips from apron1 to apron2, and the test inverts into asserting the
       bug.

    2. "expanded traded player exception" CONTAINS "traded player exception".
       That is the phrase our own cited source, the Hoops Rumors tracker,
       actually uses -- and it is EXPANDED MATCHING, not prior-season TPE use.
       They are different mechanisms that happen to share an apron, so reading
       one as the other records the wrong history even when the verdict lands
       right. For New Orleans the distinction is live: the expanded-matching
       trigger fired and the prior-season Alvarado TPE affirmatively did NOT.

    Both phrases are masked before the shorter one is searched for. This
    function first shipped missing BOTH halves of trap 2 -- it looked for the
    literal "trade exception" (which "traded player exception" does not
    contain) and for "more than 100%" but never "expanded" -- so the tracker's
    own wording classified as the EMPTY SET.
    """
    t = trigger.lower()
    masked = t.replace("non-taxpayer", "@@")
    for phrase in ("expanded traded player exception", "expanded trade exception"):
        masked = masked.replace(phrase, "@@")
    found = set()
    if "non-taxpayer" in t:
        found.add("ntmle")
    if "taxpayer midlevel" in masked:
        found.add("tmle")
    if "sign-and-trade" in t:
        found.add("snt")
    if "bi-annual" in t:
        found.add("bae")
    if "more than 100%" in t or "expanded" in t:
        found.add("expanded")
    if ("trade exception" in masked or "traded player exception" in masked
            or "prior-year exception" in masked):
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


def test_expanded_matching_does_not_read_as_prior_season_tpe_use():
    """The second substring trap, and the one that actually shipped broken.

    "expanded traded player exception" is the Hoops Rumors tracker's own
    wording for EXPANDED MATCHING, and it contains "traded player exception".
    The first version of this classifier found neither half -- it searched for
    the literal "trade exception" and for "more than 100%" -- so the live
    phrasing of our own cited source returned the EMPTY SET and would have
    failed the consistency test the moment anyone quoted the tracker.
    """
    tracker = _mechanisms("Used the expanded traded player exception to acquire AJ Johnson")
    assert tracker, (
        "the tracker's own wording classifies as nothing at all, so any row "
        "quoting our cited source fails the consistency test")
    assert tracker == {"expanded"}, tracker
    assert "tpe" not in tracker, (
        "expanded matching read as PRIOR-SEASON TPE use; they share an apron "
        "but not a mechanism, and for New Orleans the prior-season TPE "
        "affirmatively did not fire")

    real_tpe = _mechanisms("Used the prior-season Capela trade exception to absorb Devin Carter")
    assert real_tpe == {"tpe"}, real_tpe


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


def test_an_outside_signing_too_big_for_the_small_exceptions_implies_a_cap(cfg, ledger):
    """THE ONE THAT WOULD HAVE CAUGHT NEW ORLEANS, on 2026-09-10.

    A `cap=none` row is a claim, and until now nothing policed it: every other
    test in this file filters to apron1/apron2 rows, so a team could go capped
    in September and sit at "none" forever with an empty trigger and an empty
    source. New Orleans did exactly that for ten weeks.

    The screen: a team ALREADY OVER THE SALARY CAP that signs an OUTSIDE free
    agent (type "new", so no Bird rights) for more than the taxpayer midlevel
    has only apron-triggering ways to pay him. Cap room is gone by assumption,
    the room exception belongs to under-cap teams, the bi-annual and every
    minimum are smaller, and a trade exception cannot sign a free agent at
    all -- it absorbs incoming salary in a TRADE. What is left is the
    non-taxpayer midlevel or a sign-and-trade, and both hard-cap at the first
    apron. (Just above the taxpayer figure the taxpayer midlevel is still in
    range, which caps at the second, so this asserts only that SOME cap
    exists, not which.)

    League-wide this fires on three signings and names exactly one team, so it
    is a screen rather than noise. Bennedict Mathurin at $8.0M on a New
    Orleans book of $188.8M against a $164.961M cap is the one it catches.
    """
    caps = {(r.get("team") or "").strip(): (r.get("cap") or "").strip()
            for r in ledger}
    tmle, cap_line = float(cfg["tmle"]), float(cfg["cap"])
    pages = json.loads(PAGES.read_text(encoding="utf-8"))["teams"]

    screened, bad = 0, []
    for r in _rows(ROOT / "data" / "real_signings_2026.csv"):
        if (r.get("type") or "").strip() != "new":
            continue
        try:
            yr1 = float(r["yr1_M"])
        except (KeyError, TypeError, ValueError):
            continue
        team = (r.get("team") or "").strip()
        t = pages.get(team)
        if not t or yr1 <= tmle:
            continue
        before = t["payroll"] + float(t.get("dead") or 0.0) - yr1
        if before <= cap_line:
            continue
        screened += 1
        if caps.get(team) not in ("apron1", "apron2"):
            bad.append((team, r.get("player"), yr1, round(before, 1)))

    assert screened >= 2, (
        f"only {screened} signings met the screen, so it is not actually "
        f"looking at anything and would pass whatever the ledger said")
    assert not bad, (
        f"these teams were over the cap and signed an outside free agent for "
        f"more than the ${tmle}M taxpayer midlevel, which only an "
        f"apron-triggering exception can pay, yet carry no hard cap: {bad}")


def test_meltons_salary_is_the_taxpayer_midlevel_not_the_bi_annual(cfg):
    """THE REASONING, not just the row -- and the reasoning itself was wrong
    the first time, so this test is the corrected record.

    The row originally said Golden State used the BI-ANNUAL exception on
    De'Anthony Melton. It was written that way because our books carried him
    at $5,477,000, which equals the 2026-27 bi-annual exception TO THE CENT,
    and the mechanism was read off the amount.

    But $5,477,000 was never his salary. It was a Spotrac PENDING-transactions
    listing from early July that never became the deal. The Warriors offseason
    check-in of 2026-09-12 states: "Two years, $12,431,200. Second-year player
    option. Re-signed using taxpayer mid-level exception." $12,431,200 is
    exactly 2.05 times $6,064,000, which is the arithmetic signature of the
    LARGEST taxpayer-MLE contract that can exist: year one at the full
    exception, year two at the 5% ceiling.

    So the coincidence was real and did cause the bug, but it was a
    coincidence with a STALE PENDING FIGURE rather than with his pay. This
    pins the corrected figures and guards the obvious regression, which is
    somebody seeing $5,477,000 in an old note and "restoring" it.
    """
    mel = [r for r in _rows(MASTER)
           if "Melton" in (r.get("player") or "")
           and (r.get("team") or "").strip() == "GSW"]
    assert mel, "Melton is no longer on the GSW master roster"
    salary = float(mel[0]["salary_M"])
    bae, tmle = float(cfg["bae"]), float(cfg["tmle"])

    assert abs(salary - tmle) < 0.0011, (
        f"Melton is carried at ${salary}M; the sourced figure is the FULL "
        f"taxpayer midlevel ${tmle}M")
    assert abs(salary - bae) > 0.1, (
        f"Melton is back at the bi-annual figure ${bae}M. That number came "
        f"from a superseded Spotrac pending listing and is what mis-set this "
        f"row to apron1 for four months")
    assert bae != tmle, "the two exceptions are distinct figures"

    # the arithmetic signature that identifies the mechanism without the label
    assert abs(round(2.05 * tmle, 4) - 12.4312) < 0.0011, (
        "2.05x the taxpayer midlevel no longer equals the $12,431,200 total "
        "the check-in publishes, so reread the contract")


def test_a_taxpayer_midlevel_row_names_nobody_too_expensive_for_it(ledger, cfg):
    """Miami's row said the TAXPAYER midlevel on Tim Hardaway Jr., whose
    official salary is $6,065,000 -- $1,000 above the $6,064,000 taxpayer
    portion, and $11,665,000 once Klay Thompson is added. Miami's verdict
    (apron1) was right the whole time, which is exactly why nothing caught it:
    a WRONG MECHANISM under a RIGHT APRON is invisible to every other check
    here. Only comparing the named exception against the named player's actual
    pay finds it.

    THE SHAPE MATTERS, and the first version of this test got it wrong. Reading
    every dollar figure in a trigger as a salary is useless: triggers quote
    TWO-YEAR TOTALS ("2yr/$13M"), they quote the exception's own size, and a
    row naming two exceptions has no way to say which figure belongs to which.
    That version produced five false positives on a clean ledger, which is a
    test that gets weakened or deleted rather than believed.

    So this is narrow on purpose. The taxpayer midlevel is the one exception
    with a hard per-player ceiling low enough to discriminate, and it is the
    one that decides apron2 versus apron1. For any row naming it, every player
    that row names must be payable from it.
    """
    tmle = float(cfg["tmle"])
    names = {}
    for r in _rows(MASTER):
        n, t = (r.get("player") or "").strip(), (r.get("team") or "").strip()
        if n and (r.get("kind") or "").strip() == "standard":
            try:
                names.setdefault(t, {})[n] = float(r["salary_M"])
            except (KeyError, TypeError, ValueError):
                pass

    checked, bad = 0, []
    for r in ledger:
        trig = r.get("trigger") or ""
        low = trig.lower()
        if "taxpayer midlevel" not in low.replace("non-taxpayer", "@@"):
            continue
        team = r["team"].strip()
        for player, salary in names.get(team, {}).items():
            if player not in trig:
                continue
            checked += 1
            # $1K over is still over: that is precisely the Miami case.
            if salary > tmle + 0.0011:
                bad.append((team, player, salary, tmle))

    assert checked >= 1, (
        "no taxpayer-midlevel row names a player this test could price, so it "
        "is asserting nothing")
    assert not bad, (
        f"a row credits the taxpayer midlevel (${tmle}M) for a player it "
        f"cannot pay, so the mechanism is wrong even if the apron is not "
        f"(team, player, salary, limit): {bad}")


# ── the basis the engine enforces on ────────────────────────────────────────

def test_no_capped_team_is_over_its_cap_on_the_EXACT_figure(ledger, cfg):
    """INVARIANT 1, league-wide, which is what the CSV header has claimed all
    along. Until now it was pinned for Golden State alone, so the file's own
    documentation was false and the check that would have caught GSW in July
    did not exist in this module.

    Asserted on `committed`, the exact sum, not on payroll + dead. Those are
    each rounded to $0.1M for the card, and adding two rounded numbers is a
    third rounding on the one quantity every apron test reads.
    """
    pages = json.loads(PAGES.read_text(encoding="utf-8"))["teams"]
    lvl = {"apron1": float(cfg["apron1"]), "apron2": float(cfg["apron2"])}
    checked, over = 0, []
    for r in ledger:
        team, cap = r["team"].strip(), lvl.get((r.get("cap") or "").strip())
        t = pages.get(team)
        if cap is None or not t:
            continue
        committed = t.get("committed")
        assert committed is not None, (
            f"{team} publishes no exact committed figure, so this invariant "
            f"would silently fall back to the rounded sum it exists to avoid")
        checked += 1
        if committed > cap + 1e-9:
            over.append((team, round(committed, 4), cap))
    assert checked >= 20, f"only {checked} capped teams were examined"
    assert not over, (
        f"a hard cap cannot be exceeded, so these are OUR figures being wrong, "
        f"and the engine refuses every trade they appear in: {over}")


def test_the_engine_is_handed_the_exact_figure_not_the_sum_of_two_rounded_ones():
    """The defect this caught, and the shape of the fix.

    `payroll` and `dead` are published rounded to $0.1M so a team card's header
    still adds up from the player rows beneath it. The Trade Machine used to
    ADD THOSE TWO, which rounds a third time on the figure every apron and
    hard-cap test reads. It is not academic: Cleveland published $208.900M
    against an exact $208.992M, so the engine saw $0.115M of room where
    $0.023M exists -- and returned legal on trades taking back +$0.03M,
    +$0.05M and +$0.08M which break a hard cap its own message calls one that
    cannot be exceeded. Twenty-three of thirty teams drift.

    Pinned three ways: the field exists, the drift is real (so the fix is not
    a no-op), and BOTH template sites read the same helper -- the engine
    boundary and the card header computed `committed` independently before, so
    a future basis change could have split them silently.
    """
    pages = json.loads(PAGES.read_text(encoding="utf-8"))["teams"]
    missing = [a for a, t in pages.items() if t.get("committed") is None]
    assert not missing, f"teams with no exact committed figure: {missing}"

    drift = [a for a, t in pages.items()
             if abs(round(t["payroll"] + t["dead"], 6) - t["committed"]) > 0.01]
    assert len(drift) >= 5, (
        f"only {len(drift)} teams differ between the rounded sum and the exact "
        f"figure, so either the rounding was fixed upstream or this test is "
        f"no longer measuring anything")

    tpl = (ROOT / "templates" / "trade_machine.html").read_text(encoding="utf-8")
    assert tpl.count("function committedOf(") == 1, "the helper is gone or duplicated"
    assert tpl.count("committedOf(") >= 3, (
        "fewer than two call sites use the helper; the engine boundary and the "
        "card header must read the same baseline")
    assert "payroll:DATA.teams[a].payroll+(DATA.teams[a].dead" not in tpl, (
        "the engine boundary is adding the two rounded numbers again")
    assert "const committed=T.payroll+(T.dead" not in tpl, (
        "the card header is adding the two rounded numbers again")


def test_the_exact_figure_changes_a_real_verdict():
    """A behavioural control through production code, because a precision fix
    that changes no verdict is not worth a field. On the tightest capped team,
    a trade that fits under the ROUNDED figure but not the exact one must now
    be refused."""
    import trade_rules as tr
    C = tr.CBAConfig.load("2026-27")
    pages = json.loads(PAGES.read_text(encoding="utf-8"))["teams"]
    caps = {r["team"].strip(): r["cap"].strip()
            for r in _rows(ROOT / "data" / "hard_caps_2026_27.csv")}
    lvl = {"apron1": C.apron1, "apron2": C.apron2}

    # The DANGEROUS direction is the rounded sum sitting BELOW the exact one:
    # that is the engine seeing room a team does not have. The opposite sign
    # is merely over-cautious. Cleveland is understated by $0.092M against
    # $0.023M of real room, which is the whole reason this field exists.
    best = None
    for team, lab in caps.items():
        cap, t = lvl.get(lab), pages.get(team)
        if cap is None or not t or t.get("committed") is None:
            continue
        understated = t["committed"] - round(t["payroll"] + t["dead"], 6)
        if understated > 0.005 and (best is None or cap - t["committed"] < best[1]):
            best = (team, cap - t["committed"], t, cap)
    if best is None:
        pytest.skip("no capped team is currently understated by the rounding")
    team, room, t, cap = best

    def refused(payroll, net):
        tt = tr.TradeTeam(abbr=team, payroll=payroll, roster_size=14, hard_cap=cap)
        tt.in_players.append(tr.TradePlayer(name="x", salary=net))
        v, _ = tr.check_hard_caps(tr.Trade(teams=[tt], trade_date="2026-11-20"), C)
        return "HARD_CAP_EXCEEDED" in [x.code for x in v]

    net = round(room + 0.005, 4)          # over the exact cap, under the rounded one
    assert refused(t["committed"], net), (
        f"{team} has ${room:.4f}M of room and a ${net}M take-back is not refused")
    assert not refused(round(t["payroll"] + t["dead"], 6), net), (
        f"the rounded sum already refuses this, so it proves nothing about "
        f"the exact figure")
