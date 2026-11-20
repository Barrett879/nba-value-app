"""Dead money belongs in the Trade Machine's baseline, and in `room`.

Dead money is salary a team still owes players who are NOT on its roster:
buyouts, waivers, stretched contracts, partial guarantees left behind in a
trade. It counts against the cap, the tax line and BOTH APRONS exactly like a
player's salary. On 2026-11-20 it was $145.4M across 18 teams.

It appeared in no player card on the Trade Machine, and in no number on it
either: `pages/Trade_Machine.py` passed `team_pages`' `payroll`, which is
roster money, straight into the engine. So the page priced trades against a
baseline that was too small by up to $40.6M:

    MEM   $120.5M shown    $161.1M real    under an apron-1 hard cap
    MIL   $160.4M shown    $189.0M real
    PHX   $190.5M shown    $213.7M real    an error across TWO regime lines
    CLE   $206.9M shown    $209.1M real    against a $209.0M hard cap

and `/Rosters`, which has always added it, published $161.1M for Memphis on a
night the Trade Machine published $120.5M for the same team.

THE SHAPE OF THE FIX, which these tests pin:

  `payroll` stays ROSTER money, so a card header is still addable from the
  cards underneath it. `dead` is published beside it. Everything that compares
  a team to a cap, a tax line or an apron adds the two. The Python engine's
  `TradeTeam.payroll` was always documented as "committed" and simply had the
  wrong number fed to it, so the sum happens at the APP BOUNDARY
  (templates/trade_machine.html, buildTrade) and the engine itself is
  untouched -- which keeps the Python/JS parity vectors testing the same
  functions on both sides.

AND A PRECISION DEFECT THIS UNCOVERED, which is pinned here too. The spine
publishes each salary rounded to $0.1M and used to SUM THE ROUNDED ROWS.
Fourteen Cleveland contracts each moved up to $50K by rounding added to
$206.9M against a true $206.748M, and that $0.15M of accumulated error is
bigger than the $23K by which Cleveland sits UNDER its first-apron hard cap.
Adding dead money to a total built that way would have published Cleveland
$0.13M OVER a cap it is actually under, and the engine refuses every trade for
a team above its hard cap -- so the fix would have shipped a Cleveland that
could never trade. Rounding the ROWS is display; rounding the TOTAL is
arithmetic.
"""
from __future__ import annotations
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAGES = ROOT / "cache" / "team_pages.json"
SPINE = ROOT / "cache" / "team_rosters_2627.json"
DEAD = ROOT / "data" / "dead_money_2026_27.csv"
MASTER = ROOT / "data" / "master_roster.csv"
CBA = ROOT / "data" / "cba_config_2026_27.json"
HARD_CAPS = ROOT / "data" / "hard_caps_2026_27.csv"
TEMPLATE = ROOT / "templates" / "trade_machine.html"
TM_PAGE = ROOT / "pages" / "Trade_Machine.py"


def _rows(path: Path):
    return csv.DictReader(
        l for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.lstrip().startswith("#"))


@pytest.fixture(scope="module")
def pages():
    if not PAGES.exists():
        pytest.skip("team_pages absent")
    return json.loads(PAGES.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ledger():
    """Dead money per team, straight from the verified CSV."""
    out: Counter = Counter()
    for r in _rows(DEAD):
        team = (r.get("team") or "").strip()
        try:
            out[team] += float(r["amount_M"])
        except (TypeError, ValueError, KeyError):
            continue
    return out


@pytest.fixture(scope="module")
def cfg():
    return json.loads(CBA.read_text(encoding="utf-8"))


# ── the artifact carries it ──────────────────────────────────────────────────

def test_every_team_publishes_a_dead_money_figure(pages):
    """Absent is not the same as zero. A reader that has to guess which it is
    will guess zero, which is the bug."""
    missing = [a for a, t in pages["teams"].items() if "dead" not in t]
    assert not missing, f"no `dead` field on: {missing}"


def test_published_dead_money_matches_the_verified_ledger(pages, ledger):
    """The number on the page must be the number in the file somebody
    verified, team by team, not a figure that drifted from it."""
    off = {}
    for a, t in pages["teams"].items():
        want = round(ledger.get(a, 0.0), 1)
        if abs(float(t.get("dead") or 0.0) - want) > 0.0501:
            off[a] = (t.get("dead"), want)
    assert not off, f"dead money disagrees with the ledger (published, csv): {off}"


def test_the_ledger_is_not_empty_so_these_tests_can_fail(ledger):
    """A positive control. Every assertion above passes trivially against an
    empty ledger, and an empty ledger is exactly what a parse bug produces."""
    assert len(ledger) >= 10, f"only {len(ledger)} team(s) with dead money"
    assert sum(ledger.values()) > 100, f"league total ${sum(ledger.values()):.1f}M"


# ── room is apron room on COMMITTED money ────────────────────────────────────

def test_room_subtracts_dead_money(pages):
    """`room` is rendered by seo_selfheal as "$X under/over <the line it
    names>". Dead money charges against every apron and every hard cap, so
    room that ignores it overstates what a team can actually do.

    Since 2026-11-20 the line is the team's OWN binding ceiling rather than
    always the second apron, so this checks the dead-money term against
    whichever ceiling the row carries, read back out of `room_line`.
    """
    cfg = json.loads(CBA.read_text(encoding="utf-8"))
    ceil_for = {"first-apron hard cap": float(cfg["apron1"]),
                "second-apron hard cap": float(cfg["apron2"]),
                "second apron": float(cfg["apron2"])}
    off, seen = {}, set()
    for a, t in pages["teams"].items():
        line = (t.get("room_line") or "second apron").strip()
        seen.add(line)
        ceiling = ceil_for.get(line)
        assert ceiling is not None, f"{a} carries an unknown room_line {line!r}"
        # from the exact committed figure, for the same reason the spine
        # computes it that way: payroll and dead are each rounded for display.
        want = round(ceiling - float(t["committed"]), 1)
        if abs(float(t["room"]) - want) > 0.0501:
            off[a] = (t["room"], want, line)
    assert not off, f"room is not ceiling room on committed salary: {off}"
    assert len(seen) >= 2, (
        f"every team reports the same ceiling ({seen}), so the hard-cap ledger "
        f"is not reaching the spine")


def test_dead_money_actually_moves_room_somewhere(pages):
    """Guards the test above against passing because every figure is zero."""
    movers = [a for a, t in pages["teams"].items() if float(t.get("dead") or 0) > 1.0]
    assert len(movers) >= 5, f"only {len(movers)} team(s) carry over $1M of dead money"


# ── the engine is fed committed salary ───────────────────────────────────────

def test_the_trade_machine_payload_carries_dead_money():
    src = TM_PAGE.read_text(encoding="utf-8")
    assert '"dead"' in src, (
        "pages/Trade_Machine.py does not put `dead` in the payload, so the "
        "page cannot price a trade against committed salary")


def test_the_js_engine_baseline_adds_dead_money():
    """buildTrade() is the app boundary where roster money becomes committed
    money. If this line loses its `dead` term the engine silently goes back to
    granting Memphis $40.6M of headroom it does not have, and every verdict
    still renders as confidently as before."""
    js = TEMPLATE.read_text(encoding="utf-8")
    # The inline sum moved behind committedOf() on 2026-11-20, because adding
    # the two PUBLISHED (rounded) numbers rounded a third time: Cleveland
    # showed $0.115M of room against $0.023M. The helper prefers the exact
    # `committed` field and falls back to payroll + dead, so the dead term is
    # still guaranteed -- which is what this test exists to protect.
    assert "payroll:committedOf(DATA.teams[a])" in js, (
        "the JS engine baseline no longer routes through committedOf()")
    helper = js.split("function committedOf(", 1)[1].split("\n", 1)[0]
    assert "T.dead" in helper and "T.committed" in helper, (
        f"committedOf() has lost the dead-money term or the exact field: {helper}")


def test_the_card_header_uses_the_same_basis_as_the_engine():
    """`after` comes from the engine and is committed salary. If "was" stayed
    on roster money, a Memphis card with nothing moved would read
    "was $120.5M -> $161.1M" and look like the trade cost $40M."""
    js = TEMPLATE.read_text(encoding="utf-8")
    # Both sides now read ONE helper rather than each computing the sum, so a
    # future basis change cannot move the engine without moving the header.
    assert "const committed=committedOf(T);" in js, "no committed basis"
    assert js.count("committedOf(") >= 3, (
        "the engine boundary and the card header are not both on the helper")
    assert "const after=tt?payAfter(tt):committed;" in js, (
        "the header's resting figure is not the engine's basis")
    assert "was ${fmtM(committed)}" in js, "the 'was' figure is not committed"


def test_the_page_says_why_the_header_exceeds_its_cards():
    """On 18 teams the header is deliberately larger than the players listed
    under it. An unexplained gap is the same complaint the hidden draft picks
    produced: a payroll the user cannot account for."""
    js = TEMPLATE.read_text(encoding="utf-8")
    assert "dead" in js and "includes ${fmtM(T.dead)} dead" in js, (
        "nothing on the card explains the dead-money part of the header")


# ── the behaviour the CBA actually requires ──────────────────────────────────

def test_dead_money_can_decide_a_hard_cap_verdict(cfg):
    """The point of all of it, in the engine itself: a team whose ROSTER
    salary is under its hard cap but whose COMMITTED salary is over it must be
    refused. Cleveland sits $23K under a $209.0M first-apron hard cap with
    $2.2M of dead money inside that figure, so this is not hypothetical
    arithmetic."""
    import trade_rules as tr
    C = tr.CBAConfig.load("2026-27")
    cap = float(cfg["apron1"])
    roster, dead = cap - 1.0, 2.0          # under on roster, over on committed

    def verdict(payroll):
        t = tr.TradeTeam(abbr="AAA", payroll=payroll, roster_size=14,
                         hard_cap=cap)
        v, _n = tr.check_hard_caps(tr.Trade(teams=[t], trade_date="2026-11-20"), C)
        return [x.code for x in v]

    assert "HARD_CAP_EXCEEDED" not in verdict(roster), (
        "a team under its hard cap on roster salary should not be refused")
    assert "HARD_CAP_EXCEEDED" in verdict(roster + dead), (
        "committed salary over the hard cap must be refused; feeding roster "
        "salary alone is what hid it")


def test_no_team_is_published_over_a_hard_cap_it_is_actually_under(pages, cfg):
    """A hard cap is an absolute ceiling, so a team cannot BE over one: if our
    figures say it is, our figures are wrong -- and the engine then refuses
    EVERY trade for that team, with a message blaming the user's trade.

    This test shipped with Golden State EXCLUDED, as a pre-existing data
    problem it was not this change's job to fix: GSW was recorded $11.4M above
    a first-apron hard cap while carrying no dead money at all. On 2026-11-20
    that row was corrected -- the Melton signing used the TAXPAYER midlevel,
    which caps at the SECOND apron, and GSW clears that by $1.2M -- so the
    exclusion is GONE and this now holds for all thirty teams with no
    carve-out. Do not re-add one: a team over its own cap is a data bug, and
    this assertion is the only thing that says so.
    """
    lvl = {"apron1": float(cfg["apron1"]), "apron2": float(cfg["apron2"])}
    caps = {}
    for r in _rows(HARD_CAPS):
        caps[(r.get("team") or "").strip()] = lvl.get((r.get("cap") or "").strip())
    over = []
    for a, cap in caps.items():
        t = pages["teams"].get(a)
        if not t or cap is None:
            continue
        committed = round(t["payroll"] + float(t.get("dead") or 0.0), 1)
        if committed > cap + 1e-9:
            over.append((a, committed, cap))
    assert not over, (
        f"these teams are now published above their hard cap, so the Trade "
        f"Machine will refuse every trade they appear in: {over}")


# ── the precision defect the fix uncovered ───────────────────────────────────

def test_payroll_is_summed_exact_not_from_the_rounded_rows(pages):
    """Published salaries are rounded to $0.1M for display. Summing THOSE is a
    different number: Cleveland's fourteen contracts sum to $206.748M exactly
    and $206.9M from the rounded rows, and the $0.15M gap is bigger than the
    $23K margin by which Cleveland clears its hard cap.
    """
    exact: Counter = Counter()
    for r in _rows(MASTER):
        if (r.get("kind") or "").strip() != "standard":
            continue
        try:
            v = float(str(r.get("salary_M") or "").replace("$", "").replace(",", ""))
        except (TypeError, ValueError):
            continue
        exact[(r.get("team") or "").strip()] += v
    off = {}
    for a, t in pages["teams"].items():
        if a not in exact:
            continue
        want = round(exact[a], 1)
        if abs(float(t["payroll"]) - want) > 0.0501:
            off[a] = (t["payroll"], want)
    assert not off, (
        f"payroll is not the exact contract sum rounded once (published, "
        f"exact): {off}")


def test_the_rounded_rows_really_do_disagree_so_that_test_can_fail(pages):
    """A positive control for the one above: if rounding happened to be exact
    everywhere, that test would pass against a builder that still summed the
    rounded rows. At least one team's published payroll must differ from the
    sum of its own published salaries."""
    gaps = []
    for a, t in pages["teams"].items():
        rows = round(sum(p["salary"] or 0 for p in t["players"]
                         if not p.get("two_way")), 1)
        if abs(rows - float(t["payroll"])) > 0.0501:
            gaps.append((a, rows, t["payroll"]))
    assert gaps, (
        "no team's rounded rows differ from its payroll, so the exact-sum "
        "test cannot distinguish the two and proves nothing")
