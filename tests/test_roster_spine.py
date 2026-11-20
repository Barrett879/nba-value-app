"""The roster spine must be REBUILT, and it must match the verified roster.

cache/team_rosters_2627.json is what every team page and every trade card is
built from. Until 2026-11-19 nothing rebuilt it on a schedule. It had a writer
-- scripts/export_fa_spreadsheet.py -- but that is an OFFSEASON tool: it fills
roster slots from the free-agency simulation and writes a workbook to
~/Downloads, so it could never be the nightly's roster step. nightly.py said as
much in its own comment, and the comment was the defect.

Twenty days of drift cost, measured against the live artifact:

  * TWENTY signed players invisible on the Trade Machine. Every "Draft pick"
    row was by then a SIGNED standard contract in master_roster, on the right
    team, at $1.4M to $8.8M -- hidden by a filter that read the role as "not
    signed yet", while their salary stayed inside the team's payroll.
  * Mason Plumlee (SAC) and Orlando Robinson (CHI) on no card at all.
  * Nique Clifford shown on SACRAMENTO after a trade to Oklahoma City.

The split the builder makes, and that these tests pin:

  ROSTER FACTS -- team, kind, salary -- are REBUILT from data/master_roster.csv
  every night. The spine is not allowed to disagree with it.

  OFFSEASON NARRATIVE -- role, prev, unsigned -- is history, carried forward.

The age gate lives in build_team_pages (step 5b) and the remedy in step 5a
immediately before it. That ordering is deliberate: a fatal check whose only
remedy is DOWNSTREAM of it arms itself and rolls back every night forever once
it first fires.
"""
from __future__ import annotations
import csv
import datetime as dt
import json
import logging
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.disable(logging.WARNING)
from utils import SEASONS, TWO_WAY_SALARY  # noqa: E402
logging.disable(logging.NOTSET)

SPINE = ROOT / "cache" / "team_rosters_2627.json"
BUILDER = ROOT / "scripts" / "build_roster_spine.py"
PAGES_BUILDER = ROOT / "scripts" / "build_team_pages.py"
NIGHTLY = ROOT / "scripts" / "nightly.py"
MASTER = ROOT / "data" / "master_roster.csv"

MAX_AGE_DAYS = 14


def _hard(n):
    return re.sub(r"[^a-z0-9]", "", str(n).lower())


@pytest.fixture(scope="module")
def spine():
    if not SPINE.exists():
        pytest.skip("roster spine absent")
    return json.loads(SPINE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def master():
    out = {}
    for r in csv.DictReader(
            l for l in MASTER.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")):
        kind = (r.get("kind") or "").strip()
        if r.get("player") and kind in ("standard", "two_way"):
            out[_hard(r["player"])] = ((r.get("team") or "").strip(), kind,
                                       r["player"].strip())
    return out


# ── it has a builder, and it is stamped ──────────────────────────────────────

def test_the_spine_has_a_builder():
    assert BUILDER.exists(), (
        "scripts/build_roster_spine.py is gone; the spine has no scheduled "
        "builder again and will rot exactly as it did before 2026-11-19")


def test_the_builder_runs_before_the_page_build_that_gates_on_it():
    """Ordering, not presence. A fatal age gate whose only remedy runs AFTER it
    rolls back forever once it fires."""
    src = NIGHTLY.read_text(encoding="utf-8")
    spine_at = src.find("build_roster_spine.py")
    pages_at = src.find("build_team_pages.py")
    assert spine_at != -1, "nightly.py never runs scripts/build_roster_spine.py"
    assert pages_at != -1
    assert spine_at < pages_at, (
        "the roster spine is rebuilt AFTER team pages; the age gate in "
        "build_team_pages would then have no remedy ahead of it")


def test_the_spine_has_exactly_one_writer():
    """export_fa_spreadsheet.py used to write it as a side effect, which made
    an OFFSEASON simulation tool the de facto roster builder. Two writers with
    different semantics on one file is how it came to disagree with the
    roster -- and that one writes no built_at, which would now be refused.

    Resolves the VARIABLE the path is assigned to: both this builder and
    build_team_pages name the file, and only one of them writes it, so a
    whole-file grep for the name beside `write_text` reports both.
    """
    writers = []
    for f in sorted((ROOT / "scripts").glob("*.py")):
        src = f.read_text(encoding="utf-8", errors="ignore")
        for var in re.findall(r"^\s*(\w+)\s*=\s*[^\n]*team_rosters_2627\.json",
                              src, re.M):
            if re.search(rf"\b{re.escape(var)}\.write_text\s*\(", src):
                writers.append(f.name)
    assert writers == ["build_roster_spine.py"], (
        f"the roster spine is written by {writers or 'NOBODY'}; it must have "
        f"exactly one writer")


def test_the_spine_is_staged_by_the_nightly():
    """A file this script rebuilds is a file this script must ship."""
    assert '"cache/team_rosters_2627.json"' in NIGHTLY.read_text(encoding="utf-8"), (
        "the rebuilt spine is not in the nightly's staged paths, so it would "
        "be rebuilt every night and never committed")


def test_the_spine_is_stamped_and_fresh(spine):
    built = (spine.get("built_at") or "").strip()
    assert built, (
        "no built_at. File mtime lies -- git checkout, iCloud, Render's disk "
        "seed all rewrite it -- so an unstamped spine has no knowable age")
    age = (dt.date.today() - dt.date.fromisoformat(built)).days
    assert age <= MAX_AGE_DAYS, (
        f"the spine was built {built}, {age} days ago; build_team_pages "
        f"refuses over {MAX_AGE_DAYS}")


def _age_gate():
    """Import build_team_pages' age gate without running the builder."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_btp", PAGES_BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_team_pages_refuses_an_unstamped_spine():
    """BEHAVIOURAL, not a grep. The first version of this test only looked for
    "built_at" and "REFUSING" in the source, and a mutation that replaced the
    whole condition with `if False:` SURVIVED it."""
    m = _age_gate()
    with pytest.raises(SystemExit) as e:
        m.assert_spine_fresh({})
    assert "built_at" in str(e.value)


def test_build_team_pages_refuses_a_stale_spine():
    m = _age_gate()
    stale = (dt.date.today() - dt.timedelta(days=m.SPINE_MAX_DAYS + 1)).isoformat()
    with pytest.raises(SystemExit) as e:
        m.assert_spine_fresh({"built_at": stale})
    assert "REFUSING" in str(e.value)


def test_the_age_gate_is_actually_called():
    """The behavioural tests above exercise assert_spine_fresh directly, so
    they all still pass if main() simply stops calling it. Caught by the
    mutation run, not by thinking about it."""
    src = PAGES_BUILDER.read_text(encoding="utf-8")
    assert "assert_spine_fresh(ros)" in src, (
        "build_team_pages no longer calls the spine age gate, so a roster "
        "nobody has rebuilt would be published silently")


def test_build_team_pages_accepts_a_fresh_spine():
    """The gate must not be so tight that an ordinary skipped night wedges the
    nightly: the builder is step 5a and the gate is 5b, so a one-off failure
    has to be survivable."""
    m = _age_gate()
    ok = (dt.date.today() - dt.timedelta(days=m.SPINE_WARN_DAYS)).isoformat()
    assert m.assert_spine_fresh({"built_at": ok}) == m.SPINE_WARN_DAYS
    assert m.SPINE_MAX_DAYS == MAX_AGE_DAYS


# ── the facts must match the verified roster ─────────────────────────────────

def test_membership_matches_master_roster_exactly(spine, master):
    in_spine = {_hard(p["n"]): (a, p["n"])
                for a, t in spine["teams"].items() for p in t["players"]}
    missing = sorted(master[k][2] for k in master if k not in in_spine)
    extra = sorted(in_spine[k][1] for k in in_spine if k not in master)
    assert not missing, f"on master_roster but not the spine: {missing[:12]}"
    assert not extra, f"on the spine but not master_roster: {extra[:12]}"


def test_nobody_is_on_the_wrong_team(spine, master):
    wrong = []
    for a, t in spine["teams"].items():
        for p in t["players"]:
            k = _hard(p["n"])
            if k in master and master[k][0] != a:
                wrong.append((p["n"], a, master[k][0]))
    assert not wrong, f"(player, spine team, master_roster team): {wrong}"


def test_no_duplicate_players(spine):
    seen, dupes = {}, []
    for a, t in spine["teams"].items():
        for p in t["players"]:
            k = _hard(p["n"])
            if k in seen:
                dupes.append((p["n"], seen[k], a))
            seen[k] = a
    assert not dupes, f"same player on two spine rosters: {dupes}"


def test_roster_shape_invariants(spine, master):
    assert len(spine["teams"]) == 30, len(spine["teams"])
    bad = {}
    for a, t in spine["teams"].items():
        ks = [master.get(_hard(p["n"]), (None, None, None))[1] for p in t["players"]]
        n_std, n_tw = ks.count("standard"), ks.count("two_way")
        if not (14 <= n_std <= 15) or n_tw > 3:
            bad[a] = f"{n_std} standard + {n_tw} two-way"
    assert not bad, bad


def test_unsigned_never_names_a_rostered_player(spine, master):
    """`role` is a PAST-TENSE fact and is carried forward. `unsigned` is a
    PRESENT-TENSE claim, and seo_selfheal publishes it as one: "Still
    unsigned: X. No reported deal yet, and no open roster spot here."

    Carried forward blindly it had Atlanta's public team page saying that
    about CJ McCollum while the table on the SAME PAGE listed him rostered at
    $21.0M, and Memphis saying it about Javon Small, by then a Memphis
    two-way. Carry a past-tense fact; re-check a present-tense claim.
    """
    bad = [(a, u.get("n")) for a, t in spine["teams"].items()
           for u in t.get("unsigned", []) if _hard(u.get("n", "")) in master]
    assert not bad, f"named 'still unsigned' while on a roster: {bad}"


def test_unsigned_never_names_someone_out_of_the_league(spine):
    """"Still unsigned" means signable today, not retired or overseas."""
    from utils import out_of_league_players
    from utils import NameIndex
    gone = NameIndex(out_of_league_players())
    bad = [(a, u.get("n")) for a, t in spine["teams"].items()
           for u in t.get("unsigned", []) if gone.get(u.get("n", "")) is not None]
    assert not bad, f"named 'still unsigned' while out of the league: {bad}"


# ── the derived numbers must follow the facts ────────────────────────────────

def test_every_player_is_priced(spine):
    """A player with no salary is DROPPED from the board downstream, so an
    unpriced row here is an invisible player, not a blank cell."""
    unpriced = [(a, p["n"]) for a, t in spine["teams"].items()
                for p in t["players"] if p.get("salary") is None]
    assert not unpriced, unpriced[:10]


def test_two_ways_carry_the_league_scale(spine, master):
    scale = round(TWO_WAY_SALARY[SEASONS[0]] / 1e6, 1)
    wrong = [(a, p["n"], p["salary"]) for a, t in spine["teams"].items()
             for p in t["players"]
             if master.get(_hard(p["n"]), (None, None, None))[1] == "two_way"
             and p["salary"] != scale]
    assert not wrong, f"two-ways not at the ${scale}M scale: {wrong[:8]}"


def test_total_and_size_are_computed_from_the_players(spine):
    """`total` is the EXACT contract sum rounded ONCE, so it is not required to
    equal the sum of the published rows: every row is rounded to $0.1M for
    display, and fifteen of those drift up to $0.75M from the truth.

    That gap is the whole point. Summing the rounded rows put Cleveland at
    $206.9M against a real $206.748M, and the $0.15M of accumulated error is
    larger than the $23K by which Cleveland clears its first-apron hard cap --
    so once dead money went into the baseline, the rounded sum alone published
    a Cleveland that could never trade. The tolerance here therefore scales
    with the roster instead of pretending the two sums are one number, and the
    EXACT figure is pinned against master_roster in
    tests/test_dead_money_baseline.py.
    """
    off = {}
    for a, t in spine["teams"].items():
        rows = round(sum(p["salary"] or 0 for p in t["players"]), 1)
        slack = 0.05 * len(t["players"]) + 0.051
        if abs(t["total"] - rows) > slack or t["size"] != len(t["players"]):
            off[a] = (t["total"], rows, t["size"], len(t["players"]))
    assert not off, f"(total, sum of rows, size, want): {off}"
    assert all(t["total"] >= t["std_total"] - 0.051
               for t in spine["teams"].values()), "std_total exceeds total"


def test_the_seasons_are_the_current_one(spine):
    """value_season sat at 2025-26 after the rollover and only did no damage
    because nothing read it."""
    assert spine["contract_season"] == SEASONS[0], spine["contract_season"]
    assert spine["value_season"] == SEASONS[0], spine["value_season"]


def test_room_measures_the_ceiling_that_actually_BINDS_each_team(spine):
    """`room` is the only figure on a public team page that tells a reader how
    much a team can still spend, so it has to measure the line that stops it.

    It used to be apron-2 room for EVERY team, whatever hard cap the team
    carries. Twenty-one of the twenty-five capped teams are capped at the
    FIRST apron, so for them the figure measured a ceiling they cannot reach:
    Cleveland's page read "$12.8M under the second apron" on a night a hard cap
    it may not exceed sat $0.02M away. The number was true and the label
    matched it. The LINE was the wrong one to report, which is a harder defect
    to see than a wrong number.

    Now room is measured to the team's own binding ceiling and `room_line`
    names it, so a figure and its label cannot drift apart again.
    """
    import csv
    cfg = json.loads((ROOT / "data" / "cba_config_2026_27.json")
                     .read_text(encoding="utf-8"))
    lvl = {"apron1": float(cfg["apron1"]), "apron2": float(cfg["apron2"])}
    caps = {}
    hc = ROOT / "data" / "hard_caps_2026_27.csv"
    if hc.exists():
        for r in csv.DictReader(l for l in hc.read_text(encoding="utf-8").splitlines()
                                if l.strip() and not l.lstrip().startswith("#")):
            caps[(r.get("team") or "").strip()] = (r.get("cap") or "").strip()

    # against `committed`, the EXACT sum, not the rounded player rows: the
    # spine computes room from exact figures precisely so a team with $0.023M
    # of margin is not published with $0.1M, and a test that reconstructs the
    # payroll from display values measures a different quantity.
    off, labels = {}, {}
    for a, t in spine["teams"].items():
        committed = t.get("committed")
        assert committed is not None, f"{a} publishes no exact committed figure"
        ceiling = lvl.get(caps.get(a, "none"), lvl["apron2"])
        want = round(ceiling - committed, 1)
        if abs(t["room"] - want) > 0.0501:
            off[a] = (t["room"], want)
        labels[a] = t.get("room_line")
    assert not off, (
        f"room does not measure each team's binding ceiling (got, want): {off}")

    assert all(labels.values()), (
        f"teams with no room_line, so the figure says nothing about which line "
        f"it measures: {[a for a, v in labels.items() if not v]}")
    assert len({v for v in labels.values()}) >= 2, (
        f"every team carries the same room_line ({set(labels.values())}), so "
        f"either the hard-cap ledger was not read or the label is a constant")


def test_room_agrees_with_the_sentence_the_team_page_builds(spine):
    """The rendered claim must be TRUE: a team under its ceiling must not read
    'over'. seo_selfheal prints room with room_line, so this pins the sign
    against the SAME ceiling the label names."""
    import csv
    cfg = json.loads((ROOT / "data" / "cba_config_2026_27.json")
                     .read_text(encoding="utf-8"))
    lvl = {"apron1": float(cfg["apron1"]), "apron2": float(cfg["apron2"])}
    caps = {}
    hc = ROOT / "data" / "hard_caps_2026_27.csv"
    if hc.exists():
        for r in csv.DictReader(l for l in hc.read_text(encoding="utf-8").splitlines()
                                if l.strip() and not l.lstrip().startswith("#")):
            caps[(r.get("team") or "").strip()] = (r.get("cap") or "").strip()

    lying = []
    for a, t in spine["teams"].items():
        ceiling = lvl.get(caps.get(a, "none"), lvl["apron2"])
        really_over = float(t["committed"]) > ceiling
        # room rounds to $0.1M, so a team inside half a tick of its ceiling
        # publishes 0.0 and reads neither over nor under. Only a figure that
        # states the WRONG side is a lie.
        says_over = t["room"] < -0.0501
        says_under = t["room"] > 0.0501
        if (really_over and says_under) or (not really_over and says_over):
            lying.append((a, t["committed"], t["room"], t.get("room_line")))
    assert not lying, (
        f"the team page would state the wrong side of the line it names "
        f"(team, standard payroll, room, line): {lying}")


def test_the_renderer_names_the_line_rather_than_hardcoding_one():
    """The defect was a hardcoded "the second apron" in the only place room is
    ever shown to a reader. If that string comes back, the label and the figure
    can disagree again without anything failing."""
    import re
    src = (ROOT / "seo_selfheal.py").read_text(encoding="utf-8")
    blk = src.split('room = team.get("room")', 1)[1][:1200]
    assert "room_line" in blk, (
        "seo_selfheal does not read room_line, so it cannot name the ceiling "
        "the figure measures")

    # The first version of this test only banned the literal string "under the
    # second apron", and mutation testing showed that is not enough: replacing
    # the whole expression with where = "the second apron" leaves that literal
    # absent and the sentence just as wrong. Ban an apron NAME on the
    # right-hand side instead, whatever shape the hardcoding takes.
    assigns = re.findall(r"where\s*=\s*(.+)", blk)
    assert assigns, "the renderer no longer computes a name for the line"
    hard = [a for a in assigns if "apron" in a.lower()]
    assert not hard, (
        f"the line name is hardcoded rather than taken from room_line, so the "
        f"figure and its label can disagree again: {hard}")


def test_the_roster_floor_matches_check_roster_invariants():
    """A team MAY carry 13 standard contracts for up to two weeks.
    check_roster_invariants WARNS at 13 and only FAILS below it.

    This builder refused anything under 14 in its first version, which would
    have frozen the WHOLE LEAGUE's spine on a legal, correctly-swept roster --
    and step 5a is non-fatal, so the only trace would have been one warning
    line in the nightly log while every team page quietly went stale, which is
    the exact failure this builder exists to prevent.
    """
    import importlib.util
    inv = ROOT / "scripts" / "check_roster_invariants.py"
    src = inv.read_text(encoding="utf-8")
    m = re.search(r"MAX_STD,\s*MIN_STD,\s*HARD_MIN_STD,\s*MAX_TW\s*=\s*"
                  r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+)", src)
    assert m, "could not read the thresholds from check_roster_invariants"
    max_std, min_std, hard_min, max_tw = (int(x) for x in m.groups())

    spec = importlib.util.spec_from_file_location("_brs", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.MAX_STANDARD == max_std, (mod.MAX_STANDARD, max_std)
    assert mod.WARN_STANDARD == min_std, (mod.WARN_STANDARD, min_std)
    assert mod.HARD_MIN_STANDARD == hard_min, (
        f"the builder refuses below {mod.HARD_MIN_STANDARD} while "
        f"check_roster_invariants only fails below {hard_min}; a legal "
        f"{hard_min}-man roster would freeze the spine for the whole league")
    assert mod.MAX_TWO_WAY == max_tw


# ── the filter that hid twenty signed players ────────────────────────────────

def test_the_trade_machine_does_not_hide_players_by_role():
    src = (ROOT / "pages" / "Trade_Machine.py").read_text(encoding="utf-8")
    assert 'p["role"] == "Draft pick"' not in src, (
        "the Trade Machine filters on an offseason role again; every one of "
        "those rows was a signed standard contract on 2026-11-19")


def test_every_rostered_player_reaches_a_trade_card(master):
    pages = ROOT / "cache" / "team_pages.json"
    if not pages.exists():
        pytest.skip("team_pages absent")
    tp = json.loads(pages.read_text(encoding="utf-8"))["teams"]
    shown = {_hard(p["n"]) for t in tp.values() for p in t["players"]
             if p.get("real", True) and p.get("salary") is not None}
    missing = sorted(master[k][2] for k in master if k not in shown)
    assert not missing, (
        f"{len(missing)} rostered player(s) on no trade card: {missing[:12]}")
