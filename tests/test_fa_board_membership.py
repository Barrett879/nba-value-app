"""A player under contract must not appear on the free-agent board.

Barrett, 2026-11-20: "it says Quentin Grimes is going to be an UFA". He signed
4yr/$60M with the Lakers in July 2026 and is under contract to 2029-30, and
every roster file in this repo says so. The Free Agent Class page listed him
anyway, as a **UFA from PHI**, on a board headed "2027 Free Agent Board".

THE MECHANISM, and it was a correct idea applied a season too late.
pages/Free_Agent_Class.py rescues a tracked signing the classifier returns None
for, because the salary feed sometimes reports an option year as a plain
guaranteed salary and a player who genuinely reaches the market would vanish:

    _miss = Status.isna() & Player in _signed
    fa_df.loc[_miss, "Status"] = ... _signing_status   # defaults to "UFA"

That was written when the board year and the tracked signings were the SAME
offseason. A season later they are not, so the clause re-added every tracked
2026 signing the classifier had correctly excluded -- 35 of 111 -- each
labelled UFA because that is the default. All 35 were on a 2026-27 roster.

THE FIX asks the question the rescue always assumed: is he signed THROUGH the
board year? data/contract_years_2026_27.csv answers it per season, and an
option year counts as NOT under contract, which is exactly the case worth
rescuing. 31 of the 35 left; the 4 that stayed carry a 2027-28 option (P, M,
M, T), which is the whole point of the rescue surviving.
"""
from __future__ import annotations
import csv
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.disable(logging.WARNING)
import utils  # noqa: E402
from utils import normalize  # noqa: E402
logging.disable(logging.NOTSET)

PAGE = ROOT / "pages" / "Free_Agent_Class.py"
CY = ROOT / "data" / "contract_years_2026_27.csv"
ACC = ROOT / "cache" / "accuracy_tracker_v1.json"
MASTER = ROOT / "data" / "master_roster.csv"
CUR = utils.SEASONS[0]
BOARD_YEAR = int(CUR[:4]) + 1


def _rows(path: Path):
    return list(csv.DictReader(
        l for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.lstrip().startswith("#")))


@pytest.fixture(scope="module")
def flags():
    """normalized name -> {season: flag} from contract_years."""
    out = {}
    for r in _rows(CY):
        if r.get("player"):
            out[normalize(r["player"])] = dict(
                x.split(":", 1) for x in (r.get("seasons") or "").split("|") if ":" in x)
    return out


@pytest.fixture(scope="module")
def tracked():
    if not ACC.exists():
        pytest.skip("accuracy tracker absent")
    d = json.loads(ACC.read_text(encoding="utf-8"))
    return {normalize(s["player"]): s for s in d.get("signings", [])
            if s.get("model_M") is not None}


def _board_status(name, feed):
    return utils.classify_fa_status(
        name, utils.fmt_next_contract(name, feed), set(), CUR, cross_check=True)


@pytest.fixture(scope="module")
def feed():
    try:
        return utils.fetch_next_year_contracts(
            utils.season_to_espn_year(CUR), cache_v=7) or {}
    except Exception:
        pytest.skip("next-year feed unavailable")


def _signed_through(n, flags, tracked):
    """PRODUCTION CODE, not a copy. The first version of this file
    reimplemented the gate here, so mutating the real one changed nothing and
    three of four mutations survived. utils.signed_through_season is what the
    board calls."""
    return utils.signed_through_season(
        tracked[n]["player"], BOARD_YEAR, (tracked.get(n) or {}).get("years"))


def test_the_gate_splits_on_the_option_flag(flags, tracked):
    """THE SPEC, stated so a broken gate cannot pass it.

    A guaranteed board-year season means under contract; an OPTION year (P, T
    or M) means the player can reach the market and belongs on the board. Both
    directions are asserted, so a gate that always says "signed" and a gate
    that always says "free" each fail, and so does one that folds options in
    with guarantees.
    """
    gtd = [n for n in tracked if flags.get(n, {}).get(str(BOARD_YEAR)) == "G"]
    opt = [n for n in tracked if flags.get(n, {}).get(str(BOARD_YEAR)) in ("P", "T", "M")]
    assert len(gtd) >= 5 and len(opt) >= 2, (
        f"not enough tracked signings to test both directions "
        f"(guaranteed {len(gtd)}, option {len(opt)})")

    missed = [tracked[n]["player"] for n in gtd if not _signed_through(n, flags, tracked)]
    assert not missed, (
        f"these carry a guaranteed {BOARD_YEAR}-{str(BOARD_YEAR + 1)[2:]} "
        f"season and the gate still calls them free agents: {missed}")

    held = [tracked[n]["player"] for n in opt if _signed_through(n, flags, tracked)]
    assert not held, (
        f"these have an OPTION in {BOARD_YEAR}-{str(BOARD_YEAR + 1)[2:]}, so "
        f"they can reach the market, but the gate treats them as signed and "
        f"would drop them off the board: {held}")


def test_the_gate_agrees_with_the_roster(flags, tracked, feed):
    """THE GATE'S VERDICT, checked against a source it does not read.

    Restating the gate's own condition back at it would be vacuous -- the first
    version of this test did exactly that and "failed" by listing the 31
    players the gate correctly excludes. So check its ANSWER instead:
    master_roster is built from different files and knows nothing about
    contract_years, so if the gate says a man is signed through the board year
    he had better be on somebody's roster now.
    """
    roster = {normalize(r["player"]) for r in _rows(MASTER)
              if (r.get("kind") or "").strip() in ("standard", "two_way")}
    rescue_pool = [n for n, s in tracked.items()
                   if _board_status(s["player"], feed) is None]
    assert len(rescue_pool) >= 10, (
        f"only {len(rescue_pool)} tracked signings reach the rescue, so this "
        f"test is not exercising it")

    held = [n for n in rescue_pool if _signed_through(n, flags, tracked)]
    assert held, "the gate holds nobody back, so it is not gating"

    ghosts = [tracked[n]["player"] for n in held if n not in roster]
    assert not ghosts, (
        f"the gate calls these players signed through {BOARD_YEAR}-"
        f"{str(BOARD_YEAR + 1)[2:]} but no roster carries them, so it is "
        f"hiding real free agents: {ghosts}")

    through = [n for n in rescue_pool if not _signed_through(n, flags, tracked)]
    unexplained = [tracked[n]["player"] for n in through
                   if flags.get(n, {}).get(str(BOARD_YEAR)) not in ("P", "T", "M")
                   and int(tracked[n].get("years") or 0) > 1]
    assert not unexplained, (
        f"these are on multi-year deals with no option in the board year, yet "
        f"the gate lets them back onto it: {unexplained}")


def test_a_tracked_signing_with_a_real_option_year_is_labelled_as_one(flags, tracked, feed):
    """POSITIVE CONTROL. REWRITTEN 2026-11-20, when its MECHANISM changed.

    It used to assert that at least one tracked signing SURVIVES the rescue below,
    which held only because the salary feed read some option years as plain
    guaranteed salaries and classify_fa_status returned None for them.
    classify_fa_status now reads the per-season contract flags itself, so those
    players are labelled directly and never reach the rescue at all -- which made
    the old assertion empirically false the moment the upstream defect was fixed.

    The thing worth protecting was never the route, it is the OUTCOME: a tracked
    signing who can genuinely reach the market in the board year must appear, and
    appear as an OPTION rather than as a UFA on no evidence. Asserting the label
    keeps this falsifiable; asserting "he is on the board" would not, because an
    option holder is kept either by the classifier or by the rescue and the test
    could never fail.

    The rescue is not dead. It still covers a tracked signing with NO
    contract_years row whose option the feed misreads. It just has nothing to do
    today.
    """
    holders = [(n, s["player"]) for n, s in tracked.items()
               if flags.get(n, {}).get(str(BOARD_YEAR)) in ("P", "T", "M")]
    assert len(holders) >= 5, (
        f"only {len(holders)} tracked signings hold a real {BOARD_YEAR}-"
        f"{str(BOARD_YEAR + 1)[2:]} option, so this control is not exercising "
        f"anything")

    wrong = [(p, _board_status(p, feed)) for _n, p in holders
             if _board_status(p, feed) not in ("Player Option", "Team Option")]
    assert not wrong, (
        f"these tracked signings hold a real option in the board year and the "
        f"classifier does not say so, so each is either off the board or on it as "
        f"a UFA by default: {wrong[:20]}")


def test_quentin_grimes_is_not_a_free_agent(flags, tracked, feed):
    """The report, pinned by name. Four roster files agree he is a Laker under
    contract to 2029-30; only the board disagreed."""
    n = normalize("Quentin Grimes")
    assert _board_status("Quentin Grimes", feed) is None, (
        "the classifier now calls Grimes a free agent, which the roster denies")
    assert _signed_through(n, flags, tracked), (
        "Grimes is no longer read as signed through the board year, so the "
        "board would re-add him")
    roster = {normalize(r["player"]): r for r in _rows(MASTER)
              if (r.get("kind") or "").strip() == "standard"}
    assert n in roster and roster[n]["team"].strip() == "LAL", (
        "Grimes is not on the Lakers' standard roster any more; reread this")


def test_the_page_gates_the_rescue_rather_than_trusting_the_tracker():
    """Pins the shape, because the old clause reads as obviously correct: a
    tracked signing IS a free agent who came off the board, just not off THIS
    board. The membership question has to be asked explicitly."""
    src = PAGE.read_text(encoding="utf-8")
    assert "_signed_through_board_year" in src, (
        "the board no longer asks whether a rescued player is still under "
        "contract in the board year")
    blk = src.split("_miss = ", 1)[1].split("fa_df = fa_df[", 1)[0]
    assert "_signed_through_board_year" in blk, (
        "the rescue is no longer gated on it")
    assert "contract_years_2026_27.csv" in src, (
        "the page does not read the per-season contract flags")


def test_the_transcribed_option_row_is_present(flags):
    """Moritz Wagner has no Spotrac row, and both real_signings_2026.csv and
    master_roster.csv record his deal as 2yr/$19M with a SECOND-YEAR MUTUAL
    OPTION. Without the transcribed row the fallback reads "2-year deal" and
    drops him, hiding a player who can genuinely reach the market. Pinned
    because the row looks like redundant data until you delete it."""
    f = flags.get(normalize("Moritz Wagner"))
    assert f is not None, (
        "Moritz Wagner's contract_years row is gone; the board will treat his "
        "mutual option as a guaranteed second year and drop him")
    assert f.get(str(BOARD_YEAR)) == "M", (
        f"Wagner's {BOARD_YEAR} flag is {f.get(str(BOARD_YEAR))!r}, not the "
        f"mutual option both roster files record")
