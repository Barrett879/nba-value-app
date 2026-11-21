"""A signed player's 2026 deal does not decide his 2027 option.

Found by review of 900f93c, 2026-11-21. pages/Free_Agent_Class.py inferred an
option decision from mere membership of the tracked-signings dict:

    if not dec and signed:                   # "signed players reveal their call"
        if status == "Player Option": dec = "po_out"
        elif status == "Team Option": dec = "to_out"

True on the 2026 board: a player holding a 2026-27 option who signed a new 2026
deal must have opted out to reach the market. BACKWARDS on the 2027 board, where
the 2026 signing IS the contract containing the 2027-28 option. 21 live rows
published "PO Opt Out" or "TO Declined"; data/contract_years_2026_27.csv shows a
live 2027-28 option on 17 of them, and every one of those 17 is a two-year 2026
deal whose SECOND YEAR IS THAT OPTION. Each row contradicted itself, Status
"Player Option" beside Outcome "PO Opt Out", while
data/option_decisions_2027.csv says in its header that no 2027 decision exists.

THIRD INSTANCE OF ONE BUG CLASS IN THIS FILE inside 36 hours -- the _miss rescue
clause, the model_M filter, and this -- so these tests exist to make the fourth
one fail a test instead of reaching the live board.

THE PAGE'S OWN SOURCE IS THE SUBJECT. A Streamlit page cannot be imported, and
the previous two bugs both survived tests that reimplemented the logic they were
checking, so these tests lift the real _fa_outcome and the real gate expression
out of the page file with ast and execute THOSE. Change the page and the test
runs the change.
"""
from __future__ import annotations
import ast
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
BOARD_YEAR = int(utils.SEASONS[0][:4]) + 1
OPTION_FLAGS = {"P", "T", "M"}


def _page_tree():
    return ast.parse(PAGE.read_text(encoding="utf-8"))


def _lift(names, extra_globals):
    """Execute the page's OWN definitions of `names` in a controlled namespace.

    Not a copy of the logic: the nodes come from the page file as it stands on
    disk, so a change to _fa_outcome or to the gate is a change to what runs
    here. Anything the lifted code needs that is not lifted comes in through
    extra_globals, which is what lets a test vary one input at a time.
    """
    tree = _page_tree()
    wanted, found = list(names), {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            found[node.name] = node
        elif isinstance(node, (ast.Assign, ast.Try)):
            tgts = [t.id for n in ast.walk(node) if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)]
            for w in wanted:
                if w in tgts:
                    found[w] = node
    missing = [w for w in wanted if w not in found]
    assert not missing, (
        f"{PAGE.name} no longer defines {missing} at module level, so this test "
        "is no longer reading the page's own logic. Re-point it, do not delete it.")
    ns = dict(extra_globals)
    mod = ast.Module(body=[found[w] for w in wanted], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(PAGE), "exec"), ns)  # noqa: S102
    return ns


def _flags():
    rows = list(csv.DictReader(
        l for l in CY.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.lstrip().startswith("#")))
    out = {}
    for r in rows:
        if r.get("player"):
            out[normalize(r["player"])] = next(
                (x.split(":")[1] for x in (r.get("seasons") or "").split("|")
                 if x.startswith(str(BOARD_YEAR))), None)
    assert out, f"{CY.name} parsed to nothing"
    return out


def _outcome_ns(season, decisions=None, signed=None):
    """The page's _fa_outcome, with the tracker's season as the only variable."""
    return _lift(
        ["_INFER_OPTION_CALL", "_fa_outcome"],
        {"_acc": {"season": season}, "_FA_YEAR": BOARD_YEAR,
         "normalize": normalize,
         "_decisions": decisions or {},
         "_signed": signed if signed is not None else utils.tracked_signings(),
         "_OUTCOME_LABEL": {"po_in": "PO Opt In", "po_out": "PO Opt Out",
                            "to_in": "TO Picked Up", "to_out": "TO Declined"},
         "_gone_label": lambda name: None})


# --- the gate itself -------------------------------------------------------

def test_the_inference_is_off_while_the_board_outlives_the_signings_class():
    """Production values, not a constructed case: tracker season vs board year."""
    acc = utils.accuracy_tracker(refresh=True)
    assert acc, "the accuracy tracker read as empty, so this proves nothing"
    season = acc.get("season")
    assert season is not None, (
        "the tracker has no 'season', so nothing can say which class its "
        "signings belong to and the gate has no basis to be on")
    ns = _lift(["_INFER_OPTION_CALL"], {"_acc": acc, "_FA_YEAR": BOARD_YEAR})
    assert ns["_INFER_OPTION_CALL"] is (int(season) == BOARD_YEAR)
    if int(season) != BOARD_YEAR:
        assert ns["_INFER_OPTION_CALL"] is False, (
            f"the tracker holds the {season} class and the board is {BOARD_YEAR}, "
            "so a 2026 signing cannot reveal a 2027 option decision")


def test_the_inference_still_fires_on_the_signings_own_board():
    """POSITIVE CONTROL. A gate hardwired to False would pass every test above."""
    ns = _outcome_ns(str(BOARD_YEAR))
    assert ns["_INFER_OPTION_CALL"] is True
    name = next(iter(utils.tracked_signings()))
    sample = [s["player"] for s in utils.accuracy_tracker()["signings"]][:1]
    assert sample, "no signings to test with"
    got = ns["_fa_outcome"](sample[0], "Player Option", "$9.9M")
    assert "PO Opt Out" in got, (
        "on a board showing the signings' OWN class the inference must still "
        f"fire; got {got!r} for {sample[0]!r}")
    assert name  # the dict is keyed, not a list


def test_a_missing_or_unparseable_season_does_not_infer():
    """Fail SAFE: no basis means no claim."""
    for season in (None, "", "not-a-year", {}):
        ns = _lift(["_INFER_OPTION_CALL"],
                   {"_acc": {"season": season}, "_FA_YEAR": BOARD_YEAR})
        assert ns["_INFER_OPTION_CALL"] is False, f"season={season!r} inferred anyway"
    ns = _lift(["_INFER_OPTION_CALL"], {"_acc": None, "_FA_YEAR": BOARD_YEAR})
    assert ns["_INFER_OPTION_CALL"] is False, "an absent tracker inferred anyway"


# --- the published string --------------------------------------------------

def test_a_live_option_is_never_published_as_already_decided():
    """The 17 rows the bug was wrong about, named, against our own contract data.

    Every one is a two-year 2026 deal whose second year IS the option, so
    "he signed, therefore he declined it" is self-contradictory.
    """
    acc = utils.accuracy_tracker(refresh=True)
    ns = _outcome_ns(acc.get("season"))
    flags = _flags()
    signed = utils.tracked_signings()
    checked = []
    for n, s in signed.items():
        flag = flags.get(n)
        if flag not in OPTION_FLAGS:
            continue
        status = {"P": "Player Option", "T": "Team Option",
                  "M": "Player Option"}[flag]
        got = ns["_fa_outcome"](s["player"], status, "$9.9M")
        checked.append(s["player"])
        assert "Declined" not in got and "Opt Out" not in got, (
            f"{s['player']} holds a live {BOARD_YEAR}-{str(BOARD_YEAR+1)[2:]} "
            f"{status} in {CY.name} and no decision is recorded, but the board "
            f"publishes {got!r}")
        assert "Signed" in got, (
            f"{s['player']} did sign, so the outcome must still say so; got {got!r}")
    assert len(checked) >= 15, (
        f"only {len(checked)} tracked signings resolved to a live option flag; "
        "this test has stopped covering the rows the bug was about")


def test_a_recorded_decision_is_published_whatever_the_gate_says():
    """The gate suppresses an INFERENCE. Sourced data is not an inference."""
    acc = utils.accuracy_tracker(refresh=True)
    name = next(s["player"] for s in acc["signings"])
    n = normalize(name)
    for dec, label in (("po_out", "PO Opt Out"), ("to_out", "TO Declined"),
                       ("po_in", "PO Opt In"), ("to_in", "TO Picked Up")):
        ns = _outcome_ns(acc.get("season"), decisions={n: (dec, 9.9)})
        got = ns["_fa_outcome"](name, "Player Option", "$1.1M")
        assert label in got, (
            f"a recorded {dec} for {name} was not published; got {got!r}. The "
            "gate must only stop the inference, never a sourced decision.")


def test_an_unsigned_player_still_falls_back_to_his_pending_figure():
    """Nothing else in the column moved."""
    acc = utils.accuracy_tracker(refresh=True)
    ns = _outcome_ns(acc.get("season"), signed={})
    assert ns["_fa_outcome"]("Nobody At All", "Player Option", "$7.7M") == "$7.7M"
