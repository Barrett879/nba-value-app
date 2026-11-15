"""The nightly's two guards against shipping a file that exists nowhere.

Render's /data seeds a repo cache file only when that FILENAME is missing, so
the moment DATA_REV moves, production starts asking for names that may exist
nowhere. check_cache_seeding is the check for exactly that, and the nightly
excluded it from the fatal set because it "cannot pass until the new files are
staged". That excuse is true for most of what it reports and false for an
artifact that was never built: on 2026-11-14 the run bumped DATA_REV to r41 and
committed with cache/splits_2026_27_v7_r41.pkl never built, because the NBA
stats API returned nothing ten attempts running.

Two things had to change, and these pin both.

  * the excuse has a LIFETIME. The check is re-asked once everything is staged,
    and a failure there is fatal. It reads the INDEX (git ls-files), which is
    why "after git add, before git commit" is the only window where the
    question is meaningful.
  * a failed splits build carries the newest existing revision forward under
    tonight's name, instead of leaving the name missing and trusting
    _rev_path's fallback. That fallback goes to the UN-REVISIONED floor, not to
    the previous revision, and on Render the floor is frozen at whatever was
    seeded first.
"""
from __future__ import annotations
import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location("_nightly", ROOT / "scripts" / "nightly.py")
nightly = importlib.util.module_from_spec(_spec)
sys.modules["_nightly"] = nightly
_spec.loader.exec_module(nightly)          # module level only; main() is guarded


# ───────────────────────────── newest_revision ───────────────────────────────
def test_revisions_order_by_number_not_string(tmp_path):
    """r9 sorts after r40 as a string, which would carry the OLDER file
    forward and look like it worked."""
    for n in (7, 9, 33, 40):
        (tmp_path / f"splits_2026_27_v7_r{n}.pkl").write_bytes(b"x")
    assert nightly.newest_revision(tmp_path, "splits_2026_27_v7")[0] == 40


def test_revisions_ignore_mtime(tmp_path):
    """A git checkout or a copy hands an old artifact a new timestamp, so the
    ordering must not come from the clock."""
    import os
    old = tmp_path / "splits_2026_27_v7_r40.pkl"
    new = tmp_path / "splits_2026_27_v7_r9.pkl"
    old.write_bytes(b"x"); new.write_bytes(b"y")
    os.utime(old, (1, 1))                               # r40 looks ancient
    assert nightly.newest_revision(tmp_path, "splits_2026_27_v7")[0] == 40


def test_exclude_skips_the_target(tmp_path):
    for n in (33, 40):
        (tmp_path / f"splits_2026_27_v7_r{n}.pkl").write_bytes(b"x")
    ex = tmp_path / "splits_2026_27_v7_r40.pkl"
    assert nightly.newest_revision(tmp_path, "splits_2026_27_v7", exclude=ex)[0] == 33


def test_other_seasons_and_shapes_do_not_match(tmp_path):
    (tmp_path / "splits_2025_26_v7_r99.pkl").write_bytes(b"x")
    (tmp_path / "splits_2026_27_v7.pkl").write_bytes(b"floor")
    (tmp_path / "splits_2026_27_v8_r50.pkl").write_bytes(b"other formula")
    (tmp_path / "splits_2026_27_v7_rXX.pkl").write_bytes(b"not a number")
    assert nightly.newest_revision(tmp_path, "splits_2026_27_v7") is None


def test_no_directory_is_not_an_error(tmp_path):
    assert nightly.newest_revision(tmp_path / "absent", "splits_2026_27_v7") is None


def test_days_old(tmp_path):
    f = tmp_path / "x.pkl"; f.write_bytes(b"x")
    assert nightly.days_old(f) == 0
    assert nightly.days_old(tmp_path / "missing.pkl") == -1.0


# ────────────────────── the window the check has to run in ───────────────────
def _main_body():
    tree = ast.parse((ROOT / "scripts" / "nightly.py").read_text())
    return next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")


def _str_args(call):
    """Every string literal anywhere in a call's positional arguments.

    Walked, not indexed one level: the nightly stages with
    subprocess.run(["git", "add", "--"] + paths, ...), whose first argument is
    a BinOp rather than a list, so a one-level flatten finds nothing and the
    staging step goes unseen.
    """
    out = []
    for a in call.args:
        for n in ast.walk(a):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.append(n.value)
    return out


def _ordered_marks():
    """(lineno, mark) for the three operations whose ORDER is the guarantee.

    Matched on the ARGUMENTS of a subprocess call, not on the text of the
    node. The first version of this test matched any call whose source
    mentioned check_cache_seeding, and the step-4 print() that explains why the
    check is excluded mentions it by name, so the test was reading a message
    instead of the call it is supposed to be about.
    """
    out = []
    for node in ast.walk(_main_body()):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")) != "run":
            continue
        args = _str_args(node)
        if "git" in args and "add" in args:
            out.append((node.lineno, "add"))
        elif any(a.endswith("check_cache_seeding.py") for a in args):
            out.append((node.lineno, "seed"))
        elif "git" in args and "commit" in args:
            out.append((node.lineno, "commit"))
    return sorted(out)


def _seed_call_line():
    lines = [l for l, m in _ordered_marks() if m == "seed"]
    assert len(lines) == 1, f"expected one check_cache_seeding call, got {lines}"
    return lines[0]


def _calls_after_seed():
    """(lineno, callee name) for every call that follows the seed check, in
    source order.

    AST, not raw source. Reading the lines directly made the first version of
    this test match the word "rollback()" inside the COMMENT that explains why
    the reset has to come first, and conclude the order was wrong. Comments are
    not in the tree.
    """
    out = []
    for node in ast.walk(_main_body()):
        if not isinstance(node, ast.Call) or node.lineno <= _seed_call_line():
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        args = _str_args(node)
        if name == "run" and "git" in args and "reset" in args:
            out.append((node.lineno, "reset"))
        elif name == "rollback":
            out.append((node.lineno, "rollback"))
        elif name == "run" and "git" in args and "commit" in args:
            out.append((node.lineno, "commit"))
    return sorted(out)


def test_seeding_check_runs_between_staging_and_commit():
    marks = [m for _l, m in _ordered_marks()]
    assert "add" in marks, "the nightly no longer stages with git add"
    assert "seed" in marks, (
        "nothing re-runs check_cache_seeding in main(). It is excluded from the "
        "fatal set because it cannot pass before staging; if it is never asked "
        "again, a never-built artifact ships.")
    assert "commit" in marks, "the nightly no longer commits"
    assert marks.index("add") < marks.index("seed") < marks.index("commit"), (
        f"order is {marks}. check_cache_seeding reads the INDEX, so before "
        f"`git add` it always fails and after `git commit` it is too late.")


def test_a_failing_seed_check_unstages_before_rolling_back():
    """rollback() deletes the new artifacts. Leaving them staged-as-added-but-
    missing hands the next run a broken index instead of the clean tree
    rollback promises."""
    marks = [m for _l, m in _calls_after_seed()]
    assert "reset" in marks, \
        "no git reset between the failing seed check and the rollback"
    assert "rollback" in marks, "no rollback after the failing seed check"
    assert marks.index("reset") < marks.index("rollback"), \
        f"order after the seed check is {marks}; rollback deletes the new "


def test_the_seed_failure_is_fatal_not_a_warning():
    marks = [m for _l, m in _calls_after_seed()]
    assert "rollback" in marks, (
        "check_cache_seeding is re-run but its failure does not roll back. A "
        "check whose failure is explained away has stopped being a check.")


# ───────────────────── the splits step carries forward ───────────────────────
def test_splits_failure_carries_the_previous_revision_forward():
    body = ast.unparse(_main_body())
    assert "newest_revision" in body, (
        "the splits step no longer carries a previous revision forward. "
        "_rev_path falls back to the UN-REVISIONED floor, not to the previous "
        "revision, and on Render that floor is frozen at whatever /data seeded "
        "first, so leaving tonight's name missing serves an artifact of "
        "unbounded age.")
    assert "copyfile" in body, "nothing copies the carried revision into place"
    assert "copy2" not in body, (
        "copy2 preserves the source mtime, so the carried file would claim to "
        "be as old as what it came from and `ls -lt` would not show the carry")


def test_the_old_untrue_warning_is_gone():
    """It said Splits View would serve "the previous artifact through
    _rev_path's fallback". _rev_path does not walk back to the previous
    revision; it falls back to the un-revisioned floor, which was a week old."""
    src = (ROOT / "scripts" / "nightly.py").read_text()
    assert "serve \"\n" not in src
    assert "the previous artifact through _rev_path" not in src


def test_rev_path_still_falls_back_to_the_floor_only():
    """The premise behind the carry-forward. If _rev_path ever learns to walk
    back through revisions, the carry becomes redundant and this test says so
    rather than leaving two mechanisms that disagree."""
    import utils
    import re as _re
    src = _re.search(r"def _rev_path.*?\n(?=\n\ndef )",
                     (ROOT / "utils.py").read_text(), _re.S).group(0)
    assert "legacy" in src and "glob" not in src, (
        "_rev_path appears to search for other revisions now; revisit whether "
        "the nightly still needs to carry splits forward")
