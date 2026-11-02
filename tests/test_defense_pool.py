"""Invariants on the shipped defensive blend: every key is a real board player.

Written after 2026-11-02, when the artifact shipped with 354 "players" of whom
one was Darko Rajakovic, Toronto's HEAD COACH, keyed by his raw ESPN athlete id.
He reached the pool because ESPN's play-by-play names a coach as the sole
participant of a technical foul, the lineup rebuild read that as evidence he was
on the floor, and nothing between there and the artifact asked whether the key
was a person.

Runs under pytest and as a script -- tests/test_trade_rules.py is a script with
its own PASS/FAIL printing, and one directory should not hold two conventions:

    python3 -m pytest tests/test_defense_pool.py
    /opt/anaconda3/bin/python3 tests/test_defense_pool.py

Both tests read the COMMITTED artifacts, so they run in a clean checkout, and
both would have failed on the file shipped as 16ea188:

    no_bare_feed_ids      '3116516' is all digits
    every_key_is_on_the_board
                          '3116516' matches no board row, and 'Hansen Yang' is
                          the box feed's 'Yang Hansen' with the names reversed,
                          which NameIndex cannot bridge -- so the board silently
                          kept the box proxy for him
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Importing utils pulls in Streamlit's cache layer, which logs a warning per
# decorated function when there is no runtime. Silenced around the import the
# same way scripts/build_defense_blend.py does, so script mode prints its own
# PASS/FAIL lines and nothing else.
logging.disable(logging.WARNING)
from utils import DATA_REV, FORMULA_VERSION, NameIndex, SEASONS  # noqa: E402
logging.disable(logging.NOTSET)

SEASON = SEASONS[0]
_KEY = SEASON.replace("-", "_")


def _blend():
    import re
    src = (ROOT / "utils.py").read_text()
    m = re.search(rf'defense_blend_{{season.replace\(.-., ._.\)}}_(v\d+)\.json', src)
    ver = m.group(1) if m else None
    if ver is None:                      # the literal moved; find the newest file
        files = sorted((ROOT / "cache").glob(f"defense_blend_{_KEY}_v*.json"),
                       key=lambda p: int(p.stem.rsplit("_v", 1)[1]))
        if not files:
            pytest.skip("no defensive blend artifact committed")
        path = files[-1]
    else:
        path = ROOT / "cache" / f"defense_blend_{_KEY}_{ver}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return path, json.loads(path.read_text())["values"]


def _board():
    path = ROOT / "cache" / f"raw_{_KEY}_{FORMULA_VERSION}_{DATA_REV}.parquet"
    if not path.exists():
        pytest.skip(f"{path.name} not committed")
    return pd.read_parquet(path)


def test_no_bare_feed_ids():
    """A key that is all digits is an unresolved feed id, never a player."""
    path, values = _blend()
    bad = sorted(k for k in values if str(k).replace(" ", "").isdigit())
    assert not bad, (
        f"{path.name} keys {len(bad)} player(s) by a bare feed id: {bad}. "
        "An id reaching the artifact means something upstream could not name a "
        "participant and passed the id through; it arrives on the board as a "
        "plausible near-average defender. Fix the source, do not alias it."
    )


def test_every_key_is_on_the_board():
    """The board looks the blend up by its own PLAYER_NAME, so every key must
    resolve to a board row the same way the board resolves it. A key that does
    not is a value computed for nobody, and the player it was meant for keeps
    the box proxy without saying so."""
    path, values = _blend()
    board = _board()
    index = NameIndex({str(n): str(n) for n in board["Player"]})
    orphans = sorted(k for k in values if index.get(str(k)) is None)
    assert not orphans, (
        f"{path.name} has {len(orphans)} key(s) that match no board player: "
        f"{orphans}. Either the key is not a player, or it is the same player "
        "under a spelling NameIndex cannot bridge (a reversed name order is the "
        "one it cannot, by construction, since it compares surnames first). Add "
        "a def_lib.DEF_ALIASES entry for a spelling; drop a non-player."
    )


if __name__ == "__main__":
    # Script mode, matching tests/test_trade_rules.py: print a line per check
    # and exit non-zero on the first failure, so a probe reads as PASS or FAIL
    # rather than as a traceback to interpret.
    rc = 0
    for name, fn in (("no bare feed ids", test_no_bare_feed_ids),
                     ("every key is on the board", test_every_key_is_on_the_board)):
        try:
            fn()
            print(f"ok  {name}")
        except Exception as exc:                       # includes pytest's Skipped
            kind = "SKIP" if type(exc).__name__ == "Skipped" else "FAIL"
            print(f"{kind}  {name}: {exc}")
            if kind == "FAIL":
                rc = 1
    print("\nALL DEFENSIVE-POOL INVARIANTS HOLD" if rc == 0 else
          "\nDEFENSIVE-POOL INVARIANT VIOLATED")
    sys.exit(rc)
