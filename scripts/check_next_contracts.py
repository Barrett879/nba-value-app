"""Is the Next $ column actually populated, or is it a column of dashes?

    /opt/anaconda3/bin/python3 -u scripts/check_next_contracts.py [season]

The sentinel is load-bearing here, which is what makes this failure invisible:
an absent player renders as the gray dash, and the legend defines that dash as
"UFA". So a feed that returns nothing does not look broken. It looks like a
league where everybody is a free agent, which is a sentence nobody reads.

That is how it shipped. The season flipped on 2026-10-20, the code began asking
for next_contracts_2027_v7.pkl, which had never been built, and for a week every
one of the 372 players on the board showed a dash. Nothing logged, because
"empty" and "all free agents" are the same bytes.

The underlying trap is seasonal: _derive_next_year_contracts calls
fetch_salaries(espn_year + 1), and ESPN only publishes a season's salaries once
that season has begun. For eleven months of the year the live derivation CANNOT
work, so the committed artifact is the only thing standing between this column
and a wall of dashes. Rebuild it with scripts/build_next_contracts.py.

AGE (added 2026-11-13): counting rows cannot see staleness. This check passed
every night from 2026-10-27 onward on an artifact nobody was rebuilding, for
exactly the reason cache/share_meta.json sat four months out of date behind a
check that also only counted rows: "populated" and "current" are different
questions. The builder now writes a sidecar stamp recording when it RAN, and
this fails when that stamp is missing or older than MAX_AGE_DAYS.

That threshold is deliberately loose. The build step is non-fatal, so one night
of Spotrac being down ages the stamp without stopping anything; only the step
genuinely ceasing to run reaches a week. Failing sooner would block the whole
player refresh over a column that is merely a little behind.

    --selftest  break the inputs on purpose and require each check to fire

Exit code 1 on failure.
"""
from __future__ import annotations
import json, logging, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
import pandas as pd
import utils
from utils import fmt_next_contract
logging.disable(logging.NOTSET)

# Imported, not re-declared: a second copy of the threshold or the stamp path
# is the drift the nightly's own VERSIONED comment warns about.
from build_next_contracts import (MAX_AGE_DAYS, artifact_paths,  # noqa: E402
                                  stamp_path_for)

MIN_FEED = 200          # fetch_next_year_contracts' own "this scrape failed" floor
MIN_COVER = 0.50        # of the board; 2026-27 sits near 0.79


def stamp_age_days(espn_year: int, sp: Path | None = None):
    """(age_in_days, stamp_dict) or (None, None) when there is no readable
    stamp. Never raises. `sp` overrides the path so --selftest can drive THIS
    function over broken inputs instead of reimplementing it; a selftest that
    reimplements the parser only ever proves the copy works."""
    sp = sp or stamp_path_for(espn_year)
    if not sp.exists():
        return None, None
    try:
        st = json.loads(sp.read_text(encoding="utf-8"))
        built = str(st.get("built_at") or "")[:19]
        age = (time.time() - time.mktime(time.strptime(built, "%Y-%m-%dT%H:%M:%S"))) / 86400
    except Exception as e:
        # A stamp that EXISTS and cannot be read is a different fault from one
        # that was never written, and sending the reader to "run the builder"
        # when the real problem is a truncated file or a permission change
        # wastes the one message this check gets. Carry the reason out.
        return None, {"_unreadable": f"{type(e).__name__}: {e}"}
    # Negative age means the stamp is dated in the FUTURE: a clock that was
    # ahead when the builder ran, a restore, a manual date change. Treated as
    # an age of 0 it would pass the gate forever on an artifact nobody is
    # rebuilding, so it is surfaced as its own failure rather than clamped.
    return age, st


def _selftest() -> int:
    """A check that cannot fail proves nothing. Drive the REAL stamp_age_days
    over inputs that should trip it."""
    import tempfile, json as _j
    cases = []
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "s.json"

        def age_of(payload):
            if payload is None:
                f.unlink(missing_ok=True)
            else:
                f.write_text(_j.dumps(payload), encoding="utf-8")
            return stamp_age_days(0, sp=f)[0]          # the production parser

        def at(days_ago):
            return time.strftime("%Y-%m-%dT%H:%M:%S",
                                 time.localtime(time.time() - days_ago * 86400))

        a = age_of({"built_at": at(0), "entries": 341})
        cases.append(("a fresh stamp reads about 0 days", a is not None and a < 1))
        cases.append(("a fresh stamp passes the age gate",
                      a is not None and a <= MAX_AGE_DAYS))
        a = age_of({"built_at": at(30), "entries": 341})
        cases.append(("a 30-day stamp reads about 30 days", a is not None and 29 < a < 31))
        cases.append(("a 30-day stamp FAILS the age gate",
                      a is not None and a > MAX_AGE_DAYS))
        a = age_of({"built_at": at(MAX_AGE_DAYS + 1)})
        cases.append((f"a {MAX_AGE_DAYS + 1}-day stamp is just over the line",
                      a is not None and a > MAX_AGE_DAYS))
        a = age_of({"built_at": at(MAX_AGE_DAYS - 1)})
        cases.append((f"a {MAX_AGE_DAYS - 1}-day stamp is just under it",
                      a is not None and a <= MAX_AGE_DAYS))
        cases.append(("an unparseable date reads as missing",
                      age_of({"built_at": "not a date"}) is None))
        cases.append(("a stamp with no built_at reads as missing",
                      age_of({"entries": 341}) is None))
        cases.append(("an absent stamp reads as missing", age_of(None) is None))
        # A stamp dated in the FUTURE would otherwise sail through every gate:
        # it is not None, and a negative age is never > MAX_AGE_DAYS.
        a = age_of({"built_at": at(-5)})
        cases.append(("a future-dated stamp reads NEGATIVE, not 0 or None",
                      a is not None and a < 0))
        cases.append(("and a negative age does NOT satisfy the staleness gate",
                      a is not None and not (a > MAX_AGE_DAYS)))
        f.write_text("{not json", encoding="utf-8")
        age, why = stamp_age_days(0, sp=f)
        cases.append(("malformed JSON reads as missing", age is None))
        cases.append(("and carries the REASON, so the failure names the real fault",
                      isinstance(why, dict) and "_unreadable" in why))
        cases.append(("while a truly absent stamp carries no reason",
                      (f.unlink() or stamp_age_days(0, sp=f)) == (None, None)))
    cases.append(("the threshold leaves room for one bad night", MAX_AGE_DAYS >= 2))
    # the season-label helper the rollover guard depends on
    cases.append(("next label of 2026-27 is 2027-28",
                  _expected_next_label("2026-27") == "2027-28"))
    cases.append(("next label of 2029-30 is 2030-31",
                  _expected_next_label("2029-30") == "2030-31"))
    ok = True
    for label, passed in cases:
        print(("  PASS  " if passed else "  FAIL  ") + label)
        ok = ok and passed
    print("\n[PASS] selftest: every age check fires" if ok else "\n[FAIL] selftest")
    return 0 if ok else 1


def main(season: str) -> int:
    espn_year = utils.season_to_espn_year(season)
    # _rev_path, not _dc_path: this must resolve the same file the READER
    # does, or the check validates a copy production never sees.
    path = artifact_paths(espn_year)[0]
    feed = utils.fetch_next_year_contracts(espn_year, cache_v=7)

    age, stamp = stamp_age_days(espn_year)
    live = (season == utils.SEASONS[0])
    print(f"{season}  ->  {path.name}")
    print(f"  file on disk      {'yes' if path.exists() else 'NO'}")
    print(f"  feed entries      {len(feed)}")
    if age is None:
        why = (stamp or {}).get("_unreadable")
        print(f"  last rebuilt      NO STAMP ({stamp_path_for(espn_year).name} "
              + (f"unreadable: {why}" if why else "absent") + ")")
    else:
        print(f"  last rebuilt      {stamp['built_at']}  ({age:.1f} days ago, "
              f"describing {stamp.get('describes', '?')})")

    board_path = utils._rev_path(
        f"raw_{season.replace('-', '_')}_{utils.FORMULA_VERSION}.parquet")
    if not board_path.exists():
        print(f"  no board cache at {board_path.name}")
        return 1
    board = pd.read_parquet(board_path)
    vals = [fmt_next_contract(p, feed) for p in board["Player"]]
    filled = sum(1 for v in vals if v != "—")
    cover = filled / len(board) if len(board) else 0.0

    from collections import Counter
    kinds = Counter("dash" if v == "—" else "RFA" if v == "RFA"
                    else "TO" if v.endswith(" TO") else "PO" if v.endswith(" PO")
                    else "money" for v in vals)
    print(f"  board rows        {len(board)}")
    print(f"  with a value      {filled}  ({cover:.0%})")
    print(f"  breakdown         {dict(kinds)}")

    if not path.exists():
        print(f"\nFAIL: {path.name} has never been built. The live derivation cannot "
              f"build it mid-season (ESPN has no {espn_year + 1} salaries yet), so the "
              f"column will stay empty until you run scripts/build_next_contracts.py.")
        return 1
    if len(feed) < MIN_FEED:
        print(f"\nFAIL: only {len(feed)} entries, under the {MIN_FEED} floor that "
              f"fetch_next_year_contracts itself treats as a failed scrape.")
        return 1
    if cover < MIN_COVER:
        print(f"\nFAIL: only {cover:.0%} of the board has a Next $ value, under the "
              f"{MIN_COVER:.0%} floor. A column of dashes reads as 'everyone is a "
              f"free agent', which is why nobody notices it.")
        return 1
    # The stamp gates only the LIVE season. The builder writes one solely for
    # utils.SEASONS[0], so demanding it for a past season would fail
    # `refresh_live_season.py 2025-26`, which is the documented way to rebuild
    # one, on an artifact that is correct and frozen by definition.
    if not live:
        print(f"\nPASS: {filled} of {len(board)} board players carry a Next $ value. "
              f"({season} is not the live season, so no rebuild stamp is expected.)")
        return 0
    if age is None:
        why = (stamp or {}).get("_unreadable")
        print(f"\nFAIL: {stamp_path_for(espn_year).name} is "
              + (f"unreadable ({why}). Fix or delete it; " if why
                 else "absent, so there is no way to tell a current artifact from one "
                      "nobody has rebuilt since the season flipped. ")
              + "Run scripts/build_next_contracts.py --write (step 3b of the nightly "
                "writes it).")
        return 1
    if age < 0:
        print(f"\nFAIL: the stamp is dated {abs(age):.1f} days in the FUTURE "
              f"({stamp.get('built_at')}). A clock that was ahead when the builder ran "
              f"would otherwise keep this gate green forever on an artifact nobody is "
              f"rebuilding. Re-run scripts/build_next_contracts.py --write.")
        return 1
    if age > MAX_AGE_DAYS:
        print(f"\nFAIL: last rebuilt {age:.0f} days ago, over the {MAX_AGE_DAYS}-day floor. "
              f"The build step is non-fatal, so one bad night cannot reach this; this far "
              f"means step 3b has stopped running and the Next $ column is quoting money "
              f"from before whatever has happened since.")
        return 1
    if stamp.get("describes") and stamp["describes"] != _expected_next_label(season):
        print(f"\nFAIL: the artifact describes {stamp['describes']} but {season} needs "
              f"{_expected_next_label(season)}. A rollover left the builder's season "
              f"literals behind.")
        return 1
    if stamp.get("artifact") and stamp["artifact"] != path.name:
        print(f"\nFAIL: the stamp vouches for {stamp['artifact']} but the reader resolves "
              f"{path.name}. A DATA_REV bump shipped without the artifact under the new "
              f"name, so production has nothing at that name and falls back to the floor.")
        return 1
    print(f"\nPASS: {filled} of {len(board)} board players carry a Next $ value, "
          f"rebuilt {age:.1f} days ago ({path.name}).")
    return 0


def _expected_next_label(season: str) -> str:
    start = int(season.split("-")[0])
    return f"{start + 1}-{str(start + 2)[-2:]}"


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    sys.exit(main(args[0] if args else utils.SEASONS[0]))
