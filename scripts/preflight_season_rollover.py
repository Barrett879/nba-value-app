"""Preflight for the season rollover: is the site ready to make 2026-27 the
season it SCORES, instead of 2025-26?

    /opt/anaconda3/bin/python3 -u scripts/preflight_season_rollover.py

Read-only. Prints PASS / FAIL / WAIT per check and exits non-zero if anything
that would ship a wrong number is not ready.

WHY THIS EXISTS. "Rolling over" sounds like prepending one string to
utils.SEASONS, and it is not. Everything downstream is derived from SEASONS[0]:
the scored season, _NEXT_SEASON, the free-agent class year, which caches count
as immutable, and which salary cap prices a max contract. On 2026-10-20 the
season being PLAYED and the season being SCORED diverged for the first time, and
each thing keyed to the wrong one of those is a silent wrong number rather than
a crash. So the rollover is a checklist, and this is the checklist.

WAIT is not a failure. The board should keep scoring the last COMPLETE season
until the new one has enough games for the Barrett Score to mean anything. Per-
game rates and the availability multiplier are both unstable over a handful of
games, so flipping on opening night would publish noise with a straight face.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import SALARY_CAP_M, SEASONS, season_is_live  # noqa: E402

# Games a player needs before his rate stats stop being noise. A quarter of the
# season is the usual convention and is roughly when the board stabilises.
MIN_GAMES = 20

CURRENT = SEASONS[0]                                   # the season we score today
NEW = f"{int(CURRENT[:4]) + 1}-{(int(CURRENT[:4]) + 2) % 100:02d}"
NEW_NEXT = f"{int(NEW[:4]) + 1}-{(int(NEW[:4]) + 2) % 100:02d}"
NEW_FA_YEAR = int(NEW[:4]) + 1

results: list[tuple[str, str, str]] = []


def record(state: str, name: str, detail: str) -> None:
    results.append((state, name, detail))


def rows(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(
            l for l in fh if l.strip() and not l.lstrip().startswith("#")))


# ── 1. the season labels actually disagree, which is the whole premise ────────
if season_is_live(CURRENT):
    record("WAIT", "scored season still being played",
           f"{CURRENT} is still live, so there is nothing to roll over yet")
else:
    record("PASS", "scored season is complete",
           f"{CURRENT} is over, so its stats are final and its caches immutable")

if season_is_live(NEW):
    record("PASS", "new season is under way", f"{NEW} is being played")
else:
    record("WAIT", "new season has not started", f"{NEW} is not live yet")

# ── 2. the cap for the season AFTER the flip ──────────────────────────────────
# _amax_floor / _amax_cap read SALARY_CAP_M.get(_NEXT_SEASON, 165.0). A missing
# key does not raise, it prices the max tier at a stale cap, so this is exactly
# the silent-wrong-number class.
if NEW_NEXT in SALARY_CAP_M:
    record("PASS", f"salary cap for {NEW_NEXT}",
           f"${SALARY_CAP_M[NEW_NEXT]:.1f}M present, so the max tier is priced correctly")
else:
    record("FAIL", f"salary cap for {NEW_NEXT}",
           f"MISSING: _amax_floor/_amax_cap would silently fall back to $165.0M. "
           f"Add a projection to SALARY_CAP_M and cite the league memo.")

# ── 3. enough games played in the new season ─────────────────────────────────
stat_caches = sorted(ROOT.glob(f"cache/league_stats_{NEW}*")) + \
              sorted(ROOT.glob(f"cache/raw_{NEW.replace('-', '_')}*"))
if not stat_caches:
    record("WAIT", f"{NEW} stat caches",
           f"none on disk. Expected, since fetch_league_stats is only called for "
           f"seasons in SEASONS and {NEW} is not in it yet.")
else:
    best = 0
    for c in stat_caches:
        try:
            import pandas as pd
            df = pd.read_parquet(c)
            for col in ("GP", "gp", "games_played"):
                if col in df.columns and len(df):
                    best = max(best, int(df[col].max()))
                    break
        except Exception:
            continue
    if best >= MIN_GAMES:
        record("PASS", f"{NEW} games played",
               f"max GP {best} >= {MIN_GAMES}, so rate stats are stable enough to score")
    else:
        record("WAIT", f"{NEW} games played",
               f"max GP {best} < {MIN_GAMES}: flipping now would publish early-season noise")

# ── 4. no hardcoded free-agent class year left in the UI ─────────────────────
# Scan STRING TOKENS only, never raw file text. The first version of this check
# regex-matched the whole file and flagged the comment that documents the fix,
# because that comment quotes the old strings verbatim. A check that reads
# prose about the bug as the bug is worse than no check.
stale_year = []
for rel in ("app.py", "pages/Free_Agent_Class.py"):
    import io
    import tokenize
    with open(ROOT / rel, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type != tokenize.STRING:
                continue
            lit = tok.string
            if lit.lstrip("rbuf").startswith(('"""', "'''")):
                continue                              # docstrings are prose too
            if re.search(r"\b20\d\d\b", lit) and re.search(
                    r"FA watch|hits the market|[Ff]ree [Aa]gent", lit):
                stale_year.append(f"{rel}:{tok.start[0]} {lit[:52]}")
if stale_year:
    record("FAIL", "free-agent class year is hardcoded",
           "would advertise a closed class after the flip: " + "; ".join(sorted(set(stale_year))[:3]))
else:
    record("PASS", "free-agent class year is derived",
           f"advances to {NEW_FA_YEAR} on its own when SEASONS[0] moves")

# ── 5. the hand-maintained books for the new free-agent class ────────────────
# data/real_signings_2026.csv and option_decisions_2026.csv are the 2026 class.
# After the flip the class is NEW_FA_YEAR, and these files are read by NAME, so
# they do not follow. Until the new pair exists the FA page keeps scoring a
# closed offseason.
needed = [f"real_signings_{NEW_FA_YEAR}.csv", f"option_decisions_{NEW_FA_YEAR}.csv"]
missing = [n for n in needed if not (ROOT / "data" / n).exists()]
if missing:
    record("WAIT", f"{NEW_FA_YEAR} free-agent class files",
           "not created yet: " + ", ".join(missing) +
           ". The loaders name the 2026 files literally, so they need creating and wiring.")
else:
    record("PASS", f"{NEW_FA_YEAR} free-agent class files", "present")

# ── 6. the 2026-27 books keep their meaning ──────────────────────────────────
# dead_money_2026_27 and friends are the CURRENT season's books after the flip
# rather than next season's. The files do not move; only what they mean does.
books = ["dead_money_2026_27.csv", "hard_caps_2026_27.csv",
         "trade_exceptions_2026_27.csv", "contract_years_2026_27.csv",
         "salary_supplement_2026_27.csv", "cba_config_2026_27.json"]
absent = [b for b in books if not (ROOT / "data" / b).exists()]
if absent:
    record("FAIL", "2026-27 books", "missing: " + ", ".join(absent))
else:
    record("PASS", "2026-27 books",
           f"all {len(books)} present; after the flip they are the CURRENT season's "
           f"books, and {NEW_NEXT} equivalents become the forward-looking set")

# ── 7. the standing per-round check, which has bitten before ─────────────────
try:
    tracker = json.loads((ROOT / "cache" / "accuracy_tracker_v1.json").read_text())
    n_tracker = len(tracker.get("signings", []))
    n_file = len(rows(ROOT / "data" / "real_signings_2026.csv"))
    if n_tracker == n_file:
        record("PASS", "accuracy tracker in sync", f"{n_tracker} signings == real_signings rows")
    else:
        record("FAIL", "accuracy tracker out of sync",
               f"tracker {n_tracker} vs real_signings {n_file}. It is the TOP-priority "
               f"salary source and the Signed flag, so a gap renders a signed player as "
               f"an unsigned free agent. Rebuild build_accuracy_tracker.py.")
except Exception as e:                                        # noqa: BLE001
    record("FAIL", "accuracy tracker unreadable", str(e))


# ── report ───────────────────────────────────────────────────────────────────
print()
print(f"  SEASON ROLLOVER PREFLIGHT    scored: {CURRENT}  ->  {NEW}")
print(f"  after the flip, next season becomes {NEW_NEXT} and the FA class {NEW_FA_YEAR}")
print()
width = max(len(n) for _, n, _ in results)
for state, name, detail in results:
    print(f"  [{state:4}] {name:{width}}  {detail}")

fails = [r for r in results if r[0] == "FAIL"]
waits = [r for r in results if r[0] == "WAIT"]
print()
if fails:
    print(f"  NOT READY: {len(fails)} check(s) would ship a wrong number. Fix those first.")
elif waits:
    print(f"  HOLD: nothing is broken, but {len(waits)} check(s) say it is too early.")
    print(f"  Keep scoring {CURRENT} until they clear. This is the correct state on opening night.")
else:
    print("  READY: prepend the new season to utils.SEASONS and redeploy.")
print()
sys.exit(1 if fails else 0)
