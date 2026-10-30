"""Do the PAGES still work now that SEASONS[0] has moved?

    /opt/anaconda3/bin/python3 -u scripts/check_season_surfaces.py

Read-only. Prints PASS/FAIL per check, exits non-zero on any failure.

WHY THIS EXISTS, and it is worth being blunt about it.

scripts/preflight_season_rollover.py asks "is it safe to flip?" and it did that
job. It checked the salary cap for the new next-season, the stat caches, the
derived FA class year, the accuracy tracker. Then the flip happened and it
rolled forward to the NEXT flip, so every leftover from THIS one stopped being
reported on the day it became real. A gate that disarms itself by succeeding is
not a gate.

And it only ever covered DATA. Nothing checked the SURFACES, where the same
season assumption is baked in three other ways:

  1. A minutes floor sized for a complete season. DEFAULT_MIN_THRESHOLD = 500
     is a sensible noise filter in April and an empty page in October. On
     2026-10-22 the whole league topped out at 41 total minutes, so Team
     Analysis filtered 297 players to zero, then reported "salary coverage for
     2026-27 is 0%" about the empty frame and sent people elsewhere. Coverage
     was 100%. Rankings had the identical bug and was fixed a day earlier,
     because someone looked at it.

  2. Season strings written as literals. "2025-26" meant "the season being
     scored" in one place and "the last complete season" in another. After the
     flip half of them were right and half were quietly a year stale, and the
     page still rendered: Rosters presented last season's score UNTAGGED as
     though it were this season's.

  3. Hand-maintained files named by year. score_season_override.csv pins
     Keegan Murray to 2024-25, a season Rosters stopped loading the moment
     2026-27 became SEASONS[0], so the override would have gone quietly dead.

None of these raise. That is the whole problem, and it is why check 3 below
builds the real pool at each page's real default and asserts somebody survives.
"""
from __future__ import annotations

import ast
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import (BLEND_MIN_MINUTES, DEFAULT_MIN_THRESHOLD, SEASONS,  # noqa: E402
                   live_min_threshold, season_is_live)

CUR, PRIOR = SEASONS[0], SEASONS[1]
PAGES = sorted((ROOT / "pages").glob("*.py")) + [ROOT / "app.py"]

results: list[tuple[str, str, str]] = []


def record(state: str, name: str, detail: str) -> None:
    results.append((state, name, detail))


# ── 1. no page sizes a live-season control with the complete-season floor ────
# A page may still IMPORT DEFAULT_MIN_THRESHOLD (Legacy pools finished seasons
# with it, which is correct). What it must not do is hand that constant to a
# slider default while the season is live without going through
# live_min_threshold, which is exactly how Team Analysis emptied itself.
offenders = []
for page in PAGES:
    try:
        tree = ast.parse(page.read_text())
    except SyntaxError as e:
        record("FAIL", page.name, f"unparseable: {e}")
        continue
    uses_helper = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                      and n.func.id == "live_min_threshold" for n in ast.walk(tree))
    # a slider whose value= is DEFAULT_MIN_THRESHOLD, or a name assigned from it
    const_defaults = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            tgt = n.targets[0]
            if isinstance(tgt, ast.Name) and any(
                    isinstance(x, ast.Name) and x.id == "DEFAULT_MIN_THRESHOLD"
                    for x in ast.walk(n.value)):
                const_defaults.add(tgt.id)
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "slider"):
            continue
        for kw in n.keywords:
            if kw.arg != "value":
                continue
            names = {x.id for x in ast.walk(kw.value) if isinstance(x, ast.Name)}
            if (names & const_defaults or "DEFAULT_MIN_THRESHOLD" in names) \
                    and not uses_helper:
                offenders.append(f"{page.name}:{n.lineno}")
record("FAIL" if offenders else "PASS", "minutes floor scales",
       ("sized for a COMPLETE season, so the page empties early in a live one: "
        + ", ".join(offenders)) if offenders else
       "every minutes slider goes through utils.live_min_threshold")

# ── 2. no season literals standing in for the CURRENT season ────────────────
# Walk the AST, not the text, so the prose above (which quotes "2025-26"
# repeatedly) is not read as the bug it describes. The preflight learned that
# the hard way by grepping raw file text and flagging its own documentation.
#
# Not every season literal is a defect. An ORDERING comparison is a fixed
# historical boundary and never moves: Rankings asks `s < "2009-10"` because
# D-LEBRON does not exist before then, and that stays true forever. What does
# move is a literal passed somewhere as THE season: build_raw("2025-26"),
# season == "2025-26", {"season": "2026-27"}. Those mean one thing before a
# flip and another after, which is why they survive it silently.
SEASON_RE = re.compile(r"^(19|20)\d\d-\d\d$")
literals = []
for page in PAGES:
    try:
        tree = ast.parse(page.read_text())
    except SyntaxError:
        continue
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef))
                  and n.body and isinstance(n.body[0], ast.Expr)
                  and isinstance(n.body[0].value, ast.Constant)
                  and isinstance(n.body[0].value.value, str)}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if not SEASON_RE.match(node.value) or id(node) in docstrings:
            continue
        par = parent.get(id(node))
        # an ordering comparison is an epoch boundary, not a current season
        if isinstance(par, ast.Compare) and par.ops and all(
                isinstance(o, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for o in par.ops):
            continue
        literals.append(f"{page.name}:{node.lineno} {node.value!r}")
record("FAIL" if literals else "PASS", "no hardcoded season strings",
       ("a literal means one thing before a flip and another after: "
        + "; ".join(literals[:5])
        + (f" (+{len(literals)-5} more)" if len(literals) > 5 else ""))
       if literals else "every page derives its seasons from SEASONS")

# ── 3. THE BEHAVIOURAL ONE: does the page actually show anybody? ─────────────
# Everything above is structural. This builds the real pool at the real default
# and asserts the page is not empty, which is the thing a user sees.
try:
    from utils import build_ranked_projected
    df = build_ranked_projected(CUR)
    gp = int(df["GP"].max() or 0) if len(df) else 0
    default, _mx, _st = live_min_threshold(CUR, gp)
    survivors = int((df["total_min"] >= default).sum()) if len(df) else 0
    teams = int(df[df["total_min"] >= default]["Team"].nunique()) if survivors else 0
    if not len(df):
        record("FAIL", f"{CUR} pool", "empty, so every surface is empty")
    elif survivors == 0:
        record("FAIL", f"{CUR} survives its own default",
               f"{len(df)} players in the pool, 0 clear the default threshold of "
               f"{default} ({gp} game(s) played, league-high {int(df['total_min'].max())} "
               f"total minutes). Every page using this default renders empty.")
    else:
        record("PASS", f"{CUR} survives its own default",
               f"{survivors}/{len(df)} players and {teams} teams clear {default} "
               f"after {gp} game(s)")

    # and the salary join, which is what the empty frame got blamed for
    if survivors:
        cov = float((df[df["total_min"] >= default]["salary"] > 0).mean())
        record("PASS" if cov >= 0.5 else "FAIL", f"{CUR} salary coverage",
               f"{cov*100:.0f}% of shown players have a salary")
except Exception as e:                                               # noqa: BLE001
    record("FAIL", f"{CUR} pool", f"could not build: {e}")

# ── 4. the hand-maintained files the NEW season's year points at ─────────────
fa_year = int(CUR[:4]) + 1
missing = [n for n in (f"real_signings_{fa_year}.csv", f"option_decisions_{fa_year}.csv")
           if not (ROOT / "data" / n).exists()]
record("FAIL" if missing else "PASS", f"{fa_year} FA class files",
       (f"missing {missing}, so a board titled {fa_year} annotates from a closed class")
       if missing else f"present, so the {fa_year} board annotates from its own class")

# ── 5. files referenced by a DERIVED name must actually exist ────────────────
derived = {
    f"data/team_records_{PRIOR.replace('-', '_')}.csv": "Rosters' last-season leaderboard",
    f"cache/league_stats_{PRIOR.replace('-', '_')}.parquet": "Rosters/Trade Machine box line",
}
absent = [f"{p} ({why})" for p, why in derived.items() if not (ROOT / p).exists()]
record("WARN" if absent else "PASS", "derived filenames resolve",
       ("these degrade quietly rather than raising: " + "; ".join(absent))
       if absent else f"every {PRIOR}-derived file is on disk")

# ── 6. a DATA_REV bump must carry EVERY revisioned file with it ─────────────
# _rev_path falls back to the UNSUFFIXED filename when the current revision is
# missing. On Render that fallback is the ORIGINAL seeded copy, so a bump made
# to ship one corrected file silently reverts every other file that resolves
# through the same token. On 2026-10-23 bumping r4 -> r5 for the provisional
# box scores would have taken Legacy back to the mojibake names repaired the
# day before, with nothing to show for it.
try:
    from utils import DATA_REV, all_seasons_path, _raw_disk_path
    stale = []
    for label, path in (("all_seasons(0)", all_seasons_path(0)),
                        ("all_seasons(500)", all_seasons_path(500)),
                        (f"raw {CUR}", _raw_disk_path(CUR))):
        if not path.exists():
            stale.append(f"{label}: {path.name} does not exist")
        elif f"_{DATA_REV}." not in path.name:
            stale.append(f"{label}: resolved to {path.name}, which is NOT at {DATA_REV} "
                         f"(fell back, so a shipped fix is being ignored)")
    record("FAIL" if stale else "PASS", f"DATA_REV {DATA_REV} files present",
           "; ".join(stale) if stale else
           "every revisioned path resolves to a file at the current revision")
except Exception as e:                                               # noqa: BLE001
    record("FAIL", "DATA_REV files", str(e))

# ── 7. no team present in the SOURCE may be missing from the pool ───────────
# A partial scrape does not raise. _build_raw_live does
# dropna(subset=["salary"]), so any player the ESPN salary pages happened to
# miss is simply deleted, and if a run misses a cluster it deletes a whole
# roster. On 2026-10-23 a shipped build was missing all ten Washington players
# including Anthony Davis, who had been sixth on the board that morning; the
# board read 313 players instead of 319 and looked entirely normal.
#
# fetch_salaries now refuses a materially short scrape in favour of its last
# complete one. This is the check that would have caught the shipped file
# anyway, because it compares two things that must agree and does not care why.
try:
    from utils import build_raw, fetch_bref_player_stats, resolve_team_code
    pool = build_raw(CUR)
    src = fetch_bref_player_stats(CUR)
    if len(pool) and len(src):
        pool_teams = set(pool["Team"].dropna().unique())
        # The source here is BBRef, which spells three teams CHO, BRK and PHO
        # where the pool says CHA, BKN and PHX. Comparing the raw strings
        # reported those three as "dropped" on every single run, beside the one
        # that genuinely was, so the real finding arrived wearing three false
        # ones. Resolve into the pool's dialect first.
        src_teams = {resolve_team_code(t, pool_teams)
                     for t in src["TEAM_ABBREVIATION"].dropna().unique()}
        gone = sorted(src_teams - pool_teams)
        record("FAIL" if gone else "PASS", "no team dropped from the pool",
               (f"present in the stat source but ABSENT from the pool: {gone}. "
                f"That is a partial salary scrape deleting rosters, not a real "
                f"roster change.") if gone else
               f"all {len(src_teams)} source teams survive into the pool "
               f"({len(pool_teams)} total with provisional rows)")
    else:
        record("WARN", "no team dropped from the pool", "pool or source empty")
except Exception as e:                                               # noqa: BLE001
    record("FAIL", "team coverage", str(e))

# ── 8. an override may not point at a season nothing loads ───────────────────
try:
    import csv as _csv
    ovr = ROOT / "data" / "score_season_override.csv"
    pinned = set()
    if ovr.exists():
        with open(ovr) as fh:
            for r in _csv.DictReader(
                    l for l in fh if l.strip() and not l.lstrip().startswith("#")):
                if (r.get("season") or "").strip():
                    pinned.add(r["season"].strip())
    unknown = sorted(s for s in pinned if s not in SEASONS)
    record("FAIL" if unknown else "PASS", "season overrides resolve",
           (f"pinned to seasons not in SEASONS at all: {unknown}") if unknown
           else f"all {len(pinned)} pinned season(s) are loadable")
except Exception as e:                                               # noqa: BLE001
    record("FAIL", "season overrides", str(e))


# ── 9. the search box routes by the pool the hub actually resolves from ──────
# The homepage decides, for a picked or deep-linked name, whether to open the
# hub panel or send the visitor to Compare Players. That decision is a
# MEMBERSHIP TEST, and it has to be the hub's own pool.
#
# It was not. The test named build_ranked_projected (the CURRENT-season frame)
# while the hub resolves from build_ranked_blended. The two agreed until
# BLEND_MIN_MINUTES went to 0; after that the hub held 572 players and the test
# knew 396, so all 176 players still carrying a prior season were bounced off
# their own hub -- a board row that silently led somewhere else. The page threw
# nothing, which is why it survived a season-wide sweep.
#
# Checked at SOURCE level because the two sites are 700 lines apart and nothing
# at runtime forces them to agree.
try:
    _app = ast.parse((ROOT / "app.py").read_text())
    _route_builder = _pool_builder = None
    for _n in ast.walk(_app):
        # _cur_norms = {... for _p in BUILDER(...)["Player"]}
        if (isinstance(_n, ast.Assign) and len(_n.targets) == 1
                and isinstance(_n.targets[0], ast.Name) and _n.targets[0].id == "_cur_norms"):
            for _c in ast.walk(_n.value):
                if isinstance(_c, ast.Call) and isinstance(_c.func, ast.Name) \
                        and _c.func.id.startswith("build_ranked"):
                    _route_builder = _c.func.id
        # _pool = BUILDER(...)
        if (isinstance(_n, ast.Assign) and len(_n.targets) == 1
                and isinstance(_n.targets[0], ast.Name) and _n.targets[0].id == "_pool"
                and isinstance(_n.value, ast.Call) and isinstance(_n.value.func, ast.Name)
                and _n.value.func.id.startswith("build_ranked")):
            _pool_builder = _pool_builder or _n.value.func.id
    if not _route_builder or not _pool_builder:
        record("WARN", "search routes by the hub's pool",
               f"could not locate both sites in app.py (route={_route_builder}, "
               f"hub={_pool_builder}); the check has drifted from the code")
    elif _route_builder != _pool_builder:
        record("FAIL", "search routes by the hub's pool",
               f"the search box routes on {_route_builder}() but the hub resolves from "
               f"{_pool_builder}(), so every player in one and not the other is sent "
               f"away from a panel that would have rendered")
    else:
        record("PASS", "search routes by the hub's pool",
               f"both use {_pool_builder}()")
except Exception as e:                                               # noqa: BLE001
    record("FAIL", "search routes by the hub's pool", str(e))


# ---------------------------------------------------------------- 5. one floor
# Check 1 above catches a minutes floor that EMPTIES a page, because an empty
# page is loud. Rosters had the same literal and was silent: its floor did not
# drop anyone, it reached back a season instead, so the page stayed full and
# simply showed 2025-26 for every player on every roster while the hub showed
# 2026-27 for the same man. Luka read 44.0 against the hub's 50.9. Nothing
# raised, nothing emptied, and the surviving check passed throughout.
#
# The floor is not a page's decision to make. utils.BLEND_MIN_MINUTES is the
# one knob (Barrett, 2026-10-29), and a page that writes its own number has
# opted out of it in a way only a reader can notice. So: no surface may gate
# total_min on a literal.
try:
    import re as _re
    _FLOOR_NAME = _re.compile(r"MIN_MINUTES|MIN_THRESHOLD|MINUTES_FLOOR", _re.I)
    _offenders, _scanned = [], 0
    for _f in PAGES:
        try:
            _tree = ast.parse(_f.read_text(encoding="utf-8"))
        except Exception:                                            # noqa: BLE001
            continue
        _scanned += 1
        for _n in ast.walk(_tree):
            # NAME = <number>, where NAME reads like a minutes floor
            if (isinstance(_n, ast.Assign) and len(_n.targets) == 1
                    and isinstance(_n.targets[0], ast.Name)
                    and _FLOOR_NAME.search(_n.targets[0].id)
                    and isinstance(_n.value, ast.Constant)
                    and isinstance(_n.value.value, (int, float))):
                _offenders.append(f"{_f.name}:{_n.lineno} {_n.targets[0].id} = {_n.value.value}")
            # ... total_min >= <number>, the gate itself, however it is spelled
            if isinstance(_n, ast.Compare) and len(_n.comparators) == 1:
                _src = ast.dump(_n.left)
                if "total_min" in _src and isinstance(_n.comparators[0], ast.Constant) \
                        and isinstance(_n.comparators[0].value, (int, float)) \
                        and _n.comparators[0].value != 0:
                    _offenders.append(f"{_f.name}:{_n.lineno} total_min gated on "
                                      f"{_n.comparators[0].value}")
    if _offenders:
        record("FAIL", "no surface writes its own minutes floor",
               "; ".join(_offenders) + " -- use utils.BLEND_MIN_MINUTES, or the page "
               "shows a different season from the rest of the site and says nothing")
    else:
        record("PASS", "no surface writes its own minutes floor",
               f"{_scanned} surface(s) defer to BLEND_MIN_MINUTES = {BLEND_MIN_MINUTES}")
except Exception as e:                                               # noqa: BLE001
    record("FAIL", "no surface writes its own minutes floor", str(e))


print()
print(f"  SEASON SURFACES    scored: {CUR}   last complete: {PRIOR}"
      f"   live: {season_is_live(CUR)}")
print()
width = max(len(n) for _, n, _ in results)
for state, name, detail in results:
    print(f"  [{state:4}] {name:{width}}  {detail}")

fails = [r for r in results if r[0] == "FAIL"]
warns = [r for r in results if r[0] == "WARN"]
print()
print(f"  {'FAIL: ' + str(len(fails)) + ' surface(s) broken by the rollover' if fails else 'PASS: every surface agrees with SEASONS[0]' + (f', {len(warns)} warning(s)' if warns else '')}")
print()
sys.exit(1 if fails else 0)
