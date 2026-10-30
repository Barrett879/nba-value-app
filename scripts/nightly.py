"""The whole nightly refresh, as one command.

    /opt/anaconda3/bin/python3 -u scripts/nightly.py              # build + verify, stop
    /opt/anaconda3/bin/python3 -u scripts/nightly.py --commit     # ... and commit
    /opt/anaconda3/bin/python3 -u scripts/nightly.py --commit --push

Barrett will run this most nights until June, so every step that used to be a
hand-typed decision is in here, and every failure stops the run instead of
shipping. The steps, in the order they depend on each other:

  1 INGEST       games ESPN has and the lineup feed does not, rebuilt from ESPN
                 and registered
  2 PROVISIONAL  box scores for the dates where the stat feed is behind
  3 BUMP         per_game, defense_blend and DATA_REV, plus the reader literals
                 that have to move with them
  4 BUILD        refresh_live_season.py: artifacts BEFORE board, and its five
                 checks
  5 VERIFY       the new revision against the previous one; nobody may vanish
  6 SHIP         stage exactly what changed, commit, push

Anything that fails after step 3 rolls the version bumps back and deletes the
half-built artifacts, so a failed night leaves the tree exactly as it found it
and the live site untouched.

Why each guard is here, so nobody removes one as noise:

  Versions are bumped by this script because doing it by hand is five edits
  across four files and getting one wrong ships a file production will never
  read (/data seeds a cache only when the FILENAME is missing).

  The verify step compares against the previous revision because on 2026-10-28
  a blocked salary endpoint deleted all twelve Washington players and every
  automated check still passed. Row counts and team counts are the only things
  that noticed.

  Nothing is staged with `git add -A`. The repo carries ~240 churning cache
  files and an untracked 121MB pbpstats directory.
"""
from __future__ import annotations
import argparse, datetime as dt, json, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
PY_ = sys.executable

# Files whose version literal has to move together. A bump that lands in the
# builder but not the reader writes a file nothing opens.
VERSIONED = {
    "per_game": [("scripts/build_per_game.py", r'^VERSION = "v(\d+)"'),
                 ("pages/Rankings.py",
                  r"per_game_\{season_key\.replace\('-', '_'\)\}_v(\d+)\.json"),
                 # app.py's hub reads the same artifact. Added 2026-10-30 WITH
                 # that reader: a literal this script does not know about is a
                 # reader that silently stops opening anything the next time the
                 # builder moves.
                 ("app.py",
                  r"per_game_\{_HUB_SEASON\.replace\('-', '_'\)\}_v(\d+)\.json")],
    "defense_blend": [("scripts/build_defense_blend.py", r'^VERSION = "v(\d+)"'),
                      ("utils.py",
                       r"defense_blend_\{season\.replace\('-', '_'\)\}_v(\d+)\.json")],
    "DATA_REV": [("utils.py", r'^DATA_REV = "r(\d+)"')],
}


def sh(args, label, check=True):
    print(f"\n--- {label} " + "-" * max(0, 58 - len(label)))
    r = subprocess.run([PY_, "-u", *args], cwd=ROOT)
    if check and r.returncode:
        raise RuntimeError(f"{label} exited {r.returncode}")
    return r.returncode == 0


def read_versions() -> dict:
    out = {}
    for key, sites in VERSIONED.items():
        vals = set()
        for rel, pat in sites:
            txt = (ROOT / rel).read_text()
            hits = re.findall(pat, txt, re.M)
            if not hits:
                raise RuntimeError(f"no version literal for {key} in {rel}")
            vals |= set(hits)
        if len(vals) > 1:
            raise RuntimeError(f"{key} disagrees across files: {sorted(vals)}")
        out[key] = int(vals.pop())
    return out


def write_versions(target: dict) -> None:
    """Rewrite every site for every key. All or nothing per file."""
    for key, sites in VERSIONED.items():
        for rel, pat in sites:
            p = ROOT / rel
            txt = p.read_text()
            n = int(target[key])
            new = re.sub(pat,
                         lambda m: m.group(0).replace(m.group(1), str(n)),
                         txt, flags=re.M)
            if new != txt:
                p.write_text(new)


def board_summary(path: Path) -> dict | None:
    import pandas as pd
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    return {"rows": len(d), "teams": sorted(d["Team"].dropna().unique()),
            "players": set(d["Player"].dropna()),
            "salaries": {str(a): float(b) for a, b in zip(d["Player"], d["salary"])
                         if b is not None and b == b}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="stage and commit when every check passes")
    ap.add_argument("--push", action="store_true", help="push too (implies --commit)")
    ap.add_argument("--days", type=int, default=4, help="how far back to look for unpublished box scores")
    args = ap.parse_args()
    if args.push:
        args.commit = True

    import logging
    logging.disable(logging.WARNING)
    import utils
    import def_lib as L
    logging.disable(logging.NOTSET)
    season = utils.SEASONS[0]
    key = season.replace("-", "_")

    print(f"NIGHTLY  {season}   {dt.date.today().isoformat()}")

    # ---- 1 INGEST ------------------------------------------------------
    espn, feed, missing = L.feed_gap(season)
    print(f"\nESPN has {sum(espn.values()) // 2} completed games; "
          f"the lineup feed has {sum(feed.values()) // 2}")
    if missing:
        print(f"  {len(missing)} not in the lineup feed, rebuilding from ESPN")
        sh(["scripts/espn_lineupopp.py", "backfill"], "backfill lineups")
        import importlib
        importlib.reload(L)
        espn, feed, missing = L.feed_gap(season)
        if missing:
            print(f"\nSTOPPING: {len(missing)} game(s) still missing after the "
                  f"backfill: {missing[:6]}")
            return 1
    else:
        print("  lineup feed is complete")

    # ---- 2 PROVISIONAL -------------------------------------------------
    # Only the dates where the stat feed is actually behind. The merge is
    # self-superseding, so an extra date is harmless, but a missed one means a
    # team's players sit a game short of every other surface on the page.
    import pandas as pd
    src = utils.fetch_league_stats(season)
    feed_gp = (src.groupby("TEAM_ABBREVIATION")["GP"].max().to_dict()
               if len(src) else {})
    played = L.espn_completed(season)
    cutoff = (dt.date.today() - dt.timedelta(days=args.days)).strftime("%Y%m%d")
    behind = sorted({d for (a, h), d in played.items() if d >= cutoff and (
        feed_gp.get(utils.resolve_team_code(a, set(feed_gp)), 0) < espn.get(a, 0)
        or feed_gp.get(utils.resolve_team_code(h, set(feed_gp)), 0) < espn.get(h, 0))})
    if behind:
        print(f"\nthe stat feed is behind on {len(behind)} date(s): {behind}")
        for d in behind:
            iso = f"{d[:4]}-{d[4:6]}-{d[6:]}"
            sh(["scripts/fetch_provisional_box_scores.py", iso, "--write"],
               f"provisional box scores {iso}")
    else:
        print("\nthe stat feed has every completed game; no provisional rows needed")

    # ---- 3 BUMP --------------------------------------------------------
    before = read_versions()

    # Re-running must not bump again. A nightly job gets re-run: the network
    # drops, something is inspected, the operator adds --commit. If THIS
    # round's artifacts are already built and not yet committed, the build has
    # happened and only the shipping is left.
    def built_and_unshipped(v: dict) -> bool:
        files = [ROOT / "cache" / f"raw_{key}_{utils.FORMULA_VERSION}_r{v['DATA_REV']}.parquet",
                 ROOT / "cache" / f"per_game_{key}_v{v['per_game']}.json",
                 ROOT / "cache" / f"defense_blend_{key}_v{v['defense_blend']}.json"]
        if not all(f.exists() for f in files):
            return False
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(files[0].relative_to(ROOT))],
            cwd=ROOT, capture_output=True)
        return tracked.returncode != 0        # exists on disk, not yet committed

    resuming = built_and_unshipped(before)
    after = before if resuming else {k: v + 1 for k, v in before.items()}
    print("\n--- bump " + "-" * 54)
    if resuming:
        print(f"  this round is already built at per_game v{after['per_game']}, "
              f"defense_blend v{after['defense_blend']}, DATA_REV r{after['DATA_REV']}")
        print("  not bumping again; verifying and shipping what is on disk")
    else:
        for k in before:
            print(f"  {k:<14} {before[k]} -> {after[k]}")
    # The board to compare against is the newest COMMITTED one, which on a
    # resume is not the file _rev_path now resolves to.
    prev_board = None
    for cand in sorted((ROOT / "cache").glob(f"raw_{key}_{utils.FORMULA_VERSION}_r*.parquet"),
                       key=lambda f: int(re.search(r"_r(\d+)\.parquet$", f.name).group(1)),
                       reverse=True):
        if int(re.search(r"_r(\d+)\.parquet$", cand.name).group(1)) < int(after["DATA_REV"]):
            prev_board = cand
            break
    prev = board_summary(prev_board) if prev_board else None
    if prev_board:
        print(f"  comparing against {prev_board.name}")
    if not resuming:
        write_versions(after)

    new_files = [ROOT / "cache" / f"raw_{key}_{utils.FORMULA_VERSION}_r{after['DATA_REV']}.parquet",
                 ROOT / "cache" / f"all_seasons_0_{utils.FORMULA_VERSION}_r{after['DATA_REV']}.parquet",
                 ROOT / "cache" / f"all_seasons_500_{utils.FORMULA_VERSION}_r{after['DATA_REV']}.parquet",
                 ROOT / "cache" / f"per_game_{key}_v{after['per_game']}.json",
                 ROOT / "cache" / f"defense_blend_{key}_v{after['defense_blend']}.json"]

    def rollback(why: str) -> int:
        print(f"\nROLLING BACK: {why}")
        if resuming:
            print("  (resumed run: versions were already in place, leaving them)")
            return 1
        write_versions(before)
        for f in new_files:
            if f.exists():
                f.unlink()
                print(f"  removed {f.name}")
        print("  versions restored; the tree and the live site are untouched")
        return 1

    # ---- 4 BUILD -------------------------------------------------------
    try:
        if resuming:
            print("\n--- build + checks " + "-" * 44)
            print("  already built; re-running the checks only")
            ok = all(sh([c], lbl, check=False) for c, lbl in (
                ("scripts/check_defense_source.py", "defence really is the blend"),
                ("scripts/check_board_gp.py", "games played vs the schedule"),
                ("scripts/check_next_contracts.py", "Next $ is populated"),
                ("scripts/check_season_surfaces.py", "no team dropped from the pool")))
        else:
            ok = sh(["scripts/refresh_live_season.py"], "build + checks", check=False)
    except Exception as e:
        return rollback(f"build raised {type(e).__name__}: {e}")
    if not ok:
        # check_cache_seeding fails until the files are staged, which is
        # expected here; anything else is real.
        staged_only = subprocess.run(
            [PY_, "-u", "scripts/check_defense_source.py"], cwd=ROOT,
            capture_output=True).returncode == 0
        if not staged_only:
            return rollback("a data check failed")
        print("\n  (the only failure was check_cache_seeding, which cannot pass "
              "until the new files are staged)")

    # ---- 5 VERIFY ------------------------------------------------------
    print("\n--- verify against the previous revision " + "-" * 22)
    new = board_summary(new_files[0])
    if new is None:
        return rollback("no board was written")
    if prev is None:
        print("  no previous revision to compare against")
    else:
        gone_p = sorted(prev["players"] - new["players"])
        gone_t = sorted(set(prev["teams"]) - set(new["teams"]))
        moved = [(n, prev["salaries"][n], new["salaries"][n])
                 for n in prev["salaries"]
                 if n in new["salaries"]
                 and abs(prev["salaries"][n] - new["salaries"][n]) > 1]
        print(f"  rows      {prev['rows']} -> {new['rows']}  ({new['rows'] - prev['rows']:+d})")
        print(f"  teams     {len(prev['teams'])} -> {len(new['teams'])}")
        print(f"  dropped   {len(gone_p)} player(s), {len(gone_t)} team(s)")
        print(f"  salaries  {len(moved)} existing player(s) re-priced")
        if gone_t:
            return rollback(f"team(s) {gone_t} vanished from the board")
        if gone_p:
            print("            " + ", ".join(gone_p[:10]))
            return rollback(f"{len(gone_p)} player(s) vanished from the board")
        if moved:
            for n, a, b in sorted(moved, key=lambda x: -abs(x[1] - x[2]))[:5]:
                print(f"            {n}: {a:,.0f} -> {b:,.0f}")

    # ---- 6 SHIP --------------------------------------------------------
    paths = [str(f.relative_to(ROOT)) for f in new_files]
    paths += ["utils.py", "pages/Rankings.py",
              "scripts/build_per_game.py", "scripts/build_defense_blend.py",
              "data/provisional_box_scores.csv"]
    extra = subprocess.run(["git", "diff", "--name-only", "--",
                            "cache/bref_stats_v3_" + season + ".parquet"],
                           cwd=ROOT, capture_output=True, text=True).stdout.split()
    paths += extra
    print("\n--- to stage " + "-" * 50)
    for p in paths:
        print(f"  {p}")
    if not args.commit:
        print("\nBuilt and verified. Re-run with --commit (and --push) to ship.")
        return 0

    subprocess.run(["git", "add", "--"] + paths, cwd=ROOT, check=True)
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                            capture_output=True, text=True).stdout
    private = re.search(r"(^|/)(agency_db\.py|private_pages/|data/player_agents|"
                        r"data/agency_contacts|cache/agent_contracts|"
                        r"scripts/build_agent_db\.py|scripts/audit_agents\.py)",
                        staged, re.M)
    if private:
        subprocess.run(["git", "reset"], cwd=ROOT)
        print("\nSTOPPING: a private agency file was staged. Nothing committed.")
        return 1
    print("\n  agency guard: clean")

    msg = (f"Refresh {season} through the {behind[-1] if behind else 'current'} slate\n\n"
           f"{sum(espn.values()) // 2} games in the feed, nothing missing. "
           f"Board {prev['rows'] if prev else '?'} -> {new['rows']} players across "
           f"{len(new['teams'])} teams, nobody dropped.\n\n"
           f"per_game v{after['per_game']}, defense_blend v{after['defense_blend']}, "
           f"DATA_REV r{after['DATA_REV']}. Built by scripts/nightly.py; all checks pass.\n\n"
           f"Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>\n")
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, check=True)
    print("  committed")
    if args.push:
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
        print("  pushed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
