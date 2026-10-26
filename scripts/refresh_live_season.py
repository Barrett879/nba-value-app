"""Rebuild the live season's caches IN THE ORDER THEY DEPEND ON EACH OTHER.

    /opt/anaconda3/bin/python3 -u scripts/refresh_live_season.py [season]

There are four artifacts and the order is not optional:

    defense_blend  ->  board (raw + pooled)
    per_game           board

build_raw calls fetch_defense_blend and writes whatever it gets into d_lebron.
A missing blend is not an error there, because for pre-2009 seasons the box
proxy IS the right answer -- so building the board first silently scores the
whole league off a different metric. That shipped on 2026-10-25: the board was
rebuilt before build_defense_blend wrote v5, and the live site served 362
players on the box proxy, 63 of them more than two points off, with every row
present and plausible.

A milder version happened the same night: the blend was rebuilt under a name
the board had already consumed, leaving the two committed files disagreeing by
0.1-0.3 per player. Same class, smaller blast radius, equally invisible.

So this script does it in one ordered pass and then PROVES the result, rather
than leaving the order to whoever is typing at midnight.

It also handles the _rev_path build trap: build_all_seasons_combined resolves
through _rev_path, which falls back to the UNSUFFIXED pooled file when the
revisioned one does not exist yet, so the builder is handed a stale copy and
the new revision never gets written. The unsuffixed files are moved aside for
the build and put back afterwards.

Nothing is committed or pushed. That stays deliberate -- every deploy is about
45 seconds of hard downtime.
"""
from __future__ import annotations
import logging, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
logging.disable(logging.WARNING)
import utils
logging.disable(logging.NOTSET)

PY_ = sys.executable


def run(label, args):
    print(f"\n=== {label} " + "=" * max(0, 56 - len(label)))
    r = subprocess.run([PY_, "-u", *args], cwd=ROOT)
    if r.returncode:
        print(f"\nFAILED: {label} exited {r.returncode}")
    return r.returncode == 0


def tracked(path_name: str) -> bool:
    out = subprocess.run(["git", "ls-files", "--error-unmatch", f"cache/{path_name}"],
                         cwd=ROOT, capture_output=True, text=True)
    return out.returncode == 0


def main(season: str, board_only: bool = False) -> int:
    key = season.replace("-", "_")
    fv, rev = utils.FORMULA_VERSION, utils.DATA_REV
    print(f"live season {season}   FORMULA_VERSION {fv}   DATA_REV {rev}")

    # Preflight: a filename already in git is already on production's disk, and
    # /data never reseeds an existing name. Rebuilding under it changes nothing
    # live, however correct the new contents are.
    import build_per_game, build_defense_blend
    targets = {f"raw_{key}_{fv}_{rev}.parquet": "DATA_REV in utils.py"}
    if not board_only:
        targets[f"per_game_{key}_{build_per_game.VERSION}.json"] = \
            "VERSION in scripts/build_per_game.py"
        targets[f"defense_blend_{key}_{build_defense_blend.VERSION}.json"] = \
            "VERSION in scripts/build_defense_blend.py"
    live = {n: where for n, where in targets.items() if tracked(n)}
    if live:
        print("\nREFUSING: these filenames are already committed, so production "
              "already has them and will ignore new contents under the same name:")
        for n, where in live.items():
            print(f"  {n}\n      bump {where}")
        print("\nBump what is listed, then run this again.")
        return 1

    # ---- artifacts FIRST, board second. This is the whole point. ----
    if board_only:
        # Repairing a board that was built against the wrong artifacts, where
        # the artifacts themselves are correct and already live. Rebuilding
        # them would only force needless version bumps.
        print("\n--board-only: reusing the artifacts already on disk")
        for n in (f"defense_blend_{key}_{build_defense_blend.VERSION}.json",
                  f"per_game_{key}_{build_per_game.VERSION}.json"):
            f = utils.CACHE_DIR / n
            print(f"  {'ok     ' if f.exists() else 'MISSING'}  {n}")
            if not f.exists():
                print("\nREFUSING: --board-only needs the artifacts to exist already.")
                return 1
    else:
        if not run("defense blend", ["scripts/build_defense_blend.py", season]):
            return 1
        if not run("per-game panel", ["scripts/build_per_game.py", season]):
            return 1

    print("\n=== board + pooled " + "=" * 38)
    held = []
    with tempfile.TemporaryDirectory() as tmp:
        for p in (utils.CACHE_DIR / f"all_seasons_0_{fv}.parquet",
                  utils.CACHE_DIR / f"all_seasons_500_{fv}.parquet"):
            if p.exists():
                shutil.move(str(p), str(Path(tmp) / p.name))
                held.append((Path(tmp) / p.name, p))
                print(f"  held aside {p.name} (the _rev_path build trap)")
        try:
            d = utils.build_raw(season)
            print(f"  raw rows {len(d)}   d_lebron {d['d_lebron'].min():+.2f} .. "
                  f"{d['d_lebron'].max():+.2f}")
            for thr in (0, 500):
                print(f"  pooled {thr} -> {len(utils.build_all_seasons_combined(min_threshold=thr))}")
        finally:
            for src, dst in held:
                shutil.move(str(src), str(dst))
                print(f"  restored {dst.name}")

    # ---- prove it ----
    ok = True
    for label, script in (("defence really is the blend", "scripts/check_defense_source.py"),
                          ("games played vs the schedule", "scripts/check_board_gp.py")):
        ok = run(label, [script, season]) and ok
    ok = run("every cache production asks for is committed",
             ["scripts/check_cache_seeding.py"]) and ok

    print("\n" + "=" * 62)
    if ok:
        print("ALL CHECKS PASS. Stage explicitly, commit, push, verify live.")
        return 0
    print("CHECKS FAILED -- do not push. check_cache_seeding fails until you "
          "stage the new files, which is expected at this point.")
    return 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(args[0] if args else utils.SEASONS[0],
                  board_only="--board-only" in sys.argv))
