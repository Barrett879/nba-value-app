"""Will production find the caches it is about to go looking for?

    /opt/anaconda3/bin/python3 -u scripts/check_cache_seeding.py

Render's /data seeds a repo cache file only when that FILENAME is missing, and
it cannot rebuild from the network on a request path. So the moment
FORMULA_VERSION or DATA_REV changes, production starts asking for filenames
that may exist nowhere: not on its disk, because the name is new, and not in
the repo, if the rebuilt files were not committed in the same change.

That is not hypothetical. On 2026-10-25 a FORMULA_VERSION bump from v7 to v9
shipped with no v9 parquets, invalidating ~160 committed caches back to 1973-74
in one commit, and the live 2026-27 Rankings page served "No regular season
data" for twelve minutes.

Checks what is TRACKED IN GIT, not what is on disk. Local disk is exactly where
the missing files DO exist, because whoever bumped the version just built them,
which is why a gate that runs against the working tree cannot catch this.

Run it before any push that touches either constant. Exit code 1 on failure.
"""
from __future__ import annotations
import logging, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
import utils
logging.disable(logging.NOTSET)

FV, REV = utils.FORMULA_VERSION, utils.DATA_REV
LIVE = utils.SEASONS[0]


def tracked():
    out = subprocess.run(["git", "ls-files", "cache/"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


def main():
    have = tracked()
    print(f"FORMULA_VERSION = {FV}   DATA_REV = {REV}   live season = {LIVE}\n")

    # The files that carry the CURRENT season, which _rev_path revisions.
    want = [f"cache/raw_{LIVE.replace('-', '_')}_{FV}_{REV}.parquet",
            f"cache/all_seasons_0_{FV}_{REV}.parquet",
            f"cache/all_seasons_500_{FV}_{REV}.parquet"]

    bad = []
    print("files production will ask for once this lands:")
    for w in want:
        ok = w in have
        # _rev_path falls back to the unsuffixed name, so a missing revision
        # degrades to stale-but-present rather than to a rebuild -- as long as
        # that unsuffixed file is itself committed.
        legacy = w.replace(f"_{REV}.parquet", ".parquet")
        fb = (not ok) and legacy in have
        print(f"  {'ok  ' if ok else ('FALLS BACK' if fb else 'MISSING')}  {w}"
              + (f"   -> would serve stale {Path(legacy).name}" if fb else ""))
        if not ok and not fb:
            bad.append(w)

    # A formula bump also invalidates every historical season at once.
    hist = [f for f in have if f.startswith("cache/raw_") and f.endswith(".parquet")
            and f"_{FV}" in f and "playoff" not in f]
    print(f"\nhistorical caches committed at {FV}: {len(hist)}")
    if len(hist) < 20:
        print(f"  WARNING: only {len(hist)} seasons are committed at {FV}. A formula bump "
              f"invalidates every season, so each one production serves needs a {FV} file "
              f"in the repo or it will try to rebuild from the network.")
        bad.append(f"historical raw_*_{FV}.parquet set looks incomplete")

    if bad:
        print("\nFAIL: production would have nothing on disk and nothing in the repo for:")
        for b in bad:
            print(f"  {b}")
        print("\nBuild them locally and commit them IN THIS CHANGE, or revert the bump.")
        return 1
    print("\nPASS: every cache production will look for is committed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
