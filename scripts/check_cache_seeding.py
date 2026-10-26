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
import logging, re, subprocess, sys
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


# The two JSON artifacts are version-named by their OWN constants, not by
# FORMULA_VERSION or DATA_REV, and the readers spell that version as a literal.
# So the name production asks for lives in the READER, and the builder's
# constant only says what was last written. A bump to one and not the other is
# how the TOV and PF bars shipped empty on 2026-10-25: contents changed under a
# name production already had, so /data never reseeded.
ARTIFACTS = [
    # label, reader file, reader regex, builder file
    ("per_game", "pages/Rankings.py", r'per_game_\{season_key\.replace[^}]*\}_(v\d+)\.json',
     "scripts/build_per_game.py"),
    ("defense_blend", "utils.py", r'defense_blend_\{season\.replace[^}]*\}_(v\d+)\.json',
     "scripts/build_defense_blend.py"),
]


def artifact_versions(label, reader, pattern, builder):
    """What the reader asks for, and what the builder last wrote."""
    hits = set(re.findall(pattern, (ROOT / reader).read_text()))
    built = re.search(r'^VERSION = "(v\d+)"', (ROOT / builder).read_text(), re.M)
    return hits, (built.group(1) if built else None)


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

    print("\nversioned JSON artifacts (named by their own constant, not by DATA_REV):")
    for label, reader, pattern, builder in ARTIFACTS:
        asked, built = artifact_versions(label, reader, pattern, builder)
        if not asked:
            print(f"  CANNOT TELL  {label}: no version literal found in {reader}")
            bad.append(f"{label} version unreadable from {reader}")
            continue
        if len(asked) > 1:
            print(f"  INCONSISTENT  {label}: {reader} asks for {sorted(asked)}")
            bad.append(f"{label} readers disagree: {sorted(asked)}")
        for ver in sorted(asked):
            name = f"cache/{label}_{LIVE.replace('-', '_')}_{ver}.json"
            ok = name in have
            print(f"  {'ok  ' if ok else 'MISSING'}  {name}")
            if not ok:
                bad.append(name)
        if built and built not in asked:
            # Half a bump. The builder wrote a name nothing reads, and the
            # readers keep asking for the old one, which production already has.
            print(f"  PARTIAL BUMP  {builder} writes {built} but {reader} "
                  f"reads {sorted(asked)}")
            bad.append(f"{label}: builder at {built}, reader at {sorted(asked)}")

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
