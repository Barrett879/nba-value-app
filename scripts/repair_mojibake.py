"""Repair mojibake baked into cache files by the pre-fix HTML decoder.

    python3 repair_mojibake.py            # dry run, changes nothing
    python3 repair_mojibake.py --apply    # rewrite the files

utils._html_text fixed the DECODER, but every cache written before it still
holds latin-1-misread UTF-8. Pre-1996 seasons are immutable (season_is_live is
False, so _dc_fresh never refetches them), which means those files will carry
the damage forever unless it is repaired in place.

A string is only repaired when the round trip is unambiguous: it matches the
mojibake signature, encodes cleanly back to latin-1, decodes as UTF-8, and the
result carries no signature itself. Anything else is left alone. That makes the
pass idempotent and refuses to guess.
"""
from __future__ import annotations
import re, sys, shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
APPLY = "--apply" in sys.argv
MOJ = re.compile(r"[ÃÄÅÂ][\x80-\xbf\u0080-ÿ]")


def repair(s):
    """The corrected string, or None when the round trip is not unambiguous."""
    if not isinstance(s, str) or not MOJ.search(s):
        return None
    try:
        out = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    if MOJ.search(out) or out == s:
        return None
    return out


total_files = total_cells = 0
changes = {}
for p in sorted(CACHE.rglob("*.parquet")):
    try:
        df = pd.read_parquet(p)
    except Exception:
        continue
    edits, per_file = {}, []
    for c in df.columns:
        if df[c].dtype != object:
            continue
        col = df[c]
        mapped = {}
        for i, v in col.items():
            r = repair(v)
            if r is not None:
                mapped[i] = r
        if mapped:
            edits[c] = mapped
            per_file += [f"{v!r} -> {r}" for v, r in
                         list(zip([col[i] for i in mapped], mapped.values()))[:3]]
    if not edits:
        continue
    n = sum(len(m) for m in edits.values())
    total_files += 1
    total_cells += n
    changes[str(p)] = (n, list(edits), per_file[:3])
    if APPLY:
        before_shape, before_cols = df.shape, list(df.columns)
        for c, mapped in edits.items():
            for i, v in mapped.items():
                df.at[i, c] = v
        assert df.shape == before_shape and list(df.columns) == before_cols, p
        tmp = p.with_suffix(".parquet.repairtmp")
        df.to_parquet(tmp, index=False)
        shutil.move(str(tmp), str(p))

print(f"\n  {'APPLIED' if APPLY else 'DRY RUN'}: {total_cells} cells across {total_files} files\n")
for f, (n, cols, ex) in sorted(changes.items())[:8]:
    print(f"  {f}  ({n} cells in {cols})")
    for e in ex:
        print(f"      {e}")
if len(changes) > 8:
    print(f"  ... and {len(changes)-8} more files")
print()
