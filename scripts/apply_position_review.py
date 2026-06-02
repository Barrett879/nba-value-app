"""Apply the user's edits from player_positions_review.xlsx: diff each row's
Primary/Secondary against the CURRENT resolved position, and write every change
into data/player_positions_override.csv (primary-first, wins over 2K). Prints the
diff and flags any invalid cells. Self-locating cwd.

Usage:  python -u scripts/apply_position_review.py
"""
import os, sys, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
warnings.filterwarnings("ignore")
import pandas as pd
from utils import build_ranked_projected, normalize, SEASONS
import team_suitors as ts

VALID = {"PG", "SG", "SF", "PF", "C"}
xls = pd.read_excel("player_positions_review.xlsx")

# CURRENT resolved baseline (same pipeline that generated the sheet)
ranked = build_ranked_projected(max(SEASONS, key=lambda s: int(s[:4])))
posmap = ts.load_player_positions()
pick = lambda *c: next((x for x in c if x in ranked.columns), None)
pcol = pick("Player", "PLAYER_NAME", "name")
poscol = pick("position_detailed", "position", "POSITION", "Pos")
baseline = {}
for _, r in ranked.iterrows():
    nm = str(r[pcol]); bb = str(r[poscol]) if poscol and pd.notna(r[poscol]) else ""
    baseline[normalize(nm)] = ts.resolve_position(nm, bb, posmap)


def desired(p, s):
    p = ts._norm5(str(p)) if str(p).strip() and str(p).lower() != "nan" else None
    s = ts._norm5(str(s)) if str(s).strip() and str(s).lower() != "nan" else None
    if p is None:
        return None
    parts = [p] + ([s] if s and s != p else [])
    return "/".join(parts)


edits, invalid = [], []
for _, r in xls.iterrows():
    name = str(r["Player"]); nn = normalize(name)
    P, S = r.get("Primary"), r.get("Secondary")
    if str(P).strip().upper() not in VALID or (str(S).strip() and str(S).lower() != "nan"
                                               and str(S).strip().upper() not in VALID):
        invalid.append((name, P, S)); continue
    want = desired(P, S)
    if want and want != baseline.get(nn):
        edits.append((name, baseline.get(nn, "(none)"), want))

print(f"{len(edits)} position changes vs current; {len(invalid)} invalid rows skipped\n", flush=True)
for nm, old, new in edits:
    print(f"  {nm:<26} {str(old):>8}  ->  {new}", flush=True)
if invalid:
    print("\nINVALID (left unchanged):", flush=True)
    for nm, P, S in invalid:
        print(f"  {nm:<26} P={P!r} S={S!r}", flush=True)

# merge into the override CSV, preserving the comment header + existing order
path = ts.POS_OVERRIDE_PATH
lines = open(path).read().splitlines()
hdr, existing = [], {}
seen_header = False
for ln in lines:
    if not seen_header:
        hdr.append(ln)
        if ln.strip().lower().startswith("name,positions"):
            seen_header = True
        continue
    if ln.strip() and not ln.lstrip().startswith("#"):
        nm = ln.split(",")[0].strip()
        existing[normalize(nm)] = (nm, ln.split(",", 1)[1].strip() if "," in ln else "")
order = list(existing.keys())
for name, _old, new in edits:
    nn = normalize(name)
    if nn not in existing:
        order.append(nn)
    existing[nn] = (name, new)
with open(path, "w") as f:
    f.write("\n".join(hdr) + "\n")
    for nn in order:
        nm, pos = existing[nn]
        f.write(f"{nm},{pos}\n")
print(f"\nwrote {len(order)} override rows -> {path}", flush=True)
