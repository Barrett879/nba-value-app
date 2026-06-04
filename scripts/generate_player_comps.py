"""Generate every active player's comparable signings under the curated
PRIMARY-position matcher (no coarse G/F/C groups — a PF matches PFs, a C matches
Cs, etc.) and assert NO off-position comps. Writes player_comps.csv (Player,
Position, Comp 1..6) and prints any violations — proof the Chet-style SF-on-a-PF
bug is gone league-wide.

Usage:  python -u scripts/generate_player_comps.py
"""
import sys, csv, warnings
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from utils import normalize

SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "p", "__file__": str(ROOT / "pages" / "Contract_Predictor.py")}
exec(compile("".join(SRC[:cut]), "p", "exec"), ns)

fc, lh, cp = ns["find_comparables"], ns["load_historical_signings"], ns["_curated_pos"]
barr, CUR = ns["_barrett_lookup"], ns["CURRENT_SEASON"]
active = ns["active_names"]
ls = ns["fetch_league_stats"](CUR, "Regular Season")
age_by = dict(zip(ls["PLAYER_NAME"].map(normalize), ls.get("AGE", [])))
dt = ns["build_draft_tier_lookup"]()
det = ns["fetch_player_positions_detailed"](CUR, cache_v=3) or {}
hist = lh()


def feats(name):
    nn = normalize(name)
    info = dt.get(nn) or {}
    return {"name": name, "barrett_score": barr.get(name, 0.0),
            "age": age_by.get(nn) or 27, "draft_tier": info.get("draft_tier", "Undrafted"),
            "position": "", "position_detailed": det.get(nn, "")}


rows, violations = [], []
for name in sorted(active, key=lambda n: -barr.get(n, 0.0)):
    f = feats(name)
    primary = cp(f["name"], f["position_detailed"])
    comps = fc(f, hist, n=6)
    comp_names = list(comps["Player"]) if not comps.empty else []
    if not comps.empty:
        for _, r in comps.iterrows():
            if r["pos_primary"] != primary:
                violations.append(f"{name} ({primary}) -> {r['Player']} ({r['pos_primary']})")
    rows.append([name, primary, *comp_names, *[""] * (6 - len(comp_names))])

out = ROOT / "player_comps.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Player", "Position", "Comp 1", "Comp 2", "Comp 3",
                "Comp 4", "Comp 5", "Comp 6"])
    w.writerows(rows)

print(f"Wrote {len(rows)} players -> {out}")
print(f"Players with >=1 comp: {sum(1 for r in rows if r[2])}  |  "
      f"no-comp (suppressed): {sum(1 for r in rows if not r[2])}")
print(f"\nOFF-POSITION VIOLATIONS (a PF with a non-PF comp, etc.): {len(violations)}")
for v in violations[:40]:
    print("  ", v)
print("\nSample (top 12 by Barrett):")
for r in rows[:12]:
    print(f"  {r[0]:<24}{r[1]:<4}  ::  {', '.join(c for c in r[2:] if c)}")
