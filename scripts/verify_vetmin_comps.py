"""Verify the target-aware vet-min comp fix WITHOUT the Streamlit render.

Exec only the function/constant prefix of Contract_Predictor.py (everything
above the `selected = st_searchbox(` render line), then call the real
find_comparables / market-median helpers for:
  - Johni Broome  (below-replacement, the bug — should CHANGE: cheap comps in,
                   market median drops well below the old ~$4.6M floor)
  - Isaiah Joe    (positive Barrett rotation — must be IDENTICAL old vs new)
  - a clear star  (must be IDENTICAL old vs new)

Old behavior = drop ALL vet-min from the pool before matching (what shipped).
New behavior = find_comparables decides per-target.

Usage:  python -u scripts/verify_vetmin_comps.py
"""
import sys, re, warnings
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("selected = st_searchbox("))
prefix = "".join(SRC[:cut])
ns = {"__name__": "cp_prefix", "__file__": str(ROOT / "pages" / "Contract_Predictor.py")}
exec(compile(prefix, "Contract_Predictor_prefix", "exec"), ns)

find_comparables = ns["find_comparables"]
load_hist        = ns["load_historical_signings"]
get_feats        = ns["get_player_features"]
comp_dollars     = ns["_comp_salaries_in_contract_dollars"]
wmedian          = ns["_weighted_median"]
idw              = ns["_inverse_distance_weights"]

hist = load_hist(n_recent_pairs=3)
print(f"pool: {len(hist)} signings   vet-min tagged: "
      f"{int(hist['is_vet_min'].sum()) if 'is_vet_min' in hist else 'MISSING'}\n", flush=True)


def market_median(comps):
    if comps.empty:
        return float("nan")
    sal = comp_dollars(comps)
    w = idw(comps["distance"].astype(float).values) if "distance" in comps else np.ones_like(sal)
    return wmedian(sal, w) / 1e6


def show(name):
    f = get_feats(name)
    if not f:
        print(f"!! {name}: no features\n", flush=True); return
    tb = f.get("trailing_barrett", f.get("barrett_score"))
    # NEW = the shipped find_comparables (target-aware)
    new = find_comparables(f, hist, n=6)
    # OLD = emulate the prior pool (vet-min dropped for everyone)
    old = find_comparables(f, hist[~hist["is_vet_min"]].reset_index(drop=True), n=6)
    nm, om = market_median(new), market_median(old)
    flag = "←CHANGED" if abs(nm - om) > 0.05 else "identical"
    print(f"{name}  (trailing Barrett {tb:+.1f})", flush=True)
    print(f"  OLD market median: ${om:.1f}M   comps: {', '.join(old['Player'].head(4))}", flush=True)
    print(f"  NEW market median: ${nm:.1f}M   comps: {', '.join(new['Player'].head(4))}   [{flag}]\n", flush=True)


for p in ["Johni Broome", "Isaiah Joe", "Jaylen Brown"]:
    try:
        show(p)
    except Exception as e:
        print(f"!! {p}: {type(e).__name__}: {e}\n", flush=True)
