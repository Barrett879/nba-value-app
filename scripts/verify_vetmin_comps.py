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
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
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
    new = find_comparables(f, hist, n=6)
    print(f"{name}  (trailing Barrett {tb:+.1f})   market ${market_median(new):.1f}M", flush=True)
    print(f"  {'comp':<22} {'score':>6} {'|Δscore|':>8} {'salary$M':>9}", flush=True)
    sal = comp_dollars(new) / 1e6
    for i, (_, r) in enumerate(new.iterrows()):
        d = abs(r["barrett_score"] - tb)
        print(f"  {r['Player']:<22} {r['barrett_score']:>+6.1f} {d:>8.1f} {sal[i]:>9.1f}", flush=True)
    print(flush=True)


for p in ["Johni Broome", "Isaiah Joe", "Jaylen Brown"]:
    try:
        show(p)
    except Exception as e:
        print(f"!! {p}: {type(e).__name__}: {e}\n", flush=True)
