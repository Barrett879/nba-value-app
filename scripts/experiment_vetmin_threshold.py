"""Find the right VETMIN_COMP_TARGET_MAX so minimum-caliber players (low but
positive trailing Barrett — Thanasis 1.1, Amari 2.4) get their real minimum
comps included, WITHOUT dragging down genuine low-rotation players (~5-7).

For each test player, print trailing Barrett + the market median at several
candidate thresholds. Pick the lowest threshold that fixes the scrubs while
leaving rotation players' market roughly intact.

Usage:  python -u scripts/experiment_vetmin_threshold.py
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np

SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "p", "__file__": str(ROOT / "pages" / "Contract_Predictor.py")}
exec(compile("".join(SRC[:cut]), "p", "exec"), ns)

find_comparables = ns["find_comparables"]
load_hist = ns["load_historical_signings"]
get_feats = ns["get_player_features"]
comp_dollars = ns["_comp_salaries_in_contract_dollars"]
wmedian = ns["_weighted_median"]
idw = ns["_inverse_distance_weights"]
hist = load_hist(n_recent_pairs=3)

THRESHOLDS = [0.0, 3.0, 4.0, 5.0, 6.0]
PLAYERS = ["Thanasis Antetokounmpo", "Amari Williams", "Isaiah Joe",
           "Jaylen Brown", "Naji Marshall", "Royce O'Neale", "Duncan Robinson",
           # probe the 4-9 Barrett band — low-rotation vets who DID sign real
           # (non-minimum) deals; raising the threshold must not gut their market
           "Larry Nance Jr.", "Taurean Prince", "Kevin Love", "Gary Payton II",
           "Jae'Sean Tate", "Torrey Craig", "Garrison Mathews", "Jaden Springer"]


def market_of(f):
    c = find_comparables(f, hist, n=6)
    if c.empty:
        return float("nan"), []
    sal = comp_dollars(c)
    w = idw(c["distance"].astype(float).values) if "distance" in c else np.ones_like(sal)
    return wmedian(sal, w) / 1e6, list(c["Player"].head(3))


print(f"{'player':<26}{'tBarrett':>9}   " + "  ".join(f"thr={t:g}" for t in THRESHOLDS), flush=True)
for name in PLAYERS:
    f = get_feats(name)
    if not f:
        print(f"  !! {name}: no features", flush=True); continue
    tb = f.get("trailing_barrett", f.get("barrett_score"))
    cells = []
    for t in THRESHOLDS:
        ns["VETMIN_COMP_TARGET_MAX"] = t
        m, _ = market_of(f)
        cells.append(f"${m:>5.1f}M")
    print(f"{name:<26}{tb:>+9.1f}   " + "  ".join(cells), flush=True)
