"""Confirm the Jaxson Hayes -> C/PF fix: build the REAL current roster the way
the page does, then run Brandon Ingram into LAL and report who he 'upgrades over'
(should now be a forward, not the center Hayes). Self-locating cwd.
"""
import os, sys, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
warnings.filterwarnings("ignore")
from utils import build_ranked_projected, normalize, SEASONS
import team_suitors as ts

CURRENT_SEASON = max(SEASONS, key=lambda s: int(s[:4]))   # newest season, regardless of list order
print("season:", CURRENT_SEASON, flush=True)
ranked = build_ranked_projected(CURRENT_SEASON)
posmap = ts.load_player_positions()
cr = ranked.copy()
cr = cr.drop(columns=[c for c in ["position_detailed"] if c in cr.columns])  # force our resolved 'pos'
cr["pos"] = cr["Player"].map(lambda p: ts.resolve_position(p, posmap.get(normalize(p), ""), posmap))
rost = ts.build_rosters(cr)

lal = rost[rost["team"] == "LAL"].sort_values("barrett", ascending=False)
print("LAL roster (player / resolved pos / barrett):", flush=True)
for _, r in lal.iterrows():
    print(f"  {r['player']:<22} {r['pos']:<7} {r['barrett']:.1f}", flush=True)

ing_pos = ts.resolve_position("Brandon Ingram", posmap.get(normalize("Brandon Ingram"), ""), posmap)
ing_row = cr[cr["Player"].map(normalize) == normalize("Brandon Ingram")]
ing_bar = float(ing_row.iloc[0]["barrett_score"]) if not ing_row.empty else 25.0
lal_wo_ing = lal[lal["player"].map(normalize) != normalize("Brandon Ingram")]
need = ts.roster_need(ing_bar, ing_pos, lal_wo_ing)
# who's in his competing set
elig = ts._eligible_positions(ing_pos); tp = ts._primary_position(ing_pos)
comp = [r["player"] for _, r in lal_wo_ing.iterrows()
        if (ts._primary_position(r["pos"]) in elig) or (tp in ts._eligible_positions(r["pos"]))]
print(f"\nBrandon Ingram ({ing_pos}, {ing_bar:.1f}) into LAL:", flush=True)
print(f"  competes with (forward spots): {comp}", flush=True)
print(f"  -> 'upgrades over': {need['displaces']} ({need['displaces_score']})  slot {need['slot']}", flush=True)
print(f"  Jaxson Hayes in competing set? {'Jaxson Hayes' in comp}  (should be False)", flush=True)
