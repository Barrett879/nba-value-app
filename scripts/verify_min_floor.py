"""Verify the service-year-scaled minimum floor (no Streamlit render).

Exec the function/constant prefix of Contract_Predictor.py, then check:
  - min_salary_pct() scale by service years
  - predict_contract() floor for a young min player (Broome) drops below the
    old flat $2.5M, and a mid/star is unaffected (prediction well above any min)

Usage:  python -u scripts/verify_min_floor.py
"""
import sys, re, warnings
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("selected = st_searchbox("))
ns = {"__name__": "cp_prefix", "__file__": str(ROOT / "pages" / "Contract_Predictor.py")}
exec(compile("".join(SRC[:cut]), "cp_prefix", "exec"), ns)

min_salary_pct = ns["min_salary_pct"]
get_feats = ns["get_player_features"]
predict = ns["predict_contract"]
CAP = ns["SALARY_CAP_M"].get(ns["CONTRACT_SEASON"], 165.0)

print(f"CONTRACT cap = ${CAP:.1f}M   (old flat floor was 1.5% = ${0.015*CAP:.2f}M)\n", flush=True)
print("min_salary_pct scale → floor $ at this cap:", flush=True)
for s in [0, 1, 2, 5, 10, 15]:
    p = min_salary_pct(s)
    print(f"  {s:>2} yr svc:  {p*100:.2f}% = ${p*CAP:.2f}M", flush=True)

print("\npredict_contract floor check:", flush=True)
for name in ["Johni Broome", "Isaiah Joe", "Jaylen Brown"]:
    f = get_feats(name)
    if not f:
        print(f"  !! {name}: no features", flush=True); continue
    pr = predict(f)
    print(f"  {name:<14} svc={f.get('service_years')}  "
          f"predicted=${pr['predicted']/1e6:.2f}M  "
          f"floor=${pr.get('min_floor_dollars',0)/1e6:.2f}M  "
          f"low=${pr['low']/1e6:.2f}M", flush=True)
