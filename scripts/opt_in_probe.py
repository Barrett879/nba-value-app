"""Player-option decision model (concept probe): will a player exercise his
player option (opt IN, take the guaranteed money) or decline it (opt OUT) to sign
a new deal? The dominant driver is the option salary vs his projected market
value — a player keeps a $30M option rather than sign for $17M — with age as the
tiebreaker in the gray zone (older players take the security; younger players bet
on a longer deal).

Usage:  python -u scripts/opt_in_probe.py
"""
import math
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from utils import normalize, DEFAULT_MIN_THRESHOLD  # noqa: E402


def opt_in_prob(po_M: float, market_M: float, age: float) -> float:
    """Probability a player EXERCISES (opts into) his player option.
    s = surplus the option pays over the market, as a fraction of the option.
    Big positive s -> keep the money; older age nudges toward opting in."""
    if po_M <= 0:
        return 0.0
    s = (po_M - market_M) / po_M
    z = 9.0 * s + 0.16 * (age - 28.0)
    return 1.0 / (1.0 + math.exp(-z))


SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "cp", "__file__": str((ROOT / "pages" / "Contract_Predictor.py").resolve())}
exec(compile("".join(SRC[:cut]), "cp", "exec"), ns)

CUR = ns["CURRENT_SEASON"]
gpf, pc = ns["get_player_features"], ns["predict_contract"]
nc = ns["fetch_next_year_contracts"](ns["season_to_espn_year"](CUR), cache_v=7)
full = ns["build_ranked_projected"](CUR)

po = {n: v for n, v in nc.items() if (v or {}).get("type") == "player_option"}
print(f"{len(po)} player-option players in {CUR}\n")

rows = []
for _, r in full.iterrows():
    name = str(r["Player"])
    info = po.get(normalize(name))
    if not info:
        continue
    f = gpf(name, CUR)
    if not f:
        continue
    po_M = float(info["salary"]) / 1e6
    market_M = float(pc(f)["predicted"]) / 1e6
    age = float(f.get("age") or 28)
    p = opt_in_prob(po_M, market_M, age)
    rows.append((name, po_M, market_M, age, p))

rows.sort(key=lambda x: -x[4])
print(f"{'player':22}{'age':>4}{'option':>9}{'market':>9}{'Δ':>8}  {'opt-in':>7}  call")
for name, po_M, market_M, age, p in rows:
    call = "OPT IN (keep option)" if p >= 0.5 else "opt out (sign new)"
    print(f"  {name:20}{age:>4.0f}{po_M:>8.0f}M{market_M:>8.0f}M{po_M-market_M:>+7.0f}M  {p*100:>5.0f}%  {call}")
