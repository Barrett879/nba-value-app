"""Build cache/option_status_2026.json — each FA-pool player's PLAYER/TEAM option for
the upcoming season, taken from the reliable contract-end scraper (get_player_contract_info).

WHY: the live next-year-salary feed (fetch_next_year_contracts) mislabels options as
'guaranteed' and is non-deterministic — one render flags ~20 team options, the next flags
zero — so option-holders silently vanish from the Free Agent list and the PO/TO counts swing
run-to-run. classify_fa_status now reads THIS stable cache as the authoritative option source.

Run when the offseason's option picture changes:  python -u scripts/build_option_status.py
"""
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import utils
from utils import normalize, build_ranked_projected, get_player_contract_info
SEASON = utils.SEASONS[0]
_cs = int(SEASON[:4]) + 1
UPCOMING = f"{_cs}-{(_cs + 1) % 100:02d}"          # e.g. 2026-27

# Known scraper misses — force the correct option type (verified against reporting).
OVERRIDES = {
    normalize("Trae Young"): "player_option",      # ESPN/Yahoo: declined $49M player option (scraper said "guaranteed")
}


class _TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_TO()))

df = build_ranked_projected(SEASON)
players = list(dict.fromkeys(df["Player"].tolist()))
out, n_err = {}, 0
t0 = time.time()
print(f"scanning {len(players)} players for {UPCOMING} options...", flush=True)
for i, name in enumerate(players):
    n = normalize(name)
    if n in OVERRIDES:
        out[n] = {"type": OVERRIDES[n], "end_season": UPCOMING, "source": "override"}
        continue
    try:
        signal.alarm(20)
        ci = get_player_contract_info(name) or {}
        signal.alarm(0)
    except Exception:
        signal.alarm(0)
        n_err += 1
        continue
    typ, end = ci.get("last_year_type"), ci.get("end_season")
    if typ in ("player_option", "team_option") and end == UPCOMING:
        out[n] = {"type": typ, "end_season": end, "source": "scraper"}
    if (i + 1) % 40 == 0:
        print(f"  {i + 1}/{len(players)} scanned · {len(out)} options · {n_err} errs · {time.time() - t0:.0f}s", flush=True)

(ROOT / "cache" / "option_status_2026.json").write_text(
    json.dumps({"upcoming": UPCOMING, "season": SEASON, "options": out}, indent=1))
po = sum(1 for v in out.values() if v["type"] == "player_option")
to = sum(1 for v in out.values() if v["type"] == "team_option")
print(f"WROTE cache/option_status_2026.json — {len(out)} option-holders "
      f"({po} PO, {to} TO), {n_err} scrape errors, {time.time() - t0:.0f}s", flush=True)
