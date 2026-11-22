"""Build cache/option_status_<board_year>.json -- each FA-pool player's PLAYER or
TEAM option for the offseason that FOLLOWS the season being scored.

WHY IT EXISTS: the live next-year-salary feed (fetch_next_year_contracts)
mislabels some options as 'guaranteed', and a player read as guaranteed is
dropped from the free-agent board as "under contract". Measured on 2026-11-20
against the feed production actually serves (_rev_path picks the DATA_REV copy):
SIX options read as guaranteed. utils.classify_fa_status therefore consults this
stable cache FIRST, then data/contract_years_2026_27.csv, then the feed.

THE FILENAME IS DERIVED, NOT TYPED. Until 2026-11-22 it was hardcoded to
option_status_2026.json while the CONTENT rolled over with SEASONS[0], so running
this a season later would have overwritten the closed 2026 record with 2027-28
data and still left utils._load_option_status looking for a file that was never
written. The loader derives its filename the same way; keep the two in step.

NO OVERRIDES TABLE. There was one, forcing Trae Young to player_option because
the scraper read his contract as guaranteed. That was a fact about the 2026
offseason -- he DECLINED a $49M option and re-signed with Washington for
4yr/$212.8M -- and the option on the new deal is 2029-30, which
data/contract_years_2026_27.csv records as 2026:G|2027:G|2028:G|2029:P. Carried
into a 2027 file it would have asserted a 2027-28 option he does not have, and
because this cache is consulted FIRST it would have beaten the correct flag and
published him on the 2027 board four months into that contract. That is the same
shape as the bug _load_option_status was fixed for on 2026-11-20. A one-season
fact needs a season on it; if a scraper miss needs correcting again, correct it
in contract_years, which is keyed per season.

Run when the offseason's option picture changes:
    python -u scripts/build_option_status.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import utils          # the pool's keys are already normalized, so no normalize() here

SEASON = utils.SEASONS[0]
BOARD_YEAR = int(SEASON[:4]) + 1
UPCOMING = f"{BOARD_YEAR}-{(BOARD_YEAR + 1) % 100:02d}"
OUT = ROOT / "cache" / f"option_status_{BOARD_YEAR}.json"

# The pool and the contract-end map both come from the COMMITTED player-hub
# artifact, which the nightly rebuilds and which classify_fa_status itself names
# as the precomputed stand-in for get_player_contract_info ("request-path callers
# can never fall through to the live ~30-page BBRef scrape"). Reading it keeps
# this script off build_ranked_projected(), which in a bare process rebuilds and
# overwrites committed caches, and makes the build deterministic and offline.
HUB = ROOT / "cache" / "player_hub_pcv_v2.json"
hub = json.loads(HUB.read_text(encoding="utf-8"))
pool, contract_end = hub["players"], hub["contract_end"]
if not pool or not contract_end:
    raise SystemExit(f"{HUB.name} has an empty players/contract_end map; refusing "
                     "to write an option cache that would read as 'nobody has an option'")
print(f"{SEASON} -> {UPCOMING} options; pool {len(pool)}, contract_end {len(contract_end)} "
      f"(hub built {hub.get('built_at')})", flush=True)

out = {}
for n in pool:
    ci = contract_end.get(n) or {}
    if ci.get("last_year_type") in ("player_option", "team_option") and ci.get("end_season") == UPCOMING:
        out[n] = {"type": ci["last_year_type"], "end_season": UPCOMING, "source": "contract_end"}

po = sum(1 for v in out.values() if v["type"] == "player_option")
to = sum(1 for v in out.values() if v["type"] == "team_option")
OUT.write_text(json.dumps({"upcoming": UPCOMING, "season": SEASON, "options": out}, indent=1))
print(f"WROTE cache/{OUT.name} -- {len(out)} option-holders ({po} PO, {to} TO)", flush=True)
