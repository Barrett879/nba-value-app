"""Build cache/player_hub_pcv_v2.json — the Contract Predictor's model projection
(pcv) + calibrated 80% band for EVERY current-pool player, plus the request-path
precomputes the homepage must never do live: per-player CBA max pct (max_pct, for
the MAX chips on actual 2026-27 salaries), the contract-end map (contract_end,
from the BBRef scraper — the homepage deal line + FA-status cross-check read it
here instead of risking a ~30-page scrape when the pkl TTL lapses), a built_at
date and a model_stamp (sha1 prefix of the production joblib the values came from).

v1 → v2 filename bump: Render's /data disk only seeds repo cache files that are
MISSING on the disk, so new keys added to the existing v1 file would never reach
production — a new filename does.

Powers the homepage Player Hub's "Predicted contract" instantly (no live model runs
at request time) and gives the FA list's Predicted column full coverage (the old
fa_sim/fa_extra board caches only covered ~160 of ~500 players).

Safety rails (2026-07-01 lessons): socket timeout so no fetch can hang, a hard
watchdog that kills the process loudly instead of silently stalling, the
cached-parquet patch so the feature path never refetches league stats, per-player
alarm guards, and periodic flushes so an interrupted run resumes where it left off.

Run after model changes or roster moves:  python -u scripts/build_player_hub.py
"""
import socket
socket.setdefaulttimeout(30)
import faulthandler
faulthandler.dump_traceback_later(3600, exit=True)   # whole-run hard cap

import datetime
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import utils
from utils import normalize, build_ranked_projected

# Serve stats from the committed parquets — never refetch at build time.
utils._raw_disk_fresh = lambda *a, **k: True

OUT = ROOT / "cache" / "player_hub_pcv_v2.json"
V1 = ROOT / "cache" / "player_hub_pcv_v1.json"     # resume seed for the first v2 run

SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "cp", "__file__": str((ROOT / "pages" / "Contract_Predictor.py").resolve())}
t0 = time.time()
print("exec Contract_Predictor prefix...", flush=True)
exec(compile("".join(SRC[:cut]), "cp", "exec"), ns)
print(f"  prefix ready in {time.time()-t0:.1f}s", flush=True)
gpf, pcv, band_fn = ns["get_player_features"], ns["projected_contract_value"], ns["_relative_band_dollars"]


class _TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_TO()))

# Stamp of the production model artifact the pcv values come from — the homepage
# compares it to the live joblib and log-warns when the build has gone stale.
_model_path = Path(ns.get("_HISTGBM_PATH") or (ROOT / "models" / "contract_histgbm_v2.joblib"))
try:
    MODEL_STAMP = hashlib.sha1(_model_path.read_bytes()).hexdigest()[:12]
except OSError:
    MODEL_STAMP = ""
print(f"model stamp {MODEL_STAMP or '(missing artifact)'} ({_model_path.name})", flush=True)

# Contract-end map (BBRef scraper, served from its own pkl cache when fresh).
# Precomputed here so the homepage's deal line + FA-status cross-check never
# risk the ~30-page scrape inside a visitor's request.
contract_end = utils.fetch_contract_end_years()
print(f"contract_end: {len(contract_end)} players under contract", flush=True)


def _write(players: dict) -> None:
    OUT.write_text(json.dumps({
        "season": utils.SEASONS[0],
        "built_at": datetime.date.today().isoformat(),
        "model_stamp": MODEL_STAMP,
        "players": players,
        "contract_end": contract_end,
    }, indent=0))


# Resume: keep whatever a previous (interrupted) run already computed — the v1
# file seeds the first v2 run so ~500 model projections aren't recomputed.
out: dict = {}
for _src in (OUT, V1):
    if _src.exists():
        try:
            out = json.loads(_src.read_text()).get("players", {})
            if out:
                print(f"  resuming with {len(out)} already computed ({_src.name})", flush=True)
                break
        except Exception:
            out = {}

pool = build_ranked_projected(utils.SEASONS[0])
players = list(dict.fromkeys(pool["Player"].astype(str)))
todo = [p for p in players if normalize(p) not in out]
print(f"pool={len(players)} · todo={len(todo)}", flush=True)

n_err = 0
for i, name in enumerate(todo):
    try:
        signal.alarm(60)
        f = gpf(name)
        if f:
            v = float(pcv(f)) / 1e6
            b = float(band_fn(v * 1e6)) / 1e6
            # Flag predictions sitting AT the player's CBA maximum (25/30/35% of the
            # contract-season cap by service years, incl. designated bumps) so the
            # UI can label them "(Max)". Star-snap values (max minus 2-3pp) are
            # deliberately below max and stay unflagged. max_pct itself is stored
            # too: the homepage compares it to ACTUAL 2026-27 salaries for the MAX
            # chips instead of recomputing eligibility inside the request.
            is_max = False
            max_pct = None
            try:
                elig = utils.get_max_contract_eligibility(name, utils.SEASONS[0])
                max_pct = float(elig["max_pct"])
                cap_M = utils.SALARY_CAP_M.get(ns["CONTRACT_SEASON"], 165.0)
                is_max = v >= (max_pct * cap_M) - 0.05
            except Exception:
                pass
            out[normalize(name)] = {
                "player": name,
                "pcv_M": round(v, 1),
                "low_M": round(max(0.0, v - b), 1),
                "high_M": round(v + b, 1),
                "is_max": bool(is_max),
                "max_pct": max_pct,
            }
        else:
            out[normalize(name)] = {"player": name, "pcv_M": None}
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        n_err += 1
        print(f"  ERR {name}: {type(e).__name__} {str(e)[:60]}", flush=True)
    if (i + 1) % 25 == 0:
        _write(out)
        print(f"  {i+1}/{len(todo)} · {len(out)} cached · {n_err} errs · {time.time()-t0:.0f}s", flush=True)

# Backfill max_pct for players carried over from a previous run / the v1 seed
# (v1 records only had the is_max flag, not the pct itself).
todo_pct = [n for n, rec in out.items() if "max_pct" not in rec]
print(f"max_pct backfill: {len(todo_pct)} players", flush=True)
for i, n in enumerate(todo_pct):
    rec = out[n]
    try:
        signal.alarm(60)
        elig = utils.get_max_contract_eligibility(rec.get("player") or n, utils.SEASONS[0])
        rec["max_pct"] = float(elig["max_pct"])
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        rec["max_pct"] = None
        n_err += 1
        print(f"  ERR max_pct {n}: {type(e).__name__} {str(e)[:60]}", flush=True)
    if (i + 1) % 50 == 0:
        _write(out)
        print(f"  max_pct {i+1}/{len(todo_pct)} · {time.time()-t0:.0f}s", flush=True)

_write(out)
have = sum(1 for v in out.values() if v.get("pcv_M") is not None)
print(f"WROTE {OUT.name}: {have}/{len(players)} players with pcv, "
      f"{len(contract_end)} contract_end, model {MODEL_STAMP or '?'}, "
      f"{n_err} errors, {time.time()-t0:.0f}s", flush=True)
