"""Build cache/player_hub_pcv_v1.json — the Contract Predictor's model projection
(pcv) + calibrated 80% band for EVERY current-pool player.

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

OUT = ROOT / "cache" / "player_hub_pcv_v1.json"

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

# Resume: keep whatever a previous (interrupted) run already computed.
out: dict = {}
if OUT.exists():
    try:
        out = json.loads(OUT.read_text()).get("players", {})
        print(f"  resuming with {len(out)} already computed", flush=True)
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
            out[normalize(name)] = {
                "player": name,
                "pcv_M": round(v, 1),
                "low_M": round(max(0.0, v - b), 1),
                "high_M": round(v + b, 1),
            }
        else:
            out[normalize(name)] = {"player": name, "pcv_M": None}
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        n_err += 1
        print(f"  ERR {name}: {type(e).__name__} {str(e)[:60]}", flush=True)
    if (i + 1) % 25 == 0:
        OUT.write_text(json.dumps({"season": utils.SEASONS[0], "players": out}, indent=0))
        print(f"  {i+1}/{len(todo)} · {len(out)} cached · {n_err} errs · {time.time()-t0:.0f}s", flush=True)

OUT.write_text(json.dumps({"season": utils.SEASONS[0], "players": out}, indent=0))
have = sum(1 for v in out.values() if v.get("pcv_M") is not None)
print(f"WROTE {OUT.name}: {have}/{len(players)} players with pcv, {n_err} errors, {time.time()-t0:.0f}s", flush=True)
