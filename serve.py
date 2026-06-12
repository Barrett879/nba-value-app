"""Production entrypoint: warm everything BEFORE the port opens.

Render keeps the previous deploy serving until the new container starts
listening, so every second of import and cache warm-up done here is a second
no visitor ever pays. The old startCommand (`streamlit run app.py`) opened the
port immediately and made the first visitor after every deploy sit through the
whole import chain on a half-CPU box.

`streamlit run app.py` stays the dev path; Render runs `python serve.py`.
"""
import os
import sys
import time

_t0 = time.time()


def _log(msg: str) -> None:
    print(f"[serve] {time.time() - _t0:5.1f}s  {msg}", flush=True)


# ── 1. Pre-import the heavy modules into sys.modules ─────────────────────────
# Streamlit's script runner imports app code in THIS process, so anything
# loaded here is already hot when the first session runs.
_log("pre-importing heavy modules")
import pandas  # noqa: F401,E402
import plotly.express  # noqa: F401,E402  (chart pages use it)
import joblib  # noqa: F401,E402
import sklearn.ensemble  # noqa: F401,E402  (contract model deps)
import nba_api.stats.endpoints  # noqa: F401,E402  (~1s: every endpoint module)
import nba_api.stats.static.players  # noqa: F401,E402
import nba_api.stats.static.teams  # noqa: F401,E402
_log("heavy modules in sys.modules")

import utils  # noqa: E402  (seeds /data/cache from the repo copies on first boot)
_log("utils imported, disk cache seeded")

# ── 2. Warm the current season synchronously ──────────────────────────────────
# Bare mode means build_raw is fresh-or-block, so the parquet and supporting
# caches are hot AND current the moment traffic flips to this container.
for _label, _fn in [
    ("rankings frame", lambda: utils.build_ranked_projected(utils.SEASONS[0])),
    ("bref positions", lambda: utils.fetch_bref_positions(
        utils.season_to_espn_year(utils.SEASONS[0]), cache_v=3)),
    ("next-year contracts", lambda: utils.fetch_next_year_contracts(
        utils.season_to_espn_year(utils.SEASONS[0]), cache_v=7)),
    ("rookie scale", lambda: utils.fetch_rookie_scale_players(utils.SEASONS[0])),
    ("d-lebron", lambda: utils.fetch_dlebron(utils.SEASONS[0])),
    ("player name index", utils.get_all_player_names),
]:
    try:
        _fn()
        _log(f"warm: {_label}")
    except Exception as _e:  # a failed warm should never block serving
        _log(f"warm FAILED ({_label}): {_e}")

# ── 3. Open the port (traffic flips to this container now) ───────────────────
_log("starting streamlit, port opens next")
from streamlit.web.cli import main  # noqa: E402

_port = os.environ.get("PORT", "8501")
sys.argv = [
    "streamlit", "run", "app.py",
    "--server.port", _port,
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
]
sys.exit(main())
