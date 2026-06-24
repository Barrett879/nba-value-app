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
import threading
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

_port = os.environ.get("PORT", "8501")


# ── 2.5 Self-warm the first session the instant the port opens ───────────────
# Streamlit's runtime caches only fill on a real session, so the first visitor
# after a deploy paid the whole first script run (~11s on Render). This daemon
# thread waits for our own port, then drives one synthetic websocket session
# (the same handshake the frontend sends; XSRF is disabled in config.toml).
# Render flips traffic on port-open, so this races ahead of any human click
# and they land on warm runtime caches instead.
def _self_warm() -> None:
    import socket

    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", int(_port)), timeout=1):
                break
        except OSError:
            time.sleep(0.25)
    else:
        _log("self-warm: port never opened")
        return
    try:
        import websocket
        from streamlit.proto.BackMsg_pb2 import BackMsg
        from streamlit.proto.ClientState_pb2 import ClientState
        from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

        t = time.time()
        ws = websocket.create_connection(
            f"ws://127.0.0.1:{_port}/_stcore/stream",
            subprotocols=["streamlit", "PLACEHOLDER_AUTH_TOKEN"],
            timeout=120,
        )
        bm = BackMsg()
        bm.rerun_script.CopyFrom(ClientState())
        ws.send_binary(bm.SerializeToString())
        end = time.time() + 120
        while time.time() < end:
            raw = ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                fm = ForwardMsg()
                fm.ParseFromString(raw)
                if fm.WhichOneof("type") == "script_finished":
                    break
        ws.close()
        _log(f"self-warm session finished in {time.time() - t:.1f}s")
    except Exception as e:
        _log(f"self-warm failed (visitors just get the old cold first run): {e}")


threading.Thread(target=_self_warm, daemon=True).start()

# ── 2.9 SEO: patch Streamlit's static index.html ─────────────────────────────
# Streamlit serves a generic SPA shell — <title>Streamlit</title>, no meta
# description, and no body text (content arrives later over the websocket) — so
# Googlebot sees a blank page that never says "HoopsValue", and the site is
# invisible to search. Inject a real title, meta/OpenGraph tags, and a <noscript>
# content block with internal links into the file Streamlit serves at "/".
# Best-effort + idempotent: a failure (e.g. read-only site-packages) just logs
# and serving continues unaffected. Changes nothing a JS-browser visitor sees.
def _seo_html(html: str) -> str:
    """Inject a real <title>, meta/OpenGraph tags, and a crawlable <noscript> block
    (replacing Streamlit's default 'enable JavaScript' one) into the shell HTML."""
    import re as _re
    if "hv-seo-v1" in html:
        return html
    title = "HoopsValue · NBA Player Value, Contract Predictions & Rankings"
    desc = ("HoopsValue ranks every NBA player by the Barrett Score — on-court "
            "production measured against their paycheck — and predicts what any "
            "player would sign for today. Find the steals, expose the overpays, and "
            "run any team's free agency.")
    head = (
        "<!--hv-seo-v1-->"
        f'<meta name="description" content="{desc}"/>'
        '<meta name="robots" content="index, follow"/>'
        '<link rel="canonical" href="https://hoopsvalue.com/"/>'
        f'<meta property="og:title" content="{title}"/>'
        f'<meta property="og:description" content="{desc}"/>'
        '<meta property="og:type" content="website"/>'
        '<meta property="og:url" content="https://hoopsvalue.com/"/>'
        '<meta property="og:image" content="https://hoopsvalue.com/app/static/hoopsvalue_logo.png"/>'
        '<meta name="twitter:card" content="summary_large_image"/>'
    )
    nav = "".join(
        f'<li><a href="/{slug}">{label}</a></li>' for slug, label in [
            ("Contract_Predictor", "Contract Predictor — what any player would sign for today"),
            ("Rankings", "Current Rankings — every NBA player by Barrett Score"),
            ("Team_Builder", "Front Office — run a team's offseason"),
            ("Free_Agency_Simulation", "Free Agency Simulation"),
            ("Search", "Player Search"),
            ("Legacy", "Legacy — the best players ever by Barrett Score"),
        ])
    body = (
        "<noscript><header><h1>HoopsValue</h1>"
        "<p>NBA player value, contract predictions, and rankings.</p></header>"
        f"<main><p>{desc}</p><ul>{nav}</ul></main></noscript>"
    )
    if "<title>Streamlit</title>" in html:
        html = html.replace("<title>Streamlit</title>", f"<title>{title}</title>{head}")
    else:
        html = html.replace("</head>", f"<title>{title}</title>{head}</head>", 1)
    if "<noscript" in html:
        html = _re.sub(r"<noscript>.*?</noscript>", body, html, count=1, flags=_re.S)
    else:
        html = html.replace("<body>", f"<body>{body}", 1)
    return html


def _patch_seo() -> None:
    """Write the patched index.html to disk (works where site-packages is writable)."""
    try:
        import pathlib
        import streamlit as _st
        idx = pathlib.Path(_st.__file__).parent / "static" / "index.html"
        html = idx.read_text(encoding="utf-8")
        if "hv-seo-v1" not in html:
            idx.write_text(_seo_html(html), encoding="utf-8")
            _log("SEO: patched index.html on disk")
    except Exception as e:
        _log(f"SEO disk patch skipped: {e}")


def _patch_seo_inprocess() -> None:
    """Inject SEO in-memory by overriding Tornado's StaticFileHandler for index.html
    ONLY — works even when Render's runtime site-packages are read-only (the disk
    write silently no-ops there). Every other asset falls through untouched. Both
    get_content and get_content_size are overridden so Content-Length stays correct."""
    try:
        import pathlib
        import tornado.web
        import streamlit as _st
        idx = (pathlib.Path(_st.__file__).parent / "static" / "index.html").resolve()
        data = _seo_html(idx.read_text(encoding="utf-8")).encode("utf-8")
        if b"<title>HoopsValue" not in data or b"</html>" not in data.lower():
            _log("SEO in-process: transform looked invalid, skipping")
            return

        def _is_index(p):
            return str(p).replace("\\", "/").endswith("static/index.html")

        _orig_gc = tornado.web.StaticFileHandler.get_content.__func__

        def _gc(cls, abspath, start=None, end=None):
            if _is_index(abspath):
                s = 0 if start is None else start
                e = len(data) if end is None else end
                return data[s:e]
            return _orig_gc(cls, abspath, start, end)
        tornado.web.StaticFileHandler.get_content = classmethod(_gc)

        _orig_sz = tornado.web.StaticFileHandler.get_content_size

        def _sz(self):
            if _is_index(getattr(self, "absolute_path", "") or ""):
                return len(data)
            return _orig_sz(self)
        tornado.web.StaticFileHandler.get_content_size = _sz
        _log("SEO: in-process index.html injection installed")
    except Exception as e:  # never block serving over an SEO patch
        _log(f"SEO in-process patch skipped: {e}")


_patch_seo()            # disk write (works where site-packages is writable)
_patch_seo_inprocess()  # Tornado override (works on read-only runtime fs too)

# ── 3. Open the port (traffic flips to this container now) ───────────────────
_log("starting streamlit, port opens next")
from streamlit.web.cli import main  # noqa: E402
sys.argv = [
    "streamlit", "run", "app.py",
    "--server.port", _port,
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
]
sys.exit(main())
