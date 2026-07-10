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

import seo_selfheal  # noqa: E402  (canonical SEO transform, shared with the build patch)

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
    # Every season's raw frame + rankings — the career loop inside the Contract
    # Predictor's feature builder touches all of them; warming here means the
    # FIRST prediction after a deploy doesn't pay ~30-60s of cold compute.
    ("all-season frames", lambda: utils.build_all_seasons_combined(min_threshold=0)),
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
        def _run(state, label, budget):
            bm = BackMsg()
            bm.rerun_script.CopyFrom(state)
            ws.send_binary(bm.SerializeToString())
            end = time.time() + budget
            while time.time() < end:
                raw = ws.recv()
                if isinstance(raw, (bytes, bytearray)):
                    fm = ForwardMsg()
                    fm.ParseFromString(raw)
                    if fm.WhichOneof("type") == "script_finished":
                        return True
            return False

        _run(ClientState(), "home", 120)
        _log(f"self-warm home finished in {time.time() - t:.1f}s")
        # One throwaway prediction so the first real predictor user lands warm:
        # primes the model, comp pool, and the 50-season career path in-page.
        t2 = time.time()
        stc = ClientState()
        stc.page_name = "Contract_Predictor"
        stc.query_string = "player=LeBron%20James"
        ok = _run(stc, "predictor", 180)
        _log(f"self-warm predictor {'finished' if ok else 'TIMED OUT'} in {time.time() - t2:.1f}s")
        ws.close()
    except Exception as e:
        _log(f"self-warm failed (visitors just get the old cold first run): {e}")


threading.Thread(target=_self_warm, daemon=True).start()

# ── 2.9 SEO: patch Streamlit's static index.html ─────────────────────────────
# Streamlit serves a generic SPA shell (<title>Streamlit</title>, no meta
# description, and no body text; content arrives later over the websocket) so
# Googlebot sees a blank page that never says "HoopsValue", and the site is
# invisible to search. The canonical transform lives in seo_selfheal.py; here
# we apply it to the file on disk AND install an in-process Tornado fallback.
# Best-effort + idempotent: a failure (e.g. read-only site-packages) just logs
# and serving continues unaffected. Changes nothing a JS-browser visitor sees.
def _patch_seo() -> None:
    """Write the patched index.html to disk (works where site-packages is writable)."""
    if seo_selfheal.ensure_seo_patched():
        _log("SEO: index.html on disk is patched")
    else:
        _log("SEO disk patch skipped (read-only or unexpected shell)")


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
        data = seo_selfheal.seo_html(idx.read_text(encoding="utf-8")).encode("utf-8")
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


# ── 2.95 Real /robots.txt and /sitemap.xml + duplicate-host noindex ──────────
# Streamlit's catch-all StaticFileHandler answers ANY unknown path with the SPA
# shell, so /robots.txt and /sitemap.xml returned HTML. Streamlit builds its
# route table internally, so we wrap tornado.web.Application's constructor
# (only Streamlit constructs one in this process) to prepend our two handlers
# ahead of the catch-all. The same hook appends a header-only transform that
# marks responses on the duplicate *.onrender.com host with X-Robots-Tag:
# noindex, so Google folds the origin host into hoopsvalue.com. Header only:
# a redirect there would break the websocket handshake through Render.
_SITEMAP_PATHS = [
    "/", "/Rankings", "/Search", "/Legacy", "/Team_Analysis",
    "/Contract_Predictor", "/Free_Agent_Class", "/About",
]


def _install_extra_routes() -> None:
    try:
        import tornado.web

        class _RobotsHandler(tornado.web.RequestHandler):
            def get(self):
                self.set_header("Content-Type", "text/plain; charset=utf-8")
                self.write("User-agent: *\nAllow: /\n\n"
                           "Sitemap: https://hoopsvalue.com/sitemap.xml\n")

            head = get  # crawlers HEAD these; tornado drops the body itself

        class _SitemapHandler(tornado.web.RequestHandler):
            def get(self):
                urls = "".join(
                    f"<url><loc>https://hoopsvalue.com{p}</loc></url>"
                    for p in _SITEMAP_PATHS)
                self.set_header("Content-Type", "application/xml; charset=utf-8")
                self.write('<?xml version="1.0" encoding="UTF-8"?>'
                           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                           f"{urls}</urlset>")

            head = get

        class _NoindexTransform:
            """Tornado output transform: add X-Robots-Tag: noindex when the
            request came in on the *.onrender.com origin host. Skips websocket
            handshakes (101) and touches nothing else."""

            def __init__(self, request):
                host = (request.headers.get("Host") or "").split(":")[0].lower()
                self._noindex = host.endswith(".onrender.com")

            def transform_first_chunk(self, status_code, headers, chunk, finishing):
                if self._noindex and status_code != 101:
                    headers["X-Robots-Tag"] = "noindex"
                return status_code, headers, chunk

            def transform_chunk(self, chunk, finishing):
                return chunk

        _orig_app_init = tornado.web.Application.__init__

        def _app_init(self, handlers=None, *args, **kwargs):
            if handlers:
                handlers = [
                    (r"/robots\.txt", _RobotsHandler),
                    (r"/sitemap\.xml", _SitemapHandler),
                ] + list(handlers)
            _orig_app_init(self, handlers, *args, **kwargs)
            self.transforms.append(_NoindexTransform)
        tornado.web.Application.__init__ = _app_init
        _log("extra routes installed: /robots.txt /sitemap.xml + onrender noindex header")
    except Exception as e:  # never block serving over SEO routes
        _log(f"extra routes skipped: {e}")


_patch_seo()             # disk write (works where site-packages is writable)
_patch_seo_inprocess()   # Tornado override (works on read-only runtime fs too)
_install_extra_routes()  # robots/sitemap + duplicate-host noindex header

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
