"""Production entrypoint: open the port FAST, warm in the background.

The live Render service has a persistent disk attached (/data), and a disk can
mount to only ONE instance at a time, so Render cannot do zero-downtime
deploys here: the old container is stopped BEFORE the new one starts. Every
second this file spends before the port opens is therefore a second of hard
downtime on every deploy -- the opposite of the original design assumption
(pre-port warming is free only on disk-less services, where the old deploy
keeps serving until the new port opens). Barrett hit a ~30s dead window
mid-deploy on 2026-07-15; that was this file warming with the port closed.

So the boot order is inverted: install the SEO/serving patches (milliseconds),
open the port (~2-3s), and do ALL warming in one background thread -- heavy
imports, the /data seed, a synthetic first session that races human clicks,
then an idempotent cache sweep. A visitor who lands mid-warm gets the app
shell with a spinner instead of a dead site.

`streamlit run app.py` stays the dev path; Render runs `python serve.py`.
"""
import os
import sys
import threading
import time

_t0 = time.time()


def _log(msg: str) -> None:
    print(f"[serve] {time.time() - _t0:5.1f}s  {msg}", flush=True)


_port = os.environ.get("PORT", "8501")

import seo_selfheal  # noqa: E402  (imports only streamlit, which the CLI needs anyway)


# ── 2.5 Self-warm the first session the instant the port opens ───────────────
# Streamlit's runtime caches only fill on a real session, so the first visitor
# after a deploy paid the whole first script run (~11s on Render). This daemon
# thread waits for our own port, then drives one synthetic websocket session
# (the same handshake the frontend sends; XSRF is disabled in config.toml).
# Render flips traffic on port-open, so this races ahead of any human click
# and they land on warm runtime caches instead.
def _self_warm(stage: str, budget: int = 120) -> None:
    """Drive one synthetic websocket session so a real visitor lands on warm
    Streamlit runtime caches. stage: "home" or "predictor"."""
    import socket

    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", int(_port)), timeout=1):
                break
        except OSError:
            time.sleep(0.25)
    else:
        _log(f"self-warm {stage}: port never opened")
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
            timeout=budget,
        )
        state = ClientState()
        if stage == "predictor":
            # One throwaway prediction so the first real predictor user lands
            # warm: primes the model, comp pool, and the career path in-page.
            state.page_name = "Contract_Predictor"
            state.query_string = "player=LeBron%20James"
        bm = BackMsg()
        bm.rerun_script.CopyFrom(state)
        ws.send_binary(bm.SerializeToString())
        ok = False
        end = time.time() + budget
        while time.time() < end:
            raw = ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                fm = ForwardMsg()
                fm.ParseFromString(raw)
                if fm.WhichOneof("type") == "script_finished":
                    ok = True
                    break
        ws.close()
        _log(f"self-warm {stage} {'finished' if ok else 'TIMED OUT'} in {time.time() - t:.1f}s")
    except Exception as e:
        _log(f"self-warm {stage} failed (visitors just get the old cold first run): {e}")


def _background_warm() -> None:
    """All the heavy boot work, off the port-open path. Order matters: heavy
    module pre-imports (so the first script run doesn't pay them), utils import
    (seeds /data/cache on first boot), the synthetic warm sessions (race any
    human click through the real page code paths), then an idempotent data-cache
    sweep for anything the two warmed pages didn't touch. Any failure logs and
    moves on -- a failed warm must never take the site down."""
    try:
        _log("bg: pre-importing heavy modules")
        import pandas  # noqa: F401
        import plotly.express  # noqa: F401  (chart pages use it)
        import joblib  # noqa: F401
        import sklearn.ensemble  # noqa: F401  (contract model deps)
        import nba_api.stats.endpoints  # noqa: F401  (~1s: every endpoint module)
        import nba_api.stats.static.players  # noqa: F401
        import nba_api.stats.static.teams  # noqa: F401
        import utils
        _log("bg: utils imported, disk cache seeded")
    except Exception as e:
        _log(f"bg: import warm failed ({e}); sessions will import cold")
        return
    # Home session first (most visitors land there), then the data sweep does
    # the heavy compute directly, then the predictor session -- which is fast
    # once the sweep has built the all-season frames it depends on.
    _self_warm("home")
    for _label, _fn in [
        ("rankings frame", lambda: utils.build_ranked_projected(utils.SEASONS[0])),
        ("bref positions", lambda: utils.fetch_bref_positions(
            utils.season_to_espn_year(utils.SEASONS[0]), cache_v=3)),
        ("next-year contracts", lambda: utils.fetch_next_year_contracts(
            utils.season_to_espn_year(utils.SEASONS[0]), cache_v=7)),
        ("rookie scale", lambda: utils.fetch_rookie_scale_players(utils.SEASONS[0])),
        ("d-lebron", lambda: utils.fetch_dlebron(utils.SEASONS[0])),
        ("player name index", utils.get_all_player_names),
        ("all-season frames", lambda: utils.build_all_seasons_combined(min_threshold=0)),
    ]:
        try:
            _fn()
            _log(f"bg sweep: {_label}")
        except Exception as e:
            _log(f"bg sweep FAILED ({_label}): {e}")
    _self_warm("predictor", budget=180)
    _log("bg: warm complete")


threading.Thread(target=_background_warm, daemon=True).start()

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
        import pathlib
        import tornado.web
        import streamlit as _st

        # Base homepage shell served for "/" — identical to what the in-process
        # StaticFileHandler patch returns, so human visitors get the same booting
        # SPA. Precomputed once; per-request work is only the OG-tag swap.
        _base_shell = None
        try:
            _idx = (pathlib.Path(_st.__file__).parent / "static" / "index.html").resolve()
            _cand = seo_selfheal.seo_html(_idx.read_text(encoding="utf-8"))
            if "<title>HoopsValue" in _cand and "</html>" in _cand.lower():
                _base_shell = _cand
        except Exception as _e:
            _log(f"root player-OG handler: base shell unavailable ({_e})")

        # Favicon / app-icon assets served from ROOT paths with correct
        # content-types. Streamlit serves /app/static/*.svg as text/plain, which
        # Google and iOS reject; these handlers return the right type so the
        # icon actually shows instead of a generic fallback.
        _STATIC = pathlib.Path(__file__).parent / "static"
        _ASSETS = {
            "/favicon.ico": ("favicon.ico", "image/x-icon"),
            "/favicon.svg": ("favicon.svg", "image/svg+xml"),
            "/favicon-32.png": ("favicon-32.png", "image/png"),
            "/favicon-16.png": ("favicon-16.png", "image/png"),
            "/apple-touch-icon.png": ("apple-touch-icon.png", "image/png"),
            "/apple-touch-icon-precomposed.png": ("apple-touch-icon.png", "image/png"),
            "/icon-192.png": ("icon-192.png", "image/png"),
            "/icon-512.png": ("icon-512.png", "image/png"),
            "/site.webmanifest": ("site.webmanifest", "application/manifest+json"),
        }

        class _AssetHandler(tornado.web.RequestHandler):
            def initialize(self, fname, ctype):
                self._fname = fname
                self._ctype = ctype

            def get(self):
                try:
                    data = (_STATIC / self._fname).read_bytes()
                except Exception:
                    self.set_status(404)
                    return
                self.set_header("Content-Type", self._ctype)
                # short TTL: the icon is still being iterated, and Cloudflare
                # honors this at the edge — a long max-age pins a stale icon.
                self.set_header("Cache-Control", "public, max-age=3600")
                self.write(data)

            head = get

        class _RobotsHandler(tornado.web.RequestHandler):
            def get(self):
                self.set_header("Content-Type", "text/plain; charset=utf-8")
                self.write("User-agent: *\nAllow: /\n\n"
                           "Sitemap: https://hoopsvalue.com/sitemap.xml\n")

            head = get  # crawlers HEAD these; tornado drops the body itself

        class _RootHandler(tornado.web.RequestHandler):
            """Serve the homepage shell, swapping in the shared player's OG/title
            tags when the URL carries a single known ?player=. Humans still get
            the identical booting SPA; crawlers and link unfurlers read that
            player's Barrett Score and contract instead of the generic card."""

            def get(self):
                names = self.get_arguments("player")  # decoded; [] when absent
                out = _base_shell
                if len(names) == 1:  # a head-to-head (?player=A&player=B) stays generic
                    out = seo_selfheal.player_shell(_base_shell, names[0])
                self.set_header("Content-Type", "text/html; charset=utf-8")
                self.set_header("Cache-Control", "no-cache")
                self.write(out)

            head = get

        # Team pages are PARKED (Barrett, 2026-07-15): the feature is built and the
        # data pipeline works, but it's on the backburner. With the flag off the
        # route 404s (an explicit 404, so any crawled URL drops cleanly out of the
        # index) and the sitemap excludes the /team/ entries. To re-enable, set
        # HV_TEAM_PAGES=1 in the environment (Render dashboard or local shell) --
        # no code change needed. utils.team_cell reads the same flag for links.
        _team_pages_on = os.environ.get("HV_TEAM_PAGES", "").strip() == "1"

        class _TeamHandler(tornado.web.RequestHandler):
            """Serve a crawlable per-team value page at /team/<ABBR>. Unknown
            abbreviations 404 (not the SPA shell) so junk paths stay out of the
            index. A real HTTP 200 page here, not the JS app, is what Google
            reads and ranks for '<team> player value' searches."""

            def get(self, abbr):
                page = seo_selfheal.team_page_html(abbr) if _team_pages_on else None
                if not page:
                    self.set_status(404)
                    self.set_header("Content-Type", "text/html; charset=utf-8")
                    self.write('<!doctype html><meta charset="utf-8"><title>Not found'
                               '</title><p>No such team. <a href="/">HoopsValue home'
                               "</a></p>")
                    return
                self.set_header("Content-Type", "text/html; charset=utf-8")
                self.set_header("Cache-Control", "public, max-age=3600")
                self.write(page)

            head = get

        class _SitemapHandler(tornado.web.RequestHandler):
            def get(self):
                paths = list(_SITEMAP_PATHS)
                if _team_pages_on:
                    try:  # one entry per crawlable team page
                        paths += [f"/team/{a}"
                                  for a in seo_selfheal.team_pages().get("teams", {})]
                    except Exception:
                        pass
                urls = "".join(
                    f"<url><loc>https://hoopsvalue.com{p}</loc></url>"
                    for p in paths)
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
                extra = [
                    (r"/robots\.txt", _RobotsHandler),
                    (r"/sitemap\.xml", _SitemapHandler),
                    (r"/team/([A-Za-z]{2,4})", _TeamHandler),
                ]
                # favicon/app-icon assets on their canonical root paths
                # (escape the dots; these paths carry no other regex metachars)
                for _p, (_f, _c) in _ASSETS.items():
                    extra.append((_p.replace(".", r"\."), _AssetHandler,
                                  dict(fname=_f, ctype=_c)))
                if _base_shell:  # exact "/" only; every other path stays Streamlit's
                    extra.append((r"/", _RootHandler))
                handlers = extra + list(handlers)
            _orig_app_init(self, handlers, *args, **kwargs)
            self.transforms.append(_NoindexTransform)
        tornado.web.Application.__init__ = _app_init
        _log("extra routes installed: /robots.txt /sitemap.xml /team/*"
             + (" / (player OG)" if _base_shell else "")
             + " + onrender noindex header")
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
