"""Self-healing SEO patch for Streamlit's served index.html.

Streamlit serves a generic SPA shell (<title>Streamlit</title>, no meta
description, a "You need to enable JavaScript" noscript, and Streamlit's own
favicon), so crawlers index a blank, mis-titled page. This module holds the
one canonical transform and applies it to the index.html file inside the
installed streamlit package:

  - real <title> + meta description/robots/canonical + OpenGraph/Twitter tags
  - a crawlable <noscript> block with internal links
  - the favicon <link> rewritten to the HoopsValue icon (/app/static/favicon.svg,
    served by enableStaticServing; app.py's page_icon swaps it at runtime for
    JS visitors, this covers crawlers and the pre-boot shell)

Consumers:
  - serve.py (production entrypoint) calls ensure_seo_patched() at startup and
    reuses seo_html() for its in-process Tornado fallback
  - scripts/seo_patch.py calls ensure_seo_patched() at Render build time
  - app.py can call ensure_seo_patched() so even a bare `streamlit run app.py`
    (e.g. a dashboard start command that predates serve.py) self-heals: an
    already-running Streamlit serves the updated file on the very next request

ensure_seo_patched() is idempotent (marker check) and NEVER raises: any
failure (read-only site-packages, unexpected shell markup) is a silent no-op
so it can sit on the request path without ever taking the site down.
"""
import html as _html
import json
import pathlib
import re
import unicodedata

import streamlit

MARKER = "hv-seo-v8"
_OLD_MARKERS = ("hv-seo-v1", "hv-seo-v2", "hv-seo-v3", "hv-seo-v4",
                "hv-seo-v5", "hv-seo-v6", "hv-seo-v7")

TITLE = "HoopsValue · NBA Player Value, Contract Predictions & Rankings"
DESC = ("HoopsValue scores every NBA player since 1973 by their on-court value "
        "and holds it up against their salary, so you can see who's actually "
        "worth their contract and who isn't.")
# Crawlable body copy for the no-JS shell: this is most of the text Google reads
# on a Streamlit SPA, so it carries the value prop in natural language (not
# keyword stuffing) covering the searches the site actually answers.
_BODY_COPY = (
    "HoopsValue, sometimes written Hoops Value, rates every NBA player with the "
    "Barrett Score, a single value "
    "metric built from scoring, playmaking, rebounding, defense, efficiency, and "
    "availability, then holds it up against the player's salary to reveal who is "
    "underpaid, overpaid, or paid about right. Browse current NBA player rankings "
    "by value, predict any player's next contract, follow the 2026 NBA free agency "
    "class and free-agent signings, compare any two players head to head, rank "
    "every team's roster by value, and trace player value and salaries back to 1973."
)
_BODY_Q = (
    "HoopsValue helps answer questions like: which NBA players are the most "
    "underpaid and overpaid, what a player is worth on the open market, how much a "
    "free agent will sign for, and who offers the best contract value in the NBA."
)
PAGES = [
    ("Contract_Predictor", "Contract Predictor: what any NBA player would sign for today"),
    ("Rankings", "Current Rankings: every NBA player ranked by Barrett Score value"),
    ("Free_Agent_Class", "2026 NBA Free Agent Class: who is available and what they are worth"),
    ("Rosters", "NBA Rosters 2026-27: every team's contracts, cap sheet and value"),
    ("Team_Analysis", "Team Analysis: every NBA roster ranked by value"),
    ("Search", "Compare Players: head-to-head NBA player value"),
    ("Legacy", "Legacy: the best NBA players ever by Barrett Score"),
    ("About", "About HoopsValue and the Barrett Score"),
]
# Full icon set on ROOT paths (served with correct content-types by serve.py;
# Streamlit mis-serves .svg as text/plain, which crawlers/Google reject). The
# .ico covers Google + legacy, the SVG covers modern tabs, the PNGs cover the
# rest, apple-touch covers iOS, and the manifest covers PWA/Android.
# The ?v= suffix is a cache-buster: Cloudflare edge-caches these assets, so a
# bump here forces a fresh fetch of a changed icon instead of a stale one.
# Bump the number whenever the icon art changes.
_ICON_V = "2"
FAVICON_TAG = (
    f'<link rel="icon" href="/favicon.ico?v={_ICON_V}" sizes="any"/>'
    f'<link rel="icon" type="image/svg+xml" href="/favicon.svg?v={_ICON_V}"/>'
    f'<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png?v={_ICON_V}"/>'
    f'<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16.png?v={_ICON_V}"/>'
    f'<link rel="apple-touch-icon" href="/apple-touch-icon.png?v={_ICON_V}"/>'
    '<link rel="manifest" href="/site.webmanifest"/>'
)

# Purpose-built 1200x630 share card (link unfurls want a wide image, not the
# raw square logo). Bundled together so the fresh-patch and older-marker upgrade
# paths stay in sync.
OG_IMAGE = "https://hoopsvalue.com/app/static/og_card.png"
_OG_IMAGE_TAGS = (
    f'<meta property="og:image" content="{OG_IMAGE}"/>'
    '<meta property="og:image:width" content="1200"/>'
    '<meta property="og:image:height" content="630"/>'
)
_TWITTER_IMAGE_TAG = f'<meta name="twitter:image" content="{OG_IMAGE}"/>'

SITE = "https://hoopsvalue.com"

# Structured data. Two jobs: tell Google this is one named site with a search
# action (so "hoopsvalue" resolves to an entity rather than a string), and
# declare the two-word spelling as an alternateName, because nothing on the
# site ever writes "Hoops Value" and that is how people type it.
_SITE_JSONLD = {
    "@context": "https://schema.org",
    "@graph": [
        {
            "@type": "WebSite",
            "@id": f"{SITE}/#website",
            "url": f"{SITE}/",
            "name": "HoopsValue",
            "alternateName": ["Hoops Value", "hoopsvalue", "hoopsvalue.com"],
            "description": DESC,
            "inLanguage": "en-US",
            "publisher": {"@id": f"{SITE}/#org"},
            "potentialAction": {
                "@type": "SearchAction",
                "target": {"@type": "EntryPoint",
                           "urlTemplate": f"{SITE}/?player={{search_term_string}}"},
                "query-input": "required name=search_term_string",
            },
        },
        {
            "@type": "Organization",
            "@id": f"{SITE}/#org",
            "name": "HoopsValue",
            "alternateName": "Hoops Value",
            "url": f"{SITE}/",
            "logo": {"@type": "ImageObject", "url": OG_IMAGE},
            "description": DESC,
        },
    ],
}


def _jsonld_tag(obj) -> str:
    """One <script type="application/ld+json"> for obj. ensure_ascii escapes
    non-ASCII names to their unicode form, and "</" is escaped so no player
    name or copy change can close the script tag early."""
    body = json.dumps(obj, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    return f'<script type="application/ld+json">{body}</script>'


def _add_jsonld(html: str, obj) -> str:
    """Append a JSON-LD block before </head>. Additive: a page can carry the
    site graph from the base shell AND its own entity."""
    return html.replace("</head>", _jsonld_tag(obj) + "</head>", 1)


def _swap_noscript(html: str, body: str) -> str:
    """Replace the shell's crawlable no-JS block with `body`.

    The <noscript> block is most of the text a crawler reads on a Streamlit
    SPA, so a route that does not replace it is a near-duplicate of the
    homepage no matter how good its <title> is. lambda replacement: a player
    name containing a backslash must not be read as a group reference."""
    if "<noscript" not in html:
        return html
    return re.sub(r"<noscript>.*?</noscript>", lambda m: body, html,
                  count=1, flags=re.S)


def _home_body() -> str:
    """The homepage's crawlable no-JS block. One definition, used by both the
    fresh-patch and the marker-upgrade paths."""
    nav = "".join(f'<li><a href="/{slug}">{label}</a></li>' for slug, label in PAGES)
    return (
        f"<noscript><header><h1>{TITLE}</h1>"
        "<p>NBA player value, contract predictions, salary analysis, and rankings.</p>"
        f"</header><main><p>{DESC}</p><p>{_BODY_COPY}</p><p>{_BODY_Q}</p>"
        f"<ul>{nav}</ul></main></noscript>"
    )


def _ordinal(n) -> str:
    try:
        n = int(n)
    except Exception:
        return ""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


_patched_this_process = False  # skip the file read on Streamlit re-runs


def seo_html(html: str) -> str:
    """Return the shell HTML with the SEO injection applied. Idempotent: html
    already carrying the current marker comes back unchanged; html patched by
    an older version is upgraded in place (marker bump + favicon rewrite)."""
    if MARKER in html:
        return html
    if any(m in html for m in _OLD_MARKERS):
        # Head tags + noscript already injected by an older patch: bump the
        # marker, repoint the OG image at the wide share card, and apply the
        # newer favicon rewrite below.
        for m in _OLD_MARKERS:
            html = html.replace(f"<!--{m}-->", f"<!--{MARKER}-->", 1)
        # Older patches pointed og:image at the raw square logo. Swap that lone
        # tag for the share card plus its dimension hints (v1/v2 never carried
        # og:image:width/height, so this never double-injects).
        html = re.sub(r'<meta property="og:image"[^>]*/?>', _OG_IMAGE_TAGS, html, count=1)
        if 'name="twitter:image"' not in html:
            html = html.replace(
                '<meta name="twitter:card" content="summary_large_image"/>',
                '<meta name="twitter:card" content="summary_large_image"/>'
                + _TWITTER_IMAGE_TAG,
                1,
            )
    else:
        head = (
            f"<!--{MARKER}-->"
            f'<meta name="description" content="{DESC}"/>'
            '<meta name="robots" content="index, follow"/>'
            '<link rel="canonical" href="https://hoopsvalue.com/"/>'
            f'<meta property="og:title" content="{TITLE}"/>'
            f'<meta property="og:description" content="{DESC}"/>'
            '<meta property="og:type" content="website"/>'
            '<meta property="og:url" content="https://hoopsvalue.com/"/>'
            f'{_OG_IMAGE_TAGS}'
            '<meta name="twitter:card" content="summary_large_image"/>'
            f'{_TWITTER_IMAGE_TAG}'
        )
        body = _home_body()
        if "<title>Streamlit</title>" in html:
            html = html.replace("<title>Streamlit</title>", f"<title>{TITLE}</title>{head}")
        else:
            html = html.replace("</head>", f"<title>{TITLE}</title>{head}</head>", 1)
        if "<noscript" in html:
            html = re.sub(r"<noscript>.*?</noscript>", body, html, count=1, flags=re.S)
        else:
            html = html.replace("<body>", f"<body>{body}", 1)
    # Swap the favicon for the HoopsValue icon set (crawlers and the pre-boot
    # shell; the running app swaps it again via page_icon). Strip Streamlit's
    # icon link and any icon/apple/manifest links an older patch left, then
    # inject ours once so re-runs and marker upgrades never double up.
    html = re.sub(
        r'<link[^>]*rel="(?:shortcut icon|icon|apple-touch-icon|manifest)"[^>]*/?>',
        "", html, flags=re.I)
    if 'rel="apple-touch-icon"' not in html:
        html = html.replace("</head>", FAVICON_TAG + "</head>", 1)
    # Outside the fresh/upgrade branches on purpose: a shell already carrying an
    # older marker takes the upgrade path, which injects no head tags, so a
    # JSON-LD block added inside that branch would never reach a locally
    # pre-patched streamlit. Presence check keeps re-runs from doubling it.
    if "application/ld+json" not in html:
        html = _add_jsonld(html, _SITE_JSONLD)
    # The old-marker upgrade path bumped the marker and the og:image and left
    # the BODY exactly as the version that first patched the file wrote it. A
    # shell here was reporting marker v7 over a four-link nav from v1, which
    # means the crawlable copy, the nav and this JSON-LD all silently stopped
    # following the code. The body is generated from constants in this module,
    # so regenerate it every time; a shell already on the current marker
    # returned at the top and is never touched, so this stays idempotent.
    html = _swap_noscript(html, _home_body())
    return html


def ensure_seo_patched() -> bool:
    """Patch the installed streamlit static/index.html on disk if it is not
    already patched. Returns True when the file is (now) patched, False when
    it could not be. Never raises; safe to call on every script run."""
    global _patched_this_process
    if _patched_this_process:
        return True
    try:
        idx = pathlib.Path(streamlit.__file__).parent / "static" / "index.html"
        html = idx.read_text(encoding="utf-8")
        if MARKER in html:
            _patched_this_process = True
            return True
        patched = seo_html(html)
        if f"<title>{TITLE}" not in patched or "</html>" not in patched.lower():
            return False  # transform looked invalid, leave the file alone
        idx.write_text(patched, encoding="utf-8")
        _patched_this_process = True
        return True
    except Exception:
        return False  # e.g. read-only site-packages: silent no-op


# --- Per-player share metadata -------------------------------------------------
# A shared /?player=X link should unfurl with THAT player's Barrett Score and
# contract, not the generic homepage card. serve.py's root handler calls
# player_shell() per request; the lookup table is precomputed by
# scripts/build_share_meta.py (no model work on the request path).
_SHARE_META = None  # lazy {normalized_name: {"t": title, "d": desc}}


def _share_norm(name: str) -> str:
    """Match scripts/build_share_meta.py's key (utils.normalize): NFKD-fold
    accents, lowercase, strip. Kept dependency-free so serve.py stays light."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _share_meta() -> dict:
    global _SHARE_META
    if _SHARE_META is None:
        try:
            p = pathlib.Path(__file__).parent / "cache" / "share_meta.json"
            _SHARE_META = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            _SHARE_META = {}  # cache absent: every lookup is a clean miss
    return _SHARE_META


def share_meta_built_at() -> str:
    """ISO stamp of the share_meta build, or "" for a file that predates it.
    serve.py uses the date as the player URLs' <lastmod>."""
    try:
        return str(_share_meta().get("_meta", {}).get("built_at", ""))
    except Exception:
        return ""


# How many player links the /Rankings no-JS body carries. The sitemap lists
# every player, but a sitemap is a hint and an internal link is a path: without
# one, the only crawlable links to any player page lived on the team pages,
# which are parked and 404. 150 keeps the page's link count sane while covering
# everyone a value search is plausibly about.
_TOP_LINKS_N = 150
_top_links_cache = None


def _top_player_links(n: int = _TOP_LINKS_N) -> str:
    """<li> links to the n best-ranked players, with the team and score so the
    anchor text carries meaning. Memoized: one pass over share_meta per
    process, not per request."""
    global _top_links_cache
    if _top_links_cache is None:
        try:
            esc = lambda x: _html.escape(str(x), quote=True)  # noqa: E731
            recs = [r for k, r in _share_meta().items()
                    if not k.startswith("_") and isinstance(r, dict)
                    and r.get("n") and isinstance(r.get("r"), int)]
            recs.sort(key=lambda r: r["r"])
            out = []
            for r in recs[:n]:
                tail = ""
                if isinstance(r.get("s"), (int, float)):
                    tail = f' ({esc(r.get("tm", ""))}, {r["s"]:.1f} Barrett Score)'
                out.append(f'<li><a href="{esc(player_url(r["n"]))}">'
                           f'{esc(r["n"])}</a>{tail}</li>')
            _top_links_cache = "".join(out)
        except Exception:
            _top_links_cache = ""
    return _top_links_cache


# Per-page title/description for the routes in the sitemap. Without these every
# route served the homepage's tags AND — the real damage — the homepage's
# canonical, which told Google that /Rankings, /Rosters, /Contract_Predictor and
# every ?player= deep link were duplicates of "/". The sitemap asked for nine
# URLs to be indexed while the canonical disclaimed all nine.
_PAGE_META = {
    "/Rankings": (
        "NBA Player Rankings by Value: Barrett Score | HoopsValue",
        "Every NBA player ranked by the Barrett Score and set against his salary, "
        "so you can see who is underpaid, overpaid, or paid about right."),
    "/Search": (
        "Search Any NBA Player's Value and Contract | HoopsValue",
        "Look up any NBA player since 1973: Barrett Score, salary, model value, "
        "and how his production compares to what he is paid."),
    "/Legacy": (
        "NBA Career Value Leaders, 1973 to Today | HoopsValue",
        "Career Barrett Score leaders across five decades of NBA basketball, with "
        "peak seasons, longevity, and value against salary."),
    "/Rosters": (
        "NBA Rosters and 2026-27 Salary Books, All 30 Teams | HoopsValue",
        "The real 2026-27 books team by team: every contract, dead money, trade "
        "exceptions and picks, read against the cap, the tax line and both aprons."),
    "/Team_Analysis": (
        "NBA Team Value Analysis: Payroll vs Production | HoopsValue",
        "Which NBA front offices get the most on-court value per dollar, and which "
        "are paying for production they are not getting."),
    "/Contract_Predictor": (
        "NBA Contract Predictor: What Is a Player Worth? | HoopsValue",
        "Estimate what any NBA player would command on a new contract today, with "
        "comparable signings, likely suitors, and the model's track record."),
    "/Free_Agent_Class": (
        "2026 NBA Free Agent Class: Values and Predictions | HoopsValue",
        "The full 2026 NBA free agent class ranked by value, with predicted "
        "contracts and how each deal has actually landed."),
    "/Redraft": (
        "NBA Redraft: Rebuild Any Roster Under the Cap | HoopsValue",
        "Strip every team status off every player and refill 15 roster spots "
        "under the real 2026-27 cap. Contracts come with the players, and the "
        "bar tracks the floor, cap, tax and both aprons."),
    "/Trade_Machine": (
        "NBA Trade Machine: CBA-Legal Trades | HoopsValue",
        "Build NBA trades and see instantly whether they are legal under the 2023 "
        "CBA: salary matching, aprons, hard caps and trade exceptions."),
    "/About": (
        "About HoopsValue: Methodology and Data Sources",
        "How the Barrett Score is built, how the contract model is trained and "
        "tested, and where the data comes from."),
}


# Crawlable no-JS body per route. Without this every indexable route served
# the HOMEPAGE'S <noscript> block, so ten URLs shipped ten different titles
# over one identical body. On a JS app that block is most of the text Google
# reads, and near-identical bodies are what it folds together as duplicates.
# One <h1> that matches the <title>, two sentences of real description, and the
# shared nav so each route links to the others.
_PAGE_BODY = {
    "/Rankings": (
        "NBA Player Rankings by Value",
        "Every NBA player ranked by the Barrett Score, a single value metric built "
        "from scoring, playmaking, rebounding, defense, efficiency and availability. "
        "Each player's score sits next to his salary and the market value the model "
        "puts on that production.",
        "Sort by value, by salary, or by the gap between them to find who is "
        "underpaid and who is overpaid, filter by minutes played, and split a "
        "traded player's season by the teams he played for."),
    "/Search": (
        "Search Any NBA Player's Value and Contract",
        "Look up any NBA player from 1973 to today and see his Barrett Score, "
        "salary, market value, and how his production compares to what he is paid.",
        "Compare two players head to head across any season, or follow one player's "
        "value through his whole career."),
    "/Legacy": (
        "NBA Career Value Leaders, 1973 to Today",
        "Career Barrett Score leaders across five decades of NBA basketball, with "
        "peak seasons, longevity, and value measured against salary.",
        "Every season since 1973-74 is scored on the same formula, so a 1970s "
        "season and a season from this year sit on one scale."),
    "/Rosters": (
        "NBA Rosters and Salary Books, All 30 Teams",
        "Every team's roster with the real contracts behind it: salary by player, "
        "dead money, trade exceptions and draft picks, read against the cap, the "
        "tax line and both aprons.",
        "Each roster is also valued, so you can see which contracts a team is "
        "winning on and which ones it is carrying."),
    "/Team_Analysis": (
        "NBA Team Value Analysis, Payroll Against Production",
        "Which NBA front offices get the most on-court value per dollar, and which "
        "are paying for production they are not getting.",
        "Every team's payroll is set against the value its players actually "
        "produced, ranked league wide."),
    "/Contract_Predictor": (
        "NBA Contract Predictor",
        "Estimate what any NBA player would command on a new contract today. The "
        "model is trained on real signings and reports comparable deals, likely "
        "suitors, and its own track record.",
        "Change a player's role, minutes or production and watch the predicted "
        "contract move with it."),
    "/Free_Agent_Class": (
        "NBA Free Agent Class, Values and Predictions",
        "The full free agent class ranked by value, with a predicted contract for "
        "each player and, once a deal is signed, how the real one compares.",
        "Projections stay fixed once published, so the model is scored against "
        "what actually happened rather than quietly rewritten."),
    "/Redraft": (
        "NBA Redraft, Rebuild Any Roster Under the Cap",
        "Strip every team status off every player and refill fifteen roster spots "
        "under the real cap. Contracts come with the players, and the bar tracks "
        "the floor, the cap, the tax line and both aprons.",
        "Build the roster you would actually pay for and see what it costs."),
    "/Trade_Machine": (
        "NBA Trade Machine, CBA-Legal Trades",
        "Build NBA trades and see instantly whether they are legal under the 2023 "
        "CBA: salary matching, aprons, hard caps, trade exceptions and "
        "sign-and-trades.",
        "Every trade is also valued, so the legality check and the question of who "
        "wins it are answered side by side."),
    "/About": (
        "About HoopsValue and the Barrett Score",
        "How the Barrett Score is built, how the contract model is trained and "
        "tested, and where the data comes from.",
        "HoopsValue, sometimes written Hoops Value, covers every NBA player and "
        "every season back to 1973-74."),
}


def _nav_html() -> str:
    return "".join(f'<li><a href="/{slug}">{label}</a></li>' for slug, label in PAGES)


def _route_body(path: str) -> str:
    """The <noscript> block for a known route, or "" when there is none."""
    ent = _PAGE_BODY.get(path)
    if not ent:
        return ""
    h1, p1, p2 = ent
    extra = ""
    if path == "/Rankings":  # the crawl path down to the player landing pages
        links = _top_player_links()
        if links:
            extra = (f"<h2>Top {_TOP_LINKS_N} NBA players by value</h2>"
                     f"<ul>{links}</ul>")
    return (f"<noscript><header><h1>{h1}</h1></header><main>"
            f"<p>{p1}</p><p>{p2}</p>{extra}"
            f"<p>More on HoopsValue:</p><ul>{_nav_html()}</ul>"
            '<p><a href="/">HoopsValue home</a></p></main></noscript>')


def page_shell(base_html: str, path: str) -> str:
    """Return base_html with canonical/og:url, and, where known, title,
    description and the crawlable no-JS body, pointed at `path` instead of the
    homepage. Never raises; an unknown path returns base_html untouched."""
    try:
        path = "/" + (path or "").strip("/")
        if path == "/":
            return base_html
        url = "https://hoopsvalue.com" + path
        out = re.sub(r'(<link rel="canonical" href=")[^"]*(")',
                     lambda m: m.group(1) + url + m.group(2), base_html, count=1)
        out = re.sub(r'(<meta property="og:url" content=")[^"]*(")',
                     lambda m: m.group(1) + url + m.group(2), out, count=1)
        meta = _PAGE_META.get(path)
        if meta:
            t, d = _html.escape(meta[0], quote=True), _html.escape(meta[1], quote=True)
            out = re.sub(r"<title>[^<]*</title>", lambda m: f"<title>{t}</title>",
                         out, count=1)
            out = re.sub(r'(<meta name="description" content=")[^"]*(")',
                         lambda m: m.group(1) + d + m.group(2), out, count=1)
            out = re.sub(r'(<meta property="og:title" content=")[^"]*(")',
                         lambda m: m.group(1) + t + m.group(2), out, count=1)
            out = re.sub(r'(<meta property="og:description" content=")[^"]*(")',
                         lambda m: m.group(1) + d + m.group(2), out, count=1)
        body = _route_body(path)
        if body:
            out = _swap_noscript(out, body)
        return out
    except Exception:
        return base_html


def player_url(name: str) -> str:
    """The ONE canonical URL for a player page. Always built from the display
    name in the record, never from the requested string, so /?player=Nikola%20Vucevic
    and /?player=Nikola%20Vu%C4%8Devi%C4%87 both canonicalise to the same URL
    instead of competing as two pages for one player."""
    return f"{SITE}/?player=" + _urlq(name)


def player_urls() -> list:
    """[(display_name, canonical_url)] for every player with a record. Feeds
    serve.py's sitemap: the normalized keys cannot be turned back into a display
    name, so a record without "n" is skipped rather than guessed at (a short
    sitemap beats a sitemap of 404s)."""
    out = []
    for key, rec in _share_meta().items():
        if key.startswith("_") or not isinstance(rec, dict):
            continue
        nm = rec.get("n")
        if nm:
            out.append((nm, player_url(nm)))
    out.sort(key=lambda x: x[0])
    return out


def _verdict(vd):
    """Prose for the salary-against-value gap. vd is $M, positive = overpaid."""
    if not isinstance(vd, (int, float)):
        return ""
    if vd <= -2:
        return f"underpaid by ${-vd:.1f}M"
    if vd >= 2:
        return f"overpaid by ${vd:.1f}M"
    return "paid about what he is worth"


def _player_body(rec: dict, url: str) -> str:
    """The crawlable no-JS block for one player: an <h1> that matches the
    <title>, his actual numbers as a list, and links out.

    Before this, a /?player= URL carried a perfect per-player title and
    description over the HOMEPAGE'S body text, so the only page content Google
    could read never mentioned the player it was titled after."""
    esc = lambda s: _html.escape(str(s), quote=True)  # noqa: E731
    nm = esc(rec.get("n", ""))
    tm = esc(rec.get("tm", ""))
    season = esc(_share_meta().get("_meta", {}).get("season", ""))
    team_name = ""
    try:
        team_name = team_pages().get("teams", {}).get(rec.get("tm", ""), {}).get("name", "")
    except Exception:
        team_name = ""
    team_txt = esc(team_name) if team_name else tm

    items = []
    if isinstance(rec.get("s"), (int, float)):
        rank = _ordinal(rec.get("r"))
        # "sy" is set only when the score came from an EARLIER season than the
        # one the site is scoring, which is how the hub panel shows it too. A
        # fallback number presented as this season's is the same misstatement
        # the stale metadata was making, one row down.
        sy = rec.get("sy")
        label = f"Barrett Score ({esc(sy)})" if sy else "Barrett Score"
        items.append(f"<li>{label}: {rec['s']:.1f}"
                     + (f", {rank} in the NBA by value" if rank else "") + "</li>")
    if isinstance(rec.get("pd"), (int, float)):
        items.append(f"<li>{season} salary: ${rec['pd']:.1f}M</li>".lstrip())
    if isinstance(rec.get("w"), (int, float)):
        items.append(f"<li>What that production is worth: ${rec['w']:.1f}M</li>")
    v = _verdict(rec.get("vd"))
    if v:
        items.append(f"<li>Verdict: {v}</li>")
    if isinstance(rec.get("pcv"), (int, float)):
        items.append(f"<li>Projected next contract: ${rec['pcv']:.1f}M a year</li>")

    return (
        f"<noscript><header><h1>{nm}, NBA Value and Contract Prediction</h1></header>"
        f"<main><p>{esc(rec.get('d', ''))}</p>"
        f"<p>HoopsValue rates {nm} ({team_txt}) with the Barrett Score, a single "
        "value metric built from scoring, playmaking, rebounding, defense, "
        "efficiency and availability, then prices that production at the going "
        f"rate so you can see whether his contract is a bargain or an overpay.</p>"
        f"<ul>{''.join(items)}</ul>"
        f'<p><a href="{esc(url)}">Open {nm} on HoopsValue</a> for his full season '
        "history, comparable contracts and predicted next deal. You can also browse "
        '<a href="/Rankings">every NBA player ranked by value</a>, run the '
        '<a href="/Contract_Predictor">contract predictor</a>, see the '
        '<a href="/Free_Agent_Class">free agent class</a>, or look through '
        '<a href="/Rosters">every team\u2019s salary book</a>.</p>'
        "</main></noscript>")


def _player_jsonld(rec: dict, url: str) -> dict:
    """Person structured data for one player, tied back to the site graph.

    Schema.org has no Barrett Score, so the score and rank ride as
    additionalProperty PropertyValues: named, numeric, and unambiguous about
    what produced them."""
    props = []
    if isinstance(rec.get("s"), (int, float)):
        props.append({"@type": "PropertyValue", "name": "Barrett Score",
                      "value": rec["s"]})
    if isinstance(rec.get("r"), int):
        props.append({"@type": "PropertyValue", "name": "HoopsValue rank",
                      "value": rec["r"]})
    person = {"@type": "Person", "name": rec.get("n", ""),
              "jobTitle": "Basketball player", "url": url}
    team_name = ""
    try:
        team_name = team_pages().get("teams", {}).get(rec.get("tm", ""), {}).get("name", "")
    except Exception:
        team_name = ""
    if team_name or rec.get("tm"):
        person["memberOf"] = {"@type": "SportsTeam",
                              "name": team_name or rec.get("tm", "")}
    if props:
        person["additionalProperty"] = props
    return {
        "@context": "https://schema.org",
        "@type": "ProfilePage",
        "@id": url + "#page",
        "url": url,
        "name": rec.get("t", ""),
        "description": rec.get("d", ""),
        "isPartOf": {"@id": f"{SITE}/#website"},
        "mainEntity": person,
    }


def player_shell(base_html: str, player_name: str) -> str:
    """Return base_html rewritten into a real landing page for the named player:
    title, description, OG tags, a SELF-REFERENCING canonical, a crawlable no-JS
    body and Person structured data. Unknown name returns base_html unchanged.
    Rewrites whatever currently occupies those tags (pattern-based, not coupled
    to the exact TITLE/DESC constants), so it survives base-shell drift.
    Never raises."""
    try:
        rec = _share_meta().get(_share_norm(player_name or ""))
        if not rec:
            return base_html
        t = _html.escape(rec["t"], quote=True)
        d = _html.escape(rec["d"], quote=True)
        url = player_url(rec.get("n") or player_name)
        # lambda replacements: keep re.sub from interpreting \g / backslashes in
        # the player text as group references.
        out = re.sub(r"<title>[^<]*</title>", lambda m: f"<title>{t}</title>",
                     base_html, count=1)
        out = re.sub(r'(<meta name="description" content=")[^"]*(")',
                     lambda m: m.group(1) + d + m.group(2), out, count=1)
        out = re.sub(r'(<meta property="og:title" content=")[^"]*(")',
                     lambda m: m.group(1) + t + m.group(2), out, count=1)
        out = re.sub(r'(<meta property="og:description" content=")[^"]*(")',
                     lambda m: m.group(1) + d + m.group(2), out, count=1)
        # THE de-indexing bug, fixed 2026-11-12. Every ?player= page inherited
        # the homepage's <link rel="canonical" href="https://hoopsvalue.com/">,
        # which tells Google "this is a duplicate of the homepage, index that
        # instead". Several hundred player pages with good titles and real
        # numbers were disclaiming themselves, so none of them could rank.
        out = re.sub(r'(<link rel="canonical" href=")[^"]*(")',
                     lambda m: m.group(1) + url + m.group(2), out, count=1)
        out = re.sub(r'(<meta property="og:url" content=")[^"]*(")',
                     lambda m: m.group(1) + url + m.group(2), out, count=1)
        out = _swap_noscript(out, _player_body(rec, url))
        out = _add_jsonld(out, _player_jsonld(rec, url))
        return out
    except Exception:
        return base_html


# --- Crawlable team pages ------------------------------------------------------
# Real, indexable HTML at /team/<ABBR> (one landing page per NBA team). Rendered
# from cache/team_pages.json (scripts/build_team_pages.py) so the request path is
# a pure lookup + string format -- no model work. serve.py routes /team/<ABBR>
# here; unknown abbreviations return None so the handler can 404.
_TEAM_PAGES = None  # lazy {"season": str, "teams": {abbr: {...}}}


def team_pages() -> dict:
    global _TEAM_PAGES
    if _TEAM_PAGES is None:
        try:
            p = pathlib.Path(__file__).parent / "cache" / "team_pages.json"
            _TEAM_PAGES = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            _TEAM_PAGES = {"season": "", "teams": {}}
    return _TEAM_PAGES


def _money(x: float) -> str:
    return f"${x:.1f}M"


_TEAM_CSS = """
:root{--bg:#f4f6f8;--panel:#fff;--line:#e3e6eb;--ink:#16233f;--muted:#6b7280;
--navy:#16233f;--orange:#e8792b;--teal:#0fae9d;--good:#16a34a;--bad:#e0483a;
--row:#fafbfc;--logo:url('/app/static/hoopsvalue_wordmark_v2.png')}
@media (prefers-color-scheme:dark){:root{--bg:#0a0a14;--panel:#15171d;
--line:#262a33;--ink:#e6e9f2;--muted:#9aa0ac;--navy:#e2e9f8;--row:#101219;
--logo:url('/app/static/hoopsvalue_wordmark_dark_v2.png')}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
-webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}
.wrap{max-width:940px;margin:0 auto;padding:22px 18px 60px}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;
flex-wrap:wrap;padding-bottom:18px;border-bottom:1px solid var(--line)}
.brand{display:block;width:190px;height:46px;background:var(--logo) left center/contain no-repeat}
nav.top a{color:var(--muted);font-weight:600;font-size:14px;margin-left:18px}
nav.top a:hover{color:var(--orange)}
h1{font-size:30px;line-height:1.15;margin:30px 0 6px;letter-spacing:-.4px}
.sub{color:var(--muted);margin:0 0 4px;max-width:640px}
.tot{color:var(--ink);font-weight:600;margin:14px 0 20px;font-size:15px}
.tot span{color:var(--muted);font-weight:500}
.tbl{width:100%;border-collapse:collapse;background:var(--panel);
border:1px solid var(--line);border-radius:12px;overflow:hidden;font-size:15px}
.tbl th{text-align:left;font-size:12px;letter-spacing:.4px;text-transform:uppercase;
color:var(--muted);font-weight:700;padding:12px 12px;border-bottom:1px solid var(--line)}
.tbl td{padding:11px 12px;border-bottom:1px solid var(--line)}
.tbl tr:last-child td{border-bottom:none}
.tbl tbody tr:nth-child(even){background:var(--row)}
.tbl td.num,.tbl th.num{text-align:right;font-variant-numeric:tabular-nums}
.tbl a.pl{font-weight:600}
.tbl a.pl:hover{color:var(--orange)}
.sc{font-weight:700;color:var(--teal)}
.good{color:var(--good);font-weight:600}.bad{color:var(--bad);font-weight:600}
.fair{color:var(--muted)}
.rc{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.02em;
padding:2px 8px;border-radius:20px;white-space:nowrap;border:1px solid var(--line);color:var(--muted)}
.rc-rs{color:var(--teal);border-color:var(--teal)}
.rc-ns{color:var(--orange);border-color:var(--orange)}
.rc-pj{border-style:dashed}
.uns{color:var(--muted);font-size:13.5px;margin:14px 2px 0}
.uns b{color:var(--ink)}
.note{color:var(--muted);font-size:13px;margin:16px 2px 30px;max-width:680px}
.teams{margin-top:34px;padding-top:20px;border-top:1px solid var(--line)}
.teams h2{font-size:13px;letter-spacing:.5px;text-transform:uppercase;
color:var(--muted);margin:0 0 12px}
.teams a{display:inline-block;font-size:13px;font-weight:600;color:var(--muted);
padding:5px 9px;margin:0 6px 8px 0;border:1px solid var(--line);border-radius:7px}
.teams a:hover{color:var(--orange);border-color:var(--orange)}
footer{margin-top:40px;color:var(--muted);font-size:13px;text-align:center}
""".strip()


def _role_chip(role: str, real: bool):
    """Status chip label+class. Real deals read as facts ('Re-signed'/'Signed');
    a model guess for a still-unsigned FA is explicitly 'Projected' (dashed)."""
    if role == "Re-sign":
        return ("Re-signed", " rc-rs") if real else ("Projected re-sign", " rc-rs rc-pj")
    if role == "New signing":
        return ("Signed", " rc-ns") if real else ("Projected signing", " rc-ns rc-pj")
    if role == "Draft pick":
        return ("Rookie", "")
    return (role, "")


def team_page_html(abbr: str):
    """Return a full crawlable HTML page for the team's PROJECTED 2026-27 roster,
    or None if the team is unknown. Never raises (returns None on any failure)."""
    try:
        data = team_pages()
        abbr = (abbr or "").upper()
        team = data.get("teams", {}).get(abbr)
        if not team:
            return None
        # Defaults match each other and the payload rather than naming a
        # season literally: value and contract seasons are the same thing now,
        # and a literal here would outlive the next rollover.
        cseason = data.get("contract_season", "")
        vseason = data.get("value_season", cseason)
        name = team["name"]
        esc = lambda s: _html.escape(str(s), quote=True)  # noqa: E731

        rows = []
        n_proj = sum(1 for p in team["players"]
                     if p.get("role") in ("Re-sign", "New signing") and not p.get("real", True))
        for i, p in enumerate(team["players"], start=1):
            rlabel, rcls = _role_chip(p.get("role", ""), p.get("real", True))
            bar = p.get("barrett")
            bcell = (f'<span class="sc">{bar:.1f}</span>' if isinstance(bar, (int, float))
                     else '<span class="fair">&mdash;</span>')
            sal = p.get("salary")
            scell = _money(sal) if isinstance(sal, (int, float)) else "&mdash;"
            vd = p.get("vd")
            if vd is None:
                verdict = '<span class="fair">&mdash;</span>'
            elif vd <= -2:
                verdict = f'<span class="good">Underpaid {_money(-vd)}</span>'
            elif vd >= 2:
                verdict = f'<span class="bad">Overpaid {_money(vd)}</span>'
            else:
                verdict = '<span class="fair">Fair</span>'
            href = "/?player=" + _urlq(p["n"])
            rows.append(
                f'<tr><td class="num">{i}</td>'
                f'<td><a class="pl" href="{href}">{esc(p["n"])}</a></td>'
                f'<td><span class="rc{rcls}">{esc(rlabel)}</span></td>'
                f'<td class="num">{bcell}</td>'
                f'<td class="num">{scell}</td>'
                f'<td class="num">{verdict}</td></tr>')

        # cross-links to every other team (link graph for crawlers)
        others = "".join(
            f'<a href="/team/{a}">{esc(t["abbr"])}</a>'
            for a, t in sorted(data.get("teams", {}).items()))

        room = team.get("room")
        if isinstance(room, (int, float)):
            aproom = (f'{_money(room)} under the second apron' if room >= 0
                      else f'{_money(-room)} over the second apron')
        else:
            aproom = ""
        title = f"{name} {cseason} Roster, Salaries & Value | HoopsValue"
        desc = (f"The {cseason} {name} roster: every contract and reported signing, what "
                f"each player is paid, and what they are worth by the Barrett Score. See "
                f"the bargains and overpays heading into next season.")
        t_e, d_e = esc(title), esc(desc)
        url = f"https://hoopsvalue.com/team/{abbr}"

        return (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
            f"<title>{t_e}</title>"
            f'<meta name="description" content="{d_e}"/>'
            '<meta name="robots" content="index, follow"/>'
            f'<link rel="canonical" href="{url}"/>'
            '<link rel="icon" type="image/svg+xml" href="/favicon.svg?v=2"/>'
            '<meta property="og:type" content="website"/>'
            f'<meta property="og:title" content="{t_e}"/>'
            f'<meta property="og:description" content="{d_e}"/>'
            f'<meta property="og:url" content="{url}"/>'
            f'<meta property="og:image" content="{OG_IMAGE}"/>'
            '<meta property="og:image:width" content="1200"/>'
            '<meta property="og:image:height" content="630"/>'
            '<meta name="twitter:card" content="summary_large_image"/>'
            f'<meta name="twitter:image" content="{OG_IMAGE}"/>'
            f"<style>{_TEAM_CSS}</style></head><body><div class=\"wrap\">"
            '<header><a class="brand" href="/" aria-label="HoopsValue home"></a>'
            '<nav class="top"><a href="/Rankings">Rankings</a>'
            '<a href="/Team_Analysis">Team Analysis</a>'
            '<a href="/Free_Agent_Class">Free Agents</a></nav></header>'
            f"<h1>{esc(name)}, {esc(cseason)} Roster</h1>"
            f'<p class="sub">The {esc(name)} roster for {esc(cseason)}: contracts on the '
            f"books and reported free-agency signings"
            + (", plus the model&rsquo;s projection for spots still unfilled" if n_proj else "")
            + (f". Value reflects each player&rsquo;s {esc(vseason)} season, the "
               f"latest one played; {esc(cseason)} has not started yet.</p>"
               if vseason != cseason else
               f". Value reflects each player&rsquo;s {esc(vseason)} production so "
               "far this season.</p>")
            + f'<p class="tot">{team["size"]} players <span>·</span> '
            f'{_money(team["payroll"])} payroll'
            + (f' <span>·</span> {aproom}' if aproom else "")
            + (f' <span>·</span> {n_proj} still projected' if n_proj else "") + '</p>'
            '<table class="tbl"><thead><tr>'
            '<th class="num">#</th><th>Player</th><th>Status</th>'
            '<th class="num">Barrett Score</th>'
            f'<th class="num">{esc(cseason)} Salary</th>'
            '<th class="num">Value</th></tr></thead><tbody>'
            + "".join(rows) +
            "</tbody></table>"
            + ("".join([
                '<p class="uns">Still unsigned: ' + ", ".join(
                    f'<b><a href="/?player={_urlq(u["n"])}">{esc(u["n"])}</a></b>'
                    + (f' ({u["barrett"]:.1f} Barrett Score)'
                       if isinstance(u.get("barrett"), (int, float)) else "")
                    for u in team.get("unsigned", [])) +
                ". No reported deal yet, and no open roster spot here.</p>"])
               if team.get("unsigned") else "")
            + '<p class="note">Roster and salaries reflect real '
            f"{esc(cseason)} contracts and free-agency signings reported so far"
            + ("; moves marked Projected are the model&rsquo;s best guess for spots "
               "not yet filled" if n_proj else "")
            + ". The Barrett Score and value reflect each player&rsquo;s "
            + (f"{esc(vseason)} on-court production, the most recent season played, "
               "since next season&rsquo;s games have not happened. "
               if vseason != cseason else
               f"{esc(vseason)} on-court production so far this season. ")
            + "Value is what that production "
            "is worth at the going rate; the verdict compares it to the salary. "
            "Click any player for their full contract prediction.</p>"
            f'<div class="teams"><h2>Browse every team</h2>{others}</div>'
            '<footer><a href="/">HoopsValue</a> · NBA player value and contract '
            "predictions</footer></div></body></html>")
    except Exception:
        return None


def _urlq(s: str) -> str:
    """Minimal query-safe encoding for the ?player= link target."""
    import urllib.parse
    return urllib.parse.quote(str(s))


# --- Shareable trade links (Trade Machine phase 4) -----------------------------
# /Trade_Machine?trade=<b64url of {v:1,t:[teams],m:[[kind,src,id,dest]...],c:...}>
# (format mirrored from templates/trade_machine.html serialize()). The unfurl
# describes the legs from the token alone -- player names ride in the payload,
# so no roster lookups happen on the request path. Never raises.

def _trade_decode(token: str):
    import base64
    s = token.replace("-", "+").replace("_", "/")
    s += "=" * (-len(s) % 4)
    o = json.loads(base64.b64decode(s).decode("utf-8"))
    if o.get("v") != 1 or not isinstance(o.get("t"), list):
        raise ValueError("bad token")
    return o


def trade_shell(base_html: str, token: str) -> str:
    """Return base_html with title/description/OG swapped for a shared trade,
    or base_html unchanged when the token is missing/invalid."""
    try:
        if not token:
            return base_html
        o = _trade_decode(token)
        teams = [str(t)[:4] for t in o["t"][:4]]
        sends = {t: [] for t in teams}
        for mv in o.get("m", []):
            if len(mv) == 4 and mv[1] in sends:
                kind, src, ident, _dest = mv
                if kind in ("P", "T"):
                    sends[src].append(str(ident))
                elif kind in ("F", "G"):
                    nm = str(ident).rsplit("@", 1)[0]
                    sends[src].append(f"{nm} (sign-and-trade)")
                else:
                    parts = str(ident).split(":")
                    yr = parts[0]
                    rnd = "second" if parts[2:3] == ["2"] else "first"
                    if kind == "S":
                        sends[src].append(f"a {yr} pick swap")
                    else:
                        sends[src].append(f"a {yr} {rnd}-round pick")
        legs = []
        for t in teams:
            names = sends[t][:3]
            if not names:
                continue
            extra = len(sends[t]) - len(names)
            legs.append(f"{t} send " + ", ".join(names)
                        + (f" and {extra} more" if extra > 0 else ""))
        title = f"NBA Trade Proposal: {' / '.join(teams)} | HoopsValue"
        desc = (("; ".join(legs) + ". ") if legs else "") + \
            "See whether this trade works under the 2023 CBA and who wins it on value."
        t_e = _html.escape(title, quote=True)
        d_e = _html.escape(desc[:290], quote=True)
        out = re.sub(r"<title>[^<]*</title>", lambda m: f"<title>{t_e}</title>",
                     base_html, count=1)
        out = re.sub(r'(<meta name="description" content=")[^"]*(")',
                     lambda m: m.group(1) + d_e + m.group(2), out, count=1)
        out = re.sub(r'(<meta property="og:title" content=")[^"]*(")',
                     lambda m: m.group(1) + t_e + m.group(2), out, count=1)
        out = re.sub(r'(<meta property="og:description" content=")[^"]*(")',
                     lambda m: m.group(1) + d_e + m.group(2), out, count=1)
        return out
    except Exception:
        return base_html
