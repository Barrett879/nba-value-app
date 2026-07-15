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
import pathlib
import re

import streamlit

MARKER = "hv-seo-v4"
_OLD_MARKERS = ("hv-seo-v1", "hv-seo-v2", "hv-seo-v3")

TITLE = "HoopsValue · NBA Player Value, Contract Predictions & Rankings"
DESC = ("HoopsValue scores every NBA player since 1973 by their on-court value "
        "and holds it up against their salary, so you can see who's actually "
        "worth their contract and who isn't.")
# Crawlable body copy for the no-JS shell: this is most of the text Google reads
# on a Streamlit SPA, so it carries the value prop in natural language (not
# keyword stuffing) covering the searches the site actually answers.
_BODY_COPY = (
    "HoopsValue rates every NBA player with the Barrett Score, a single value "
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
    ("Team_Analysis", "Team Analysis: every NBA roster ranked by value"),
    ("Search", "Compare Players: head-to-head NBA player value"),
    ("Legacy", "Legacy: the best NBA players ever by Barrett Score"),
    ("About", "About HoopsValue and the Barrett Score"),
]
FAVICON_TAG = '<link rel="icon" type="image/svg+xml" href="/app/static/favicon.svg" />'

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
        nav = "".join(f'<li><a href="/{slug}">{label}</a></li>' for slug, label in PAGES)
        body = (
            f"<noscript><header><h1>{TITLE}</h1>"
            "<p>NBA player value, contract predictions, salary analysis, and rankings.</p></header>"
            f"<main><p>{DESC}</p><p>{_BODY_COPY}</p><p>{_BODY_Q}</p>"
            f"<ul>{nav}</ul></main></noscript>"
        )
        if "<title>Streamlit</title>" in html:
            html = html.replace("<title>Streamlit</title>", f"<title>{TITLE}</title>{head}")
        else:
            html = html.replace("</head>", f"<title>{TITLE}</title>{head}</head>", 1)
        if "<noscript" in html:
            html = re.sub(r"<noscript>.*?</noscript>", body, html, count=1, flags=re.S)
        else:
            html = html.replace("<body>", f"<body>{body}", 1)
    # Swap Streamlit's favicon for the HoopsValue icon (crawlers and the
    # pre-boot shell; the running app swaps it again via page_icon).
    html = re.sub(r'<link rel="shortcut icon"[^>]*>', FAVICON_TAG, html, count=1)
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
