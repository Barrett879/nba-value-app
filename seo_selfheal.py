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

MARKER = "hv-seo-v5"
_OLD_MARKERS = ("hv-seo-v1", "hv-seo-v2", "hv-seo-v3", "hv-seo-v4")

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
# Full icon set on ROOT paths (served with correct content-types by serve.py;
# Streamlit mis-serves .svg as text/plain, which crawlers/Google reject). The
# .ico covers Google + legacy, the SVG covers modern tabs, the PNGs cover the
# rest, apple-touch covers iOS, and the manifest covers PWA/Android.
FAVICON_TAG = (
    '<link rel="icon" href="/favicon.ico" sizes="any"/>'
    '<link rel="icon" type="image/svg+xml" href="/favicon.svg"/>'
    '<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png"/>'
    '<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16.png"/>'
    '<link rel="apple-touch-icon" href="/apple-touch-icon.png"/>'
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
    # Swap the favicon for the HoopsValue icon set (crawlers and the pre-boot
    # shell; the running app swaps it again via page_icon). Strip Streamlit's
    # icon link and any icon/apple/manifest links an older patch left, then
    # inject ours once so re-runs and marker upgrades never double up.
    html = re.sub(
        r'<link[^>]*rel="(?:shortcut icon|icon|apple-touch-icon|manifest)"[^>]*/?>',
        "", html, flags=re.I)
    if 'rel="apple-touch-icon"' not in html:
        html = html.replace("</head>", FAVICON_TAG + "</head>", 1)
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


def player_shell(base_html: str, player_name: str) -> str:
    """Return base_html with the title/description/OG tags swapped for the named
    player's, or base_html unchanged when the name is unknown. Rewrites whatever
    currently occupies those four tags (pattern-based, not coupled to the exact
    TITLE/DESC constants), so it survives base-shell drift. Never raises."""
    try:
        rec = _share_meta().get(_share_norm(player_name or ""))
        if not rec:
            return base_html
        t = _html.escape(rec["t"], quote=True)
        d = _html.escape(rec["d"], quote=True)
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
.note{color:var(--muted);font-size:13px;margin:16px 2px 30px;max-width:680px}
.teams{margin-top:34px;padding-top:20px;border-top:1px solid var(--line)}
.teams h2{font-size:13px;letter-spacing:.5px;text-transform:uppercase;
color:var(--muted);margin:0 0 12px}
.teams a{display:inline-block;font-size:13px;font-weight:600;color:var(--muted);
padding:5px 9px;margin:0 6px 8px 0;border:1px solid var(--line);border-radius:7px}
.teams a:hover{color:var(--orange);border-color:var(--orange)}
footer{margin-top:40px;color:var(--muted);font-size:13px;text-align:center}
""".strip()


def team_page_html(abbr: str):
    """Return a full crawlable HTML page for the team abbreviation, or None if
    the team is unknown. Never raises (returns None on any failure)."""
    try:
        data = team_pages()
        abbr = (abbr or "").upper()
        team = data.get("teams", {}).get(abbr)
        if not team:
            return None
        season = data.get("season", "")
        name = team["name"]
        tot = team["tot"]
        esc = lambda s: _html.escape(str(s), quote=True)  # noqa: E731

        rows = []
        for i, p in enumerate(team["players"], start=1):
            vd = p["vd"]  # salary - market; >0 overpaid
            if vd <= -2:
                verdict = f'<span class="good">Underpaid {_money(-vd)}</span>'
            elif vd >= 2:
                verdict = f'<span class="bad">Overpaid {_money(vd)}</span>'
            else:
                verdict = '<span class="fair">Fair</span>'
            href = "/?player=" + _urlq(p["n"])
            rows.append(
                f'<tr><td class="num">{i}</td>'
                f'<td><a class="pl" href="{href}">{esc(p["n"])}</a></td>'
                f'<td class="num"><span class="sc">{p["s"]:.1f}</span></td>'
                f'<td class="num">{_money(p["sal"])}</td>'
                f'<td class="num">{_money(p["val"])}</td>'
                f'<td class="num">{verdict}</td></tr>')

        # cross-links to every other team (link graph for crawlers)
        others = "".join(
            f'<a href="/team/{a}">{esc(t["abbr"])}</a>'
            for a, t in sorted(data.get("teams", {}).items()))

        title = f"{name} Player Value & Contracts ({season}) | HoopsValue"
        desc = (f"Every {name} player ranked by the Barrett Score and measured against "
                f"their salary. See who is underpaid, overpaid, and what each player is "
                f"worth on the {season} roster.")
        t_e, d_e = esc(title), esc(desc)
        url = f"https://hoopsvalue.com/team/{abbr}"

        return (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
            f"<title>{t_e}</title>"
            f'<meta name="description" content="{d_e}"/>'
            '<meta name="robots" content="index, follow"/>'
            f'<link rel="canonical" href="{url}"/>'
            '<link rel="icon" type="image/svg+xml" href="/app/static/favicon.svg"/>'
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
            f"<h1>{esc(name)} Player Value &amp; Contracts</h1>"
            f'<p class="sub">Every player on the {esc(name)} ranked by the Barrett Score '
            f"and held up against what they are paid, {esc(season)}.</p>"
            f'<p class="tot">{tot["n"]} players <span>·</span> {_money(tot["sal"])} payroll '
            f'<span>·</span> {_money(tot["val"])} market value</p>'
            '<table class="tbl"><thead><tr>'
            '<th class="num">#</th><th>Player</th><th class="num">Barrett Score</th>'
            '<th class="num">Salary</th><th class="num">Market Value</th>'
            '<th class="num">Verdict</th></tr></thead><tbody>'
            + "".join(rows) +
            "</tbody></table>"
            '<p class="note">The Barrett Score rates a player\'s on-court value from '
            "scoring, playmaking, rebounding, defense, efficiency, and availability. "
            "Market value is what that production is worth at the going rate; the verdict "
            "compares it to actual salary. Click any player for their full contract "
            "prediction.</p>"
            f'<div class="teams"><h2>Browse every team</h2>{others}</div>'
            '<footer><a href="/">HoopsValue</a> · NBA player value and contract '
            "predictions</footer></div></body></html>")
    except Exception:
        return None


def _urlq(s: str) -> str:
    """Minimal query-safe encoding for the ?player= link target."""
    import urllib.parse
    return urllib.parse.quote(str(s))
