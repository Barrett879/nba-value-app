"""Patch Streamlit's served index.html for SEO. Run at BUILD time (render.yaml
buildCommand) so the change bakes into the deployed image regardless of runtime
filesystem writability; serve.py also calls the same logic at startup as a
backup. Idempotent (marker hv-seo-v1), never fatal.

Streamlit serves a generic shell — <title>Streamlit</title>, no meta description,
and a <noscript>You need to enable JavaScript...</noscript> — so Google indexes a
blank, mis-titled page. This injects a real title + meta/OpenGraph tags and swaps
the default noscript for real HoopsValue content with internal links.

Usage:  python scripts/seo_patch.py
"""
import pathlib
import re
import sys

TITLE = "HoopsValue · NBA Player Value, Contract Predictions & Rankings"
DESC = ("HoopsValue ranks every NBA player by the Barrett Score — on-court "
        "production measured against their paycheck — and predicts what any player "
        "would sign for today. Find the steals, expose the overpays, and run any "
        "team's free agency.")
PAGES = [
    ("Contract_Predictor", "Contract Predictor — what any player would sign for today"),
    ("Rankings", "Current Rankings — every NBA player by Barrett Score"),
    ("Search", "Player Search"),
    ("Legacy", "Legacy — the best players ever by Barrett Score"),
]


def patch() -> bool:
    try:
        import streamlit as _st
    except Exception as e:  # streamlit not importable -> nothing to do
        print(f"[seo_patch] streamlit import failed: {e}")
        return False
    idx = pathlib.Path(_st.__file__).parent / "static" / "index.html"
    try:
        html = idx.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[seo_patch] cannot read {idx}: {e}")
        return False
    if "hv-seo-v1" in html:
        print("[seo_patch] already patched")
        return True
    head = (
        "<!--hv-seo-v1-->"
        f'<meta name="description" content="{DESC}"/>'
        '<meta name="robots" content="index, follow"/>'
        '<link rel="canonical" href="https://hoopsvalue.com/"/>'
        f'<meta property="og:title" content="{TITLE}"/>'
        f'<meta property="og:description" content="{DESC}"/>'
        '<meta property="og:type" content="website"/>'
        '<meta property="og:url" content="https://hoopsvalue.com/"/>'
        '<meta property="og:image" content="https://hoopsvalue.com/app/static/hoopsvalue_logo.png"/>'
        '<meta name="twitter:card" content="summary_large_image"/>'
    )
    nav = "".join(f'<li><a href="/{s}">{l}</a></li>' for s, l in PAGES)
    body = ("<noscript><header><h1>HoopsValue</h1>"
            "<p>NBA player value, contract predictions, and rankings.</p></header>"
            f"<main><p>{DESC}</p><ul>{nav}</ul></main></noscript>")
    if "<title>Streamlit</title>" in html:
        html = html.replace("<title>Streamlit</title>", f"<title>{TITLE}</title>{head}")
    else:
        html = html.replace("</head>", f"<title>{TITLE}</title>{head}</head>", 1)
    if "<noscript" in html:
        html = re.sub(r"<noscript>.*?</noscript>", body, html, count=1, flags=re.S)
    else:
        html = html.replace("<body>", f"<body>{body}", 1)
    try:
        idx.write_text(html, encoding="utf-8")
    except Exception as e:
        print(f"[seo_patch] cannot write {idx}: {e}")
        return False
    print(f"[seo_patch] patched {idx}")
    return True


if __name__ == "__main__":
    ok = patch()
    sys.exit(0)   # never fail the build over SEO
