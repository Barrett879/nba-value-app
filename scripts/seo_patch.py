"""Patch Streamlit's served index.html for SEO. Run at BUILD time (render.yaml
buildCommand) so the change bakes into the deployed image regardless of runtime
filesystem writability; serve.py also applies the same logic at startup as a
backup. Idempotent (marker check), never fatal.

Streamlit serves a generic shell (<title>Streamlit</title>, no meta description,
and a <noscript>You need to enable JavaScript...</noscript>), so Google indexes a
blank, mis-titled page. The canonical transform lives in seo_selfheal.py at the
repo root: real title + meta/OpenGraph tags, a crawlable noscript block with
internal links, and the favicon link rewritten to the HoopsValue icon.

Usage:  python scripts/seo_patch.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import seo_selfheal  # noqa: E402


if __name__ == "__main__":
    if seo_selfheal.ensure_seo_patched():
        print("[seo_patch] index.html patched (or already patched)")
    else:
        print("[seo_patch] skipped: could not patch index.html")
    sys.exit(0)   # never fail the build over SEO
