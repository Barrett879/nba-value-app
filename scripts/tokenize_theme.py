#!/usr/bin/env python3
"""Rewrite hardcoded CSS colours -> theme tokens, CSS regions ONLY.

The light/dark theme works by swapping CSS custom properties (see
utils.THEME_BASE_CSS / THEME_LIGHT_CSS). For that to reach a colour, the colour
must be written as `var(--token)` rather than a hex/rgba literal. This script
does that rewrite mechanically, but ONLY inside:

  1. <style> ... </style> blocks
  2. inline  style="..."  /  style='...'  attribute values

It deliberately LEAVES every other hex alone — crucially the colour kwargs
passed to plotly/altair/matplotlib in Python (e.g. marker_color="#e63946"),
which must stay literal because chart engines can't resolve CSS variables.

Matching is on COMPLETE colour tokens (`#rgb`, `#rrggbb`, `rgba(...)`), so
`#fff` never clobbers the `#fff` inside `#ffffff`. Unmapped colours are left
untouched (reported with --report).

Usage:
    python scripts/tokenize_theme.py FILE [FILE ...]            # dry-run diff
    python scripts/tokenize_theme.py --apply FILE [FILE ...]    # write in place
    python scripts/tokenize_theme.py --report FILE [FILE ...]   # list unmapped
"""
from __future__ import annotations

import difflib
import re
import sys

# ── colour -> token map (keys NORMALISED: hex lowercased; rgba spaces stripped)
COLOR_MAP = {
    # surfaces
    "#0a0a14": "var(--bg-base)",
    "#0a0a0a": "var(--bg-nav)",
    "#15171d": "var(--panel-solid)",
    "#1a1a2e": "var(--panel-2)",
    "#1a2e1a": "var(--tint-good)",
    "#1a2a1a": "var(--tint-good)",
    "#2e1a1a": "var(--tint-bad)",
    "rgba(20,20,42,0.55)": "var(--panel)",
    "rgba(20,20,42,0.7)": "var(--panel)",
    "rgba(30,30,56,0.85)": "var(--panel-hover)",
    "rgba(80,80,110,0.35)": "var(--panel-line)",
    "rgba(250,250,250,0.55)": "var(--panel)",
    # hairlines (white-alpha dividers / tracks)
    "rgba(255,255,255,0.15)": "var(--hairline)",
    "rgba(255,255,255,0.12)": "var(--hairline)",
    "rgba(255,255,255,0.10)": "var(--hairline)",
    "rgba(255,255,255,0.1)": "var(--hairline)",
    "rgba(255,255,255,0.08)": "var(--hairline)",
    "rgba(255,255,255,0.07)": "var(--hairline)",
    "rgba(255,255,255,0.06)": "var(--hairline)",
    "rgba(255,255,255,0.05)": "var(--hairline-soft)",
    "rgba(255,255,255,0.04)": "var(--hairline-soft)",
    "rgba(255,255,255,0.03)": "var(--hairline-soft)",
    # near-white text/overlay variants (Rankings)
    "rgba(250,250,250,0.85)": "var(--fg-1)",
    "rgba(250,250,250,0.6)": "var(--fg-3)",
    "rgba(250,250,250,0.12)": "var(--hairline)",
    # era-tinted dark surfaces (Legacy)
    "#0a1a2a": "var(--panel-2)",
    "#0a2a0a": "var(--tint-good)",
    "#2a1a0a": "var(--tint-bad)",
    # text ramp
    "#ffffff": "var(--fg-1)",
    "#fff": "var(--fg-1)",
    "#cdcdd5": "var(--fg-2)",
    "#cfcfd6": "var(--fg-2)",
    "#d0d0d6": "var(--fg-2)",
    "#e8e8ee": "var(--fg-2)",
    "#ddd": "var(--fg-2)",
    "#aaaaaa": "var(--fg-3)",
    "#aaa": "var(--fg-3)",
    "#9a9aa3": "var(--fg-4)",
    "#9aa0aa": "var(--fg-4)",
    "#999999": "var(--fg-4)",
    "#999": "var(--fg-4)",
    "#8a8a93": "var(--fg-4)",
    "#888": "var(--fg-4)",
    "#7a7a85": "var(--fg-5)",
    "#777": "var(--fg-5)",
    "#6a6a72": "var(--fg-6)",
    "#666": "var(--fg-6)",
    "#5a5a62": "var(--fg-6)",
    "#555555": "var(--fg-6)",
    "#555": "var(--fg-6)",
    # accents (nudge across modes via tokens)
    "#e63946": "var(--accent-red)",
    "#16d4c1": "var(--accent-teal)",
    "#2ecc71": "var(--value-good)",
    "#27ae60": "var(--value-good)",
    "#a8e6a8": "var(--value-good-s)",
    "#e74c3c": "var(--value-bad)",
    "#f1a8a8": "var(--value-bad-s)",
    "#f1c40f": "var(--gold)",
    "#3498db": "var(--blue)",
    "#f39c12": "var(--orange)",
    "#9b59b6": "var(--purple)",
    "#7ec8e8": "var(--sky)",
    "#4cc9f0": "var(--sky)",
    "#f0b35b": "var(--amber)",
}

# one colour token: a hex (#rgb/#rrggbb/#rrggbbaa) or an rgb()/rgba() call
COLOR_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)")
# CSS regions: a whole <style> block, or one inline style="..."/style='...'
STYLE_BLOCK_RE = re.compile(r"<style>.*?</style>", re.DOTALL)
INLINE_STYLE_RE = re.compile(r"""style=(["'])(.*?)\1""", re.DOTALL)


def _norm(color: str) -> str:
    c = color.strip().lower()
    if c.startswith("rgb"):
        c = re.sub(r"\s+", "", c)
    return c


def _convert_region(text: str, unmapped: set[str]) -> str:
    def repl(m: re.Match) -> str:
        raw = m.group(0)
        tok = COLOR_MAP.get(_norm(raw))
        if tok is None:
            unmapped.add(_norm(raw))
            return raw
        return tok
    return COLOR_RE.sub(repl, text)


def convert(src: str, unmapped: set[str]) -> str:
    # inline style="..." first (these can live outside <style> blocks)
    def inline_repl(m: re.Match) -> str:
        q, body = m.group(1), m.group(2)
        return f"style={q}{_convert_region(body, unmapped)}{q}"
    src = INLINE_STYLE_RE.sub(inline_repl, src)
    # then whole <style> blocks
    src = STYLE_BLOCK_RE.sub(lambda m: _convert_region(m.group(0), unmapped), src)
    return src


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    report = "--report" in argv
    files = [a for a in argv if not a.startswith("--")]
    if not files:
        print(__doc__)
        return 1
    for path in files:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        unmapped: set[str] = set()
        out = convert(src, unmapped)
        n = sum(1 for _ in difflib.unified_diff(src.splitlines(), out.splitlines()))
        if report:
            print(f"\n{path}: unmapped colours in CSS regions:")
            for c in sorted(unmapped):
                print(f"   {c}")
            continue
        if apply:
            if out != src:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(out)
                print(f"✓ {path}: rewritten ({n} diff lines)")
            else:
                print(f"– {path}: no change")
        else:
            diff = difflib.unified_diff(
                src.splitlines(keepends=True), out.splitlines(keepends=True),
                fromfile=path, tofile=path + " (tokenized)",
            )
            sys.stdout.writelines(diff)
            if unmapped:
                print(f"\n# {path}: {len(unmapped)} unmapped colours (left as-is): "
                      + ", ".join(sorted(unmapped)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
