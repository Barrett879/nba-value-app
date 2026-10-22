"""Every HTML response must be decoded through utils._html_text.

    /opt/anaconda3/bin/python3 -u scripts/check_html_decoding.py

Read-only. Prints PASS/FAIL per module, exits non-zero on any failure.

WHY THIS EXISTS. requests guesses a response's encoding from the HTTP
Content-Type header and falls back to ISO-8859-1 when no charset is declared.
Basketball-Reference sends a bare "text/html", so r.text decoded every UTF-8
name as latin-1 and "Luka Dončić" arrived as "Luka DonÄ\x8diÄ\x87".

The damage is silent and it is not cosmetic. normalize() folds accents, so a
correctly decoded "Dončić" matches ESPN's ASCII "Doncic" and the salary join
succeeds. Mojibake normalizes to "dona\x8dia\x87", the join misses, and
_build_raw_live's dropna(subset=["salary"]) drops the player out of the pool.
On 2026-10-22 Luka Dončić was missing from the Lakers' rankings for exactly
this reason, with fourteen other accented names, and nothing raised.

The reason it needs a CHECK and not just a fix is that it had already been
fixed. Four call sites carried a hand-written r.encoding = "utf-8", two with a
comment naming Dončić and Jokić. The sites added afterwards did not get it,
because nothing connected them. A per-site fix for a per-site mistake decays
every time someone adds a site; this check is what makes the helper the only
way through.

It parses the AST rather than grepping, so a comment that quotes r.text as
prose (this file is full of them, and so is the helper) cannot be read as a
violation. The preflight check learned that lesson the hard way.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# utils._html_text IS the decoder, so it is the one place allowed to touch
# r.text and r.encoding directly.
DECODER = "_html_text"

MODULES = ["utils.py", "serve.py", "app.py"] + \
          sorted(str(p.relative_to(ROOT)) for p in (ROOT / "pages").glob("*.py")) + \
          sorted(str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("*.py"))

results: list[tuple[str, str, str]] = []


def enclosing_funcs(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def check(rel: str) -> None:
    path = ROOT / rel
    if not path.exists():
        return
    try:
        src = path.read_text()
        tree = ast.parse(src)
    except (SyntaxError, UnicodeDecodeError) as e:
        results.append(("FAIL", rel, f"unparseable: {e}"))
        return

    bad_text: list[str] = []
    bad_enc: list[str] = []

    # ax.text(...) is matplotlib drawing a label, not a response body being
    # read. Collecting the Attribute nodes that are a call's target lets the
    # walk below tell "read the .text property" from "call the .text method".
    # The first version of this check reported generate_incremental_2020.py
    # for an axis annotation, which is the same false-positive class the
    # rollover preflight hit when it grepped raw text instead of parsing.
    called = {id(n.func) for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}

    for fn in enclosing_funcs(tree):
        if fn.name == DECODER:
            continue                       # the decoder itself
        for n in ast.walk(fn):
            # <something>.text  where <something> is a plain name (r, resp, ...)
            if isinstance(n, ast.Attribute) and n.attr == "text" and \
                    isinstance(n.value, ast.Name) and id(n) not in called:
                bad_text.append(f"{rel}:{n.lineno} {fn.name}() reads {n.value.id}.text")
            # <something>.encoding = ...
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Attribute) and t.attr == "encoding":
                        bad_enc.append(f"{rel}:{t.lineno} {fn.name}() sets .encoding by hand")

    # A file that CALLS the helper must also be able to resolve it. Routing a
    # call site through _html_text without importing it swaps a silent
    # mojibake bug for a NameError at scrape time, and the first version of
    # this check passed exactly that state in build_agent_db.py: absence of
    # r.text is not the same thing as presence of a working decoder.
    calls_helper = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == DECODER
        for n in ast.walk(tree))
    if calls_helper and rel != "utils.py":
        imported = any(
            (isinstance(n, ast.ImportFrom) and
             any(a.name == DECODER for a in n.names))
            for n in ast.walk(tree))
        if not imported:
            bad_text.append(f"{rel} calls {DECODER}() but never imports it (NameError)")

    detail = bad_text + bad_enc
    if detail:
        results.append(("FAIL", rel, "; ".join(detail[:6]) +
                        (f" (+{len(detail) - 6} more)" if len(detail) > 6 else "")))
    else:
        results.append(("PASS", rel, "no direct HTML decoding"))


for rel in MODULES:
    check(rel)

# The helper must exist and must honour an explicit server charset. A version
# that blanket-forces UTF-8 would corrupt ESPN's salary pages, which really do
# declare iso-8859-1: the mirror of the bug this check is here to prevent.
try:
    sys.path.insert(0, str(ROOT))
    import utils
    fn = getattr(utils, DECODER, None)
    if not callable(fn):
        results.append(("FAIL", f"utils.{DECODER}", "missing"))
    else:
        class _R:
            def __init__(self, ct, body, enc=None):
                self.headers = {"Content-Type": ct}
                self.content = body
                self._enc = enc or "iso-8859-1"
            @property
            def encoding(self):
                return self._enc
            @encoding.setter
            def encoding(self, v):
                self._enc = v
            @property
            def text(self):
                return self.content.decode(self._enc, errors="replace")

        name = "Luka Dončić"
        utf8 = name.encode("utf-8")

        # 1. bare text/html + <meta charset="utf-8">: the BBRef case
        r1 = _R("text/html", b'<meta charset="utf-8">' + utf8)
        ok1 = name in fn(r1)
        # 2. server declares utf-8: honoured
        r2 = _R("text/html; charset=utf-8", utf8, enc="utf-8")
        ok2 = name in fn(r2)
        # 3. server explicitly declares iso-8859-1: NOT overridden
        latin = "José".encode("iso-8859-1")
        r3 = _R("text/html; charset=iso-8859-1", latin, enc="iso-8859-1")
        ok3 = "José" in fn(r3)
        # 4. no charset anywhere: utf-8 default
        r4 = _R("text/html", utf8)
        ok4 = name in fn(r4)
        # 5. a charset Python cannot load must not raise
        try:
            fn(_R("text/html", b'<meta charset="x-bogus-42">' + utf8))
            ok5 = True
        except Exception:
            ok5 = False

        cases = [("meta utf-8 (BBRef)", ok1), ("header utf-8", ok2),
                 ("header iso-8859-1 preserved", ok3), ("no charset defaults utf-8", ok4),
                 ("unknown charset survives", ok5)]
        failed = [n for n, o in cases if not o]
        results.append(("FAIL" if failed else "PASS", f"utils.{DECODER} behaviour",
                        ("broken: " + ", ".join(failed)) if failed
                        else f"all {len(cases)} decoding cases correct"))
except Exception as e:                                               # noqa: BLE001
    results.append(("FAIL", f"utils.{DECODER} behaviour", f"could not test: {e}"))


print()
print("  HTML DECODING    every response parsed as HTML goes through "
      f"utils.{DECODER}")
print()
width = max(len(n) for _, n, _ in results)
shown = [r for r in results if r[0] == "FAIL"] + [r for r in results if r[0] != "FAIL"]
for state, name, detail in shown:
    print(f"  [{state:4}] {name:{width}}  {detail}")

fails = [r for r in results if r[0] == "FAIL"]
print()
if fails:
    print(f"  FAIL: {len(fails)} module(s) decode HTML without the helper. "
          f"Accented names will be silently dropped from joins.")
else:
    print(f"  PASS: {len(results) - 1} modules clean, helper behaviour verified")
print()
sys.exit(1 if fails else 0)
