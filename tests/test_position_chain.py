"""The position chain: what the site shows when no source knows the player.

group_flex("") returns SF/PF, because _norm5 falls back to a generic forward.
So before the roster layer, a player none of the curated files listed was
DISPLAYED at small forward with nothing marking it as a default: 60 players on
2026-11-13, among them Robert Williams, Tacko Fall, Felix Okpara and Graham Ike
(all centres) and six point guards.

These tests pin the three properties that fix depends on:

  * the roster layer replaces an INVENTED answer and never a real one, so a
    fallback naming a real spot still wins (the measurement that produced this
    rule: putting the roster above pos_detailed moved AJ Green from SG/PG to
    SF/PF, and 64 others);
  * a roster row with no position contributes NOTHING, or _norm5("") would put
    the invented SF/PF back in through the layer meant to remove it;
  * every request-path call site passes the layer, because one that forgets is
    a page quietly disagreeing with the others, which is the bug class this
    change exists to close.
"""
from __future__ import annotations
import ast
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import team_suitors as ts                                       # noqa: E402

SPOTS = {"PG", "SG", "SF", "PF", "C"}


# ───────────────────────────── the predicate ─────────────────────────────────
@pytest.mark.parametrize("label, parts", [
    ("C", ["C"]), ("PG", ["PG"]), ("C/PF", ["C", "PF"]), ("SG/SF", ["SG", "SF"]),
    ("c/pf", ["C", "PF"]), ("PF-C", ["PF", "C"]), ("SF|PF", ["SF", "PF"]),
    ("PG/PG", ["PG"]),                       # deduped
    # a GROUP or nothing names no spot, and that is what lets the chain tell a
    # real fallback from a guess
    ("G", []), ("F", []), ("Guard", []), ("Forward", []), ("Wing", []),
    ("Center", []), ("", []), ("   ", []), (None, []),
    ("PG/Forward", []),                      # mixed: not specific
])
def test_spot_parts(label, parts):
    assert ts._spot_parts(label) == parts


@pytest.mark.parametrize("raw, want", [
    ("C", "C"), ("C/PF", "C/PF"), ("PG", "PG"), ("SF/SG", "SF/SG"),
    ("PG/PF", "PG/PF"),
    # a group label is all the feed said, so expand it to the group
    ("G", "SG/PG"), ("F", "SF/PF"),
    # blank contributes nothing. _norm5("") is "SF", so a bug here would feed
    # the invented answer back in through the layer meant to remove it.
    ("", ""), ("   ", ""), (None, ""),
])
def test_canon_position(raw, want):
    assert ts._canon_position(raw) == want


def test_canon_never_loses_a_named_spot():
    """group_flex keeps only the PRIMARY, so it turns C/PF into C and SF/SG
    into SF/PF. _canon_position must not route a specific label through it."""
    assert ts.group_flex("C/PF") == "C"          # the trap, pinned
    assert ts.group_flex("SF/SG") == "SF/PF"
    assert ts._canon_position("C/PF") == "C/PF"
    assert ts._canon_position("SF/SG") == "SF/SG"


# ───────────────────────────── the roster layer ──────────────────────────────
def test_load_roster_positions_shape():
    items = ts.load_roster_positions()
    assert len(items) > 400, "master_roster should carry hundreds of positions"
    for name, pos in items.items():
        assert name.strip() == name and name, f"unkeyed name {name!r}"
        assert pos, f"{name}: empty position entered the map"
        for part in pos.split("/"):
            assert part in SPOTS, f"{name}: {pos!r} is not a 5-spot string"


def test_blank_roster_rows_are_absent():
    """Every row with a blank pos must be missing from the map, not present
    with a guessed value."""
    path = ROOT / "data" / "master_roster.csv"
    with open(path) as fh:
        rows = list(csv.DictReader(l for l in fh if l.strip()
                                   and not l.lstrip().startswith("#")))
    blank = [(r.get("player") or "").strip() for r in rows
             if (r.get("player") or "").strip() and not (r.get("pos") or "").strip()]
    assert blank, "expected at least one blank-pos row to exercise this"
    items = ts.load_roster_positions()
    assert [n for n in blank if n in items] == []


def test_load_roster_positions_tolerates_a_missing_file(tmp_path):
    assert ts.load_roster_positions(tmp_path / "nope.csv") == {}


def test_load_roster_positions_tolerates_wrong_columns(tmp_path):
    p = tmp_path / "other.csv"
    p.write_text("name,positions\nSomeone,PG\n")
    assert ts.load_roster_positions(p) == {}


def test_index_resolves_spelling_variants():
    """The layer must go through the repo's one name matcher. master_roster
    says "C.J. McCollum" where the board says "CJ McCollum"; a bare normalize()
    key is what dropped seven players' next contracts and $95.5M in October."""
    idx = ts.roster_position_index()
    assert idx.get("CJ McCollum") == idx.get("C.J. McCollum") is not None
    assert idx.get("PJ Hall") == idx.get("P.J. Hall") is not None
    assert idx.get("A Player Who Does Not Exist") is None


def test_index_is_memoized(tmp_path):
    first = ts.roster_position_index()
    assert ts.roster_position_index() is first, "rebuilt per call: app.py "\
        "resolves a position inside a per-player loop"
    other = tmp_path / "roster.csv"
    other.write_text("player,pos\nMade Upperson,C\n")
    swapped = ts.roster_position_index(other)
    assert swapped is not first and swapped.get("Made Upperson") == "C"
    assert ts.roster_position_index() is not swapped


def test_memo_key_includes_the_path(tmp_path):
    """Two files of the same size and mtime must not share a cache entry.

    A memo keyed on (mtime, size) alone looks correct against master_roster
    because any other file differs in one of them; it fails only on a
    collision, which is exactly the case a test has to construct.
    """
    import os
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("player,pos\nAaa Aaa,C\n")
    b.write_text("player,pos\nBbb Bbb,P\n")       # same byte length
    assert a.stat().st_size == b.stat().st_size, "fixture sizes drifted"
    st = a.stat()
    os.utime(b, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert b.stat().st_mtime_ns == st.st_mtime_ns
    assert ts.roster_position_index(a).get("Aaa Aaa") == "C"
    assert ts.roster_position_index(b).get("Bbb Bbb") is not None, \
        "served a's cached index for b: the memo key dropped the path"
    assert ts.roster_position_index(b).get("Aaa Aaa") is None


def test_exact_index_fallback_matches_the_nameindex_shape():
    """If utils cannot be imported the layer degrades to exact keys. It must
    still answer .get(display_name), and must return None rather than a wrong
    position when it misses."""
    shim = ts._ExactIndex({"Robert Williams": "C"})
    assert shim.get("Robert Williams") == "C"
    assert shim.get("robert williams") == "C"
    assert shim.get("Robert Williams III") is None
    assert shim.get("Nobody", "x") == "x"


# ───────────────────────────── the chain order ───────────────────────────────
def test_chain_without_a_roster_is_unchanged():
    """Every pre-existing caller passes three arguments and must resolve
    exactly as it did, including to the invented default."""
    assert ts.resolve_position("Robert Williams", "", {}) == "SF/PF"
    assert ts.resolve_position("Someone", "Guard", {}) == "SG/PG"
    assert ts.resolve_position("Someone", "SG", {"someone": "PG"}) == "PG"


def test_roster_replaces_the_invented_answer():
    idx = ts.roster_position_index()
    assert ts.resolve_position("Robert Williams", "", {}, idx) == "C"
    assert ts.resolve_position("Tacko Fall", "", {}, idx) == "C"
    assert ts.resolve_position("Braden Smith", "", {}, idx) == "PG"


def test_roster_beats_a_coarse_group_fallback():
    idx = ts.roster_position_index()
    # ESPN's feed only says Guard / Forward / Center
    assert ts.resolve_position("Tobe Awaka", "Forward", {}, idx) == "PF"
    assert ts.resolve_position("Braden Smith", "Guard", {}, idx) == "PG"


def test_a_specific_fallback_still_beats_the_roster():
    """The regression this ordering exists for. AJ Green's detailed feed says
    SG, which is right; master_roster says SF/PF, which is not. Putting the
    roster above a specific fallback moved him and 64 others."""
    idx = ts.roster_position_index()
    assert idx.get("A.J. Green") == "SF/PF", "fixture moved; pick another row"
    assert ts.resolve_position("A.J. Green", "SG", {}, idx) == "SG/PG"
    assert ts.resolve_position("A.J. Green", "PG/SG", {}, idx) == "PG/SG"


def test_curated_sources_still_win_over_the_roster():
    idx = ts.roster_position_index()
    assert ts.resolve_position("Robert Williams", "", {"robert williams": "PF"},
                               idx) == "PF"


def test_roster_misses_fall_through_exactly_as_before():
    idx = ts.roster_position_index()
    assert ts.resolve_position("Nobody At All", "", {}, idx) == "SF/PF"
    assert ts.resolve_position("Nobody At All", "Center", {}, idx) == "C"


# ──────────────────────── every call site passes it ──────────────────────────
# The comp gate is held back on purpose; see _curated_pos's docstring. Anything
# else that resolves a position on the request path must pass the layer, or it
# is a page quietly disagreeing with the others.
_EXEMPT = {("pages/Contract_Predictor.py", "_curated_pos_map")}
_REQUEST_PATH = ["app.py", "pages/Rankings.py", "pages/Team_Analysis.py",
                 "pages/Free_Agent_Class.py", "pages/Contract_Predictor.py",
                 "scripts/build_static.py"]


def _exempt_token(call) -> str | None:
    """The helper the call's POS_MAP argument (the 3rd) comes from.

    It has to be the third argument specifically, not the first identifier
    followed by "(" anywhere in the arguments: a call that merely MENTIONS
    _curated_pos_map() in some other slot, such as using it as the fallback,
    would otherwise inherit the comp gate's exemption and skip the check
    entirely while being wrong in production.
    """
    args = call.args
    if len(args) < 3:
        return None
    node = args[2]
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Call):
        f = node.func
        return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
    return getattr(node, "id", None)


def _is_a_real_layer(node) -> bool:
    """Is this argument plausibly a live lookup, rather than something that
    silently switches the layer off?

    resolve_position treats None as "no layer" and an empty mapping as a miss,
    so counting arguments is not enough: `resolve_position(n, fb, pos_map,
    None)` has four positional arguments and no layer at all. A name, an
    attribute or a call is accepted; a constant or an empty literal is not.
    """
    if isinstance(node, ast.Constant):
        return False
    if isinstance(node, (ast.Dict, ast.Set, ast.List, ast.Tuple)):
        return bool(getattr(node, "elts", None) or getattr(node, "keys", None))
    return isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Subscript,
                             ast.BoolOp, ast.IfExp))


def _resolve_calls(path: Path):
    """One record per resolve_position call: where it is, how many positional
    arguments it has, whether a roster was passed and whether that roster is a
    real lookup, which helper fed its pos_map, the enclosing expression's
    source, and the call node itself."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name != "resolve_position":
            continue
        kw = next((k.value for k in node.keywords if k.arg == "roster"), None)
        roster = node.args[3] if len(node.args) > 3 else kw
        out.append({
            "lineno": node.lineno,
            "n_pos": len(node.args),
            "roster": roster,
            "roster_is_real": roster is not None and _is_a_real_layer(roster),
            "token": _exempt_token(node),
            "src": ast.unparse(node),
            "node": node,
        })
    return out


def _enclosing_exprs(path: Path):
    """The source of every expression that CONTAINS a resolve_position call, so
    a test can ask whether the override is consulted in the same breath."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.BoolOp, ast.IfExp, ast.Call, ast.Lambda)):
            continue
        src = ast.unparse(node)
        if "resolve_position" in src:
            out.append(src)
    return out


@pytest.mark.parametrize("rel", _REQUEST_PATH)
def test_every_request_path_call_passes_the_roster_layer(rel):
    calls = _resolve_calls(ROOT / rel)
    assert calls, f"{rel}: no resolve_position call found; did it move?"
    for c in calls:
        if any(rel == e[0] and c["token"] == e[1] for e in _EXEMPT):
            continue
        assert c["roster"] is not None, (
            f"{rel}:{c['lineno']} resolves a position without the roster layer "
            f"({c['src']}). A player no curated file lists will be shown at an "
            f"invented SF/PF. Pass the index from position_layers(), or add "
            f"the call to _EXEMPT with the reason.")
        assert c["roster_is_real"], (
            f"{rel}:{c['lineno']} passes {ast.unparse(c['roster'])!r} as the "
            f"roster layer, which resolve_position treats as no layer at all. "
            f"Counting arguments is not enough: None and {{}} both disable the "
            f"fix while looking like it is there.")


# data/position_overrides.csv is EDITORIAL and outranks every scraped position,
# so every surface that SHOWS a position has to apply it. Three pages did not,
# and read a different position from the board for 39 of 45 overridden players.
_SHOWS_A_POSITION = ["app.py", "pages/Rosters.py", "pages/Rankings.py",
                     "pages/Free_Agent_Class.py", "pages/Team_Analysis.py",
                     "scripts/build_static.py"]


@pytest.mark.parametrize("rel", _SHOWS_A_POSITION)
def test_every_displayed_position_applies_the_overrides(rel):
    """The override must be CONSULTED in the same expression as the resolve,
    not merely imported somewhere in the file.

    The first version of this test asserted only that the identifier appeared
    in the source. It passed with the consult surgically deleted from all five
    call sites, because the import line alone satisfied it: Kevin Durant read
    SG/SF on Rankings against PF/SF on the board and the suite was green.
    """
    if rel == "pages/Rosters.py":
        # no resolve_position at all: it reads the override index, then
        # master_roster's pos directly
        src = (ROOT / rel).read_text()
        assert "_posov.get(" in src or "_posov_idx.get(" in src, \
            f"{rel} no longer consults the override index"
        return
    exprs = _enclosing_exprs(ROOT / rel)
    assert exprs, f"{rel}: no expression contains a resolve_position call"
    unguarded = [e for e in exprs
                 if "_posov_idx.get(" not in e and "_curated_pos_map(" not in e]
    # an enclosing Call node is reported as well as the BoolOp around it, so
    # only fail when NO expression at that site consults the override
    by_site = {}
    for c in _resolve_calls(ROOT / rel):
        if c["token"] in (e[1] for e in _EXEMPT):
            continue
        hits = [e for e in exprs if c["src"] in e]
        by_site[c["lineno"]] = any("_posov_idx.get(" in h for h in hits)
    assert by_site, f"{rel}: every resolve_position call is exempt; check _EXEMPT"
    bad = [ln for ln, ok in by_site.items() if not ok]
    assert not bad, (
        f"{rel}: the resolve_position call(s) at line(s) {bad} do not consult "
        f"_posov_idx in the same expression. data/position_overrides.csv is "
        f"EDITORIAL and outranks every scraped position; a surface that skips "
        f"it reads a different position from the board (41 of 45 overridden "
        f"players, before 2026-11-13). Unguarded: {unguarded[:1]}")


# ── the gate: both layers describe the CURRENT roster ────────────────────────
def test_position_layers_are_empty_for_a_historical_season():
    import utils
    ov, rp = utils.position_layers(utils.SEASONS[0])
    assert ov.get("Kevin Durant") == "PF/SF" and rp is not None

    old_season = utils.SEASONS[-1]
    assert old_season != utils.SEASONS[0]
    ov_h, rp_h = utils.position_layers(old_season)
    assert rp_h is None, "the 2026-27 roster must not reach a historical board"
    assert ov_h.get("Kevin Durant") is None, \
        "a 2026-27 editorial position must not reach a historical board"


def test_the_namesake_regression_stays_fixed():
    """NameIndex is suffix-insensitive and matches a short given name to a
    longer one, BY DESIGN. Applied to a historical board that means the wrong
    person: 1983-84 Rob Williams, a 6'1" point guard, took Robert Williams
    III's "C"; 1997-98 Jaren Jackson took his son's editorial "PF/C"; 2000-01
    Gary Payton took Gary Payton II's "SF/SG"."""
    import utils
    live_ov, live_rp = utils.position_layers(utils.SEASONS[0])
    # the lookups DO match across the generations, which is the hazard
    assert live_rp.get("Rob Williams") == live_rp.get("Robert Williams")
    assert live_ov.get("Jaren Jackson") == live_ov.get("Jaren Jackson Jr.")
    # ...and the gate is what keeps them off a historical board
    old_ov, old_rp = utils.position_layers(utils.SEASONS[-1])
    assert old_rp is None
    for nm in ("Rob Williams", "Jaren Jackson", "Gary Payton", "Gary Trent",
               "Tim Hardaway", "Jabari Smith"):
        assert old_ov.get(nm) is None, f"{nm} still picks up a 2026-27 position"
        assert ts.resolve_position(nm, "", {}, old_rp) == "SF/PF", \
            f"{nm} still picks up a 2026-27 roster position"


_GATED = ["app.py", "pages/Rankings.py", "pages/Team_Analysis.py",
          "pages/Free_Agent_Class.py", "scripts/build_static.py"]


@pytest.mark.parametrize("rel", _GATED)
def test_display_surfaces_take_their_layers_from_the_one_gate(rel):
    """A page that builds the layers itself has its own answer to "which
    season is this", and that is how the historical boards broke."""
    src = (ROOT / rel).read_text()
    assert "position_layers(" in src, f"{rel} does not go through the gate"
    # ...and it has to be IMPORTED, not just called. Deleting the import line
    # while leaving the call is a module-level NameError on every page load,
    # and no test in this repo imports a page (they need a Streamlit runtime),
    # so nothing else would notice.
    tree = ast.parse(src)
    imported = {a.name or "" for n in ast.walk(tree)
                if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert "position_layers" in imported, (
        f"{rel} calls position_layers() without importing it: NameError on "
        f"every load")
    for direct in ("_ts.roster_position_index()", "position_override_index()"):
        assert direct not in src, (
            f"{rel} calls {direct} directly instead of position_layers(season). "
            f"Both layers describe the CURRENT roster; a surface with a season "
            f"picker must not apply them to a historical board.")


def test_the_memo_is_published_in_one_rebind():
    """A two-step publish (clear, then update) leaves a window where a
    concurrent reader finds the value gone and raises out of a module-scope
    binding nothing wraps in a try."""
    src = (ROOT / "team_suitors.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "roster_position_index")
    calls = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    for forbidden in (".clear(", ".update(", ".pop(", ".setdefault("):
        assert not any(forbidden in c for c in calls), (
            f"roster_position_index mutates its memo in place ({forbidden}); "
            f"publish the new (stamp, index) pair as a single rebind instead")


def test_the_overrides_reach_the_pages_through_one_loader():
    """Four copies of this loader is how three pages came to disagree with the
    board. utils.position_overrides is the one copy; a page must not grow its
    own reader of the CSV again."""
    import re as _re
    for rel in _SHOWS_A_POSITION:
        src = (ROOT / rel).read_text()
        reads = _re.findall(r'open\([^)]*position_overrides\.csv', src)
        reads += _re.findall(r'_read_csv\([^)]*position_overrides\.csv', src)
        assert not reads, (
            f"{rel} opens position_overrides.csv itself ({reads}); call "
            f"utils.position_override_index() instead")


def test_overrides_win_over_both_lower_layers():
    import utils
    idx = utils.position_override_index()
    assert idx.get("Kevin Durant") == "PF/SF"
    assert idx.get("Jaren Jackson Jr.") == "PF/C"
    # and the roster layer must not be what answers them
    rp = ts.roster_position_index()
    assert rp.get("Jaren Jackson Jr.") != "PF/C", \
        "fixture moved: pick an override that disagrees with master_roster"


def test_conditional_override_rows_resolve_against_the_injury_feed():
    """A row with alt_pos + alt_when_out must return one or the other, never
    the raw pair, and never blank."""
    import csv as _csv
    import utils
    path = ROOT / "data" / "position_overrides.csv"
    with open(path) as fh:
        rows = [r for r in _csv.DictReader(l for l in fh if l.strip()
                                           and not l.lstrip().startswith("#"))
                if (r.get("alt_pos") or "").strip()
                and (r.get("alt_when_out") or "").strip()]
    assert rows, "expected at least one conditional row to exercise this"

    # Drive the branch BOTH ways. Asserting only that the answer is one of the
    # two values is a tautology: it cannot tell "resolved against the feed"
    # from "feed ignored" or "feed read backwards", and inverting the test at
    # utils.py left 59 tests green while Draymond Green was listed at centre
    # whenever Porzingis was HEALTHY.
    raw = utils.position_overrides.__wrapped__          # past st.cache_data
    real = utils.fetch_injuries
    try:
        for r in rows:
            who, pos, alt, dep = (r["player"].strip(), r["pos"].strip(),
                                  r["alt_pos"].strip(), r["alt_when_out"].strip())
            utils.fetch_injuries = lambda: {dep: {"status": "Out"}}
            assert raw().get(who) == alt, (
                f"{who} should be {alt} while {dep} is OUT")
            utils.fetch_injuries = lambda: {dep: {"status": "Day-To-Day"}}
            assert raw().get(who) == pos, (
                f"{who} should be {pos} while {dep} is only day-to-day: "
                f"injury_is_out deliberately does not count day-to-day")
            utils.fetch_injuries = lambda: {}
            assert raw().get(who) == pos, f"{who} should be {pos} with no feed"
            utils.fetch_injuries = lambda: (_ for _ in ()).throw(RuntimeError("feed down"))
            assert raw().get(who) == pos, (
                f"{who}: a broken feed must fall back to the normal position, "
                f"not drop the row")
    finally:
        utils.fetch_injuries = real


def test_an_unconditional_override_ignores_the_injury_feed():
    import utils
    raw = utils.position_overrides.__wrapped__
    real = utils.fetch_injuries
    try:
        utils.fetch_injuries = lambda: {"Kristaps Porzingis": {"status": "Out"},
                                        "Brandon Ingram": {"status": "Out"}}
        assert raw().get("Kevin Durant") == "PF/SF"
    finally:
        utils.fetch_injuries = real


def test_the_exemption_still_describes_something_real():
    """A whitelist that stops matching is a whitelist that hides the next
    regression. The exemption has to name a call that is STILL three-argument:
    if the comp gate ever gains the roster layer, this entry is dead weight
    pointing at nothing, and the next unguarded call site slips past it.
    """
    for rel, token in _EXEMPT:
        matched = [c for c in _resolve_calls(ROOT / rel) if c["token"] == token]
        assert matched, (
            f"{rel}: nothing calls resolve_position with {token}() as its "
            f"pos_map any more; drop it from _EXEMPT or point it at the new call")
        assert all(c["roster"] is None for c in matched), (
            f"{rel}: the {token}() calls now pass the roster layer, so the "
            f"_EXEMPT entry is dead. Remove it.")
