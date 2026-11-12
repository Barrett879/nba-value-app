"""What a crawler sees. No network, no server, no board build.

Every assertion here corresponds to a defect that was live on 2026-11-12:

  * /?player=X served <link rel="canonical" href="https://hoopsvalue.com/">,
    which tells Google the page is a duplicate of the homepage and should not
    be indexed. Several hundred player pages were disclaiming themselves.
  * Those pages carried the HOMEPAGE'S <noscript> body, so the only text a
    crawler could read never mentioned the player it was titled after.
  * Ten indexable routes shipped ten titles over one identical body.
  * The marker-upgrade path bumped the version and left the body from whatever
    release first patched the file.
  * The sitemap listed eleven URLs and no player pages, and nothing crawlable
    linked to one.

A test that only asserts the new behaviour would pass against a shell that
still carried the old canonical, so several of these assert the OLD output is
GONE rather than that the new output is present.
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import seo_selfheal as S  # noqa: E402

# A minimal stand-in for Streamlit's SPA shell.
RAW = ('<!doctype html><html lang="en"><head><meta charset="utf-8"/>'
       '<title>Streamlit</title><link rel="shortcut icon" href="./favicon.png"/>'
       '</head><body><noscript>You need to enable JavaScript to run this app.'
       '</noscript><div id="root"></div></body></html>')


@pytest.fixture(scope="module")
def shell():
    return S.seo_html(RAW)


@pytest.fixture(scope="module")
def records():
    recs = [(k, v) for k, v in S._share_meta().items()
            if not k.startswith("_") and isinstance(v, dict)]
    if not recs:
        pytest.skip("cache/share_meta.json absent; run scripts/build_share_meta.py")
    return recs


def _tag(html, pattern):
    m = re.search(pattern, html)
    return m.group(1) if m else None


def _canonical(html):
    return _tag(html, r'<link rel="canonical" href="([^"]*)"')


def _noscript(html):
    m = re.search(r"<noscript>(.*?)</noscript>", html, re.S)
    return m.group(1) if m else ""


def _jsonld(html):
    return [json.loads(b.replace("<\\/", "</")) for b in
            re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)]


# --- the shell ---------------------------------------------------------------

def test_shell_is_idempotent(shell):
    assert S.seo_html(shell) == shell


def test_shell_has_one_canonical(shell):
    assert shell.count('rel="canonical"') == 1


def test_site_jsonld(shell):
    blocks = _jsonld(shell)
    assert len(blocks) == 1
    nodes = {n["@type"]: n for n in blocks[0]["@graph"]}
    assert "WebSite" in nodes and "Organization" in nodes
    assert nodes["WebSite"]["potentialAction"]["@type"] == "SearchAction"
    # "hoops value" as two words is how people type it, and nothing on the site
    # ever spelled it that way, so there was no signal tying the query to the
    # brand at all.
    assert "Hoops Value" in nodes["WebSite"]["alternateName"]
    assert "Hoops Value" in shell


def test_marker_upgrade_regenerates_the_body(shell):
    """A shell on an older marker must come out carrying the CURRENT body."""
    stale = shell.replace(S.MARKER, "hv-seo-v3", 1)
    stale = re.sub(r"<noscript>.*?</noscript>",
                   "<noscript><h1>Old</h1><ul><li><a href='/A'>A</a></li></ul></noscript>",
                   stale, count=1, flags=re.S)
    up = S.seo_html(stale)
    assert S.MARKER in up
    assert "<h1>Old</h1>" not in up
    assert up.count('<li><a href="/') == len(S.PAGES)
    assert "application/ld+json" in up


# --- player landing pages ----------------------------------------------------

def test_player_canonical_is_not_the_homepage(shell, records):
    _k, rec = records[0]
    page = S.player_shell(shell, rec["n"])
    assert _canonical(page) != "https://hoopsvalue.com/"
    assert _canonical(page).startswith("https://hoopsvalue.com/?player=")
    assert _tag(page, r'<meta property="og:url" content="([^"]*)"') == _canonical(page)


def test_ascii_and_accented_requests_share_one_canonical(shell):
    """Both spellings must resolve to ONE canonical or they compete as two
    pages for one player."""
    accented = "Nikola Vučević"
    a = S.player_shell(shell, "Nikola Vucevic")
    b = S.player_shell(shell, accented)
    if a == shell:
        pytest.skip("player not in this board")
    assert _canonical(a) == _canonical(b)
    # and the one they share is the DISPLAY spelling, not the request's
    assert _canonical(a) == S.player_url(accented)
    assert _canonical(a) != S.player_url("Nikola Vucevic")


def test_player_body_names_the_player(shell, records):
    _k, rec = records[0]
    body = _noscript(S.player_shell(shell, rec["n"]))
    assert rec["n"] in body
    assert rec["n"] in body.split("</h1>")[0]
    # the homepage's copy must not be what a crawler reads here
    assert "HoopsValue rates every NBA player" not in body


def test_player_jsonld(shell, records):
    _k, rec = records[0]
    blocks = _jsonld(S.player_shell(shell, rec["n"]))
    assert len(blocks) == 2, "site graph + this player"
    pj = blocks[1]
    assert pj["@type"] == "ProfilePage"
    assert pj["mainEntity"]["@type"] == "Person"
    assert pj["mainEntity"]["name"] == rec["n"]
    assert pj["isPartOf"]["@id"] == "https://hoopsvalue.com/#website"


@pytest.mark.parametrize("name", ["Not A Real Player", "", None])
def test_unknown_player_changes_nothing(shell, name):
    assert S.player_shell(shell, name) == shell


def test_no_page_prints_one_number_two_ways(shell, records):
    """The description formats from the stored value, and the body formats from
    the same stored value, so a page cannot say $2.4M in one and $2.5M in the
    other. It did, before the builder rounded once."""
    money = re.compile(r"\$(\d+\.\d)M")
    bad = []
    for _k, rec in records:
        body = _noscript(S.player_shell(shell, rec["n"]))
        in_body = set(money.findall(body))
        for fig in money.findall(rec["d"]):
            if fig not in in_body:
                bad.append((rec["n"], fig, sorted(in_body)))
                break
    assert not bad, f"{len(bad)} page(s) disagree with themselves, e.g. {bad[:3]}"


def test_fallback_scores_name_their_season(shell, records):
    """A score carried over from a previous season must say so, exactly as the
    hub panel does. Otherwise the page states a number for a season the player
    has not played."""
    tagged = [r for _k, r in records if r.get("sy")]
    if not tagged:
        pytest.skip("no fallback rows in this board")
    for rec in tagged[:25]:
        assert rec["sy"] in rec["d"]
        assert rec["sy"] in _noscript(S.player_shell(shell, rec["n"]))


# --- section routes ----------------------------------------------------------

@pytest.mark.parametrize("route", sorted(S._PAGE_BODY))
def test_route_is_self_canonical_with_its_own_body(shell, route):
    page = S.page_shell(shell, route)
    assert _canonical(page) == "https://hoopsvalue.com" + route
    body = _noscript(page)
    assert body and body != _noscript(shell)
    assert "—" not in body, "no em dashes in copy (house style)"


def test_every_route_body_is_distinct(shell):
    bodies = [_noscript(S.page_shell(shell, r)) for r in S._PAGE_BODY]
    assert len(set(bodies)) == len(bodies)


def test_homepage_route_is_untouched(shell):
    assert S.page_shell(shell, "/") == shell


def test_rankings_links_down_to_player_pages(shell):
    """The sitemap is a hint; a link is a path. Before this the only crawlable
    links to a player page were on the team pages, which are parked and 404."""
    body = _noscript(S.page_shell(shell, "/Rankings"))
    assert body.count('href="https://hoopsvalue.com/?player=') >= 100


# --- sitemap inputs ----------------------------------------------------------

def test_player_urls_cover_every_record(records):
    urls = S.player_urls()
    assert len(urls) == len(records)
    assert len({u for _n, u in urls}) == len(urls)
    assert all(u.startswith("https://hoopsvalue.com/?player=") for _n, u in urls)
    # a <loc> carrying a raw & or < is invalid XML and voids the whole sitemap
    assert not [u for _n, u in urls if set("&<>\"") & set(u)]


def test_build_stamp_present():
    assert len(S.share_meta_built_at()) >= 10


# --- team pages (parked behind HV_TEAM_PAGES, must still render clean) -------

def test_team_page_renders_without_contradicting_itself():
    page = S.team_page_html("LAL")
    if page is None:
        pytest.skip("cache/team_pages.json absent")
    # read "Value reflects each player's 2026-27 season, the latest one played,
    # 2026-27 has not started yet" once and the bug is obvious; it shipped
    # because value_season and contract_season became equal at the rollover.
    assert "has not started yet" not in page or \
        S.team_pages().get("value_season") != S.team_pages().get("contract_season")
    assert 'href="/team/' in page, "cross-links must reach the sibling pages"
    assert 'href="/Rosters?team=' not in page


def test_unknown_team_is_none():
    assert S.team_page_html("ZZZ") is None
