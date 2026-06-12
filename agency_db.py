"""Private: NBA player -> agents -> agency database (pure data layer).

LOCAL-ONLY feature for the private HoopsValue fork. No Streamlit, no network at
import. Loads the curated agent map from data/player_agents.csv with
data/player_agents_override.csv layered on top, keyed on the SAME normalize(name)
the rest of the app uses so it joins cleanly to the Barrett Score player universe.

MULTI-AGENT model: a player can have several listed representatives (HoopsHype
lists 2+ for ~half the league, and the source order is NOT a reliable "primary"),
so each record stores a LIST of agents. The base file may carry several rows per
player, one per agent. The override file FULLY REPLACES a player's agent list, so
a correction can fix a wrong agent, drop a bad one, add the real one(s), or blank
a player outright - the curator has full control of any player they touch.

Data flow:
  - data/player_agents.csv          base map (rows: name,agent,agency,source,as_of),
                                     rewritten by scripts/build_agent_db.py.
  - data/player_agents_override.csv  hand-maintained corrections; replace-by-player.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pandas as pd

_DATA = Path(__file__).parent / "data"
AGENTS_PATH = _DATA / "player_agents.csv"
AGENTS_OVERRIDE_PATH = _DATA / "player_agents_override.csv"
CONTRACTS_PATH = Path(__file__).parent / "cache" / "agent_contracts_v1.json"
AGENCY_CONTACTS_PATH = _DATA / "agency_contacts.csv"

FIELDS = ("name", "agent", "agency", "source", "as_of")
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _normalize(name: str) -> str:
    """Identical to utils.normalize (NFKD strip-accents + lower + strip). Kept
    local so this module stays import-light and usable from bare scripts."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _clean(v) -> str:
    s = str(v).strip() if v is not None else ""
    return "" if s.lower() == "nan" else s


def _read_grouped(path: Path | str) -> dict:
    """{normalized_name: {"name", "agents": [{agent, agency, source, as_of}, ...]}}.

    Several rows for one player accumulate into that player's agents list (same
    agent name is not added twice). A row with a blank agent still registers the
    player with an empty list - which, via the override, is how you explicitly
    blank a player. '#'-prefixed lines are comments.
    """
    out: dict = {}
    try:
        df = pd.read_csv(path, comment="#", skip_blank_lines=True, dtype=str)
    except Exception:
        return out
    if "name" not in df.columns:
        return out
    for _, row in df.iterrows():
        nm = _clean(row.get("name"))
        if not nm:
            continue
        key = _normalize(nm)
        rec = out.setdefault(key, {"name": nm, "agents": []})
        rec["name"] = nm
        agent = _clean(row.get("agent"))
        if agent and not any(a["agent"].lower() == agent.lower() for a in rec["agents"]):
            rec["agents"].append({
                "agent": agent,
                "agency": _clean(row.get("agency")),
                "source": _clean(row.get("source")),
                "as_of": _clean(row.get("as_of")),
            })
    return out


def load_player_agents(base_path: Path | str = AGENTS_PATH,
                       override_path: Path | str = AGENTS_OVERRIDE_PATH) -> dict:
    """Base map with the override layered on top. The override REPLACES a player's
    whole agent list (not field-by-field), so a curator who touches a player owns
    that player's representation. Missing/blank files contribute nothing."""
    base = _read_grouped(base_path)
    for key, rec in _read_grouped(override_path).items():
        base[key] = rec
    return base


def has_agent(rec: dict | None) -> bool:
    """A record counts as covered only if it lists at least one agent."""
    return bool(rec and rec.get("agents"))


def agent_names(rec: dict | None) -> list[str]:
    """All agent names for a record, in file order."""
    return [a["agent"] for a in (rec or {}).get("agents", []) if a.get("agent")]


def agency_names(rec: dict | None) -> list[str]:
    """Distinct agency names for a record, in first-seen order (usually one)."""
    out: list[str] = []
    for a in (rec or {}).get("agents", []):
        ag = a.get("agency")
        if ag and ag not in out:
            out.append(ag)
    return out


def representation_line(name: str, agent_map: dict, index: dict | None = None) -> str:
    """One-line 'Represented by ...' string for a player, or '' if unknown. When
    all agents share one agency it reads 'A, B (Agency)'; when agencies differ it
    pairs each agent with its own 'A (Agency 1), B (Agency 2)'. No em dashes."""
    rec = match(name, agent_map, index)
    agents = [a for a in (rec or {}).get("agents", []) if a.get("agent")]
    if not agents:
        return ""
    if len(agents) == 1 and agents[0]["agent"].lower().startswith("self-represent"):
        return "Self-represented"
    agys = agency_names(rec)
    if len(agys) <= 1:
        suffix = f" ({agys[0]})" if agys else ""
        return "Represented by " + ", ".join(a["agent"] for a in agents) + suffix
    parts = [a["agent"] + (f" ({a['agency']})" if a.get("agency") else "") for a in agents]
    return "Represented by " + ", ".join(parts)


def _loose(key: str) -> str:
    """Period-insensitive variant of an already-normalized key, so the app's
    "a.j. lawson" matches HoopsHype's "aj lawson"."""
    return key.replace(".", "")


def _base(key: str) -> str:
    """Drop periods AND a trailing generational suffix (jr/sr/ii/iii/iv/v), so
    the app's "michael porter jr." meets HoopsHype's suffix-less "michael porter".
    Used only as an ambiguity-checked last resort (see build_index)."""
    toks = key.replace(".", "").split()
    if toks and toks[-1] in _SUFFIXES:
        toks = toks[:-1]
    return " ".join(toks)


def build_index(agent_map: dict) -> dict:
    """Fallback lookup indices for match(): a period-insensitive 'loose' map and
    a suffix-insensitive 'base' map. The base map EXCLUDES any key two different
    players share (e.g. a father/son "gary payton" / "gary payton ii"), so a
    suffix strip can never silently mismatch them. Build once, pass into match()."""
    loose: dict = {}
    base: dict = {}
    base_dup: set = set()
    for k, rec in agent_map.items():
        loose.setdefault(_loose(k), rec)
        bk = _base(k)
        if bk in base and base[bk] is not rec:
            base_dup.add(bk)
        else:
            base.setdefault(bk, rec)
    for bk in base_dup:
        base.pop(bk, None)
    return {"loose": loose, "base": base}


def match(name: str, agent_map: dict, index: dict | None = None) -> dict | None:
    """Look up a player's record in tiers: exact normalized, then period-
    insensitive (Jr./Sr./initials), then ambiguity-checked suffix-insensitive.
    None if all miss. Nickname gaps (Nic vs Nicolas) are intentionally NOT bridged
    here - those belong in data/player_agents_override.csv."""
    k = _normalize(name)
    rec = agent_map.get(k)
    if rec:
        return rec
    if index is None:
        index = build_index(agent_map)
    return index["loose"].get(_loose(k)) or index["base"].get(_base(k))


# ── Contract-by-agent (from the scraped HoopsHype payload) ───────────────────

def load_agent_contracts(path: Path | str = CONTRACTS_PATH) -> dict:
    """The raw per-agent client + contract payload written by build_agent_db.py.
    {} if absent (e.g. before the scraper has ever run)."""
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def load_agency_contacts(path: Path | str = AGENCY_CONTACTS_PATH) -> dict:
    """{agency_lower: {website, hq, phone, email, status}} of published BUSINESS
    contact per agency (no personal contact for individuals). {} if absent."""
    out: dict = {}
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return out
    for _, r in df.iterrows():
        ag = str(r.get("agency", "")).strip()
        if ag:
            out[ag.lower()] = {f: str(r.get(f, "")).strip()
                               for f in ("website", "hq", "phone", "email", "status")}
    return out


def _client_book(c: dict, from_season: int | None) -> dict | None:
    """One client's contract summary from their seasons array, or None if they
    carry no live contract in range. The HoopsHype seasons array is a player's
    FULL salary history (past years included), so from_season (a HoopsHype
    end-year int, e.g. 2026 for 2025-26) counts only current+future years - the
    live book, not money already paid. None counts every season (career total).
    Seasons are deduped to one row per year defensively."""
    by_year: dict = {}
    for s in c.get("seasons", []):
        if s.get("terminated") or (s.get("salary") or 0) <= 0:
            continue
        y = s.get("season")
        if from_season is not None and (y or 0) < from_season:
            continue
        if y not in by_year or (s.get("salary") or 0) > (by_year[y].get("salary") or 0):
            by_year[y] = s
    seasons = list(by_year.values())
    if not seasons:
        return None
    opts = set()
    for s in seasons:
        if s.get("playerOption"):
            opts.add("PO")
        if s.get("teamOption"):
            opts.add("TO")
        if s.get("qualifyingOffer"):
            opts.add("QO")
        if s.get("twoWayContract"):
            opts.add("2W")
    return {
        "player": c.get("playerName"),
        "playerID": str(c.get("playerID") or c.get("playerName")),
        "years": len(seasons),
        "total": sum(s.get("salary") or 0 for s in seasons),
        "options": "/".join(sorted(opts)),
    }


def _resolve_agency(agent: str, raw_agency: str, agency_of: dict | None) -> str:
    """The agent's agency, preferring a corrected mapping (agent name -> agency)
    so books reflect override fixes (e.g. Drew Gross now WME, not Roc Nation)
    rather than the raw scrape stored in the payload."""
    if agency_of:
        return agency_of.get((agent or "").lower(), raw_agency)
    return raw_agency


def agent_books(payload: dict, from_season: int | None = None,
                agency_of: dict | None = None) -> list[dict]:
    """Per-agent "book" summaries (client list, contract values, total book),
    sorted by book value descending. Pass agency_of {agent_lower: agency} to show
    corrected agencies instead of the payload's raw scrape."""
    out: list[dict] = []
    for a in payload.get("agents", {}).values():
        clients = [cb for c in a.get("clients", []) if (cb := _client_book(c, from_season))]
        if not clients:
            continue
        clients.sort(key=lambda x: -x["total"])
        out.append({
            "agent": a.get("agent"),
            "agency": _resolve_agency(a.get("agent"), a.get("agency"), agency_of),
            "n_clients": len(clients),
            "book_total": sum(c["total"] for c in clients),
            "clients": clients,
        })
    out.sort(key=lambda x: -x["book_total"])
    return out


def agency_books(payload: dict, from_season: int | None = None,
                 agency_of: dict | None = None) -> list[dict]:
    """Per-AGENCY book summaries: each agency's unique clients (deduped by player
    so co-agents at the same agency don't double-count), total contract value and
    client count, sorted by book value. agency_of {agent_lower: agency} applies
    the override corrections; a player repped across two agencies counts in both."""
    by_agency: dict = {}                              # agency -> {playerID: client}
    for a in payload.get("agents", {}).values():
        agency = _resolve_agency(a.get("agent"), a.get("agency"), agency_of)
        if not agency:
            continue
        bucket = by_agency.setdefault(agency, {})
        for c in a.get("clients", []):
            cb = _client_book(c, from_season)
            if cb:
                bucket.setdefault(cb["playerID"], cb)
    out = []
    for agency, m in by_agency.items():
        clients = sorted(m.values(), key=lambda x: -x["total"])
        out.append({
            "agency": agency,
            "n_clients": len(clients),
            "book_total": sum(c["total"] for c in clients),
            "clients": clients,
        })
    out.sort(key=lambda x: -x["book_total"])
    return out
