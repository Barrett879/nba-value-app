"""Build the player -> agents -> agency base map from HoopsHype.

One-shot offline builder (run locally, like the other scripts/). It pulls the
full agent roster from HoopsHype's agents index, then each agent's client list,
and writes:

  - data/player_agents.csv         the base map (name,agent,agency,source,as_of),
                                   ONE ROW PER (player, agent) - a player with
                                   several representatives gets several rows.
  - cache/agent_contracts_v1.json  the richer per-agent client + contract payload
                                   (playerID, agentIDs, salary by season, options),
                                   kept for the future contract-by-agent phase.

HoopsHype is a Next.js app, so every page server-embeds its data in a
__NEXT_DATA__ JSON island. Plain requests + json is enough - no JS rendering.
A player's full agent list comes from each contract's `agentIDs`; HoopsHype's
order is NOT a reliable "primary first", so all agents are kept (agency_db lists
them, and the override curates).

Polite by construction: browser headers, rate-limited between requests, never
overwrites the CSV with an empty result. Run:
    python scripts/build_agent_db.py [--limit N] [--agent SLUG]
    python scripts/build_agent_db.py --from-cache   # rebuild CSV from the saved
                                                     # payload, no network
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
OUT_CSV = DATA / "player_agents.csv"
OUT_RAW = CACHE / "agent_contracts_v1.json"

BASE = "https://www.hoopshype.com/salaries/agents/"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
SLEEP = 0.7          # politeness pause between per-agent requests
TIMEOUT = 15


def _next_data(html: str) -> dict:
    m = re.search(r'id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        raise ValueError("no __NEXT_DATA__ script island on page")
    return json.loads(m.group(1))


def _queries(nd: dict) -> list:
    return nd["props"]["pageProps"]["dehydratedState"]["queries"]


def fetch_agent_index() -> list[dict]:
    """Every agent from the index page: list of {slug, agent, agency}."""
    r = requests.get(BASE, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    nd = _next_data(r.text)
    best: list = []
    for q in _queries(nd):
        d = q.get("state", {}).get("data") or {}
        agents = (d.get("agents") or {}).get("agents")
        if isinstance(agents, list) and agents and "agentName" in agents[0]:
            if len(agents) > len(best):
                best = agents
    if not best:
        raise ValueError("agent roster not found in index __NEXT_DATA__")
    out = []
    for a in best:
        slug = a.get("id") or a.get("slug")
        if slug:
            out.append({"slug": slug,
                        "agent": (a.get("agentName") or "").strip(),
                        "agency": (a.get("agencyName") or "").strip()})
    return out


def fetch_agent_clients(slug: str) -> list[dict]:
    """Contract records for one agent: [{playerName, playerID, agentIDs, seasons}]."""
    r = requests.get(f"{BASE}{slug}/", headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    nd = _next_data(r.text)
    for q in _queries(nd):
        d = q.get("state", {}).get("data") or {}
        contracts = (d.get("contracts") or {}).get("contracts")
        if isinstance(contracts, list) and contracts and "playerName" in contracts[0]:
            return contracts
    return []


def _rows_from(imap: dict, players: dict) -> list[tuple[str, str, str]]:
    """imap: {slug:{agent,agency}}; players: {pid:{name,agentIDs}} -> sorted
    (player, agent, agency) rows, one per resolvable agent, dupes removed."""
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for p in players.values():
        for aid in p["agentIDs"]:
            a = imap.get(aid)
            if not a or not a.get("agent"):
                continue
            key = (p["name"].lower(), a["agent"].lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append((p["name"], a["agent"], a["agency"]))
    rows.sort(key=lambda r: (r[0].lower(), r[1].lower()))
    return rows


def _write_csv(rows: list[tuple[str, str, str]], as_of: str) -> None:
    DATA.mkdir(exist_ok=True)
    tmp = OUT_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "agent", "agency", "source", "as_of"])
        for nm, agent, agency in rows:
            w.writerow([nm, agent, agency, "hoopshype", as_of])
    tmp.replace(OUT_CSV)
    n_players = len({nm for nm, _, _ in rows})
    print(f"wrote {len(rows)} agent rows for {n_players} players -> {OUT_CSV}")


def _players_from_clients(agents_by_slug: dict) -> dict:
    """{playerID: {name, agentIDs}} collected once across all agent pages (the
    same player carries the same agentIDs list wherever they appear)."""
    players: dict = {}
    for a in agents_by_slug.values():
        for c in a["clients"]:
            nm = (c.get("playerName") or "").strip()
            if not nm:
                continue
            pid = str(c.get("playerID") or nm.lower())
            if pid not in players:
                players[pid] = {"name": nm, "agentIDs": list(c.get("agentIDs") or [])}
    return players


def build_live(limit: int | None, only_agent: str | None) -> int:
    index = fetch_agent_index()
    imap = {a["slug"]: a for a in index}      # FULL roster: agentID -> {agent, agency}
    print(f"index: {len(index)} agents")

    crawl = index
    if only_agent:
        crawl = [a for a in index if a["slug"] == only_agent]
        if not crawl:
            print(f"agent slug {only_agent!r} not in index")
            return 1
    if limit:
        crawl = crawl[:limit]
        print(f"(limited to first {len(crawl)} agents)")

    raw: dict = {}
    for i, a in enumerate(crawl, 1):
        slug = a["slug"]
        try:
            clients = fetch_agent_clients(slug)
        except Exception as e:                      # one bad page must not kill the run
            print(f"  [{i}/{len(crawl)}] {slug}: ERROR {type(e).__name__} {e}")
            time.sleep(SLEEP)
            continue
        raw[slug] = {"agent": a["agent"], "agency": a["agency"], "clients": clients}
        print(f"  [{i}/{len(crawl)}] {slug}: {len(clients)} clients ({a['agency']})")
        time.sleep(SLEEP)

    as_of = datetime.now(timezone.utc).strftime("%Y-%m")
    rows = _rows_from(imap, _players_from_clients(raw))
    if not rows:
        print("ABORT: scraped 0 rows, leaving existing files untouched")
        return 1
    _write_csv(rows, as_of)

    CACHE.mkdir(exist_ok=True)
    tmpr = OUT_RAW.with_suffix(".json.tmp")
    tmpr.write_text(json.dumps({"as_of": as_of, "source": "hoopshype", "agents": raw}, indent=1))
    tmpr.replace(OUT_RAW)
    print(f"wrote raw contract payload ({len(raw)} agents) -> {OUT_RAW}")
    return 0


def build_from_cache() -> int:
    """Rebuild the CSV from the saved payload - no network. Use after changing
    the row format so a re-scrape is not needed."""
    if not OUT_RAW.exists():
        print(f"no cached payload at {OUT_RAW}; run a live build first")
        return 1
    payload = json.loads(OUT_RAW.read_text())
    agents = payload["agents"]
    imap = {slug: {"agent": a["agent"], "agency": a["agency"]} for slug, a in agents.items()}
    rows = _rows_from(imap, _players_from_clients(agents))
    if not rows:
        print("ABORT: 0 rows from cache, leaving CSV untouched")
        return 1
    _write_csv(rows, payload.get("as_of", "cache"))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build player->agents->agency map from HoopsHype.")
    ap.add_argument("--limit", type=int, default=None, help="crawl only the first N agents (testing)")
    ap.add_argument("--agent", type=str, default=None, help="crawl only this agent slug (probe)")
    ap.add_argument("--from-cache", action="store_true",
                    help="rebuild the CSV from cache/agent_contracts_v1.json, no network")
    args = ap.parse_args()
    if args.from_cache:
        sys.exit(build_from_cache())
    sys.exit(build_live(limit=args.limit, only_agent=args.agent))
