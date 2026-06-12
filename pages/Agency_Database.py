import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from utils import (
    SEASONS,
    build_ranked_projected,
    render_page_chrome, render_nav, render_footer,
    html_table, stat_cards, _bootstrap_warm,
)
from agency_db import (
    load_player_agents, has_agent, build_index, match, agent_names, agency_names,
    load_agent_contracts, agent_books, agency_books, load_agency_contacts,
)


def _fmt_money(d) -> str:
    d = float(d or 0)
    if abs(d) >= 1e9:
        return f"${d / 1e9:.2f}B"
    if abs(d) >= 1e6:
        return f"${d / 1e6:.1f}M"
    if abs(d) >= 1e3:
        return f"${d / 1e3:.0f}K"
    return f"${d:.0f}"


_AGENCY_CONTACTS = load_agency_contacts()


def _agency_contact_md(agency: str) -> str:
    """A subtle one-line contact string for an agency (website link, HQ, phone,
    email), or '' if we have no published business contact for it."""
    c = _AGENCY_CONTACTS.get((agency or "").lower())
    if not c:
        return ""
    parts = []
    if c["website"]:
        dom = c["website"].split("//")[-1].strip("/").split("/")[0]
        parts.append(f"[{dom}]({c['website']})")
    for f in ("hq", "phone", "email"):
        if c[f]:
            parts.append(c[f])
    return "  ·  ".join(parts)

st.set_page_config(page_title="Agency Database", layout="wide")
render_page_chrome()
_bootstrap_warm()
# "Agency Database" is deliberately NOT in utils._NAV_PAGES, so no nav link
# renders for it. The page is reachable only by direct /Agency_Database URL,
# which keeps this private build out of the public-looking site navigation.
render_nav("Agency Database")

st.title("Agency Database")
st.caption(
    "Private build, not linked in the site nav. Every active NBA player mapped to "
    "their agents and agency, scraped from HoopsHype with manual corrections layered "
    "on top. A player can have more than one representative, and all are listed."
)

CURRENT_SEASON = SEASONS[0]

agent_map = load_player_agents()
_idx = build_index(agent_map)

current = build_ranked_projected(CURRENT_SEASON)
if current is None or current.empty:
    st.warning("Current-season rankings are still warming up. Reload in a moment.")
    render_footer()
    st.stop()
current = current.copy()

barrett = dict(zip(current["Player"], current["barrett_score"].fillna(0.0)))

rows = []
all_agents: set = set()
agency_clients: dict = defaultdict(set)   # agency -> {player}
agency_agents: dict = defaultdict(set)    # agency -> {agent}
agency_roster: dict = defaultdict(lambda: defaultdict(list))  # agency -> player -> [agents there]
covered_barrett: dict = {}                 # player -> barrett (covered only)

for name in current["Player"].tolist():
    rec = match(name, agent_map, _idx)
    ags = agent_names(rec)
    agys = agency_names(rec)
    b = round(float(barrett.get(name, 0.0)), 1)
    covered = has_agent(rec)
    if covered:
        covered_barrett[name] = b
        all_agents.update(ags)
        for a in rec["agents"]:
            if a.get("agency"):
                agency_clients[a["agency"]].add(name)
                agency_agents[a["agency"]].add(a["agent"])
                agency_roster[a["agency"]][name].append(a["agent"])
    rows.append({
        "Player":    name,
        "Agent":     ", ".join(ags) if ags else "—",
        "Agency":    ", ".join(agys) if agys else "—",
        "Barrett":   b,
        "_covered":  covered,
        "_agencies": agys,
    })
df = pd.DataFrame(rows)

# ── Coverage summary ──────────────────────────────────────────────────────────
n_total = len(df)
n_cov = int(df["_covered"].sum())
pct = (n_cov / n_total * 100.0) if n_total else 0.0

stat_cards([
    ("Active players", f"{n_total:,}",            "var(--accent-teal)"),
    ("With an agent",  f"{n_cov:,}",              "var(--accent-teal)"),
    ("Coverage",       f"{pct:.0f}%",             "var(--accent-teal)" if pct >= 90 else "var(--accent-red)"),
    ("Agencies",       f"{len(agency_clients):,}", "var(--accent-teal)"),
    ("Agents",         f"{len(all_agents):,}",     "var(--accent-teal)"),
])

st.divider()

# ── Player table ──────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], vertical_alignment="center")
with left:
    agencies = ["All agencies"] + sorted(agency_clients.keys())
    pick = st.selectbox("Filter by agency", agencies)
with right:
    only_cov = st.checkbox("Only players with a known agent", value=False)

view = df
if pick != "All agencies":
    view = view[view["_agencies"].apply(lambda lst: pick in lst)]
if only_cov:
    view = view[view["_covered"]]
view = view.sort_values(["_covered", "Barrett"], ascending=[False, False])

html_table(
    view[["Player", "Agent", "Agency", "Barrett"]],
    numeric=["Barrett"],
    aligns={"Barrett": "right"},
    formatters={"Barrett": lambda v: f"{v:.1f}"},
    height=620,
)

# ── Agencies by active client count ───────────────────────────────────────────
if agency_clients:
    agg = pd.DataFrame([
        {
            "Agency": agency,
            "Clients": len(clients),
            "Agents": len(agency_agents[agency]),
            "Avg Barrett": round(sum(covered_barrett[p] for p in clients) / len(clients), 1),
        }
        for agency, clients in agency_clients.items()
    ]).sort_values("Clients", ascending=False)
    st.subheader("Agencies by active client count")
    html_table(
        agg,
        numeric=["Clients", "Agents", "Avg Barrett"],
        aligns={"Clients": "right", "Agents": "right", "Avg Barrett": "right"},
        formatters={"Avg Barrett": lambda v: f"{v:.1f}"},
        height=420,
    )

# ── Players grouped by agency ─────────────────────────────────────────────────
if agency_clients:
    st.divider()
    st.subheader("Players by agency")
    st.caption("Pick an agency to see its active roster, ranked by Barrett Score.")
    _ag_opts = sorted(agency_clients, key=lambda a: (-len(agency_clients[a]), a))
    _ag = st.selectbox("Agency", _ag_opts, key="agency_roster_pick", label_visibility="collapsed")
    _cm = _agency_contact_md(_ag)
    if _cm:
        st.caption(_cm)
    _roster = agency_roster[_ag]
    _avg = round(sum(covered_barrett[p] for p in _roster) / len(_roster), 1) if _roster else 0.0
    stat_cards([
        ("Players", f"{len(_roster):,}", "var(--accent-teal)"),
        ("Agents", f"{len(agency_agents[_ag]):,}", "var(--accent-teal)"),
        ("Avg Barrett", f"{_avg:.1f}", "var(--accent-red)"),
    ])
    _rdf = pd.DataFrame([
        {"Player": p, "Agent(s)": ", ".join(ags), "Barrett": covered_barrett[p]}
        for p, ags in _roster.items()
    ]).sort_values("Barrett", ascending=False)
    html_table(
        _rdf,
        numeric=["Barrett"],
        aligns={"Barrett": "right"},
        formatters={"Barrett": lambda v: f"{v:.1f}"},
        height=520,
    )

# ── Books: contract value by agency, then by agent ────────────────────────────
_payload = load_agent_contracts()
# HoopsHype's season field is the START year (e.g. 2025 = the 2025-26 season),
# so include season >= the current start year to count the current season onward.
_cut = int(CURRENT_SEASON.split("-")[0])
# Corrected agency per agent, so books reflect override fixes (e.g. Drew Gross ->
# WME) rather than the raw scrape stored in the contract payload.
_agency_of = {a["agent"].lower(): a["agency"]
              for rec in agent_map.values() for a in rec["agents"] if a.get("agency")}

_abooks = agency_books(_payload, from_season=_cut, agency_of=_agency_of)
if _abooks:
    st.divider()
    st.subheader("Agency books")
    st.caption(
        f"Total value of the current and future NBA contracts each agency "
        f"represents ({CURRENT_SEASON} onward), from HoopsHype."
    )
    _adf = pd.DataFrame([
        {"Agency": b["agency"], "Clients": b["n_clients"],
         "Avg / client": (b["book_total"] / b["n_clients"]) if b["n_clients"] else 0,
         "Book value": b["book_total"]}
        for b in _abooks
    ])
    html_table(
        _adf,
        numeric=["Clients", "Avg / client", "Book value"],
        aligns={"Clients": "right", "Avg / client": "right", "Book value": "right"},
        formatters={"Avg / client": _fmt_money, "Book value": _fmt_money},
        height=420,
    )

    # Drill into one agency's book
    _amap = {b["agency"]: b for b in _abooks}
    _apick = st.selectbox("Agency", list(_amap.keys()), key="agency_book_pick", label_visibility="collapsed")
    _ab = _amap[_apick]
    _cm = _agency_contact_md(_apick)
    if _cm:
        st.caption(_cm)
    stat_cards([
        ("Clients", f"{_ab['n_clients']:,}", "var(--accent-teal)"),
        ("Book value", _fmt_money(_ab["book_total"]), "var(--accent-red)"),
        ("Avg / client", _fmt_money(_ab["book_total"] / _ab["n_clients"]) if _ab["n_clients"] else "$0", "var(--accent-teal)"),
    ])
    _abdf = pd.DataFrame([
        {"Player": c["player"], "Years": c["years"],
         "Contract value": c["total"], "Options": c["options"] or "—"}
        for c in _ab["clients"]
    ])
    html_table(
        _abdf,
        numeric=["Years", "Contract value"],
        aligns={"Years": "right", "Contract value": "right"},
        formatters={"Contract value": _fmt_money},
        height=440,
    )

_books = agent_books(_payload, from_season=_cut, agency_of=_agency_of)
if _books:
    st.divider()
    st.subheader("Agent books")
    st.caption(f"Pick an agent to see the clients and total value on their book ({CURRENT_SEASON} onward).")
    _opts = {f"{b['agent']} ({b['agency']})": b for b in _books}
    _bpick = st.selectbox("Agent", list(_opts.keys()), key="agency_db_agent_book", label_visibility="collapsed")
    _b = _opts[_bpick]
    _cm = _agency_contact_md(_b["agency"])
    if _cm:
        st.caption(_cm)
    stat_cards([
        ("Clients", f"{_b['n_clients']:,}", "var(--accent-teal)"),
        ("Active book value", _fmt_money(_b["book_total"]), "var(--accent-red)"),
    ])
    _bdf = pd.DataFrame([
        {"Player": c["player"], "Years": c["years"],
         "Contract value": c["total"], "Options": c["options"] or "—"}
        for c in _b["clients"]
    ])
    html_table(
        _bdf,
        numeric=["Years", "Contract value"],
        aligns={"Years": "right", "Contract value": "right"},
        formatters={"Contract value": _fmt_money},
        height=440,
    )

st.caption(
    "Coverage grows by running  python scripts/build_agent_db.py  (HoopsHype) "
    "and hand fixing edge cases in data/player_agents_override.csv."
)
render_footer()
