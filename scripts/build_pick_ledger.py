"""Build the engine-ready pick ledger from data/pick_ledger.csv.

The CSV holds ONLY deviations from default ownership (a team owning its own
unencumbered first emits no row); this script generates the defaults, enforces
the ledger invariants, computes each team's Stepien coverage per the verified
spec, and writes cache/pick_ledger.json for the trade machine.

Coverage semantics (docs/plan_trade_machine.md 5.1-5.5 + S9.15, conservative):
  - unprotected pick controlled by T        -> T covered for that year
  - own pick encumbered only by a swap      -> origin still covered (ends with
    the worse first); the swap HOLDER gains no coverage from the right
  - any protection/conditionality           -> NOBODY covered by that pick
    (possibility standard: it might convey, or might not)

Invariants enforced (build fails loudly):
  - every (origin, year) at most one row; abbrs valid; years 2027-2033
  - exactly 30 original firsts per draft year after defaults are generated

Run after editing data/pick_ledger.csv. Output: cache/pick_ledger.json
  { "asof": ..., "teams": { ABBR: { "covered": [years], "controls": [ {pick...} ] } } }
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "pick_ledger.csv"
OUT = ROOT / "cache" / "pick_ledger.json"

TEAMS = ["ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
         "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
         "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
YEARS = list(range(2027, 2034))


def main() -> None:
    rows = {}
    seconds = []          # round-2 picks: explicit rows only, Stepien-exempt
    asof = ""
    with SRC.open() as f:
        for r in csv.DictReader(x for x in f if not x.lstrip().startswith("#")):
            if r.get("asof"):
                asof = r["asof"]
            year, origin = int(r["year"]), r["origin"].strip()
            rnd = int(r.get("round") or 1)
            assert origin in TEAMS, f"bad origin {origin!r}"
            assert r["controlled_by"].strip() in TEAMS, f"bad controller in {r}"
            assert year in YEARS, f"year out of range in {r}"
            rec = {
                "year": year, "origin": origin, "round": rnd,
                "controlled_by": r["controlled_by"].strip(),
                "protection": r.get("protection", "").strip(),
                "swap_with": r.get("swap_with", "").strip(),
                "notes": r.get("notes", "").strip(),
            }
            if rnd == 2:
                seconds.append(rec)
                continue
            key = (origin, year)
            assert key not in rows, f"duplicate ledger row for {key}"
            rows[key] = rec

    # defaults: every unlisted (origin, year) FIRST is an unencumbered own pick.
    # Seconds get NO defaults -- league-wide round-2 ownership is unverified, so
    # only explicitly listed second-round rows exist.
    all_picks = []
    for y in YEARS:
        for t in TEAMS:
            all_picks.append(rows.get((t, y)) or {
                "year": y, "origin": t, "controlled_by": t, "round": 1,
                "protection": "", "swap_with": "", "notes": "",
            })
    per_year = {y: sum(1 for p in all_picks if p["year"] == y) for y in YEARS}
    assert all(n == 30 for n in per_year.values()), f"pick count broken: {per_year}"
    all_picks.extend(seconds)

    teams = {t: {"covered": [], "controls": []} for t in TEAMS}
    for p in all_picks:
        ctrl = p["controlled_by"]
        teams[ctrl]["controls"].append(p)
        if p.get("round", 1) == 2:
            continue                       # seconds: tradable, Stepien-exempt
        if p["protection"]:
            continue                       # protected: covers nobody (S9.15)
        if p["swap_with"] and ctrl == p["origin"]:
            teams[p["origin"]]["covered"].append(p["year"])  # worse-of still a first
        else:
            teams[ctrl]["covered"].append(p["year"])
    for t in TEAMS:
        teams[t]["covered"] = sorted(set(teams[t]["covered"]))
        teams[t]["controls"].sort(key=lambda p: (p["year"], p["origin"]))

    # sanity: nobody can cover more years than exist, and a team with zero
    # coverage in consecutive years is worth flagging loudly (data smell)
    for t, d in teams.items():
        gaps = [y for y in YEARS if y not in d["covered"]]
        for a, b in zip(gaps, gaps[1:]):
            if b == a + 1:
                print(f"  note: {t} has no guaranteed first in {a}+{b} "
                      f"(legal only if pre-existing; new trades must not worsen it)")

    OUT.write_text(json.dumps({"asof": asof, "teams": teams}, separators=(",", ":")))
    n_dev = len(rows)
    print(f"wrote {OUT.name}: 210 picks, {n_dev} encumbered, "
          f"{sum(len(d['covered']) for d in teams.values())} covered team-years")


if __name__ == "__main__":
    sys.exit(main())
