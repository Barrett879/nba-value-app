"""Build cache/next_contracts_<espn_year>_v7.pkl: what every player is owed NEXT season.

    /opt/anaconda3/bin/python3 -u scripts/build_next_contracts.py [--write]

Why this exists: the live derivation cannot work during a season. It calls
fetch_salaries(espn_year + 1), and ESPN only publishes a season's salaries once
that season has begun, so from October to the following autumn it returns zero
rows. The 2026_v7 file on disk only exists because it was built on 2026-10-19,
the day before the season flipped, while ESPN still counted 2026-27 as "next".

After the 2026-10-20 flip the code began asking for next_contracts_2027_v7.pkl,
which had never been built. No stale copy meant no fallback, the sanity gate
returned a 4-entry dict uncached, and the Next $ column served "—" for all 362
players for a week. Nothing logged, because an empty Next $ is indistinguishable
from a league of free agents.

So the money comes from Spotrac's per-team yearly tables, which carry a column
per future season, and the OPTION TYPE comes from data/contract_years_2026_27.csv,
which was compiled from those same pages and verified. Two sources for the same
fact, cross-checked against each other, because that is the only kind of check
that catches a wrong-but-plausible value.

Absence is meaningful and must stay meaningful: a player with no entry renders
as the gray sentinel, which the legend defines as UFA. 182 of the 492 players on
the 2026-27 books genuinely have no 2027-28 season, and they SHOULD show it.
"""
from __future__ import annotations
import io, logging, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
import pandas as pd, requests
import utils
from utils import normalize
logging.disable(logging.NOTSET)

NEXT_LABEL = "2027-28"           # the season this file describes
CSV = ROOT / "data" / "contract_years_2026_27.csv"
MARK = "2027"                    # key into the CSV's `seasons` field

HDRS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"}

SLUG = {
    "ATL": "atlanta-hawks", "BOS": "boston-celtics", "BKN": "brooklyn-nets",
    "CHA": "charlotte-hornets", "CHI": "chicago-bulls", "CLE": "cleveland-cavaliers",
    "DAL": "dallas-mavericks", "DEN": "denver-nuggets", "DET": "detroit-pistons",
    "GSW": "golden-state-warriors", "HOU": "houston-rockets", "IND": "indiana-pacers",
    "LAC": "la-clippers", "LAL": "los-angeles-lakers", "MEM": "memphis-grizzlies",
    "MIA": "miami-heat", "MIL": "milwaukee-bucks", "MIN": "minnesota-timberwolves",
    "NOP": "new-orleans-pelicans", "NYK": "new-york-knicks", "OKC": "oklahoma-city-thunder",
    "ORL": "orlando-magic", "PHI": "philadelphia-76ers", "PHX": "phoenix-suns",
    "POR": "portland-trail-blazers", "SAC": "sacramento-kings", "SAS": "san-antonio-spurs",
    "TOR": "toronto-raptors", "UTA": "utah-jazz", "WAS": "washington-wizards",
}

# The CSV's per-season markers. N is non-guaranteed: real money on the books that
# is not locked in, and the four-type vocabulary fmt_next_contract speaks has no
# word for it, so it renders as plain money. M is a MUTUAL option, which neither
# the TO nor the PO chip describes honestly, so it also renders plain rather than
# asserting a mechanism that is not the one in the contract. Q is undocumented in
# the CSV's own legend; checked against Spotrac directly rather than guessed, and
# all three carry ordinary salaries (McDaniels $28,006,898, Proctor $2,525,901),
# so it is plain money too.
TYPE_OF = {"G": "guaranteed", "N": "guaranteed", "M": "guaranteed",
           "Q": "guaranteed", "T": "team_option", "P": "player_option"}
SKIP_MARK = {"W"}                # two-way: no next-season salary to quote

# Spotrac glues the cap percentage straight onto the money with no separator:
# "$28,006,89815.9%" is $28,006,898 at 15.9%. Stripping a trailing percentage
# first does NOT work -- the greedy \d+ eats back into the salary ("89815.9%"
# comes off and leaves "$28,006,"), which turned Jokic into $0.1M and McDaniels
# into $0.0M in the first build. The comma grouping is the real delimiter: take
# $ + up to three digits + as many ",ddd" groups as follow, and stop. The
# percentage never has a comma, so it cannot be swallowed.
_MONEY = re.compile(r"\$(\d{1,3}(?:,\d{3})*)")
_MONEY_ABBR = re.compile(r"\$(\d+(?:\.\d+)?)\s*([MK])")


def parse_cell(raw) -> tuple[float | None, str | None]:
    """(salary, marker) from one Spotrac season cell.

    Shapes seen: '$30,000,00017.0%', 'UFA / $31.5M17.9%', 'Two-Way', ''.
    'UFA / $X' is a CAP HOLD, not money owed -- quoting it as next season's
    salary would say a free agent is under contract for it.
    """
    s = " ".join(str(raw or "").split())
    if not s or s.lower() in {"nan", "-", "--"}:
        return None, "empty"
    if "two-way" in s.lower():
        return None, "two_way"
    up = s.upper()
    if up.startswith("UFA") or "UFA /" in up:
        return None, "ufa"
    if up.startswith("RFA") or "RFA /" in up:
        # "RFA / $15.1M8.6%" is a cap hold on a restricted free agent, not a
        # contract. fmt_next_contract renders type "rfa" as the RFA chip with
        # no figure, which is the honest reading: nobody is owed that money.
        # Parsed as a salary it came out as $15, which is how the floor check
        # found these twenty rows.
        return None, "rfa"
    m = _MONEY.search(s)
    if m:
        return float(m.group(1).replace(",", "")), None
    m = _MONEY_ABBR.search(s)
    if m:
        mult = 1_000_000 if m.group(2) == "M" else 1_000
        return float(m.group(1)) * mult, None
    return None, "unparsed:" + s[:24]


def spotrac_name(raw) -> str:
    """The player's actual name out of Spotrac's doubled cell.

    The cell reads "Johnson  Jalen Johnson" -- surname, two spaces, full name --
    so normalizing it whole matches nothing, which is how the first run of this
    script reported 302 players "not found on the team page" while every one of
    them was right there. Split on the run of 2+ spaces and keep the last field;
    that survives hyphens ("Alexander-Walker  Nickeil Alexander-Walker") and
    initials ("McCollum  C.J. McCollum").
    """
    parts = re.split(r"\s{2,}", str(raw or "").strip())
    return " ".join(parts[-1].split())


def team_money(abbr: str) -> dict:
    """{normalized name: (salary|None, marker)} from one team's yearly page.

    Every table on the page is read, but the FIRST occurrence of a name wins:
    the active roster table comes before the dead-money and cap-hold tables, and
    a cap hold must never shadow a real contract.
    """
    url = f"https://www.spotrac.com/nba/{SLUG[abbr]}/yearly/"
    r = requests.get(url, headers=HDRS, timeout=25)
    r.raise_for_status()
    out: dict = {}
    seen_active = False
    for t in pd.read_html(io.StringIO(r.text)):
        cols = {str(c).strip(): c for c in t.columns}
        if NEXT_LABEL not in cols:
            continue
        # The first table carrying a season column is the active roster; the
        # Dead Money and Cap Hold tables follow it. A dead-money row has no
        # future salary, so letting one shadow a real contract would read as
        # "no money next season" for a player who has plenty.
        kind = "active" if not seen_active else "other"
        seen_active = True
        name_col = t.columns[0]
        for _, row in t.iterrows():
            disp = spotrac_name(row[name_col])
            if not disp or disp.lower().startswith("incomplete roster"):
                continue
            key = normalize(disp)
            if key in out and out[key][0] == "active":
                continue
            out[key] = (kind, abbr, disp, *parse_cell(row[cols[NEXT_LABEL]]))
    return out


def main(write: bool) -> int:
    books = pd.read_csv(CSV, comment="#")
    print(f"{len(books)} players on the 2026-27 books, {books.team.nunique()} teams")
    print(f"money from Spotrac /yearly/, option type from {CSV.name}\n")

    scraped: dict = {}
    for i, abbr in enumerate(sorted(SLUG), 1):
        try:
            scraped[abbr] = team_money(abbr)
            print(f"  [{i:2d}/30] {abbr}  {len(scraped[abbr]):3d} rows")
        except Exception as e:
            print(f"  [{i:2d}/30] {abbr}  FAILED {type(e).__name__}: {str(e)[:60]}")
            scraped[abbr] = {}
        time.sleep(0.4)

    failed = [a for a, v in scraped.items() if not v]
    if failed:
        print(f"\nREFUSING: {len(failed)} team page(s) gave nothing: {failed}")
        print("A partial scrape would silently turn those rosters into free agents.")
        return 1

    # ONE league-wide map, not thirty team-scoped ones. The CSV records the
    # team a player was on when it was compiled, and players move: the first
    # run of this looked each player up only on his CSV team and reported 11
    # "not found", among them LeBron James and Brandon Ingram, every one of
    # whom was on another team's page with his money intact. A team-scoped
    # join silently drops exactly the players whose situation changed.
    league: dict = {}
    for abbr, rows in scraped.items():
        for key, rec in rows.items():
            if key in league and league[key][0] == "active" and rec[0] != "active":
                continue
            league[key] = rec
    ni = utils.NameIndex({k: k for k in league})
    print(f"\n{len(league)} distinct names across all 30 pages "
          f"({sum(1 for v in league.values() if v[0] == 'active')} on active rosters)")

    contracts: dict = {}
    no_money, unmatched, disagree, skipped, moved, csv_missing = [], [], [], [], [], []

    # Which players have 2027-28 money is decided by SPOTRAC, not by the CSV.
    # The CSV is a snapshot and it goes stale: it records Stephen Curry's deal
    # as ending in 2026-27, while Spotrac's own Warriors page shows $55,700,000
    # for 2027-28. Driving membership off the CSV skipped him entirely and
    # rendered the gray sentinel, which reads as "free agent" for a player owed
    # $55.7M. The CSV's job is the option TYPE, which Spotrac's yearly cell does
    # not carry; when it has nothing to say, a plain cell means plain money.
    by_name = {}
    for _, row in books.iterrows():
        marks = {q.split(":")[0]: q.split(":")[1]
                 for q in str(row.get("seasons") or "").split("|") if ":" in q}
        for key in {normalize(str(row["player"])),
                    normalize(str(row.get("spotrac_player") or row["player"]))}:
            by_name[key] = (str(row["team"]), str(row["player"]), marks.get(MARK))

    for key, (kind, found_on, disp, salary, flag) in sorted(league.items()):
        if kind != "active":
            continue                      # dead money and cap holds owe nothing
        csv_team, csv_player, mark = by_name.get(key, (None, None, None))
        if flag == "rfa":
            contracts[normalize(csv_player or disp)] = {"salary": 0.0, "type": "rfa"}
            continue
        if flag == "ufa":
            if mark is not None:
                # Spotrac says free agent, the CSV says a season exists. Two
                # readings of one fact; record it rather than pick a winner.
                disagree.append((csv_team or found_on, csv_player or disp, mark,
                                 "spotrac says UFA"))
            continue
        if salary is None:
            if mark is not None and mark not in SKIP_MARK:
                no_money.append((csv_team or found_on, csv_player or disp, mark, flag))
            continue
        if mark in SKIP_MARK:
            skipped.append((found_on, disp, mark))
            continue
        if mark is None:
            # On a roster with money next season, but the CSV has no 2027 entry:
            # either it predates the deal or the player is not in it at all.
            csv_missing.append((found_on, disp, salary))
        elif csv_team and csv_team != found_on:
            moved.append((csv_player, f"{csv_team} -> {found_on}", kind))
        t = TYPE_OF.get(mark, "guaranteed")
        contracts[normalize(csv_player or disp)] = {"salary": float(salary), "type": t}

    # Players Spotrac's own pages omit entirely. Filled only where the scrape
    # found nothing, so a supplement row can never override a live figure, and
    # every row cites a source -- once it reaches the column a wrong number
    # here is indistinguishable from a scraped one.
    sup_path = ROOT / "data" / "next_contract_supplement_2027_28.csv"
    supplemented = []
    if sup_path.exists():
        sup = pd.read_csv(sup_path, comment="#")
        for _, srow in sup.iterrows():
            key = normalize(str(srow["player"]))
            if key in contracts:
                continue
            contracts[key] = {"salary": float(srow["salary"]),
                              "type": str(srow["type"]).strip()}
            supplemented.append((str(srow["player"]), float(srow["salary"]),
                                 str(srow["type"]).strip()))

    from collections import Counter
    print(f"\n{len(contracts)} players with {NEXT_LABEL} money")
    if supplemented:
        print(f"  supplemented (Spotrac omits them): {len(supplemented)}")
        for r in supplemented:
            print("    ", r[0], f"${r[1]:,.0f}", r[2])
    print("  types:", dict(Counter(v["type"] for v in contracts.values())))
    print(f"  genuine UFAs (no {MARK} season, render as the gray sentinel): "
          f"{sum(1 for _, r in books.iterrows() if MARK not in str(r.get('seasons') or ''))}")
    for label, rows in (("on a different team than the CSV records", moved),
                        ("Spotrac has money, the CSV has no 2027 season", csv_missing),
                        ("two-way, no salary to quote", skipped),
                        ("on the books but not found on the team page", unmatched),
                        ("found but no parsable money", no_money),
                        ("THE TWO SOURCES DISAGREE", disagree)):
        if rows:
            print(f"\n  {label}: {len(rows)}")
            for r in rows[:12]:
                print("    ", "  ".join(str(x) for x in r))

    # A mis-parse does not look like an error, it looks like a cheap contract.
    # The 2026-27 minimum is a bit over $1.2M and two-ways are excluded above,
    # so anything under $500k here is a parsing failure wearing a salary.
    tiny = sorted(((k, v["salary"]) for k, v in contracts.items()
                   if v["type"] != "rfa" and v["salary"] < 500_000),
                  key=lambda x: x[1])
    if tiny:
        print(f"\n  IMPLAUSIBLY SMALL ({len(tiny)}), almost certainly mis-parsed:")
        for k, v in tiny[:12]:
            print(f"     {k}  ${v:,.0f}")
        print("\nREFUSING: a mis-parsed salary is indistinguishable from a real "
              "cheap contract once it reaches the column.")
        return 1
    paid = [v["salary"] for v in contracts.values() if v["type"] != "rfa"]
    lo, hi = min(paid), max(paid)
    print(f"  salary range ${lo:,.0f} .. ${hi:,.0f}")

    if len(contracts) < 200:
        print(f"\nREFUSING: only {len(contracts)} entries. fetch_next_year_contracts "
              f"treats anything under 200 as a failed scrape, so this would never "
              f"be served anyway.")
        return 1

    espn_year = utils.season_to_espn_year(utils.SEASONS[0])
    path = utils._dc_path(f"next_contracts_{espn_year}_v7.pkl")
    if write:
        utils._pkl_save(path, contracts)
        print(f"\nwrote {path.name}")
    else:
        print(f"\ndry run, nothing written. Pass --write to save {path.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main("--write" in sys.argv))
