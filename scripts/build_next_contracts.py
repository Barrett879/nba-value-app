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
import io, json, logging, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
import pandas as pd, requests
import utils
from utils import normalize, _html_text
logging.disable(logging.NOTSET)

# Every season literal is DERIVED, not typed. Hardcoded, they survive a
# rollover looking correct and are not: the output filename comes from
# SEASONS[0] via season_to_espn_year, so at the next flip this would have
# written 2027-28 money into next_contracts_2028_v7.pkl, a file that is
# supposed to describe 2028-29. Spotrac still publishes a 2027-28 column, so
# the lookup would succeed, nothing would refuse, and the Next $ column would
# serve a whole league one season off. See docs/season_rollover.md.
CUR_SEASON = utils.SEASONS[0]                       # "2026-27"
_START = int(CUR_SEASON.split("-")[0])              # 2026
NEXT_LABEL = f"{_START + 1}-{str(_START + 2)[-2:]}"  # "2027-28", the season described
MARK = str(_START + 1)                              # "2027", key into `seasons`
CSV = ROOT / "data" / f"contract_years_{CUR_SEASON.replace('-', '_')}.csv"
SUP = ROOT / "data" / f"next_contract_supplement_{NEXT_LABEL.replace('-', '_')}.csv"
# How long the committed artifact may go without a rebuild before
# check_next_contracts fails. Next-season money moves on trades, waivers and
# extensions, so a week is generous and still well inside the window where a
# traded player's figure would be wrong.
MAX_AGE_DAYS = 7

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


def artifact_paths(espn_year: int):
    """(revisioned, floor, stamp) paths for the Next $ artifact.

    THE REVISIONED NAME IS NOT COSMETIC. Render's /data disk seeds a repo cache
    file only when its FILENAME is missing, so re-committing the same name
    never reaches the live site: the first copy the disk ever saw wins forever
    (utils.DATA_REV's own note, and the 347-vs-379 board incident on
    2026-11-07). next_contracts_<year>_v7.pkl has been on that disk since the
    first deploy after 2026-10-27, so wiring a nightly rebuild under that name
    would have rebuilt it every night, committed it every night, and changed
    nothing a visitor ever sees.

    So the live artifact gets utils.DATA_REV in its name, exactly as the splits
    pickle does, and a one-time un-revisioned FLOOR copy stays behind for
    _rev_path to fall back to, so a night the scrape fails degrades to
    stale-but-present rather than to a column of dashes.
    """
    base = f"next_contracts_{espn_year}_v7.pkl"
    return (utils._rev_path(base),
            utils.CACHE_DIR / base,
            utils.CACHE_DIR / f"next_contracts_{espn_year}_v7_{utils.DATA_REV}.stamp.json")


def stamp_path_for(espn_year: int) -> Path:
    """Sidecar recording when this artifact was last REBUILT. Read by
    scripts/check_next_contracts.py; see the note where it is written."""
    return artifact_paths(espn_year)[2]


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
    # _html_text, never r.text: requests falls back to latin-1 when a response
    # carries no charset, which turns every accented name into mojibake and
    # drops it from the NameIndex join. This file prices "Next $", so a dropped
    # name reads as a player with no contract next season. Flagged by
    # scripts/check_html_decoding.py, which is why that guard exists.
    for t in pd.read_html(io.StringIO(_html_text(r))):
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
    # A missing CSV must stop the run, not fall back to another season's. The
    # option TYPE comes from here; without it every team option and player
    # option silently renders as plain guaranteed money, which is a different
    # contract than the one the player actually has.
    if not CSV.exists():
        print(f"REFUSING: {CSV.relative_to(ROOT)} does not exist.")
        print(f"The season rolled to {CUR_SEASON} and this file has to be compiled "
              f"for it before Next $ can name option types. See docs/season_rollover.md.")
        return 1
    books = pd.read_csv(CSV, comment="#")
    print(f"season {CUR_SEASON}, describing {NEXT_LABEL} money (CSV mark {MARK!r})")
    print(f"{len(books)} players on the {CUR_SEASON} books, {books.team.nunique()} teams")
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
            # Spotrac's "RFA / $X" is a cap hold, so type rfa is right and the
            # CSV cannot overrule it. But when the CSV records a live option
            # for the same player the two sources genuinely disagree, and that
            # has to be REPORTED, not just resolved: this is how Tidjane Salaun
            # went from "$10.4M TO" to "RFA" between two builds with nothing in
            # the run saying so. Only the ufa branch used to log a conflict, so
            # an rfa conflict structurally could not appear in the report.
            if mark in ("T", "P"):
                disagree.append((csv_team or found_on, csv_player or disp, mark,
                                 "spotrac says RFA cap hold"))
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

    # Players the CSV has but no Spotrac page mentions at all. `unmatched` was
    # declared and printed and never appended to, so this always reported zero
    # and the builder could not see a player vanishing from all 30 pages --
    # which is exactly how LeBron James had to be found by hand and put in the
    # supplement. Walk the BOOKS, not the scrape, because the scrape by
    # definition cannot contain who is missing from it.
    _seen = set(league)
    for _, row in books.iterrows():
        keys = {normalize(str(row["player"])),
                normalize(str(row.get("spotrac_player") or row["player"]))}
        if keys & _seen:
            continue
        marks_r = {q.split(":")[0]: q.split(":")[1]
                   for q in str(row.get("seasons") or "").split("|") if ":" in q}
        if marks_r.get(MARK) in SKIP_MARK:
            continue          # two-way: nothing to quote next season anyway
        unmatched.append((str(row["team"]), str(row["player"]),
                          marks_r.get(MARK) or "no " + MARK))

    # Players Spotrac's own pages omit entirely. Filled only where the scrape
    # found nothing, so a supplement row can never override a live figure, and
    # every row cites a source -- once it reaches the column a wrong number
    # here is indistinguishable from a scraped one.
    sup_path = SUP
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

    # ---- key the artifact by the spellings the READER will use -------------
    # utils.fmt_next_contract looks this dict up with a bare
    # normalize(board_name). The artifact is keyed by Spotrac's and the CSV's
    # spellings, and those differ from the board's for a long tail of players,
    # so their money was simply invisible: measured on 2026-11-13, SEVEN board
    # players rendered the gray dash -- which the legend defines as UFA --
    # while their records sat in this file under another spelling, $95.5M in
    # all. Bobby Portis Jr. vs "bobby portis", Nic Claxton vs
    # "nicolas claxton", Herbert Jones vs "herb jones", P.J. Washington vs
    # "pj washington", Ronald Holland II vs "ron holland ii".
    #
    # utils.NameIndex is the repo's answer to exactly this and was already
    # being CONSTRUCTED here and then never read. It resolves short given names
    # to long ones only when the surname matches and exactly one player fits,
    # which is what keeps Baba Miller off Brandon Miller.
    #
    # Aliases are ADDED, never substituted: every existing key still resolves,
    # so nothing that works today can break, and a record is reachable by both
    # spellings.
    aliased = []
    try:
        bpath = utils._rev_path(
            f"raw_{CUR_SEASON.replace('-', '_')}_{utils.FORMULA_VERSION}.parquet")
        if bpath.exists():
            idx = utils.NameIndex({k: k for k in contracts})
            for bn in pd.read_parquet(bpath, columns=["Player"])["Player"].astype(str):
                k = normalize(bn)
                if k in contracts:
                    continue
                hit = idx.get(bn)
                if hit and hit in contracts:
                    contracts[k] = contracts[hit]
                    aliased.append((bn, hit, contracts[hit]["salary"],
                                    contracts[hit]["type"]))
        else:
            print(f"  note: no board at {bpath.name}, skipping the alias pass")
    except Exception as e:
        print(f"  note: alias pass skipped ({type(e).__name__}: {e})")

    from collections import Counter
    # An alias is a second KEY onto an existing record, so counting keys would
    # report seven extra "players" and double-count their types. Distinct
    # records is the honest figure; the key count is what the reader indexes.
    n_alias = len(aliased)
    n_records = len(contracts) - n_alias
    print(f"\n{n_records} players with {NEXT_LABEL} money"
          + (f", reachable under {len(contracts)} keys" if n_alias else ""))
    if aliased:
        print(f"  aliased to the board's spelling: {len(aliased)} "
              f"(${sum(a[2] for a in aliased):,.0f} that rendered as the UFA dash)")
        for bn, key, sal, t in aliased[:12]:
            print(f"     {bn}  <- {key!r}  ${sal:,.0f} {t}")
    if supplemented:
        print(f"  supplemented (Spotrac omits them): {len(supplemented)}")
        for r in supplemented:
            print("    ", r[0], f"${r[1]:,.0f}", r[2])
    _alias_keys = {normalize(a[0]) for a in aliased}
    print("  types:", dict(Counter(v["type"] for k, v in contracts.items()
                                   if k not in _alias_keys)))
    print(f"  genuine UFAs (no {MARK} season, render as the gray sentinel): "
          f"{sum(1 for _, r in books.iterrows() if MARK not in str(r.get('seasons') or ''))}")
    for label, rows in (("on a different team than the CSV records", moved),
                        (f"Spotrac has money, the CSV has no {MARK} season", csv_missing),
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

    if n_records < 200:
        print(f"\nREFUSING: only {n_records} entries. fetch_next_year_contracts "
              f"treats anything under 200 as a failed scrape, so this would never "
              f"be served anyway.")
        return 1

    espn_year = utils.season_to_espn_year(CUR_SEASON)
    _, floor_path, stamp_path = artifact_paths(espn_year)
    # Write the CURRENT revision, not whatever _rev_path currently resolves to:
    # on the first run of a new DATA_REV that would still be the previous
    # revision's file, so the new name would never be created and Render would
    # never see this build.
    path = utils.CACHE_DIR / (f"next_contracts_{espn_year}_v7_{utils.DATA_REV}.pkl")
    if write:
        utils._pkl_save(path, contracts)
        # _pkl_save swallows every write failure and returns None (utils.py's
        # bare `except Exception: logger.warning(...)`), so "it returned" is not
        # "it landed". Without this, a full disk would leave the old artifact in
        # place under a fresh stamp certifying a rebuild that never happened --
        # the age gate would then pass forever on frozen data, which is the
        # exact failure this stamp exists to catch.
        try:
            landed = utils._pkl_load(path)
        except Exception as e:
            print(f"\nREFUSING: {path.name} could not be read back ({type(e).__name__}). "
                  f"Not writing a stamp for a build that did not land.")
            return 1
        if not isinstance(landed, dict) or len(landed) != len(contracts):
            print(f"\nREFUSING: {path.name} read back as "
                  f"{len(landed) if hasattr(landed, '__len__') else type(landed).__name__}, "
                  f"expected {len(contracts)}. Not writing a stamp for it.")
            return 1
        # One-time floor under the un-revisioned name, for _rev_path to fall
        # back to. Written only when absent, for the seeding reason above.
        if not floor_path.exists():
            utils._pkl_save(floor_path, contracts)
            print(f"wrote {floor_path.name}  (fallback floor, written once)")
        # A SIDECAR, not a key inside the dict. utils._derive_next_year_contracts
        # iterates contracts.values() for its option-flag sanity gate, and
        # fmt_next_contract/classify_fa_status/team_payroll all read records by
        # name, so a "_meta" entry would be a fake player that one of them
        # eventually trips over. A separate file cannot.
        #
        # It records when the BUILDER RAN, not when the data last changed. That
        # is the distinction that matters: next-season money is mostly static,
        # so "the artifact has not changed in a week" is normal while "the
        # builder has not run in a week" means the step is dead. A git commit
        # date cannot tell those apart, which is how share_meta.json sat four
        # months stale behind a check that only counted rows.
        from collections import Counter as _C
        stamp = {
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "season": CUR_SEASON,
            "describes": NEXT_LABEL,
            "espn_year": espn_year,
            "entries": n_records,          # distinct players
            "keys": len(contracts),        # including board-spelling aliases
            "aliases": n_alias,
            "types": dict(_C(v["type"] for k, v in contracts.items()
                             if k not in _alias_keys)),
            "data_rev": utils.DATA_REV,
            "artifact": path.name,          # which file this stamp vouches for
        }
        stamp_path.write_text(json.dumps(stamp, indent=1, sort_keys=True) + "\n",
                              encoding="utf-8")
        print(f"\nwrote {path.name} and {stamp_path.name}")
    else:
        print(f"\ndry run, nothing written. Pass --write to save {path.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main("--write" in sys.argv))
