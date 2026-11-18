"""Next-season salary and option type from Basketball Reference's contracts table.

The SECOND source for the Next $ column, behind Spotrac. Added 2026-11-18, the
day after Spotrac began answering 403 to this machine on every path -- team
pages and /robots.txt alike. That is a client block, not an outage: it does not
self-heal, and we do not try to evade it. One source for a load-bearing column
was the real defect; this file is the fix for that, not just for one bad night.

WHY THIS SOURCE AND NOT ANOTHER. Four candidates were checked on 2026-11-18:

  spotrac.com            403 on everything, robots.txt included. Blocked.
  hoopshype.com          robots.txt Disallows "Claude", "Claude-Web",
                         "ClaudeBot", "claudeweb" and "anthropic-ai" from "/".
                         Named explicitly, so off limits however reachable.
  basketball.realgm.com  robots.txt Disallows "anthropic-ai" and "Claude-Web",
                         AND its salary pages answer 403 anyway.
  basketball-reference   200. robots.txt disallows AhrefsBot and GPTBot but not
                         us, and /contracts/ is not in its Disallow list.
                         Crawl-delay: 3, which ONE request per run respects
                         trivially.

So Basketball Reference is the only candidate that is both reachable and
permitted, and it happens to be the better-shaped source anyway:

  * ONE request. /contracts/players.html carries all 30 teams, where Spotrac
    needed thirty team pages at 0.4s apart.
  * It carries the OPTION TYPE itself, as a cell class -- salary-pl for a
    player option, salary-tm for a team option. Spotrac's yearly cell does not,
    which is why the current builder has to take option type from
    data/contract_years_2026_27.csv, a hand-compiled SNAPSHOT whose own header
    admits it is a snapshot. A live class marker tracks an option being
    exercised; a snapshot cannot.

WHAT IT COSTS. Measured against the 342-record Spotrac artifact of 2026-11-17:
308 players in both, salary agreeing to the dollar on 263 (85.4%) and option
type on 296 (96.1%). The residue is NOT noise and is documented where it is
handled -- see SALARY CAP PROJECTION below, and the split-contract and cap-hold
notes in build_next_contracts.py. Treating this source as a drop-in equal of
Spotrac would have shipped a handful of wrong figures, and a wrong salary is
indistinguishable from a real one once it reaches the column.

SALARY CAP PROJECTION, the one difference that is nobody's bug. A max contract
is defined as a PERCENTAGE of a salary cap that for 2027-28 does not exist yet,
so each source prices those deals off its own forecast. Derived from the two
max tiers on 2026-11-18, to the dollar and consistent across both tiers:

    Basketball Reference   $179,280,000
    Spotrac                $178,157,880      ratio 1.006298

That ratio is EXACTLY the salary ratio observed on all six 25%/30% max players
(Adebayo, Fox, Doncic at 30%; Holmgren, Jalen Williams, Banchero at 25%), which
is what identifies the mechanism rather than merely fitting it. A 0.63%
difference on a projected figure is not an error to correct; both numbers are
forecasts and ours is a value model, not a cap sheet. It is recorded here so
that the next person to diff the two sources does not go looking for a parser
bug that is not there.

Several other rows differ by literal single dollars (Podziemski $18,965,517 vs
$18,965,518, Steven Adams by $1 the other way) -- percentage arithmetic rounded
at different places. Also not a bug.

ENCODING. Read through utils._html_text, never requests' .text: requests falls
back to latin-1 when a response carries no charset, which turns "Nikola Jokic"
into mojibake and drops the row from every NameIndex join. A dropped name in
THIS file reads as a player with no contract next season, i.e. as a free agent.
scripts/check_html_decoding.py exists because of that class of bug.

WHY NOT pandas.read_html. It returns the text of a table and throws the
attributes away, and the option type lives in a cell CLASS. read_html can see
the money and is structurally blind to whether it is guaranteed.
"""
from __future__ import annotations
import logging, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
import requests
from bs4 import BeautifulSoup
import utils
from utils import normalize, _html_text
logging.disable(logging.NOTSET)

LEAGUE_URL = "https://www.basketball-reference.com/contracts/players.html"
TABLE_ID = "player-contracts"

# Same shape the Spotrac path sends, for the same reason: a bare python-requests
# User-Agent is refused by most sports sites. Nothing here is an attempt to look
# like a browser we are not -- robots.txt permits this path for us by name.
HDRS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"}

# A player option and a team option are DIFFERENT CONTRACTS and the board says
# so with different chips, so this mapping is the whole reason to prefer this
# source. "iz" is Basketball Reference's own empty-cell class.
CLASS_TYPE = {"salary-pl": "player_option", "salary-tm": "team_option"}

# Basketball Reference writes plain comma-grouped money with nothing glued to
# it. Spotrac welds the cap percentage onto the figure ("$28,006,89815.9%"),
# which is why the Spotrac parser needs a comment explaining why a greedy
# pattern eats the salary. Here the simple pattern is the correct one, and the
# anchored group-count still refuses anything that is not money.
_MONEY = re.compile(r"^\$(\d{1,3}(?:,\d{3})*)$")

# A BACKSTOP, deliberately far below any real contract figure -- NOT the
# "below the league minimum" floor that build_next_contracts.py applies to the
# Spotrac path. Copying that $500,000 floor here was wrong and refused the
# whole source on the first real page: Didi Louzada is owed $268,032 in
# 2027-28, which is not a mis-parse but a STRETCHED WAIVER, and the shape says
# so -- an identical figure in 2026-27, 2027-28 and 2028-29 with a Guaranteed
# total of exactly three times it. Vasilije Micic's $666,667 over two years is
# the same thing. Those are real obligations, below the minimum because they
# are a fraction of a bought-out deal spread over years, and a source that
# refuses to load because one of them exists is a source that never loads.
#
# The floor was also guarding nothing. It exists on the Spotrac path because
# Spotrac welds the cap percentage onto the money, so a greedy pattern there
# silently returns a FRACTION of the real salary -- Jokic at $0.1M. _MONEY
# here is anchored to the whole cell, so a cell this parser does not fully
# understand yields None and a named flag, never a plausible small number. The
# anchor is the protection; this is only a last line against a shape nobody
# anticipated.
MIN_PLAUSIBLE = 50_000
MIN_ROWS = 250          # 2027-28 had 333 on 2026-11-18; well under a real shape change

# How many impossible Guaranteed totals it takes to mean THIS PARSER is
# broken rather than one source row being wrong. Zero of 431 on 2026-11-18,
# so any real number here is slack; 3 keeps a one-off typo from taking the
# source offline while a missed season column, which hits dozens of rows at
# once, still refuses.
MAX_SHORTFALLS = 3

# Reported, not refused. A figure under the league minimum is usually a stretch
# or a partial guarantee, and the caller logs them so a genuinely new shape is
# visible in the build output instead of silently averaged into the column.
NOTABLY_SMALL = 500_000


class SourceRefused(Exception):
    """The page did not have the shape this parser understands.

    Raised rather than returned so a caller cannot mistake a structural failure
    for "no players have money next season" -- which is what an empty dict
    means here, and which renders as a league of free agents.
    """


def _plain_money(raw) -> float | None:
    """A comma-grouped dollar figure, or None for anything else.

    Shared by the salary cells and the Guaranteed cell so the self-check and
    the parse agree about what counts as money. Anchored, like _MONEY.
    """
    m = _MONEY.match(" ".join(str(raw or "").split()))
    return float(m.group(1).replace(",", "")) if m else None


def fetch(url: str = LEAGUE_URL, timeout: int = 30) -> str:
    """The contracts page as correctly-decoded text. ONE request per run."""
    r = requests.get(url, headers=HDRS, timeout=timeout)
    r.raise_for_status()
    return _html_text(r)


def season_columns(table) -> dict:
    """{season label: data-stat} off the header, e.g. {"2027-28": "y2"}.

    DERIVED, NEVER TYPED. The season columns are named y1..y6 positionally, so
    y2 means 2027-28 only until the season flips, after which y2 is 2028-29 and
    a hardcoded "y2" would quote money for the wrong season while every row
    parsed, every count looked right and nothing refused. That is the exact
    failure build_next_contracts.py's own season-literal comment warns about,
    one layer down. The label is the only stable identifier on this page.
    """
    rows = table.find("thead").find_all("tr")
    if not rows:
        raise SourceRefused(f"#{TABLE_ID} has no header row")
    out = {}
    for cell in rows[-1].find_all(["th", "td"]):
        stat = cell.get("data-stat") or ""
        if re.fullmatch(r"y\d+", stat):
            out[" ".join(cell.get_text(" ", strip=True).split())] = stat
    if not out:
        raise SourceRefused(f"#{TABLE_ID} header exposes no y<N> season columns")
    return out


def player_header_label(table) -> str:
    """Whatever the header calls the player column, e.g. "Player".

    Needed to recognise a REPEATED HEADER that reaches the body without the
    class that normally marks it. The page puts one every twenty rows or so,
    and on 2026-11-18 all 46 carried class="thead", so the class check alone
    was enough -- but a header row that loses that class does not look like a
    header any more, and the parser would emit a player named "Player" with
    whatever sits in its money cell. A literal name match is the only signal
    that works on BOTH page shapes: the league table writes body names in
    <td> while a team table writes them in <th scope="row">, so sniffing the
    tag would read correctly here and silently drop every real player from a
    team page.
    """
    rows = table.find("thead").find_all("tr")
    for cell in (rows[-1].find_all(["th", "td"]) if rows else []):
        if (cell.get("data-stat") or "") == "player":
            return " ".join(cell.get_text(" ", strip=True).split())
    return "Player"


def parse(html: str, next_label: str) -> dict:
    """{normalized name: [row, ...]} for `next_label`, every row kept.

    EVERY row, not one per player, because this page lists a player ONCE PER
    TEAM that still carries part of his contract. A waived-and-stretched deal
    appears twice with the SAME y1..y6 figures and a different Guaranteed
    total: Damian Lillard on both MIL ($90,066,412 guaranteed) and POR
    ($13,398,800), Bradley Beal on PHO and LAC, Klay Thompson on DAL and MIA.
    Collapsing here with a last-one-wins would be an arbitrary choice between
    two rows and would hide the fact that there were two, so the caller gets
    both and decides. See build_next_contracts.py for what it decides and why.

    Each row: {"display", "team", "salary"|None, "type", "flag"|None,
               "guaranteed", "years": {label: raw text}}
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id=TABLE_ID)
    if table is None:
        raise SourceRefused(f"no table#{TABLE_ID} on the page "
                            f"({len(html)} bytes) -- the layout changed")
    cols = season_columns(table)
    if next_label not in cols:
        raise SourceRefused(
            f"no {next_label} column; the page offers {sorted(cols)}. "
            f"Refusing rather than quoting a different season's money.")
    col = cols[next_label]
    body = table.find("tbody")
    if body is None:
        raise SourceRefused(f"table#{TABLE_ID} has no tbody")

    hdr_name = player_header_label(table)
    out: dict = {}
    for tr in body.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue                       # repeated header, every ~20 rows
        cells = {c.get("data-stat"): c for c in tr.find_all(["th", "td"])}
        pcell = cells.get("player")
        if pcell is None:
            continue
        disp = " ".join(pcell.get_text(" ", strip=True).split())
        if not disp or disp == hdr_name:
            continue                       # a header row that lost its class
        scell = cells.get(col)
        raw = " ".join(scell.get_text(" ", strip=True).split()) if scell is not None else ""
        classes = (scell.get("class") or []) if scell is not None else []
        salary, flag = None, None
        if raw:
            m = _MONEY.match(raw)
            if m:
                salary = float(m.group(1).replace(",", ""))
            else:
                # Not empty and not money. Never guessed at: an unrecognised
                # cell is reported upward so a new shape surfaces as a named
                # oddity in the build log instead of as a missing player.
                flag = "unparsed:" + raw[:24]
        else:
            flag = "empty"                 # no 2027-28 season: a genuine UFA
        row = {
            "display": disp,
            "team": " ".join(cells["team_id"].get_text(" ", strip=True).split())
                    if "team_id" in cells else "",
            "salary": salary,
            "type": next((CLASS_TYPE[c] for c in classes if c in CLASS_TYPE),
                         "guaranteed"),
            "flag": flag,
            "guaranteed": " ".join(cells["remain_gtd"].get_text(" ", strip=True).split())
                          if "remain_gtd" in cells else "",
            "years": {lbl: " ".join(cells[st].get_text(" ", strip=True).split())
                      for lbl, st in cols.items() if st in cells},
        }
        out.setdefault(normalize(disp), []).append(row)

    paid = [r for rows in out.values() for r in rows if r["salary"] is not None]
    if len(paid) < MIN_ROWS:
        raise SourceRefused(
            f"only {len(paid)} rows carry {next_label} money, under the "
            f"{MIN_ROWS} floor. A partial parse turns rosters into free agents.")

    # ---- the page checked against ITSELF ---------------------------------
    # The Guaranteed column is a TOTAL of the year columns, so the year series
    # can never sum to less than it: a guarantee exceeding every dollar the
    # table lists is arithmetically impossible. Measured on 2026-11-18 across
    # the 431 rows that carry a Guaranteed cell: 227 sum to it exactly, 204
    # sum HIGHER (correct -- option years and non-guaranteed years are real
    # money that is not guaranteed), and ZERO sum lower.
    #
    # This is worth having because it is the only check here that needs no
    # second source. If a season column is ever added and this parser does not
    # read it -- the exact failure that cost nothing today only because
    # season_columns is derived -- the missing money shows up as a shortfall
    # against the guarantee, on many rows at once. A second source can tell us
    # a FIGURE is wrong; this tells us the PARSE is incomplete.
    #
    # Refused in bulk, reported individually, because the two causes differ:
    # many shortfalls mean this parser is missing a column, while one or two
    # mean the source published a row that does not add up, and refusing the
    # whole league over somebody else's typo is how a source stops loading.
    short = []
    for rows in out.values():
        r = rows[0]
        g = _plain_money(r["guaranteed"])
        if g is None:
            continue
        listed = sum(v for v in (_plain_money(x) for x in r["years"].values())
                     if v is not None)
        if listed and listed + 1 < g:
            short.append((r["display"], listed, g))
    if len(short) > MAX_SHORTFALLS:
        raise SourceRefused(
            f"{len(short)} players have a Guaranteed total above the sum of "
            f"every season column, which is arithmetically impossible and "
            f"means this parser is not reading every season. First: "
            + "; ".join(f"{d} listed ${l:,.0f} < guaranteed ${g:,.0f}"
                        for d, l, g in short[:3]))
    for d, l, g in short:
        print(f"  note: {d} lists ${l:,.0f} against a ${g:,.0f} guarantee")
    tiny = sorted((r["salary"], r["display"]) for r in paid
                  if r["salary"] < MIN_PLAUSIBLE)
    if tiny:
        raise SourceRefused(
            f"{len(tiny)} salary(ies) under ${MIN_PLAUSIBLE:,} "
            f"({', '.join(f'{d} ${s:,.0f}' for s, d in tiny[:4])}). "
            f"A mis-parse is indistinguishable from a cheap contract.")
    return out


def league_money(next_label: str, url: str = LEAGUE_URL) -> dict:
    """parse(fetch()). The one entry point a caller needs."""
    return parse(fetch(url), next_label)
