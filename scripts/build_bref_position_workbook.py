#!/usr/bin/env python3
"""Every player's most-played positions, measured, against the ones HoopsValue shows.

    /opt/anaconda3/bin/python3 scripts/build_bref_position_workbook.py [end_year] [threshold]

BBRef's play-by-play table estimates the share of a player's minutes spent at
each of the five spots. That is the closest thing to a measured answer for
"what does this guy actually play", as opposed to what a roster page labels him.
This writes three position answers side by side for every rostered player:

  this season   BBRef's 2026-27 shares, one league-wide request
  career        the same shares weighted over every season back to 2003-04
  HoopsValue    what the site itself shows, and which file decided it

One request per season (scripts/fetch_bref_position_pct), all cached under
cache/bref_pbp/, so a re-run costs nothing and BBRef's rate limit never bites.

CAREER is aggregated from the season pages, NOT scraped per player: BBRef's own
career row is a minutes-weighted mean of the seasons, so summing 24 league
pages reproduces it for 23 requests instead of one page per player. Verified
against BBRef's published career row for the 11 player pages already in the
cache: all 11 agree to within 1 percentage point and 0 career minutes.

TWO HoopsValue position columns, because the site has two and they disagree.
The Rankings board and player hub resolve position through app.py's chain
(position_overrides.csv, then the 2K/reviewed/override trio, then ESPN's coarse
Guard/Forward/Center); the /Rosters depth chart reads position_overrides.csv
then master_roster.csv. A single column would have had to pick one and hide the
split, which runs to 85 of 463 players. "Pos source" names the file that
decided the board value, so the disagreements sort into causes instead of
piling up as noise.

VALUES, NOT FORMULAS, deliberately. The derived columns could be INDEX/MATCH
over the shares, but LibreOffice is not installed on this machine, so no
formula written here could be machine-verified before shipping, and an
unverified formula in the headline column is worse than a computed value: it
also reads as EMPTY in Quick Look, Numbers' preview and anything reading cached
values, because openpyxl writes formulas with no cached result. The shares sit
beside every answer so any row can be checked by eye, and the second-spot
threshold is a script argument, so a different cut is one re-run away.

TIES: BBRef reports whole percentages, so two spots can be exactly equal
(Brandin Podziemski, 50 SG / 50 SF on 2026-11-13). Sorting by share alone would
leave the order to Python's dict insertion and read as arbitrary. Ties break
toward the more guard-ward spot (PG before SG before SF before PF before C),
which is also arbitrary but is at least stated and stable across runs. The
season block and the career block run through the SAME ranker, so the tie rule
and the zero-second rule cannot drift apart between them.
"""
from __future__ import annotations
import csv
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

warnings.filterwarnings("ignore")

from openpyxl import Workbook                                   # noqa: E402
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # noqa: E402
from openpyxl.utils import get_column_letter                    # noqa: E402

import fetch_bref_position_pct as B                             # noqa: E402
import team_suitors as _ts                                      # noqa: E402
import utils                                                    # noqa: E402
from utils import NameIndex, normalize                           # noqa: E402

SPOTS = B.SPOTS                                   # PG SG SF PF C
OUT = ROOT / "exports" / "BBRef_Positions_2026-27.xlsx"
FONT = "Arial"
SRC = "https://www.basketball-reference.com/leagues/NBA_{}_play-by-play.html"
# A second position is named only when it reaches this share of the minutes.
# Below it the player is listed at one spot: 96 percent SF is not a SF/PF.
THRESHOLD = 0.10
# Play-by-play position data starts in 1996-97, but no active player debuted
# before 2003-04 (LeBron), so that is where a complete career can begin. A
# player whose aggregate starts AT this floor and has no earlier season is
# complete by construction; verify() re-checks the assumption every run.
CAREER_FROM = 2004
# Below these the estimate rests on almost nothing, and the block is greyed.
THIN_SEASON_MP = 100
THIN_CAREER_MP = 500

THIN = Side(style="thin", color="D9D9D9")
HEAD_FILL = PatternFill("solid", fgColor="16233F")
CAREER_FILL = PatternFill("solid", fgColor="2B3A55")
HV_FILL = PatternFill("solid", fgColor="3C2B55")
NOTE_FILL = PatternFill("solid", fgColor="FFF7E6")
BAND_FILL = PatternFill("solid", fgColor="F3F6FB")
NO_FILL = PatternFill("solid", fgColor="FCE4E4")
GREY = "909090"


def season_label(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[-2:]}"


# ---------------------------------------------------------------- ranking ----
def rank(shares: list, threshold: float) -> dict:
    """The top two spots behind a list of five percentages.

    ONE ranker for both the season block and the career block, so the tie rule
    and the zero-second rule are the same code and cannot diverge.

    Ties break toward the more guard-ward spot by taking SPOTS order as the
    secondary key, so a 50/50 row is stable across runs instead of depending on
    dict order. A share of zero is not a second position, it is the absence of
    one: Tyrese Maxey at 100 percent PG would otherwise be listed "2nd
    position: SG, 0%", which reads like evidence for something nobody observed.
    """
    if not shares or sum(shares) <= 0:
        return {"top": "", "top_share": None, "second": "", "second_share": None,
                "label": ""}
    order = sorted(range(5), key=lambda j: (-shares[j], j))
    a, b = order[0], order[1]
    second = round(shares[b] / 100.0, 4)
    return {"top": SPOTS[a], "top_share": round(shares[a] / 100.0, 4),
            "second": SPOTS[b] if second > 0 else "",
            "second_share": second if second > 0 else None,
            "label": f"{SPOTS[a]}/{SPOTS[b]}" if second >= threshold else SPOTS[a]}


def primary(pos: str) -> str:
    """The first-listed spot of a HoopsValue position string: 'SG/SF' -> 'SG'.

    The first position is the load-bearing one. position_overrides.csv says so
    in its own header: it decides which depth column a player lands in, which
    is the whole point of most of those rows."""
    for part in str(pos).replace("|", "/").split("/"):
        if part.strip():
            return part.strip().upper()
    return ""


# ------------------------------------------------------- HoopsValue's view ----
def _csv_rows(path: Path) -> list:
    """DictReader over a repo CSV, skipping the comment banner every one has."""
    try:
        with open(path) as fh:
            return list(csv.DictReader(l for l in fh if l.strip()
                                       and not l.lstrip().startswith("#")))
    except Exception:
        return []


def hoopsvalue_positions(espn_year: int) -> tuple:
    """What the site shows, and which file decided it.

    Returns (board_idx, roster_idx, source_idx, conditional, counts) where the
    first three are NameIndex lookups keyed by any spelling of a player's name.

    board_idx reproduces app.py's chain for the Rankings board and player hub:
    position_overrides.csv wins, then team_suitors.resolve_position over the
    2K / reviewed / override trio, then master_roster.csv's verified `pos`,
    then group_flex of ESPN's coarse Guard/Forward/Center. roster_idx is what
    the /Rosters depth chart reads: the same override file, then the same
    master_roster `pos`, so since the roster layer was added to the site on
    2026-11-13 the two columns agree far more often than they used to.

    Lookups go through NameIndex, never a bare normalize() key: master_roster
    says "C.J. McCollum" where BBRef prints "CJ McCollum", and an exact key
    reports a player who is on the roster as absent from it.

    A position_overrides row carrying alt_pos + alt_when_out swaps only while
    the named player is listed OUT. That is resolved against a live injury feed
    on the site and is deliberately NOT applied here: a spreadsheet that bakes
    in today's injury list is wrong tomorrow and says nothing about it. The
    normal position is used and the affected players are named on Notes.
    """
    ov, cond = {}, {}
    for r in _csv_rows(ROOT / "data" / "position_overrides.csv"):
        nm, pos = (r.get("player") or "").strip(), (r.get("pos") or "").strip()
        if not nm or not pos:
            continue
        ov[nm] = pos
        alt, dep = (r.get("alt_pos") or "").strip(), (r.get("alt_when_out") or "").strip()
        if alt and dep:
            cond[nm] = (pos, alt, dep)
    ov_idx = NameIndex(ov)

    roster = {}
    for r in _csv_rows(ROOT / "data" / "master_roster.csv"):
        nm = (r.get("player") or "").strip()
        if nm:
            roster[nm] = {"team": (r.get("team") or "").strip(),
                          "pos": (r.get("pos") or "").strip(),
                          "kind": (r.get("kind") or "").strip()}
    roster_idx = NameIndex(roster)

    # The three files behind resolve_position, kept SEPARATE so a row's source
    # can be named. load_player_positions() layers them into one dict and the
    # provenance is gone, which is exactly what the "Pos source" column needs.
    import pandas as pd

    def names_in(path) -> set:
        try:
            df = pd.read_csv(path, comment="#", skip_blank_lines=True)
        except Exception:
            return set()
        if "name" not in df.columns or "positions" not in df.columns:
            return set()
        return {normalize(str(n).strip()) for n, q in zip(df["name"], df["positions"])
                if str(n).strip() and str(q).strip() and str(q).lower() != "nan"}

    L_2k = names_in(_ts.POS_2K_PATH)
    L_rev = names_in(_ts.POS_REVIEWED_PATH)
    L_ovr = names_in(_ts.POS_OVERRIDE_PATH)
    pos_map = _ts.load_player_positions()
    roster_layer = _ts.roster_position_index()
    try:
        coarse = utils.fetch_bref_positions(espn_year, cache_v=3)
    except Exception:
        coarse = {}

    def resolve(name: str) -> tuple:
        n = normalize(name)
        hit = ov_idx.get(name)
        if hit:
            return hit, "position_overrides.csv (hand-set)"
        pos = _ts.resolve_position(name, coarse.get(n, ""), pos_map, roster_layer)
        if n in L_ovr:
            return pos, "player_positions_override.csv"
        if n in L_rev:
            return pos, "player_positions_reviewed.csv"
        if n in L_2k:
            return pos, "player_positions_2k.csv (NBA 2K26)"
        # The roster layer, which outranks ESPN's group because ESPN only ever
        # says Guard / Forward / Center and that names no spot. Added to the
        # site on 2026-11-13; this column has to follow the chain or the one
        # instrument built to audit the SF/PF default reports it as still
        # present for the 56 players it just fixed.
        if roster_layer.get(name):
            return pos, "master_roster.csv (verified roster)"
        if coarse.get(n):
            return pos, f"ESPN group only ({coarse.get(n)})"
        # group_flex("") returns SF/PF, so a player the chain knows NOTHING
        # about is shown at small forward. That is a default, not a reading,
        # and the column has to say so or the value reads as an answer.
        return pos, "no data (SF/PF default)"

    return ov_idx, roster_idx, resolve, cond, roster


# ---------------------------------------------------------------- building ----
def collect(end_year: int, threshold: float) -> tuple:
    """One row per rostered player, plus anyone with minutes who is not on the
    roster yet. Returns (rows, stats)."""
    label = season_label(end_year)
    print(f"fetching BBRef play-by-play for {label} (NBA_{end_year}) ...")
    season = B.season_pcts(end_year)
    if not season:
        raise SystemExit("REFUSING: no play-by-play rows came back. BBRef may "
                         "not publish this table for the season yet, or the "
                         "page moved.")
    print(f"  {len(season)} players with {label} minutes")
    print(f"aggregating career shares, {season_label(CAREER_FROM)} to {label} ...")
    career, hist = B.career_pcts(CAREER_FROM, end_year)
    print(f"  {len(career)} players across {end_year - CAREER_FROM + 1} seasons")

    _ov, _roster_idx, resolve, cond, roster = hoopsvalue_positions(end_year)
    season_idx = NameIndex({(r.get("name") or k): r for k, r in season.items()})

    # The spine is the UNION: everyone BBRef has 2026-27 minutes for, plus
    # every rostered non-free-agent who has not played yet. Driving it off the
    # season table alone dropped 62 players under contract, Jimmy Butler and
    # Kristaps Porzingis among them, which is precisely the case a career
    # column exists to answer.
    rows, seen = [], set()

    def add(name: str, team: str, kind: str, srec: dict | None) -> None:
        key = normalize(name)
        if key in seen:
            return
        seen.add(key)
        sh = ([float((srec or {}).get(s, 0.0) or 0.0) for s in SPOTS]
              if srec else [])
        if sh and sum(sh) <= 0:
            sh = []
        cr = career.get(key) or {}
        csh = [float(cr.get(s, 0.0) or 0.0) for s in SPOTS] if cr else []
        if csh and sum(csh) <= 0:
            csh = []
        hv, src = resolve(name)
        rst = _ov.get(name) or (roster.get(name) or _roster_idx.get(name) or {}).get("pos", "")
        s_rank, c_rank = rank(sh, threshold), rank(csh, threshold)
        rows.append({
            "player": name,
            "team": team or (srec or {}).get("team", ""),
            "kind": kind,
            "hv": hv, "roster": rst, "src": src,
            "mp": B._num((srec or {}).get("mp")) if srec else None,
            "cmp": int(cr["mp"]) if cr.get("mp") else None,
            "shares": sh, "cshares": csh,
            "s": s_rank, "c": c_rank,
            "first_season": min((e for e, _ in hist.get(key, [])), default=None),
            "agree_s": ("" if not s_rank["top"] or not hv else
                        "yes" if primary(hv) == s_rank["top"] else "no"),
            "agree_c": ("" if not c_rank["top"] or not hv else
                        "yes" if primary(hv) == c_rank["top"] else "no"),
        })

    for key, srec in season.items():
        nm = srec.get("name") or key
        hit = _roster_idx.get(nm) or {}
        add(nm, hit.get("team", "") or srec.get("team", ""),
            hit.get("kind", "") or "not on master_roster", srec)
    unplayed = 0
    for nm, rec in roster.items():
        if rec.get("kind") == "free_agent" or season_idx.get(nm):
            continue
        add(nm, rec.get("team", ""), rec.get("kind", ""), None)
        unplayed += 1

    # A partial table is worse than none: a player missing from this file reads
    # as a player with no position, and nothing downstream could tell the
    # difference. Refuse rather than ship a short one.
    played = sum(1 for r in rows if r["shares"])
    if played < 300:
        raise SystemExit(f"REFUSING: only {played} players with {label} position "
                         f"data. A full season table runs 450+; this looks like "
                         f"a truncated page.")

    # played first, by minutes; then the unplayed, by career minutes, so a
    # starter on the shelf sits above a two-way who has never played.
    rows.sort(key=lambda r: (r["mp"] is None, -(r["mp"] or 0), -(r["cmp"] or 0),
                             r["player"]))
    stats = {"played": played, "unplayed": unplayed, "rows": len(rows),
             "career_n": len(career), "label": label, "cond": cond}
    return rows, stats


COLS = [
    # (header, width, kind) - kind drives number format and alignment
    ("Player", 26, "text"),
    ("Team", 7, "mid"),
    ("Pos (HoopsValue)", 16, "mid"),
    ("Pos (roster)", 12, "mid"),
    ("Pos source", 30, "text"),
    ("Positions 26-27", 15, "mid"),
    ("Positions career", 16, "mid"),
    ("Primary agrees 26-27", 12, "mid"),
    ("Primary agrees career", 12, "mid"),
    ("Minutes 26-27", 11, "int"),
    ("Top position", 11, "mid"),
    ("Top share", 9, "pct"),
    ("2nd position", 11, "mid"),
    ("2nd share", 9, "pct"),
    ("PG", 6, "pct"), ("SG", 6, "pct"), ("SF", 6, "pct"),
    ("PF", 6, "pct"), ("C", 6, "pct"),
    ("Career minutes", 12, "int"),
    ("Career PG", 8, "pct"), ("Career SG", 8, "pct"), ("Career SF", 8, "pct"),
    ("Career PF", 8, "pct"), ("Career C", 8, "pct"),
]
# which columns belong to which block, for the greying rules
SEASON_BLOCK = list(range(6, 7)) + list(range(10, 20))     # F, J..S
CAREER_BLOCK = [7] + list(range(20, 26))                   # G, T..Y


def build(end_year: int, threshold: float = THRESHOLD) -> int:
    rows, stats = collect(end_year, threshold)
    label = stats["label"]

    wb = Workbook()
    ws = wb.active
    ws.title = f"Positions {label}"

    ws.append([c[0] for c in COLS])
    for i, (h, _w, _k) in enumerate(COLS, start=1):
        c = ws.cell(row=1, column=i)
        c.font = Font(name=FONT, bold=True, size=10, color="FFFFFF")
        c.fill = (CAREER_FILL if h.startswith("Career") or h == "Positions career"
                  or h == "Primary agrees career"
                  else HV_FILL if h.startswith("Pos ") else HEAD_FILL)
        c.alignment = Alignment(horizontal="center", vertical="center",
                                wrap_text=True)
        c.border = Border(bottom=THIN)

    FIRST = 2
    for n, r in enumerate(rows):
        x = FIRST + n
        vals = [r["player"], r["team"], r["hv"], r["roster"], r["src"],
                r["s"]["label"], r["c"]["label"], r["agree_s"], r["agree_c"],
                r["mp"], r["s"]["top"], r["s"]["top_share"],
                r["s"]["second"], r["s"]["second_share"]]
        vals += [round(v / 100.0, 4) for v in r["shares"]] or [None] * 5
        vals += [r["cmp"]]
        vals += [round(v / 100.0, 4) for v in r["cshares"]] or [None] * 5
        thin_s = not r["shares"] or (r["mp"] or 0) < THIN_SEASON_MP
        thin_c = not r["cshares"] or (r["cmp"] or 0) < THIN_CAREER_MP
        for i, v in enumerate(vals, start=1):
            c = ws.cell(row=x, column=i, value=v)
            kind = COLS[i - 1][2]
            grey = (thin_s and i in SEASON_BLOCK) or (thin_c and i in CAREER_BLOCK)
            c.font = Font(name=FONT, size=10,
                          color=GREY if grey else "000000", italic=grey)
            c.border = Border(bottom=THIN)
            if kind == "int":
                c.number_format = "#,##0"
            elif kind == "pct":
                c.number_format = "0%"
            if kind in ("mid", "pct", "int"):
                c.alignment = Alignment(horizontal="center")
            # the career block gets a tint so the eye can tell the two
            # measured answers apart without reading the header every time
            if i in CAREER_BLOCK:
                c.fill = BAND_FILL
            if i in (8, 9) and v == "no":
                c.fill = NO_FILL

    last = FIRST + len(rows) - 1
    for (h, w, _k), i in zip(COLS, range(1, len(COLS) + 1)):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLS))}{last}"

    _notes(wb, end_year, threshold, rows, stats)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(rows)} rows: "
          f"{stats['played']} with {label} minutes, {stats['unplayed']} without)")
    return 0


def _notes(wb, end_year: int, threshold: float, rows: list, stats: dict) -> None:
    from collections import Counter
    label = stats["label"]
    src = Counter(r["src"] for r in rows if r["src"])
    split = [r for r in rows if r["hv"] and r["roster"] and r["hv"] != r["roster"]]
    dis_s = sum(1 for r in rows if r["agree_s"] == "no")
    dis_c = sum(1 for r in rows if r["agree_c"] == "no")
    nt = wb.create_sheet("Notes")
    nt.column_dimensions["A"].width = 30
    nt.column_dimensions["B"].width = 100
    lines = [
        ("Position shares, measured against HoopsValue", ""),
        ("Season", label),
        ("Source", SRC.format(end_year)),
        ("Career span", f"{season_label(CAREER_FROM)} through {label} "
                        f"({end_year - CAREER_FROM + 1} seasons)"),
        ("Fetched", time.strftime("%Y-%m-%d %H:%M")),
        ("Rows", f"{stats['rows']} ({stats['played']} with {label} minutes, "
                 f"{stats['unplayed']} under contract but yet to play)"),
        ("", ""),
        ("Second-spot threshold", threshold),
        ("", "Below this share the player is listed at one spot only. To cut it "
             "differently, re-run: scripts/build_bref_position_workbook.py "
             f"{end_year} 0.15"),
        ("", ""),
        ("What the shares are",
         "BBRef's estimate, from play-by-play, of the share of each player's minutes "
         "spent at each of the five spots. It is measured, not a roster label."),
        ("Career shares",
         "A minutes-weighted mean over every season back to "
         f"{season_label(CAREER_FROM)}, aggregated from the league pages rather "
         "than scraped per player. BBRef's own career row is the same weighted "
         "mean, so the two agree: checked against BBRef's published career row "
         "for the 11 player pages in the cache, all 11 within 1 percentage "
         "point and 0 career minutes. Play-by-play data starts in 1996-97 and "
         "no active player debuted before 2003-04, so no active career is clipped."),
        ("Career of a rookie",
         "Equals his 26-27 line, because that is his whole career. Read Career "
         "minutes before treating the two columns as independent evidence."),
        ("Why rows do not total 100",
         "BBRef rounds each share to a whole percent, so a row can total 99 or 101."),
        ("Top two",
         "Computed from the five shares and written as values, not formulas. "
         "LibreOffice is not installed on this machine, so no formula could be "
         "machine-verified before shipping, and an unverified formula also reads "
         "as empty in any previewer. The shares sit beside every answer."),
        ("Ties",
         "Two spots can be exactly equal. Ties break toward the more guard-ward "
         "spot (PG, SG, SF, PF, C in that order): arbitrary, but stated and "
         "stable across runs. The season and career blocks share one ranker."),
        ("", ""),
        ("Pos (HoopsValue)", "WHAT THE SITE SHOWS"),
        ("", "The position on the Rankings board and the player hub. It resolves "
             "through position_overrides.csv first, then the 2K / reviewed / "
             "override trio, then ESPN's coarse Guard-Forward-Center group."),
        ("Pos (roster)",
         "What the /Rosters depth chart reads: the same override file, then "
         "master_roster.csv's scraped `pos`. It is roster data, usually more "
         "specific than the board's."),
        ("Pos source",
         "Which file decided Pos (HoopsValue). This is the column to filter on: "
         "it sorts the disagreements into causes."),
        ("The two columns disagree",
         f"for {len(split)} of {sum(1 for r in rows if r['hv'] and r['roster'])} "
         f"players with both populated. Far fewer than before 2026-11-13, when "
         f"the board had no roster layer and fell back to a GROUP for anyone "
         f"the curated files missed: group_flex returns SF/PF for an unknown "
         f"player and for ESPN's 'Forward', and SG/PG for 'Guard'. What is "
         f"left is mostly a curated file disagreeing with the roster feed. "
         f"Filter Pos source to see which file answered each row."),
        ("", ""),
        ("Primary agrees",
         "The FIRST-listed spot of Pos (HoopsValue) against BBRef's top spot. "
         "The first position is the load-bearing one: position_overrides.csv's "
         "own header says it decides which depth column a player lands in. A "
         "full-string match would call 'PG' against 'PG/SG' a disagreement, "
         "which it is not. Disagreements: "
         f"{dis_s} against {label}, {dis_c} against career."),
        ("Conditional overrides NOT applied",
         "position_overrides.csv can swap a position only while a named player "
         "is listed OUT, resolved against a live injury feed on the site. A "
         "spreadsheet that baked in today's injury list would be wrong tomorrow "
         "and say nothing about it, so the normal position is used here. The "
         "rows: " + "; ".join(f"{k} is {v[0]}, becomes {v[1]} while {v[2]} is out"
                              for k, v in stats["cond"].items()) + "."),
        ("Free agents excluded",
         "master_roster rows marked free_agent are left out: the file answers "
         "what rostered players play."),
        ("Greyed cells",
         f"The 26-27 block greys under {THIN_SEASON_MP} minutes this season; the "
         f"career block greys under {THIN_CAREER_MP} career minutes. The two "
         f"rules are independent, so a thin season does not grey a long career."),
        ("", ""),
        ("SAMPLE SIZE, read this",
         f"The {label} shares are {label} only, and the season is young. Check "
         f"Minutes 26-27 before trusting that block: a player with 20 minutes "
         f"has a position estimate built from almost nothing. The career block "
         f"is the one to lean on for a veteran, and says nothing extra for a "
         f"rookie."),
        ("", ""),
        ("Pos source breakdown", ""),
    ]
    for s, n in src.most_common():
        lines.append(("", f"{n:4}  {s}"))
    for i, (k, v) in enumerate(lines, start=1):
        a = nt.cell(row=i, column=1, value=k)
        b = nt.cell(row=i, column=2, value=v)
        a.font = Font(name=FONT, size=10, bold=bool(k))
        b.font = Font(name=FONT, size=10)
        b.alignment = Alignment(wrap_text=True, vertical="top")
        if k in ("Position shares, measured against HoopsValue",
                 "SAMPLE SIZE, read this", "Pos (HoopsValue)"):
            a.fill = NOTE_FILL
            b.fill = NOTE_FILL
    nt["B8"].number_format = "0%"
    nt["B8"].font = Font(name=FONT, size=10, color="0000FF", bold=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    yr = int(args[0]) if args else 2027
    th = float(args[1]) if len(args) > 1 else THRESHOLD
    sys.exit(build(yr, th))
