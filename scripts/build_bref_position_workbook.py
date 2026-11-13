#!/usr/bin/env python3
"""Every player's most-played positions this season, straight from Basketball Reference.

    /opt/anaconda3/bin/python3 scripts/build_bref_position_workbook.py [end_year]

BBRef's play-by-play table estimates the share of a player's minutes spent at
each of the five spots. That is the closest thing to a measured answer for
"what does this guy actually play", as opposed to what a roster page labels him.

One request for the whole league (scripts/fetch_bref_position_pct.season_pcts),
cached under cache/bref_pbp/, so a re-run costs nothing and BBRef's rate limit
is never a factor.

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
which is also arbitrary but is at least stated and stable across runs.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from openpyxl import Workbook                                   # noqa: E402
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # noqa: E402
from openpyxl.utils import get_column_letter                    # noqa: E402

import fetch_bref_position_pct as B                             # noqa: E402

SPOTS = B.SPOTS                                   # PG SG SF PF C
OUT = ROOT / "exports" / "BBRef_Positions_2026-27.xlsx"
FONT = "Arial"
SRC = "https://www.basketball-reference.com/leagues/NBA_{}_play-by-play.html"
# A second position is named only when it reaches this share of the minutes.
# Below it the player is listed at one spot: 96 percent SF is not a SF/PF.
THRESHOLD = 0.10

THIN = Side(style="thin", color="D9D9D9")
HEAD_FILL = PatternFill("solid", fgColor="16233F")
NOTE_FILL = PatternFill("solid", fgColor="FFF7E6")
INPUT_FILL = PatternFill("solid", fgColor="FFFF00")


def season_label(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[-2:]}"


def build(end_year: int, threshold: float = THRESHOLD) -> int:
    label = season_label(end_year)
    print(f"fetching BBRef play-by-play for {label} (NBA_{end_year}) ...")
    data = B.season_pcts(end_year)
    if not data:
        print("REFUSING: no play-by-play rows came back. BBRef may not publish "
              "this table for the season yet, or the page moved.")
        return 1

    rows = []
    for _key, rec in data.items():
        name = rec.get("name") or _key
        shares = [float(rec.get(s, 0.0) or 0.0) for s in SPOTS]
        if sum(shares) <= 0:
            continue                       # no position data: nothing to rank
        rows.append({"player": name, "team": rec.get("team", ""),
                     "mp": B._num(rec.get("mp")), "shares": shares})

    # A partial table is worse than none: a player missing from this file reads
    # as a player with no position, and nothing downstream could tell the
    # difference. Refuse rather than ship a short one.
    if len(rows) < 300:
        print(f"REFUSING: only {len(rows)} players with position data. A full "
              f"season table runs 450+; this looks like a truncated page.")
        return 1

    # Rank each player's five shares. Ties break toward the more guard-ward
    # spot by taking SPOTS order as the secondary key, so a 50/50 row is stable
    # across runs instead of depending on dict order.
    for r in rows:
        order = sorted(range(5), key=lambda j: (-r["shares"][j], j))
        a, b = order[0], order[1]
        r["top"], r["top_share"] = SPOTS[a], round(r["shares"][a] / 100.0, 4)
        # A share of zero is not a second position, it is the absence of one.
        # Tyrese Maxey at 100 percent PG would otherwise be listed "2nd
        # position: SG, 0%", which reads like evidence for something nobody
        # observed. Blank says what the data says.
        second = round(r["shares"][b] / 100.0, 4)
        r["second"] = SPOTS[b] if second > 0 else ""
        r["second_share"] = second if second > 0 else None
        r["label"] = (f"{SPOTS[a]}/{SPOTS[b]}"
                      if second >= threshold else SPOTS[a])

    rows.sort(key=lambda r: (-r["mp"], r["player"]))
    print(f"{len(rows)} players with {label} play-by-play position data")

    wb = Workbook()
    ws = wb.active
    ws.title = f"Positions {label}"

    # ---- header ------------------------------------------------------------
    headers = (["Player", "Team", "Minutes",
                "Top position", "Top share",
                "2nd position", "2nd share",
                "Positions"] + SPOTS)
    ws.append(headers)
    for i, _h in enumerate(headers, start=1):
        c = ws.cell(row=1, column=i)
        c.font = Font(name=FONT, bold=True, size=10, color="FFFFFF")
        c.fill = HEAD_FILL
        c.alignment = Alignment(horizontal="center", vertical="center",
                                wrap_text=True)
        c.border = Border(bottom=THIN)

    FIRST = 2
    for n, r in enumerate(rows):
        x = FIRST + n
        ws.cell(row=x, column=1, value=r["player"])
        ws.cell(row=x, column=2, value=r["team"])
        ws.cell(row=x, column=3, value=r["mp"])
        ws.cell(row=x, column=4, value=r["top"])
        ws.cell(row=x, column=5, value=r["top_share"])
        ws.cell(row=x, column=6, value=r["second"])
        ws.cell(row=x, column=7, value=r["second_share"])
        ws.cell(row=x, column=8, value=r["label"])
        for j, v in enumerate(r["shares"]):
            ws.cell(row=x, column=9 + j, value=round(v / 100.0, 4))

    last = FIRST + len(rows) - 1
    for x in range(FIRST, last + 1):
        for i in range(1, 14):
            c = ws.cell(row=x, column=i)
            c.font = Font(name=FONT, size=10)
            c.border = Border(bottom=THIN)
            if i == 3:
                c.number_format = "#,##0"
            elif i in (5, 7) or 9 <= i <= 13:
                c.number_format = "0%"
            if i in (2, 4, 6, 8):
                c.alignment = Alignment(horizontal="center")
        # a thin sample is the one thing that makes a row untrustworthy, so
        # grey the whole row rather than leaving the reader to check Minutes
        if rows[x - FIRST]["mp"] < 100:
            for i in range(1, 14):
                ws.cell(row=x, column=i).font = Font(name=FONT, size=10,
                                                     color="909090", italic=True)

    widths = {"A": 26, "B": 7, "C": 9, "D": 13, "E": 10, "F": 13, "G": 10, "H": 12}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for i in range(9, 14):
        ws.column_dimensions[get_column_letter(i)].width = 7
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:M{last}"

    # ---- notes -------------------------------------------------------------
    nt = wb.create_sheet("Notes")
    nt.column_dimensions["A"].width = 30
    nt.column_dimensions["B"].width = 96
    lines = [
        ("Basketball Reference position shares", ""),
        ("Season", label),
        ("Source", SRC.format(end_year)),
        ("Fetched", time.strftime("%Y-%m-%d %H:%M")),
        ("Players", len(rows)),
        ("", ""),
        ("Second-spot threshold", threshold),
        ("", "Below this share the player is listed at one spot only. To cut it "
             "differently, re-run: scripts/build_bref_position_workbook.py "
             f"{end_year} 0.15"),
        ("", ""),
        ("What the shares are",
         "BBRef's estimate, from play-by-play, of the share of each player's minutes "
         "spent at each of the five spots. It is measured, not a roster label."),
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
         "stable across runs."),
        ("Greyed rows",
         "Under 100 minutes this season. The position estimate is built from "
         "almost nothing; treat it as a placeholder."),
        ("", ""),
        ("SAMPLE SIZE, read this",
         f"These are {label} shares only, and the season is young. Check the "
         f"Minutes column before trusting a row: a player with 20 minutes has a "
         f"position estimate built from almost nothing."),
    ]
    for i, (k, v) in enumerate(lines, start=1):
        a = nt.cell(row=i, column=1, value=k)
        b = nt.cell(row=i, column=2, value=v)
        a.font = Font(name=FONT, size=10, bold=bool(k) and not str(k).startswith(" "))
        b.font = Font(name=FONT, size=10)
        b.alignment = Alignment(wrap_text=True, vertical="top")
        if k in ("Basketball Reference position shares", "EDIT THIS",
                 "SAMPLE SIZE, read this"):
            a.fill = NOTE_FILL
            b.fill = NOTE_FILL
    nt["B8"].number_format = "0%"
    nt["B8"].font = Font(name=FONT, size=10, color="0000FF", bold=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(rows)} players)")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    yr = int(args[0]) if args else 2027
    th = float(args[1]) if len(args) > 1 else THRESHOLD
    sys.exit(build(yr, th))
