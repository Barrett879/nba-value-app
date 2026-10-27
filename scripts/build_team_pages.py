"""Precompute crawlable team-page data: each team's 2026-27 roster, reality-first.

Source of truth is scripts/export_fa_spreadsheet.py, which assembles the
2026-27 rosters from real contracts + REPORTED real signings (the verified
tracker), with the model projecting only still-unsigned spots, and dumps
cache/team_rosters_2627.json. This script joins each player's market value so
the page can show a value verdict, then writes cache/team_pages.json for
serve.py to render at /team/<ABBR>.

Roster and salaries are 2026-27 (real where reported, marked projected where
not); value/Barrett Score reflect 2025-26 (the latest season actually played).

Pipeline: run export_fa_spreadsheet.py FIRST (it dumps the roster JSON), then
this. Re-run both after the FA projection / signings update.

Output: cache/team_pages.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import utils  # noqa: E402

ROS = ROOT / "cache" / "team_rosters_2627.json"
OUT = ROOT / "cache" / "team_pages.json"


def main() -> None:
    ros = json.loads(ROS.read_text(encoding="utf-8"))
    # Market value ($M) per player = the projected market salary from the season
    # this file SAYS it values players on, which is the last COMPLETE one.
    #
    # Taken from the roster JSON, deliberately, not from SEASONS[0]. Those were
    # the same thing when this was written and stopped being the same thing at
    # the 2026-10-20 rollover: SEASONS[0] became 2026-27, so a rebuild would
    # have graded every contract against a four-game sample while the payload
    # went on labelling itself value_season 2025-26. The label and the numbers
    # now come from one place and cannot disagree.
    value_season = ros["value_season"]
    df = utils.build_ranked_projected(value_season)

    # Hand-set position overrides (data/position_overrides.csv), the same file
    # the Rosters depth chart honours. They are EDITORIAL: a judgement about the
    # role a player actually fills, which the scraped roster position often gets
    # wrong. Applied HERE rather than in each page so one file feeds the Trade
    # Machine, Redraft and Rosters alike.
    #
    # It matters more here than anywhere else, because the trade card shows only
    # the PRIMARY position, so an override that merely reorders a pair is the
    # whole displayed value. Without this, 39 players rendered the wrong single
    # position: Kevin Durant as SG, Devin Booker as SG, Lauri Markkanen as PF,
    # Jaren Jackson Jr. as C.
    #
    # Only the base `pos` is taken. The alt_pos / alt_when_out pair is keyed to
    # the live injury feed and reshapes a depth chart while someone is out; a
    # trade card is about the player, not tonight's rotation.
    import csv as _csv
    overrides = utils.NameIndex()
    ov_path = ROOT / "data" / "position_overrides.csv"
    if ov_path.exists():
        for _r in _csv.DictReader(
                ln for ln in ov_path.read_text(encoding="utf-8").splitlines()
                if not ln.lstrip().startswith("#")):
            if (_r.get("player") or "").strip() and (_r.get("pos") or "").strip():
                overrides.add(_r["player"].strip(), _r["pos"].strip())
    applied = 0
    # NameIndex, not a plain dict: the roster feed writes "C.J. McCollum" and
    # "Nicolas Claxton" where the board has "CJ McCollum" and "Nic Claxton", and
    # an exact-key miss left those players with no market value at all.
    mval = utils.NameIndex({str(r["Player"]): float(r["projected_salary"]) / 1e6
                            for _, r in df.iterrows()})

    teams = {}
    for abbr, t in ros["teams"].items():
        players = []
        for p in t["players"]:
            sal = p.get("salary")
            val = mval.get(p["n"])
            vd = round(sal - val, 1) if (val is not None and sal is not None) else None
            _ov = overrides.get(p["n"])
            _pos = p.get("pos", "")
            if _ov and _ov != _pos:
                _pos = _ov
                applied += 1
            players.append({
                "n": p["n"], "role": p["role"], "pos": _pos,
                "barrett": p.get("barrett"), "bs_yr": p.get("bs_yr"),
                "salary": sal,
                "value": round(val, 1) if val is not None else None,
                "vd": vd,  # salary - market value; >0 overpay, <0 bargain
                "real": p.get("real", True),  # False = projected move for an unsigned FA
            })
        players.sort(key=lambda x: -(x["salary"] or 0))  # stars first
        teams[abbr] = {"abbr": abbr, "name": t["name"], "size": t["size"],
                       "payroll": t["total"], "room": t["room"], "players": players,
                       "unsigned": t.get("unsigned", [])}

    payload = {
        "value_season": value_season,
        "contract_season": ros["contract_season"],
        "cap_M": ros["cap_M"], "apron2_M": ros["apron2_M"],
        "teams": dict(sorted(teams.items())),
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT.name}  ({len(teams)} teams, {OUT.stat().st_size // 1024}KB, "
          f"value {payload['value_season']} / roster {payload['contract_season']})")
    print(f"  position overrides applied: {applied}")
    for probe in ("LAL", "DEN", "OKC"):
        t = teams.get(probe)
        if t:
            top = t["players"][0]
            print(f"  {probe} {t['name']}: {t['size']} players, ${t['payroll']}M payroll; "
                  f"top ${top['salary']}M {top['n']} ({top['role']})")


if __name__ == "__main__":
    main()
