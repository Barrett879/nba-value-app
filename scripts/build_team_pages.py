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
    # Market value ($M) per player = the board's 2025-26 projected market salary,
    # keyed by normalized name. Used to grade each 2026-27 contract.
    df = utils.build_ranked_projected(utils.SEASONS[0])
    mval = {utils.normalize(str(r["Player"])): float(r["projected_salary"]) / 1e6
            for _, r in df.iterrows()}

    teams = {}
    for abbr, t in ros["teams"].items():
        players = []
        for p in t["players"]:
            sal = p.get("salary")
            val = mval.get(utils.normalize(str(p["n"])))
            vd = round(sal - val, 1) if (val is not None and sal is not None) else None
            players.append({
                "n": p["n"], "role": p["role"], "pos": p.get("pos", ""),
                "barrett": p.get("barrett"),
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
        "value_season": ros["value_season"],
        "contract_season": ros["contract_season"],
        "cap_M": ros["cap_M"], "apron2_M": ros["apron2_M"],
        "teams": dict(sorted(teams.items())),
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT.name}  ({len(teams)} teams, {OUT.stat().st_size // 1024}KB, "
          f"value {payload['value_season']} / roster {payload['contract_season']})")
    for probe in ("LAL", "DEN", "OKC"):
        t = teams.get(probe)
        if t:
            top = t["players"][0]
            print(f"  {probe} {t['name']}: {t['size']} players, ${t['payroll']}M payroll; "
                  f"top ${top['salary']}M {top['n']} ({top['role']})")


if __name__ == "__main__":
    main()
