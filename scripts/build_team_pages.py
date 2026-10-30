"""Precompute crawlable team-page data: each team's 2026-27 roster, reality-first.

Source of truth is scripts/export_fa_spreadsheet.py, which assembles the
2026-27 rosters from real contracts + REPORTED real signings (the verified
tracker), with the model projecting only still-unsigned spots, and dumps
cache/team_rosters_2627.json. This script joins each player's market value so
the page can show a value verdict, then writes cache/team_pages.json for
serve.py to render at /team/<ABBR>.

Roster and salaries are 2026-27 (real where reported, marked projected where
not); value and Barrett Score reflect the CURRENT season, with a player who has
not played yet carrying his last complete one, tagged per player in bs_yr. The
season actually used is written into the payload as value_season, and every
caption derives from that field rather than naming a season of its own.

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
    # Market value ($M) per player = the projected market salary from the
    # CURRENT season, on Barrett's instruction (2026-10-29: "set the hub to
    # current season, along with every other feature in the site").
    #
    # This was previously taken from the roster JSON's own value_season, to
    # stop a DATA_REV bump silently re-grading every contract against a
    # four-game sample while the payload still said 2025-26. That risk has not
    # gone away, it has been accepted deliberately and made visible: the
    # payload records the season actually used, and every caption derives from
    # that field, so the page always names the basis it is showing. Changing
    # this line changes the label with it.
    value_season = utils.SEASONS[0]
    # BLENDED, not projected: current-season numbers where a player has any,
    # his last complete season where he has none, and score_season recording
    # which. Straight build_ranked_projected on the current season left 54
    # players with no value at all because they have not played yet, among them
    # Jimmy Butler, Brandon Ingram, Porzingis and Claxton. A trade card showing
    # nothing for Jimmy Butler is worse than one showing last season, and this
    # is the same mechanism the hub uses, so the two surfaces agree.
    df = utils.build_ranked_blended(value_season)
    _season_by = utils.NameIndex({str(r["Player"]): str(r.get("score_season") or value_season)
                                  for _, r in df.iterrows()})
    # The CARD's Barrett score, which until now came from the roster JSON and
    # so was whatever season that file was built against, regardless of what
    # the values beside it said.
    _score_by = utils.NameIndex({str(r["Player"]): float(r["barrett_score"])
                                 for _, r in df.iterrows()
                                 if r.get("barrett_score") == r.get("barrett_score")})

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
    # data/master_roster.csv, used for the TWO things it alone knows and
    # nothing else. Deliberately NOT as the roster spine: it carries no Barrett
    # score, no projected signings and no `real` flag, and it is hand-edited,
    # so making it the spine would bind a nightly pipeline to an editing
    # cadence. Reconciling against it keeps both properties.
    #
    #   kind == free_agent  ->  not on anybody's roster. The `team` column on
    #       those rows is the team that WAIVED him, which is why joining on
    #       team alone leaves a waived player sitting on a trade card.
    #   pos                 ->  filled only where the roster feed has none.
    #
    # Its TEAM column is NOT used. It had Walter Clayton Jr. and Micah Peavy on
    # MEM on 2026-10-29 while both had already played for DAL and OKC, so the
    # live stat feed is the fresher source for where a player actually is.
    # Absence from master_roster means KEEP: only an explicit free_agent row
    # removes anybody.
    import csv as _csv
    mr_kind, mr_pos = utils.NameIndex(), utils.NameIndex()
    mr_path = ROOT / "data" / "master_roster.csv"
    if mr_path.exists():
        for _r in _csv.reader(ln for ln in mr_path.read_text(encoding="utf-8").splitlines()
                              if not ln.lstrip().startswith("#")):
            # headerless: _, team, player, salary, kind, pos, note
            if len(_r) < 6 or not _r[2].strip() or _r[2].strip() == "player":
                continue
            mr_kind.add(_r[2].strip(), _r[4].strip())
            if _r[5].strip() and _r[5].strip().lower() != "nan":
                mr_pos.add(_r[2].strip(), _r[5].strip())

    overrides = utils.NameIndex()
    ov_path = ROOT / "data" / "position_overrides.csv"
    if ov_path.exists():
        for _r in _csv.DictReader(
                ln for ln in ov_path.read_text(encoding="utf-8").splitlines()
                if not ln.lstrip().startswith("#")):
            if (_r.get("player") or "").strip() and (_r.get("pos") or "").strip():
                overrides.add(_r["player"].strip(), _r["pos"].strip())
    applied = filled = prior_rows = 0
    waived: list = []
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
            if mr_kind.get(p["n"]) == "free_agent":
                waived.append((abbr, p["n"]))
                continue
            _ov = overrides.get(p["n"])
            _pos = p.get("pos", "")
            if _ov and _ov != _pos:
                _pos = _ov
                applied += 1
            elif not _pos.strip() or _pos.strip() == "\u2014":
                _mp = mr_pos.get(p["n"])
                if _mp:
                    _pos = _mp
                    filled += 1
            _bs = _score_by.get(p["n"])
            _byr = _season_by.get(p["n"]) or value_season
            if _bs is not None and _byr != value_season:
                prior_rows += 1
            players.append({
                "n": p["n"], "role": p["role"], "pos": _pos,
                # Score and its season from the blend, falling back to the
                # roster JSON only when the blend has never seen the player.
                # bs_yr is set ONLY when the row is not the current season, so
                # the card tags exactly the numbers that need explaining.
                "barrett": (round(_bs, 1) if _bs is not None else p.get("barrett")),
                # When the blend has a score, its season decides the tag
                # ABSOLUTELY: current means no tag. Falling through to the
                # roster JSON's bs_yr in that case kept a stale "24-25" beside
                # a current-season number for Haliburton and Kyrie, which is a
                # mislabel rather than a missing label. The roster value is
                # only a fallback for a player the blend has never scored.
                "bs_yr": ((_byr if _byr != value_season else None)
                          if _bs is not None else p.get("bs_yr")),
                "salary": sal,
                "value": round(val, 1) if val is not None else None,
                "vd": vd,  # salary - market value; >0 overpay, <0 bargain
                "real": p.get("real", True),  # False = projected move for an unsigned FA
            })
        players.sort(key=lambda x: -(x["salary"] or 0))  # stars first
        teams[abbr] = {"abbr": abbr, "name": t["name"], "size": t["size"],
                       "payroll": t["total"], "room": t["room"], "players": players,
                       "unsigned": t.get("unsigned", [])}

    # A hand-edited file that goes wrong should cost a few rows, never a roster.
    # Anything past this is a corrupted or wildly stale master_roster, not a
    # real transaction wave, and refusing beats writing it.
    _cap = 40
    if len(waived) > _cap:
        raise SystemExit(
            f"REFUSING: master_roster would drop {len(waived)} players as free "
            f"agents, over the {_cap} cap. That is a bad file, not a roster "
            f"change. Nothing written.")

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
    print(f"  positions filled from master_roster: {filled}")
    print(f"  players still on their {utils.SEASONS[1]} line (tagged bs_yr): {prior_rows}")
    print(f"  dropped as free agents per master_roster: {len(waived)}")
    for _ab, _n in waived:
        print(f"     {_ab}  {_n}")
    for probe in ("LAL", "DEN", "OKC"):
        t = teams.get(probe)
        if t:
            top = t["players"][0]
            print(f"  {probe} {t['name']}: {t['size']} players, ${t['payroll']}M payroll; "
                  f"top ${top['salary']}M {top['n']} ({top['role']})")


if __name__ == "__main__":
    main()
