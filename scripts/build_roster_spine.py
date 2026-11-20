#!/usr/bin/env python3
"""Rebuild cache/team_rosters_2627.json, the roster spine, from master_roster.

WHY THIS EXISTS
---------------
The spine is what scripts/build_team_pages.py builds the Trade Machine and the
crawlable team pages from. Until 2026-11-19 nothing rebuilt it on a schedule.
It had a writer -- scripts/export_fa_spreadsheet.py -- but that script is an
OFFSEASON tool: it fills roster slots from the free-agency SIMULATION
(cache/fa_sim_v1.json) and writes a workbook to ~/Downloads. Running it nightly
in November would re-project free agents onto teams, which is not a roster.

So the file sat at 2026-10-30 while the league moved, and nightly.py said so in
its own comment: "It does not regenerate the ROSTER ... which follows
transactions, not nights." Nothing followed the transactions.

WHAT IT COST, measured on 2026-11-19 against the live artifact:

  * TWENTY signed players invisible on the Trade Machine. Every "Draft pick"
    row in the spine is a SIGNED standard contract in master_roster, on the
    right team, at $1.4M to $8.8M -- but pages/Trade_Machine.py hid them,
    because in July "Draft pick" meant "drafted, not yet signed". Their money
    was still inside the team's payroll, so a user saw a payroll they could not
    account for and a roster they could not fill.
  * Mason Plumlee (SAC) and Orlando Robinson (CHI) on no card at all.
  * Nique Clifford shown on SACRAMENTO while master_roster has him on OKC.

THE SPLIT THIS FILE MAKES
-------------------------
Two different things were tangled in one artifact:

  ROSTER FACTS -- who is on which team, on what kind of deal, for how much.
      data/master_roster.csv is the verified source and is swept by hand; it
      was re-verified 2026-11-19. These are REBUILT here, every night.

  OFFSEASON NARRATIVE -- `role` (Re-sign / New signing / Draft pick), `prev`
      (the team he came from) and `unsigned`. These describe the 2026 offseason
      and are history, not state. They are CARRIED FORWARD from the previous
      spine, and only derived afresh for somebody it had never seen.

Everything else -- total, size, room -- is recomputed from the facts.

NOT A SOURCE OF TRUTH FOR SCORES. `barrett` and `bs_yr` are carried forward
only because build_team_pages uses them as a last-ditch fallback for a player
the blend has never scored; it overwrites them from the blend otherwise.

Run:  /opt/anaconda3/bin/python3 scripts/build_roster_spine.py
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import utils  # noqa: E402

ROS = ROOT / "cache" / "team_rosters_2627.json"
MASTER = ROOT / "data" / "master_roster.csv"
SIGNINGS = ROOT / "data" / "real_signings_2026.csv"
CBA = ROOT / "data" / "cba_config_2026_27.json"
DEAD = ROOT / "data" / "dead_money_2026_27.csv"
HARDCAPS = ROOT / "data" / "hard_caps_2026_27.csv"

ROSTERED = ("standard", "two_way")

# The same invariants scripts/check_roster_invariants.py enforces on
# master_roster, repeated here because that script is NOT part of the nightly:
# this builder is the last thing standing between a bad hand edit and a public
# roster. Refusing costs one stale night; writing costs a wrong league.
# 15 is the hard maximum. 14 is the ordinary minimum, but a team MAY sit at 13
# for up to two weeks, which is why scripts/check_roster_invariants.py WARNS at
# 13 and only FAILS below it. This builder refused anything under 14 in its
# first version, which would have frozen the WHOLE LEAGUE's spine on a roster
# that is legal and correctly swept -- and 5a is non-fatal, so the only trace
# would have been one warning line in the nightly log.
MAX_STANDARD, WARN_STANDARD, HARD_MIN_STANDARD = 15, 14, 13
MAX_TWO_WAY = 3
MIN_TEAMS = 30
MIN_ROWS, MAX_ROWS = 480, 560


def _hard(name: str) -> str:
    """Letters and digits only. Identity key, never a display or data join.

    utils.NameIndex cannot bridge a HYPHEN against a SPACE -- its key strips
    punctuation but keeps spaces -- and that is exactly how the old spine
    spells "David Jones-Garcia" against master_roster's "David Jones Garcia".
    On 2026-11-19 that put one man on the Spurs card twice.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _rows(path: Path):
    return csv.DictReader(
        l for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.lstrip().startswith("#"))


def main() -> int:
    if not ROS.exists():
        raise SystemExit(
            f"REFUSING: {ROS.name} does not exist. This script REBUILDS the "
            f"spine; it does not create one from nothing, because team names, "
            f"`role`, `prev` and `unsigned` are carried forward from it.")
    old = json.loads(ROS.read_text(encoding="utf-8"))
    old_teams = old.get("teams", {})

    # Previous spine, indexed for carry-forward. Keyed per TEAM as well as
    # globally: a player's `role` describes his arrival at a particular team,
    # so it must not follow him through a trade.
    prev_on_team: dict[tuple[str, str], dict] = {}
    for _abbr, _t in old_teams.items():
        for _p in _t.get("players", []):
            prev_on_team[(_abbr, _hard(_p["n"]))] = _p

    # Offseason signings, for a player the old spine never had.
    signed: dict[str, str] = {}
    if SIGNINGS.exists():
        for r in _rows(SIGNINGS):
            if r.get("player"):
                signed[_hard(r["player"])] = (r.get("type") or "").strip().lower()

    # Men who are out of the league entirely cannot be "still unsigned"
    # either: that phrase means signable today.
    try:
        gone = utils.NameIndex(utils.out_of_league_players())
    except Exception:                                    # never block a build
        gone = utils.NameIndex()

    cfg = json.loads(CBA.read_text(encoding="utf-8")) if CBA.exists() else {}
    cap_m = round(float(cfg.get("cap") or old.get("cap_M") or 0), 1)
    apron2_m = round(float(cfg.get("apron2") or old.get("apron2_M") or 0), 1)
    # EXACT ceilings for the arithmetic. apron2_m is rounded to $0.1M because
    # it is PUBLISHED as apron2_M and shown to readers, but rounding a ceiling
    # before subtracting a payroll from it is the same precision leak as
    # summing rounded rows: $209.015M became $209.0M, which moved Atlanta's
    # published room by $0.06M and is the whole margin on a team like
    # Cleveland.
    apron1_x = float(cfg.get("apron1") or 0)
    apron2_x = float(cfg.get("apron2") or old.get("apron2_M") or 0)

    # ── Each team's BINDING ceiling ──────────────────────────────────────────
    # `room` used to be apron-2 room for every team, whatever hard cap the team
    # actually carries. Twenty-one of the twenty-five capped teams are capped
    # at the FIRST apron, so the figure measured a line they cannot reach:
    # Cleveland's public team page read "$12.8M under the second apron" on a
    # night a hard cap it may not exceed sat $0.02M away. The number was true
    # and the label matched it; the LINE was the wrong one to report.
    # So publish room to the ceiling that actually binds, and publish the name
    # of that ceiling beside it, so the figure and its label cannot drift
    # apart the way they just did.
    binding = {}
    if HARDCAPS.exists() and apron1_x and apron2_x:
        for r in _rows(HARDCAPS):
            lvl = (r.get("cap") or "").strip()
            abbr_h = (r.get("team") or "").strip()
            if not abbr_h:
                continue
            if lvl == "apron1":
                binding[abbr_h] = (apron1_x, "first-apron hard cap")
            elif lvl == "apron2":
                binding[abbr_h] = (apron2_x, "second-apron hard cap")

    # ── Dead money ───────────────────────────────────────────────────────────
    # Salary a team still owes players who are NOT on its roster: buyouts,
    # waivers, stretches, partial guarantees left behind in a trade. It counts
    # against the cap, the tax line and BOTH APRONS exactly like a player's
    # salary, so it belongs in `room` and in the baseline the Trade Machine
    # prices a trade against. $145.4M of it across 18 teams on 2026-11-20, and
    # Memphis alone carries $40.6M -- a third of its real cap sheet.
    #
    # Published as its OWN field rather than folded into `total`, and that is
    # deliberate: `total` is the sum of the player rows, and a header a visitor
    # cannot add up from the cards below it is a header he will not trust.
    # Readers that need COMMITTED salary add the two; `room` already has.
    #
    # Non-fatal if the file is missing. A spine with no dead money understates
    # every apron figure, which is the status quo this fixes, but refusing to
    # build the league's rosters over one cap input would be worse.
    dead_raw: dict[str, float] = {}
    if DEAD.exists():
        for r in _rows(DEAD):
            abbr_d, amt = (r.get("team") or "").strip(), _num(r.get("amount_M"))
            if abbr_d and amt is not None:
                dead_raw[abbr_d] = dead_raw.get(abbr_d, 0.0) + amt
    # Rounded for DISPLAY only. `committed` below sums `dead_raw`, because
    # rounding dead money to $0.1M before adding it to salary is the same
    # precision leak as summing rounded player rows: Cleveland's $2.2440M
    # became $2.2000M, which is $0.044M on a team with $0.067M of room.
    dead_by_team = {k: round(v, 1) for k, v in dead_raw.items()}
    print(f"  dead money: ${sum(dead_by_team.values()):.1f}M across "
          f"{len(dead_by_team)} team(s)"
          + ("" if DEAD.exists() else "  (NO FILE -- apron figures understated)"))
    tw_season = utils.SEASONS[0]
    if tw_season not in utils.TWO_WAY_SALARY:
        raise SystemExit(
            f"REFUSING: utils.TWO_WAY_SALARY has no entry for {tw_season}, so "
            f"every two-way would be priced at $0.0M. Nothing written.")
    tw_salary = round(utils.TWO_WAY_SALARY[tw_season] / 1e6, 1)

    # ── the facts ────────────────────────────────────────────────────────────
    by_team: dict[str, list] = {}
    seen: dict[str, tuple[str, str]] = {}
    dupes = []
    for r in _rows(MASTER):
        name = (r.get("player") or "").strip()
        kind = (r.get("kind") or "").strip()
        team = (r.get("team") or "").strip()
        if not name or name == "player" or kind not in ROSTERED:
            continue
        if not team:
            raise SystemExit(
                f"REFUSING: master_roster has {name} on a {kind} deal with no "
                f"team. Nothing written.")
        key = _hard(name)
        if key in seen:
            dupes.append((name, seen[key], (team, kind)))
        seen[key] = (team, kind)
        by_team.setdefault(team, []).append((name, kind, r))
    if dupes:
        raise SystemExit(
            f"REFUSING: {len(dupes)} player(s) on two rosters in "
            f"master_roster: {dupes[:6]}. Nothing written.")

    # ── guards, before anything is written ───────────────────────────────────
    if len(by_team) != MIN_TEAMS:
        raise SystemExit(
            f"REFUSING: master_roster yields {len(by_team)} teams, not "
            f"{MIN_TEAMS}: {sorted(by_team)}. Nothing written.")
    bad, at_floor = {}, {}
    for abbr, members in by_team.items():
        n_std = sum(1 for _, k, _ in members if k == "standard")
        n_tw = sum(1 for _, k, _ in members if k == "two_way")
        if n_std > MAX_STANDARD or n_std < HARD_MIN_STANDARD or n_tw > MAX_TWO_WAY:
            bad[abbr] = f"{n_std} standard + {n_tw} two-way"
        elif n_std < WARN_STANDARD:
            at_floor[abbr] = n_std
    if bad:
        raise SystemExit(
            f"REFUSING: {len(bad)} team(s) over {MAX_STANDARD} standard, below "
            f"the temporary floor of {HARD_MIN_STANDARD}, or over "
            f"{MAX_TWO_WAY} two-way: {bad}. Fix data/master_roster.csv. "
            f"Nothing written.")
    total_rows = sum(len(v) for v in by_team.values())
    if not (MIN_ROWS <= total_rows <= MAX_ROWS):
        raise SystemExit(
            f"REFUSING: {total_rows} rostered players, outside the sane band "
            f"{MIN_ROWS}-{MAX_ROWS}. That is a broken file, not a transaction "
            f"wave. Nothing written.")
    missing_names = [a for a in by_team if not (old_teams.get(a) or {}).get("name")]
    if missing_names:
        raise SystemExit(
            f"REFUSING: no team name carried forward for {missing_names}. "
            f"Nothing written.")

    # ── assemble ─────────────────────────────────────────────────────────────
    dropped_unsigned = []
    for _a, _t in old_teams.items():
        for _u in _t.get("unsigned", []):
            if _hard(_u.get("n", "")) in seen or gone.get(_u.get("n", "")) is not None:
                dropped_unsigned.append((_a, _u.get("n")))

    kind_of = {_hard(n): k for members in by_team.values() for n, k, _ in members}
    if at_floor:
        print(f"  WARNING at the temporary {HARD_MIN_STANDARD}-man floor, legal "
              f"for two weeks: {at_floor}")
    teams, carried, derived, moved = {}, 0, 0, []
    for abbr in sorted(by_team):
        players = []
        # Salaries are published rounded to $0.1M, which is the only precision
        # a card ever shows. SUMMING the rounded numbers is a different thing,
        # and it is wrong: fourteen Cleveland contracts each moved up to $50K
        # by rounding add up to $206.9M against a true $206.748M, and that
        # $0.15M of accumulated error is larger than the $23K by which
        # Cleveland sits UNDER its first-apron hard cap. Rounding the ROWS is
        # display; rounding the TOTAL is arithmetic. Keep the exact figures for
        # the totals and round once, at the end.
        exact: list[tuple[str, float]] = []
        for name, kind, r in by_team[abbr]:
            key = _hard(name)
            was = prev_on_team.get((abbr, key))

            sal = _num(r.get("salary_M"))
            if not sal:
                # master_roster leaves two-ways at 0; the scale is the number.
                # Anyone else falls back to what the spine last knew, because a
                # missing salary DROPS a player from the board downstream.
                sal = tw_salary if kind == "two_way" else (
                    (was or {}).get("salary"))

            if was is not None:
                role = was.get("role") or "Under contract"
                carried += 1
            else:
                # New to this team. His old role described a different roster,
                # so it is not carried through a move.
                t = signed.get(key)
                role = ("Re-sign" if t == "resign"
                        else "New signing" if t else "Under contract")
                derived += 1
                if any(k == key for (_a, k) in prev_on_team):
                    moved.append(name)

            pos = (r.get("pos") or "").strip()
            if not pos or pos.lower() == "nan":
                pos = (was or {}).get("pos") or ""

            players.append({
                "n": name,
                "role": role,
                "pos": pos,
                "prev": (was or {}).get("prev") or abbr,
                "barrett": (was or {}).get("barrett"),
                "bs_yr": (was or {}).get("bs_yr"),
                "salary": round(float(sal), 1) if sal is not None else None,
                # Every row here is an ACTUAL contract in master_roster. `real`
                # was False only for a projected free-agent landing, which is
                # a thing that can no longer be in this file.
                "real": True,
            })
            exact.append((kind, float(sal) if sal is not None else 0.0))
        players.sort(key=lambda x: -(x["salary"] or 0))
        total = round(sum(s for _k, s in exact), 1)
        # `room` is room under the SECOND APRON, not under the cap. The only
        # reader renders it as a sentence -- seo_selfheal: "$X under the second
        # apron" / "over the second apron" -- so the basis is not a free
        # choice. Writing cap room here put 26 of 30 crawlable team pages into
        # a false "over the second apron", each contradicting the apron2_M in
        # its own payload: Atlanta would have read "$30.6M over" while sitting
        # $26M under.
        #
        # And on STANDARD money, because two-way salary does not count toward
        # the apron, and because build_team_pages already computes `payroll`
        # that way. Two fields on two bases is how a page comes to disagree
        # with itself.
        #
        # PLUS DEAD MONEY, which is not on any roster and so appears in no
        # player row, but counts against the apron in full. Leaving it out let
        # the Trade Machine show Memphis $101.2M of apron room when it has
        # $60.6M, and quoted its payroll at $120.5M on a night /Rosters quoted
        # $161.1M for the same team. A number nobody is playing for is still a
        # number the cap charges you for.
        std_exact = sum(s for k, s in exact if k != "two_way")
        std_total = round(std_exact, 1)
        dead_m = dead_by_team.get(abbr, 0.0)
        teams[abbr] = {
            "abbr": abbr,
            "name": old_teams[abbr]["name"],
            "total": total,
            # Standard-contract money, summed exact and rounded once. Published
            # so build_team_pages reads it instead of subtracting two-way money
            # out of `total`: a subtraction between two already-rounded numbers
            # loses precision a second time, on the figure every apron and hard
            # cap is tested against.
            "std_total": std_total,
            # THE FIGURE THE ENGINE SHOULD TEST, exact and summed ONCE.
            # `std_total` and `dead` are each rounded to $0.1M for display,
            # and the Trade Machine used to ADD THOSE TWO ROUNDED NUMBERS.
            # That is a third rounding on the one quantity every apron and
            # hard-cap test reads, and it is not academic: Cleveland published
            # $208.900M against an exact $208.992M, so the engine saw $0.115M
            # of room where $0.023M exists and returned legal=True on trades
            # taking back +$0.03M, +$0.05M and +$0.08M that break a hard cap
            # its own message calls one that cannot be exceeded.
            # Published ADDITIVELY: `payroll`, `dead` and `room` keep their
            # shape because four test modules and export_fa_spreadsheet read
            # them, and the payroll/dead split is what keeps a team card's
            # header addable from the player rows under it.
            "committed": round(std_exact + dead_raw.get(abbr, 0.0), 6),
            "size": len(players),
            "dead": dead_m,
            # room to the ceiling that BINDS this team, named by room_line.
            # An uncapped team is measured to the second apron, which it may
            # legally exceed but not without the harshest restrictions.
            # From the EXACT figures, rounded ONCE. Subtracting the two
            # already-rounded display numbers rounds a third time on the one
            # quantity this field exists to state: Cleveland published $0.1M
            # of room against an exact $0.023M.
            "room": round(binding.get(abbr, (apron2_x, ""))[0]
                          - std_exact - dead_raw.get(abbr, 0.0), 1),
            "room_line": binding.get(abbr, (apron2_x, "second apron"))[1],
            "players": players,
            # `unsigned` is carried forward but FILTERED, and the difference
            # from `role` is the whole point. A role is a PAST-TENSE fact --
            # "he was a new signing" stays true forever. `unsigned` is a
            # PRESENT-TENSE claim, and seo_selfheal publishes it as one:
            # "Still unsigned: X. No reported deal yet, and no open roster
            # spot here." Carried forward blindly it had Atlanta's public team
            # page saying that about CJ McCollum while the TABLE ON THE SAME
            # PAGE listed him rostered at $21.0M, and Memphis saying it about
            # Javon Small, by then a Memphis two-way.
            #
            # Carry a past-tense fact; re-check a present-tense claim.
            "unsigned": [u for u in (old_teams.get(abbr) or {}).get("unsigned", [])
                         if _hard(u.get("n", "")) not in seen
                         and gone.get(u.get("n", "")) is None],
        }

    payload = {
        # Was hardcoded "2025-26" and went stale the moment the season rolled;
        # nothing read it, which is the only reason it did no damage.
        "value_season": utils.SEASONS[0],
        "contract_season": utils.SEASONS[0],
        "cap_M": cap_m,
        "apron2_M": apron2_m,
        # The stamp this file never had. File mtime lies (git checkout, iCloud
        # re-materialisation, Render's disk seed), so the only trustworthy age
        # is one the builder wrote. build_team_pages refuses on a stale one.
        "built_at": _dt.date.today().isoformat(),
        "teams": dict(sorted(teams.items())),
    }
    ROS.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    n_std = sum(1 for t in teams.values() for p in t["players"]
                if (p["salary"] or 0) > tw_salary + 0.05)
    print(f"wrote {ROS.name}  ({len(teams)} teams, {total_rows} rostered, "
          f"{ROS.stat().st_size // 1024}KB)")
    print(f"  roles: {carried} carried forward, {derived} derived")
    if moved:
        print(f"  changed team since the last spine ({len(moved)}): "
              f"{', '.join(sorted(moved)[:10])}")
    if dropped_unsigned:
        print(f"  dropped from `unsigned`, no longer true ({len(dropped_unsigned)}): "
              + ", ".join(f"{a} {n}" for a, n in dropped_unsigned))
    print(f"  payroll: ${sum(t['total'] for t in teams.values()):,.1f}M total, "
          f"~{n_std} above the two-way scale")
    for probe in ("LAL", "SAS", "OKC"):
        t = teams[probe]
        print(f"  {probe}: {t['size']} rostered, ${t['total']}M, "
              f"top ${t['players'][0]['salary']}M {t['players'][0]['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
