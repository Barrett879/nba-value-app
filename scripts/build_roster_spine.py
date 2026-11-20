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
        players.sort(key=lambda x: -(x["salary"] or 0))
        total = round(sum(p["salary"] or 0 for p in players), 1)
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
        std_total = round(
            sum(p["salary"] or 0 for p in players
                if kind_of.get(_hard(p["n"])) != "two_way"), 1)
        teams[abbr] = {
            "abbr": abbr,
            "name": old_teams[abbr]["name"],
            "total": total,
            "size": len(players),
            "room": round(apron2_m - std_total, 1),
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
