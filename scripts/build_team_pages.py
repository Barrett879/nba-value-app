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
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import utils  # noqa: E402

ROS = ROOT / "cache" / "team_rosters_2627.json"


SPINE_WARN_DAYS = 3
SPINE_MAX_DAYS = 14


def assert_spine_fresh(ros: dict, today=None) -> int:
    """Raise unless the roster spine was rebuilt recently. Returns its age.

    A module-level function, not three lines inside main(), so it can be
    TESTED. The first version was inline and the only guard on it was a test
    grepping the source for "built_at" and "REFUSING" -- which a mutation that
    disabled the gate entirely survived, because both strings stayed put.
    """
    built = (ros.get("built_at") or "").strip()
    if not built:
        raise SystemExit(
            "REFUSING: cache/team_rosters_2627.json carries no built_at, so "
            "its age cannot be known. File mtime lies (git checkout, iCloud "
            "re-materialisation, Render's disk seed), which is why the age "
            "has to be one the builder wrote. Run "
            "scripts/build_roster_spine.py. Nothing written.")
    today = today or _dt.date.today()
    age = (today - _dt.date.fromisoformat(built)).days
    if age > SPINE_MAX_DAYS:
        raise SystemExit(
            f"REFUSING: the roster spine was last built {built} ({age} days "
            f"ago), over the {SPINE_MAX_DAYS}-day limit. That is a roster "
            f"nobody is maintaining, and every team page and trade card is "
            f"built from it. Run scripts/build_roster_spine.py. "
            f"Nothing written.")
    if age > SPINE_WARN_DAYS:
        print(f"  WARNING roster spine is {age} days old (built {built})")
    return age


def _hard(name: str) -> str:
    """A last-ditch identity key: letters and digits only, no spaces.

    utils.NameIndex is the right tool for name joins and is used first
    everywhere below. It cannot bridge a HYPHEN against a SPACE, because its
    key strips punctuation but keeps spaces: "David Jones-Garcia" and "David
    Jones Garcia" share no key in either direction, and the surname fallback
    compares "jonesgarcia" against "jones garcia".

    That is exactly how the roster spine spells him against master_roster and
    the live NBA feed, and on 2026-11-19 it put the same man on the Spurs card
    TWICE -- once as a standard "Re-sign" with no score, once as a two-way --
    which also inflated San Antonio's standard count by one.

    This key is deliberately NOT used for display or for joining data; only to
    answer "have I already emitted this person", where a false match costs one
    skipped row and a false miss publishes a duplicate.
    """
    import re
    return re.sub(r"[^a-z0-9]", "", str(name).lower())
OUT = ROOT / "cache" / "team_pages.json"


def main() -> None:
    ros = json.loads(ROS.read_text(encoding="utf-8"))
    # THE SPINE MUST BE FRESH. It had no scheduled builder until 2026-11-19
    # and sat at 2026-10-30 while the league moved: twenty signed players
    # invisible on the Trade Machine, two on no card at all, and Nique
    # Clifford on Sacramento after he was traded to Oklahoma City. Nothing
    # failed and nothing warned, because a stale roster looks exactly like a
    # quiet one.
    #
    # The remedy (scripts/build_roster_spine.py) runs BEFORE this check, as
    # nightly step 5a, so this cannot become a gate that arms itself: a check
    # whose only fix lives downstream of it rolls back forever once it fires.
    assert_spine_fresh(ros)
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
    # (team, player) for every two-way deal. master_roster is the ONLY file
    # that records them: the ESPN salary feed omits two-way contracts
    # altogether, which is why the roster JSON carries whichever handful
    # happened to be written into it as projected signings and no others.
    mr_two_way: list = []
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
            if _r[4].strip() == "two_way":
                mr_two_way.append((_r[1].strip(), _r[2].strip()))

    overrides = utils.NameIndex()
    ov_path = ROOT / "data" / "position_overrides.csv"
    if ov_path.exists():
        for _r in _csv.DictReader(
                ln for ln in ov_path.read_text(encoding="utf-8").splitlines()
                if not ln.lstrip().startswith("#")):
            if (_r.get("player") or "").strip() and (_r.get("pos") or "").strip():
                overrides.add(_r["player"].strip(), _r["pos"].strip())
    _twi = utils.NameIndex({_n: True for _n in utils.two_way_players()})
    _tw_hard = {_hard(_n) for _n in utils.two_way_players()}
    _spine_tw: dict = {}          # two-way rows the spine carried, per team
    _spine_tw_sal: dict = {}      # and their money, which `total` includes
    _spine_pos = utils.NameIndex()
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
            if _twi.get(p["n"]) is not None or _hard(p["n"]) in _tw_hard:
                # Two-ways are emitted below, ALL of them, from master_roster.
                # Keeping the spine's copy here and adding only the missing
                # ones left the file with two authorities for the same fact
                # and the team could disagree: the spine still had Sean
                # Pedulla on Houston after master_roster moved him to
                # Sacramento, which is a FOURTH Houston two-way. One source
                # for the set and one for the team, or this recurs.
                #
                # His spine position is kept as a last fallback, since
                # master_roster leaves `pos` blank for most two-ways.
                _spine_tw[abbr] = _spine_tw.get(abbr, 0) + 1
                _spine_tw_sal[abbr] = _spine_tw_sal.get(abbr, 0.0) + float(sal or 0)
                if (p.get("pos") or "").strip() not in ("", "\u2014"):
                    _spine_pos.add(p["n"], p["pos"])
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
                # A two-way is NOT a standard contract: it is invisible to
                # salary matching and to the 15-man limit, and the CBA engine
                # already models all of that behind TradePlayer.two_way. It
                # just had no way to know, because nothing on the wire said
                # so. See pages/Trade_Machine.py.
                "two_way": False,     # every two-way is emitted below
            })
        players.sort(key=lambda x: -(x["salary"] or 0))  # stars first
        # `size` is the spine's own roster count. It is NOT the standard-
        # contract count: it includes the two-ways skipped above AND the
        # players dropped as free agents, so deriving std_size by subtracting
        # two-ways left LAC reading 16 standard contracts (Jamarion Sharp,
        # waived, was still in the 17).
        #
        # So COUNT, do not subtract. At this point `players` holds exactly the
        # standard contracts -- free agents dropped, two-ways not yet appended
        # -- which is the number the 15-man limit is about. Projected signings
        # are counted, deliberately: `payroll` already carries their money, so
        # excluding them from the roster would have the cap sheet and the
        # roster count disagree about the same hypothetical player.
        #
        # Measured 2026-11-19: no team exceeds 15 under this definition. The
        # old one had TWELVE teams over before any trade was proposed -- ten
        # from counting two-ways as standard contracts, and HOU and LAC from
        # counting players this builder had already dropped as free agents.
        # Payroll is STANDARD-contract money. A two-way salary does not count
        # against the cap, the apron or the tax, and the engine already
        # excludes two-ways from salary_out/salary_in -- so leaving their money
        # inside the baseline made the arithmetic inconsistent: sending a
        # two-way out credited nothing while his salary stayed in `payroll`.
        # Indiana sits $0.6M under a hard cap with $2.0M of two-way money
        # inside its total, which is enough to call a legal trade illegal.
        # The spine was also inconsistent team to team, carrying whichever
        # two-ways it happened to list (LAL 0.7, IND 2.0, BKN 0.0).
        # `payroll` is ROSTER money: what the player cards below it add up to.
        # `dead` is what the team owes players who are NOT on the roster and so
        # appear in no card. COMMITTED salary -- the number the cap, the tax
        # line and both aprons are measured against, and the only honest
        # baseline for a trade -- is the two added. They are published apart so
        # the card header stays addable from the cards; every cap comparison
        # adds them, and `room` already has.
        teams[abbr] = {"abbr": abbr, "name": t["name"], "size": t["size"],
                       "std_size": len(players),
                       # The spine sums the exact contracts and rounds once;
                       # prefer that over subtracting two-way money out of an
                       # already-rounded `total`, which rounds twice on the
                       # one figure every apron and hard-cap test reads.
                       "payroll": round(float(
                           t["std_total"] if t.get("std_total") is not None
                           else t["total"] - _spine_tw_sal.get(abbr, 0.0)), 1),
                       "dead": round(float(t.get("dead") or 0.0), 1),
                       # exact committed salary, summed once; see the spine
                       "committed": (round(float(t["committed"]), 6)
                                     if t.get("committed") is not None else None),
                       "room": t["room"], "players": players,
                       "unsigned": t.get("unsigned", [])}

    # ── Two-ways the roster JSON never heard of ──────────────────────────────
    # The spine carried 19 of the league's 89 two-way players on 2026-11-19,
    # and the 19 were arbitrary: whichever two-way deals happened to be
    # written in as projected signings last time the file was touched, which
    # was 2026-10-30. 29 of 30 teams were wrong. On the Trade Machine that
    # read as Arthur Kaluma being the only two-way the Lakers could trade,
    # with Chris Manon and Carson Cooper absent.
    #
    # master_roster is used here the same way it is used above: for the thing
    # it alone knows. Above, which players are free agents; here, who is on a
    # two-way. Adding rather than replacing, because the spine is still the
    # source for standard contracts, projected signings and the `real` flag.
    #
    # The TEAM comes from the blend only for a CURRENT-SEASON row, and
    # otherwise from master_roster.
    #
    # Both halves of that were learned the hard way. master_roster's team
    # column lags -- it had Walter Clayton Jr. and Micah Peavy on MEM while
    # both had already played for DAL and OKC -- so the live stat feed wins
    # where it has a live opinion. But a FALLBACK row's Team is last season's
    # team, and taking it put Drew Timme on the Lakers (where he played in
    # 2025-26, not where he is) and Kobe Bufkin there too, giving LAL five
    # two-ways. The three-slot guard below is what caught it.
    # NO SILENT ZERO. `.get(season, 0)` would price all 89 two-ways at $0.0M
    # the first season nobody adds a key, and the guard that exists for this
    # table (check_season_surfaces) checks SEASONS[0] while this reads the
    # SPINE's contract_season -- two keys that drift apart every July.
    _tw_season = ros["contract_season"]
    if _tw_season not in utils.TWO_WAY_SALARY:
        raise SystemExit(
            f"REFUSING: utils.TWO_WAY_SALARY has no entry for {_tw_season}, so "
            f"every two-way would be priced at $0.0M. Add the season's two-way "
            f"scale. Nothing written.")
    _tw_sal = round(utils.TWO_WAY_SALARY[_tw_season] / 1e6, 1)
    _mteam = utils.NameIndex({
        str(r["Player"]): str(r["Team"]) for _, r in df.iterrows()
        if str(r.get("Team") or "").strip()
        and str(r.get("score_season") or value_season) == value_season})
    _known = list(teams.keys())
    _have = utils.NameIndex({p["n"]: True for t in teams.values() for p in t["players"]})
    _have_hard = {_hard(p["n"]) for t in teams.values() for p in t["players"]}
    added: list = []
    for _mr_team, _name in mr_two_way:
        if _have.get(_name) is not None or _hard(_name) in _have_hard:
            # A duplicate master_roster row, or a name that collides with a
            # standard contract. Either is a data error; say so and skip,
            # rather than putting one man on two cards.
            print(f"  WARNING two-way {_name}: already on a roster card; skipped")
            continue
        _cand = _mteam.get(_name) or _mr_team
        _abbr = utils.resolve_team_code(_cand, _known)
        if _abbr not in teams:
            print(f"  WARNING two-way {_name}: no team ({_cand!r}); skipped")
            continue
        _bs2 = _score_by.get(_name)
        _byr2 = _season_by.get(_name) or value_season
        _val2 = mval.get(_name)
        teams[_abbr]["players"].append({
            "n": _name,
            "role": "Two-way",
            "pos": (overrides.get(_name) or mr_pos.get(_name)
                    or _spine_pos.get(_name) or ""),
            "barrett": round(_bs2, 1) if _bs2 is not None else None,
            "bs_yr": (_byr2 if _byr2 != value_season else None) if _bs2 is not None else None,
            "salary": _tw_sal,
            "value": round(_val2, 1) if _val2 is not None else None,
            "vd": round(_tw_sal - _val2, 1) if _val2 is not None else None,
            "real": True,
            "two_way": True,
        })
        _have.add(_name, True)
        _have_hard.add(_hard(_name))
        added.append((_abbr, _name))
    for _t in teams.values():
        _t["players"].sort(key=lambda x: -(x["salary"] or 0))
        # The engine needs this: a two-way is invisible to the 15-man count, so
        # nothing else bounds how many a team may hold. See
        # trade_rules.check_cash_and_roster.
        _t["two_way_size"] = sum(1 for _p in _t["players"] if _p["two_way"])

    # Three two-way slots per team (data/cba_config_2026_27.json two_way_slots).
    # A team over that is a master_roster error, not a roster: refuse rather
    # than publish an illegal roster to the Trade Machine, which is the one
    # surface that reasons about roster legality.
    try:
        _slots = int(json.loads(
            (ROOT / "data" / "cba_config_2026_27.json").read_text())["two_way_slots"])
    except Exception:
        _slots = 3
    _over = {a: n for a, n in
             ((a, sum(1 for p in t["players"] if p["two_way"])) for a, t in teams.items())
             if n > _slots}
    if _over:
        raise SystemExit(
            f"REFUSING: {_over} would carry more than {_slots} two-way contracts. "
            f"Fix data/master_roster.csv. Nothing written.")

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
        # When this script last re-joined scores and values onto the roster.
        # NOT when the roster changed: that is cache/team_rosters_2627.json,
        # which follows transactions rather than nights.
        #
        # Stamped because nothing could previously tell how old this file was.
        # It carries no marker of its own, the nightly did not rebuild it
        # until 2026-11-02, and "are these updated?" had to be answered by
        # reading git log. The page prints it now.
        "built_at": _dt.date.today().isoformat(),
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
    _n_tw = sum(1 for t in teams.values() for p in t["players"] if p["two_way"])
    print(f"  two-way contracts: {_n_tw} from master_roster "
          f"({sum(_spine_tw.values())} of them were also in the roster JSON)")
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
