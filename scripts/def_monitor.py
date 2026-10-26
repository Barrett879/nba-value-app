"""Re-run the blend-vs-box-line question as the season accumulates games.

    /opt/anaconda3/bin/python3 -u scripts/def_monitor.py [season] [--no-fetch]

The back-test said a points-per-possession ridge (option 13) adds +0.03 to +0.08
Spearman on top of the shipped box proxy, with a blend weight of 0.15-0.30. That
was measured on FINISHED seasons against FINISHED D-LEBRON. The live season has
neither, so this asks the same question with what exists today.

TWO benchmarks, with OPPOSITE biases, which is the point:

  forward holdout   Fit both metrics on each team's first k games, then score
                    them against what happened in the games AFTER k: opponent
                    points per possession with the player off the floor minus
                    with him on, so higher is a better defender. Independent
                    games, so the noise is independent -- but the target IS an
                    on/off quantity, which structurally flatters the ridge.

                    MEASURED, not assumed: at 19 team games this target has a
                    split-half reliability of +0.04, so NOTHING can correlate
                    with it and every metric scored -0.01 to -0.04 against it.
                    That is the published finding about defence arriving in our
                    own data (Ilardi put single-season defensive RAPM's standard
                    error near 5 points per 100), not a bug. Reliability is
                    therefore recomputed every run and reported beside the
                    scores, with holdout_interpretable false while it is under
                    0.15 -- which is the honest state early in a season. It
                    should cross that line as games accumulate, and the log will
                    show the crossing.

  current D-LEBRON  Checked every run and used the moment BBall Index publishes
                    2026-27. D-LEBRON shrinks RAPM toward a BOX prior, so it
                    flatters the proxy instead.

A blend that wins on both is real. A blend that only wins on the holdout is the
holdout's bias. The proxy's box line is cumulative-to-date, so the holdout games
leak into the proxy side -- that makes a proxy win uninformative and a blend win
conservative. Each run snapshots the box line with its game count, so once
enough snapshots exist a later pass can difference them and remove that leak.

Also logged: board churn against the previous run. When the top of the board
stops moving, the sample has arrived, regardless of what either benchmark says.

Appends one line to cache/def_monitor/log.jsonl. Writes nothing the site reads.
Designed to be run unattended: network failures are logged, never raised.
"""
from __future__ import annotations
import json, logging, re, sys, time, urllib.request, warnings
import datetime as dt
from pathlib import Path


def _quiet_import():
    """Import the heavy modules without 4 KB of Streamlit noise on stderr.

    utils carries @st.cache_data decorators, and importing it outside a
    Streamlit runtime logs one warning per decorator -- 31 of them. Under
    launchd that lands in StandardErrorPath, where it would bury a real error,
    so an empty stderr log should mean the run was healthy.

    setLevel on those loggers does NOT work and was tried: Streamlit configures
    its own loggers when it is imported and overwrites a level set beforehand,
    and STREAMLIT_LOGGER_LEVEL=error does not reach them either (both measured,
    31 lines each). Suppressing records globally for the duration of the import
    and restoring immediately after is what actually works.
    """
    logging.disable(logging.WARNING)
    try:
        import numpy, pandas                      # noqa: F401
        import def_lib, def_options               # noqa: F401
        import def_backtest                       # noqa: F401
        import utils                              # noqa: F401
    finally:
        logging.disable(logging.NOTSET)
    # One warning survives the import window because fetch_bref_player_stats is
    # @st.cache_data and warns when it is CALLED, not when it is defined. Now
    # that Streamlit has configured its own loggers, setLevel on them sticks --
    # which it does not before the import.
    for n in ("streamlit.runtime.caching.cache_data_api",
              "streamlit.runtime.caching.cache_resource_api"):
        logging.getLogger(n).setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

def _season_being_played(d=None):
    """The season currently ON COURT, derived from the date.

    Deliberately NOT utils.SEASONS[0], which is the season being SCORED and has
    already diverged from the one being played once this year (see
    docs/season_rollover.md). A hardcoded label here would also strand this
    script every October, and the whole point is that it runs unattended.
    """
    d = d or dt.date.today()
    y = d.year if d.month >= 10 else d.year - 1
    return f"{y}-{str((y + 1) % 100).zfill(2)}"


SEASON = next((a for a in sys.argv[1:] if not a.startswith("-")), _season_being_played())
FETCH = "--no-fetch" not in sys.argv
# per season, so a run on a finished season for testing can never be compared
# for churn against the live one
OUT = ROOT / "cache" / "def_monitor" / SEASON
PB = ROOT / "cache" / "pbpstats"
LOG = OUT / "log.jsonl"
UA = {"User-Agent": "Mozilla/5.0"}
PAUSE, LAM13 = 2.5, 200.0
WGRID = (0.15, 0.20, 0.25, 0.30)
MIN_ON_HOLD, MIN_OFF_HOLD = 60.0, 40.0


def http(url, tries=4):
    d = 4.0
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=45).read())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(d); d *= 1.8


def ingest(season, note):
    """Refresh the game list and pull any lineup file we do not have yet."""
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        gl = http(f"https://api.pbpstats.com/get-games/nba?Season={season}"
                  f"&SeasonType=Regular%20Season")["results"]
    except Exception as e:
        note.append(f"game list fetch failed: {type(e).__name__}")
        return 0
    # MERGE, never overwrite: games backfilled from ESPN are absent from this
    # API list by definition, and a blind write would silently delete them on
    # the next run. Anything with a lineup file on disk is kept.
    gfp = PB / f"games_{season}.json"
    keep = []
    if gfp.exists():
        try:
            known = {str(g.get("GameId")) for g in gl}
            for g in json.loads(gfp.read_text()):
                gid = str(g.get("GameId"))
                if gid not in known and (PB / f"lineupopp_{gid}.json").exists():
                    keep.append(g)
        except Exception as e:
            note.append(f"could not read existing game index: {type(e).__name__}")
    if keep:
        note.append(f"kept {len(keep)} locally backfilled game(s) absent from the feed")
    gfp.write_text(json.dumps(list(gl) + keep))
    gl = sorted(gl, key=lambda g: (str(g.get("Date")), str(g.get("GameId"))))
    got = 0
    for g in gl:
        fp = PB / f"lineupopp_{g['GameId']}.json"
        if fp.exists():
            continue
        try:
            d = http(f"https://api.pbpstats.com/get-game-stats?GameId={g['GameId']}"
                     f"&Type=LineupOpponent")
        except Exception as e:
            note.append(f"{g['GameId']} fetch failed: {type(e).__name__}")
            continue
        if not isinstance(d, dict) or "stats" not in d:
            note.append(f"{g['GameId']} returned no stats block")
            continue
        fp.write_text(json.dumps(d)); got += 1
        time.sleep(PAUSE)
    return got


def report():
    """Print the accumulated series. Runs nothing, fetches nothing."""
    if not LOG.exists():
        print(f"  no runs logged yet for {SEASON} ({LOG})")
        return 0
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    print(f"\n  {SEASON} defensive-blend monitor, {len(rows)} runs\n")
    print(f"  {'date':<12}{'gms':>4}{'pool':>6}{'rel':>7}{'ok?':>5}"
          f"{'D-LEB proxy':>12}{'blend':>8}{'gain':>7}{'churn':>7}{'top25':>6}")
    for r in rows:
        if "fatal" in r:
            print(f"  {r.get('date','?'):<12}  FATAL {r['fatal'][:60]}")
            continue
        dl = r.get("dlebron") or {}
        best = max((dl.get(f"w{w:.2f}") or -9 for w in WGRID), default=-9)
        gain = (best - dl["proxy"]) if dl.get("proxy") is not None and best > -9 else None
        f = lambda v, w=8, d=3: (f"{v:>{w}.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")
        print(f"  {r.get('date',''):<12}{r.get('team_games',0):>4}"
              f"{r.get('holdout_players') or 0:>6}{f(r.get('holdout_reliability'),7)}"
              f"{('yes' if r.get('holdout_interpretable') else 'no'):>5}"
              f"{f(dl.get('proxy'),12)}{f(best if best > -9 else None)}{f(gain,7)}"
              f"{f(r.get('stability'),7)}{(r.get('top25_held') if r.get('top25_held') is not None else '-'):>6}")
    print(f"\n  latest: {rows[-1].get('verdict', rows[-1].get('fatal','?'))}\n")
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    note, today = [], dt.date.today().isoformat()
    fetched = ingest(SEASON, note) if FETCH else 0

    _quiet_import()
    import numpy as np, pandas as pd
    import def_lib as L, def_options as O
    import utils
    from def_backtest import shipped_proxy, truth

    def zs(s):
        sd = s.std()
        return (s - s.mean()) / (sd if sd and sd == sd else 1.0)

    def rho(a, b):
        j = pd.DataFrame({"a": a}).join(pd.Series(b, name="b"), how="inner").dropna()
        return round(float(j.a.corr(j.b, method="spearman")), 4) if len(j) > 25 else None

    full = L.onoff(SEASON)
    n = max((v[3] for v in full.values()), default=0)
    n_min = min((v[3] for v in full.values()), default=0)
    rec = {"date": today, "season": SEASON, "team_games": int(n),
           "fewest_games_any_player": int(n_min),
           # game ids are 002 + the two-digit season start year + five digits
           "lineup_files": len(list(PB.glob(f"lineupopp_002{SEASON[2:4]}*.json"))),
           "fetched_this_run": fetched, "notes": note}

    lsp = ROOT / "cache" / f"league_stats_{SEASON.replace('-', '_')}.parquet"
    rec["box_source"] = "cached parquet" if lsp.exists() else "live BBRef fetch"

    # The two feeds run on different clocks. The box line is BBRef and can be a
    # day ahead of pbpstats' lineups, so some teams have fewer lineup games than
    # their players' box lines claim, and the two halves of the blend then see
    # different numbers of games for those players. team_games above is the MAX,
    # which hides it, so the spread is recorded explicitly.
    from collections import Counter
    per_team = Counter()
    for _g, ab, _r in L.lineup_rows(SEASON):
        per_team[ab] += 1
    rec["lineup_games_per_team"] = {str(k): v for k, v in
                                    sorted(Counter(per_team.values()).items())}

    # Completeness against ESPN, because pbpstats' OWN list omits games it has
    # not processed and comparing the feed to that list reported "healthy".
    # team_games above is the MAX, which hid it too: it read 2 while four teams
    # were short a played game and Walker Kessler, who had played twice, was
    # being scored off one.
    try:
        espn, feed, missing = L.feed_gap(SEASON)
        rec["games_played_espn"] = sum(espn.values()) // 2
        rec["games_in_feed"] = len(L.games(SEASON))
        rec["games_missing_from_feed"] = len(missing)
        rec["missing_games"] = [f"{m} {d}" for m, d in missing][:40]
        rec["teams_short"] = {t: [espn[t], feed.get(t, 0)]
                              for t in sorted(espn) if feed.get(t, 0) < espn[t]}
        if missing:
            note.append(f"lineup feed is missing {len(missing)} played games; "
                        f"{len(rec['teams_short'])} teams are short")
    except Exception as e:
        note.append(f"completeness check failed: {type(e).__name__}")

    # box-line snapshot, so a later pass can difference out the holdout leak
    try:
        box = L.players(SEASON)[["PLAYER_NAME", "key", "GP", "MIN", "BLK", "STL",
                                 "DREB", "PF"]]
        box.to_parquet(OUT / f"box_{today}_{int(n)}g.parquet")
        rec["box_snapshot"] = f"box_{today}_{int(n)}g.parquet"
    except Exception as e:
        note.append(f"box snapshot failed: {type(e).__name__}")

    # ---- is the SHIPPED defensive blend stale? -------------------------
    # The blend cannot be computed on Render, so the live site's defence is
    # only as fresh as the last push. Nothing used to notice that. This
    # recomputes what a rebuild WOULD write and measures it against the
    # artifact currently committed, so the staleness is a number in the log
    # rather than something somebody has to remember.
    #
    # It deliberately stops short of committing or pushing. A daily
    # unattended push would mean a 45-130 second outage EVERY DAY, which is
    # the deploy-spam the project's own notes warn against, and the deploy
    # is the one step that should stay deliberate.
    try:
        import importlib, glob
        bdb = importlib.import_module("build_defense_blend")
        shipped_path = sorted(glob.glob(str(ROOT / "cache" /
                              f"defense_blend_{SEASON.replace('-', '_')}_v*.json")))
        if shipped_path:
            shipped = json.loads(Path(shipped_path[-1]).read_text())
            sv = shipped.get("values") or {}
            _df, _scaled, _mu, _sd = bdb.compute(SEASON)
            live = {str(k): round(float(v), 4) for k, v in
                    zip(_df.player.astype(str), _scaled)}
            both = set(sv) & set(live)
            moved = {k: round(live[k] - sv[k], 3) for k in both
                     if abs(live[k] - sv[k]) > 0.05}
            worst = sorted(moved.items(), key=lambda kv: -abs(kv[1]))[:5]
            rec["blend_drift"] = {
                "shipped_file": Path(shipped_path[-1]).name,
                "shipped_generated": (shipped.get("_meta") or {}).get("generated"),
                "shipped_lineup_games": (shipped.get("_meta") or {}).get("lineup_games"),
                "lineup_games_now": len(L.games(SEASON)),
                "players_shipped": len(sv), "players_now": len(live),
                "players_moved_over_0.05": len(moved),
                "max_move": max((abs(v) for v in moved.values()), default=0.0),
                "biggest_moves": [f"{k} {v:+.2f}" for k, v in worst],
                "new_players": sorted(set(live) - set(sv))[:10],
            }
            stale = (len(moved) or set(live) ^ set(sv))
            rec["blend_is_stale"] = bool(stale)
            if stale:
                note.append(
                    f"SHIPPED BLEND IS STALE: {len(moved)} players moved by more than "
                    f"0.05, {len(set(live) - set(sv))} new. To refresh and deploy, bump "
                    f"VERSION in build_defense_blend.py and DATA_REV in utils.py, rerun "
                    f"build_defense_blend.py, rebuild the three revisioned caches, then "
                    f"check_cache_seeding.py and push.")
    except Exception as e:
        note.append(f"blend drift check failed: {type(e).__name__}: {str(e)[:90]}")

    # ---- is the SHIPPED per-game panel data stale? ---------------------
    # Second artifact with the same property as the blend: it cannot be built
    # on Render, so the Rankings row panel is only as fresh as the last push.
    # One unwatched artifact is an oversight; two is a pattern, so it gets the
    # same check rather than a note in somebody's head.
    try:
        import glob as _g
        # Sorted NUMERICALLY, not lexicographically: plain sorted() puts v10
        # before v4, so the moment the version reaches double digits [-1]
        # silently starts picking v9 and the staleness check reads the wrong
        # artifact while reporting success.
        _pg = sorted(_g.glob(str(ROOT / "cache" /
                     f"per_game_{SEASON.replace('-', '_')}_v*.json")),
                     key=lambda f: int(re.search(r"_v(\d+)\.json$", f).group(1)))
        if _pg:
            _shipped = json.loads(Path(_pg[-1]).read_text())
            _sv = _shipped.get("values") or {}
            _src = L.players(SEASON)
            _live = {str(n): float(g) for n, g in zip(_src["PLAYER_NAME"], _src["GP"])}
            _moved = [k for k in set(_sv) & set(_live)
                      if abs(float(_sv[k].get("GP", 0)) - _live[k]) > 0.5]
            rec["per_game"] = {
                "file": Path(_pg[-1]).name,
                "generated": (_shipped.get("_meta") or {}).get("generated"),
                "players_shipped": len(_sv),
                "players_with_more_games_now": len(_moved),
                "new_players": len(set(_live) - set(_sv)),
            }
            if _moved or (set(_live) - set(_sv)):
                rec["per_game_is_stale"] = True
                note.append(
                    f"SHIPPED PANEL DATA IS STALE: {len(_moved)} players have played "
                    f"more games than the artifact records and "
                    f"{len(set(_live) - set(_sv))} are new. Rerun "
                    f"scripts/build_per_game.py and push.")
            else:
                rec["per_game_is_stale"] = False
    except Exception as e:
        note.append(f"per-game drift check failed: {type(e).__name__}: {str(e)[:80]}")

    # ---- does any team's board show more games than it has played? ------
    # A consistency check cannot catch a double count: the frame stays
    # perfectly consistent with itself while every per-game average for that
    # team is divided by the wrong denominator. So this compares against a
    # source that counts games a DIFFERENT way -- the board comes from the
    # stat feed, the truth comes from the schedule. On 2026-10-25 the
    # provisional merge folded already-counted games back in for the three
    # franchises whose BBRef code differs from their NBA code, and Charlotte
    # showed 5 games on a night it had played 3.
    try:
        _bp = utils._rev_path(
            f"raw_{SEASON.replace('-', '_')}_{utils.FORMULA_VERSION}.parquet")
        if _bp.exists():
            _board = pd.read_parquet(_bp)
            _played, _, _ = L.feed_gap(SEASON)
            _over, _unknown = [], []
            for _t, _grp in _board.groupby("Team"):
                _gp = float(pd.to_numeric(_grp["GP"], errors="coerce").max())
                _truth = _played.get(_t)
                if _truth is None:
                    _unknown.append(_t)          # a code that leaked through untranslated
                elif _gp > _truth:
                    _over.append((_t, _gp, _truth))
            rec["board_gp"] = {"file": _bp.name,
                               "teams_over": [list(x) for x in _over],
                               "teams_not_in_schedule": _unknown}
            if _over or _unknown:
                rec["board_gp_is_wrong"] = True
                note.append(
                    f"BOARD GAMES PLAYED IS WRONG: {len(_over)} team(s) show more games "
                    f"than they have played{' and ' + str(len(_unknown)) + ' team code(s) are not in the schedule' if _unknown else ''}. "
                    f"Run scripts/check_board_gp.py; see utils.resolve_team_code.")
            else:
                rec["board_gp_is_wrong"] = False
    except Exception as e:
        note.append(f"board GP check failed: {type(e).__name__}: {str(e)[:80]}")

    # ---- is the board's defence actually the blend? --------------------
    # d_lebron has two possible sources, the on/off blend and the box proxy,
    # and they are indistinguishable by shape. build_raw silently takes the
    # proxy when the blend artifact is missing, which is correct for pre-2009
    # seasons and catastrophic for this one. On 2026-10-25 the board was built
    # before build_defense_blend wrote the new version and the live site served
    # 362 players on the proxy, 63 of them more than two points off.
    try:
        _bp = utils._rev_path(
            f"raw_{SEASON.replace('-', '_')}_{utils.FORMULA_VERSION}.parquet")
        _bl = utils.fetch_defense_blend(SEASON)
        if _bp.exists():
            if not _bl:
                rec["defense_source_is_wrong"] = True
                note.append("BOARD DEFENCE IS THE BOX PROXY: there is no blend artifact "
                            "for this season, so every player is on the fallback metric.")
            else:
                _bd = pd.read_parquet(_bp)
                _ni = utils.NameIndex(dict(_bl))
                _n, _off = 0, 0
                for _p, _v in zip(_bd["Player"], _bd["d_lebron"]):
                    _w = _ni.get(str(_p))
                    if _w is None:
                        continue
                    _n += 1
                    if abs(float(_v) - float(_w)) > 0.005:
                        _off += 1
                rec["defense_source"] = {"file": _bp.name, "shared_players": _n,
                                         "disagreeing": _off}
                if _n and _off:
                    rec["defense_source_is_wrong"] = True
                    note.append(
                        f"BOARD DEFENCE DOES NOT MATCH THE BLEND: {_off} of {_n} shared "
                        f"players disagree, so the board was built against a different "
                        f"artifact. Run scripts/check_defense_source.py; rebuild with "
                        f"scripts/refresh_live_season.py.")
                elif not _n:
                    rec["defense_source_is_wrong"] = True
                    note.append("BOARD DEFENCE: not one board player matched the blend by "
                                "name. That is a join failure, not a coverage gap.")
                else:
                    rec["defense_source_is_wrong"] = False
    except Exception as e:
        note.append(f"defence source check failed: {type(e).__name__}: {str(e)[:80]}")

    if n < 4:
        rec["verdict"] = f"waiting for sample: {n} team games, a forward holdout needs 4"
    else:
        try:
            k = n - max(1, n // 3)
            rec["fit_games"], rec["holdout_games"] = int(k), int(n - k)

            # ---- the target: opponent efficiency in games the fit never saw ----
            hold = L.onoff(SEASON, None, k)
            tgt = {}
            for p, (on, off, tm, g) in hold.items():
                if on.get("OffPoss", 0) < MIN_ON_HOLD or off.get("OffPoss", 0) < MIN_OFF_HOLD:
                    continue
                tgt[L.box_key(SEASON, p)] = (off["Points"] / off["OffPoss"]
                                             - on["Points"] / on["OffPoss"])
            tgt = pd.Series(tgt, dtype=float)
            rec["holdout_players"] = int(len(tgt))

            # how much signal does the target itself carry? split the season in two
            # and correlate the target with itself. This bounds every score below:
            # a metric cannot track a target that does not track itself.
            def _tgt(lo, hi):
                d = {}
                for p, (on, off, tm, g) in L.onoff(SEASON, hi, lo).items():
                    if on.get("OffPoss", 0) < MIN_ON_HOLD / 2 or \
                            off.get("OffPoss", 0) < MIN_OFF_HOLD / 2:
                        continue
                    d[L.box_key(SEASON, p)] = (off["Points"] / off["OffPoss"]
                                               - on["Points"] / on["OffPoss"])
                return pd.Series(d, dtype=float)
            half = n // 2
            rel = rho(_tgt(0, half), _tgt(half, n)) if half >= 2 else None
            rec["holdout_reliability"] = rel
            rec["holdout_reliability_window"] = int(half)
            rec["holdout_interpretable"] = bool(rel is not None and rel >= 0.15)

            df = O.build(SEASON, prior=None, max_games=k)
            px = zs(pd.Series(np.asarray(shipped_proxy(df), float), index=df.key))
            o13 = zs(pd.Series(np.asarray(O.opt13_possession_outcome(
                df, SEASON, lam=LAM13, max_games=k), float), index=df.key))

            rec["holdout"] = {"proxy": rho(px, tgt), "opt13": rho(o13, tgt),
                              **{f"w{w:.2f}": rho(zs((1 - w) * px + w * o13), tgt)
                                 for w in WGRID}}

            # ---- the other benchmark, the moment it exists ---------------------
            try:
                t = truth(SEASON)
                rec["dlebron_players"] = int(len(t))
                if len(t) > 50:
                    full_df = O.build(SEASON, prior=None)
                    fpx = zs(pd.Series(np.asarray(shipped_proxy(full_df), float),
                                       index=full_df.key))
                    fo13 = zs(pd.Series(np.asarray(O.opt13_possession_outcome(
                        full_df, SEASON, lam=LAM13), float), index=full_df.key))
                    rec["dlebron"] = {"proxy": rho(fpx, t), "opt13": rho(fo13, t),
                                      **{f"w{w:.2f}": rho(zs((1 - w) * fpx + w * fo13), t)
                                         for w in WGRID}}
            except Exception as e:
                rec["dlebron_players"] = 0
                note.append(f"no current-season D-LEBRON yet ({type(e).__name__})")

    # ---- churn against the previous run --------------------------------
            board = zs(0.80 * px + 0.20 * o13)
            prev = sorted(OUT.glob("board_*.json"))
            if prev:
                try:
                    old = pd.Series(json.loads(prev[-1].read_text()), dtype=float)
                    rec["churn_vs"] = prev[-1].name
                    rec["stability"] = rho(board, old)
                    a = set(board.nlargest(25).index); b = set(old.nlargest(25).index)
                    rec["top25_held"] = len(a & b)
                except Exception as e:
                    note.append(f"churn compare failed: {type(e).__name__}")
            (OUT / f"board_{today}_{int(n)}g.json").write_text(
                json.dumps({k2: round(float(v), 5) for k2, v in board.items()}))

            def _best(blk):
                w = max(WGRID, key=lambda x: (blk.get(f"w{x:.2f}") or -9))
                return w, blk.get(f"w{w:.2f}"), (blk.get(f"w{w:.2f}") or 0) - (blk.get("proxy") or 0)

            if rec.get("dlebron"):
                w, v, gain = _best(rec["dlebron"])
                rec["verdict"] = (f"{n} team games: vs current D-LEBRON, proxy "
                                  f"{rec['dlebron'].get('proxy')}, best blend w={w:.2f} -> {v} "
                                  f"(gain {gain:+.3f}). THIS is the usable number.")
            elif rec["holdout_interpretable"]:
                w, v, gain = _best(rec["holdout"])
                rec["verdict"] = (f"{n} team games: no D-LEBRON yet; holdout target is now "
                                  f"interpretable (reliability {rel}), proxy "
                                  f"{rec['holdout'].get('proxy')}, best blend w={w:.2f} -> {v} "
                                  f"(gain {gain:+.3f})")
            else:
                rec["verdict"] = (f"{n} team games: NOTHING DECIDABLE YET. Current-season "
                                  f"D-LEBRON unpublished, and the holdout target's own "
                                  f"split-half reliability is {rel}, below 0.15, so the "
                                  f"holdout scores in this record mean nothing. "
                                  f"Top-25 churn is the only live signal.")
        except Exception as e:
            # BBRef or pbpstats having a bad day must not erase the row; the series
            # is more useful with a gap that says why than with a missing date.
            note.append(f"metrics failed: {type(e).__name__}: {str(e)[:120]}")
            rec["verdict"] = (f"{n} team games: metrics could not be computed this run "
                              f"({type(e).__name__}). Source: {rec['box_source']}.")
    rec["notes"] = note
    with LOG.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))
    print(f"\n  {rec['verdict']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(report() if "--report" in sys.argv else main())
    except Exception as e:
        OUT.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as fh:
            fh.write(json.dumps({"date": dt.date.today().isoformat(),
                                 "fatal": f"{type(e).__name__}: {e}"}) + "\n")
        print(f"FATAL {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
