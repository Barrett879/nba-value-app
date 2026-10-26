"""Back-test the defensive options on a season where a trusted answer exists.

    /opt/anaconda3/bin/python3 -u scripts/def_backtest.py

2024-25 is the newest season with both cached lineup data (260 games, ~17 per
team) and a finished full-season D-LEBRON. Each option is rebuilt at 2, 5, 10
and 17 team games and scored against that full-season D-LEBRON, so the question
is not "does this option agree with D-LEBRON at the end" but "how few games does
it need to get there". The prior-season channels come from 2023-24, which is
information a real October run would genuinely have.

Two honesty notes, both printed in the output:

  * the box columns (BLK/STL/DREB per game) come from the FINISHED 2024-25
    season, so at a 2-game cap any option that uses them is reading a rate it
    could not have known. The thinned column re-draws those counts as
    Poisson(games x rate)/games, averaged over DRAWS realisations, which keeps
    the mean and restores the variance a short sample really has.
  * D-LEBRON shrinks RAPM toward a box prior by construction, so it is not an
    independent referee for box-heavy options. It is the best available one.

Offline. Writes nothing the site reads.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import def_lib as L, def_options as O
from utils import normalize, _DLEBRON_PROXY_COEFS as PX

SEASON, PRIOR = "2024-25", "2023-24"
CAPS = (2, 5, 10, 17)
DRAWS = 20
BOXCOLS = ("BLK", "STL", "DREB")


def truth(season):
    d = pd.read_parquet(ROOT / "cache" / "dlebron_all.parquet")
    d = d[d.Season == season]
    ls = pd.read_parquet(ROOT / "cache" / f"league_stats_{season.replace('-', '_')}.parquet")
    m = d.merge(ls[["PLAYER_ID", "PLAYER_NAME"]], left_on="nba_id", right_on="PLAYER_ID")
    m["key"] = m.PLAYER_NAME.map(normalize)
    return m.drop_duplicates("key").set_index("key")["D-LEBRON"]


def shipped_proxy(df):
    """What the site scores today: the OLS box proxy, per-game inputs."""
    return (PX["intercept"] + PX["STL"] * df.STL + PX["BLK"] * df.BLK
            + PX["DREB"] * df.DREB + PX["PF"] * df.PF)


def options(cap):
    return [(1, "two-channel, frozen scale", O.opt1_two_channel, True),
            (3, "expected points from shot diet", O.opt3_expected_points, False),
            (4, "career-possession blend",
             lambda d: O.opt4_career_blend(d, season=SEASON), True),
            (5, "own-prior empirical Bayes", O.opt5_own_prior, True),
            (6, "ridge on opponent rim share",
             lambda d: O.opt6_ridge_rimshare(d, SEASON, max_games=cap), False),
            (7, "observed over expected", O.opt7_obs_over_exp, False),
            (11, "Markov reward pricing", O.opt11_markov, False),
            (13, "ridge on opponent points/poss",
             lambda d: O.opt13_possession_outcome(d, SEASON, max_games=cap), False),
            (0, "SHIPPED box proxy (baseline)", shipped_proxy, True)]


def sp(s, t):
    """Spearman of a score against the truth, on the players both cover."""
    j = pd.DataFrame({"s": s.values, "k": s.index}).set_index("k").join(t.rename("t"), how="inner")
    j = j.dropna()
    return (float(j.s.corr(j.t, method="spearman")), len(j)) if len(j) > 30 else (np.nan, len(j))


def thin(df, games, rng):
    """Re-draw the per-game box rates as if only `games` games had been played."""
    d = df.copy()
    for c in BOXCOLS:
        lam = np.clip(d[c].values, 0, None) * games
        d[c] = rng.poisson(lam) / games
    return d


def main():
    t = truth(SEASON)
    print(f"\n{'=' * 90}")
    print(f"  BACK-TEST  {SEASON}   truth = full-season D-LEBRON ({len(t)} players)")
    print(f"  prior-season channels from {PRIOR}; lineups capped at the first N games per team")
    print(f"{'=' * 90}")

    curve, pool, built = {}, {}, {}
    for cap in CAPS:
        df = O.build(SEASON, prior=PRIOR, max_games=cap)
        built[cap] = df
        pool[cap] = len(df)
        rng = np.random.default_rng(20241025)
        for n, name, fn, usesbox in options(cap):
            try:
                s = pd.Series(np.asarray(fn(df), dtype=float), index=df.key)
            except Exception as e:
                curve[(n, cap)] = (np.nan, np.nan, f"{type(e).__name__}")
                continue
            r, cov = sp(s, t)
            rt = np.nan
            if usesbox:
                vals = []
                for _ in range(DRAWS):
                    try:
                        st = pd.Series(np.asarray(fn(thin(df, cap, rng)), dtype=float), index=df.key)
                        vals.append(sp(st, t)[0])
                    except Exception:
                        pass
                rt = float(np.nanmean(vals)) if vals else np.nan
            curve[(n, cap)] = (r, rt, cov)
        print(f"  built cap={cap:>2}: {len(df)} qualifying players")

    print(f"\n{'-' * 90}")
    print("  SPEARMAN vs full-season D-LEBRON, by games of lineup data")
    print("  (thinned) = box rates re-drawn for that sample size; blank = option uses no box stats")
    print(f"{'-' * 90}")
    hdr = "".join(f"{c:>7}g" for c in CAPS)
    print(f"  {'option':<34}{hdr}   {'thinned at 2g / 17g':>22}")
    order = [n for n, _, _, _ in options(CAPS[-1])]
    for n, name, _, usesbox in options(CAPS[-1]):
        row = "".join(
            (f"{curve[(n, c)][0]:>+8.3f}" if curve[(n, c)][0] == curve[(n, c)][0] else f"{'n/a':>8}")
            for c in CAPS)
        tail = ""
        if usesbox:
            a, b = curve[(n, CAPS[0])][1], curve[(n, CAPS[-1])][1]
            tail = f"   {a:>+10.3f} / {b:>+8.3f}" if a == a else ""
        tag = f"#{n}" if n else "--"
        print(f"  {tag:>3} {name:<30}{row}{tail}")

    df = built[CAPS[-1]]
    inpool = t.reindex(df.key).dropna()
    dl = inpool.sort_values(ascending=False)
    top10 = set(dl.head(10).index); bot10 = set(dl.tail(10).index)
    name_of = dict(zip(df.key, df.player))
    print(f"\n{'-' * 90}")
    print(f"  D-LEBRON's own best and worst 10 inside our {len(inpool)}-player pool")
    print(f"{'-' * 90}")
    for i in range(10):
        a, b = dl.index[i], dl.index[-(i + 1)]
        print(f"   {i+1:>2}. {str(name_of.get(a, a))[:24]:24} {dl.iloc[i]:+6.2f}"
              f"      {i+1:>2}. {str(name_of.get(b, b))[:24]:24} {dl.iloc[-(i+1)]:+6.2f}")

    print(f"\n{'-' * 90}")
    print(f"  EACH OPTION AT {CAPS[-1]} GAMES -- best/worst 10, ** = in D-LEBRON's own best/worst 10")
    print(f"{'-' * 90}")
    for n, name, fn, _ in options(CAPS[-1]):
        try:
            s = pd.Series(np.asarray(fn(df), dtype=float), index=df.index)
        except Exception as e:
            print(f"\n  option {n} failed: {e}"); continue
        d = df.copy(); d["s"] = s
        bt, wt = d.nlargest(10, "s"), d.nsmallest(10, "s")
        hb = sum(1 for k in bt.key if k in top10); hw = sum(1 for k in wt.key if k in bot10)
        tag = f"OPTION {n}" if n else "SHIPPED PROXY"
        print(f"\n  {tag}  {name}   [best {hb}/10, worst {hw}/10 vs D-LEBRON]")
        for i in range(10):
            b, w = bt.iloc[i], wt.iloc[i]
            mb = "**" if b.key in top10 else "  "
            mw = "**" if w.key in bot10 else "  "
            print(f"  {mb}{i+1:>2}. {str(b.player)[:22]:22} {b.s:+8.2f}   "
                  f"{mw}{i+1:>2}. {str(w.player)[:22]:22} {w.s:+8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
