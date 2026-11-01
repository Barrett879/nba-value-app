"""Build the shippable defensive blend for the live season.

    /opt/anaconda3/bin/python3 -u scripts/build_defense_blend.py [season]

Writes cache/defense_blend_<season>_v1.json: {player name: value} on D-LEBRON's
own scale, for the season the site is scoring while BBall Index has not yet
published a real D-LEBRON for it.

score = 0.70 * z(option 1) + 0.30 * z(option 4), then rescaled to D-LEBRON's
observed mean and standard deviation so it is a drop-in for the term base_score
already multiplies by 2. Shipping the raw z would silently compress the
defensive term by about 10%, since a z has sd 1.00 and D-LEBRON's is 1.11.

It CANNOT be computed on Render: it needs the pbpstats lineup cache plus the
ESPN rebuilds, which is already 5 MB and heads toward ~190 MB across a season.
So it is precomputed here and committed, like player_hub_pcv_v2.json. The
consequence, stated plainly because it is easy to forget: the site's defensive
value is as fresh as the last push, not hourly.

Coverage is partial by design. Players below the possession floor get no blend
value and keep the box proxy; the split is printed and recorded in the file.
"""
from __future__ import annotations
import json, logging, sys
import datetime as dt
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
logging.disable(logging.WARNING)
import numpy as np, pandas as pd
import def_lib as L, def_options as O
logging.disable(logging.NOTSET)

# ── The composite ────────────────────────────────────────────────────────────
# Rebuilt 2026-11-01 after Barrett asked why Trae Young scored +2.03 on defence.
# He has been between the 0th and 8th percentile of real D-LEBRON for seven
# straight seasons; the old formula had him at the 88th. The cause was that the
# whole thing was built from lineup ON/OFF channels, which are the noisiest
# defensive measurement in the published record: Basketball Index put defensive
# four-factor on/off at 0.21-0.47 year over year across 2,895 player pairs,
# against 0.87 for box-anchored measures, and BPM's own author calls raw
# one-season on/off "basically useless". Our own test agreed: every on/off
# channel scored under +0.22 against real D-LEBRON, and the corner-3 channel
# that drove Trae's number scored -0.05.
#
# Measured on finished seasons against real D-LEBRON (Spearman, 40+ GP,
# 15+ mpg). Weights were chosen on 2024-25 and HELD OUT on 2023-24:
#
#                                    2024-25   2023-24
#     the old on/off blend            +0.591    +0.599
#     DBPM alone                      +0.820    +0.826
#     this composite                  +0.872    +0.895
#
# DBPM is legal under the current-season constraint: its coefficients were fit
# on multi-year RAPM but are now fixed constants, so applying them needs only
# this season's box score. That is the same thing utils._DLEBRON_PROXY_COEFS
# already does. What is NOT legal, and is not used, is a player-specific prior
# carried across seasons (EPM, DARKO, DRIP).
#
# The FAST channel is blocks and defensive rebound rate, which arrive earliest:
# DARKO's published half-lives are ~41 games for blocks and ~46 for defensive
# rebounds against ~75 for steal rate, and our own disjoint split-half on this
# season agreed independently (DREB +0.52, BLK +0.34, STL +0.07). Steals are
# deliberately NOT in it: they are the slowest-arriving channel and they are
# what a gambling guard accumulates.
W_DBPM, W_FAST, W_ONOFF = 0.60, 0.30, 0.10

# Minutes shrink, in TOTAL minutes. Every channel here is a rate, and a rate on
# a tiny denominator is not a measurement: the first build of this composite put
# Kyle Anderson first on 20 total minutes and Charles Bassey second on 26, ahead
# of Wembanyama. Shrinking toward the pool mean by minutes played puts
# Wembanyama, Dyson Daniels and Walker Kessler back in the top three and drops
# those two to 63rd and 62nd.
#
# Why a shrink at all, rather than converting to per-game impact: Barrett asked
# exactly that, and per-game is the more consistent basis WITHIN base_score,
# where every other term is per-game volume. It loses on a bigger consistency.
# The site scores 1973 to today and every finished season's defensive term is
# real D-LEBRON, which is a RATE. A per-game live season would be incomparable
# with the fifty-odd seasons beside it, which is the same sin the old metadata
# warning flagged about the stretched scale.
#
# THIS IS THE ONE NUMBER HERE THAT IS NOT BACKTESTED, and that is worth saying.
# The finished-season test cannot see it: that sample is filtered to 40+ games
# and 15+ mpg, where every value of M scores +0.872 because nobody is short of
# minutes. It is set instead as the lightest value that clears sub-30-minute
# players out of the top of the live board. At 7 games the pool median is ~115
# total minutes, so a median player carries 115/265 = 0.43 of his raw signal.
#
# It cannot be backtested, and not for want of trying: D-LEBRON only covers
# rotation players, so the smallest player in ANY finished-season validation
# sample has 500 total minutes. The population this constant exists to handle --
# 20 to 150 minutes -- is absent from every available target, and the accuracy
# curve is flat (0.861 to 0.863) at every value of M as a result.
SHRINK_MIN = 150.0

W1 = 0.70          # inside the on/off sub-blend, which is now only 10% of the whole
VERSION = "v14"  # v7: 2026-10-27 slate (4 games)

# Removed 2026-11-01. compute() scales to D-LEBRON's own mean and sd now, and
# _meta["scaled_to"]["endpoints"] carries the OBSERVED range instead, which is
# what pages/Rankings.py sizes its defensive bar from.


def zs(s):
    sd = s.std()
    return (s - s.mean()) / (sd if sd and sd == sd else 1.0)


def compute(season, prior=None):
    """The scaled defensive value per player. ONE definition, so the daily
    monitor measures drift against exactly what a rebuild would write rather
    than against its own re-derivation of the formula."""
    prior = prior or ("2024-25" if season == "2026-27" else "2023-24")
    dl = pd.read_parquet(ROOT / "cache" / "dlebron_all.parquet")
    recent = dl[dl.Season.isin(["2024-25", "2025-26"])]["D-LEBRON"].astype(float)
    mu, sd = float(recent.mean()), float(recent.std())

    df = O.build(season, prior=prior)
    z1 = zs(pd.Series(np.asarray(O.opt1_two_channel(df), float)))
    z4 = zs(pd.Series(np.asarray(O.opt4_career_blend(df, season=season), float)))
    onoff = zs(W1 * z1 + (1 - W1) * z4)

    # Box spine and the fast channel, both from THIS season only.
    box = L.players(season).set_index("key")
    key = df.key.astype(str)
    dbpm = pd.Series(box.dbpm.reindex(key).values, index=df.index)
    mins = pd.Series(box.MIN.reindex(key).values, index=df.index).replace(0, np.nan)
    per36 = lambda c: pd.Series(box[c].reindex(key).values, index=df.index) / mins * 36
    blk36, dreb36 = per36("BLK"), per36("DREB")
    fast = zs(blk36.fillna(blk36.median())) + zs(dreb36.fillna(dreb36.median()))

    # A player with no dbpm yet keeps the other two, renormalised, rather than
    # being dropped or silently zero-filled. On 2026-11-01 that was 30 of 349,
    # among them Robert Williams III and Bobby Portis Jr.
    has = dbpm.notna()
    comp = pd.Series(index=df.index, dtype=float)
    if has.any():
        comp[has] = (W_DBPM * zs(dbpm[has]) + W_FAST * zs(fast[has])
                     + W_ONOFF * zs(onoff[has]))
    if (~has).any():
        comp[~has] = (W_FAST * zs(fast[~has]) + W_ONOFF * zs(onoff[~has])) \
                     / (W_FAST + W_ONOFF)

    # Rates need a denominator before they mean anything. See SHRINK_MIN.
    total_min = (mins * pd.Series(box.GP.reindex(key).values, index=df.index))
    comp = zs(comp) * (total_min / (total_min + SHRINK_MIN)).fillna(0.0)

    # On D-LEBRON's OWN scale, which is what this was always supposed to be: a
    # drop-in for the term base_score multiplies. The piecewise stretch to
    # [-4, +6] that used to live here forced the extremes onto fixed endpoints
    # regardless of sample, which made the live season's spread wider than any
    # real season's (sd 1.76 against D-LEBRON's 1.11) and meant one player's
    # input moved everyone, since the endpoints are pool-relative.
    scaled = zs(comp) * sd + mu
    return df, scaled, mu, sd


def main():
    season = sys.argv[1] if len(sys.argv) > 1 else "2026-27"
    df, scaled, mu, sd = compute(season)

    espn, feed, missing = L.feed_gap(season)
    rebuilt = [str(g.get("GameId")) for g in L.games(season)
               if str(g.get("GameId")).startswith("E")]

    out = {
        "_meta": {
            "season": season, "generated": dt.date.today().isoformat(),
            "formula": (f"{W_DBPM:.2f} * z(DBPM) + {W_FAST:.2f} * z(BLK/36 + DREB/36) "
                        f"+ {W_ONOFF:.2f} * z({W1:.2f} opt1 + {1 - W1:.2f} opt4), "
                        f"shrunk by min/(min+{SHRINK_MIN:.0f})"),
            "validation": {"method": "Spearman vs real D-LEBRON, 40+ GP and 15+ mpg",
                           "weights_fit_on": "2024-25", "held_out_on": "2023-24",
                           "old_onoff_blend": {"2024-25": 0.591, "2023-24": 0.599},
                           "dbpm_alone": {"2024-25": 0.820, "2023-24": 0.826},
                           "this_composite": {"2024-25": 0.872, "2023-24": 0.895}},
            "scaled_to": {# The OBSERVED range, not fixed endpoints. pages/Rankings.py
                          # reads this to size its defensive bar and falls back to
                          # (-4, 6) if it is missing, which would mis-scale every bar
                          # now that the scale is D-LEBRON's rather than a stretch.
                          "endpoints": [round(float(scaled.min()), 2),
                                        round(float(scaled.max()), 2)],
                          "method": "z, then D-LEBRON's own mean and sd",
                          "dlebron_reference": {"mean": round(mu, 4),
                                                "sd": round(sd, 4)},
                          "note": ("ON D-LEBRON's scale, so historical seasons and "
                                   "this one are like for like. Replaced a piecewise "
                                   "stretch to [-4, +6] that forced the extremes onto "
                                   "fixed endpoints regardless of sample.")},
            "players": int(len(df)),
            "team_games_max": int(df.g.max()), "team_games_min": int(df.g.min()),
            "lineup_games": len(L.games(season)),
            "games_played_espn": sum(espn.values()) // 2,
            "games_rebuilt_from_espn": len(rebuilt),
            "games_still_missing": len(missing),
            "caveat": ("Partial coverage: players below the possession floor are "
                       "absent and must keep the box proxy. Not computable on "
                       "Render, so this is only as fresh as the last push."),
        },
        # SORTED, so the file is deterministic. Written in frame order it came
        # out reordered on every rebuild, producing 79-line diffs with zero
        # values changed -- which makes "has anything actually moved?"
        # unanswerable by diff, and that is what the daily drift check reads.
        "values": {k: round(float(v), 4) for k, v in
                   sorted(zip(df.player.astype(str), scaled), key=lambda kv: kv[0])},
    }
    path = ROOT / "cache" / f"defense_blend_{season.replace('-', '_')}_{VERSION}.json"
    path.write_text(json.dumps(out, indent=1))

    print(f"wrote {path}")
    print(f"  {len(out['values'])} players, {df.g.min():.0f}-{df.g.max():.0f} team games")
    print(f"  {W_DBPM:.2f} DBPM + {W_FAST:.2f} fast(BLK+DREB) + {W_ONOFF:.2f} on/off,"
          f" on D-LEBRON's own scale (mean {mu:+.3f}, sd {sd:.3f})")
    print(f"  blend range {scaled.min():+.2f} .. {scaled.max():+.2f}"
          f"  (D-LEBRON's own: -2.94 .. +3.99)")
    print(f"  {len(rebuilt)} of {len(L.games(season))} lineup games rebuilt from ESPN,"
          f" {len(missing)} still missing")
    top = df.assign(v=scaled.values).nlargest(5, "v")
    bot = df.assign(v=scaled.values).nsmallest(5, "v")
    print("  top 5:   " + ", ".join(f"{r.player} {r.v:+.2f}" for _, r in top.iterrows()))
    print("  worst 5: " + ", ".join(f"{r.player} {r.v:+.2f}" for _, r in bot.iterrows()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
