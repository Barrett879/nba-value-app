"""Build the shippable defensive blend for the live season.

    /opt/anaconda3/bin/python3 -u scripts/build_defense_blend.py [season]

Writes cache/defense_blend_<season>_<VERSION>.json: {player name: value}, for
the season the site is scoring while BBall Index has not yet published a real
D-LEBRON for it.

score = w_base * z(the pre-Trae blend) + W_DBPM_LITE * z(DBPM)
      + w_chg  * z(charges drawn, as a posterior rate)

where the pre-Trae blend is W1 * z(option 1) + (1 - W1) * z(option 4), w_chg is
W_CHG damped by how many charges the league has drawn, and w_base is whatever is
left. Then shrunk by min/(min+SHRINK_MIN) and stretched PIECEWISE ABOUT ZERO so
the best defender lands exactly on POS_MAX and the worst exactly on NEG_MIN.
base_score multiplies it by utils.defense_weight(season), 1.75 for the live
season.

The spine is the pre-Trae-Young blend, restored on 2026-11-02 at Barrett's
request after the 2026-11-01 DBPM rebuild; the DBPM and charges terms were added
the same day, also his, and both are deliberately small. See the notes above the
weights for what each one is worth and for what is known to be wrong with them.

The stretch is a display decision, Barrett's, 2026-10-25. It is deliberately
NOT D-LEBRON's own scale: real D-LEBRON across the 17 seasons in
cache/dlebron_all.parquet runs -3.55 .. +4.41 and has never reached either
endpoint, so the live season is wider than any finished one and the all-time
pool favours current players structurally. compute() still returns D-LEBRON's
mean and sd and main() still writes them to _meta, so what the alternative
scale would have been stays on the record beside what shipped.

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
#     THE BLEND BELOW, which ships     +0.591    +0.599
#     DBPM alone                       +0.820    +0.826
#     the DBPM composite, reverted     +0.872    +0.895
#
# Everything above is the research record of the rebuild that this file no
# longer implements, kept because it is the measurement the next change should
# argue with. The one live claim in it is the last row's absence: the shipping
# formula is the FIRST row.
#
# DBPM would be legal under the current-season constraint, for whenever this is
# revisited: its coefficients were fit on multi-year RAPM but are now fixed
# constants, so applying them needs only this season's box score, the same thing
# utils._DLEBRON_PROXY_COEFS already does. What is NOT legal, and has never been
# used, is a player-specific prior carried across seasons (EPM, DARKO, DRIP).
#
# REVERTED 2026-11-02, Barrett's call: "i kind of want to change it back to
# before i made the trae young comment, and then adjust from there". So the
# composite is the pre-Trae on/off blend again, W1 * option 1 + (1-W1) * option
# 4, and W_DBPM / W_FAST / W_ONOFF are gone with the channels they weighted.
#
# What that costs, measured and stated rather than discovered later: against
# real D-LEBRON on finished seasons the pre-Trae blend scored +0.535 (2024-25)
# and +0.591 (2023-24); the DBPM composite it replaces scored +0.872 and +0.895.
# Trae Young returns to +2.19, 43rd of 353, which is the number that started
# this. None of that is an argument against the revert -- it is Barrett's call
# and he made it knowing the figures -- but it is why the rebuild existed.
#
# For adjusting from here, the useful measurement is that the pre-Trae metric's
# two halves pull in opposite directions. Its BOX half, which lives inside
# option 1 as z(sqrt BLK) + z(STL) + 0.3 z(DREB) at 55% of option 1, scores
# +0.716 alone on 11 seasons. The full blend scores +0.535. The ON/OFF channels
# are what drag it down, and they are 45% of option 1 plus all of option 4.
# Raising the box half is the cheapest adjustment available and it is also the
# playmaking-heavy direction, since steals carry a full z-term there.

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
VERSION = "v21"  # v7: 2026-10-27 slate (4 games)

# The display scale, Barrett's call on 2026-10-25 (c9c804f, then e1778ff pulled
# the floor in from -6 to -4). The stretch is piecewise about zero, so the best
# defender lands exactly on POS_MAX, the worst exactly on NEG_MIN, zero stays
# zero and every sign in between is preserved. Piecewise is what lets the two
# ends move independently.
#
# Restored 2026-11-01 after the composite rebuild dropped it. The rebuild put
# the values on D-LEBRON's own mean and sd, which is the right scale for
# comparing this season against finished ones but is NOT what the site was
# agreed to show. The two choices are independent: the stretch is monotone and
# fixes zero, so it changes no defensive ranking, only the magnitudes.
POS_MAX = 6.0
NEG_MIN = -4.0

# ── The slight DBPM blend, and charges drawn ─────────────────────────────────
# Barrett, 2026-11-02: "can we include charges drawn and blend slightly with
# adjusted defensive box plus minus?"
#
# W_DBPM_LITE is the "slightly". There is NO interior optimum to find: accuracy
# against real D-LEBRON rises monotonically with it, so the number is a judgement
# about how much of this metric's character to keep, not a fitted value. Graded
# on the two seasons that have both lineup caches and real D-LEBRON, 2024-25 as
# the fit and 2023-24 as the HELD-OUT season, at W_CHG = 0.10:
#
#     W_DBPM_LITE   2024-25   2023-24    mean
#         0.00       +0.584    +0.608    +0.596   <- the pre-Trae blend alone
#         0.15       +0.666    +0.694    +0.680   <- shipped, closes ~35%
#         0.30       +0.743    +0.768    +0.755
#         0.60       +0.836    +0.845    +0.840
#
# CHARGES DRAWN is the defensive charge stat, def_lib loads it separately from
# "Charge Fouls", which is charges COMMITTED and is an offensive act. Two things
# about it are worth knowing before anyone raises W_CHG:
#
#   It correlates NEGATIVELY with real D-LEBRON on its own, about -0.02 to -0.06
#   in every season and every functional form. It still helps in the blend
#   (+0.009 on the fit season, +0.011 held out, monotone to at least 0.20), which
#   is a suppressor effect, not a measurement of charge-taking being valuable.
#
#   A z-score of a nearly-all-zero count is STRUCTURALLY A SPIKE, and no amount
#   of shrinking the rate fixes it, because z re-expands whatever the shrink
#   compressed. On 2026-11-02 the league had drawn 23 charges in total, 19
#   players had one and two had two, and the raw z ran -0.49 .. +7.17: Draymond
#   Green's two charges in 142 minutes would have moved him most of a composite
#   sd. Hence both guards below.
W_DBPM_LITE = 0.15      # the "slightly"; monotone, so this is a taste dial
W_CHG = 0.10            # full-season weight for charges drawn, before damping
CHG_PRIOR_EVENTS = 1.0  # per-player prior, in league-average events
CHG_LEAGUE_E0 = 300.0   # league charges before the channel reaches full weight
CHG_CLIP = 3.0          # cap on the charges z, which is one-sided by nature


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
    base = zs(W1 * z1 + (1 - W1) * z4)

    box = L.players(season).set_index("key")
    key = df.key.astype(str)
    mins = pd.Series(box.MIN.reindex(key).values, index=df.index).replace(0, np.nan)
    total_min = (mins * pd.Series(box.GP.reindex(key).values, index=df.index))

    # Charges drawn, as a posterior rate rather than a raw one: a player with one
    # charge in 60 minutes is not twice the defender of one with the same charge
    # in 120. CHG_PRIOR_EVENTS worth of league-average rate is the prior, the z
    # is clipped because the distribution is one-sided while counts are small,
    # and the whole channel is then damped by how many charges THE LEAGUE has
    # drawn. That last factor is the honest one: early in a season the channel
    # carries no information, and that is a fact about the league's sample, not
    # about any player. On 2026-11-02 it stood at 23/(23+300) = 0.071, so the
    # channel is nearly inert and opens up as the season fills.
    chg_total = float(df.CHGD.sum())
    rate0 = chg_total / max(float(total_min.sum()), 1.0)
    prior_min = CHG_PRIOR_EVENTS / rate0 if rate0 > 0 else 0.0
    chg = zs((df.CHGD + CHG_PRIOR_EVENTS)
             / (total_min.fillna(0.0) + prior_min) * 36).clip(-CHG_CLIP, CHG_CLIP)
    w_chg = W_CHG * (chg_total / (chg_total + CHG_LEAGUE_E0))

    # DBPM where it exists. A player without one keeps the other two terms
    # renormalised rather than being handed the pool median, which would score
    # "no measurement yet" as "exactly average".
    dbz = zs(pd.Series(box.dbpm.reindex(key).values, index=df.index))
    has = dbz.notna()
    w_base = 1.0 - W_DBPM_LITE - w_chg
    comp = pd.Series(index=df.index, dtype=float)
    comp[has] = (w_base * base[has] + W_DBPM_LITE * dbz[has] + w_chg * chg[has])
    if (~has).any():
        comp[~has] = ((w_base * base[~has] + w_chg * chg[~has])
                      / (w_base + w_chg))

    # Rates need a denominator before they mean anything. See SHRINK_MIN. This
    # is the ONE piece of the 2026-11-01 rebuild kept across the revert, and
    # Barrett was told so: the pre-Trae metric had no shrink, and without it
    # Kawhi Leonard is the 4th best defender in the league on 52 total minutes
    # and Collin Gillespie the 10th on 106. Set SHRINK_MIN = 0.0 to have the
    # pre-Trae metric with nothing added at all.
    comp = zs(comp) * (total_min / (total_min + SHRINK_MIN)).fillna(0.0)

    # Stretched piecewise about zero onto [NEG_MIN, POS_MAX]. The pivot is
    # zero, which is the pool mean BEFORE the stretch; it is not the mean
    # after, and that distinction is the whole of the first cost below.
    #
    # Three costs, all real, all measured on the 2026-11-01 pool, none of them
    # a reason not to do it -- the stretch is the agreed display scale -- but
    # each one is why the rebuild had dropped it, so none of them gets to be
    # rediscovered by surprise:
    #
    # 1. IT SHIFTS THE MEAN, because the two halves scale independently and
    #    nothing makes the pre-stretch z symmetric. MEASURED ON THE 2026-11-02
    #    POOL, not carried over from the receipts: this blend runs -1.93 .. +4.20,
    #    so it is RIGHT-skewed and the negative half is multiplied by 2.07 while
    #    the positive half is multiplied by 1.43. The pool mean lands at -0.250,
    #    which at weight 1.75 is about -0.44 base score to every covered live
    #    player. The DBPM composite it replaced was left-skewed the other way
    #    (-4.63 .. +3.12, mean +0.412, about +0.73 base score), so THE REVERT
    #    FLIPS THE SIGN of this shift: live players were being favoured against
    #    the all-time pool and are now penalised against it. The receipts record
    #    this blend at +0.04 on the 2026-10-25 pool; it is -0.250 today, which is
    #    why the number gets re-measured and not quoted. It is UNIFORM within the
    #    season either way, so it moves no rank on the live board; it lands
    #    entirely on cross-season comparisons.
    #
    # 2. THE ENDPOINTS ARE POOL-RELATIVE, so one player moves everyone. Both
    #    ends are set by whoever is most extreme, which on a short season is
    #    often a low-minutes player: on 2026-11-01 the negative anchor had 101
    #    total minutes, and dropping him rescaled every negative player by
    #    about +59%. SHRINK_MIN damps that, it does not remove it.
    #
    # 3. IT IS NOT D-LEBRON'S SCALE, so the live season is wider than any
    #    finished one (real D-LEBRON: -3.55 .. +4.41 across the 17 seasons in
    #    cache/dlebron_all.parquet, never once reaching either endpoint).
    #
    # mu and sd are still returned and still written to _meta, so the scale
    # this would have had is on the record beside the scale it has.
    z = zs(comp)
    pos, neg = float(z.max()), float(z.min())
    scaled = z.copy()
    scaled[z > 0] = z[z > 0] * (POS_MAX / pos) if pos > 0 else 0.0
    scaled[z < 0] = z[z < 0] * (abs(NEG_MIN) / abs(neg)) if neg < 0 else 0.0
    # The term weights go back as a dict rather than as more positional values:
    # this tuple has grown twice already and each time it broke def_monitor,
    # which unpacks it. Callers that only want the values keep working, and the
    # monitor now absorbs anything appended here.
    terms = {"w_base": float(w_base), "w_chg": float(w_chg),
             "chg_total": chg_total,
             "chg_damping": chg_total / (chg_total + CHG_LEAGUE_E0)}
    return df, scaled, mu, sd, neg, pos, terms


def main():
    season = sys.argv[1] if len(sys.argv) > 1 else "2026-27"
    df, scaled, mu, sd, z_lo, z_hi, terms = compute(season)
    w_base, w_chg = terms["w_base"], terms["w_chg"]
    chg_total = terms["chg_total"]

    espn, feed, missing = L.feed_gap(season)
    rebuilt = [str(g.get("GameId")) for g in L.games(season)
               if str(g.get("GameId")).startswith("E")]

    out = {
        "_meta": {
            "season": season, "generated": dt.date.today().isoformat(),
            "formula": (f"base * z({W1:.2f} opt1 + {1 - W1:.2f} opt4) + "
                        f"{W_DBPM_LITE:.2f} * z(DBPM) + chg * z(charges drawn), "
                        f"shrunk by min/(min+{SHRINK_MIN:.0f})"),
            "term_weights": {"pre_trae_blend": round(float(w_base), 4),
                             "dbpm": W_DBPM_LITE,
                             "charges_drawn": round(float(w_chg), 4),
                             "charges_full_weight": W_CHG,
                             "charges_league_total": int(chg_total),
                             "charges_damping": round(
                                 chg_total / (chg_total + CHG_LEAGUE_E0), 4)},
            "validation": {"method": "Spearman vs real D-LEBRON, 40+ GP and 15+ mpg",
                           # Reverted to this blend on 2026-11-02 at Barrett's
                           # request. The figures are recorded rather than
                           # removed so the artifact says what it is worth.
                           "this_blend": {"2024-25": 0.591, "2023-24": 0.599},
                           "dbpm_composite_replaced": {"2024-25": 0.872,
                                                       "2023-24": 0.895},
                           "reverted": "2026-11-02, Barrett's call"},
            "scaled_to": {# pages/Rankings.py sizes its defensive bar from these
                          # and falls back to (-4, 6) if the key is missing, so
                          # the fallback and the real value agree again.
                          "endpoints": [NEG_MIN, POS_MAX],
                          "method": "piecewise linear about zero",
                          # What the scale WOULD be if it were left on D-LEBRON's
                          # own mean and sd. Kept so the stretch factor at each
                          # end is recoverable from the artifact alone.
                          "dlebron_reference": {"mean": round(mu, 4),
                                                "sd": round(sd, 4)},
                          "observed_before_stretch": [round(float(z_lo), 2),
                                                      round(float(z_hi), 2)],
                          "warning": ("NOT on D-LEBRON's scale. Historical seasons "
                                      "carry real D-LEBRON, which has never run "
                                      "outside -3.55 .. +4.41 in 17 seasons, so "
                                      "this season's defenders sit on a wider scale "
                                      "than any past season and the all-time pool "
                                      "favours current players structurally. The "
                                      "endpoints are pool-relative: one player's "
                                      "input moves everyone.")},
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
    print(f"  {w_base:.3f} pre-Trae blend + {W_DBPM_LITE:.2f} DBPM"
          f" + {w_chg:.3f} charges drawn ({int(chg_total)} league-wide,"
          f" damping {chg_total / (chg_total + CHG_LEAGUE_E0):.3f}),"
          f" stretched to [{NEG_MIN:+.1f}, {POS_MAX:+.1f}] piecewise about zero")
    print(f"  pre-stretch z {z_lo:+.2f} .. {z_hi:+.2f}, so the negative half is"
          f" scaled x{abs(NEG_MIN) / abs(z_lo):.2f} and the positive half"
          f" x{POS_MAX / z_hi:.2f} (D-LEBRON would be mean {mu:+.3f}, sd {sd:.3f})")
    print(f"  blend range {scaled.min():+.2f} .. {scaled.max():+.2f}"
          f"  (real D-LEBRON over 17 seasons: -3.55 .. +4.41)")
    print(f"  {len(rebuilt)} of {len(L.games(season))} lineup games rebuilt from ESPN,"
          f" {len(missing)} still missing")
    top = df.assign(v=scaled.values).nlargest(5, "v")
    bot = df.assign(v=scaled.values).nsmallest(5, "v")
    print("  top 5:   " + ", ".join(f"{r.player} {r.v:+.2f}" for _, r in top.iterrows()))
    print("  worst 5: " + ", ".join(f"{r.player} {r.v:+.2f}" for _, r in bot.iterrows()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
