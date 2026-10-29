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

W1 = 0.70
VERSION = "v9"  # v7: 2026-10-27 slate (4 games)

# The endpoints the defensive value is stretched to. Scaling is PIECEWISE about
# zero, so the best defender lands exactly on POS_MAX, the worst exactly on
# NEG_MIN, zero stays zero and the sign of everyone in between is preserved.
# That is what lets the negative end be moved on its own, to -4 or -3, without
# touching the positive end.
POS_MAX = 6.0
NEG_MIN = -4.0


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
    blend = zs(W1 * z1 + (1 - W1) * z4)
    pos, neg = float(blend.max()), float(blend.min())
    scaled = blend.copy()
    scaled[blend > 0] = blend[blend > 0] * (POS_MAX / pos) if pos > 0 else 0.0
    scaled[blend < 0] = blend[blend < 0] * (abs(NEG_MIN) / abs(neg)) if neg < 0 else 0.0
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
            "formula": f"{W1:.2f} * z(option 1) + {1 - W1:.2f} * z(option 4)",
            "scaled_to": {"endpoints": [NEG_MIN, POS_MAX],
                          "method": "piecewise linear about zero",
                          "dlebron_reference": {"mean": round(mu, 4),
                                                "sd": round(sd, 4)},
                          "warning": ("NOT on D-LEBRON's scale. Historical seasons "
                                      "carry real D-LEBRON, roughly -3 to +4, so "
                                      "this season's defenders are stretched wider "
                                      "than any past season's and the all-time pool "
                                      "is no longer like for like.")},
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
    print(f"  stretched to [{NEG_MIN:+.1f}, {POS_MAX:+.1f}] piecewise about zero"
          f"   (D-LEBRON's own reference: mean {mu:+.3f}, sd {sd:.3f})")
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
