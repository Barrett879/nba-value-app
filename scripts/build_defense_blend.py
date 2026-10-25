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
VERSION = "v1"
CLIP = (-3.0, 4.0)          # D-LEBRON's own observed range is -2.94 to +3.99


def zs(s):
    sd = s.std()
    return (s - s.mean()) / (sd if sd and sd == sd else 1.0)


def main():
    season = sys.argv[1] if len(sys.argv) > 1 else "2026-27"
    prior = "2024-25" if season == "2026-27" else "2023-24"

    # D-LEBRON's real distribution, so the blend lands on the same scale
    dl = pd.read_parquet(ROOT / "cache" / "dlebron_all.parquet")
    recent = dl[dl.Season.isin(["2024-25", "2025-26"])]["D-LEBRON"].astype(float)
    mu, sd = float(recent.mean()), float(recent.std())

    df = O.build(season, prior=prior)
    z1 = zs(pd.Series(np.asarray(O.opt1_two_channel(df), float)))
    z4 = zs(pd.Series(np.asarray(O.opt4_career_blend(df, season=season), float)))
    blend = zs(W1 * z1 + (1 - W1) * z4)
    scaled = (blend * sd + mu).clip(*CLIP)

    espn, feed, missing = L.feed_gap(season)
    rebuilt = [str(g.get("GameId")) for g in L.games(season)
               if str(g.get("GameId")).startswith("E")]

    out = {
        "_meta": {
            "season": season, "generated": dt.date.today().isoformat(),
            "formula": f"{W1:.2f} * z(option 1) + {1 - W1:.2f} * z(option 4)",
            "scaled_to": {"source": "D-LEBRON 2024-25 and 2025-26",
                          "mean": round(mu, 4), "sd": round(sd, 4),
                          "clip": list(CLIP)},
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
        "values": {str(p): round(float(v), 4)
                   for p, v in zip(df.player, scaled)},
    }
    path = ROOT / "cache" / f"defense_blend_{season.replace('-', '_')}_{VERSION}.json"
    path.write_text(json.dumps(out, indent=1))

    print(f"wrote {path}")
    print(f"  {len(out['values'])} players, {df.g.min():.0f}-{df.g.max():.0f} team games")
    print(f"  scaled to D-LEBRON: mean {mu:+.3f}, sd {sd:.3f}, clipped to {CLIP}")
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
