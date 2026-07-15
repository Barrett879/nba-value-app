"""Precompute the homepage "Biggest movers" card: players whose Barrett Score
changed most vs last season.

Rule: rotation minutes (>= 1000 total) in BOTH seasons, so the delta reflects a
genuine year-over-year performance change, not an availability swing (an injured
star's score cratering is real but obvious). Writes cache/movers_v1.json, read
by app.py on the homepage -- NO request-path computation. Re-run after the
season's raw cache updates (new filename _vN if the schema/season changes, per
the /data seeding rule).
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "cache" / "all_seasons_0_v7.parquet"
OUT = ROOT / "cache" / "movers_v1.json"

CUR, PREV = "2025-26", "2024-25"
MIN_FLOOR = 1000   # total minutes required in BOTH seasons
N = 5              # risers + fallers shown


def main() -> None:
    df = pd.read_parquet(SRC)
    cur = df[df["Season"] == CUR][["Player", "PLAYER_ID", "Team", "barrett_score", "total_min"]]
    prev = (df[df["Season"] == PREV][["PLAYER_ID", "barrett_score", "total_min"]]
            .rename(columns={"barrett_score": "prev_score", "total_min": "prev_min"}))
    m = cur.merge(prev, on="PLAYER_ID")
    m = m[(m["total_min"] >= MIN_FLOOR) & (m["prev_min"] >= MIN_FLOOR)].copy()
    m["delta"] = m["barrett_score"] - m["prev_score"]

    def rows(frame):
        out = []
        for _, r in frame.iterrows():
            out.append({
                "player": str(r["Player"]),
                "team": str(r["Team"]),
                "cur": round(float(r["barrett_score"]), 2),
                "prev": round(float(r["prev_score"]), 2),
                "delta": round(float(r["delta"]), 2),
            })
        return out

    payload = {
        "cur_season": CUR,
        "prev_season": PREV,
        "min_floor": MIN_FLOOR,
        "risers": rows(m.sort_values("delta", ascending=False).head(N)),
        "fallers": rows(m.sort_values("delta").head(N)),
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT.name}  ({len(payload['risers'])} risers, {len(payload['fallers'])} fallers, "
          f"pool={len(m)} with {MIN_FLOOR}+ min both seasons)")
    print("  top riser:", payload["risers"][0]["player"], f"+{payload['risers'][0]['delta']}")
    print("  top faller:", payload["fallers"][0]["player"], payload["fallers"][0]["delta"])


if __name__ == "__main__":
    sys.exit(main())
