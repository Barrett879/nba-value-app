"""Precompute crawlable team-page data: every team's roster ranked by value.

serve.py serves a real HTML page at /team/<ABBR> (Google-indexable, unique
content per team) rendered from this JSON -- no model work on the request path.
Re-run after the board/model updates. One page per NBA team gives 30 landing
pages targeting "<team> player value / contracts / who's overpaid" searches,
each cross-linking into the app.

Output: cache/team_pages.json = {
  "season": "2025-26",
  "teams": { "LAL": {"abbr","name","players":[{n,s,sal,val,vd,r}], "tot":{...}} }
}
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import utils  # noqa: E402

OUT = ROOT / "cache" / "team_pages.json"


def _team_names() -> dict:
    """abbreviation -> full team name (falls back to the abbr itself)."""
    try:
        from nba_api.stats.static import teams as _t
        return {t["abbreviation"]: t["full_name"] for t in _t.get_teams()}
    except Exception as e:
        print("warn: nba_api team names unavailable:", e)
        return {}


def main() -> None:
    season = utils.SEASONS[0]
    df = utils.build_ranked_projected(season)
    names = _team_names()

    teams = {}
    for abbr, g in df.groupby("Team"):
        abbr = str(abbr)
        if not abbr or abbr in ("nan", "None"):
            continue
        g = g.sort_values("value_diff")  # most underpaid first
        players = []
        for r, (_, row) in enumerate(g.iterrows(), start=1):
            vd = float(row["value_diff"])  # salary - market; >0 overpaid
            players.append({
                "n": str(row["Player"]),
                "s": round(float(row["barrett_score"]), 1),
                "sal": round(float(row["salary"]) / 1e6, 1),
                "val": round(float(row["projected_salary"]) / 1e6, 1),
                "vd": round(vd / 1e6, 1),
                "r": int(row["score_rank"]),  # league-wide value rank
            })
        tot = {
            "n": len(players),
            "sal": round(float(g["salary"].sum()) / 1e6, 1),
            "val": round(float(g["projected_salary"].sum()) / 1e6, 1),
        }
        teams[abbr] = {"abbr": abbr, "name": names.get(abbr, abbr),
                       "players": players, "tot": tot}

    payload = {"season": season, "teams": dict(sorted(teams.items()))}
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT.name}  ({len(teams)} teams, {OUT.stat().st_size // 1024}KB)")
    for probe in ("LAL", "DEN", "OKC"):
        t = teams.get(probe)
        if t:
            top = t["players"][0]
            print(f"  {probe} {t['name']}: {t['tot']['n']} players, "
                  f"best value {top['n']} ({top['s']}, {top['vd']:+}M)")


if __name__ == "__main__":
    main()
