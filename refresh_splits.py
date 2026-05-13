"""
Run this script to pre-fetch splits data for all seasons and save to disk.
The Streamlit app reads from these files instantly — no API calls at load time.

Usage:
    python refresh_splits.py              # refresh all seasons
    python refresh_splits.py 2024-25      # refresh one season

Availability multiplier (Option C):
  For traded players, the denominator is the number of games their team actually
  played DURING their stint, not 82. Computed from PlayerGameLog + TeamGameLog.
  For non-traded players, denominator is 82 as normal.
"""
import re
import sys
import time
import math
import unicodedata
import io
import requests
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    playergamelogs,
    teamgamelogs,
)
from nba_api.stats.static import teams as nba_teams_static

SEASONS  = ["2025-26", "2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20",
            "2018-19", "2017-18", "2016-17", "2015-16", "2014-15", "2013-14", "2012-13",
            "2011-12", "2010-11", "2009-10"]
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def season_to_espn_year(season: str) -> int:
    return int(season.split("-")[0]) + 1


def api_call(fn, *args, **kwargs):
    time.sleep(0.8)
    return fn(*args, **kwargs)


def base_score(row) -> float:
    d_lebron = row["d_lebron"] if "d_lebron" in row.index else 0
    return (
        row["PTS"] + row["AST"] * 2 + row["OREB"] / 2 + row["DREB"] / 3
        + row["BLK"] / 2 + row["STL"] / 1.5 - row["TOV"] / 1.5 - row["PF"] / 3
        + d_lebron
    )


def availability_multiplier(gp: float, team_games: float, total_min: float) -> float:
    """Option C: denominator is team games played during stint, not always 82."""
    return 0.75 + 0.25 * math.sqrt((gp / team_games) * min(total_min / 2500, 1))


# ── D-LEBRON fetch ─────────────────────────────────────────────────────────────

def fetch_dlebron(season: str) -> dict:
    """Returns {player_id (int): d_lebron (float)} for the given season."""
    try:
        r = requests.post(
            "https://fanspo.com/bbi-role-explorer/api/lebron_dashboard_data",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.bball-index.com/",
                "Content-Type": "application/json",
            },
            json={},
            timeout=15,
        )
        players = r.json()["players"]
        df = pd.DataFrame(players)
        season_df = df[df["Season"] == season][["nba_id", "D-LEBRON"]].dropna()
        return {int(row["nba_id"]): float(row["D-LEBRON"]) for _, row in season_df.iterrows()}
    except Exception as e:
        print(f"  WARN: D-LEBRON fetch failed: {e}")
        return {}


# ── Salary fetch ───────────────────────────────────────────────────────────────

def fetch_salaries(espn_year: int) -> dict:
    sal = {}
    for page in range(1, 15):
        url = f"https://www.espn.com/nba/salaries/_/year/{espn_year}/page/{page}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            break
        df = pd.read_html(io.StringIO(str(tables[0])))[0]
        data = df[df[0] != "RK"]
        if data.empty:
            break
        for _, row in data.iterrows():
            name = re.sub(r",\s*\w+$", "", str(row[1])).strip()
            salary_str = str(row[3]).replace("$", "").replace(",", "")
            try:
                sal[normalize(name)] = float(salary_str)
            except ValueError:
                pass
        time.sleep(0.3)
    return sal


# ── Team game log cache (one fetch per team per season) ────────────────────────

def fetch_all_team_gamelogs(season: str) -> dict:
    """Returns {team_abbrev: DataFrame of game dates sorted ascending}."""
    print("  Fetching team game logs (30 calls)...")
    result = {}
    for team in nba_teams_static.get_teams():
        try:
            ep = api_call(
                teamgamelogs.TeamGameLogs,
                team_id_nullable=team["id"],
                season_nullable=season,
            )
            df = ep.get_data_frames()[0]
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            result[team["abbreviation"]] = df.sort_values("GAME_DATE").reset_index(drop=True)
        except Exception as e:
            print(f"    WARN: team log failed for {team['abbreviation']}: {e}")
    print(f"  Got logs for {len(result)} teams")
    return result


def team_games_in_range(team_log: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    mask = (team_log["GAME_DATE"] >= start) & (team_log["GAME_DATE"] <= end)
    return max(int(mask.sum()), 1)


# ── Player game log → stint date ranges ───────────────────────────────────────

def fetch_stint_team_games(player_id: int, season: str, team_abbrevs: list,
                            team_logs: dict) -> dict:
    """
    Returns {team_abbrev: team_games_played_during_stint}.
    Falls back to GP if game log unavailable.
    """
    try:
        ep = api_call(
            playergamelogs.PlayerGameLogs,
            player_id_nullable=player_id,
            season_nullable=season,
        )
        gl = ep.get_data_frames()[0]
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    except Exception:
        return {t: 82 for t in team_abbrevs}

    result = {}
    for abbrev in team_abbrevs:
        stint_games = gl[gl["TEAM_ABBREVIATION"] == abbrev]
        if stint_games.empty or abbrev not in team_logs:
            result[abbrev] = 82
            continue
        start = stint_games["GAME_DATE"].min()
        end   = stint_games["GAME_DATE"].max()
        result[abbrev] = team_games_in_range(team_logs[abbrev], start, end)

    return result


# ── Main build ─────────────────────────────────────────────────────────────────

def build_splits(season: str) -> pd.DataFrame:
    print(f"\n{'='*50}")
    print(f"Building splits for {season}")
    print(f"{'='*50}")

    print("  Fetching salaries from ESPN...")
    sal_dict = fetch_salaries(season_to_espn_year(season))
    print(f"  Got {len(sal_dict)} salaries")

    # Step 1: one call per team for player stats
    TOTALS_COLS = ["GP", "MIN", "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF", "FGA", "FTA"]
    all_rows = []
    teams = nba_teams_static.get_teams()

    print(f"  Fetching player stats per team ({len(teams)} calls)...")
    for i, team in enumerate(teams):
        print(f"  [{i+1:2d}/{len(teams)}] {team['abbreviation']}...", end=" ", flush=True)
        try:
            ep = api_call(
                leaguedashplayerstats.LeagueDashPlayerStats,
                season=season,
                per_mode_detailed="Totals",
                team_id_nullable=team["id"],
            )
            df = ep.get_data_frames()[0].copy()
            df["TEAM_ABBREVIATION"] = team["abbreviation"]
            all_rows.append(df)
            print(f"{len(df)} players")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    combined["salary"] = combined["PLAYER_NAME"].apply(lambda n: sal_dict.get(normalize(n)))
    combined = combined.dropna(subset=["salary"])

    # Step 2: identify traded players
    player_counts = combined.groupby("PLAYER_ID").size()
    traded_ids = set(player_counts[player_counts > 1].index)
    print(f"\n  Traded players: {len(traded_ids)}")

    # Step 3: fetch team game logs (for Option C denominators)
    team_logs = fetch_all_team_gamelogs(season)

    # Step 4: for each traded player fetch game log → get stint team game counts
    print(f"  Fetching player game logs for traded players ({len(traded_ids)} calls)...")
    stint_team_games: dict[int, dict[str, int]] = {}  # {player_id: {team_abbrev: games}}

    for i, pid in enumerate(traded_ids):
        prows = combined[combined["PLAYER_ID"] == pid]
        abbrevs = prows["TEAM_ABBREVIATION"].tolist()
        name = prows.iloc[0]["PLAYER_NAME"]
        print(f"  [{i+1:2d}/{len(traded_ids)}] {name}...", end=" ", flush=True)
        stg = fetch_stint_team_games(int(pid), season, abbrevs, team_logs)
        stint_team_games[pid] = stg
        print(" | ".join(f"{a}:{stg.get(a,'?')}" for a in abbrevs))

    # Step 5: add TOT rows for traded players
    tot_rows = []
    for pid in traded_ids:
        prows = combined[combined["PLAYER_ID"] == pid]
        tot = prows.iloc[0].copy()
        for col in TOTALS_COLS:
            if col in prows.columns:
                tot[col] = prows[col].sum()
        tot["TEAM_ABBREVIATION"] = "TOT"
        tot_rows.append(tot)

    if tot_rows:
        combined = pd.concat([combined, pd.DataFrame(tot_rows)], ignore_index=True)

    # Step 6: merge D-LEBRON (full-season metric, same value for all stints)
    print("  Fetching D-LEBRON from bball-index...")
    dlebron_dict = fetch_dlebron(season)
    print(f"  Got D-LEBRON for {len(dlebron_dict)} players")
    combined["d_lebron"] = combined["PLAYER_ID"].map(dlebron_dict).fillna(0)

    # Step 7: totals → per-game, then Barrett Score with correct denominator
    for col in ["MIN", "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF", "FGA", "FTA"]:
        combined[col] = combined[col] / combined["GP"]

    # TS% efficiency adjustment
    combined["ts_pct"] = combined["PTS"] / (2 * (combined["FGA"] + 0.44 * combined["FTA"])).replace(0, float("nan"))
    league_avg_ts = (combined["ts_pct"] * combined["GP"]).sum() / combined["GP"].sum()
    K_EFF = 0.15
    MIN_FGA = 2.0
    def eff_adj(row):
        if row["FGA"] < MIN_FGA or pd.isna(row["ts_pct"]):
            return 0.0
        return float(min(max(K_EFF * (row["ts_pct"] - league_avg_ts) * 100, -2), 2))
    combined["efficiency_adj"] = combined.apply(eff_adj, axis=1)

    combined["total_min"] = (combined["MIN"] * combined["GP"]).round(0).astype(int)
    combined["base_score"] = combined.apply(base_score, axis=1) + combined["efficiency_adj"]

    def get_avail_mult(row):
        pid  = row["PLAYER_ID"]
        team = row["TEAM_ABBREVIATION"]
        gp   = row["GP"]
        tmin = row["total_min"]
        if pid in traded_ids and team != "TOT":
            # Option C: use actual team games played during stint
            denom = stint_team_games.get(pid, {}).get(team, 82)
        else:
            # Non-traded or TOT row: use 82
            denom = 82
        return availability_multiplier(gp, denom, tmin)

    combined["avail_mult"]    = combined.apply(get_avail_mult, axis=1)
    combined["barrett_score"] = combined["base_score"] * combined["avail_mult"]

    # Step 8: drop duplicate non-traded rows, keep all traded rows
    non_traded = combined[~combined["PLAYER_ID"].isin(traded_ids)].drop_duplicates(subset=["PLAYER_ID"])
    traded     = combined[combined["PLAYER_ID"].isin(traded_ids)]
    combined   = pd.concat([non_traded, traded], ignore_index=True)

    combined = combined.rename(columns={
        "PLAYER_NAME": "Player", "TEAM_ABBREVIATION": "Team", "MIN": "MPG"
    })
    combined["score_rank"]  = combined["barrett_score"].rank(ascending=False, method="min").astype(int)
    combined["salary_rank"] = combined["salary"].rank(ascending=False, method="min").astype(int)
    combined["rank_diff"]   = combined["salary_rank"] - combined["score_rank"]

    return combined[["Player", "Team", "GP", "MPG", "total_min",
                      "base_score", "avail_mult", "barrett_score",
                      "salary", "score_rank", "salary_rank", "rank_diff"]]


def cache_path(season: str) -> Path:
    return CACHE_DIR / f"splits_{season.replace('-', '_')}.pkl"


def refresh(season: str):
    df = build_splits(season)
    if df.empty:
        print(f"  No data for {season}, skipping save.")
        return
    path = cache_path(season)
    with open(path, "wb") as f:
        pickle.dump({"data": df, "fetched_at": datetime.now()}, f)
    print(f"\n  Saved {len(df)} rows → {path}")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else SEASONS
    for s in targets:
        refresh(s)
    print("\nDone.")
