import math
import time
import io
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import unicodedata
from pathlib import Path
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import leaguedashplayerstats, playercareerstats, playerindex
from nba_api.stats.static import players as nba_players_static
from nba_api.stats.static import teams as nba_teams_static

CACHE_DIR = Path(__file__).parent / "cache"

st.set_page_config(page_title="Barrett Score", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-left: 0.5rem; padding-right: 0.5rem; max-width: 100%; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    [data-testid="stAppViewerBadge"] { display: none !important; }
    [data-testid="stBottom"] { display: none !important; }
    .viewerBadge_container__r5tak { display: none !important; }
    .styles_viewerBadge__CvC9N { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.title("Barrett Score — NBA Contract Value Rankings")
st.caption("A stat-driven ranking of every NBA player's contract value — who's underpaid, who's overpaid, and who's available.")

with st.expander("How is this calculated?"):
    st.markdown(
        "**Base Score** = PTS + AST×2 + OREB÷2 + DREB÷3 + BLK÷2 + STL÷1.5 − TOV÷1.5 − PF÷3 + D-LEBRON×2 + Eff. Adj×2  *(per game)*\n\n"
        "**Eff. Adj** = clamp(0.15 × (TS% − Lg Avg TS%) × 100, −4, +4)\n\n"
        "**Barrett Score** = Base Score × (0.75 + 0.25 × √((GP/82) × min(MIN/2500, 1)))\n\n"
        "*The availability multiplier scales down players who have missed significant time, rewarding durability.*"
    )

SEASONS = ["2025-26", "2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20",
           "2018-19", "2017-18", "2016-17", "2015-16", "2014-15", "2013-14", "2012-13",
           "2011-12", "2010-11", "2009-10", "2008-09", "2007-08", "2006-07"]
DEFAULT_MIN_THRESHOLD = 500

# Actual games played per season (shortened seasons due to lockout/COVID)
SEASON_GAMES_LOOKUP = {
    "2020-21": 72, "2019-20": 72, "2011-12": 66,
}


def season_to_espn_year(season: str) -> int:
    return int(season.split("-")[0]) + 1


# ── Data fetching ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Fetching league stats...")
def fetch_league_stats(season: str) -> pd.DataFrame:
    time.sleep(0.5)
    result = None
    delay = 1
    while result is None:
        try:
            result = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
            )
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 30)
    return result.get_data_frames()[0]


@st.cache_data(ttl=86400)
def fetch_league_avg_ts(season: str) -> float:
    """Weighted league-average TS% for a given season."""
    try:
        stats = fetch_league_stats(season)
        ts = stats["PTS"] / (2 * (stats["FGA"] + 0.44 * stats["FTA"])).replace(0, float("nan"))
        return float((ts * stats["GP"]).sum() / stats["GP"].sum())
    except Exception:
        return 0.570


@st.cache_data(ttl=3600, show_spinner="Fetching salary data...")
def fetch_salaries(espn_year: int) -> pd.DataFrame:
    rows = []
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
        rows.append(data)
    combined = pd.concat(rows, ignore_index=True)
    combined.columns = ["rank", "name_pos", "team", "salary"]
    combined["name"] = combined["name_pos"].str.replace(r",\s*\w+$", "", regex=True).str.strip()
    combined["salary"] = combined["salary"].str.replace(r"[\$,]", "", regex=True).astype(float)
    return combined[["name", "salary"]]


@st.cache_data(ttl=3600, show_spinner="Fetching player splits...")
def fetch_player_season_splits(player_id: int, season: str,
                                d_lebron_val: float = 0.0,
                                league_avg_ts: float = 0.57,
                                season_games: int = 82) -> pd.DataFrame:
    return _player_season_splits_raw(player_id, season, d_lebron_val, league_avg_ts, season_games)


@st.cache_data(ttl=3600, show_spinner="Fetching career trend...")
def fetch_career_trend(player_id: int, num_seasons: int = 5) -> pd.DataFrame:
    """Barrett Score for each of the player's last N seasons, with real D-LEBRON."""

    # Full D-LEBRON table — one call, all seasons, already cached
    dlebron_all = fetch_dlebron_all()
    player_dlebron = {}
    if not dlebron_all.empty:
        pdf = dlebron_all[dlebron_all["nba_id"].astype(str) == str(player_id)]
        player_dlebron = {row["Season"]: float(row["D-LEBRON"]) for _, row in pdf.iterrows()}

    career = None
    delay = 1
    while career is None:
        try:
            career = playercareerstats.PlayerCareerStats(player_id=player_id)
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 30)

    df = career.get_data_frames()[0]

    # One row per season: prefer TOT row for traded players
    rows_out = []
    for season_id, grp in df.groupby("SEASON_ID"):
        if not str(season_id)[0:2] in ("19", "20"):
            continue
        tot = grp[grp["TEAM_ABBREVIATION"] == "TOT"]
        row = tot.iloc[0].copy() if not tot.empty else grp.iloc[0].copy()
        if len(grp) > 1 and tot.empty:
            for col in ["GP", "MIN", "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF", "FGA", "FTA"]:
                if col in grp.columns:
                    row[col] = grp[col].sum()
        rows_out.append(row)

    if not rows_out:
        return pd.DataFrame()

    cdf = pd.DataFrame(rows_out).sort_values("SEASON_ID", ascending=False).head(num_seasons)

    for col in ["MIN", "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF", "FGA", "FTA"]:
        cdf[col] = cdf[col] / cdf["GP"]

    cdf["d_lebron"] = cdf["SEASON_ID"].map(lambda s: player_dlebron.get(s, 0.0))
    cdf["ts_pct"] = cdf["PTS"] / (2 * (cdf["FGA"] + 0.44 * cdf["FTA"])).replace(0, float("nan"))
    cdf["league_avg_ts"] = cdf["SEASON_ID"].map(fetch_league_avg_ts)
    cdf["efficiency_adj"] = cdf.apply(
        lambda r: float(min(max(0.15 * (r["ts_pct"] - r["league_avg_ts"]) * 100, -4), 4))
        if r.get("FGA", 0) >= 2.0 and not pd.isna(r["ts_pct"]) else 0.0, axis=1
    )
    cdf["total_min"] = (cdf["MIN"] * cdf["GP"]).round(0).astype(int)
    cdf["base_score"] = cdf.apply(base_score, axis=1) + cdf["efficiency_adj"] * 2
    cdf["sg"] = cdf["SEASON_ID"].map(lambda s: SEASON_GAMES_LOOKUP.get(s, 82))
    cdf["avail_mult"] = cdf.apply(
        lambda r: availability_multiplier(r["GP"], r["total_min"], int(r["sg"])), axis=1
    )
    cdf["barrett_score"] = cdf["base_score"] * cdf["avail_mult"]

    return cdf[["SEASON_ID", "GP", "MIN", "total_min",
                "base_score", "avail_mult", "barrett_score"]].rename(
        columns={"SEASON_ID": "Season", "MIN": "MPG"}
    ).sort_values("Season")


def _player_season_splits_raw(player_id: int, season: str,
                               d_lebron_val: float = 0.0,
                               league_avg_ts: float = 0.57,
                               season_games: int = 82) -> pd.DataFrame:
    career = None
    delay = 1
    while career is None:
        try:
            career = playercareerstats.PlayerCareerStats(player_id=player_id)
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 30)
    df = career.get_data_frames()[0]
    rows = df[df["SEASON_ID"] == season].copy()
    if rows.empty:
        return pd.DataFrame()

    for col in ["PTS", "AST", "OREB", "DREB", "REB", "BLK", "STL", "TOV", "PF", "MIN", "FGA", "FTA"]:
        rows[col] = rows[col] / rows["GP"]

    rows["d_lebron"] = d_lebron_val
    rows["ts_pct"] = rows["PTS"] / (2 * (rows["FGA"] + 0.44 * rows["FTA"])).replace(0, float("nan"))
    rows["efficiency_adj"] = rows.apply(
        lambda r: float(min(max(0.15 * (r["ts_pct"] - league_avg_ts) * 100, -4), 4))
        if r["FGA"] >= 2.0 and not pd.isna(r["ts_pct"]) else 0.0, axis=1
    )

    rows["total_min"] = (rows["MIN"] * rows["GP"]).round(0).astype(int)
    rows["base_score"] = rows.apply(base_score, axis=1) + rows["efficiency_adj"] * 2
    rows["avail_mult"] = rows.apply(
        lambda r: availability_multiplier(r["GP"], r["total_min"], season_games), axis=1
    )
    rows["barrett_score"] = rows["base_score"] * rows["avail_mult"]

    return rows[["TEAM_ABBREVIATION", "GP", "MIN", "total_min",
                 "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF",
                 "d_lebron", "ts_pct", "efficiency_adj",
                 "base_score", "avail_mult", "barrett_score"]].rename(columns={
        "TEAM_ABBREVIATION": "Team",
        "MIN": "MPG",
    })


@st.cache_data(ttl=3600, show_spinner="Building splits table — loading once, fast for everyone after…")
def build_splits_data_live(season: str, salary_lookup: tuple) -> pd.DataFrame:
    # salary_lookup: ((normalized_name, salary), ...) — hashable for cache key
    sal_dict = dict(salary_lookup)

    TOTALS_COLS = ["GP", "MIN", "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF"]

    def _fetch_team(team: dict):
        """Fetch one team's player stats — retries with backoff on failure."""
        delay = 1
        while True:
            try:
                ep = leaguedashplayerstats.LeagueDashPlayerStats(
                    season=season,
                    per_mode_detailed="Totals",
                    team_id_nullable=team["id"],
                )
                df = ep.get_data_frames()[0]
                if df.empty:
                    return None
                df = df.copy()
                df["TEAM_ABBREVIATION"] = team["abbreviation"]
                return df
            except Exception:
                time.sleep(delay)
                delay = min(delay * 2, 30)

    teams = nba_teams_static.get_teams()
    all_rows = []
    # 6 workers → ~5× faster than sequential; stays within NBA API rate limits
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_team, t): t for t in teams}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_rows.append(result)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)

    # Match salaries
    combined["salary"] = combined["PLAYER_NAME"].apply(
        lambda n: sal_dict.get(normalize(n))
    )
    combined = combined.dropna(subset=["salary"])

    # Capture season game count before TOT rows are added (TOT GP = sum of stints,
    # which would inflate the max). Max GP of any real player row = games played this season.
    season_games = int(combined["GP"].max())

    # For traded players (appear in 2+ team rows), add a TOT row
    player_counts = combined.groupby("PLAYER_ID").size()
    traded_ids = set(player_counts[player_counts > 1].index)

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

    # Convert totals → per-game
    pg_cols = ["MIN", "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF"]
    for col in pg_cols:
        combined[col] = combined[col] / combined["GP"]

    combined["total_min"]     = (combined["MIN"] * combined["GP"]).round(0).astype(int)
    combined["base_score"]    = combined.apply(base_score, axis=1)
    combined["avail_mult"]    = combined.apply(
        lambda r: availability_multiplier(r["GP"], r["total_min"], season_games), axis=1
    )
    combined["barrett_score"] = combined["base_score"] * combined["avail_mult"]

    # For non-traded players, drop the redundant duplicate (they appear once per team call, correctly)
    non_traded_mask = ~combined["PLAYER_ID"].isin(traded_ids)
    non_traded = combined[non_traded_mask].drop_duplicates(subset=["PLAYER_ID"])
    traded     = combined[combined["PLAYER_ID"].isin(traded_ids)]
    combined   = pd.concat([non_traded, traded], ignore_index=True)

    combined = combined.rename(columns={"PLAYER_NAME": "Player", "TEAM_ABBREVIATION": "Team", "MIN": "MPG"})
    combined["score_rank"]  = combined["barrett_score"].rank(ascending=False, method="min").astype(int)
    combined["salary_rank"] = combined["salary"].rank(ascending=False, method="min").astype(int)
    combined["rank_diff"]   = combined["salary_rank"] - combined["score_rank"]

    return combined[["Player", "Team", "GP", "MPG", "total_min",
                      "base_score", "avail_mult", "barrett_score",
                      "salary", "score_rank", "salary_rank", "rank_diff"]]


@st.cache_data(ttl=3600)
def load_splits_from_disk(season: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"splits_{season.replace('-', '_')}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        cached = pickle.load(f)
    return cached["data"]


def build_splits_data(season: str, salary_lookup: tuple) -> pd.DataFrame:
    disk = load_splits_from_disk(season)
    if disk is not None:
        return disk
    return build_splits_data_live(season, salary_lookup)


@st.cache_data(ttl=86400)
def get_player_id_map() -> dict:
    return {p["full_name"]: p["id"] for p in nba_players_static.get_players()}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_bref_positions(espn_year: int, cache_v: int = 3) -> dict:
    """
    Returns {normalized_name: "Guard"|"Forward"|"Center"} from ESPN salary pages.
    ESPN uses exactly G / F / C in the 'Player Name, X' cell — simple and reliable.
    """
    _pos_map = {"G": "Guard", "F": "Forward", "C": "Center"}
    result: dict = {}
    try:
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
            data.columns = ["rank", "name_pos", "team", "salary"]
            for name_pos in data["name_pos"]:
                s = str(name_pos)
                if ", " not in s:
                    continue
                name, pos = s.rsplit(", ", 1)
                mapped = _pos_map.get(pos.strip().upper())
                if mapped:
                    result[normalize(name.strip())] = mapped
    except Exception:
        pass
    return result



@st.cache_data(ttl=86400, show_spinner=False)
def fetch_rookie_scale_players(season: str) -> set:
    """
    Returns a set of normalized names for players on rookie scale contracts.
    Rookie scale = first-round draft picks in their first 4 NBA seasons.
    e.g. for 2025-26, that's first-round picks drafted 2022–2025.
    """
    try:
        end_year = int(season.split("-")[0]) + 1          # e.g. 2025-26 → 2026
        rookie_draft_years = set(range(end_year - 3, end_year + 1))  # 2023–2026

        idx = playerindex.PlayerIndex(season=season)
        df_idx = idx.get_data_frames()[0]

        rookies: set = set()
        for _, row in df_idx.iterrows():
            try:
                draft_year  = int(row["DRAFT_YEAR"])
                draft_round = int(row["DRAFT_ROUND"])
            except (ValueError, TypeError):
                continue
            if draft_year in rookie_draft_years and draft_round == 1:
                full_name = f"{row['PLAYER_FIRST_NAME']} {row['PLAYER_LAST_NAME']}".strip()
                rookies.add(normalize(full_name))
        return rookies
    except Exception:
        return set()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_next_year_contracts(espn_year: int, cache_v: int = 7) -> dict:
    """
    Returns {normalized_name: {"salary": float, "type": str}} for next season (espn_year + 1).
    type is one of: "guaranteed", "team_option", "player_option".
    Players not in the dict have no deal next year → displayed as "—".

    Dollar amounts : ESPN next-year salary page (reliable).
    Option type    : Spotrac free-agents page, second table.
                     Type column encodes "PLAYER / $X" or "TEAM / $X" for options.
    """
    next_year = espn_year + 1
    contracts: dict = {}
    _hdrs = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    # Step 1: ESPN next year for dollar amounts
    try:
        sal_df = fetch_salaries(next_year)
        for _, row in sal_df.iterrows():
            contracts[normalize(row["name"])] = {"salary": row["salary"], "type": "guaranteed"}
    except Exception:
        pass

    # Step 2: Spotrac free-agents page — second table has Type column.
    # Confirmed structure: Player | Pos | Age | YOE | Prev Team | Prev AAV | Type
    # Type values: "PLAYER / $49.0M"  → player option
    #              "TEAM / $18.0M"    → team option
    #              "UFA / Bird"       → expiring guaranteed (no option)
    try:
        r = requests.get(
            "https://www.spotrac.com/nba/free-agents/",
            headers=_hdrs, timeout=15,
        )
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            tables = soup.find_all("table")
            # Second table (index 1) is the free-agents list with the Type column
            fa_table = tables[1] if len(tables) >= 2 else (tables[0] if tables else None)
            if fa_table is not None:
                tdf = pd.read_html(io.StringIO(str(fa_table)))[0]
                cols_lower = {str(c).lower().strip(): c for c in tdf.columns}
                name_col  = tdf.columns[0]
                type_col  = cols_lower.get("type")
                if type_col is not None:
                    for _, row in tdf.iterrows():
                        norm = normalize(str(row[name_col]))
                        if norm not in contracts:
                            continue
                        t = str(row[type_col]).upper().strip()
                        if t.startswith("PLAYER"):
                            contracts[norm]["type"] = "player_option"
                        elif t.startswith("CLUB"):   # Spotrac labels team options as "CLUB / $X"
                            contracts[norm]["type"] = "team_option"
    except Exception:
        pass

    return contracts


# ── Score calculation ──────────────────────────────────────────────────────────

def base_score(row) -> float:
    d_lebron = row["d_lebron"] if "d_lebron" in row.index else 0
    return (
        row["PTS"]
        + row["AST"] * 2
        + row["OREB"] / 2
        + row["DREB"] / 3
        + row["BLK"] / 2
        + row["STL"] / 1.5
        - row["TOV"] / 1.5
        - row["PF"] / 3
        + d_lebron * 2
    )


@st.cache_data(ttl=3600, show_spinner="Fetching D-LEBRON defensive ratings...")
def fetch_dlebron_all() -> pd.DataFrame:
    """Fetches all D-LEBRON data in one call — every player, every season back to 2009-10."""
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
        df = pd.DataFrame(r.json()["players"])
        return df[["nba_id", "Season", "D-LEBRON"]].dropna()
    except Exception:
        return pd.DataFrame(columns=["nba_id", "Season", "D-LEBRON"])


def fetch_dlebron(season: str) -> dict:
    """Returns {player_id (int): d_lebron (float)} for a single season."""
    df = fetch_dlebron_all()
    if df.empty:
        return {}
    season_df = df[df["Season"] == season]
    return {int(row["nba_id"]): float(row["D-LEBRON"]) for _, row in season_df.iterrows()}


def availability_multiplier(gp: float, total_min: float, season_games: int = 82) -> float:
    # Scale the minute cap to actual season length so a 66-game lockout season
    # isn't penalized vs a full 82-game season.
    min_cap = season_games * (2500 / 82)
    return 0.75 + 0.25 * math.sqrt((gp / season_games) * min(total_min / min_cap, 1))


# ── Name matching ──────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


# ── Build raw data ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Building rankings...")
def build_raw(season: str) -> pd.DataFrame:
    stats = fetch_league_stats(season).copy()
    salaries = fetch_salaries(season_to_espn_year(season))

    sal_lookup = {normalize(n): s for n, s in zip(salaries["name"], salaries["salary"])}
    stats["salary"] = stats["PLAYER_NAME"].apply(lambda n: sal_lookup.get(normalize(n)))
    stats = stats.dropna(subset=["salary"])

    dlebron = fetch_dlebron(season)
    stats["d_lebron"] = stats["PLAYER_ID"].map(dlebron).fillna(0)

    # TS% efficiency adjustment — rewards efficient scorers, penalises inefficient ones
    stats["ts_pct"] = stats["PTS"] / (2 * (stats["FGA"] + 0.44 * stats["FTA"])).replace(0, float("nan"))
    league_avg_ts = (stats["ts_pct"] * stats["GP"]).sum() / stats["GP"].sum()
    K_EFF = 0.15
    MIN_FGA = 2.0
    def eff_adj(row):
        if row["FGA"] < MIN_FGA or pd.isna(row["ts_pct"]):
            return 0.0
        return float(min(max(K_EFF * (row["ts_pct"] - league_avg_ts) * 100, -4), 4))
    stats["efficiency_adj"] = stats.apply(eff_adj, axis=1)

    # Use actual games played this season as denominator — handles lockout/COVID seasons.
    season_games = int(stats["GP"].max())

    stats["total_min"] = (stats["MIN"] * stats["GP"]).round(0).astype(int)
    stats["base_score"] = stats.apply(base_score, axis=1) + stats["efficiency_adj"] * 2
    stats["avail_mult"] = stats.apply(
        lambda r: availability_multiplier(r["GP"], r["total_min"], season_games), axis=1
    )
    stats["barrett_score"] = stats["base_score"] * stats["avail_mult"]

    return stats[[
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "total_min",
        "base_score", "avail_mult", "barrett_score", "salary",
        "d_lebron", "ts_pct", "efficiency_adj",
    ]].rename(columns={
        "PLAYER_NAME": "Player",
        "TEAM_ABBREVIATION": "Team",
        "MIN": "MPG",
    })


def apply_rankings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score_rank"]  = df["barrett_score"].rank(ascending=False, method="min").astype(int)
    df["salary_rank"] = df["salary"].rank(ascending=False, method="min").astype(int)
    df["rank_diff"]   = df["salary_rank"] - df["score_rank"]
    return df


def apply_projections(df: pd.DataFrame) -> pd.DataFrame:
    """Projected salary = the actual salary of whoever holds the same rank position by salary.
    Rank 1 by Barrett Score → deserves the rank-1 salary (highest paid player's contract), etc."""
    df = df.copy()
    salaries_by_rank = df.sort_values("salary", ascending=False)["salary"].values
    n = len(salaries_by_rank)
    df["projected_salary"] = df["score_rank"].apply(
        lambda r: float(salaries_by_rank[min(int(r) - 1, n - 1)])
    )
    df["value_diff"] = df["salary"] - df["projected_salary"]  # positive = overpaid
    return df


# ── UI ─────────────────────────────────────────────────────────────────────────

ctrl_l, ctrl_mid, ctrl_r = st.columns([1, 1, 1])
with ctrl_l:
    season = st.selectbox("Season", SEASONS, index=0)
with ctrl_r:
    min_threshold = st.slider(
        "Min total minutes", min_value=0, max_value=1500,
        value=DEFAULT_MIN_THRESHOLD, step=50,
        help="Hides players below this threshold. Ranks are always computed on the full pool.",
    )

raw = build_raw(season)
df = apply_rankings(raw)
df = apply_projections(df)
df = df[df["total_min"] >= min_threshold]

# salary_lookup is needed by the splits tab — computed once here (cheap)
salary_lookup = tuple(
    (normalize(row["Player"]), row["salary"])
    for _, row in raw.iterrows()
)

# Positions from ESPN (PG/SG → Guard, SF/PF → Forward, C → Center)
# Much more reliable than the NBA API's PlayerIndex which mislabels wing players.
_bref_positions = fetch_bref_positions(season_to_espn_year(season), cache_v=3)
df["position"] = df["Player"].map(
    lambda n: _bref_positions.get(normalize(n), "")
)

_next_contracts = fetch_next_year_contracts(season_to_espn_year(season), cache_v=7)
_rookie_scale   = fetch_rookie_scale_players(season)

# ── Background cache warming ───────────────────────────────────────────────────
# After the current season loads, silently pre-warm the 3 most recent other
# seasons so switching seasons is instant for all users.
def _warm_season(s: str) -> None:
    """Pre-populate @st.cache_data for rankings data for a given season.
    Splits are intentionally excluded — they're slow (30 API calls) and load
    lazily only when the user opens the Splits toggle."""
    try:
        espn_year = season_to_espn_year(s)
        build_raw(s)
        fetch_bref_positions(espn_year, cache_v=3)
        fetch_next_year_contracts(espn_year, cache_v=7)
        fetch_rookie_scale_players(s)
    except Exception:
        pass  # Never crash the main app from a background thread

_seasons_to_warm = [s for s in SEASONS if s != season][:3]
for _ws in _seasons_to_warm:
    threading.Thread(target=_warm_season, args=(_ws,), daemon=True).start()

def _fmt_salary(player_name: str, salary_dollars: float) -> str:
    """Format salary as '$X.XXM'. Rookie-scale players are colored purple via style, no text marker."""
    return f"${salary_dollars / 1_000_000:.2f}M"

def _fmt_next_contract(player_name: str) -> str:
    info = _next_contracts.get(normalize(player_name))
    if info is None:
        return "—"
    sal_m = info["salary"] / 1_000_000
    if info["type"] == "team_option":
        return f"${sal_m:.1f}M TO"
    if info["type"] == "player_option":
        return f"${sal_m:.1f}M PO"
    return f"${sal_m:.1f}M"

df["next_contract"] = df["Player"].apply(_fmt_next_contract)

# MPG and season_games come directly from the PerGame stats in df (fast, no splits needed).
# The splits tab loads its own data lazily when the user opens it.
season_games = int(raw["GP"].max())
splits_mpg_lookup = df.set_index("Player")["MPG"]

st.caption(
    f"**{len(df)}** players ranked · "
    f"**{(df['value_diff'] < -5_000_000).sum()}** underpaid (earning \$5M+ below projection) · "
    f"**{(df['value_diff'] > 5_000_000).sum()}** overpaid (earning \$5M+ above projection)"
)
st.divider()

tab_rankings, tab_projector, tab_teams, tab_fa = st.tabs(["Rankings", "Salary Projector", "Team Analysis", "Free Agent Class"])


def color_rank_diff(val):
    try:
        n = int(str(val).replace("+", ""))
    except ValueError:
        return ""
    if n > 20:   return "color: #2ecc71; font-weight: bold"
    if n > 0:    return "color: #a8e6a8"
    if n < -20:  return "color: #e74c3c; font-weight: bold"
    if n < 0:    return "color: #f1a8a8"
    return ""


def style_rookie_salary(row):
    """Color the Salary cell purple for rookie-scale players (row-wise so we can check Player name)."""
    result = pd.Series('', index=row.index)
    if 'Player' in row.index and normalize(str(row['Player'])) in _rookie_scale:
        if 'Salary' in row.index:
            result['Salary'] = 'color: #a855f7; font-weight: bold'
    return result


def color_next_contract(val):
    s = str(val)
    if s == "—":
        return "color: #555555"          # gray  — no deal / UFA
    if " TO" in s:
        return "color: #f39c12; font-weight: bold"   # orange — team option
    if " PO" in s:
        return "color: #3498db; font-weight: bold"   # blue   — player option
    return ""                            # white — fully guaranteed




# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Rankings
# ══════════════════════════════════════════════════════════════════════════════

with tab_rankings:
    player_id_map_full = {row["Player"]: int(row["PLAYER_ID"]) for _, row in df.iterrows()}
    dlebron_lookup = fetch_dlebron(season)
    league_avg_ts = float((raw["ts_pct"] * raw["GP"]).sum() / raw["GP"].sum())
    if not dlebron_lookup:
        st.info("⚠️ D-LEBRON data is not available for this season — defensive ratings are set to 0 for all players. "
                "Barrett Scores reflect only box-score statistics.")

    def render_splits_panel(player_name, season):
        if player_name not in player_id_map_full:
            return
        pid = player_id_map_full[player_name]
        d_leb = float(dlebron_lookup.get(pid, 0.0))
        splits = fetch_player_season_splits(pid, season, d_leb, league_avg_ts, season_games)
        if splits.empty:
            st.info("No per-team split data available.")
            return
        # Build interleaved stat + contribution rows
        rows_out = []
        row_styles = []
        for i, (_, r) in enumerate(splits.iterrows()):
            is_tot = r["Team"] == "TOT"
            ts_str = f"{r['ts_pct']*100:.1f}%" if not pd.isna(r["ts_pct"]) else "—"
            # Stat row
            rows_out.append({
                "#": i + 1, "Team": r["Team"],
                "GP": str(int(r["GP"])), "MPG": f"{r['MPG']:.2f}", "Total MIN": str(int(r["total_min"])),
                "PTS": f"{r['PTS']:.2f}", "AST": f"{r['AST']:.2f}",
                "OREB": f"{r['OREB']:.2f}", "DREB": f"{r['DREB']:.2f}",
                "BLK": f"{r['BLK']:.2f}", "STL": f"{r['STL']:.2f}",
                "TOV": f"{r['TOV']:.2f}", "PF": f"{r['PF']:.2f}",
                "D-LEBRON": f"{r['d_lebron']:.2f}", "TS%": ts_str,
                "Eff. Adj": f"{r['efficiency_adj']:.2f}",
                "Base Score": f"{r['base_score']:.2f}",
                "Avail ×": f"{r['avail_mult']:.3f}",
                "Barrett Score": f"{r['barrett_score']:.2f}",
            })
            row_styles.append("tot_stat" if is_tot else "stat")
            # Contribution row (weighted values)
            rows_out.append({
                "#": "", "Team": "↳ Value",
                "GP": "-", "MPG": "-", "Total MIN": "-",
                "PTS": f"{r['PTS']:.2f}",
                "AST": f"{r['AST'] * 2:.2f}",
                "OREB": f"{r['OREB'] / 2:.2f}",
                "DREB": f"{r['DREB'] / 3:.2f}",
                "BLK": f"{r['BLK'] / 2:.2f}",
                "STL": f"{r['STL'] / 1.5:.2f}",
                "TOV": f"{-(r['TOV'] / 1.5):.2f}",
                "PF": f"{-(r['PF'] / 3):.2f}",
                "D-LEBRON": f"{r['d_lebron'] * 2:.2f}", "TS%": "-",
                "Eff. Adj": f"{r['efficiency_adj'] * 2:.2f}",
                "Base Score": f"{r['base_score']:.2f}",
                "Avail ×": f"{r['avail_mult']:.3f}",
                "Barrett Score": f"{r['barrett_score']:.2f}",
            })
            row_styles.append("tot_contrib" if is_tot else "contrib")

        fmt = pd.DataFrame(rows_out)

        def style_rows(row):
            s = row_styles[row.name]
            if s == "tot_stat":
                return ["font-weight: bold; background-color: #2a2a2a"] * len(row)
            if s == "tot_contrib":
                return ["font-weight: bold; background-color: #222; color: #999; font-style: italic"] * len(row)
            if s == "contrib":
                return ["color: #888; font-style: italic; background-color: #1c1c1c"] * len(row)
            return [""] * len(row)

        st.dataframe(
            fmt.style.apply(style_rows, axis=1),
            column_config={
                "GP":           st.column_config.TextColumn(help="Games played with this team during the season."),
                "MPG":          st.column_config.TextColumn(help="Minutes per game with this team."),
                "Total MIN":    st.column_config.TextColumn(help="Total minutes played with this team."),
                "PTS":          st.column_config.NumberColumn(help="Points per game."),
                "AST":          st.column_config.NumberColumn(help="Assists per game."),
                "OREB":         st.column_config.NumberColumn(help="Offensive rebounds per game."),
                "DREB":         st.column_config.NumberColumn(help="Defensive rebounds per game."),
                "BLK":          st.column_config.NumberColumn(help="Blocks per game."),
                "STL":          st.column_config.NumberColumn(help="Steals per game."),
                "TOV":          st.column_config.NumberColumn(help="Turnovers per game."),
                "PF":           st.column_config.NumberColumn(help="Personal fouls per game."),
                "D-LEBRON":     st.column_config.NumberColumn(help="Defensive LEBRON — estimated points prevented per game vs average. Full-season metric, same across all stints."),
                "TS%":          st.column_config.TextColumn(help="True Shooting % — scoring efficiency across 2s, 3s, and free throws. PTS / (2 × (FGA + 0.44 × FTA)). League avg ~57%."),
                "Eff. Adj":     st.column_config.NumberColumn(help="Efficiency adjustment added to Base Score. clamp(0.15 × (TS% − League Avg TS%) × 100, −4, +4). Rewards efficient scorers, penalises inefficient ones."),
                "Base Score":   st.column_config.NumberColumn(help="PTS + AST×2 + OREB÷2 + DREB÷3 + BLK÷2 + STL÷1.5 − TOV÷1.5 − PF÷3 + D-LEBRON×2 + Eff. Adj. Raw per-game value before the availability multiplier."),
                "Avail ×":      st.column_config.NumberColumn(help="Availability multiplier (0.75–1.00). Rewards health and heavy minutes. 0.75 + 0.25 × √((GP/team games) × min(Total MIN/2500, 1))."),
                "Barrett Score":st.column_config.NumberColumn(help="Base Score × Availability Multiplier. The final contract value rating."),
            },
            use_container_width=True,
            hide_index=True,
        )

    # ── Compare players (multiselect) ─────────────────────────────────────────
    all_player_names = sorted(df["Player"].unique().tolist())
    compare_selected = st.multiselect(
        "Compare players",
        options=all_player_names,
        max_selections=10,
        placeholder="Select up to 10 players to compare…",
        key="rankings_multiselect",
    )
    st.session_state["rankings_selected"] = compare_selected

    panel_placeholder = st.empty()

    # ── Search / sort / team filters ──────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
    with col_a:
        search = st.text_input("Filter by player name", "")
    with col_b:
        view = st.selectbox("Sort by", [
            "Barrett Score (best first)", "Salary (highest first)",
            "Most Underpaid", "Most Overpaid",
        ])
    with col_c:
        team_options = ["All"] + sorted(df["Team"].unique().tolist())
        team_filter = st.selectbox("Team", team_options)
    with col_d:
        pos_options = ["All", "Guard", "Forward", "Center"]
        pos_filter = st.selectbox("Position", pos_options)

    display = df.copy()
    if search:
        display = display[display["Player"].str.contains(search, case=False)]
    if team_filter != "All":
        display = display[display["Team"] == team_filter]
    if pos_filter != "All":
        display = display[display["position"] == pos_filter]

    sort_map = {
        "Barrett Score (best first)": ("score_rank", True),
        "Salary (highest first)":     ("salary_rank", True),
        "Most Underpaid":             ("value_diff", True),
        "Most Overpaid":              ("value_diff", False),
    }
    sort_col, sort_asc = sort_map[view]
    display = display.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    tog_a, tog_b, tog_rest = st.columns([1, 1, 6])
    with tog_a:
        advanced = st.toggle("Advanced view", value=False)
    with tog_b:
        show_splits = st.toggle("Splits View", value=False,
                                help="Per-team stints ranked together. Team-switchers appear as separate rows.")

    def color_value_diff(val):
        try:
            n = float(val)
        except (ValueError, TypeError):
            return ""
        if n > 20:   return "color: #e74c3c; font-weight: bold"
        if n > 5:    return "color: #f1a8a8"
        if n < -20:  return "color: #2ecc71; font-weight: bold"
        if n < -5:   return "color: #a8e6a8"
        return ""

    if show_splits:
        splits_df = build_splits_data(season, salary_lookup)

    if show_splits and splits_df is not None:
        # ── Splits table ──────────────────────────────────────────────────────
        if search:
            matched = splits_df[splits_df["Player"].str.contains(search, case=False)]["Player"].unique()
            sdisplay = splits_df[splits_df["Player"].isin(matched)].copy()
        else:
            sdisplay = splits_df[splits_df["total_min"] >= min_threshold].copy()
        if team_filter != "All":
            team_players = sdisplay[sdisplay["Team"] == team_filter]["Player"].unique()
            sdisplay = sdisplay[
                (sdisplay["Team"] == team_filter) |
                ((sdisplay["Player"].isin(team_players)) & (sdisplay["Team"] == "TOT"))
            ]
        if pos_filter != "All":
            pos_players = set(df[df["position"] == pos_filter]["Player"])
            sdisplay = sdisplay[sdisplay["Player"].isin(pos_players)]

        # Per-stint GP/MPG always from splits_df (actual totals per team).
        # Scores: non-traded players and TOT rows use main df (matches rankings exactly).
        #         Individual stints of traded players use splits_df (genuine per-team perf).
        season_scores = df.set_index("Player")[["base_score", "avail_mult", "barrett_score"]]
        proj_lookup   = df.set_index("Player")[["salary", "projected_salary", "value_diff",
                                                "score_rank", "salary_rank", "rank_diff",
                                                "d_lebron", "ts_pct"]]
        traded_players = set(splits_df[splits_df["Team"] == "TOT"]["Player"])

        sdisplay = sdisplay[["Player", "Team", "GP", "total_min",
                              "base_score", "avail_mult", "barrett_score"]].copy()
        sdisplay["MPG"] = (sdisplay["total_min"] / sdisplay["GP"]).round(2)

        # Override scores with main df for non-traded players and TOT rows
        use_season = ~sdisplay["Player"].isin(traded_players) | (sdisplay["Team"] == "TOT")
        for col in ["base_score", "avail_mult", "barrett_score"]:
            sdisplay.loc[use_season, col] = sdisplay.loc[use_season, "Player"].map(season_scores[col])

        # For individual stints of traded players, recalculate avail_mult using
        # team_games (max GP on that team this season) as denominator so that
        # missed games within the stint are penalized. 2500-min cap stays at
        # full-season scale — partial stints naturally score lower on both axes.
        # games_possible ≈ max GP of any player on that team (best proxy for
        # "how many games did the team play while this player was on the roster")
        team_games = splits_df[splits_df["Team"] != "TOT"].groupby("Team")["GP"].max()
        MINS_PER_GAME_CAP = 2500 / 82  # ≈ 30.5 — full-season minute cap per game
        stint_mask = sdisplay["Player"].isin(traded_players) & (sdisplay["Team"] != "TOT")
        if stint_mask.any():
            sdisplay.loc[stint_mask, "avail_mult"] = sdisplay[stint_mask].apply(
                lambda r: 0.75 + 0.25 * math.sqrt(
                    min(r["total_min"] / (team_games.get(r["Team"], season_games) * MINS_PER_GAME_CAP), 1)
                ), axis=1
            )
            sdisplay.loc[stint_mask, "barrett_score"] = (
                sdisplay.loc[stint_mask, "base_score"] * sdisplay.loc[stint_mask, "avail_mult"]
            )

        sdisplay = sdisplay[["Player", "Team", "GP", "MPG",
                              "base_score", "avail_mult", "barrett_score"]].join(
                              proj_lookup, on="Player", how="left")

        # Sort AFTER join so value_diff/salary come from the authoritative main df.
        # Use actual values (not ranks) so per-stint rows sort by their real scores.
        s_sort_map = {
            "Barrett Score (best first)": ("barrett_score", False),
            "Salary (highest first)":     ("salary", False),
            "Most Underpaid":             ("value_diff", True),
            "Most Overpaid":              ("value_diff", False),
        }
        s_col, s_asc = s_sort_map[view]
        sdisplay = sdisplay.sort_values(s_col, ascending=s_asc).reset_index(drop=True)

        def highlight_tot_row(row):
            if row["Team"] == "TOT":
                return ["font-weight: bold; background-color: #1e2a1e"] * len(row)
            if row["Player"] in traded_players:
                return ["background-color: #1a1f2e; color: #a0b0d0"] * len(row)
            return [""] * len(row)

        # Next year contract for splits view
        _nc_lookup = df.set_index("Player")["next_contract"]
        sdisplay["Next $"] = sdisplay["Player"].map(_nc_lookup).fillna("—")

        if advanced:
            sfmt = sdisplay[["Player", "Team", "GP", "MPG",
                              "base_score", "avail_mult", "barrett_score",
                              "score_rank", "salary", "projected_salary", "value_diff",
                              "salary_rank", "rank_diff", "d_lebron", "ts_pct", "Next $"]].copy()
            _sfmt_sal_raw            = sfmt["salary"].values
            sfmt["salary"]           = [_fmt_salary(p, s) for p, s in zip(sfmt["Player"].values, _sfmt_sal_raw)]
            sfmt["projected_salary"] = sfmt["projected_salary"] / 1_000_000
            sfmt["value_diff"]       = sfmt["value_diff"] / 1_000_000
            sfmt["ts_pct"]           = sfmt["ts_pct"] * 100
            sfmt.columns = ["Player", "Team", "GP", "MPG",
                            "Base Score", "Avail ×", "Barrett Score",
                            "Score Rank", "Salary", "Proj. Salary", "Δ Market",
                            "Salary Rank", "Rank Diff", "D-LEBRON", "TS%", "Next $"]
            sfmt.insert(0, "#", range(1, len(sfmt) + 1))
            s_style = sfmt.style.map(color_value_diff, subset=["Δ Market"]) \
                                .map(color_rank_diff, subset=["Rank Diff"]) \
                                .map(color_next_contract, subset=["Next $"]) \
                                .apply(style_rookie_salary, axis=1) \
                                .apply(highlight_tot_row, axis=1)
            s_col_config = {
                "Next $":        st.column_config.TextColumn("Next $",
                    help="Next season salary. White = guaranteed. Orange (TO) = team option. Blue (PO) = player option. Gray — = UFA.",
                    width="medium"),
                "MPG":           st.column_config.NumberColumn(format="%.2f"),
                "Base Score":    st.column_config.NumberColumn(format="%.2f"),
                "Avail ×":       st.column_config.NumberColumn(format="%.3f"),
                "Barrett Score": st.column_config.NumberColumn(format="%.2f"),
                "Salary":        st.column_config.TextColumn(
                    help="Player's actual salary this season. Purple = rookie scale contract (first-round pick, years 1–4)."),
                "Proj. Salary":  st.column_config.NumberColumn(format="$%.2fM"),
                "Δ Market":      st.column_config.NumberColumn(format="$%.2fM"),
                "D-LEBRON":      st.column_config.NumberColumn(format="%.2f"),
                "TS%":           st.column_config.NumberColumn(format="%.1f%%"),
            }
        else:
            sfmt = sdisplay[["Player", "Team", "barrett_score", "salary",
                              "projected_salary", "value_diff", "Next $"]].copy()
            _sfmt_sal_raw            = sfmt["salary"].values
            sfmt["salary"]           = [_fmt_salary(p, s) for p, s in zip(sfmt["Player"].values, _sfmt_sal_raw)]
            sfmt["projected_salary"] = sfmt["projected_salary"] / 1_000_000
            sfmt["value_diff"]       = sfmt["value_diff"] / 1_000_000
            sfmt.columns = ["Player", "Team", "Barrett Score", "Salary", "Proj. Salary", "Δ Market", "Next $"]
            sfmt.insert(0, "#", range(1, len(sfmt) + 1))
            s_style = sfmt.style.map(color_value_diff, subset=["Δ Market"]) \
                                .map(color_next_contract, subset=["Next $"]) \
                                .apply(style_rookie_salary, axis=1) \
                                .apply(highlight_tot_row, axis=1)
            s_col_config = {
                "Next $":        st.column_config.TextColumn("Next $",
                    help="Next season salary. White = guaranteed. Orange (TO) = team option. Blue (PO) = player option. Gray — = UFA.",
                    width="medium"),
                "Barrett Score": st.column_config.NumberColumn(format="%.2f"),
                "Salary":        st.column_config.TextColumn(width="medium",
                    help="Player's actual salary this season. Purple = rookie scale contract (first-round pick, years 1–4)."),
                "Proj. Salary":  st.column_config.NumberColumn(format="$%.2fM"),
                "Δ Market":      st.column_config.NumberColumn(format="$%.2fM"),
            }

        st.dataframe(s_style, column_config=s_col_config,
                     use_container_width=True, hide_index=True, height=600)
        dl_col, cap_col = st.columns([1, 5])
        with dl_col:
            st.download_button(
                "⬇ Export CSV",
                data=sfmt.to_csv(index=False),
                file_name=f"barrett_score_splits_{season}.csv",
                mime="text/csv",
                key="splits_csv",
            )
        with cap_col:
            st.caption(
                f"{len(sdisplay)} rows shown. **TOT** = full season combined. "
                "Players who switched teams mid-season appear as separate stints. "
                "**Δ Market** = Actual − Projected. "
                "Purple salary = rookie scale. "
                "**Next $**: white = guaranteed · orange (TO) = team option · blue (PO) = player option · gray = UFA."
            )
    else:
        # ── Rankings table ────────────────────────────────────────────────────
        display_fmt = display[[
            "Player", "Team", "GP",
            "base_score", "avail_mult", "barrett_score",
            "score_rank", "salary", "projected_salary", "value_diff", "salary_rank", "rank_diff",
            "d_lebron", "ts_pct",
        ]].copy()
        display_fmt["MPG"] = display_fmt["Player"].map(splits_mpg_lookup)
        display_fmt = display_fmt[["Player", "Team", "GP", "MPG",
                                   "base_score", "avail_mult", "barrett_score",
                                   "score_rank", "salary", "projected_salary", "value_diff",
                                   "salary_rank", "rank_diff", "d_lebron", "ts_pct"]]
        _sal_raw                        = display_fmt["salary"].values
        display_fmt["salary"]           = [_fmt_salary(p, s) for p, s in zip(display_fmt["Player"].values, _sal_raw)]
        display_fmt["projected_salary"] = display_fmt["projected_salary"] / 1_000_000
        display_fmt["value_diff"]       = display_fmt["value_diff"] / 1_000_000
        display_fmt["ts_pct"]           = display_fmt["ts_pct"] * 100
        display_fmt.columns = [
            "Player", "Team", "GP", "MPG",
            "Base Score", "Avail ×", "Barrett Score",
            "Score Rank", "Salary", "Proj. Salary", "Δ Market", "Salary Rank", "Rank Diff",
            "D-LEBRON", "TS%",
        ]
        display_fmt.insert(0, "#", range(1, len(display_fmt) + 1))
        display_fmt["Next $"] = display["next_contract"].values
        if not advanced:
            display_fmt = display_fmt[["#", "Player", "Team", "Barrett Score", "Salary", "Proj. Salary", "Δ Market", "Next $"]]

        style = display_fmt.style
        if "Rank Diff" in display_fmt.columns:
            style = style.map(color_rank_diff, subset=["Rank Diff"])
        if "Δ Market" in display_fmt.columns:
            style = style.map(color_value_diff, subset=["Δ Market"])
        if "Next $" in display_fmt.columns:
            style = style.map(color_next_contract, subset=["Next $"])
        style = style.apply(style_rookie_salary, axis=1)

        col_config = {
            "Next $":        st.column_config.TextColumn("Next $",
                help="Next season salary. White = guaranteed. Orange (TO) = team option. Blue (PO) = player option. Gray — = unrestricted free agent.",
                width="medium"),
            "Barrett Score": st.column_config.NumberColumn(format="%.2f",
                help="Base Score × Availability Multiplier."),
            "Salary":        st.column_config.TextColumn(
                help="Player's actual salary this season. Purple = rookie scale contract (first-round pick, years 1–4)."),
            "Proj. Salary":  st.column_config.NumberColumn(format="$%.2fM",
                help="Salary earned by whoever holds the same rank by pay."),
            "Δ Market":      st.column_config.NumberColumn(format="$%.2fM",
                help="Actual − Projected. Positive (red) = overpaid. Negative (green) = underpaid."),
        }
        if advanced:
            col_config.update({
                "GP":         st.column_config.NumberColumn(help="Games played this season."),
                "MPG":        st.column_config.NumberColumn(format="%.2f", help="Minutes per game."),
                "Base Score": st.column_config.NumberColumn(format="%.2f",
                    help="PTS + AST×2 + OREB÷2 + DREB÷3 + BLK÷2 + STL÷1.5 − TOV÷1.5 − PF÷3 + D-LEBRON×2 + Eff. Adj."),
                "Avail ×":    st.column_config.NumberColumn(format="%.3f",
                    help="0.75 + 0.25 × √((GP/82) × min(Total MIN/2500, 1))."),
                "Score Rank": st.column_config.NumberColumn(help="Rank by Barrett Score."),
                "Salary Rank":st.column_config.NumberColumn(help="Rank by actual salary."),
                "Rank Diff":  st.column_config.NumberColumn(help="Salary Rank − Score Rank. Positive = underpaid."),
                "D-LEBRON":   st.column_config.NumberColumn(format="%.2f",
                    help="Points prevented per game vs average. From bball-index.com."),
                "TS%":        st.column_config.NumberColumn(format="%.1f%%",
                    help="True Shooting %. League avg ~57%."),
            })

        table_height = min(600, max(100, len(display_fmt) * 35 + 40))
        st.dataframe(style, column_config=col_config,
                     use_container_width=True, hide_index=True, height=table_height)
        dl_col_r, cap_col_r = st.columns([1, 5])
        with dl_col_r:
            st.download_button(
                "⬇ Export CSV",
                data=display_fmt.to_csv(index=False),
                file_name=f"barrett_score_{season}.csv",
                mime="text/csv",
                key="rankings_csv",
            )
        with cap_col_r:
            if advanced:
                st.caption("**Rank Diff** = Salary Rank − Score Rank. **Δ Market** = Actual − Projected (red = overpaid, green = underpaid). "
                           "Purple salary = rookie scale contract (1st-round pick, yrs 1–4). "
                           "**Next $**: white = guaranteed · orange (TO) = team option · blue (PO) = player option · gray = UFA.")
            else:
                st.caption("**Proj. Salary** = what this player would earn paid by Barrett Score rank. **Δ Market** = Actual − Projected. "
                           "Purple salary = rookie scale contract (1st-round pick, yrs 1–4). "
                           "**Next $**: white = guaranteed · orange (TO) = team option · blue (PO) = player option · gray = UFA.")

    # ── Fill panel placeholder (above multiselect) ────────────────────────────
    new_selected = st.session_state.get("rankings_selected", [])
    if new_selected:
        with panel_placeholder.container():

            # ── Combined 5-year trend chart for all selected players ──────────
            all_trends = []
            for name in new_selected:
                if name in player_id_map_full:
                    t = fetch_career_trend(player_id_map_full[name], num_seasons=20)
                    if not t.empty:
                        t = t.copy()
                        t["Player"] = name
                        all_trends.append(t)

            if all_trends:
                trend_df = pd.concat(all_trends, ignore_index=True)

                available_seasons = sorted(trend_df["Season"].unique().tolist())
                sel_a, sel_b = st.columns(2)
                with sel_a:
                    start_season = st.selectbox("From", available_seasons,
                                                index=max(0, len(available_seasons) - 5),
                                                key="trend_start")
                with sel_b:
                    end_season = st.selectbox("To", available_seasons,
                                              index=len(available_seasons) - 1,
                                              key="trend_end")
                start_i = available_seasons.index(start_season)
                end_i   = available_seasons.index(end_season)
                if start_i > end_i:
                    start_i, end_i = end_i, start_i
                selected_seasons = available_seasons[start_i : end_i + 1]
                trend_df = trend_df[trend_df["Season"].isin(selected_seasons)]
                trend_df = trend_df.sort_values("Season")
                n_seasons = len(selected_seasons)
                fig_trend = px.line(
                    trend_df, x="Season", y="barrett_score",
                    color="Player", markers=True,
                    labels={"barrett_score": "Barrett Score", "Season": ""},
                    title=f"Barrett Score — {n_seasons}-Season Trend",
                    height=340,
                    category_orders={"Season": selected_seasons},
                )
                # Star the current season for each player
                cur = trend_df[trend_df["Season"] == season]
                if not cur.empty:
                    fig_trend.add_scatter(
                        x=cur["Season"], y=cur["barrett_score"],
                        mode="markers",
                        marker=dict(size=14, symbol="star", color="white",
                                    line=dict(width=1, color="black")),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                fig_trend.update_traces(line=dict(width=2.5), marker=dict(size=8),
                                        selector=dict(mode="lines+markers"))
                fig_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.15)",
                    font_color="white",
                    margin=dict(l=50, r=50, t=40, b=80),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="",
                               type="category",
                               categoryorder="array",
                               categoryarray=selected_seasons),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Barrett Score"),
                    legend=dict(
                        orientation="h",
                        x=0.5, xanchor="center",
                        y=-0.2, yanchor="top",
                        title="",
                    ),
                )
                st.plotly_chart(fig_trend, use_container_width=True,
                                config={"displayModeBar": False})
                no_dlebron = [s for s in selected_seasons if s < "2009-10"]
                caption = "★ = current season"
                if no_dlebron:
                    caption += (f"  ·  ⚠️ D-LEBRON unavailable for "
                                f"{', '.join(no_dlebron)} — defensive ratings set to 0 those seasons")
                st.caption(caption)

            # ── Per-player splits tables ──────────────────────────────────────
            for name in new_selected:
                title_col, btn_col = st.columns([20, 1])
                with title_col:
                    st.subheader(f"{name} — {season}")
                with btn_col:
                    if st.button("✕", key=f"x_{name}", help="Remove"):
                        updated = [n for n in new_selected if n != name]
                        st.session_state["rankings_selected"] = updated
                        st.session_state["rankings_multiselect"] = updated
                        st.rerun()
                render_splits_panel(name, season)
            st.caption("**TOT** = full season combined (bold).")
            st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Splits View
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Salary Projector
# ══════════════════════════════════════════════════════════════════════════════

with tab_projector:
    st.caption(
        "Projected salary = the actual salary of whoever holds the same rank position by pay. "
        "If Jokic is #1 by Barrett Score, he deserves the #1 salary (Curry's contract). "
        "No invented numbers — every projected salary is a real contract on the books."
    )

    proj = df.copy()
    proj["Actual $M"]     = proj["salary"] / 1e6
    proj["Proj. $M"]      = proj["projected_salary"] / 1e6
    proj["Δ $M"]          = proj["value_diff"] / 1e6
    proj["Barrett Score"] = proj["barrett_score"].round(2)
    proj["Score Rank"]    = proj["score_rank"]

    ph_col, ph_sort = st.columns([2, 1])
    with ph_col:
        proj_search = st.text_input("Highlight player", "", key="proj_search",
                                    placeholder="Type a name to highlight on the chart…")
    with ph_sort:
        proj_sort = st.selectbox("Table order", ["Most Overpaid", "Most Underpaid"], key="proj_sort")

    # Scatter: Projected $M (x) vs Actual $M (y)
    # On the y=x diagonal = fairly paid; above = overpaid; below = underpaid
    axis_max = proj[["Actual $M", "Proj. $M"]].max().max() * 1.05

    fig = px.scatter(
        proj,
        x="Proj. $M",
        y="Actual $M",
        color="Δ $M",
        color_continuous_scale="RdYlGn_r",
        color_continuous_midpoint=0,
        hover_name="Player",
        hover_data={
            "Team": True,
            "Score Rank": True,
            "Barrett Score": ":.2f",
            "Actual $M": ":.1f",
            "Proj. $M": ":.1f",
            "Δ $M": ":.1f",
        },
        labels={"Proj. $M": "Projected Salary ($M)", "Actual $M": "Actual Salary ($M)"},
        height=520,
    )

    # y=x "fairly paid" diagonal
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=axis_max, y1=axis_max,
        line=dict(color="rgba(255,255,255,0.5)", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=axis_max * 0.72, y=axis_max * 0.82,
        text="Fairly paid", showarrow=False,
        font=dict(color="rgba(255,255,255,0.5)", size=11),
        textangle=-40,
    )

    # Highlight searched players
    if proj_search:
        hi = proj[proj["Player"].str.contains(proj_search, case=False)]
        fig.add_traces(px.scatter(
            hi, x="Proj. $M", y="Actual $M",
            text="Player",
        ).update_traces(
            marker=dict(size=14, color="yellow", line=dict(color="black", width=1)),
            textposition="top center",
            name="Highlighted",
        ).data)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
        font_color="white",
        coloraxis_colorbar=dict(title="Δ ($M)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickprefix="$", ticksuffix="M",
                   range=[0, axis_max]),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickprefix="$", ticksuffix="M",
                   range=[0, axis_max]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

    # Table
    tbl = proj[["Player", "Team", "Score Rank", "Barrett Score",
                "Proj. $M", "Actual $M", "Δ $M"]].copy()
    tbl["Proj. $M"]  = tbl["Proj. $M"].round(2)
    tbl["Actual $M"] = tbl["Actual $M"].round(2)
    tbl["Δ $M"]      = tbl["Δ $M"].round(2)

    if proj_sort == "Most Overpaid":
        tbl = tbl.sort_values("Δ $M", ascending=False)
    else:
        tbl = tbl.sort_values("Δ $M", ascending=True)
    tbl = tbl.reset_index(drop=True)
    tbl.insert(0, "#", range(1, len(tbl) + 1))

    def color_delta(val):
        try:
            n = float(val)
        except (ValueError, TypeError):
            return ""
        if n > 20:  return "color: #e74c3c; font-weight: bold"
        if n > 5:   return "color: #f1a8a8"
        if n < -20: return "color: #2ecc71; font-weight: bold"
        if n < -5:  return "color: #a8e6a8"
        return ""

    st.dataframe(
        tbl.style.map(color_delta, subset=["Δ $M"]),
        column_config={
            "Proj. $M":  st.column_config.NumberColumn(format="$%.2fM"),
            "Actual $M": st.column_config.NumberColumn(format="$%.2fM"),
            "Δ $M":      st.column_config.NumberColumn(format="$%.2fM"),
            "Barrett Score": st.column_config.NumberColumn(format="%.2f"),
        },
        use_container_width=True,
        hide_index=True,
        height=500,
    )
    st.caption(
        "**Proj. Salary** = salary of whoever is the same rank by pay (e.g. score rank #1 → highest salary on the roster). "
        "**Δ $M** = Actual − Projected. Positive (red) = overpaid. Negative (green) = underpaid."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Team Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab_teams:
    st.caption(
        "Which front offices are getting the most value? "
        "Payroll efficiency = how much a team is over- or under-paying relative to what their "
        "players' Barrett Scores say they deserve. Negative = underpaying (efficient). "
        "Positive = overpaying (inefficient)."
    )

    team_grp = df.groupby("Team").agg(
        players        = ("Player",           "count"),
        actual_payroll = ("salary",           "sum"),
        proj_payroll   = ("projected_salary", "sum"),
        net_delta      = ("value_diff",       "sum"),
        avg_score      = ("barrett_score",    "mean"),
    ).reset_index()

    team_grp["actual_payroll"] /= 1e6
    team_grp["proj_payroll"]   /= 1e6
    team_grp["net_delta"]      /= 1e6
    team_grp["avg_score"]       = team_grp["avg_score"].round(2)

    team_grp = team_grp.sort_values("net_delta")  # most underpaid first

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig_teams = px.bar(
        team_grp,
        x="Team", y="net_delta",
        color="net_delta",
        color_continuous_scale="RdYlGn_r",
        color_continuous_midpoint=0,
        labels={"net_delta": "Net Δ ($M)", "Team": ""},
        title="Payroll Efficiency by Team  (negative = underpaying, positive = overpaying)",
        height=420,
        text=team_grp["net_delta"].apply(lambda v: f"${v:+.1f}M"),
    )
    fig_teams.update_traces(textposition="outside", textfont_size=10)
    fig_teams.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.15)",
        font_color="white",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", categoryorder="total ascending"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickprefix="$", ticksuffix="M"),
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig_teams, use_container_width=True, config={"displayModeBar": False})

    # ── Table ─────────────────────────────────────────────────────────────────
    # Add best and worst value player per team
    best_player  = df.loc[df.groupby("Team")["value_diff"].idxmin(), ["Team", "Player"]].set_index("Team")["Player"]
    worst_player = df.loc[df.groupby("Team")["value_diff"].idxmax(), ["Team", "Player"]].set_index("Team")["Player"]

    team_tbl = team_grp.copy()
    team_tbl["Best Value"]  = team_tbl["Team"].map(best_player)
    team_tbl["Worst Value"] = team_tbl["Team"].map(worst_player)
    team_tbl.columns = ["Team", "Players", "Actual $M", "Proj. $M",
                         "Net Δ $M", "Avg Score", "Best Value", "Worst Value"]
    team_tbl.insert(0, "#", range(1, len(team_tbl) + 1))

    def color_net(val):
        try:
            n = float(val)
        except (ValueError, TypeError):
            return ""
        if n < -20: return "color: #2ecc71; font-weight: bold"
        if n < 0:   return "color: #a8e6a8"
        if n > 20:  return "color: #e74c3c; font-weight: bold"
        if n > 0:   return "color: #f1a8a8"
        return ""

    st.dataframe(
        team_tbl.style.map(color_net, subset=["Net Δ $M"]),
        column_config={
            "Actual $M":  st.column_config.NumberColumn(format="$%.1fM",
                help="Sum of all qualifying players' actual salaries."),
            "Proj. $M":   st.column_config.NumberColumn(format="$%.1fM",
                help="Sum of what those players would earn paid by Barrett Score rank."),
            "Net Δ $M":   st.column_config.NumberColumn(format="$%.1fM",
                help="Actual − Projected. Negative (green) = team is getting value. Positive (red) = overpaying."),
            "Avg Score":  st.column_config.NumberColumn(format="%.2f",
                help="Average Barrett Score across qualifying players on the team."),
            "Best Value": st.column_config.TextColumn(help="Most underpaid player (lowest Actual − Projected)."),
            "Worst Value":st.column_config.TextColumn(help="Most overpaid player (highest Actual − Projected)."),
        },
        use_container_width=True,
        hide_index=True,
        height=min(600, len(team_tbl) * 35 + 40),
    )
    st.caption(
        f"Based on **{len(df)}** players with ≥ {min_threshold} total minutes. "
        "Only players in the rankings pool are included — rookies on minimum deals may not appear."
    )

    # ── Drill down into a team ────────────────────────────────────────────────
    st.divider()
    drill_team = st.selectbox("Drill into a team", [""] + sorted(df["Team"].unique().tolist()),
                              key="team_drill")
    if drill_team:
        team_players = df[df["Team"] == drill_team].copy()
        team_players["salary"]           /= 1e6
        team_players["projected_salary"] /= 1e6
        team_players["value_diff"]       /= 1e6
        team_players = team_players.sort_values("value_diff")
        team_players = team_players[["Player", "barrett_score", "score_rank",
                                      "salary", "projected_salary", "value_diff"]].copy()
        team_players.columns = ["Player", "Barrett Score", "Score Rank",
                                  "Salary $M", "Proj. $M", "Δ $M"]
        team_players.insert(0, "#", range(1, len(team_players) + 1))
        st.dataframe(
            team_players.style.map(color_net, subset=["Δ $M"]),
            column_config={
                "Barrett Score": st.column_config.NumberColumn(format="%.2f"),
                "Salary $M":     st.column_config.NumberColumn(format="$%.2fM"),
                "Proj. $M":      st.column_config.NumberColumn(format="$%.2fM"),
                "Δ $M":          st.column_config.NumberColumn(format="$%.2fM"),
            },
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Free Agent Class
# ══════════════════════════════════════════════════════════════════════════════

with tab_fa:
    st.caption(
        "Every player whose contract situation makes them available this offseason — "
        "UFAs, player options (they may opt out), and team options (team may decline). "
        "Ranked by Barrett Score — a GM's draft board."
    )

    # ── Classify each player's FA status from next_contract string ────────────
    def _fa_status(nc: str) -> str | None:
        if nc == "—":
            return "UFA"
        if " PO" in nc:
            return "Player Option"
        if " TO" in nc:
            return "Team Option"
        return None   # guaranteed deal — not a free agent

    fa_df = df.copy()
    fa_df["Status"] = fa_df["next_contract"].apply(_fa_status)
    fa_df = fa_df[fa_df["Status"].notna()].copy()

    # ── Summary metrics ───────────────────────────────────────────────────────
    n_ufa = (fa_df["Status"] == "UFA").sum()
    n_po  = (fa_df["Status"] == "Player Option").sum()
    n_to  = (fa_df["Status"] == "Team Option").sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Free Agents", len(fa_df))
    m2.metric("Unrestricted (UFA)", n_ufa)
    m3.metric("Player Options", n_po, help="Player can opt out and hit the market")
    m4.metric("Team Options", n_to,  help="Team may decline, making player available")

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    fa_col_a, fa_col_b, fa_col_c, fa_col_d = st.columns([2, 1, 1, 1])
    with fa_col_a:
        fa_search = st.text_input("Filter by name", "", key="fa_search")
    with fa_col_b:
        fa_status_filter = st.selectbox(
            "Status", ["All", "UFA", "Player Option", "Team Option"], key="fa_status"
        )
    with fa_col_c:
        fa_pos_filter = st.selectbox(
            "Position", ["All", "Guard", "Forward", "Center"], key="fa_pos"
        )
    with fa_col_d:
        fa_team_opts = ["All"] + sorted(fa_df["Team"].unique().tolist())
        fa_team_filter = st.selectbox("Team", fa_team_opts, key="fa_team")

    fa_display = fa_df.copy()
    if fa_search:
        fa_display = fa_display[fa_display["Player"].str.contains(fa_search, case=False)]
    if fa_status_filter != "All":
        fa_display = fa_display[fa_display["Status"] == fa_status_filter]
    if fa_pos_filter != "All":
        fa_display = fa_display[fa_display["position"] == fa_pos_filter]
    if fa_team_filter != "All":
        fa_display = fa_display[fa_display["Team"] == fa_team_filter]

    fa_display = fa_display.sort_values("barrett_score", ascending=False).reset_index(drop=True)

    # ── Build display table ───────────────────────────────────────────────────
    fa_fmt = fa_display[[
        "Player", "Team", "position", "Status",
        "barrett_score", "salary", "projected_salary", "value_diff", "next_contract",
    ]].copy()

    fa_fmt["salary"]           = [_fmt_salary(p, s) for p, s in zip(fa_fmt["Player"].values, fa_fmt["salary"].values)]
    fa_fmt["projected_salary"] = fa_fmt["projected_salary"] / 1_000_000
    fa_fmt["value_diff"]       = fa_fmt["value_diff"] / 1_000_000

    fa_fmt.columns = [
        "Player", "Team", "Pos", "Status",
        "Barrett Score", "Salary", "Proj. Value", "Δ Market", "Next $",
    ]
    fa_fmt.insert(0, "#", range(1, len(fa_fmt) + 1))

    def color_fa_status(val):
        if val == "UFA":
            return "color: #aaaaaa"
        if val == "Player Option":
            return "color: #3498db; font-weight: bold"
        if val == "Team Option":
            return "color: #f39c12; font-weight: bold"
        return ""

    fa_style = (
        fa_fmt.style
        .map(color_fa_status,    subset=["Status"])
        .map(color_next_contract, subset=["Next $"])
        .apply(style_rookie_salary, axis=1)
        .map(color_value_diff,    subset=["Δ Market"])
    )

    st.dataframe(
        fa_style,
        column_config={
            "Barrett Score": st.column_config.NumberColumn(format="%.2f",
                help="Base Score × Availability Multiplier. Higher = more valuable."),
            "Salary":        st.column_config.TextColumn(width="medium",
                help="Current season salary. Purple = rookie scale contract (1st-round pick, yrs 1–4)."),
            "Proj. Value":   st.column_config.NumberColumn(format="$%.2fM",
                help="What this player would earn if paid by their Barrett Score rank. "
                     "Useful baseline for contract negotiations."),
            "Δ Market":      st.column_config.NumberColumn(format="$%.2fM",
                help="Actual − Projected. Negative (green) = currently underpaid — "
                     "expect them to command a raise. Positive (red) = overpaid relative to production."),
            "Next $":        st.column_config.TextColumn("Next $",
                help="Option value or — for UFAs. Blue (PO) = player option. Orange (TO) = team option.",
                width="medium"),
            "Status":        st.column_config.TextColumn(
                help="UFA = unrestricted free agent. Player Option = player controls opt-out. "
                     "Team Option = team controls whether to keep player."),
            "Pos":           st.column_config.TextColumn("Pos", width="small"),
        },
        use_container_width=True,
        hide_index=True,
        height=min(800, max(200, len(fa_fmt) * 35 + 40)),
    )

    fa_dl_col, fa_cap_col = st.columns([1, 5])
    with fa_dl_col:
        st.download_button(
            "⬇ Export CSV",
            data=fa_fmt.to_csv(index=False),
            file_name=f"barrett_score_free_agents_{season}.csv",
            mime="text/csv",
            key="fa_csv",
        )
    with fa_cap_col:
        st.caption(
            f"**{len(fa_display)}** free agents shown · "
            "**Proj. Value** = salary of the player at the same Barrett Score rank in the current pool — "
            "a market-rate anchor for what this player should cost. "
            "**Δ Market**: green = underpaid (will demand raise) · red = overpaid (value risk)."
        )

    # ── Position breakdown bar chart ─────────────────────────────────────────
    if not fa_display.empty:
        st.divider()
        st.subheader("Position breakdown")
        pos_status = (
            fa_display.groupby(["position", "Status"])
            .size()
            .reset_index(name="count")
        )
        pos_status = pos_status[pos_status["position"] != ""]
        if not pos_status.empty:
            fig_fa = px.bar(
                pos_status,
                x="position", y="count",
                color="Status",
                color_discrete_map={
                    "UFA":           "#aaaaaa",
                    "Player Option": "#3498db",
                    "Team Option":   "#f39c12",
                },
                barmode="stack",
                labels={"position": "", "count": "Players", "Status": ""},
                height=320,
                category_orders={"position": ["Guard", "Forward", "Center"]},
                text_auto=True,
            )
            fig_fa.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.15)",
                font_color="white",
                margin=dict(t=20, b=20),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1),
            )
            st.plotly_chart(fig_fa, use_container_width=True, config={"displayModeBar": False})
