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
import streamlit.components.v1 as components
import plotly.express as px
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import leaguedashplayerstats, playercareerstats, playerindex
from nba_api.stats.static import players as nba_players_static
from nba_api.stats.static import teams as nba_teams_static

# CACHE_DIR — use Render's persistent disk if mounted, otherwise local fallback.
# On Render: attach a disk, set mount path to /data, size 1 GB.
# Locally or on ephemeral deploys: falls back to repo-root /cache (wiped on restart).
_RENDER_DISK = Path("/data/cache")
CACHE_DIR = _RENDER_DISK if _RENDER_DISK.parent.exists() else Path(__file__).parent / "cache"

SEASONS = [
    "2025-26", "2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20",
    "2018-19", "2017-18", "2016-17", "2015-16", "2014-15", "2013-14", "2012-13",
    "2011-12", "2010-11", "2009-10", "2008-09", "2007-08", "2006-07",
]
DEFAULT_MIN_THRESHOLD = 500

# Actual games played per season (shortened seasons due to lockout/COVID)
SEASON_GAMES_LOOKUP = {
    "2020-21": 72, "2019-20": 72, "2011-12": 66,
}

# ── Salary supplement ──────────────────────────────────────────────────────────
# ESPN's historical salary rankings omit players who exercised player options
# (they're listed as "cap holds" rather than active salaries in some years).
# This supplement covers confirmed gaps so those players still appear in rankings.
SALARY_SUPPLEMENT: dict[str, dict[str, float]] = {
    "2017-18": {
        "lebron james": 33_285_709,
        "kevin durant": 25_000_000,
    },
    "2015-16": {
        "lebron james": 22_970_500,
    },
}

COMMON_CSS = """
<style>
    /* Push page content below the fixed nav bar */
    .main .block-container {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        padding-top: 3.8rem !important;
        max-width: 100%;
    }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"]        { display: none !important; }
    [data-testid="stDecoration"]     { display: none !important; }
    [data-testid="stStatusWidget"]   { display: none !important; }
    [data-testid="stAppViewerBadge"] { display: none !important; }
    [data-testid="stBottom"]         { display: none !important; }
    [data-testid="stSidebarNav"]     { display: none !important; }
    [data-testid="stSidebar"]        { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    /* Fixed top nav bar */
    .top-nav {
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 9999;
        display: flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0 1.5rem;
        height: 3rem;
        background: #0a0a0a;
        border-bottom: 1px solid #222;
        flex-wrap: nowrap;
    }
    .top-nav a {
        text-decoration: none;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #aaa;
        border: 1px solid transparent;
        transition: all 0.15s;
        white-space: nowrap;
    }
    .top-nav a:hover { border-color: #e63946; color: #fff; text-decoration: none; }
    .top-nav a.active { background: #e63946; border-color: #e63946; color: #fff; }
    .top-nav .home-link {
        color: #666;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 0.3rem 0.7rem;
        margin-right: 0.25rem;
        border: none;
    }
    .top-nav .home-link:hover { color: #fff; border: none; }
    .top-nav .divider { color: #333; font-size: 0.75rem; margin: 0 0.1rem; user-select: none; }
</style>
"""

_NAV_PAGES = [
    ("Current Rankings",  "/Rankings"),
    ("Visualizer",        "/Salary_Projector"),
    ("Team Analysis",     "/Team_Analysis"),
    ("Current Free Agents", "/Free_Agent_Class"),
    ("Legacy",            "/Legacy"),
]

def render_nav(current: str) -> None:
    """Render the top nav bar. Pass the current page title to highlight it."""
    links = '<a class="home-link" href="/" target="_top">Home</a><span class="divider">|</span>'
    for label, url in _NAV_PAGES:
        css_class = "active" if label == current else ""
        links += f'<a class="{css_class}" href="{url}" target="_top">{label}</a>'
    st.markdown(f'<div class="top-nav">{links}</div>', unsafe_allow_html=True)


# ── Name matching ──────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def season_to_espn_year(season: str) -> int:
    return int(season.split("-")[0]) + 1


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


def availability_multiplier(gp: float, total_min: float, season_games: int = 82) -> float:
    min_cap = season_games * (2500 / 82)
    return 0.75 + 0.25 * math.sqrt((gp / season_games) * min(total_min / min_cap, 1))


# ── Salary formatting ──────────────────────────────────────────────────────────

def _fmt_salary(player_name: str, salary_dollars: float) -> str:
    """Format salary as '$X.XXM'."""
    return f"${salary_dollars / 1_000_000:.2f}M"


def fmt_next_contract(player_name: str, next_contracts: dict) -> str:
    info = next_contracts.get(normalize(player_name))
    if info is None:
        return "—"
    if info["type"] == "rfa":
        return "RFA"
    sal_m = info["salary"] / 1_000_000
    if info["type"] == "team_option":
        return f"${sal_m:.1f}M TO"
    if info["type"] == "player_option":
        return f"${sal_m:.1f}M PO"
    return f"${sal_m:.1f}M"


# ── Styling helpers ────────────────────────────────────────────────────────────

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


def color_next_contract(val):
    s = str(val)
    if s == "—":
        return "color: #555555"
    if " TO" in s:
        return "color: #f39c12; font-weight: bold"
    if " PO" in s:
        return "color: #3498db; font-weight: bold"
    return ""


def style_rookie_salary(row, rookie_scale: set):
    """Color the Salary cell purple for rookie-scale players."""
    result = pd.Series('', index=row.index)
    if 'Player' in row.index and normalize(str(row['Player'])) in rookie_scale:
        if 'Salary' in row.index:
            result['Salary'] = 'color: #a855f7; font-weight: bold'
    return result


# ── Data fetching ──────────────────────────────────────────────────────────────

# ── Generic disk-cache helpers (pickle for dicts/sets, parquet for DataFrames) ─
def _dc_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name

def _dc_fresh(path: Path, season: str | None = None, ttl: int | None = None) -> bool:
    if not path.exists():
        return False
    effective_ttl = ttl if ttl is not None else (3600 if season == SEASONS[0] else 86400)
    return (time.time() - path.stat().st_mtime) < effective_ttl

def _pkl_load(path: Path):
    return pickle.loads(path.read_bytes())

def _pkl_save(path: Path, obj) -> None:
    try:
        path.write_bytes(pickle.dumps(obj))
    except Exception:
        pass


@st.cache_data(ttl=3600, show_spinner="Fetching league stats...")
def fetch_league_stats(season: str) -> pd.DataFrame:
    path = _dc_path(f"league_stats_{season.replace('-','_')}.parquet")
    if _dc_fresh(path, season=season):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
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
    df = result.get_data_frames()[0]
    try:
        df.to_parquet(path, index=False)
    except Exception:
        pass
    return df


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


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_hoopshype_salaries(season: str) -> dict:
    """Fallback salary source from HoopsHype. Returns {normalized_name: salary}."""
    start_year = season.split("-")[0]
    end_year = str(int(start_year) + 1)
    urls_to_try = [
        f"https://hoopshype.com/salaries/players/{start_year}-{end_year}/",
        f"https://hoopshype.com/salaries/players/{start_year}-{end_year[-2:]}/",
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    for url in urls_to_try:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            table = (
                soup.find("table", {"class": lambda c: c and "hh-salaries" in c})
                or soup.find("table", class_="hh-salaries-table")
                or soup.find("table")
            )
            if not table:
                continue
            result = {}
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if len(cols) < 2:
                    continue
                name = cols[1].get_text(strip=True)
                if not name:
                    continue
                for col_idx in (2, 3):
                    if col_idx >= len(cols):
                        break
                    sal_text = cols[col_idx].get_text(strip=True).replace("$", "").replace(",", "")
                    try:
                        result[normalize(name)] = float(sal_text)
                        break
                    except ValueError:
                        continue
            if result:
                return result
        except Exception:
            continue
    return {}


@st.cache_data(ttl=3600, show_spinner="Fetching player splits...")
def fetch_player_season_splits(player_id: int, season: str,
                                d_lebron_val: float = 0.0,
                                league_avg_ts: float = 0.57,
                                season_games: int = 82) -> pd.DataFrame:
    return _player_season_splits_raw(player_id, season, d_lebron_val, league_avg_ts, season_games)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_monthly_scores(player_id: int, season: str,
                          d_lebron_val: float = 0.0,
                          league_avg_ts_val: float = 0.57) -> pd.DataFrame:
    """Cumulative season-to-date Barrett Score at the end of each calendar month.

    Each row represents all games played from opening night through the last
    game of that month — so January's score includes Oct + Nov + Dec + Jan.
    Availability multiplier uses team games played through that month (not 82)
    so a player who misses no games in the first 30 gets full credit.
    """
    from nba_api.stats.endpoints import playergamelog, teamgamelog

    # ── Player game log ───────────────────────────────────────────────────────
    gl = None
    delay = 1
    while gl is None:
        try:
            gl = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 30)

    pdf = gl.get_data_frames()[0]
    if pdf.empty:
        return pd.DataFrame()

    pdf["GAME_DATE"] = pd.to_datetime(pdf["GAME_DATE"])
    pdf = pdf.sort_values("GAME_DATE").reset_index(drop=True)

    # MIN arrives as "36:24" strings in some seasons — convert to decimal
    if pdf["MIN"].dtype == object:
        def _parse_min(m):
            try:
                parts = str(m).split(":")
                return float(parts[0]) + (float(parts[1]) / 60 if len(parts) > 1 else 0)
            except Exception:
                return 0.0
        pdf["MIN"] = pdf["MIN"].apply(_parse_min)
    pdf["MIN"] = pd.to_numeric(pdf["MIN"], errors="coerce").fillna(0.0)

    # ── Team game log (to know how many games the team has played each month) ─
    # PlayerGameLog has no Team_ID column — derive team from MATCHUP field
    # e.g. "LAL vs. GSW" or "LAL @ GSW" — the player's team is always first.
    team_dates = pd.Series(dtype="datetime64[ns]")
    try:
        team_abbr = pdf["MATCHUP"].iloc[0].split()[0]
        _teams    = {t["abbreviation"]: t["id"] for t in nba_teams_static.get_teams()}
        team_id   = _teams.get(team_abbr)
        if team_id:
            tgl = None
            delay = 1
            while tgl is None:
                try:
                    tgl = teamgamelog.TeamGameLog(team_id=team_id, season=season)
                except Exception:
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
            tdf = tgl.get_data_frames()[0]
            tdf["GAME_DATE"] = pd.to_datetime(tdf["GAME_DATE"])
            team_dates = tdf["GAME_DATE"].sort_values().reset_index(drop=True)
    except Exception:
        pass

    months = sorted(pdf["GAME_DATE"].dt.to_period("M").unique())
    rows = []

    for month in months:
        cutoff = month.to_timestamp(how="end")

        # Player's cumulative games through month end
        sub = pdf[pdf["GAME_DATE"] <= cutoff]
        gp = len(sub)
        if gp < 3:
            continue

        # Team games played through month end (for accurate avail multiplier)
        if not team_dates.empty:
            team_gp = int((team_dates <= cutoff).sum())
        else:
            team_gp = SEASON_GAMES_LOOKUP.get(season, 82)  # fallback to season max

        total_min = float(sub["MIN"].sum())

        # Cumulative per-game averages
        pts  = sub["PTS"].sum()  / gp
        ast  = sub["AST"].sum()  / gp
        oreb = sub["OREB"].sum() / gp
        dreb = sub["DREB"].sum() / gp
        blk  = sub["BLK"].sum()  / gp
        stl  = sub["STL"].sum()  / gp
        tov  = sub["TOV"].sum()  / gp
        pf   = sub["PF"].sum()   / gp

        total_fga = sub["FGA"].sum()
        total_fta = sub["FTA"].sum()
        denom = 2 * (total_fga + 0.44 * total_fta)
        ts_pct = sub["PTS"].sum() / denom if denom > 0 else float("nan")

        eff_adj = 0.0
        if (total_fga / gp) >= 2.0 and not pd.isna(ts_pct):
            eff_adj = float(min(max(0.15 * (ts_pct - league_avg_ts_val) * 100, -4), 4))

        bs = (pts + ast * 2 + oreb / 2 + dreb / 3 + blk / 2 + stl / 1.5
              - tov / 1.5 - pf / 3 + d_lebron_val * 2 + eff_adj * 2)

        # avail_mult uses team_gp as the season-games denominator so that
        # appearing in 30/30 team games = full availability, not 30/82
        min_cap = team_gp * (2500 / 82)
        avail = 0.75 + 0.25 * math.sqrt(
            (gp / team_gp) * min(total_min / min_cap, 1)
        )
        barrett = bs * avail

        rows.append({
            "Month":         month.strftime("%b '%y"),
            "month_order":   str(month),
            "GP":            gp,
            "team_GP":       team_gp,
            "base_score":    round(bs, 2),
            "avail_mult":    round(avail, 3),
            "barrett_score": round(barrett, 2),
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner="Fetching career trend...")
def fetch_career_trend(player_id: int, num_seasons: int = 5) -> pd.DataFrame:
    """Barrett Score for each of the player's last N seasons, with real D-LEBRON."""
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
    sal_dict = dict(salary_lookup)

    TOTALS_COLS = ["GP", "MIN", "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF"]

    def _fetch_team(team: dict):
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
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_team, t): t for t in teams}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_rows.append(result)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)

    combined["salary"] = combined["PLAYER_NAME"].apply(
        lambda n: sal_dict.get(normalize(n))
    )
    combined = combined.dropna(subset=["salary"])

    season_games = int(combined["GP"].max())

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

    pg_cols = ["MIN", "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF"]
    for col in pg_cols:
        combined[col] = combined[col] / combined["GP"]

    combined["total_min"]     = (combined["MIN"] * combined["GP"]).round(0).astype(int)
    combined["base_score"]    = combined.apply(base_score, axis=1)
    combined["avail_mult"]    = combined.apply(
        lambda r: availability_multiplier(r["GP"], r["total_min"], season_games), axis=1
    )
    combined["barrett_score"] = combined["base_score"] * combined["avail_mult"]

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
    """Returns {normalized_name: "Guard"|"Forward"|"Center"} from ESPN salary pages."""
    path = _dc_path(f"bref_positions_{espn_year}_v{cache_v}.pkl")
    if _dc_fresh(path, ttl=86400):
        try:
            return _pkl_load(path)
        except Exception:
            pass
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
    _pkl_save(path, result)
    return result


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_rookie_scale_players(season: str) -> set:
    """Returns a set of normalized names for players on rookie scale contracts."""
    path = _dc_path(f"rookie_scale_{season.replace('-','_')}.pkl")
    if _dc_fresh(path, ttl=86400):
        try:
            return _pkl_load(path)
        except Exception:
            pass
    try:
        end_year = int(season.split("-")[0]) + 1
        rookie_draft_years = set(range(end_year - 3, end_year + 1))

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
        _pkl_save(path, rookies)
        return rookies
    except Exception:
        return set()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_next_year_contracts(espn_year: int, cache_v: int = 7) -> dict:
    """
    Returns {normalized_name: {"salary": float, "type": str}} for next season.
    type is one of: "guaranteed", "team_option", "player_option", "rfa".
    """
    path = _dc_path(f"next_contracts_{espn_year}_v{cache_v}.pkl")
    if _dc_fresh(path, ttl=86400):
        try:
            return _pkl_load(path)
        except Exception:
            pass
    next_year = espn_year + 1
    contracts: dict = {}
    _hdrs = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        sal_df = fetch_salaries(next_year)
        for _, row in sal_df.iterrows():
            contracts[normalize(row["name"])] = {"salary": row["salary"], "type": "guaranteed"}
    except Exception:
        pass

    try:
        r = requests.get(
            "https://www.spotrac.com/nba/free-agents/",
            headers=_hdrs, timeout=15,
        )
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            tables = soup.find_all("table")
            fa_table = tables[1] if len(tables) >= 2 else (tables[0] if tables else None)
            if fa_table is not None:
                tdf = pd.read_html(io.StringIO(str(fa_table)))[0]
                cols_lower = {str(c).lower().strip(): c for c in tdf.columns}
                name_col  = tdf.columns[0]
                type_col  = cols_lower.get("type")
                if type_col is not None:
                    for _, row in tdf.iterrows():
                        norm = normalize(str(row[name_col]))
                        t = str(row[type_col]).upper().strip()
                        if t.startswith("PLAYER"):
                            if norm in contracts:
                                contracts[norm]["type"] = "player_option"
                        elif t.startswith("CLUB") or t.startswith("TEAM"):
                            if norm in contracts:
                                contracts[norm]["type"] = "team_option"
                        elif t.startswith("RFA"):
                            contracts.setdefault(norm, {"salary": 0.0, "type": "rfa"})
    except Exception:
        pass

    _pkl_save(path, contracts)
    return contracts


@st.cache_data(ttl=3600, show_spinner="Fetching D-LEBRON defensive ratings...")
def fetch_dlebron_all() -> pd.DataFrame:
    """Fetches all D-LEBRON data in one call — every player, every season back to 2009-10."""
    path = _dc_path("dlebron_all.parquet")
    if _dc_fresh(path, ttl=3600):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
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
        df = df[["nba_id", "Season", "D-LEBRON"]].dropna()
        try:
            df.to_parquet(path, index=False)
        except Exception:
            pass
        return df
    except Exception:
        return pd.DataFrame(columns=["nba_id", "Season", "D-LEBRON"])


def fetch_dlebron(season: str) -> dict:
    """Returns {player_id (int): d_lebron (float)} for a single season."""
    df = fetch_dlebron_all()
    if df.empty:
        return {}
    season_df = df[df["Season"] == season]
    return {int(row["nba_id"]): float(row["D-LEBRON"]) for _, row in season_df.iterrows()}


# ── Build raw data ─────────────────────────────────────────────────────────────

def _raw_disk_path(season: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"raw_{season.replace('-', '_')}.parquet"

def _raw_disk_fresh(season: str) -> bool:
    """True if the on-disk parquet is still within its TTL."""
    p = _raw_disk_path(season)
    if not p.exists():
        return False
    # Current season refreshes every hour; historical seasons every 30 days
    ttl = 3600 if season == SEASONS[0] else 30 * 86_400
    return (time.time() - p.stat().st_mtime) < ttl

@st.cache_data(ttl=3600, show_spinner="Building rankings...")
def build_raw(season: str) -> pd.DataFrame:
    # ── Disk cache hit: load parquet instead of hitting the APIs ──────────────
    if _raw_disk_fresh(season):
        try:
            return pd.read_parquet(_raw_disk_path(season))
        except Exception:
            pass  # corrupted file — fall through to live fetch

    # Fetch stats + salaries in parallel — saves ~5s on cold cache misses
    with ThreadPoolExecutor(max_workers=2) as _pool:
        _stats_f = _pool.submit(fetch_league_stats, season)
        _sal_f   = _pool.submit(fetch_salaries, season_to_espn_year(season))
        stats    = _stats_f.result().copy()
        salaries = _sal_f.result()

    sal_lookup = {normalize(n): s for n, s in zip(salaries["name"], salaries["salary"])}
    stats["salary"] = stats["PLAYER_NAME"].apply(lambda n: sal_lookup.get(normalize(n)))

    supplement = SALARY_SUPPLEMENT.get(season, {})
    if supplement:
        missing_mask = stats["salary"].isna()
        stats.loc[missing_mask, "salary"] = stats.loc[missing_mask, "PLAYER_NAME"].apply(
            lambda n: supplement.get(normalize(n), float("nan"))
        )

    missing_mask = stats["salary"].isna()
    if missing_mask.any():
        hh_lookup = fetch_hoopshype_salaries(season)
        if hh_lookup:
            stats.loc[missing_mask, "salary"] = stats.loc[missing_mask, "PLAYER_NAME"].apply(
                lambda n: hh_lookup.get(normalize(n), float("nan"))
            )

    stats = stats.dropna(subset=["salary"])

    dlebron = fetch_dlebron(season)
    stats["d_lebron"] = stats["PLAYER_ID"].map(dlebron).fillna(0)

    stats["ts_pct"] = stats["PTS"] / (2 * (stats["FGA"] + 0.44 * stats["FTA"])).replace(0, float("nan"))
    league_avg_ts = (stats["ts_pct"] * stats["GP"]).sum() / stats["GP"].sum()
    K_EFF = 0.15
    MIN_FGA = 2.0

    def eff_adj(row):
        if row["FGA"] < MIN_FGA or pd.isna(row["ts_pct"]):
            return 0.0
        return float(min(max(K_EFF * (row["ts_pct"] - league_avg_ts) * 100, -4), 4))

    stats["efficiency_adj"] = stats.apply(eff_adj, axis=1)

    season_games = int(stats["GP"].max())

    stats["total_min"] = (stats["MIN"] * stats["GP"]).round(0).astype(int)
    stats["base_score"] = stats.apply(base_score, axis=1) + stats["efficiency_adj"] * 2
    stats["avail_mult"] = stats.apply(
        lambda r: availability_multiplier(r["GP"], r["total_min"], season_games), axis=1
    )
    stats["barrett_score"] = stats["base_score"] * stats["avail_mult"]

    result = stats[[
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "total_min",
        "base_score", "avail_mult", "barrett_score", "salary",
        "d_lebron", "ts_pct", "efficiency_adj",
    ]].rename(columns={
        "PLAYER_NAME": "Player",
        "TEAM_ABBREVIATION": "Team",
        "MIN": "MPG",
    })

    # ── Persist to disk so future cold-starts skip the API entirely ───────────
    try:
        result.to_parquet(_raw_disk_path(season), index=False)
    except Exception:
        pass

    return result


def warm_all_seasons() -> None:
    """Background-warm historical seasons. Current season is already warmed
    synchronously by _bootstrap_warm, so we skip it here to avoid duplicate work.
    """
    def _warm(s: str) -> None:
        try:
            build_raw(s)
            fetch_bref_positions(season_to_espn_year(s), cache_v=3)
            fetch_next_year_contracts(season_to_espn_year(s), cache_v=7)
            fetch_rookie_scale_players(s)
        except Exception:
            pass

    def _run_pool() -> None:
        historical = SEASONS[1:]  # current season already warm
        with ThreadPoolExecutor(max_workers=5) as pool:
            pool.map(_warm, historical)
        # After all individual seasons are warm, pre-build the combined legacy dataset
        try:
            build_all_seasons_combined()
        except Exception:
            pass
        try:
            fetch_draft_classes()
        except Exception:
            pass

    threading.Thread(target=_run_pool, daemon=True).start()


@st.cache_resource
def _bootstrap_warm() -> None:
    """Fires once per server process. Current season is warmed synchronously so
    the first user never hits a cold build_raw. Historical seasons warm in a
    background thread so startup doesn't block.
    """
    # Warm current season + its supporting data synchronously — eliminates
    # the race condition where background threads haven't finished when the
    # first user hits the Rankings page.
    try:
        build_ranked_projected(SEASONS[0])
        fetch_bref_positions(season_to_espn_year(SEASONS[0]), cache_v=3)
        fetch_next_year_contracts(season_to_espn_year(SEASONS[0]), cache_v=7)
        fetch_rookie_scale_players(SEASONS[0])
    except Exception:
        pass
    # Remaining seasons in the background
    warm_all_seasons()


@st.cache_data(ttl=3600, show_spinner=False)
def build_all_seasons_combined(min_threshold: int = DEFAULT_MIN_THRESHOLD) -> pd.DataFrame:
    """Load every season, apply per-season rankings + projections, and concatenate.

    Rankings/projections are applied *within* each season so score_rank and
    value_diff are always comparable within a year.  The Season column is added
    so cross-season analysis can group/filter by year.
    """
    path = _dc_path(f"all_seasons_{min_threshold}.parquet")
    if _dc_fresh(path, ttl=3600):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    frames: list[pd.DataFrame] = []
    for season in SEASONS:
        try:
            raw      = build_raw(season)
            ranked   = apply_rankings(raw)
            projected = apply_projections(ranked)
            filt     = projected[projected["total_min"] >= min_threshold].copy()
            filt["Season"] = season
            frames.append(filt)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    try:
        combined.to_parquet(path, index=False)
    except Exception:
        pass
    return combined


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_draft_classes() -> pd.DataFrame:
    """Draft history from the NBA: Player, draft_year (int), round, pick."""
    path = _dc_path("draft_history.parquet")
    if _dc_fresh(path, ttl=86400):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    try:
        from nba_api.stats.endpoints import drafthistory
        time.sleep(0.6)
        df = drafthistory.DraftHistory().get_data_frames()[0]
        keep = [c for c in ["PLAYER_NAME", "SEASON", "ROUND_NUMBER", "ROUND_PICK", "OVERALL_PICK"] if c in df.columns]
        df = df[keep].copy().rename(columns={"PLAYER_NAME": "Player", "SEASON": "draft_year"})
        df["draft_year"] = pd.to_numeric(df["draft_year"], errors="coerce")
        df = df.dropna(subset=["draft_year"])
        df["draft_year"] = df["draft_year"].astype(int)
        df["player_norm"] = df["Player"].apply(normalize)
        try:
            df.to_parquet(path, index=False)
        except Exception:
            pass
        return df
    except Exception:
        return pd.DataFrame(columns=["Player", "draft_year", "player_norm",
                                     "ROUND_NUMBER", "ROUND_PICK", "OVERALL_PICK"])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_player_career_all_seasons(player_name: str) -> pd.DataFrame:
    """Return every season a player appears in raw data, regardless of minutes played.

    Uses per-season apply_rankings so score_rank reflects their true league rank
    that year.  No minutes threshold — injury years, cameo seasons all included.
    """
    name_norm = normalize(player_name)
    frames: list[pd.DataFrame] = []
    for season in SEASONS:
        try:
            raw  = build_raw(season)
            mask = raw["Player"].apply(normalize) == name_norm
            if not mask.any():
                continue
            ranked = apply_rankings(raw)
            row    = ranked[mask].copy()
            row["Season"] = season
            frames.append(row)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["_season_year"] = combined["Season"].apply(lambda s: int(s.split("-")[0]))
    combined = combined.sort_values("_season_year").reset_index(drop=True)
    return combined


@st.cache_data(ttl=3600, show_spinner=False)
def build_ranked_projected(season: str) -> pd.DataFrame:
    """Full pipeline — build_raw + apply_rankings + apply_projections — cached."""
    return apply_projections(apply_rankings(build_raw(season)))


def apply_rankings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score_rank"]  = df["barrett_score"].rank(ascending=False, method="min").astype(int)
    df["salary_rank"] = df["salary"].rank(ascending=False, method="min").astype(int)
    df["rank_diff"]   = df["salary_rank"] - df["score_rank"]
    return df


def apply_projections(df: pd.DataFrame) -> pd.DataFrame:
    """Projected salary = the actual salary of whoever holds the same rank position by salary."""
    df = df.copy()
    salaries_by_rank = df.sort_values("salary", ascending=False)["salary"].values
    n = len(salaries_by_rank)
    df["projected_salary"] = df["score_rank"].apply(
        lambda r: float(salaries_by_rank[min(int(r) - 1, n - 1)])
    )
    df["value_diff"] = df["salary"] - df["projected_salary"]  # positive = overpaid
    return df
