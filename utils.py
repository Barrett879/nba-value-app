import math
import re
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
    # ─── Pre-2006 era (no D-LEBRON; defensive rating falls back to 0) ───────
    "2005-06", "2004-05", "2003-04", "2002-03", "2001-02", "2000-01", "1999-00",
    # ─── Jordan-era Bulls + lockout (salary data scraped from BBRef teams) ──
    "1998-99", "1997-98", "1996-97",
    # ─── Pre-1996 era (BBRef per-game stats fallback; salaries from BBRef) ─
    # NBA Stats API returns nothing here — fetch_bref_player_stats provides
    # the per-game stats. D-LEBRON unavailable (set to 0). Salaries via BBRef
    # team pages. Covers Jordan's full Bulls run, all of Magic/Bird,
    # late Kareem, Dr. J, Moses Malone, prime Hakeem.
    "1995-96", "1994-95", "1993-94", "1992-93", "1991-92", "1990-91",
    "1989-90", "1988-89", "1987-88", "1986-87", "1985-86", "1984-85",
]
DEFAULT_MIN_THRESHOLD = 500

# ── League-wide pace by season (possessions per 48 minutes) ────────────────────
# Source: Basketball Reference league-averages page. Used to era-adjust volume
# stats (PTS, AST, REB, BLK, STL, TOV, PF) so a 25 PPG game in 1985 (high-pace)
# is normalized against a 25 PPG game in 2003 (dead-ball). D-LEBRON and the
# TS% efficiency adjustment are already era-relative so they don't get scaled.
LEAGUE_PACE: dict[str, float] = {
    "2025-26": 99.0,   # in-progress estimate
    "2024-25": 99.1, "2023-24": 98.5, "2022-23": 99.2, "2021-22": 97.2,
    "2020-21": 99.2, "2019-20": 100.3, "2018-19": 100.0,
    "2017-18": 97.3, "2016-17": 96.4, "2015-16": 95.8, "2014-15": 93.9,
    "2013-14": 93.9, "2012-13": 92.0, "2011-12": 91.3, "2010-11": 92.1,
    "2009-10": 92.7, "2008-09": 91.7, "2007-08": 92.4, "2006-07": 91.9,
    "2005-06": 90.5, "2004-05": 90.9, "2003-04": 90.1, "2002-03": 91.0,
    "2001-02": 90.7, "2000-01": 91.3, "1999-00": 93.1, "1998-99": 88.9,
    "1997-98": 90.3, "1996-97": 90.1, "1995-96": 91.8, "1994-95": 92.9,
    "1993-94": 95.1, "1992-93": 96.8, "1991-92": 96.6, "1990-91": 97.8,
    "1989-90": 98.3, "1988-89": 100.6, "1987-88": 99.6, "1986-87": 100.8,
    "1985-86": 102.1, "1984-85": 102.1,
}
# Reference pace = roughly the average across all seasons we cover. Volume stats
# get scaled toward this number so dead-ball-era players get a boost and
# Showtime/modern-era players get a small haircut.
REFERENCE_PACE = 96.0


def pace_factor(season: str) -> float:
    """Multiplier to bring a season's volume stats onto a pace-neutral baseline.
    Returns 1.0 for unknown seasons (no adjustment)."""
    p = LEAGUE_PACE.get(season)
    if not p or p <= 0:
        return 1.0
    return REFERENCE_PACE / p


def pace_adjusted_barrett(base_score: float, d_lebron: float,
                          efficiency_adj: float, avail_mult: float,
                          season: str) -> float:
    """Era-adjusted Barrett Score.

    The 'volume' portion of base_score (PTS + AST×1.5 + REB terms + BLK/STL/TOV
    /PF terms) gets scaled by pace_factor. D-LEBRON and the TS%-based efficiency
    adjustment are already era-relative (D-LEBRON is RAPM-based, efficiency is
    measured against league-avg TS% that season) so they're left alone.
    """
    volume = base_score - d_lebron * 2 - efficiency_adj * 2
    adjusted_base = volume * pace_factor(season) + d_lebron * 2 + efficiency_adj * 2
    return adjusted_base * avail_mult

# Actual games played per season (shortened seasons due to lockout/COVID)
SEASON_GAMES_LOOKUP = {
    "2020-21": 72, "2019-20": 72, "2011-12": 66,
    "1998-99": 50,  # lockout-shortened
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
    # ── Pre-2006 supplement ────────────────────────────────────────────────
    # ESPN's older salary pages cap around ~180 players and exclude many
    # rookie-scale contracts and veteran minimums. Fill in legends so they
    # show up in legacy career comparisons.
    "2005-06": {
        "lebron james":   4_621_800,   # rookie scale yr 3
    },
    "2004-05": {
        "lebron james":   4_320_360,   # rookie scale yr 2
    },
    "2003-04": {
        "lebron james":   4_018_920,   # rookie scale yr 1
        "carmelo anthony": 3_603_480,   # #3 pick
        "dwyane wade":    2_581_440,   # #5 pick
        "chris bosh":     3_036_240,   # #4 pick
    },
    "2002-03": {
        "michael jordan": 1_030_000,   # vet minimum, Wizards
    },
    "2001-02": {
        "michael jordan": 1_000_000,   # vet minimum, Wizards comeback
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
    ("Search Player",     "/Search"),
    ("Legacy",            "/Legacy"),
    ("Team Analysis",     "/Team_Analysis"),
    ("Trades",            "/Trades"),
    ("Current Free Agents", "/Free_Agent_Class"),
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
        + row["AST"] * 1.5
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


# ── Basketball Reference team-page salary scraper ─────────────────────────────
# For seasons before 1999-2000 ESPN's salary pages don't exist. BBRef has
# salary tables on each team's season page (hidden inside HTML comments to
# defeat scrapers — we extract them anyway). One season costs ~30 HTTP
# requests so the result is cached aggressively to disk.

_BREF_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)


def _bref_team_abbrs(end_year: int) -> list[str]:
    """All NBA team abbreviations active in the given season (end-year).
    Raises on rate-limit so the caller can back off instead of caching empty.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}.html"
    r = requests.get(url, headers={"User-Agent": _BREF_UA}, timeout=15)
    if r.status_code == 429:
        raise RuntimeError(f"BBRef rate-limited on {url}")
    if r.status_code != 200:
        return []
    return sorted(set(re.findall(rf"/teams/([A-Z]{{3}})/{end_year}\.html", r.text)))


def _bref_team_salaries(team_abbr: str, end_year: int) -> list[tuple[str, float]]:
    """Scrape one team's salary table (hidden in an HTML comment on BBRef).
    Raises on rate-limit so the caller can back off.
    """
    url = f"https://www.basketball-reference.com/teams/{team_abbr}/{end_year}.html"
    try:
        r = requests.get(url, headers={"User-Agent": _BREF_UA}, timeout=15)
        if r.status_code == 429:
            raise RuntimeError(f"BBRef rate-limited on {url}")
        if r.status_code != 200:
            return []
        from bs4 import Comment
        soup = BeautifulSoup(r.text, "html.parser")
        for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
            if 'id="salaries2"' not in comment:
                continue
            inner = BeautifulSoup(comment, "html.parser")
            tbl = inner.find("table", id="salaries2")
            if tbl is None or tbl.find("tbody") is None:
                continue
            rows = []
            for tr in tbl.find("tbody").find_all("tr"):
                cells = tr.find_all(["th", "td"])
                if len(cells) < 3:
                    continue
                player = cells[1].get_text(strip=True)
                sal_txt = cells[2].get_text(strip=True)
                if not player or not sal_txt:
                    continue
                try:
                    salary = float(re.sub(r"[\$,]", "", sal_txt))
                    rows.append((player, salary))
                except ValueError:
                    continue
            return rows
        return []
    except Exception:
        return []


@st.cache_data(ttl=30 * 86_400, show_spinner=False)
def fetch_bref_salaries(season: str) -> dict:
    """Aggregated salary lookup from BBRef team pages.

    Returns {normalized_name: salary}. Used as a fallback when ESPN doesn't
    have salary data for the season (i.e. pre-1999-2000). Caches both in
    Streamlit's cache and to disk so we only hit BBRef once per season.
    """
    end_year = season_to_espn_year(season)
    disk_path = _dc_path(f"bref_salaries_{season}.pkl")
    if _dc_fresh(disk_path, ttl=30 * 86_400):
        try:
            return _pkl_load(disk_path)
        except Exception:
            pass

    result: dict[str, float] = {}
    try:
        abbrs = _bref_team_abbrs(end_year)
    except RuntimeError:
        # Rate-limited fetching team list — bail out without caching empty
        return {}

    for i, abbr in enumerate(abbrs):
        try:
            for player, salary in _bref_team_salaries(abbr, end_year):
                result[normalize(player)] = salary
        except RuntimeError:
            # Rate-limited mid-season — stop, don't cache partial result
            return result if len(result) > 50 else {}
        except Exception:
            pass
        # Be polite to BBRef — sleep between requests to avoid rate limiting
        if i < len(abbrs) - 1:
            time.sleep(1.5)

    # Only cache to disk if we got a meaningful result. Empty caches are evil
    # because they look fresh but defeat the fallback chain.
    if len(result) > 50:
        try:
            _pkl_save(disk_path, result)
        except Exception:
            pass
    return result


# ── BBRef per-game stats scraper ─────────────────────────────────────────────
# NBA Stats API returns empty for 1995-96 and earlier. For pre-1996 seasons we
# fall back to Basketball Reference's per-game stats table. Output schema
# matches fetch_league_stats so this is a drop-in replacement upstream of
# build_raw's formula.
@st.cache_data(ttl=30 * 86_400, show_spinner=False)
def fetch_bref_player_stats(season: str) -> pd.DataFrame:
    """Per-game player stats from BBRef. Returns DataFrame with the same
    columns the NBA Stats API would return (PLAYER_ID, PLAYER_NAME,
    TEAM_ABBREVIATION, GP, MIN, PTS, AST, OREB, DREB, BLK, STL, TOV, PF,
    FGA, FTA). Used only as a fallback when NBA API has no data."""
    end_year = season_to_espn_year(season)
    # v2: cache filename bumped 2026-04-30 to orphan v1 parquets that were
    # written without TEAM_ABBREVIATION (rename map missed BBRef's "Team"
    # column). v2 files are guaranteed to have the right schema.
    disk_path = _dc_path(f"bref_stats_v2_{season}.parquet")
    if _dc_fresh(disk_path, ttl=30 * 86_400):
        try:
            cached = pd.read_parquet(disk_path)
            required = {"PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "PTS"}
            if required.issubset(cached.columns):
                return cached
            # Silently fall through and re-fetch if cache somehow lacks columns
        except Exception:
            pass

    url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_per_game.html"
    r = requests.get(url, headers={"User-Agent": _BREF_UA}, timeout=15)
    if r.status_code == 429:
        raise RuntimeError(f"BBRef rate-limited on {url}")
    if r.status_code != 200:
        return pd.DataFrame()

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", id="per_game_stats")
    if table is None:
        # Fallback — sometimes BBRef hides the per_game table inside a comment
        from bs4 import Comment
        for c in soup.find_all(string=lambda x: isinstance(x, Comment)):
            if 'id="per_game_stats"' in c:
                inner = BeautifulSoup(c, "html.parser")
                table = inner.find("table", id="per_game_stats")
                if table is not None:
                    break
    if table is None:
        return pd.DataFrame()

    df = pd.read_html(io.StringIO(str(table)))[0]

    # BBRef sprinkles repeated header rows through the table — drop those
    if "Player" in df.columns:
        df = df[df["Player"] != "Player"].reset_index(drop=True)

    # For traded players BBRef shows a TOT/2TM row + per-team rows. Keep the
    # combined row (TOT, 2TM, 3TM, ...) when present, otherwise the single row.
    team_col = "Team" if "Team" in df.columns else ("Tm" if "Tm" in df.columns else None)
    keep = []
    for player in df["Player"].dropna().unique():
        rows = df[df["Player"] == player]
        if len(rows) > 1 and team_col:
            # Combined-team rows historically labeled TOT, but newer BBRef uses
            # 2TM / 3TM / 4TM tags depending on stints.
            mask = rows[team_col].astype(str).str.match(r"^(TOT|\dTM)$", na=False)
            combined = rows[mask]
            keep.append(combined.iloc[0] if not combined.empty else rows.iloc[0])
        else:
            keep.append(rows.iloc[0])
    df = pd.DataFrame(keep).reset_index(drop=True)

    # Flatten multi-level column headers if pd.read_html returned them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]

    # Rename to NBA Stats API schema. BBRef has used both "Tm" and "Team" over
    # time, so accept either. Build the rename map only for columns that exist.
    rename_map: dict[str, str] = {}
    for src, dst in [
        ("Player",  "PLAYER_NAME"),
        ("Team",    "TEAM_ABBREVIATION"),  # current BBRef column name
        ("Tm",      "TEAM_ABBREVIATION"),  # legacy column name
        ("G",       "GP"),
        ("MP",      "MIN"),
        ("ORB",     "OREB"),
        ("DRB",     "DREB"),
    ]:
        if src in df.columns:
            rename_map[src] = dst
    df = df.rename(columns=rename_map)

    # Coerce numerics — BBRef returns strings
    numeric_cols = ["GP", "MIN", "FGA", "FTA", "PTS", "AST", "OREB", "DREB",
                    "BLK", "STL", "TOV", "PF"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill missing stats with 0 (TOV pre-1977, BLK/STL pre-1973). Formula
    # tolerates this — those components just contribute 0.
    for c in ["TOV", "BLK", "STL", "OREB", "DREB", "PF"]:
        if c not in df.columns:
            df[c] = 0
        else:
            df[c] = df[c].fillna(0)

    # Pre-1973 OREB/DREB weren't tracked separately — only TRB. Split it 30/70
    # (rough league average) so the formula doesn't double-credit total rebounds.
    if "TRB" in df.columns and df["OREB"].sum() == 0 and df["DREB"].sum() == 0:
        df["TRB"] = pd.to_numeric(df["TRB"], errors="coerce").fillna(0)
        df["OREB"] = (df["TRB"] * 0.30).round(2)
        df["DREB"] = (df["TRB"] * 0.70).round(2)

    # Map BBRef player names to NBA Stats player IDs where possible. Static
    # list covers most players including pre-1996. For unknown players we
    # generate a stable synthetic negative ID from the name hash.
    def _resolve_pid(name: str) -> int:
        try:
            results = nba_players_static.find_players_by_full_name(name)
            if results:
                return int(results[0]["id"])
        except Exception:
            pass
        return -(abs(hash(name)) % (10 ** 9))
    df["PLAYER_ID"] = df["PLAYER_NAME"].apply(_resolve_pid)

    # Final column set + dropna on the essentials
    df = df.dropna(subset=["GP", "MIN", "PTS"])
    keep_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN",
                 "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF",
                 "FGA", "FTA"]
    df = df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)

    if len(df) > 50:
        try:
            df.to_parquet(disk_path, index=False)
        except Exception:
            pass
    return df


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
            eff_adj = float(min(max(0.15 * (ts_pct - league_avg_ts_val) * 100, -6), 6))

        bs = (pts + ast * 1.5 + oreb / 2 + dreb / 3 + blk / 2 + stl / 1.5
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


@st.cache_data(ttl=3600, show_spinner=False)
def trade_side_summary(player_names: tuple[str, ...], season: str) -> dict:
    """Summarize a list of players in a given season — their Barrett Scores
    and salaries. Used by the Trades page. Player names matched by normalize().
    Returns dict with: rows (DataFrame), found (list), missing (list),
    barrett_total (float), salary_total (float)."""
    if not _raw_disk_fresh(season):
        # Don't trigger fresh build_raw on view-time requests
        return {"rows": pd.DataFrame(), "found": [], "missing": list(player_names),
                "barrett_total": 0.0, "salary_total": 0.0}

    try:
        df = apply_projections(apply_rankings(build_raw(season)))
    except Exception:
        return {"rows": pd.DataFrame(), "found": [], "missing": list(player_names),
                "barrett_total": 0.0, "salary_total": 0.0}

    if df.empty:
        return {"rows": pd.DataFrame(), "found": [], "missing": list(player_names),
                "barrett_total": 0.0, "salary_total": 0.0}

    name_norms = {normalize(n) for n in player_names}
    matched = df[df["Player"].apply(lambda n: normalize(n) in name_norms)].copy()
    found_norms = {normalize(n) for n in matched["Player"]}
    missing = [n for n in player_names if normalize(n) not in found_norms]

    return {
        "rows":          matched.reset_index(drop=True),
        "found":         list(matched["Player"]),
        "missing":       missing,
        "barrett_total": float(matched["barrett_score"].sum()) if not matched.empty else 0.0,
        "salary_total":  float(matched["salary"].sum())        if not matched.empty else 0.0,
    }


# ── Historical trades preloaded for the Trades page ───────────────────────────
# season = the season the trade ENDED (i.e., where most of the players land
# post-trade). year_after = next season, used to show how each side did after.
# Keep names exactly as they appear in NBA Stats data so normalize() matches.
HISTORICAL_TRADES = [
    {
        "name":        "Kevin Garnett to Boston (2007)",
        "season":      "2007-08",
        "year_after":  "2008-09",
        "side_a_team": "Boston Celtics",
        "side_a":      ["Kevin Garnett"],
        "side_b_team": "Minnesota Timberwolves",
        "side_b":      ["Al Jefferson", "Ryan Gomes", "Sebastian Telfair", "Gerald Green", "Theo Ratliff"],
        "notes":       "Boston also sent two future first-round picks. Won the title that year.",
    },
    {
        "name":        "Pau Gasol to Lakers (2008)",
        "season":      "2007-08",
        "year_after":  "2008-09",
        "side_a_team": "Los Angeles Lakers",
        "side_a":      ["Pau Gasol"],
        "side_b_team": "Memphis Grizzlies",
        "side_b":      ["Kwame Brown", "Javaris Crittenton", "Aaron McKie"],
        "notes":       "Lakers also got Marc Gasol's draft rights. Reached 3 straight Finals; won 2 titles.",
    },
    {
        "name":        "James Harden to Houston (2012)",
        "season":      "2012-13",
        "year_after":  "2013-14",
        "side_a_team": "Houston Rockets",
        "side_a":      ["James Harden", "Cole Aldrich", "Daequan Cook", "Lazar Hayward"],
        "side_b_team": "Oklahoma City Thunder",
        "side_b":      ["Kevin Martin", "Jeremy Lamb"],
        "notes":       "Plus 2 firsts + 2nd to OKC. One of the most lopsided trades of the modern era.",
    },
    {
        "name":        "Kawhi Leonard to Toronto (2018)",
        "season":      "2018-19",
        "year_after":  "2019-20",
        "side_a_team": "Toronto Raptors",
        "side_a":      ["Kawhi Leonard", "Danny Green"],
        "side_b_team": "San Antonio Spurs",
        "side_b":      ["DeMar DeRozan", "Jakob Poeltl"],
        "notes":       "Toronto won the championship that season. Kawhi left in free agency the next summer.",
    },
    {
        "name":        "Anthony Davis to Lakers (2019)",
        "season":      "2019-20",
        "year_after":  "2020-21",
        "side_a_team": "Los Angeles Lakers",
        "side_a":      ["Anthony Davis"],
        "side_b_team": "New Orleans Pelicans",
        "side_b":      ["Lonzo Ball", "Brandon Ingram", "Josh Hart"],
        "notes":       "Lakers won the title with AD + LeBron. New Orleans got 3 firsts in the deal too.",
    },
    {
        "name":        "Kyrie Irving to Boston (2017)",
        "season":      "2017-18",
        "year_after":  "2018-19",
        "side_a_team": "Boston Celtics",
        "side_a":      ["Kyrie Irving"],
        "side_b_team": "Cleveland Cavaliers",
        "side_b":      ["Isaiah Thomas", "Jae Crowder", "Ante Zizic"],
        "notes":       "Boston also got the Brooklyn pick (used on Collin Sexton).",
    },
    {
        "name":        "Pierce/Garnett to Brooklyn (2013)",
        "season":      "2013-14",
        "year_after":  "2014-15",
        "side_a_team": "Brooklyn Nets",
        "side_a":      ["Paul Pierce", "Kevin Garnett", "Jason Terry"],
        "side_b_team": "Boston Celtics",
        "side_b":      ["Gerald Wallace", "Kris Humphries", "Marshon Brooks", "Keith Bogans"],
        "notes":       "Boston received 3 unprotected firsts. Cornerstone of Brooklyn's prolonged decline.",
    },
    {
        "name":        "Jimmy Butler to Philadelphia (2018)",
        "season":      "2018-19",
        "year_after":  "2019-20",
        "side_a_team": "Philadelphia 76ers",
        "side_a":      ["Jimmy Butler", "Justin Patton"],
        "side_b_team": "Minnesota Timberwolves",
        "side_b":      ["Robert Covington", "Dario Šarić", "Jerryd Bayless"],
        "notes":       "Butler walked to Miami the following summer in a sign-and-trade.",
    },
    {
        "name":        "Allen Iverson to Detroit (2008)",
        "season":      "2008-09",
        "year_after":  "2009-10",
        "side_a_team": "Detroit Pistons",
        "side_a":      ["Allen Iverson"],
        "side_b_team": "Denver Nuggets",
        "side_b":      ["Chauncey Billups", "Antonio McDyess", "Cheikh Samb"],
        "notes":       "Denver pivoted to a deeper team; Detroit's veteran core fell apart.",
    },
]


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_player_names(min_seasons: int = 1) -> list[str]:
    """All player names that appear in any season, sorted by career-average
    Barrett Score (highest first, GP-weighted). Used to populate autocomplete
    dropdowns. Includes EVERY player who appeared in a game — including
    bench players, two-way contracts, and cameo seasons (Bronny, etc.)."""
    try:
        # Pass min_threshold=0 so even players with <500 total minutes show up.
        # Sorts by career-avg Barrett, so legends still rise to the top —
        # bench players just settle to the bottom of the list.
        all_seasons = build_all_seasons_combined(min_threshold=0)
        if all_seasons.empty:
            return []
        # GP-weighted career average — so a 17-game cameo doesn't drag a
        # legend's career avg down to a role-player level.
        career = (
            all_seasons.groupby("Player")
            .apply(lambda g: pd.Series({
                "avg_score": (g["barrett_score"] * g["GP"]).sum() / g["GP"].sum()
                             if g["GP"].sum() > 0 else g["barrett_score"].mean(),
                "n_seasons": g["Season"].nunique(),
            }))
            .reset_index()
        )
        career = career[career["n_seasons"] >= min_seasons]
        career = career.sort_values("avg_score", ascending=False)
        return career["Player"].tolist()
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_player_full_career(player_name: str) -> pd.DataFrame:
    """Full per-season career stats for one player: raw counting stats from
    fetch_league_stats joined with Barrett Score / rank from build_raw +
    apply_rankings. One row per season the player appeared in.

    Only reads seasons that are already on disk — view-time requests must
    NEVER trigger fresh BBRef scrapes. seed_cache.py populates the disk."""
    name_norm = normalize(player_name)
    rows: list[dict] = []
    for season in SEASONS:
        if not _raw_disk_fresh(season):
            continue
        try:
            stats = fetch_league_stats(season)
            # Pre-1996 fallback: NBA Stats API returns empty, so pull per-game
            # stats from BBRef (same source build_raw uses for those years).
            if stats.empty or "PLAYER_NAME" not in stats.columns:
                try:
                    stats = fetch_bref_player_stats(season)
                except Exception:
                    stats = pd.DataFrame()
            if stats.empty or "PLAYER_NAME" not in stats.columns:
                continue
            mask = stats["PLAYER_NAME"].apply(normalize) == name_norm
            if not mask.any():
                continue
            raw_row = stats[mask].iloc[0]

            ranked = apply_rankings(build_raw(season))
            mask2 = ranked["Player"].apply(normalize) == name_norm
            if not mask2.any():
                continue
            br_row = ranked[mask2].iloc[0]

            barrett_raw = float(br_row["barrett_score"])
            barrett_pace = pace_adjusted_barrett(
                float(br_row.get("base_score", 0)),
                float(br_row.get("d_lebron", 0)),
                float(br_row.get("efficiency_adj", 0)),
                float(br_row.get("avail_mult", 1.0)),
                season,
            )
            rows.append({
                "Season":         season,
                "Team":           raw_row["TEAM_ABBREVIATION"],
                "GP":             int(raw_row["GP"]),
                "MPG":            float(raw_row["MIN"]),
                "PTS":            float(raw_row["PTS"]),
                "AST":            float(raw_row["AST"]),
                "REB":            float(raw_row.get("OREB", 0)) + float(raw_row.get("DREB", 0)),
                "STL":            float(raw_row["STL"]),
                "BLK":            float(raw_row["BLK"]),
                "TOV":            float(raw_row["TOV"]),
                "TS%":            float(br_row.get("ts_pct", 0)) * 100,
                "Barrett Score":  barrett_raw,
                "Barrett (Pace)": barrett_pace,
                "Score Rank":     int(br_row["score_rank"]),
                "Total Players":  int(len(ranked)),
                "Salary":         float(br_row.get("salary", 0) or 0),
            })
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["_year"] = result["Season"].apply(lambda s: int(s.split("-")[0]))
    result = result.sort_values("_year").drop(columns=["_year"]).reset_index(drop=True)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_career_trend(player_id: int, num_seasons: int = 5) -> pd.DataFrame:
    """Barrett Score per season pulled directly from build_raw — guaranteed to
    match the stat panel since both use the same LeagueDashPlayerStats source.

    IMPORTANT: only reads seasons that are already on disk. View-time requests
    must NEVER trigger fresh BBRef scrapes (those take ~50s each and would
    block home-page rendering). Cache-population is seed_cache.py's job.
    """
    player_info = nba_players_static.find_player_by_id(player_id)
    if not player_info:
        return pd.DataFrame()
    name_norm = normalize(player_info["full_name"])

    rows = []
    for season in SEASONS:
        if not _raw_disk_fresh(season):
            # Season not yet seeded — skip rather than trigger a fresh fetch
            # that would hang this request for tens of seconds.
            continue
        try:
            raw = build_raw(season)
            mask = raw["Player"].apply(normalize) == name_norm
            if not mask.any():
                continue
            r = raw[mask].iloc[0]
            rows.append({
                "Season":        season,
                "GP":            int(r["GP"]),
                "MPG":           float(r["MPG"]),
                "total_min":     int(r["total_min"]),
                "base_score":    float(r["base_score"]),
                "avail_mult":    float(r["avail_mult"]),
                "barrett_score": float(r["barrett_score"]),
            })
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["_year"] = result["Season"].apply(lambda s: int(s.split("-")[0]))
    result = result.sort_values("_year").drop(columns=["_year"]).reset_index(drop=True)
    return result.tail(num_seasons).reset_index(drop=True)


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

# Bump this when the Barrett Score formula changes — old parquet caches with
# previous formula values are then ignored and rebuilt on demand.
#   v1: AST × 2, TS efficiency cap ±4 (original formula)
#   v2: AST × 1.5, TS efficiency cap ±6 (rebalanced 2026-04)
#   v3: Box Score Defense fallback for pre-2009 seasons (BLK*1.5 + STL*1.5
#       + DREB*0.15 - PF*0.4, centered on league avg, clipped to [-5, 6])
#   v4: Pre-1996 seasons keep all players (salary filled with 0 instead of
#       dropna), so MJ's pre-96 ranks are out of full league not just the
#       ~50 players with matched salary data.
FORMULA_VERSION = "v4"


def _raw_disk_path(season: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"raw_{season.replace('-', '_')}_{FORMULA_VERSION}.parquet"

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

    # Pre-1996 fallback: NBA Stats API returns empty for those years, so
    # scrape per-game stats from BBRef instead. This unlocks Magic, Bird,
    # prime Jordan, and (with 1973+) full Kareem from his Lakers years.
    if stats.empty:
        try:
            stats = fetch_bref_player_stats(season).copy()
        except Exception:
            stats = pd.DataFrame()
        if stats.empty:
            return pd.DataFrame()

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

    # Pre-2000 fallback: ESPN doesn't have those years, so scrape BBRef team
    # pages for salaries. We trigger this only when most rows are still missing
    # (cheap heuristic — current/recent seasons should be 95%+ filled by ESPN).
    if stats["salary"].isna().mean() > 0.5:
        bref_lookup = fetch_bref_salaries(season)
        if bref_lookup:
            still_missing = stats["salary"].isna()
            stats.loc[still_missing, "salary"] = stats.loc[still_missing, "PLAYER_NAME"].apply(
                lambda n: bref_lookup.get(normalize(n), float("nan"))
            )

    # Pre-1996 has sparse salary coverage (BBRef team pages miss most players,
    # HoopsHype starts ~1990, ESPN starts ~2000). Dropping salary-less rows
    # would leave only ~50 players per season → bogus "1/50" ranks for legends
    # like MJ. For those years, keep everyone with valid stats and fill missing
    # salaries with 0 (already documented as "salary unavailable" in the UI).
    pre_1996 = int(season.split("-")[0]) < 1996
    if pre_1996:
        stats["salary"] = stats["salary"].fillna(0)
    else:
        stats = stats.dropna(subset=["salary"])

    dlebron = fetch_dlebron(season)
    stats["d_lebron"] = stats["PLAYER_ID"].map(dlebron).fillna(0)

    # ── Box Score Defense fallback for pre-2009-10 seasons ────────────────────
    # D-LEBRON only goes back to 2009-10. For older seasons we compute a
    # box-score defensive estimate calibrated to roughly match D-LEBRON's
    # ±5 scale so the same `d_lebron * 2` weighting in base_score works.
    #
    # Formula: BLK*1.5 + STL*1.5 + DREB*0.15 - PF*0.4, centered on the
    # league average among qualified players (GP >= 20). Empirically this
    # gives elite shot-blockers/wing defenders ~+3 to +4, league-average
    # defenders ~0, and weak defenders ~-1.
    if not dlebron:  # empty dict = season has no D-LEBRON coverage
        _box = (stats["BLK"] * 1.5
                + stats["STL"] * 1.5
                + stats["DREB"] * 0.15
                - stats["PF"]   * 0.4)
        qualified_mask = stats["GP"] >= 20
        if qualified_mask.any():
            _league_avg_box = float(_box[qualified_mask].mean())
        else:
            _league_avg_box = float(_box.mean()) if len(_box) else 0.0
        stats["d_lebron"] = (_box - _league_avg_box).clip(-5, 6)

    stats["ts_pct"] = stats["PTS"] / (2 * (stats["FGA"] + 0.44 * stats["FTA"])).replace(0, float("nan"))
    league_avg_ts = (stats["ts_pct"] * stats["GP"]).sum() / stats["GP"].sum()
    K_EFF = 0.15
    MIN_FGA = 2.0

    def eff_adj(row):
        if row["FGA"] < MIN_FGA or pd.isna(row["ts_pct"]):
            return 0.0
        return float(min(max(K_EFF * (row["ts_pct"] - league_avg_ts) * 100, -6), 6))

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
        fetch_dlebron(SEASONS[0])
    except Exception:
        pass
    # Remaining seasons in the background
    warm_all_seasons()


@st.cache_resource(ttl=3600, show_spinner=False)
def build_all_seasons_combined(min_threshold: int = DEFAULT_MIN_THRESHOLD) -> pd.DataFrame:
    """Load every season, apply per-season rankings + projections, and concatenate.

    Rankings/projections are applied *within* each season so score_rank and
    value_diff are always comparable within a year.  The Season column is added
    so cross-season analysis can group/filter by year.

    NOTE: Uses @st.cache_resource (singleton, no copy on hit) instead of
    @st.cache_data. Callers MUST .copy() before mutating columns.
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


@st.cache_resource(ttl=3600, show_spinner=False)
def build_ranked_projected(season: str) -> pd.DataFrame:
    """Full pipeline — build_raw + apply_rankings + apply_projections — cached.

    NOTE: Uses @st.cache_resource (singleton, no copy on hit) instead of
    @st.cache_data. Callers MUST .copy() before mutating columns.
    """
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
    if df.empty:
        df["projected_salary"] = pd.Series(dtype=float)
        df["value_diff"]       = pd.Series(dtype=float)
        return df
    salaries_by_rank = df.sort_values("salary", ascending=False)["salary"].values
    n = len(salaries_by_rank)
    df["projected_salary"] = df["score_rank"].apply(
        lambda r: float(salaries_by_rank[min(int(r) - 1, n - 1)])
    )
    df["value_diff"] = df["salary"] - df["projected_salary"]  # positive = overpaid
    return df
