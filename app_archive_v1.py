import math
import time
import io
import pickle
import unicodedata
from pathlib import Path
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import leaguedashplayerstats, playercareerstats
from nba_api.stats.static import players as nba_players_static
from nba_api.stats.static import teams as nba_teams_static

CACHE_DIR = Path(__file__).parent / "cache"

st.set_page_config(page_title="Barrett Score", layout="wide")
st.title("Barrett Score — NBA Contract Value Rankings")
st.caption(
    "Base Score = PTS + AST×2 + OREB÷2 + DREB÷3 + BLK÷2 + STL÷1.5 − TOV÷1.5 − PF÷3 + D-LEBRON + Eff. Adj  (per game)  "
    "· Eff. Adj = clamp(0.15 × (TS% − Lg Avg TS%) × 100, −2, +2)  "
    "· Barrett Score = Base Score × (0.75 + 0.25 × √((GP/82) × min(MIN/2500, 1)))"
)

SEASONS = ["2025-26", "2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20",
           "2018-19", "2017-18", "2016-17", "2015-16", "2014-15", "2013-14", "2012-13",
           "2011-12", "2010-11", "2009-10"]
DEFAULT_MIN_THRESHOLD = 500


def season_to_espn_year(season: str) -> int:
    return int(season.split("-")[0]) + 1


# ── Data fetching ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Fetching league stats...")
def fetch_league_stats(season: str) -> pd.DataFrame:
    time.sleep(0.5)
    endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
    )
    return endpoint.get_data_frames()[0]


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
                                league_avg_ts: float = 0.57) -> pd.DataFrame:
    return _player_season_splits_raw(player_id, season, d_lebron_val, league_avg_ts)


def _player_season_splits_raw(player_id: int, season: str,
                               d_lebron_val: float = 0.0,
                               league_avg_ts: float = 0.57) -> pd.DataFrame:
    time.sleep(0.5)
    career = playercareerstats.PlayerCareerStats(player_id=player_id)
    df = career.get_data_frames()[0]
    rows = df[df["SEASON_ID"] == season].copy()
    if rows.empty:
        return pd.DataFrame()

    for col in ["PTS", "AST", "OREB", "DREB", "REB", "BLK", "STL", "TOV", "PF", "MIN", "FGA", "FTA"]:
        rows[col] = rows[col] / rows["GP"]

    rows["d_lebron"] = d_lebron_val
    rows["ts_pct"] = rows["PTS"] / (2 * (rows["FGA"] + 0.44 * rows["FTA"])).replace(0, float("nan"))
    rows["efficiency_adj"] = rows.apply(
        lambda r: float(min(max(0.15 * (r["ts_pct"] - league_avg_ts) * 100, -2), 2))
        if r["FGA"] >= 2.0 and not pd.isna(r["ts_pct"]) else 0.0, axis=1
    )

    rows["total_min"] = (rows["MIN"] * rows["GP"]).round(0).astype(int)
    rows["base_score"] = rows.apply(base_score, axis=1) + rows["efficiency_adj"]
    rows["avail_mult"] = rows.apply(
        lambda r: availability_multiplier(r["GP"], r["total_min"]), axis=1
    )
    rows["barrett_score"] = rows["base_score"] * rows["avail_mult"]

    return rows[["TEAM_ABBREVIATION", "GP", "MIN", "total_min",
                 "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF",
                 "d_lebron", "ts_pct", "efficiency_adj",
                 "base_score", "avail_mult", "barrett_score"]].rename(columns={
        "TEAM_ABBREVIATION": "Team",
        "MIN": "MPG",
    })


@st.cache_data(ttl=3600, show_spinner="Building splits table (30 API calls — loads once per session)…")
def build_splits_data_live(season: str, salary_lookup: tuple) -> pd.DataFrame:
    # salary_lookup: ((normalized_name, salary), ...) — hashable for cache key
    sal_dict = dict(salary_lookup)

    TOTALS_COLS = ["GP", "MIN", "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF"]

    all_rows = []
    for team in nba_teams_static.get_teams():
        time.sleep(0.3)
        try:
            ep = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="Totals",
                team_id_nullable=team["id"],
            )
            df = ep.get_data_frames()[0]
        except Exception:
            continue
        if df.empty:
            continue
        df = df.copy()
        # Tag with the team we actually queried (API returns current-team abbreviation)
        df["TEAM_ABBREVIATION"] = team["abbreviation"]
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)

    # Match salaries
    combined["salary"] = combined["PLAYER_NAME"].apply(
        lambda n: sal_dict.get(normalize(n))
    )
    combined = combined.dropna(subset=["salary"])

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
        lambda r: availability_multiplier(r["GP"], r["total_min"]), axis=1
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
        + d_lebron
    )


@st.cache_data(ttl=3600, show_spinner="Fetching D-LEBRON defensive ratings...")
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
    except Exception:
        return {}


def availability_multiplier(gp: float, total_min: float) -> float:
    return 0.75 + 0.25 * math.sqrt((gp / 82) * min(total_min / 2500, 1))


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
        return float(min(max(K_EFF * (row["ts_pct"] - league_avg_ts) * 100, -2), 2))
    stats["efficiency_adj"] = stats.apply(eff_adj, axis=1)

    stats["total_min"] = (stats["MIN"] * stats["GP"]).round(0).astype(int)
    stats["base_score"] = stats.apply(base_score, axis=1) + stats["efficiency_adj"]
    stats["avail_mult"] = stats.apply(
        lambda r: availability_multiplier(r["GP"], r["total_min"]), axis=1
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

# Build splits data — cached after first load
salary_lookup = tuple(
    (normalize(row["Player"]), row["salary"])
    for _, row in raw.iterrows()
)
splits_df = build_splits_data(season, salary_lookup)

st.caption(
    f"**{len(df)}** players ranked · "
    f"**{(df['rank_diff'] > 10).sum()}** underpaid (rank diff > +10) · "
    f"**{(df['rank_diff'] < -10).sum()}** overpaid (rank diff < −10)"
)
st.divider()

tab_rankings, tab_splits, tab_projector = st.tabs(["Rankings", "Splits View", "Salary Projector"])


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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Rankings
# ══════════════════════════════════════════════════════════════════════════════

with tab_rankings:
    player_id_map_full = {row["Player"]: int(row["PLAYER_ID"]) for _, row in df.iterrows()}
    dlebron_lookup = fetch_dlebron(season)
    league_avg_ts = float((raw["ts_pct"] * raw["GP"]).sum() / raw["GP"].sum())

    def render_splits_panel(player_name, season):
        if player_name not in player_id_map_full:
            return
        pid = player_id_map_full[player_name]
        d_leb = float(dlebron_lookup.get(pid, 0.0))
        splits = fetch_player_season_splits(pid, season, d_leb, league_avg_ts)
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
                "D-LEBRON": f"{r['d_lebron']:.2f}", "TS%": "-",
                "Eff. Adj": f"{r['efficiency_adj']:.2f}",
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
                "Eff. Adj":     st.column_config.NumberColumn(help="Efficiency adjustment added to Base Score. clamp(0.15 × (TS% − League Avg TS%) × 100, −2, +2). Rewards efficient scorers, penalises inefficient ones."),
                "Base Score":   st.column_config.NumberColumn(help="PTS + AST×2 + OREB÷2 + DREB÷3 + BLK÷2 + STL÷1.5 − TOV÷1.5 − PF÷3 + D-LEBRON + Eff. Adj. Raw per-game value before the availability multiplier."),
                "Avail ×":      st.column_config.NumberColumn(help="Availability multiplier (0.75–1.00). Rewards health and heavy minutes. 0.75 + 0.25 × √((GP/team games) × min(Total MIN/2500, 1))."),
                "Barrett Score":st.column_config.NumberColumn(help="Base Score × Availability Multiplier. The final contract value rating."),
            },
            use_container_width=True,
            hide_index=True,
        )

    # ── Compare players (multiselect persists across searches) ───────────────
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
    col_a, col_b, col_c = st.columns([2, 1, 1])
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

    display = df.copy()
    if search:
        display = display[display["Player"].str.contains(search, case=False)]
    if team_filter != "All":
        display = display[display["Team"] == team_filter]

    sort_map = {
        "Barrett Score (best first)": ("score_rank", True),
        "Salary (highest first)":     ("salary_rank", True),
        "Most Underpaid":             ("value_diff", True),
        "Most Overpaid":              ("value_diff", False),
    }
    sort_col, sort_asc = sort_map[view]
    display = display.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)
    player_names_ordered = display["Player"].tolist()

    advanced = st.toggle("Advanced view", value=False)

    display_fmt = display[[
        "Player", "Team", "GP", "MPG",
        "base_score", "avail_mult", "barrett_score",
        "score_rank", "salary", "projected_salary", "value_diff", "salary_rank", "rank_diff",
        "d_lebron", "ts_pct",
    ]].copy()
    display_fmt["salary"]           = display_fmt["salary"] / 1_000_000
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

    if not advanced:
        display_fmt = display_fmt[["#", "Player", "Team", "Barrett Score", "Salary", "Proj. Salary", "Δ Market"]]

    # ── Main table ────────────────────────────────────────────────────────────
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

    style = display_fmt.style
    if "Rank Diff" in display_fmt.columns:
        style = style.applymap(color_rank_diff, subset=["Rank Diff"])
    if "Δ Market" in display_fmt.columns:
        style = style.applymap(color_value_diff, subset=["Δ Market"])

    col_config = {
        "Barrett Score": st.column_config.NumberColumn(format="%.2f",
            help="Base Score × Availability Multiplier. Combines per-game production, defense, and efficiency with a playing-time bonus."),
        "Salary":        st.column_config.NumberColumn(format="$%.2fM",
            help="Player's actual salary this season."),
        "Proj. Salary":  st.column_config.NumberColumn(format="$%.2fM",
            help="The real salary earned by whoever holds the same rank by pay. Score rank #1 → highest salary on the books, etc."),
        "Δ Market":      st.column_config.NumberColumn(format="$%.2fM",
            help="Actual Salary minus Projected Salary. Positive (red) = overpaid vs their performance rank. Negative (green) = underpaid."),
    }
    if advanced:
        col_config.update({
            "GP":         st.column_config.NumberColumn(
            help="Games played this season."),
            "MPG":        st.column_config.NumberColumn(format="%.2f",
            help="Minutes per game."),
            "Base Score": st.column_config.NumberColumn(format="%.2f",
            help="PTS + AST×2 + OREB÷2 + DREB÷3 + BLK÷2 + STL÷1.5 − TOV÷1.5 − PF÷3 + D-LEBRON + Eff. Adj. Raw per-game value before the availability multiplier."),
            "Avail ×":    st.column_config.NumberColumn(format="%.3f",
            help="Availability multiplier (0.75–1.00). Rewards players who stay healthy and log heavy minutes. Formula: 0.75 + 0.25 × √((GP/82) × min(Total MIN/2500, 1))."),
            "Score Rank": st.column_config.NumberColumn(
            help="Player's rank by Barrett Score among all players this season."),
            "Salary Rank":st.column_config.NumberColumn(
            help="Player's rank by actual salary among all players this season."),
            "Rank Diff":  st.column_config.NumberColumn(
            help="Salary Rank minus Score Rank. Positive (green) = paid less than their performance warrants. Negative (red) = paid more."),
            "D-LEBRON":   st.column_config.NumberColumn(format="%.2f",
            help="Defensive LEBRON — estimates points prevented per game relative to average. From bball-index.com. Positive = above-average defender."),
            "TS%":        st.column_config.NumberColumn(format="%.1f%%",
            help="True Shooting % — measures scoring efficiency across 2s, 3s, and free throws. Formula: PTS / (2 × (FGA + 0.44 × FTA)). League average is ~57%."),
        })

    table_height = min(600, max(100, len(display_fmt) * 35 + 40))
    st.dataframe(
        style,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        height=table_height,
    )
    if advanced:
        st.caption(
            "**Rank Diff** = Salary Rank − Score Rank. "
            "**Δ Market** = Actual − Projected Salary (red = overpaid, green = underpaid). "
            "**Avail ×** = availability multiplier (0.75–1.00)."
        )
    else:
        st.caption("**Proj. Salary** = what this player would earn if paid according to their Barrett Score rank. **Δ Market** = Actual − Projected (green = underpaid, red = overpaid).")

    # ── Fill panel placeholder (above search) ────────────────────────────────
    new_selected = st.session_state.get("rankings_selected", [])
    if new_selected:
        with panel_placeholder.container():
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

with tab_splits:
    st.caption(
        "Every player-team combination ranked together. "
        "Traded players appear as separate rows (DAL, LAL, TOT). "
        "Non-traded players appear once. Salary is the full season value."
    )

    # splits_df already built above the tabs

    if splits_df is not None:
        # Filter controls
        sc_a, sc_b, sc_c = st.columns([2, 1, 1])
        with sc_a:
            s_search = st.text_input("Filter by player name", "", key="splits_search")
        with sc_b:
            s_sort = st.selectbox("Sort by", [
                "Barrett Score (best first)", "Most Underpaid", "Most Overpaid",
            ], key="splits_sort")
        with sc_c:
            s_team_opts = ["All"] + sorted(splits_df["Team"].unique().tolist())
            s_team = st.selectbox("Team", s_team_opts, key="splits_team")

        if s_search:
            # When searching by name, show ALL rows for matching players regardless of
            # the minutes threshold — so traded players always have complete stint breakdowns.
            matched = splits_df[splits_df["Player"].str.contains(s_search, case=False)]["Player"].unique()
            sdisplay = splits_df[splits_df["Player"].isin(matched)].copy()
        else:
            sdisplay = splits_df[splits_df["total_min"] >= min_threshold].copy()
        if s_team != "All":
            sdisplay = sdisplay[sdisplay["Team"] == s_team]

        s_sort_map = {
            "Barrett Score (best first)": ("score_rank", True),
            "Most Underpaid":             ("rank_diff", False),
            "Most Overpaid":              ("rank_diff", True),
        }
        s_col, s_asc = s_sort_map[s_sort]
        sdisplay = sdisplay.sort_values(s_col, ascending=s_asc).reset_index(drop=True)

        sfmt = sdisplay[["Player", "Team", "GP", "MPG",
                          "base_score", "avail_mult", "barrett_score",
                          "score_rank", "salary", "salary_rank", "rank_diff"]].copy()
        sfmt["salary"] = sfmt["salary"] / 1_000_000
        sfmt.columns = [
            "Player", "Team", "GP", "MPG",
            "Base Score", "Avail ×", "Barrett Score",
            "Score Rank", "Salary", "Salary Rank", "Rank Diff",
        ]

        def highlight_tot_row(row):
            if row["Team"] == "TOT":
                return ["font-weight: bold; background-color: #1e2a1e"] * len(row)
            return [""] * len(row)

        sfmt.insert(0, "#", range(1, len(sfmt) + 1))
        st.dataframe(
            sfmt.style.applymap(color_rank_diff, subset=["Rank Diff"])
                      .apply(highlight_tot_row, axis=1),
            column_config={
                "MPG":          st.column_config.NumberColumn(format="%.2f"),
                "Base Score":   st.column_config.NumberColumn(format="%.2f"),
                "Avail ×":      st.column_config.NumberColumn(format="%.3f"),
                "Barrett Score":st.column_config.NumberColumn(format="%.2f"),
                "Salary":       st.column_config.NumberColumn(format="$%.2fM"),
            },
            use_container_width=True,
            hide_index=True,
            height=600,
        )
        st.caption(
            f"{len(sdisplay)} rows shown. "
            "**TOT** rows (bold/green tint) = full season. "
            "**Rank Diff** compares each row's Barrett Score rank to the player's salary rank."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Salary Projector
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
        tbl.style.applymap(color_delta, subset=["Δ $M"]),
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

