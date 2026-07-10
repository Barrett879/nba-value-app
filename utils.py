"""Shared logic + UI helpers for the HoopsValue Streamlit app.

╔══════════════════════════════════════════════════════════════════════════╗
║  TABLE OF CONTENTS                                                       ║
║                                                                          ║
║  1.  Configuration constants                                             ║
║      - CACHE_DIR, SEASONS, DEFAULT_MIN_THRESHOLD                         ║
║      - SALARY_CAP_M, cap_dollars()                                       ║
║      - Contract calibration: AGE / POSITION multipliers, thresholds      ║
║      - age_bucket()                                                      ║
║      - LEAGUE_PACE, pace_factor()                                        ║
║      - SALARY_SUPPLEMENT (pre-1990 hand-curated)                         ║
║      - SEASON_GAMES_LOOKUP                                               ║
║      - FORMULA_VERSION + PLAYOFF_VERSION (cache invalidation)            ║
║      - Logger                                                            ║
║                                                                          ║
║  2.  CSS + nav rendering                                                 ║
║      - COMMON_CSS                                                        ║
║      - _HIDE_BADGE_SCRIPT, render_page_chrome()                          ║
║      - render_nav(), render_playoff_toggle()                             ║
║      - render_barrett_score_explainer()                                  ║
║                                                                          ║
║  3.  Text + name normalization                                           ║
║      - normalize(), season_to_espn_year()                                ║
║                                                                          ║
║  4.  Scoring formula                                                     ║
║      - base_score(), availability_multiplier()                           ║
║                                                                          ║
║  5.  Formatting helpers                                                  ║
║      - _fmt_salary(), fmt_next_contract()                                ║
║      - color_rank_diff(), color_value_diff(), color_next_contract()      ║
║      - style_rookie_salary()                                             ║
║                                                                          ║
║  6.  Disk cache helpers                                                  ║
║      - _dc_path(), _dc_fresh(), _pkl_load(), _pkl_save()                 ║
║                                                                          ║
║  7.  Data sources (network scrapers)                                     ║
║      - NBA Stats API: fetch_league_stats(), fetch_dlebron(), …           ║
║      - BBRef: fetch_bref_player_stats(), fetch_bref_salaries(),          ║
║        fetch_playoff_rounds(), fetch_player_positions_detailed()         ║
║      - Spotrac/HoopsHype: fetch_next_year_contracts(),                   ║
║        fetch_hoopshype_salaries(), fetch_rookie_scale_players()          ║
║      - ESPN: fetch_bref_positions() (legacy coarse positions)            ║
║                                                                          ║
║  8.  Ranking pipeline                                                    ║
║      - build_raw(), raw per-season stats + Barrett Score                ║
║      - apply_rankings(), apply_projections()                             ║
║      - build_ranked_projected(), entry point used by every page         ║
║      - warm_all_seasons(), _bootstrap_warm()                             ║
║      - build_all_seasons_combined() (Legacy page)                        ║
║                                                                          ║
║  9.  Career / player lookup                                              ║
║      - get_all_player_names(), get_player_id_map()                       ║
║      - fetch_player_full_career(), fetch_career_trend()                  ║
║      - fetch_season_component_distribution()                             ║
║      - fetch_player_career_all_seasons(),                                ║
║        fetch_player_career_with_rank()                                   ║
║                                                                          ║
║  10. Splits + monthly                                                    ║
║      - fetch_player_season_splits(), fetch_monthly_scores()              ║
║      - build_splits_data(), load_splits_from_disk()                      ║
║                                                                          ║
║  11. Trades (page disabled but logic preserved)                          ║
║      - trade_side_summary(), HISTORICAL_TRADES                           ║
║                                                                          ║
║  12. Draft                                                               ║
║      - fetch_draft_classes()                                             ║
║                                                                          ║
║  CACHE STRATEGY                                                          ║
║                                                                          ║
║  Three layers of caching, in order from fastest to slowest:              ║
║                                                                          ║
║  a. @st.cache_data, in-memory per Streamlit session. Keyed by all       ║
║     function arguments. Use TTL for data that ages (current-season       ║
║     stats: 1h; historical: 24h+; truly static: no TTL).                  ║
║                                                                          ║
║  b. Disk cache (CACHE_DIR/*.{parquet,pkl}), survives process restart.   ║
║     Filenames carry a version suffix to invalidate on schema changes.    ║
║     Conventions:                                                         ║
║       raw_<season>_<FORMULA_VERSION>.parquet, main ranking output   ║
║       raw_<season>_playoff_<PLAYOFF_VERSION>_<FORMULA_VERSION>.parquet   ║
║       splits_<season>_<FORMULA_VERSION>.pkl                              ║
║       bref_positions_<year>_v<cache_v>.pkl, per-scraper cache_v   ║
║       positions_detailed_<year>_v<cache_v>.pkl                           ║
║       bref_salaries_<season>.pkl                                         ║
║                                                                          ║
║     Bump FORMULA_VERSION when base_score() weights change so stale       ║
║     ranks don't poison fresh ones. Bump per-scraper cache_v when         ║
║     that scraper's parser changes.                                       ║
║                                                                          ║
║  c. Network fetch (slow, last resort). Always behind @st.cache_data,     ║
║     always wrapped in try/except + logger.warning so failures surface.   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
import html
import logging
import math
import re
import time
import io
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import unicodedata
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup


# ── Lazy nba_api modules ──────────────────────────────────────────────────────
# Importing nba_api.stats.endpoints costs ~1s locally (~3-4s on Render's
# half-CPU box) because the package __init__ pulls in EVERY endpoint module.
# Warm-parquet page loads never touch the API, so defer the import to first
# live use. The proxy swaps itself for the real module after that first touch,
# so steady-state access is native speed. (plotly.express was also imported
# here and never used: it dragged xarray in on every cold start, now gone.)
class _LazyModule:
    def __init__(self, path: str, alias: str):
        self._path, self._alias = path, alias

    def __getattr__(self, attr):
        import importlib
        mod = importlib.import_module(self._path)
        globals()[self._alias] = mod
        return getattr(mod, attr)


leaguedashplayerstats = _LazyModule("nba_api.stats.endpoints.leaguedashplayerstats", "leaguedashplayerstats")
playercareerstats     = _LazyModule("nba_api.stats.endpoints.playercareerstats", "playercareerstats")
playerindex           = _LazyModule("nba_api.stats.endpoints.playerindex", "playerindex")
nba_players_static    = _LazyModule("nba_api.stats.static.players", "nba_players_static")
nba_teams_static      = _LazyModule("nba_api.stats.static.teams", "nba_teams_static")


# ── Logging ──────────────────────────────────────────────────────────────────
# Centralized logger so cache misses, scrape failures, and parse errors don't
# silently disappear. View logs in `render logs` for the live app or in your
# terminal locally. Set HOOPSVALUE_LOG=DEBUG for verbose output.
logger = logging.getLogger("hoopsvalue")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[hoopsvalue] %(levelname)s %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(os.environ.get("HOOPSVALUE_LOG", "WARNING").upper())

# CACHE_DIR — use Render's persistent disk if mounted, otherwise local fallback.
# On Render: attach a disk, set mount path to /data, size 1 GB.
# Locally or on ephemeral deploys: falls back to repo-root /cache (wiped on restart).
_RENDER_DISK = Path("/data/cache")
CACHE_DIR = _RENDER_DISK if _RENDER_DISK.parent.exists() else Path(__file__).parent / "cache"


def _seed_disk_cache_from_repo() -> None:
    """Copy the cache committed in the repo (./cache, shipped in the deploy image)
    into the persistent disk for any files the disk doesn't already have.

    On Render CACHE_DIR is /data/cache; a fresh/empty disk would otherwise force
    the app to cold-fetch the entire NBA API on first load. Seeding it from the
    committed snapshot means it serves from local disk instead, seconds, not
    minutes. Gap-fill only (never clobbers fresher data the disk accumulated),
    runs once per process at import, and can never break the app: any failure
    just logs and the normal network fallback still works.
    """
    import shutil
    repo_cache = Path(__file__).parent / "cache"
    if CACHE_DIR == repo_cache or not repo_cache.is_dir():
        return  # local/dev already reads the repo cache directly — nothing to seed
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        have = set(os.listdir(CACHE_DIR))
        copied = 0
        for src in repo_cache.iterdir():
            if src.is_file() and src.name not in have:
                shutil.copy2(src, CACHE_DIR / src.name)
                copied += 1
        if copied:
            logger.info("seeded %d cache files from repo -> %s", copied, CACHE_DIR)
    except Exception as e:  # noqa: BLE001 — seeding is best-effort, never fatal
        logger.warning("disk-cache seed skipped: %s", e)


_seed_disk_cache_from_repo()

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
    # ─── 1970s / early-80s era (full counting stats: BLK, STL, TOV, OREB,
    # DREB all start 1973-74 — earlier seasons would need a different
    # formula). Salaries: pre-1990 the BBRef team pages don't have data,
    # so most rows show $0 (the pre_1996 fillna path handles it cleanly).
    # Covers: rookie Magic / Bird, Dr. J's full prime, Kareem at his peak,
    # Wilt's last year, Walt Frazier, Rick Barry, prime Moses Malone.
    # Playoff bracket sizes varied (8-12 teams) until 1983-84's 16-team
    # standard — round-credit math still works, just with fewer R1 teams.
    "1983-84", "1982-83", "1981-82", "1980-81", "1979-80",
    "1978-79", "1977-78", "1976-77", "1975-76", "1974-75", "1973-74",
]
DEFAULT_MIN_THRESHOLD = 500

# ── Salary cap by season ($M) ──────────────────────────────────────────────────
# NBA / Spotrac historical cap. Pre-1984 had no cap. Single source of truth
# for the analyzer scripts (analyze_accuracy.py, optimize_weights.py, etc.)
# and the Contract Predictor page. Update one place, every script picks it up.
SALARY_CAP_M = {
    "1984-85":   3.6,  "1985-86":   4.2,  "1986-87":   4.9,  "1987-88":   6.2,
    "1988-89":   7.2,  "1989-90":   9.8,  "1990-91":  11.9,  "1991-92":  12.5,
    "1992-93":  14.0,  "1993-94":  15.2,  "1994-95":  15.9,  "1995-96":  23.0,
    "1996-97":  24.4,  "1997-98":  26.9,  "1998-99":  30.0,  "1999-00":  34.0,
    "2000-01":  35.5,  "2001-02":  42.5,  "2002-03":  40.3,  "2003-04":  43.8,
    "2004-05":  43.9,  "2005-06":  49.5,  "2006-07":  53.1,  "2007-08":  55.6,
    "2008-09":  58.7,  "2009-10":  57.7,  "2010-11":  58.0,  "2011-12":  58.0,
    "2012-13":  58.0,  "2013-14":  58.7,  "2014-15":  63.1,  "2015-16":  70.0,
    "2016-17":  94.1,  "2017-18":  99.1,  "2018-19": 101.9,  "2019-20": 109.1,
    "2020-21": 109.1,  "2021-22": 112.4,  "2022-23": 123.7,  "2023-24": 136.0,
    "2024-25": 140.6,  "2025-26": 154.6,
    # 2026-27 is the season a new contract signed "today" would start —
    # NBA projection $165M (Shams/ESPN, ~$10M rise). Used to price the
    # Contract Predictor's output in next-season dollars.
    "2026-27": 165.0,
}


def cap_dollars(season: str, fallback_M: float = 154.6) -> float:
    """Salary cap in actual dollars (not millions) for a season.
    Falls back to current-season cap if unknown (pre-1984)."""
    return SALARY_CAP_M.get(season, fallback_M) * 1_000_000


# ── Contract Predictor calibration constants ──────────────────────────────────
# Age multipliers — fit on 2014-22 real new contracts. A 33-year-old with the
# same Score as a 27-year-old signs for about 28% less.
CONTRACT_AGE_MULTIPLIERS = {
    "≤22":   0.890,
    "23-25": 0.971,
    "26-28": 1.000,
    "29-31": 1.000,
    "32-34": 0.723,
    "35+":   0.574,
}

# Position multipliers — Centers are systematically overprojected by the
# box-score-heavy Barrett Score (rebounds aren't paid like points).
CONTRACT_POSITION_MULTIPLIERS = {
    "Guard":   0.971,
    "Forward": 0.949,
    "Center":  0.810,
    "Unknown": 0.960,
}

# Out-of-sample median |error| on the new-contract test set is 1.8% of cap.
# Confidence band = ~2× median ≈ 3.6% of cap.
CONFIDENCE_BAND_PCT_OF_CAP = 0.036

# Filter thresholds — single source of truth across the analyzer scripts and
# the Contract Predictor page.
HEALTHY_SEASON_GP   = 40    # min GP for a season to count as "healthy"
NEW_CONTRACT_PCT    = 0.25  # ≥25% YoY salary change = "new deal" proxy
SUPERMAX_CAP_PCT    = 0.28  # base ≥ this fraction of cap → suppress pos mult
TOP_N_DIRECTIONAL   = 20    # top-N underpaid/overpaid calls per season


def age_bucket(age) -> str:
    """Five-bucket age classification used by the contract calibration layer.
    Stays consistent across the Contract Predictor page and all analyzer
    scripts so a tweak here propagates everywhere.

    NOTE: kept for backward-compat with analyzer scripts that compare
    bucket-based calibration to the new tiered model. Production
    prediction now uses tiered_age_multiplier() below."""
    if pd.isna(age) if hasattr(pd, "isna") else age is None:
        return "UNK"
    try:
        age = int(age)
    except (TypeError, ValueError):
        return "UNK"
    if age <= 22: return "≤22"
    if age <= 25: return "23-25"
    if age <= 28: return "26-28"
    if age <= 31: return "29-31"
    if age <= 34: return "32-34"
    return "35+"


def tiered_age_multiplier(age, career_score: float,
                          current_rank: int | None = None) -> tuple[float, str]:
    """Tier-aware age decline curve. Returns (multiplier, tier_name).

    Three populations age very differently in real NBA contracts:

      ELITE, Top 30 by career-weighted score, OR career_score ≥ 28.
                  These guys (LeBron, Curry, KD, Harden) hold their
                  contract value until ~37 then decline slowly (1.5% / yr).
                  Their body fails before the market does.

      ROTATION, Top 100, OR career_score ≥ 18. Real NBA starters who
                  take moderate age discounts (~3% / yr past 28).

      DEPTH, Everyone else. Bench / role guys who get steeply
                  discounted past 28 (6% / yr), flooded out by younger
                  options at similar production.

    All tiers get an additional decline past age 35 (body fails). The
    floor is 0.40, a 40+ year-old depth player isn't completely
    worthless but is signing at vet-minimum tier.

    Replaces the bucket-based age multiplier (CONTRACT_AGE_MULTIPLIERS).
    Continuous (no discontinuities at bucket boundaries) and tier-aware
    (the average aging player and the aging star are different
    populations).
    """
    if age is None or (hasattr(pd, "isna") and pd.isna(age)):
        return 1.0, "Unknown"
    try:
        age = float(age)
    except (TypeError, ValueError):
        return 1.0, "Unknown"

    # No age penalty in prime years.
    if age <= 28:
        return 1.0, "Prime (≤28)"

    # Tier classification — production rank takes priority because it's
    # the freshest signal. Fall back to career-score threshold when rank
    # is unknown.
    score = float(career_score or 0)
    rank_ok = current_rank is not None and current_rank > 0
    if (rank_ok and current_rank <= 30) or score >= 28:
        tier = "Elite"
        decline_rate    = 0.015  # 1.5% per year past 28
        body_fail_rate  = 0.010  # extra 1% per year past 34
    elif (rank_ok and current_rank <= 100) or score >= 18:
        tier = "Rotation"
        decline_rate    = 0.030
        body_fail_rate  = 0.020
    else:
        tier = "Depth"
        decline_rate    = 0.060
        body_fail_rate  = 0.040

    base_decline  = decline_rate   * (age - 28)
    extra_decline = body_fail_rate * max(0.0, age - 34)
    multiplier = max(0.40, 1.0 - base_decline - extra_decline)
    return round(multiplier, 3), tier


def durability_multiplier(career_df, lookback_seasons: int = 3
                          ) -> tuple[float, str, float]:
    """Contract discount based on chronic availability.

    Real GMs separate two things when projecting contracts:
      1. Production rate (what you do when on the floor)  → rate score
      2. Durability     (how often you're actually on the floor) → THIS

    The rate-score input to the model treats Curry's one-anomaly 41-GP
    season the same as a healthy 75-GP year (he's elite either way).
    But the durability question is real for chronic cases, Embiid
    averaging 30 GP/yr over 3 years signals teams can't bank on him.
    That deserves its own multiplier separate from production.

    Computed on trailing N seasons (default 3). All seasons count toward
    the GP sum, including ones below the GP ≥ 40 "healthy" filter, we
    explicitly want to PENALIZE missing games here, not exclude them.

    Returns: (multiplier, tier_label, trailing_gp_ratio).

    Tiers:
        >= 0.85 trailing avail (~70 GP/yr): healthy   → ×1.00
        >= 0.70 (~58 GP/yr):                mild      → ×0.95
        >= 0.55 (~45 GP/yr):                moderate  → ×0.88
        >= 0.40 (~33 GP/yr):                chronic   → ×0.78
        < 0.40:                              severe    → ×0.65

    Rookies / players with <2 seasons: returns ×1.00 ("insufficient
    history") so we don't double-penalize them on top of the rookie
    scale lock.
    """
    if career_df.empty:
        return 1.0, "no career data", 0.0
    recent = career_df.tail(lookback_seasons)
    if len(recent) < 2:
        return 1.0, "insufficient history", 0.0

    total_gp = float(recent["GP"].sum())
    expected = len(recent) * 82  # max possible GP across the window
    avail = total_gp / expected if expected > 0 else 0.0

    if avail >= 0.85: return 1.00, "Healthy",  avail
    if avail >= 0.70: return 0.95, "Mild",     avail
    if avail >= 0.55: return 0.88, "Moderate", avail
    if avail >= 0.40: return 0.78, "Chronic",  avail
    return 0.65, "Severe", avail


def playoff_bonus_multiplier(playoff_career_df, lookback_seasons: int = 3
                              ) -> tuple[float, str, float, int]:
    """Bonus multiplier for proven recent playoff performers.

    Why one-way (bonus only, no penalty):
      - Lottery teams can't help that their team didn't make the playoffs.
        Penalizing a star on a 25-win team for "no playoff exposure" would
        confuse "playing for a bad team" with "underperforming under
        pressure." The market doesn't do that.
      - Players who DO make playoffs AND perform well get a real bump
        from GMs (Jokic, Tatum, SGA premium). We capture that.

    Why MOST RECENT (not 3-year trailing average):
      - GMs price the next contract off the freshest playoff impression.
        Bruce Brown after the 2023 BKN run, Wiggins after the 2022 title,
        Rui after the 2023 WCF run, all got paid off ONE playoff
        performance, not a smoothed multi-year average.
      - 3-year trailing dilutes a strong recent run with weak earlier
        cameos. Rui's 2024-25 playoff Barrett ~20 gets dragged toward
        ~14 by lower-role years on Washington.

    Requires ≥4 GP in the most recent playoff appearance (filters
    2-game cameos / DNP first-round sweeps that aren't representative
    of a real postseason role).

    Returns: (multiplier, tier_label, most_recent_playoff_barrett, gp).

    Tiers (most recent playoff Barrett, with availability):
        >= 31:  Elite playoff    → ×1.15   (Jokic Finals MVP tier)
        >= 24:  Strong playoff   → ×1.10   (SGA / Tatum)
        >= 16:  Solid playoff    → ×1.05   (Rui WCF / Bruce Brown BKN)
        < 16:   Average / role   → ×1.00

    Magnitudes calibrated to be meaningful without overwhelming the rate-
    score base. Real-world playoff signings (Bruce Brown +244%, Rui ~+200%)
    are outliers driven by free-agent market dynamics the model can't see;
    these multipliers capture the systematic playoff premium, not the
    breakout-event magnitude.
    """
    if playoff_career_df is None or playoff_career_df.empty:
        return 1.0, "No playoff data", 0.0, 0

    # Filter out cameo appearances (DNP, single-series sweeps where the
    # player barely contributed). ≥4 GP = at least one round played.
    healthy = playoff_career_df[playoff_career_df["GP"] >= 4]
    if healthy.empty:
        return 1.0, "No qualifying playoff data", 0.0, 0

    # Use the MOST RECENT qualifying playoff appearance as the tier
    # driver. This matches what GMs actually price off.
    most_recent = healthy.iloc[-1]
    gp = int(most_recent["GP"])
    playoff_barrett = float(most_recent["Barrett Score"])

    if playoff_barrett >= 31:
        return 1.15, "Elite playoff",   playoff_barrett, gp
    if playoff_barrett >= 24:
        return 1.10, "Strong playoff",  playoff_barrett, gp
    if playoff_barrett >= 16:
        return 1.05, "Solid playoff",   playoff_barrett, gp
    return 1.0, "Average playoff",  playoff_barrett, gp


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
    # ─── 1970s / early-80s pace (high-pace, low-efficiency era) ───────────
    "1983-84": 101.4, "1982-83": 103.1, "1981-82": 100.9, "1980-81": 101.4,
    "1979-80": 103.1, "1978-79": 108.1, "1977-78": 107.8, "1976-77": 106.5,
    "1975-76": 105.5, "1974-75": 105.6, "1973-74": 107.8,
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
    # ── Current-season fill-ins ────────────────────────────────────────────
    # Players who played real minutes but are absent from the ESPN salary feed
    # (two-way deals, late-season conversions the feed hasn't caught), so they'd
    # otherwise be dropped for a missing salary. Without this they vanish from
    # the Rankings entirely.
    # ── Current-season fill-ins ────────────────────────────────────────────
    # Players who played real minutes but are absent from the ESPN salary feed
    # (two-ways, late-season conversions, 10-day deals), so they'd otherwise be
    # dropped for a missing salary and vanish from the Rankings. Salaries were
    # verified per player (Spotrac/HoopsHype/SalarySwish); sub-two-way prorated
    # figures floored to the $636,435 two-way scale for clean display. Nick Smith
    # Jr. (LAL) and Tolu Smith (DET) each replaced a waived player -- see
    # data/roster_corrections.csv.
    "2025-26": {
        "nick smith jr.": 636_435,
        "tolu smith": 636_435,
        "aaron holiday": 3_080_921,
        "alijah martin": 636_435,
        "bez mbeng": 636_435,
        "blake hinson": 636_435,
        "branden carlson": 636_435,
        "brooks barnhizer": 636_435,
        "caleb love": 636_435,
        "chaney johnson": 636_435,
        "charles bassey": 2_296_274,
        "chris youngblood": 636_435,
        "christian koloko": 636_435,
        "cormac ryan": 636_435,
        "curtis jones": 636_435,
        "daeqwon plowden": 636_435,
        "dalen terry": 636_435,
        "david jones garcia": 636_435,
        "dejon jarreau": 2_296_274,
        "e.j. liddell": 636_435,
        "elijah harkless": 636_435,
        "ethan thompson": 636_435,
        "garrison mathews": 2_031_929,
        "isaiah crawford": 636_435,
        "isaiah livers": 636_435,
        "jahmai mashack": 636_435,
        "jahmir young": 636_435,
        "jalen slawson": 636_435,
        "jamal cain": 636_435,
        "javon small": 636_435,
        "javonte cooke": 636_435,
        "jeremiah robinson-earl": 636_435,
        "john poulakidas": 636_435,
        "johnny juzang": 636_435,
        "julian reese": 636_435,
        "kennedy chandler": 636_435,
        "kevin mccullar jr.": 636_435,
        "killian hayes": 3_018_158,
        "kj simpson": 636_435,
        "kobe sanders": 636_435,
        "koby brea": 636_435,
        "lachlan olbrich": 636_435,
        "leaky black": 636_435,
        "lj cryer": 636_435,
        "luke travers": 636_435,
        "mac mcclung": 2_378_870,
        "malachi smith": 636_435,
        "malevy leons": 636_435,
        "marjon beauchamp": 636_435,
        "max shulga": 636_435,
        "miles kelly": 636_435,
        "moussa cisse": 636_435,
        "nate williams": 636_435,
        "oscar tshiebwe": 636_435,
        "pete nance": 977_689,
        "pj hall": 636_435,
        "rayan rupert": 636_435,
        "rayj dennis": 636_435,
        "ron harper jr.": 636_435,
        "sharife cooper": 636_435,
        "taelon peter": 636_435,
        "tony bradley": 2_940_876,
        "trey jemison iii": 636_435,
        "tyler burton": 636_435,
        "tyrese martin": 2_191_897,
        "tyson etienne": 636_435,
        "tyty washington jr.": 636_435,
        "yuki kawamura": 636_435,
    },
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
    # ─── Pre-1990 supplement (HISTORICAL APPROXIMATIONS) ────────────────────
    # NBA salary disclosures were not consistently public pre-1990. Numbers
    # below are drawn from contemporary newspaper reporting, biographies, and
    # documented contract milestones; precise per-year figures sometimes vary
    # between sources by ~5-15%. Treat as best-effort estimates for the top
    # ~10 highest-paid players each season. Most other players in these eras
    # show $0 salary (genuine data gap — see PRE_1990_SALARY_NOTE).
    "1989-90": {
        "magic johnson":    2_500_000,
        "larry bird":       3_600_000,
        "patrick ewing":    2_750_000,
        "michael jordan":   2_500_000,
        "hakeem olajuwon":  2_500_000,
        "david robinson":   2_500_000,   # rookie deal, big number
        "charles barkley":  2_000_000,
        "akeem olajuwon":   2_500_000,   # alt spelling used pre-'91
        "karl malone":      1_500_000,
        "isiah thomas":     1_500_000,
    },
    "1988-89": {
        "magic johnson":    2_500_000,
        "larry bird":       2_500_000,
        "michael jordan":   2_000_000,
        "hakeem olajuwon":  2_000_000,
        "akeem olajuwon":   2_000_000,
        "patrick ewing":    1_700_000,
        "moses malone":     1_900_000,
        "isiah thomas":     1_500_000,
    },
    "1987-88": {
        "larry bird":       1_950_000,
        "magic johnson":    2_500_000,
        "michael jordan":   845_000,    # final year of rookie scale
        "kareem abdul-jabbar": 2_000_000,
        "akeem olajuwon":   1_800_000,
        "moses malone":     1_800_000,
        "patrick ewing":    1_700_000,
    },
    "1986-87": {
        "larry bird":       1_800_000,
        "magic johnson":    2_500_000,
        "kareem abdul-jabbar": 2_000_000,
        "patrick ewing":    1_700_000,
        "moses malone":     1_800_000,
        "akeem olajuwon":   1_500_000,
        "michael jordan":   630_000,    # rookie scale yr 3
    },
    "1985-86": {
        "patrick ewing":    1_700_000,   # 10-yr/$30M lifetime deal start
        "larry bird":       1_800_000,
        "magic johnson":    1_500_000,
        "kareem abdul-jabbar": 2_000_000,
        "akeem olajuwon":   1_500_000,
        "moses malone":     1_800_000,
        "michael jordan":   630_000,    # rookie scale yr 2
    },
    "1984-85": {
        "kareem abdul-jabbar": 1_500_000,
        "magic johnson":    1_500_000,
        "larry bird":       1_800_000,
        "moses malone":     2_200_000,
        "michael jordan":   550_000,    # rookie scale yr 1
        "akeem olajuwon":   1_500_000,  # rookie deal
    },
    "1983-84": {
        "moses malone":     2_200_000,   # 76ers post-championship
        "magic johnson":    1_200_000,
        "larry bird":       1_800_000,
        "kareem abdul-jabbar": 1_500_000,
        "julius erving":    800_000,
        "ralph sampson":    1_500_000,   # rookie deal
    },
    "1982-83": {
        "moses malone":     2_000_000,   # signed massive 76ers deal
        "magic johnson":    1_000_000,
        "larry bird":       1_500_000,
        "kareem abdul-jabbar": 1_500_000,
        "julius erving":    800_000,
    },
    "1981-82": {
        "magic johnson":    1_000_000,   # 25-yr/$25M deal averaged
        "larry bird":       1_000_000,
        "julius erving":    800_000,
        "moses malone":     900_000,
        "kareem abdul-jabbar": 1_000_000,
    },
    "1980-81": {
        "kareem abdul-jabbar": 1_000_000,
        "magic johnson":    1_000_000,
        "julius erving":    800_000,
        "larry bird":       800_000,
        "moses malone":     800_000,
    },
    "1979-80": {
        "kareem abdul-jabbar": 1_000_000,
        "magic johnson":    500_000,     # rookie deal
        "larry bird":       650_000,     # rookie deal (Celtics)
        "julius erving":    700_000,
        "moses malone":     500_000,
    },
    "1978-79": {
        "kareem abdul-jabbar": 600_000,
        "julius erving":    600_000,
        "bill walton":      600_000,
        "george gervin":    400_000,
        "moses malone":     400_000,
    },
    "1977-78": {
        "kareem abdul-jabbar": 625_000,
        "julius erving":    600_000,
        "bill walton":      500_000,
        "pete maravich":    500_000,
        "george mcginnis":  500_000,
    },
    "1976-77": {
        "kareem abdul-jabbar": 625_000,
        "julius erving":    600_000,     # to 76ers post-ABA merger
        "pete maravich":    500_000,
        "bill walton":      250_000,
    },
    "1975-76": {
        "kareem abdul-jabbar": 625_000,  # Lakers signing
        "pete maravich":    500_000,
        "walt frazier":     400_000,
        "rick barry":       400_000,
    },
    "1974-75": {
        "kareem abdul-jabbar": 450_000,  # last Bucks year
        "walt frazier":     400_000,
        "rick barry":       250_000,     # returning from ABA
        "bob mcadoo":       350_000,
    },
    "1973-74": {
        "wilt chamberlain": 450_000,     # final season (Lakers)
        "kareem abdul-jabbar": 350_000,
        "walt frazier":     350_000,
        "bob mcadoo":       300_000,
        "jerry west":       350_000,
    },
}

# Disclaimer shown anywhere we display pre-1990 salary-derived metrics.
PRE_1990_SALARY_NOTE = (
    "⚠️ Pre-1990 NBA salary disclosures were inconsistent and not always public. "
    "Figures shown for these seasons come from a hand-curated supplement of "
    "documented historical contracts (newspaper archives, biographies, BBRef "
    "where available). Coverage is limited to the ~10 highest-paid stars per "
    "year, so most other players show $0 salary, which is the genuine data gap. "
    "Use rank/score for these seasons; salary-derived metrics (Δ Market, Proj. "
    "Salary) are best-effort estimates."
)

# ════════════════════════════════════════════════════════════════════════════
#  THEME TOKENS  —  single source of truth for the light/dark system
# ════════════════════════════════════════════════════════════════════════════
# The DARK token values are chosen to *exactly equal* the colours that used to
# be hardcoded across the app, so rewriting `#fff` -> `var(--fg-1)` is a pixel
# no-op in dark mode. Light mode injects THEME_LIGHT_CSS, which redefines the
# same tokens with white-surface values. Accents mostly hold; teal/green/gold/
# value-tints nudge for legibility on white. inject_theme() (called by every
# page's chrome) emits the base + the light override when light is active.

#: Default theme. The design refresh ships LIGHT by default; dark is opt-in via
#: the nav toggle. (config.toml base is set to "light" to match, so iframe
#: components like the player searchbox — which CSS can't reach — render light.)
THEME_DEFAULT_DARK = False

THEME_BASE_CSS = """
<style>
    :root {
        /* surfaces */
        --app-bg:      #0a0a14;                     /* .stApp base (flat)       */
        --bg-base:     #0a0a14;
        --bg-nav:      #0a0a0a;
        --panel:       rgba(20, 20, 42, 0.55);      /* default card / strip     */
        --panel-solid: #15171d;                     /* opaque card              */
        --panel-2:     #1a1a2e;                      /* secondary dark panel     */
        --panel-hover: rgba(30, 30, 56, 0.85);
        --panel-line:  rgba(80, 80, 110, 0.35);     /* card hairline border     */
        --hairline:    rgba(255, 255, 255, 0.08);   /* divider                  */
        --hairline-soft:rgba(255, 255, 255, 0.04);  /* faint track              */
        --nav-border:  #222;
        --nav-divider: #333;
        /* tinted value-card surfaces */
        --tint-good:   #1a2e1a;
        --tint-bad:    #2e1a1a;
        --tint-even:   #1a1a2e;
        /* text ramp */
        --fg-1: #ffffff;   /* headings / primary    */
        --fg-2: #cdcdd5;   /* body                  */
        --fg-3: #aaaaaa;   /* secondary / nav rest  */
        --fg-4: #8a8a93;   /* captions / eyebrows   */
        --fg-5: #777777;   /* faint meta            */
        --fg-6: #666666;   /* disabled / home-link  */
        /* brand accents */
        --accent-red:  #e63946;
        --accent-teal: #16d4c1;
        --value-good:  #2ecc71;
        --value-good-s:#a8e6a8;
        --value-bad:   #e74c3c;
        --value-bad-s: #f1a8a8;
        --gold:        #f1c40f;
        --blue:        #3498db;
        --orange:      #f39c12;
        --purple:      #9b59b6;
        --sky:         #7ec8e8;
        --amber:       #f0b35b;   /* caveat / model chips */
        /* logo metals (on dark) */
        --logo-copper: #b06a38;
        --logo-sage:   #4f8a68;
        --logo-tag:    #8a8d98;
        /* swoosh wordmark: navy reads as near-white on dark surfaces */
        --logo-navy:   #e2e9f8;
        --logo-orange: #f6863a;
        /* hero wordmark image, dark variant (navy recolored light) */
        --logo-img: url('/app/static/hoopsvalue_wordmark_dark.png');
        /* elevation */
        --shadow-card: 0 4px 16px rgba(0, 0, 0, 0.35);
        /* table polish (zebra rows + in-cell score bars) */
        --row-tint: rgba(255, 255, 255, 0.025);
        --bar-tint: rgba(22, 212, 193, 0.16);
    }
    /* Paint the app surface + native text from tokens so every page follows the
       theme regardless of config.toml base (which can't flip at runtime).
       html/body too, so the dark Streamlit default never seams through below
       short pages or during overscroll. */
    html, body, .stApp { background: var(--app-bg) !important; }
    .stApp, body { color: var(--fg-2); }
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3 { color: var(--fg-1) !important; }

    /* Native widget labels (selectbox / input / multiselect / slider / checkbox
       / radio) follow the token text colour. config.toml base can't flip at
       runtime, so without this they stay dark (black-on-dark) in dark mode. */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label,
    .stCheckbox label, .stRadio label,
    [data-baseweb="form-control-label"] {
        color: var(--fg-2) !important;
    }

    /* Native Streamlit widgets painted from tokens so they follow BOTH themes.
       (config.toml base="light" handles iframe components like the searchbox,
       which CSS can't reach; these token rules handle the styleable widgets in
       light AND dark.) */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div {
        background: var(--panel-solid) !important;
        border-color: var(--panel-line) !important;
        color: var(--fg-1) !important;
    }
    div[data-baseweb="select"] svg { fill: var(--fg-3) !important; }
    /* Dropdown option list (selectbox / multiselect popover). Target by ROLE,
       not just `ul li`, recent BaseWeb renders options as [role=option] that
       don't always match the `ul[menu] li` descendant selector, which left the
       option text near-black on the dark menu. Cover the container + the option
       text (and its inner nodes) so it follows the theme. */
    ul[data-baseweb="menu"], div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] [role="listbox"] {
        background: var(--panel-solid) !important;
    }
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] [role="option"] *,
    ul[data-baseweb="menu"] [role="option"],
    ul[data-baseweb="menu"] li {
        color: var(--fg-2) !important;
    }
    [data-baseweb="popover"] [role="option"]:hover,
    ul[data-baseweb="menu"] [role="option"]:hover,
    ul[data-baseweb="menu"] li:hover {
        background: var(--panel-hover) !important;
    }
    [data-testid="stTextInput"] div[data-baseweb="input"],
    [data-testid="stNumberInput"] div[data-baseweb="input"],
    [data-testid="stTextInput"] div[data-baseweb="base-input"],
    div[data-baseweb="input"] {
        background: var(--panel-solid) !important;
        border-color: var(--panel-line) !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    div[data-baseweb="input"] input { color: var(--fg-1) !important; }
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: var(--fg-4) !important; }
    [data-testid="stExpander"] details {
        background: var(--panel) !important;
        border-color: var(--panel-line) !important;
    }
    /* Default st.button / st.download_button (the ✕ remove buttons, Export CSV)
, Streamlit's light-config white doesn't follow the runtime dark theme.
       The brightness/theme button keeps its own chrome-stripped look via a
       higher-specificity rule in COMMON_CSS (loads after this). */
    [data-testid="stButton"] button, .stButton button,
    [data-testid="stDownloadButton"] button, .stDownloadButton button {
        background: var(--panel-solid) !important;
        border-color: var(--panel-line) !important;
        color: var(--fg-2) !important;
    }
    [data-testid="stButton"] button:hover, .stButton button:hover,
    [data-testid="stDownloadButton"] button:hover, .stDownloadButton button:hover {
        background: var(--panel-hover) !important;
        border-color: var(--fg-5) !important;
        color: var(--fg-1) !important;
    }
    /* st.spinner, the "Loading …" bar renders on a white track in light config. */
    [data-testid="stSpinner"] {
        color: var(--fg-3) !important;
        background: transparent !important;
    }
    [data-testid="stSpinner"] > div { background: transparent !important; }
    [data-testid="stSpinner"] i, [data-testid="stSpinner"] svg { border-color: var(--accent-teal) !important; }
</style>
"""

THEME_LIGHT_CSS = """
<style>
    :root {
        /* surfaces -> light */
        --app-bg:      linear-gradient(180deg, #fbfcfd 0%, #eef1f4 100%);
        --bg-base:     #f4f6f8;
        --bg-nav:      #ffffff;
        --panel:       #ffffff;
        --panel-solid: #ffffff;
        --panel-2:     #eef1f4;
        --panel-hover: #f1f3f6;
        --panel-line:  #e3e6eb;
        --hairline:    rgba(20, 22, 40, 0.10);
        --hairline-soft:rgba(20, 22, 40, 0.05);
        --nav-border:  #e3e6eb;
        --nav-divider: #c9ccd3;
        /* tinted value-card surfaces -> pale */
        --tint-good:   #eafaf1;
        --tint-bad:    #fdeceb;
        --tint-even:   #eef3f8;
        /* text ramp -> dark-on-light */
        --fg-1: #14142a;
        --fg-2: #3a3d48;
        --fg-3: #585c68;
        --fg-4: #71757f;
        --fg-5: #9aa0ab;
        --fg-6: #b3b8c2;
        /* accents nudged darker for white-bg contrast */
        --accent-teal: #0fae9d;
        --value-good:  #16a34a;
        --value-good-s:#2fbb6e;
        --value-bad:   #dc3a2c;
        --value-bad-s: #e7584b;
        --gold:        #9a6a00;   /* dark amber, not yellow, on white */
        --amber:       #a8730a;
        --orange:      #b45f06;
        --purple:      #7d3fa8;
        --blue:        #2471a3;   /* accent text: >= 4.5:1 on white/panel-2 */
        --sky:         #146c94;   /* player-name links + RFA chips, same hue */
        /* logo metals (on light) */
        --logo-copper: #985729;
        --logo-sage:   #3d6f52;
        --logo-tag:    #7a7d88;
        /* swoosh wordmark: reference navy + orange on light surfaces */
        --logo-navy:   #1c2c5b;
        --logo-orange: #f2711d;
        /* hero wordmark image, light variant (original navy + orange) */
        --logo-img: url('/app/static/hoopsvalue_wordmark.png');
        /* elevation -> soft light card shadow */
        --shadow-card: 0 1px 2px rgba(20,22,40,.06), 0 4px 14px rgba(20,22,40,.07);
        /* table polish (zebra rows + in-cell score bars) */
        --row-tint: rgba(20, 22, 40, 0.028);
        --bar-tint: rgba(15, 174, 157, 0.15);
    }
</style>
"""


# Copies the active ?theme=... URL param onto every internal <a> link, so a
# full-reload page navigation (each tab is a fresh Streamlit session) carries
# the chosen theme. Runs in the components iframe; reaches the app via
# window.parent. Idempotent + re-runs on Streamlit re-renders (MutationObserver).
_THEME_PERSIST_SCRIPT = """
<script>
(function () {
    function sync() {
        try {
            var W = window.parent;
            var theme = new URLSearchParams(W.location.search).get('theme');
            if (theme !== 'dark' && theme !== 'light') return;
            W.document.querySelectorAll('a[href]').forEach(function (a) {
                var h = a.getAttribute('href');
                if (!h || h.charAt(0) === '#') return;
                var internal = h.charAt(0) === '/' || h.indexOf(W.location.origin) === 0;
                if (!internal) return;
                if (/[?&]theme=/.test(h)) {
                    a.setAttribute('href', h.replace(/([?&])theme=(dark|light)/, '$1theme=' + theme));
                } else {
                    a.setAttribute('href', h + (h.indexOf('?') === -1 ? '?' : '&') + 'theme=' + theme);
                }
            });
        } catch (e) {}
    }
    sync();
    try {
        new MutationObserver(sync).observe(
            window.parent.document.documentElement, { childList: true, subtree: true });
    } catch (e) {}
})();
</script>
"""


def inject_theme() -> None:
    """Emit the theme tokens (dark base + light override when active).

    Call once near the top of every page's chrome, BEFORE any CSS that
    references the tokens. The chosen theme persists across page navigations
    via the ?theme=... URL param: render_theme_toggle() writes it, this seeds
    st.session_state['theme_dark'] from it on a fresh session, and
    _THEME_PERSIST_SCRIPT copies it onto every internal link.
    """
    # Initialise the theme once per session (before the toggle widget claims the
    # key) from the URL ?theme=, else the default. Setting it here AND passing a
    # widget value= would make Streamlit warn, so render_theme_toggle() relies
    # purely on this session_state value (no value=).
    if "theme_dark" not in st.session_state:
        qp = st.query_params.get("theme")
        st.session_state["theme_dark"] = (
            (qp == "dark") if qp in ("dark", "light") else THEME_DEFAULT_DARK
        )
    st.markdown(THEME_BASE_CSS, unsafe_allow_html=True)
    if not st.session_state.get("theme_dark", THEME_DEFAULT_DARK):
        st.markdown(THEME_LIGHT_CSS, unsafe_allow_html=True)
    import streamlit.components.v1 as _components
    _components.html(_THEME_PERSIST_SCRIPT, height=0)


def theme_fig(fig):
    """Make a Plotly figure follow the active theme.

    Charts render server-side and can't read CSS variables, so we set the
    transparent canvas + the axis font/grid colours from the current mode here.
    Call it inline at the plot site: ``st.plotly_chart(theme_fig(fig), ...)``.
    A no-op-equivalent in dark (same white/► values the charts already used);
    in light it swaps to dark-on-white axes + faint dark gridlines. Series and
    on-bar label colours are left to the chart (saturated accents read on both).
    """
    dark = st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
    # Hover tooltip: Plotly's default is a light box with light text, which is
    # unreadable in dark mode. Pin an explicit, theme-matched hoverlabel in BOTH
    # modes (dark panel + light text on dark; light panel + dark text on light).
    try:
        if dark:
            fig.update_layout(hoverlabel=dict(
                bgcolor="#1a1a2e", bordercolor="#2c2c40",
                font=dict(color="#e8e8f0")))
        else:
            fig.update_layout(hoverlabel=dict(
                bgcolor="#ffffff", bordercolor="rgba(20,22,40,0.18)",
                font=dict(color="#14142a")))
    except Exception:
        pass
    if dark:
        # Charts are authored for the dark canvas, but a few don't set every text
        # colour (e.g. the Plotly title can render dark-on-dark — see the Team
        # Analysis bar). Force all chart text to a light colour so nothing is
        # invisible: global font, title, axis titles/ticks, legend, colorbar.
        _LT = "#e8e8f0"
        try:
            fig.update_layout(font_color=_LT,
                              legend_font_color=_LT, legend_title_font_color=_LT)
            # Only touch the title when one exists: setting title_font on a
            # title-less figure creates {font:...} with no text, which
            # plotly.js renders as the literal string "undefined".
            if getattr(fig.layout.title, "text", None):
                fig.update_layout(title_font_color=_LT)
            for axis in (fig.update_xaxes, fig.update_yaxes):
                axis(color=_LT, tickfont_color=_LT, title_font_color=_LT)
            # Colorbars live on each trace's marker (px.bar/px.scatter) — recolour
            # their tick + title text too.
            for tr in fig.data:
                try:
                    cb = getattr(getattr(tr, "marker", None), "colorbar", None)
                    # graph_objects materialize attributes on ASSIGNMENT, so only
                    # touch colorbars that already carry properties - writing to a
                    # pristine one creates it and a phantom colorbar renders.
                    if cb is not None and cb.to_plotly_json():
                        cb.tickfont = dict(color=_LT)
                        if getattr(cb, "title", None) is not None:
                            cb.title.font = dict(color=_LT)
                except Exception:
                    pass
            # Same trap: update_coloraxes on a figure with no coloraxis
            # CREATES one, conjuring a phantom default colorbar.
            if fig.layout.coloraxis.to_plotly_json():
                fig.update_coloraxes(colorbar_tickfont_color=_LT,
                                     colorbar_title_font_color=_LT)
            for ann in (fig.layout.annotations or []):
                try:
                    if ann.font is None or not ann.font.color:
                        ann.font.color = _LT
                except Exception:
                    pass
        except Exception:
            pass
        return fig
    font, grid = "#3a3d48", "rgba(20,22,40,0.10)"
    try:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=font,
        )
        # color= retints the whole axis (line+ticks+labels); tickfont_color
        # beats any explicit light tick colour the chart baked in for dark.
        for axis in (fig.update_xaxes, fig.update_yaxes):
            axis(gridcolor=grid, zerolinecolor=grid, color=font,
                 tickfont_color=font, title_font_color=font)
        # Bright golds wash out on white — swap chart line/marker/annotation
        # yellows for a dark amber that reads on a light background.
        # (includes the CSS named "yellow", not just hex.)
        YELLOW = {"#f1c40f", "#ecbe1a", "#e3b121", "#f0b35b", "yellow"}
        GOLD_LT = "#a87400"
        # Dark marker outlines (picked for dark mode) read as black rings on
        # white — drop them to a white outline so the dots are clean.
        # (includes the CSS named "black", not just hex.)
        DARK_OUTLINE = {"#14142a", "#1a1a2e", "#0a0a14", "#15171d", "#2a2a2a",
                        "#000000", "#000", "black"}
        # Peak-season stars are authored white-fill-on-dark; on the pale canvas
        # a white star (whose outline we've also just whitened) disappears.
        # Invert them — dark navy fill + white outline — mirroring dark mode.
        WHITE_FILL = {"white", "#fff", "#ffffff"}
        STAR_FILL_LT = "#14142a"

        def _fix(c):
            if not isinstance(c, str):
                return c
            cl = c.lower().replace(" ", "")
            if cl in YELLOW:
                return GOLD_LT
            # rgb()/rgba() bright yellow (e.g. the translucent career-arc line)
            if "241,196,15" in cl:
                return "rgba(168,116,0,0.75)"
            return c
        for tr in fig.data:
            try:
                if getattr(tr, "line", None) is not None and getattr(tr.line, "color", None):
                    tr.line.color = _fix(tr.line.color)
                if getattr(tr, "marker", None) is not None and getattr(tr.marker, "color", None):
                    tr.marker.color = _fix(tr.marker.color)
                _ml = getattr(tr.marker, "line", None) if getattr(tr, "marker", None) is not None else None
                if _ml is not None and getattr(_ml, "color", None):
                    if str(_ml.color).lower().replace(" ", "") in DARK_OUTLINE:
                        _ml.color = "#ffffff"
                # A white-filled star marker would vanish on the light canvas
                # (its outline is now white too). Flip it to a dark navy fill so
                # the peak reads — the inverse of the dark-mode white star.
                _sym = getattr(tr.marker, "symbol", None) if getattr(tr, "marker", None) is not None else None
                if _sym is not None and "star" in str(_sym):
                    if str(getattr(tr.marker, "color", "")).lower().replace(" ", "") in WHITE_FILL:
                        tr.marker.color = STAR_FILL_LT
                    if _ml is not None:
                        _ml.color = "#ffffff"
                        if not getattr(_ml, "width", None):
                            _ml.width = 1.5
                if getattr(tr, "textfont", None) is not None and getattr(tr.textfont, "color", None):
                    tr.textfont.color = _fix(tr.textfont.color)
            except Exception:
                pass
        for ann in (fig.layout.annotations or []):
            try:
                if ann.font and ann.font.color:
                    ann.font.color = _fix(ann.font.color)
            except Exception:
                pass
    except Exception:
        pass
    return fig


def value_color(v, vmin, vmax):
    """Red (career-low) → gold (mid) → green (career-high) for a score ``v``
    within ``[vmin, vmax]``. Returns a ``"rgb(r,g,b)"`` string."""
    if vmax <= vmin:
        return "#f1c40f"
    t = (v - vmin) / (vmax - vmin)
    if t < 0.5:
        (r1, g1, b1), (r2, g2, b2), f = (0xe7, 0x4c, 0x3c), (0xf1, 0xc4, 0x0f), t * 2
    else:
        (r1, g1, b1), (r2, g2, b2), f = (0xf1, 0xc4, 0x0f), (0x2e, 0xcc, 0x71), (t - 0.5) * 2
    return "rgb(%d,%d,%d)" % (
        int(r1 + (r2 - r1) * f), int(g1 + (g2 - g1) * f), int(b1 + (b2 - b1) * f))


# Absolute Barrett-Score tiers, one colour per 10-point band. Unlike
# value_color (relative to a single player's own range), these are fixed — so a
# 45 reads the same colour on every chart, making players comparable by colour.
_TIER_COLORS = [
    "#e74c3c",  # 0–10   red
    "#ef7e3b",  # 10–20  orange
    "#f0b429",  # 20–30  amber
    "#a3c644",  # 30–40  lime
    "#2ecc71",  # 40–50  green
    "#16a085",  # 50+    teal
]
_TIER_LABELS = ["0–10", "10–20", "20–30", "30–40", "40–50", "50+"]


def tier_color(score):
    """Absolute colour for a Barrett Score, bucketed into 10-point tiers
    (0–10, 10–20, … 50+). See ``_TIER_COLORS``."""
    try:
        idx = int(float(score) // 10)
    except (TypeError, ValueError):
        return _TIER_COLORS[0]
    return _TIER_COLORS[max(0, min(idx, len(_TIER_COLORS) - 1))]


def _rgb_tuple(c):
    """Parse ``"#rrggbb"`` or ``"rgb(r,g,b)"`` / ``"rgba(...)"`` → (r, g, b)."""
    c = str(c).strip()
    if c.startswith("#"):
        c = c[1:]
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
    inside = c[c.find("(") + 1:c.find(")")]
    p = inside.split(",")
    return (int(float(p[0])), int(float(p[1])), int(float(p[2])))


def gradient_points(x_idx, y_vals, colors, sub=60):
    """Dense (xs, ys, css_colours) interpolating position *and* colour between
    each consecutive node, for drawing a gradient 'line' as a fine marker trace.

    A marker trace (not a line) is used on purpose: theme_fig leaves
    ``marker.color`` *arrays* untouched, so the value palette survives the
    light-mode yellow-swap (a single-colour line would get its gold rewritten).
    """
    xs, ys, cs = [], [], []
    n = len(x_idx)
    for i in range(n - 1):
        c0, c1 = _rgb_tuple(colors[i]), _rgb_tuple(colors[i + 1])
        last = i == n - 2
        for s in range(sub + (1 if last else 0)):
            t = s / sub
            xs.append(x_idx[i] + (x_idx[i + 1] - x_idx[i]) * t)
            ys.append(y_vals[i] + (y_vals[i + 1] - y_vals[i]) * t)
            cs.append("rgb(%d,%d,%d)" % (
                round(c0[0] + (c1[0] - c0[0]) * t),
                round(c0[1] + (c1[1] - c0[1]) * t),
                round(c0[2] + (c1[2] - c0[2]) * t)))
    return xs, ys, cs


def _toggle_theme() -> None:
    """Flip the active theme and mirror it into ?theme=... (survives navigation)."""
    new_dark = not st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
    st.session_state["theme_dark"] = new_dark
    st.query_params["theme"] = "dark" if new_dark else "light"


def render_theme_toggle() -> bool:
    """Brightness button in the nav, backed by st.session_state['theme_dark']:
    a moon in light mode (click → dark) and a sun in dark mode (click → light).

    on_click mirrors the choice into ?theme=... so it persists when you navigate
    between tabs (each tab is a fresh full-reload session). inject_theme() has
    already initialised st.session_state['theme_dark']. Returns True when dark.
    """
    dark = st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
    st.button(
        "",
        key="theme_btn",
        help="Switch between the light and dark themes.",
        on_click=_toggle_theme,
        icon=":material/light_mode:" if dark else ":material/dark_mode:",
    )
    return st.session_state.get("theme_dark", THEME_DEFAULT_DARK)


# ── Themed, sortable HTML table ─────────────────────────────────────────────
# Native st.dataframe is drawn on a canvas locked to config.toml's theme, so it
# can't follow the runtime light/dark toggle. html_table() renders a real HTML
# table that themes via the same tokens, with delegated JS click-to-sort.
HV_TABLE_CSS = """
<style>
.hv-table-wrap{overflow:auto;border:1px solid var(--panel-line);border-radius:10px;
  background:var(--panel-solid);box-shadow:var(--shadow-card);
  margin:0.7rem 0 1.7rem;display:block;}
table.hv-table{width:100%;border-collapse:collapse;font-size:0.85rem;
  font-variant-numeric:tabular-nums;}
.hv-table thead th{position:sticky;top:0;z-index:1;background:var(--panel-2);
  color:var(--fg-4);text-transform:uppercase;font-size:0.68rem;letter-spacing:0.04em;
  font-weight:700;padding:0.6rem 0.7rem;cursor:pointer;white-space:nowrap;
  border-bottom:2px solid var(--panel-line);user-select:none;}
.hv-table thead th:hover{color:var(--fg-2);}
.hv-table thead th:focus-visible{outline:2px solid var(--accent-red);outline-offset:-2px;}
.hv-table thead th .sort-ind{margin-left:0.15em;font-size:0.85em;color:var(--accent-red);}
.hv-table tbody td{padding:0.5rem 0.7rem;color:var(--fg-2);
  border-bottom:1px solid var(--hairline-soft);white-space:nowrap;}
.hv-table tbody tr:nth-child(even) td{background:var(--row-tint);}
.hv-table tbody tr:hover td{background:var(--panel-hover);}  /* must follow zebra */
.hv-table tbody tr:last-child td{border-bottom:none;}
/* Status / label chips inside table cells (and reused by page banners). */
.hv-chip{display:inline-block;padding:0.05rem 0.45rem;border-radius:999px;
  font-size:0.64rem;font-weight:800;letter-spacing:0.04em;border:1px solid currentColor;
  white-space:nowrap;vertical-align:middle;}
.hv-chip.ufa{color:var(--gold);}
.hv-chip.rfa{color:var(--sky);}
.hv-chip.po{color:var(--purple);}
.hv-chip.to{color:var(--orange);}
.hv-chip.signed,.hv-chip.muted{color:var(--fg-5);border-color:var(--hairline);}
.hv-chip.max{color:var(--gold);border-radius:4px;font-size:0.6rem;
  padding:0 0.3rem;margin-right:0.35rem;}
/* Team-color dot (per-team background classes are emitted by the page). */
.tdot{display:inline-block;width:8px;height:8px;border-radius:50%;
  margin-right:0.4rem;background:var(--fg-5);vertical-align:baseline;}
</style>
"""

_HV_SORT_SCRIPT = """
<script>
(function () {
  var doc = window.parent.document;
  if (doc.__hvSortInit) return;
  doc.__hvSortInit = true;
  function hvSort(th) {
    var tr = th.parentNode, table = th.closest('table'), tbody = table.querySelector('tbody');
    var idx = Array.prototype.indexOf.call(tr.children, th);
    var numeric = th.hasAttribute('data-num');
    var asc = th.getAttribute('data-dir') !== 'asc';
    Array.prototype.forEach.call(tr.children, function (o) {
      o.removeAttribute('data-dir');
      o.setAttribute('aria-sort', 'none');
      var s = o.querySelector('.sort-ind'); if (s) s.textContent = '';
    });
    th.setAttribute('data-dir', asc ? 'asc' : 'desc');
    th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
    var ind = th.querySelector('.sort-ind'); if (ind) ind.textContent = asc ? ' ▲' : ' ▼';
    var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
    rows.sort(function (a, b) {
      var av = a.children[idx].getAttribute('data-v'), bv = b.children[idx].getAttribute('data-v');
      if (numeric) {
        av = parseFloat(av); bv = parseFloat(bv);
        if (isNaN(av)) av = -Infinity; if (isNaN(bv)) bv = -Infinity;
        return asc ? av - bv : bv - av;
      }
      av = (av || '').toLowerCase(); bv = (bv || '').toLowerCase();
      return av < bv ? (asc ? -1 : 1) : av > bv ? (asc ? 1 : -1) : 0;
    });
    rows.forEach(function (r) { tbody.appendChild(r); });
  }
  doc.addEventListener('click', function (e) {
    var th = e.target.closest ? e.target.closest('.hv-table th[data-sortable]') : null;
    if (!th) return;
    hvSort(th);
  });
  doc.addEventListener('keydown', function (e) {
    if (e.key !== 'Enter' && e.key !== ' ' && e.key !== 'Spacebar') return;
    var th = e.target.closest ? e.target.closest('.hv-table th[data-sortable]') : null;
    if (!th) return;
    if (e.key !== 'Enter') e.preventDefault();  /* Space must not scroll the page */
    hvSort(th);
  });
})();
</script>
"""


def html_table(df, *, formatters=None, styles=None, aligns=None,
               numeric=None, helps=None, height=560, row_style=None,
               max_rows=1000, raw=None):
    """Render a DataFrame as a themed, sortable HTML table (follows light/dark).

    formatters: {col: value -> display str}          (default str(value))
    styles:     {col: (value, row_dict) -> css str}  e.g. 'color:var(--value-good)'
    aligns:     {col: 'left'|'right'|'center'}        (default left)
    numeric:    iterable of cols sorted by raw numeric value
    helps:      {col: tooltip text}                   (header title=)
    row_style:  row_dict -> css str for the <tr> (e.g. peak-season highlight)
    max_rows:   cap rendered rows (native grids virtualise; raw HTML doesn't, so
                very long tables would bloat the DOM). A note row flags the cap.
    raw:        iterable of cols whose FORMATTED value is trusted HTML and rendered
                unescaped (e.g. an <a> player link built by our own formatter —
                never user input). All other columns stay escaped.
    """
    import html
    formatters = formatters or {}
    styles = styles or {}
    aligns = aligns or {}
    numeric = set(numeric or [])
    helps = helps or {}
    raw = set(raw or [])
    cols = list(df.columns)

    head = []
    for c in cols:
        al = aligns.get(c, "left")
        num = ' data-num="1"' if c in numeric else ""
        tip = f' title="{html.escape(str(helps[c]), quote=True)}"' if c in helps else ""
        head.append(
            f'<th data-sortable="1" tabindex="0" scope="col" aria-sort="none"'
            f'{num}{tip} style="text-align:{al}">'
            f'{html.escape(str(c))}<span class="sort-ind"></span></th>'
        )
    n_total = len(df)
    df_view = df.head(max_rows) if (max_rows and n_total > max_rows) else df
    rows_html = []
    for _, row in df_view.iterrows():
        rd = row.to_dict()
        tds = []
        for c in cols:
            v = rd[c]
            # NaN-/type-safe: a formatter raising on one cell (e.g. int(NaN) on
            # sparse historical data) must not blow up the whole table.
            if c in formatters:
                try:
                    _fmt = formatters[c]
                    # Row-aware formatters: a 2-arg callable gets (value, row_dict) —
                    # lets display text depend on other columns (e.g. "(Max)" prefix)
                    # without affecting the numeric sort value.
                    if getattr(_fmt, "__code__", None) is not None and _fmt.__code__.co_argcount >= 2:
                        disp = _fmt(v, rd)
                    else:
                        disp = _fmt(v)
                except Exception:
                    disp = ""
            elif v is None or (isinstance(v, float) and v != v):
                disp = ""
            else:
                disp = str(v)
            al = aligns.get(c, "left")
            stl = styles[c](v, rd) if c in styles else ""
            sv = v if c in numeric else disp
            extra = f";{stl}" if stl else ""
            cell = disp if c in raw else html.escape(disp)
            tds.append(
                f'<td data-v="{html.escape(str(sv), quote=True)}" '
                f'style="text-align:{al}{extra}">{cell}</td>'
            )
        rstyle = row_style(rd) if row_style else ""
        tr_open = f'<tr style="{rstyle}">' if rstyle else "<tr>"
        rows_html.append(tr_open + "".join(tds) + "</tr>")
    if max_rows and n_total > max_rows:
        rows_html.append(
            f'<tr><td colspan="{len(cols)}" style="text-align:center;'
            f'color:var(--fg-5);font-style:italic;padding:0.6rem">'
            f'Showing top {max_rows:,} of {n_total:,} rows</td></tr>'
        )
    st.markdown(
        f'<div class="hv-table-wrap" style="max-height:{height}px">'
        f'<table class="hv-table"><thead><tr>{"".join(head)}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def stat_cards(items):
    """Render a row of colour-accented summary stat cards (themed light/dark).

    items: list of (label, value, color_css[, sub]) tuples, where color_css is
    a token like 'var(--accent-teal)'. Optional 4th element is a small sub-line.
    """
    import html
    out = []
    for it in items:
        label, value, color = it[0], it[1], it[2]
        sub = it[3] if len(it) > 3 else ""
        sub_html = f'<div class="hv-stat-s">{html.escape(str(sub))}</div>' if sub else ""
        out.append(
            f'<div class="hv-stat" style="--c:{color}">'
            f'<div class="hv-stat-v">{html.escape(str(value))}</div>'
            f'<div class="hv-stat-l">{html.escape(str(label))}</div>{sub_html}</div>'
        )
    st.markdown(f'<div class="hv-stats">{"".join(out)}</div>', unsafe_allow_html=True)


COMMON_CSS = """
<style>
    /* Theme tokens (:root) are injected separately by inject_theme() so the
       home page (which doesn't use this stylesheet) shares one source of
       truth. COMMON_CSS just *references* the tokens (var(--...)). */

    /* Trim the page side padding. Streamlit's default is a hefty 5rem each
       side, which wastes width on wide screens; .main no longer matches the
       1.51 DOM, so target the testid too. (Top padding is handled below.) */
    .main .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 100%;
    }
    /* A touch more vertical breathing room between stacked elements (the
       toggle wrappers keep their own gap:0 !important and are unaffected). */
    [data-testid="stVerticalBlock"] { gap: 1.7rem; }
    /* The expander is a bordered "box" with no margin of its own, so whatever
       follows (e.g. the Season selector) sits flush against it. Give it clear
       separation below. */
    [data-testid="stExpander"] { margin-bottom: 1.6rem; }
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
        background: var(--bg-nav);
        border-bottom: 1px solid var(--nav-border);
        flex-wrap: nowrap;
    }
    .top-nav a {
        text-decoration: none;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--fg-3);
        border: 1px solid transparent;
        transition: all 0.15s;
        white-space: nowrap;
    }
    .top-nav a:hover { border-color: var(--accent-red); color: var(--fg-1); text-decoration: none; }
    .top-nav a.active { background: var(--accent-red); border-color: var(--accent-red); color: #fff; }
    .top-nav .home-link {
        color: var(--fg-6);
        font-size: 0.82rem;
        font-weight: 500;
        padding: 0.3rem 0.7rem;
        margin-right: 0.25rem;
        border: none;
    }
    .top-nav .home-link:hover { color: var(--fg-1); border: none; }
    .top-nav .divider { color: var(--nav-divider); font-size: 0.75rem; margin: 0 0.1rem; user-select: none; }

    /* ── Responsive nav: keep the links clear of the pinned brightness button ──
       Only the theme button is pinned (position:fixed) top-right; reserve its
       narrow width so the tabs don't slide under it, and shrink the tabs as the
       viewport narrows. Playoff mode is rendered in-page, not pinned here. */
    .top-nav { padding-right: 3.5rem; }
    @media (max-width: 1100px) {
        .top-nav a, .top-nav .home-link {
            padding-left: 0.5rem; padding-right: 0.5rem; font-size: 0.78rem;
        }
    }
    @media (max-width: 940px) {
        .top-nav .divider { display: none; }
        .top-nav a, .top-nav .home-link {
            padding-left: 0.4rem; padding-right: 0.4rem; font-size: 0.73rem;
        }
    }
    @media (max-width: 870px) {
        .top-nav a, .top-nav .home-link {
            padding-left: 0.3rem; padding-right: 0.3rem; font-size: 0.67rem;
        }
    }
    /* Below ~760px even the shrunken links overflow and the rightmost ones
       clipped under the fixed theme button. Let the bar scroll sideways
       (hidden scrollbar) instead; the flex end-spacer guarantees the last
       link can scroll clear of the toggle even where a scroll container's
       trailing padding isn't honoured. */
    @media (max-width: 760px) {
        .top-nav {
            overflow-x: auto;
            overflow-y: hidden;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
            padding-right: 4rem;
        }
        .top-nav::-webkit-scrollbar { display: none; }
        .top-nav::after { content: ""; flex: 0 0 3.25rem; }
    }

    /* Pin the playoff-mode toggle to the top-right of the nav, just left of the
       brightness button. Only rendered (in an st.container(key=...)) on pages
       where playoff mode changes the content, so it's out of the way but handy. */
    .st-key-playoff_nav_toggle {
        position: fixed !important;
        top: 0.5rem !important;
        right: 3.75rem !important;
        z-index: 10001 !important;
        margin: 0 !important;
        padding: 0 !important;
        width: auto !important;
        background: transparent !important;
    }
    .st-key-playoff_nav_toggle [data-testid="stToggle"] {
        background: transparent;
        padding: 0;
    }
    /* Toggle label colour to match nav text */
    .st-key-playoff_nav_toggle label p {
        color: var(--fg-3) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    .st-key-playoff_nav_toggle:hover label p { color: var(--fg-1) !important; }

    /* Theme (brightness) button, pinned to the far top-right of the nav,
       vertically centered within the 3rem-tall bar. */
    .st-key-theme_nav_toggle {
        position: fixed !important;
        top: 0 !important;
        height: 3rem !important;
        right: 1rem !important;
        z-index: 10001 !important;
        margin: 0 !important;
        padding: 0 !important;
        width: auto !important;
        background: transparent !important;
        display: flex !important;
        align-items: center !important;
    }
    /* Brightness icon button (moon in light, sun in dark), strip Streamlit's
       button chrome down to just the icon. */
    .st-key-theme_nav_toggle button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 0.3rem !important;
        /* Full nav height + flex-center the icon, so it's vertically centered
           regardless of the wrapper chain (container align-items alone left the
           glyph pinned to the top). */
        height: 3rem !important;
        min-height: 3rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 1 !important;
        font-size: 1.3rem !important;
        color: var(--fg-3) !important;
    }
    .st-key-theme_nav_toggle button [data-testid="stIconMaterial"] {
        font-size: 1.3rem !important;
    }
    .st-key-theme_nav_toggle button:hover { color: var(--fg-1) !important; background: transparent !important; }
    .st-key-theme_nav_toggle button:active,
    .st-key-theme_nav_toggle button:focus,
    .st-key-theme_nav_toggle button:focus-visible {
        box-shadow: none !important; background: transparent !important; color: var(--fg-1) !important;
    }

    /* In-page Playoff-mode toggle (on the title row): push it to the right edge
       of its column so it lines up under the brightness button. The toggle's
       wrapper gets class .st-key-playoff_mode from its widget key. */
    .st-key-playoff_mode {
        display: flex !important;
        justify-content: flex-end !important;
    }

    /* Make the toggle's wrapper hierarchy vanish from layout entirely.
       display:contents removes an element from the box tree (its children
       render in its place); since our only child is position:fixed, no
       flow space is claimed. Apply to every ancestor that might wrap the
       keyed container so it doesn't matter which one Streamlit creates. */
    .element-container:has(.st-key-playoff_nav_toggle),
    [data-testid="element-container"]:has(.st-key-playoff_nav_toggle),
    [data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-playoff_nav_toggle),
    [data-testid="stVerticalBlock"]:has(.st-key-playoff_nav_toggle):not(.st-key-playoff_nav_toggle),
    .element-container:has(.st-key-theme_nav_toggle),
    [data-testid="element-container"]:has(.st-key-theme_nav_toggle),
    [data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-theme_nav_toggle),
    [data-testid="stVerticalBlock"]:has(.st-key-theme_nav_toggle):not(.st-key-theme_nav_toggle) {
        display: contents !important;
    }

    /* Belt-and-suspenders: zero out any remaining wrapper dimensions in
       case display:contents isn't enough on some browser. */
    .element-container:has(.st-key-playoff_nav_toggle),
    [data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-playoff_nav_toggle),
    [data-testid="stVerticalBlock"]:has(.st-key-playoff_nav_toggle):not(.st-key-playoff_nav_toggle),
    .element-container:has(.st-key-theme_nav_toggle),
    [data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-theme_nav_toggle),
    [data-testid="stVerticalBlock"]:has(.st-key-theme_nav_toggle):not(.st-key-theme_nav_toggle) {
        min-height: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
    }

    /* Page block-container top padding, fits the fixed nav bar (3rem tall)
       plus breathing room before the page title. 5.5rem leaves a clean ~2.5rem
       gap between the nav and the title. */
    .main .block-container,
    section.main > .block-container,
    [data-testid="stMain"] .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-top: 5.5rem !important;
    }

    /* Same trick for the components.html hide-badge iframe, height=0 in
       the call but Streamlit's component wrapper still claims default
       space. Narrow these selectors to height="0" iframes only so other
       components.html embeds (like the Search "Share this view" widget)
       still render. */
    [data-testid="stCustomComponentV1"]:has(iframe[height="0"]),
    [data-testid="stCustomComponentV1"] iframe[height="0"],
    [data-testid="stIFrame"] iframe[height="0"] {
        display: none !important;
    }
    .element-container:has(iframe[height="0"]) {
        display: contents !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* ── Global typographic rhythm ────────────────────────────────────────
       Adds consistent breathing room sitewide between page titles, their
       captions, and the content blocks below. Without these, st.title sits
       flush against the next widget/section and st.caption sits flush
       against the title above it. */
    [data-testid="stHeading"]:has(h1) {
        margin-bottom: 0.85rem !important;
    }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaption"] {
        margin-top: 0.45rem !important;
        margin-bottom: 0.7rem !important;
    }
    /* If a caption sits directly above a divider, tighten the gap below
       the caption so the divider hugs the line above it (not the section
       below). */
    [data-testid="stCaptionContainer"] + hr,
    [data-testid="stCaption"]          + hr {
        margin-top: 0.35rem !important;
    }
    /* Info / success / warning / error alerts get a touch of breathing
       room from whatever widget they sit beneath (e.g. the "X players
       indexed..." info box below the player multiselect on Search). */
    [data-testid="stAlert"],
    [data-testid="stAlertContainer"] {
        margin-top: 0.85rem !important;
    }

    /* Colour-accented stat cards (utils.stat_cards), themed summary metrics. */
    .hv-stats { display:flex; gap:0.7rem; flex-wrap:wrap; margin:0.3rem 0 0; padding:1rem 0 1.7rem; }
    .hv-stat {
        flex:1 1 0; min-width:120px; background:var(--panel-solid);
        border:1px solid var(--panel-line); border-top:3px solid var(--c);
        border-radius:10px; padding:0.85rem 0.7rem 0.75rem; text-align:center;
        box-shadow:var(--shadow-card); transition:transform .12s ease;
    }
    .hv-stat:hover { transform:translateY(-2px); }
    .hv-stat-v { font-size:1.7rem; font-weight:800; line-height:1.05; color:var(--c); }
    .hv-stat-l { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.05em;
                 color:var(--fg-4); margin-top:0.4rem; font-weight:600; }
    .hv-stat-s { font-size:0.7rem; color:var(--fg-5); margin-top:0.2rem; }

    /* st.tabs labels, Streamlit's default inactive colour is a faint grey that
       disappears on the dark background. Bind to theme tokens so the inactive
       tabs stay legible (mid-grey) and the active one takes the brand red on
       both themes. Targets the tab buttons + their inner text node. */
    .stTabs [data-baseweb="tab-list"] button[role="tab"],
    .stTabs [data-baseweb="tab-list"] button[role="tab"] p {
        color: var(--fg-3) !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-list"] button[role="tab"]:hover,
    .stTabs [data-baseweb="tab-list"] button[role="tab"]:hover p {
        color: var(--fg-1) !important;
    }
    .stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] p {
        color: var(--accent-red) !important;
    }
    /* The active-tab underline highlight follows the brand red too. */
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent-red) !important; }
</style>
"""

_NAV_PAGES = [
    ("Current Rankings",   "/Rankings"),
    ("Compare Players",    "/Search"),
    ("Legacy",             "/Legacy"),
    ("Team Analysis",      "/Team_Analysis"),
    # Trades tab removed — page lives at /Trades_disabled.py (kept for
    # easy revival) and a backup of the verdict-aware version is in
    # Trades_backup.py at repo root.
    ("Contract Predictor", "/Contract_Predictor"),
    # Team Builder (Front Office) tab removed from users' view — page lives at
    # /Team_Builder_disabled.py (kept at repo root for easy revival). Its model
    # also lives inside Contract Predictor's team mode. Restore by moving the file
    # back into pages/ as Team_Builder.py and re-adding the nav entry below.
    # ("Team Builder",       "/Team_Builder"),
    # Free Agency Sim tab removed from users' view — page lives at
    # /Free_Agency_Simulation_disabled.py (kept at repo root for easy revival).
    # Restore by moving the file back into pages/ as Free_Agency_Simulation.py
    # and re-adding the nav entry below.
    # ("Free Agency Sim",    "/Free_Agency_Simulation"),
    ("Current Free Agents", "/Free_Agent_Class"),
    # Track Record tab removed — page lives at /Track_Record_disabled.py
    # (kept at repo root for easy revival). Restore by moving the file
    # back into pages/ as Track_Record.py and re-adding the nav entry below.
    # ("Track Record",       "/Track_Record"),
]

_PLAYOFF_HELP = (
    "Replace regular-season stats with postseason stats for the selected "
    "season(s). Salaries stay the same (one annual contract). Defense uses "
    "box-score fallback. Availability is based on each team's depth-of-run, "
    "so Finals MVPs outrank first-round stars with similar per-game production."
)

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT-INTERNALS SELECTOR INDEX — check these after any Streamlit upgrade.
# The site's chrome reaches into Streamlit's private DOM (built against 1.51);
# none of these are public API and any release can break them. One checklist:
#
#   1. stLayoutWrapper :has() container targeting (app.py) — the homepage hub
#      quadrants and board pills are styled via
#      [data-testid="stLayoutWrapper"]:has(> .st-key-hub_q*) and
#      [data-testid="stVerticalBlock"]:has(> [data-testid="stLayoutWrapper"]
#      .st-key-board_view); 1.51 nests columns/fragments under stLayoutWrapper,
#      so if that wrapper is renamed or the nesting changes, the hub cards and
#      pills-to-table spacing silently lose their styling.
#   2. button[kind="pillsActive"] (app.py) — the active board-view pill keys off
#      the private `kind` attribute Streamlit stamps on st.pills buttons.
#   3. The hidden "Made with Streamlit" badge hack — three layers must all keep
#      working: the COMMON_CSS display:none rules ([data-testid=
#      "stAppViewerBadge"] / stBottom / stToolbar / stDecoration / ...), the
#      _HIDE_BADGE_SCRIPT MutationObserver below (removes [class*="viewerBadge"]
#      nodes — HASHED CSS-module class names, matched by prefix only), and the
#      height-0 iframe collapse rules in COMMON_CSS ([data-testid=
#      "stCustomComponentV1"]:has(iframe[height="0"]) etc.) that keep the badge/
#      sort/face-guard component iframes from leaving visible gaps.
#   4. data-testid + .st-key-* hooks throughout THEME_BASE_CSS / COMMON_CSS —
#      stMainBlockContainer, stVerticalBlock, stSidebar, stWidgetLabel, the
#      baseweb select/input internals, stTabs internals, and the keyed-container
#      pinning (.st-key-theme_nav_toggle / .st-key-playoff_nav_toggle). The
#      st.container(key=...) class is stable API, but the wrapper chain these
#      rules style against is not.
#
# After upgrading Streamlit: load the homepage + one subpage and verify (a) no
# sidebar/toolbar/badge flash, (b) hub quadrant cards styled, (c) board pills
# styled + active pill highlighted, (d) theme + playoff toggles pinned top-right,
# (e) tables still click-to-sort, (f) no stray empty iframe gaps.
# ══════════════════════════════════════════════════════════════════════════════
_HIDE_BADGE_SCRIPT = """
<script>
    function hideBadge() {
        try {
            const doc = window.parent.document;
            [
                '[data-testid="stAppViewerBadge"]',
                '[data-testid="stBottom"]',
                '[data-testid="stToolbar"]',
                '[data-testid="stStatusWidget"]',
                '[class*="viewerBadge"]',
                '[class*="ViewerBadge"]',
            ].forEach(sel => doc.querySelectorAll(sel).forEach(el => el.remove()));
        } catch(e) {}
    }
    hideBadge();
    new MutationObserver(hideBadge).observe(document.documentElement, { childList: true, subtree: true });
</script>
"""


# ══════════════════════════════════════════════════════════════════════════════
# HV visual kit — shared by the homepage and all pages (rails, team colors,
# headshots, sparklines). Emitted site-wide by render_page_chrome().
# ══════════════════════════════════════════════════════════════════════════════

# Team accent colors: rails, dots, rings, and low-alpha washes ONLY (never text,
# never a full background). Brightest recognizable color per team so an 8px dot
# reads on both the dark and light panel surfaces (navy/black brands swapped for
# their secondary: BKN/SAS silver, DEN/IND/NOP/UTA gold, MIN lake blue).
TEAM_HEX = {
    "ATL": "#E03A3E", "BOS": "#007A33", "BKN": "#8A8D8F", "CHA": "#00788C", "CHI": "#CE1141",
    "CLE": "#A31D3C", "DAL": "#0064B1", "DEN": "#FEC524", "DET": "#C8102E", "GSW": "#1D428A",
    "HOU": "#CE1141", "IND": "#FDBB30", "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9",
    "MIA": "#B62435", "MIL": "#1E7A44", "MIN": "#236192", "NOP": "#B4975A", "NYK": "#F58426",
    "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#E56020", "POR": "#E03A3E",
    "SAC": "#5A2D81", "SAS": "#8A8D8F", "TOR": "#CE1141", "UTA": "#F9A01B", "WAS": "#E31837",
}


def hex_rgba(h: str, alpha: float) -> str:
    """'#RRGGBB' -> 'rgba(r,g,b,a)'. Plotly and per-player CSS can't read theme
    tokens, so team tints are computed server-side from TEAM_HEX."""
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def hex_darken(h: str, f: float = 0.72) -> str:
    h = h.lstrip("#")
    return "#" + "".join(f"{int(int(h[i:i + 2], 16) * f):02x}" for i in (0, 2, 4))


def hex_is_light(h: str) -> bool:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) > 175


@st.cache_data(show_spinner=False)
def _headshot_id_map() -> dict:
    """normalize(name) -> nba.com player id. nba_api's static players table is a
    bundled offline json (no network); active players win name collisions."""
    try:
        from nba_api.stats.static import players as _np
        m = {}
        for p in _np.get_players():
            k = normalize(p["full_name"])
            if k not in m or p.get("is_active"):
                m[k] = p["id"]
        return m
    except Exception:
        return {}


def face_img(name: str, css_class: str) -> str:
    """Lazy circular headshot <img> from the NBA CDN, or '' when the id is
    unknown. 404s hide themselves via the capture-phase guard in the chrome
    (Streamlit strips inline onerror= attributes)."""
    pid = _headshot_id_map().get(normalize(str(name)))
    if not pid:
        return ""
    return (f'<img class="{css_class}" loading="lazy" decoding="async" '
            f'src="https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png" alt="">')


def spark_svg(scores: list, w: int = 72, h: int = 20) -> str:
    """Tiny inline career-shape polyline."""
    if not scores or len(scores) < 2:
        return ""
    lo, hi = min(scores), max(scores)
    rng = (hi - lo) or 1.0
    pts = " ".join(f"{i / (len(scores) - 1) * (w - 2) + 1:.1f},"
                   f"{h - 2 - (s - lo) / rng * (h - 4):.1f}"
                   for i, s in enumerate(scores))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="vertical-align:middle">'
            f'<polyline points="{pts}" fill="none" stroke="var(--logo-sage)" '
            f'stroke-width="1.5" stroke-linejoin="round"/></svg>')


def render_rail(kicker: str, title: str, count: str | None = None, meta: str | None = None) -> None:
    """Ruled section header: kicker + title + optional count pill + right meta.

    An empty kicker is omitted entirely (no stray gap), so callers can hide the
    eyebrow label by passing "".
    """
    bits = []
    if kicker:
        bits.append(f'<span class="k">{html.escape(kicker)}</span>')
    bits.append(f'<span class="t">{html.escape(title)}</span>')
    if count:
        bits.append(f'<span class="n">{html.escape(count)}</span>')
    if meta:
        bits.append(f'<span class="meta">{html.escape(meta)}</span>')
    st.markdown(f'<div class="hv-rail">{"".join(bits)}</div>', unsafe_allow_html=True)


# Rail headers, mini headshots, and the 30 team-dot colors, site-wide.
HV_KIT_CSS = ("""
<style>
.hv-rail{display:flex;align-items:center;gap:.65rem;margin:0;padding:1.6rem 0 .7rem;}
.hv-rail::before{content:"";width:4px;height:1.05em;background:var(--accent-red);
    border-radius:2px;flex:0 0 auto;}
.hv-rail .k{font-size:.64rem;font-weight:800;letter-spacing:.11em;
    text-transform:uppercase;color:var(--fg-4);white-space:nowrap;}
.hv-rail .t{font-size:1.02rem;font-weight:800;color:var(--fg-1);white-space:nowrap;}
.hv-rail .n{font-size:.72rem;font-weight:700;color:var(--fg-4);background:var(--panel-2);
    border:1px solid var(--panel-line);border-radius:99px;padding:.1rem .55rem;
    font-variant-numeric:tabular-nums;white-space:nowrap;}
.hv-rail .meta{font-size:.72rem;color:var(--fg-4);white-space:nowrap;order:10;}
.hv-rail::after{content:"";flex:1;height:1px;background:var(--hairline);order:9;}
[data-testid="stMarkdownContainer"]:has(.hv-rail){margin-bottom:0 !important;}
.hv-mini-wrap{display:inline-block;width:24px;height:24px;border-radius:50%;
    background:var(--panel-2);margin-right:0.45rem;vertical-align:middle;
    overflow:hidden;flex:0 0 auto;}
img.hv-mini-face{width:24px;height:24px;border-radius:50%;object-fit:cover;
    object-position:center 15%;display:block;}
"""
    + "".join(f".tdot.tdot-{k}{{background:{v}}}" for k, v in TEAM_HEX.items())
    + "</style>")


# Hide headshots whose CDN image 404s. Runs from a components iframe as a
# capture-phase listener on the parent document (img error events don't bubble).
FACE_GUARD_SCRIPT = """
<script>
(function () {
    var doc = window.parent.document;
    if (doc.__hvFaceGuard) return;
    doc.__hvFaceGuard = true;
    doc.addEventListener('error', function (e) {
        var t = e.target;
        if (t && t.tagName === 'IMG' &&
            (t.classList.contains('fp-face') || t.classList.contains('hub-face') ||
             t.classList.contains('hv-mini-face'))) {
            t.style.display = 'none';
        }
    }, true);
})();
</script>
"""


def render_page_chrome() -> None:
    """One-call page chrome: COMMON_CSS + the hide-badge MutationObserver.

    Call this once near the top of every page right after st.set_page_config.
    Replaces ~20 lines of identical components.html + st.markdown boilerplate
    that used to be copy-pasted into every page. Adding new pages? Just call
    render_page_chrome() and you're done, no need to remember which iframe
    script to copy.
    """
    import streamlit.components.v1 as _components
    inject_theme()                       # tokens first — COMMON_CSS references them
    st.markdown(COMMON_CSS, unsafe_allow_html=True)
    st.markdown(HV_TABLE_CSS, unsafe_allow_html=True)
    st.markdown(HV_KIT_CSS, unsafe_allow_html=True)  # rails, mini faces, team dots
    _components.html(_HIDE_BADGE_SCRIPT, height=0)
    _components.html(_HV_SORT_SCRIPT, height=0)      # delegated click-to-sort
    _components.html(FACE_GUARD_SCRIPT, height=0)    # hide 404 headshots


def render_nav(current: str) -> None:
    """Render the top nav bar with the playoff toggle pinned right.

    The toggle widget is rendered inside an st.container(key=...) and CSS
    pulls the resulting `.st-key-playoff_nav_toggle` element into a
    position:fixed slot at the top-right of the viewport (same row as the
    nav links). State is shared via st.session_state.playoff_mode, so every
    page sees the change immediately on its next render.
    """
    links = '<a class="home-link" href="/" target="_top">Home</a><span class="divider">|</span>'
    for label, url in _NAV_PAGES:
        css_class = "active" if label == current else ""
        links += f'<a class="{css_class}" href="{url}" target="_top">{label}</a>'
    st.markdown(f'<div class="top-nav">{links}</div>', unsafe_allow_html=True)

    # Theme button — keyed container, pinned to the top-right via CSS. The
    # playoff toggle is NOT pinned here: it's rendered in-page (render_playoff_
    # toggle) only on pages where playoff mode actually changes the content.
    with st.container(key="theme_nav_toggle"):
        render_theme_toggle()


_FOOTER_X_PATH = ("M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83"
                  "L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z")

_FOOTER_HTML = f"""
<style>
.hv-footer {{ margin-top:3.5rem; padding-top:1.4rem; font-size:0.85rem; }}
.hv-foot-top {{ text-align:center; color:var(--fg-3); margin-bottom:0.5rem; }}
.hv-foot-top a {{ color:var(--amber); text-decoration:underline; font-weight:600; }}
.hv-foot-top a:hover {{ filter:brightness(1.12); }}
.hv-foot-disc {{ text-align:center; color:var(--fg-5); font-size:0.78rem; line-height:1.45;
    max-width:760px; margin:0 auto 1.2rem; }}
.hv-foot-rule {{ border-top:1px solid var(--panel-line); margin:0 0 1rem; }}
.hv-foot-bottom {{ display:flex; justify-content:space-between; align-items:center;
    flex-wrap:wrap; gap:0.5rem 1.1rem; color:var(--fg-5); }}
.hv-foot-bottom a {{ color:var(--fg-3); text-decoration:none; }}
.hv-foot-bottom a:hover {{ color:var(--fg-1); }}
.hv-foot-right {{ display:flex; align-items:center; gap:0.85rem; }}
.hv-foot-right .sep {{ color:var(--panel-line); }}
.hv-foot-ico {{ display:inline-flex; line-height:0; }}
.hv-foot-ico svg {{ width:15px; height:15px; fill:var(--fg-3); transition:fill .12s ease; }}
.hv-foot-ico:hover svg {{ fill:var(--fg-1); }}
</style>
<div class="hv-footer">
  <div class="hv-foot-top">Enjoying HoopsValue? <a href="mailto:contact@hoopsvalue.com">Share feedback</a></div>
  <div class="hv-foot-disc">HoopsValue.com is an independent project and is not affiliated with,
    endorsed by, or sponsored by the National Basketball Association.</div>
  <div class="hv-foot-rule"></div>
  <div class="hv-foot-bottom">
    <div class="hv-foot-left">© 2026 HoopsValue.com. All rights reserved.</div>
    <div class="hv-foot-right">
      <a href="/About" target="_top">About</a>
      <span class="sep">|</span>
      <a href="mailto:contact@hoopsvalue.com">contact@hoopsvalue.com</a>
      <span class="sep">|</span>
      <a href="https://x.com/HoopsValue" target="_blank" rel="noopener">@HoopsValue</a>
      <a class="hv-foot-ico" href="https://x.com/HoopsValue" target="_blank" rel="noopener" aria-label="X">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="{_FOOTER_X_PATH}"/></svg></a>
    </div>
  </div>
</div>
"""


def render_footer() -> None:
    """Site-wide footer, call once at the very bottom of every page (and before
    any st.stop() that ends a normal page state, e.g. the Contract Predictor modes)."""
    st.markdown(_FOOTER_HTML, unsafe_allow_html=True)


def render_playoff_toggle() -> bool:
    """In-page playoff-mode toggle, backed by st.session_state.playoff_mode.

    Rendered near the top of the pages where playoff mode affects the content
    (Rankings, Team Analysis, Search, Legacy), NOT on pages it doesn't touch
    (Contract Predictor, Free Agents, Home). setdefault + no value= avoids the
    "set via Session State" warning. Returns True when playoff mode is on.
    """
    st.session_state.setdefault("playoff_mode", False)
    return st.toggle(
        "Playoff mode",
        key="playoff_mode",
        help=_PLAYOFF_HELP,
    )


def render_barrett_score_explainer() -> None:
    """Drop-in "What is the Barrett Score?" expander used on every page.

    Single source of truth so the copy stays in sync sitewide. Called near
    the top of each page (under the page title / caption) on Rankings,
    Search, Legacy, Team Analysis, and Free Agent Class.
    """
    with st.expander("What is the Barrett Score?"):
        st.markdown(
            "The Barrett Score combines scoring, playmaking, rebounding, defense, efficiency, "
            "and availability into one player value metric. Each player's score is compared "
            "against real NBA contracts by matching the highest scores with the highest "
            "salaries, creating an estimated contract value for every player. The result "
            "shows who is underpaid, overpaid, or being paid roughly in line with their "
            "on-court value."
        )


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
    """Availability multiplier, punishes missed games + light minutes.

    Range: 0.30 → 1.00. Sqrt curve so missing 20% of the season costs ~10%
    rather than a linear 20%, but missing 80% costs much more than the
    previous v4 formula (0.30 floor instead of 0.75). An 18-game season
    at 30 MPG now lands ~0.62 instead of ~0.80.

    Inputs:
      gp:           games played
      total_min:    total minutes played
      season_games: games in this season (66 for 2011-12 lockout,
                    72 for COVID years, 82 otherwise)
    """
    min_cap = season_games * (2500 / 82)
    return 0.30 + 0.70 * math.sqrt(min(total_min / min_cap, 1.0))


# ── Salary formatting ──────────────────────────────────────────────────────────

def _fmt_salary(player_name: str, salary_dollars: float) -> str:
    """Format salary as '$X.XXM'."""
    return f"${salary_dollars / 1_000_000:.2f}M"


def fmt_next_contract(player_name: str, next_contracts: dict) -> str:
    # NOTE: classify_fa_status is coupled to this exact output format. Its
    # primary path uses this formatter as a consistency check (structured feed
    # fields are only trusted when they re-format to the caller's string), and
    # its fallback path PARSES these strings ("—" / "RFA" / trailing " PO" /
    # " TO"). Change the format here and that fallback parser together.
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


# ── Player-option decision model ─────────────────────────────────────────────
OPTION_OPT_IN_THRESHOLD = 0.5


def option_opt_in_prob(option_M: float, market_M: float, age) -> float:
    """Probability a player EXERCISES (opts into) his player option instead of
    declining it to sign a new deal. The dominant driver is how much more the
    option pays than his projected market value, a player keeps a $30M option
    rather than sign for $17M, with age breaking gray-zone ties (older players
    take the guaranteed money; younger ones bet on a longer deal). Returns 0–1.

    Rule-based / economic, not a trained classifier: there's no labelled history
    of opt-in/out decisions to fit, but the option-vs-market surplus IS the real
    driver and this captures the clear cases transparently.
    """
    if not option_M or option_M <= 0:
        return 0.0
    a = float(age) if age is not None else 28.0
    surplus = (option_M - market_M) / option_M          # premium the option pays over the market
    z = max(-30.0, min(30.0, 9.0 * surplus + 0.16 * (a - 28.0)))
    return 1.0 / (1.0 + math.exp(-z))


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


_OPTION_STATUS_CACHE = None


def _load_option_status() -> dict:
    """Precomputed {normalized_name: {'type': player_option|team_option}} for the upcoming
    season, built offline from the reliable contract scraper (scripts/build_option_status.py).
    Authoritative because the live next-year-salary feed mislabels options as 'guaranteed'
    and is non-deterministic. Loaded once; empty dict if the cache is absent."""
    global _OPTION_STATUS_CACHE
    if _OPTION_STATUS_CACHE is None:
        import json
        try:
            d = json.loads((Path(__file__).parent / "cache" / "option_status_2026.json").read_text())
            _OPTION_STATUS_CACHE = d.get("options", {})
        except Exception:
            _OPTION_STATUS_CACHE = {}
    return _OPTION_STATUS_CACHE


# classify_fa_status resolves status PRIMARILY from this memoized structured
# feed, not from the display string. Memoized per process because per-player
# classification must never turn into per-player pkl reads — or, when the feed
# is in its degraded uncached state (partial scrape, nothing written to disk),
# per-player live scrapes.
_NEXT_FEED_MEMO: dict = {}          # espn_year -> (monotonic read time, feed dict | None)
_NEXT_FEED_TTL = 300                # seconds between disk-cache re-reads


def _next_year_feed(cur_season: str) -> dict | None:
    """The structured next-year salary feed for `cur_season`'s offseason
    ({normalized_name: {"salary": float, "type": ...}} from
    fetch_next_year_contracts), or None when it can't be read. Bounded: at most
    one fetch call per _NEXT_FEED_TTL per season per process (normally a fresh
    disk-cache hit, since every caller fetches the same feed just before
    classifying), and a failed refresh keeps serving the last good feed
    (stale-beats-empty)."""
    try:
        yr = season_to_espn_year(cur_season)
    except Exception:
        return None
    now = time.monotonic()
    hit = _NEXT_FEED_MEMO.get(yr)
    if hit is not None and (now - hit[0]) < _NEXT_FEED_TTL:
        return hit[1]
    try:
        feed = fetch_next_year_contracts(yr, cache_v=7) or None
    except Exception:
        feed = None
    if feed is None and hit is not None:
        feed = hit[1]               # keep the last good feed on a failed refresh
    _NEXT_FEED_MEMO[yr] = (now, feed)
    return feed


def classify_fa_status(name: str, next_contract_str: str, rookie_set: set,
                       cur_season: str, cross_check: bool = True,
                       include_rookie_options: bool = False,
                       contract_end_map: dict | None = None) -> str | None:
    """A player's free-agency status THIS offseason: 'UFA' / 'RFA' /
    'Player Option' / 'Team Option', or None when he is NOT a free agent
    (still under contract). THE single source of truth — used by the Free Agent
    Class page, the home-page FA summary, the Contract Predictor FA toggle, and
    scripts/build_fa_board.py, so a player's status can never disagree across
    them.

    PRIMARY PATH: status comes from the salary feed's STRUCTURED fields
    (fetch_next_year_contracts -> {"salary", "type"}), fetched internally via
    _next_year_feed — so a copy tweak to fmt_next_contract's wording can't
    silently reclassify players. `next_contract_str` (fmt_next_contract's
    display string) still matters two ways:
      - consistency gate: the structured fields are trusted only when they
        re-format to exactly the string the caller is displaying, so the
        classification can never disagree with what's on screen; and
      - FALLBACK source: when the feed is unreadable here or the caller worked
        from a different snapshot, the string itself is parsed (that parser is
        coupled to fmt_next_contract's exact output format — see the note
        there).

    The salary feed OMITS some players who hold options or are signed (2nd-round
    rookies, two-ways, certain option years) — they read as missing/'—'. For
    those, cross-check the contract-end scraper (get_player_contract_info):
      - deal ends this season             -> UFA (a genuine expiring free agent)
      - option on the upcoming season      -> Player/Team Option
      - signed beyond the upcoming season  -> None (under contract; not an FA)
    e.g. Austin Reaves' player option and Will Richard's rookie deal are both
    missing from the salary feed; the cross-check classifies the first as a
    Player Option and drops the second. cross_check=False skips the scraper
    (use for non-current seasons, where today's contract data wouldn't apply).

    contract_end_map: optional precomputed contract-end dict (same shape as
    fetch_contract_end_years' result, e.g. the player-hub build cache's
    contract_end key). When given, the cross-check reads it INSTEAD of
    get_player_contract_info, so request-path callers can never fall through
    to the live ~30-page BBRef scrape on a lapsed pkl TTL."""
    # Authoritative option source first: a stable precomputed cache from the contract
    # scraper, because the live salary feed mislabels options as 'guaranteed' (so option
    # holders like Brook Lopez / Kuminga / Dort would otherwise read as "under contract"
    # and be dropped). Current season only (where today's contract data applies).
    if cross_check:
        _opt = _load_option_status().get(normalize(name))
        if _opt:
            # Rookie-scale option years (e.g. Wembanyama / Amen Thompson / Keyonte George's
            # year-4 team option) are auto-exercised formalities, NOT free agency — the player
            # is still on his rookie deal and isn't available, so drop him from the list
            # (unless the caller opts to include them for fun).
            if normalize(name) in rookie_set and not include_rookie_options:
                return None
            if _opt.get("type") == "player_option":
                return "Player Option"
            if _opt.get("type") == "team_option":
                return "Team Option"

    def _missing() -> str | None:
        # No feed entry for the player: rookie-scale -> RFA; otherwise cross-check
        # the contract-end scraper for feed-omitted option-holders / signed players.
        if normalize(name) in rookie_set:
            return "RFA"
        if cross_check:
            _cs = int(cur_season[:4]) + 1
            _contract_season = f"{_cs}-{(_cs + 1) % 100:02d}"   # upcoming season
            ci = (contract_end_map.get(normalize(name))
                  if contract_end_map is not None
                  else get_player_contract_info(name)) or {}
            end, last = ci.get("end_season"), ci.get("last_year_type")
            if end and end > cur_season:
                if end == _contract_season and last == "player_option":
                    return "Player Option"
                if end == _contract_season and last == "team_option":
                    return "Team Option"
                return None        # signed beyond this offseason -> not an FA
        return "UFA"

    s = next_contract_str
    # PRIMARY: the feed's structured fields — but only when they re-format to
    # the exact string the caller passed (same snapshot), so status and the
    # displayed string can never disagree.
    feed = _next_year_feed(cur_season)
    if feed is not None and fmt_next_contract(name, feed) == s:
        info = feed.get(normalize(name))
        if info is None:
            return _missing()
        t = info.get("type")
        if t == "rfa":
            return "RFA"
        if t == "player_option":
            return "Player Option"
        if t == "team_option":
            return "Team Option"
        return None                # "guaranteed" -> under contract, not an FA
    # FALLBACK: parse the display string (feed unreadable, or the caller's
    # snapshot differs). Coupled to fmt_next_contract's exact output format.
    if s == "RFA":
        return "RFA"
    if s == "—":
        return _missing()
    if " PO" in s:
        return "Player Option"
    if " TO" in s:
        return "Team Option"
    if not re.fullmatch(r"\$\d+(\.\d+)?M", s or ""):
        # Not a string fmt_next_contract produces — a format drift would land
        # here and silently read as "under contract", so make it loud.
        logger.warning("classify_fa_status: unrecognized next-contract string %r "
                       "for %s — treating as under contract", s, name)
    return None


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
    # Historical seasons are immutable — trust the committed disk cache for a
    # month so the live site never re-scrapes old Basketball-Reference pages
    # (a version bump changes the filename, so it still busts when data changes).
    # Current season auto-refreshes hourly DURING the season; in the offseason
    # (Jul–Sep) its stats are final too, so treat it as immutable — the hourly
    # "refresh" was pure churn that forced live NBA-API calls at request time.
    if ttl is not None:
        effective_ttl = ttl
    elif season is not None and season != SEASONS[0]:
        return True          # historical season caches are immutable (version-busted)
    elif season == SEASONS[0] and time.localtime().tm_mon not in (7, 8, 9):
        effective_ttl = 3600
    else:
        effective_ttl = 2_592_000
    return (time.time() - path.stat().st_mtime) < effective_ttl

def _pkl_load(path: Path):
    return pickle.loads(path.read_bytes())

def _pkl_save(path: Path, obj) -> None:
    # Atomic write (tmp sibling + os.replace), mirroring _atomic_to_parquet:
    # a writer killed mid-write (deploy SIGTERM) would otherwise leave a
    # truncated pkl whose fresh mtime makes stale-beats-empty trust it.
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}-{threading.get_ident()}")
    try:
        tmp.write_bytes(pickle.dumps(obj))
        os.replace(tmp, path)
    except Exception as e:
        # A full/unwritable cache disk silently degrades every scraper to
        # "never caches" — surface it in the logs rather than failing silently.
        logger.warning("cache write failed for %s: %s", path, e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet atomically (tmp file + os.replace, same filesystem).

    pyarrow writes in place, so a writer killed mid-write (deploy SIGTERM
    killing daemon threads, a one-shot script exiting) would otherwise leave a
    truncated file whose fresh mtime makes it look valid. With the tmp+replace
    dance, readers only ever see the old-complete or new-complete file and a
    kill just abandons the tmp file.
    """
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}-{threading.get_ident()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


@st.cache_data(ttl=3600, show_spinner="Fetching league stats...")
def fetch_league_stats(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Per-game player stats for one season.

    season_type controls regular season vs playoffs, accepts NBA Stats API's
    own values: "Regular Season" (default) or "Playoffs". Each variant gets
    its own disk cache so the two modes don't clobber each other.
    """
    suffix = "_playoff" if season_type == "Playoffs" else ""
    path = _dc_path(f"league_stats_{season.replace('-','_')}{suffix}.parquet")
    stale = None
    if path.exists():
        try:
            stale = pd.read_parquet(path)
        except Exception:
            stale = None
    if stale is not None and _dc_fresh(path, season=season):
        return stale
    # Live fetch — BOUNDED, and a readable-but-stale parquet always beats blocking:
    # stats.nba.com errors/blocks under load (July-1 FA frenzy), and the old
    # infinite-retry loop here hung every Contract Predictor prediction on the
    # "Fetching league stats..." spinner. Stale-while-error instead.
    time.sleep(0.5)
    result = None
    delay = 1
    for _ in range(3 if stale is not None else 8):
        try:
            result = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
                season_type_all_star=season_type,
                timeout=15,
            )
            break
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 15)
    if result is None:
        if stale is not None:
            logger.warning("league stats refresh failed for %s — serving stale parquet", season)
            return stale
        return pd.DataFrame()
    df = result.get_data_frames()[0]
    try:
        _atomic_to_parquet(df, path)
    except Exception:
        pass
    return df


@st.cache_data(ttl=3600, show_spinner="Fetching advanced stats...")
def fetch_advanced_stats(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Per-game ADVANCED player stats for one season (NBA Stats API
    MeasureType=Advanced). Returns USG_PCT, TS_PCT, PIE, AST_PCT, REB_PCT,
    OREB_PCT, DREB_PCT, OFF_RATING, DEF_RATING, NET_RATING, EFG_PCT, AST_TO,
    AST_RATIO, TM_TOV_PCT, PACE alongside PLAYER_ID/PLAYER_NAME.

    Available back to 1996-97. Disk-cached per season+type. These are the
    possession- and team-context metrics that can't be derived from the
    box score alone (usage rate, on/off ratings, impact estimate)."""
    suffix = "_playoff" if season_type == "Playoffs" else ""
    path = _dc_path(f"adv_stats_{season.replace('-','_')}{suffix}.parquet")
    stale = None
    if path.exists():
        try:
            stale = pd.read_parquet(path)
        except Exception:
            stale = None
    if stale is not None and _dc_fresh(path, season=season):
        return stale
    from nba_api.stats.endpoints import leaguedashplayerstats as _ldps
    time.sleep(0.5)
    result = None
    delay = 1
    attempts = 0
    # Bounded + fail-fast (timeout=15); a stale parquet always beats blocking the
    # request — same stale-while-error treatment as fetch_league_stats.
    while result is None and attempts < (3 if stale is not None else 8):
        try:
            result = _ldps.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
                season_type_all_star=season_type,
                measure_type_detailed_defense="Advanced",
                timeout=15,
            )
        except Exception:
            attempts += 1
            time.sleep(delay)
            delay = min(delay * 2, 15)
    if result is None:
        if stale is not None:
            logger.warning("advanced stats refresh failed for %s — serving stale parquet", season)
            return stale
        return pd.DataFrame()
    df = result.get_data_frames()[0]
    try:
        _atomic_to_parquet(df, path)
    except Exception:
        pass
    return df


@st.cache_data(ttl=86400)
@st.cache_data(ttl=3600, show_spinner="Fetching salary data...")
def fetch_salaries(espn_year: int) -> pd.DataFrame:
    rows = []
    for page in range(1, 15):
        url = f"https://www.espn.com/nba/salaries/_/year/{espn_year}/page/{page}"
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        except Exception:
            break                     # partial pages beat hanging/raising mid-scrape
        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            break
        df = pd.read_html(io.StringIO(str(tables[0])))[0]
        data = df[df[0] != "RK"]
        if data.empty:
            break
        rows.append(data)
    if not rows:
        # ESPN returned no salary tables (uncovered year / outage). Return an
        # empty frame with the expected columns rather than crashing build_raw
        # on pd.concat([]) — the caller tolerates a missing salary lookup.
        return pd.DataFrame(columns=["name", "salary"])
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


# ── BBRef playoff round scraper ───────────────────────────────────────────────
# Maps each postseason team to the deepest round they reached. Used by
# build_raw's playoff branch to weight availability by depth-of-run rather
# than raw GP — so a 7-game first-round loss and a 4-game first-round sweep
# both count as "1 round played", and the four Finals teams (champion +
# Finals loser per year) all land at round 4 even when their GP differs.
@st.cache_data(ttl=30 * 86_400, show_spinner=False)
def fetch_playoff_rounds(season: str) -> dict:
    """Returns {team_abbr: rounds_played} for one postseason.

    rounds_played is 1-4:
      1 = first-round exit
      2 = conf semis exit
      3 = conf finals exit
      4 = Finals (winner OR loser)

    Scraped once from Basketball Reference's playoff summary page and cached
    to disk. Empty dict if the page returns nothing (very old seasons /
    network failure / 429); callers should fall back to GP-only heuristic.
    """
    end_year = season_to_espn_year(season)
    disk_path = _dc_path(f"bref_playoff_rounds_{season}.pkl")
    if _dc_fresh(disk_path, ttl=30 * 86_400):
        try:
            return _pkl_load(disk_path)
        except Exception:
            pass

    url = f"https://www.basketball-reference.com/playoffs/NBA_{end_year}.html"
    try:
        r = requests.get(url, headers={"User-Agent": _BREF_UA}, timeout=15)
    except Exception as e:
        logger.warning("BBRef GET failed for %s: %s", url, e)
        return {}
    # Retry once after a backoff if we hit BBRef's rate limit
    if r.status_code == 429:
        wait = int(r.headers.get("Retry-After", 60))
        time.sleep(wait)
        try:
            r = requests.get(url, headers={"User-Agent": _BREF_UA}, timeout=15)
        except Exception as e:
            logger.warning("BBRef retry GET failed for %s: %s", url, e)
            return {}
    if r.status_code != 200:
        return {}

    # BBRef hides some series tables inside HTML comments. Strip the comment
    # markers so the parser sees one continuous document — but DON'T parse
    # them as tables, we only want the raw HTML for header/link scanning.
    content = r.text.replace("<!--", "").replace("-->", "")

    header_re    = re.compile(r"<h[23][^>]*>([^<]+)</h[23]>", re.IGNORECASE)
    team_link_re = re.compile(rf"/teams/([A-Z]{{3}})/{end_year}[/.]")

    def _round_for(text: str):
        t = text.strip().lower()
        # ORDER MATTERS — conference rounds must match before plain "finals"
        if ("conference final" in t or "eastern final" in t or "east finals" in t
                or "western final" in t or "west finals" in t):
            return 3
        if "semifinal" in t or "conference semi" in t or "second round" in t:
            return 2
        if "first round" in t or "conference quarter" in t or "quarterfinal" in t:
            return 1
        if "final" in t:  # bare "Finals" or "NBA Finals" — round 4
            return 4
        return None

    round_markers = []  # (byte position, round_num)
    for match in header_re.finditer(content):
        rn = _round_for(match.group(1))
        if rn is not None:
            round_markers.append((match.start(), rn))
    round_markers.sort()

    team_rounds: dict[str, int] = {}
    for i, (pos, round_num) in enumerate(round_markers):
        end_pos = round_markers[i + 1][0] if i + 1 < len(round_markers) else len(content)
        segment = content[pos:end_pos]
        for team in set(team_link_re.findall(segment)):
            if round_num > team_rounds.get(team, 0):
                team_rounds[team] = round_num

    if team_rounds:
        try:
            _pkl_save(disk_path, team_rounds)
        except Exception:
            pass
    return team_rounds


# ── BBRef per-game stats scraper ─────────────────────────────────────────────
# NBA Stats API returns empty for 1995-96 and earlier. For pre-1996 seasons we
# fall back to Basketball Reference's per-game stats table. Output schema
# matches fetch_league_stats so this is a drop-in replacement upstream of
# build_raw's formula.
@st.cache_data(ttl=30 * 86_400, show_spinner=False)
def fetch_bref_player_stats(season: str, playoffs: bool = False) -> pd.DataFrame:
    """Per-game player stats from BBRef. Returns DataFrame with the same
    columns the NBA Stats API would return (PLAYER_ID, PLAYER_NAME,
    TEAM_ABBREVIATION, GP, MIN, PTS, AST, OREB, DREB, BLK, STL, TOV, PF,
    FGA, FTA). Used as a fallback when NBA API has no data.

    playoffs=True pulls postseason per-game stats from BBRef's playoff
    page, used for pre-1996 playoff seasons since the NBA Stats API
    doesn't return playoff data either for those years.
    """
    end_year = season_to_espn_year(season)
    # v2: cache filename bumped 2026-04-30 to orphan v1 parquets that were
    #     written without TEAM_ABBREVIATION (rename map missed BBRef's "Team"
    #     column). v2 files are guaranteed to have the right schema.
    # v3: bumped 2026-05-09 to orphan v2 parquets that kept BBRef's Hall-of-
    #     Fame asterisk in player names ("Michael Jordan*"). v3 strips it.
    suffix = "_playoff" if playoffs else ""
    disk_path = _dc_path(f"bref_stats_v3_{season}{suffix}.parquet")
    if _dc_fresh(disk_path, ttl=30 * 86_400):
        try:
            cached = pd.read_parquet(disk_path)
            required = {"PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "PTS"}
            if required.issubset(cached.columns):
                return cached
            # Silently fall through and re-fetch if cache somehow lacks columns
        except Exception:
            pass

    if playoffs:
        url = f"https://www.basketball-reference.com/playoffs/NBA_{end_year}_per_game.html"
    else:
        url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_per_game.html"
    r = requests.get(url, headers={"User-Agent": _BREF_UA}, timeout=15)
    # Rate-limited? Wait and retry once before giving up — BBRef caps at
    # roughly 20 req/min, and the seed script can blow through that on
    # pre-1996 seasons. The Retry-After header isn't always populated, so
    # default to 60 seconds (BBRef's typical penalty).
    if r.status_code == 429:
        wait = int(r.headers.get("Retry-After", 60))
        time.sleep(wait)
        r = requests.get(url, headers={"User-Agent": _BREF_UA}, timeout=15)
        if r.status_code == 429:
            raise RuntimeError(f"BBRef rate-limited on {url} (retry also failed)")
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

    # Strip BBRef's Hall-of-Fame marker (trailing '*' on enshrined players,
    # e.g. "Michael Jordan*"). Otherwise normalize() doesn't match modern
    # NBA Stats API names and the same player gets two entries across data
    # sources.
    if "PLAYER_NAME" in df.columns:
        df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.rstrip("*").str.strip()

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
            _atomic_to_parquet(df, disk_path)
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
    game of that month, so January's score includes Oct + Nov + Dec + Jan.
    Availability multiplier uses team games played through that month (not 82)
    so a player who misses no games in the first 30 gets full credit.
    """
    from nba_api.stats.endpoints import playergamelog, teamgamelog

    # ── Player game log ───────────────────────────────────────────────────────
    gl = None
    delay = 1
    while gl is None:
        try:
            gl = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=15)
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
        # appearing in 30/30 team games = full availability, not 30/82.
        # v5 formula: 0.30 floor, sqrt of total_min/cap only.
        min_cap = team_gp * (2500 / 82)
        avail = 0.30 + 0.70 * math.sqrt(min(total_min / min_cap, 1.0))
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
def trade_side_summary(player_names: tuple[str, ...], season: str,
                       playoffs: bool = False) -> dict:
    """Summarize a list of players in a given season, their Barrett Scores
    and salaries. Used by the Trades page. Player names matched by normalize().
    Returns dict with: rows (DataFrame), found (list), missing (list),
    barrett_total (float), salary_total (float).

    playoffs=True uses postseason scores instead of regular-season.
    """
    if not _raw_disk_exists(season, playoffs):
        # Don't trigger fresh build_raw on view-time requests
        return {"rows": pd.DataFrame(), "found": [], "missing": list(player_names),
                "barrett_total": 0.0, "salary_total": 0.0}

    try:
        df = apply_projections(apply_rankings(build_raw(season, playoffs)))
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
        "side_a_picks": "Plus Boston's 2009 + 2012 firsts",
        "side_b_picks": "Picks landed at #5 (Jonny Flynn) and #20 (Wayne Ellington)",
        "winner":      "side_a",
        "verdict":     "Celtics. Garnett anchored the historic Big Three defense and won the 2008 title in his first year. Minnesota got volume, no stars.",
        "key_points": [
            "Boston won the 2007-08 championship in Garnett's first year",
            "Big Three made the Finals twice (won 2008, lost 2010)",
            "Minnesota missed the playoffs every year of the rebuild window",
        ],
    },
    {
        "name":        "Pau Gasol to Lakers (2008)",
        "season":      "2007-08",
        "year_after":  "2008-09",
        "side_a_team": "Los Angeles Lakers",
        "side_a":      ["Pau Gasol"],
        "side_b_team": "Memphis Grizzlies",
        "side_b":      ["Kwame Brown", "Javaris Crittenton", "Aaron McKie"],
        "side_a_picks": "Plus Marc Gasol's draft rights, became multi-time All-Star",
        "side_b_picks": "Lakers 2008 + 2010 first-round picks",
        "winner":      "side_a",
        "verdict":     "Lakers. Three Finals in three years, two titles, generational frontcourt. One of the most lopsided trades in NBA history at the time it was made.",
        "key_points": [
            "Lakers made the Finals 3 straight years (2008, 2009, 2010)",
            "Won championships in 2008-09 and 2009-10",
            "Memphis eventually got Marc Gasol back too, but won zero playoff series with the original return",
        ],
    },
    {
        "name":        "James Harden to Houston (2012)",
        "season":      "2012-13",
        "year_after":  "2013-14",
        "side_a_team": "Houston Rockets",
        "side_a":      ["James Harden", "Cole Aldrich", "Daequan Cook", "Lazar Hayward"],
        "side_b_team": "Oklahoma City Thunder",
        "side_b":      ["Kevin Martin", "Jeremy Lamb"],
        "side_a_picks": "—",
        "side_b_picks": "Two firsts + a second from Houston (landed Steven Adams, Mitch McGary, Alex Abrines)",
        "winner":      "side_a",
        "verdict":     "Houston, by a mile. Harden became MVP and the franchise centerpiece for nearly a decade. Adams was nice; not nearly enough.",
        "key_points": [
            "Harden won 2017-18 MVP in Houston",
            "Rockets made the Western Finals twice (2015, 2018) with Harden as #1 option",
            "OKC's return: zero deep playoff runs traceable to this trade's assets",
        ],
    },
    {
        "name":        "Kawhi Leonard to Toronto (2018)",
        "season":      "2018-19",
        "year_after":  "2019-20",
        "side_a_team": "Toronto Raptors",
        "side_a":      ["Kawhi Leonard", "Danny Green"],
        "side_b_team": "San Antonio Spurs",
        "side_b":      ["DeMar DeRozan", "Jakob Poeltl"],
        "side_a_picks": "—",
        "side_b_picks": "Protected 2019 Raptors first (landed at #29, Keldon Johnson)",
        "winner":      "side_a",
        "verdict":     "Toronto. Won the 2019 title, the only championship in franchise history, even though Kawhi left in free agency that summer. Worth it any day.",
        "key_points": [
            "Raptors won the 2018-19 championship in Kawhi's only season",
            "Kawhi was Finals MVP",
            "DeRozan + Poeltl gave SAS playoff appearances but no second-round wins",
        ],
    },
    {
        "name":        "Anthony Davis to Lakers (2019)",
        "season":      "2019-20",
        "year_after":  "2020-21",
        "side_a_team": "Los Angeles Lakers",
        "side_a":      ["Anthony Davis"],
        "side_b_team": "New Orleans Pelicans",
        "side_b":      ["Lonzo Ball", "Brandon Ingram", "Josh Hart"],
        "side_a_picks": "—",
        "side_b_picks": "Three firsts (became Trey Murphy III, Dyson Daniels) + pick swaps",
        "winner":      "side_a",
        "verdict":     "Lakers. Won the 2020 title in year one with AD as Finals co-MVP-caliber force. Picks may bear long-term fruit for NOP, but a ring beats potential.",
        "key_points": [
            "Lakers won the 2019-20 championship, Bubble title",
            "AD was a top-3 player on a Finals team alongside prime LeBron",
            "Pelicans rebuild produced Murphy + Daniels but zero playoff series wins from this haul",
        ],
    },
    {
        "name":        "Kyrie Irving to Boston (2017)",
        "season":      "2017-18",
        "year_after":  "2018-19",
        "side_a_team": "Boston Celtics",
        "side_a":      ["Kyrie Irving"],
        "side_b_team": "Cleveland Cavaliers",
        "side_b":      ["Isaiah Thomas", "Jae Crowder", "Ante Zizic"],
        "side_a_picks": "—",
        "side_b_picks": "Unprotected 2018 Nets pick (became Collin Sexton, #8)",
        "winner":      "wash",
        "verdict":     "Wash. Boston got two seasons of Kyrie (no Finals, weird locker room). Cleveland got Sexton from the Nets pick but made the Finals once more before LeBron left and the wheels came off.",
        "key_points": [
            "Cavaliers made the 2018 Finals (lost to GSW), final LeBron Finals run",
            "Boston went to the Eastern Finals in 2018 (Kyrie injured) and 2019 (Kyrie disgruntled), no Finals",
            "Kyrie left for Brooklyn in 2019; Sexton was traded to Utah in 2022",
        ],
    },
    {
        "name":        "Pierce/Garnett to Brooklyn (2013)",
        "season":      "2013-14",
        "year_after":  "2014-15",
        "side_a_team": "Brooklyn Nets",
        "side_a":      ["Paul Pierce", "Kevin Garnett", "Jason Terry"],
        "side_b_team": "Boston Celtics",
        "side_b":      ["Gerald Wallace", "Kris Humphries", "Marshon Brooks", "Keith Bogans"],
        "side_a_picks": "—",
        "side_b_picks": "Three unprotected firsts + pick swap (became Jaylen Brown, Jayson Tatum via swap, others)",
        "winner":      "side_b",
        "verdict":     "Celtics. Possibly the worst trade of the modern era for the receiving star team. Nets cratered for half a decade and handed Boston their next dynasty's foundation (Brown + Tatum).",
        "key_points": [
            "Boston used the picks to draft Jaylen Brown (#3, 2016) and acquire Jayson Tatum (#3, 2017)",
            "Tatum + Brown won the 2023-24 championship",
            "Nets won zero playoff series with Pierce + Garnett, missed playoffs three of next five years",
        ],
    },
    {
        "name":        "Jimmy Butler to Philadelphia (2018)",
        "season":      "2018-19",
        "year_after":  "2019-20",
        "side_a_team": "Philadelphia 76ers",
        "side_a":      ["Jimmy Butler", "Justin Patton"],
        "side_b_team": "Minnesota Timberwolves",
        "side_b":      ["Robert Covington", "Dario Šarić", "Jerryd Bayless"],
        "side_a_picks": "—",
        "side_b_picks": "Protected 2022 Sixers first",
        "winner":      "wash",
        "verdict":     "Wash, leaning Philly. Sixers got a heroic Kawhi-game-7 playoff run from Butler before he bolted to Miami. Wolves got rotation pieces who left within a year.",
        "key_points": [
            "Butler walked to Miami via sign-and-trade summer 2019",
            "Sixers lost in 7 to eventual champion Raptors (Kawhi's Game 7 dagger)",
            "Covington + Šarić were both traded out of Minnesota within a year",
        ],
    },
    {
        "name":        "Allen Iverson to Detroit (2008)",
        "season":      "2008-09",
        "year_after":  "2009-10",
        "side_a_team": "Detroit Pistons",
        "side_a":      ["Allen Iverson"],
        "side_b_team": "Denver Nuggets",
        "side_b":      ["Chauncey Billups", "Antonio McDyess", "Cheikh Samb"],
        "side_a_picks": "—",
        "side_b_picks": "—",
        "winner":      "side_b",
        "verdict":     "Denver, decisively. Billups led Nuggets to their first Western Finals in 24 years. Detroit's veteran core fell apart immediately; AI played 54 games and was waived.",
        "key_points": [
            "Denver made the 2008-09 Western Conference Finals (lost to Lakers)",
            "Iverson lasted 54 games in Detroit, fit was a disaster",
            "Pistons missed the playoffs the following year for the first time in 8 seasons",
        ],
    },
]


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_player_names(min_seasons: int = 1) -> list[str]:
    """All player names that appear in any season, sorted by career-average
    Barrett Score (highest first, GP-weighted). Used to populate autocomplete
    dropdowns. Includes EVERY player who appeared in a game, including
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
def fetch_player_full_career(player_name: str, playoffs: bool = False) -> pd.DataFrame:
    """Full per-season career stats for one player: raw counting stats from
    fetch_league_stats joined with Barrett Score / rank from build_raw +
    apply_rankings. One row per season the player appeared in.

    playoffs=True uses postseason data, pulls from the playoff-cached
    league_stats and raw parquets. Pre-1996 playoff data isn't seeded yet
    (different BBRef URL), so those seasons just won't appear.

    Only reads seasons that are already on disk, view-time requests must
    NEVER trigger fresh BBRef scrapes. seed_cache.py populates the disk."""
    name_norm = normalize(player_name)
    season_type = "Playoffs" if playoffs else "Regular Season"

    # FAST PATH: the combined all-seasons frame already holds every season's ranked
    # output (barrett, rank, salary, ts/d-lebron/avail...) as ONE memoized parquet
    # read. The legacy loop below re-ranked all ~53 seasons at request time —
    # ~minutes of CPU on Render's half-core box for the FIRST prediction after
    # every deploy (the "contract predictor takes forever" reports). Counting
    # stats (PTS/AST/...) come from the per-season stats parquets, which are
    # plain cached reads — no ranking work. Falls back to the loop on any gap.
    try:
        # Direct read-only load of the persisted combined parquet. Never call the
        # builder here: it may decide to REBUILD all ~53 seasons (and even hit the
        # live stats API for missing ones) inside the user's prediction — the
        # profiled cause of multi-minute predictions. Boot (serve.py) owns rebuilds.
        if playoffs:
            _cpath = _dc_path(f"all_seasons_0_playoff_{PLAYOFF_VERSION}_{FORMULA_VERSION}.parquet")
        else:
            _cpath = _dc_path(f"all_seasons_0_{FORMULA_VERSION}.parquet")
        combined = pd.read_parquet(_cpath) if _cpath.exists() else pd.DataFrame()
        need = {"Player", "Season", "barrett_score", "score_rank", "GP", "MPG",
                "Team", "avail_mult", "ts_pct", "d_lebron", "efficiency_adj", "salary"}
        if len(combined) and need.issubset(combined.columns):
            mask = combined["Player"].astype(str).apply(normalize) == name_norm
            mine = combined[mask]
            if len(mine):
                season_sizes = combined["Season"].value_counts()
                fast_rows: list[dict] = []
                for _, br in mine.iterrows():
                    season = br["Season"]
                    stat_row = None
                    try:
                        stats = fetch_league_stats(season, season_type)
                        if not stats.empty and "PLAYER_NAME" in stats.columns:
                            m2 = stats["PLAYER_NAME"].apply(normalize) == name_norm
                            if m2.any():
                                stat_row = stats[m2].iloc[0]
                    except Exception:
                        stat_row = None
                    _g = (lambda k, d=0.0: float(stat_row.get(k, d)) if stat_row is not None else d)
                    barrett_canonical = float(br["barrett_score"])
                    barrett_raw = float(br.get("barrett_score_raw", barrett_canonical))
                    fast_rows.append({
                        "Season":        season,
                        "Team":          br["Team"],
                        "GP":            int(br["GP"]),
                        "MPG":           float(br["MPG"]),
                        "PTS":           _g("PTS"), "AST": _g("AST"),
                        "OREB":          _g("OREB"), "DREB": _g("DREB"),
                        "REB":           _g("OREB") + _g("DREB"),
                        "STL":           _g("STL"), "BLK": _g("BLK"),
                        "TOV":           _g("TOV"), "PF": _g("PF"),
                        "TS%":           float(br.get("ts_pct", 0) or 0) * 100,
                        "D-LEBRON":      float(br.get("d_lebron", 0) or 0),
                        "EffAdj":        float(br.get("efficiency_adj", 0) or 0),
                        "Avail":         float(br.get("avail_mult", 1.0) or 1.0),
                        "Barrett Score": barrett_canonical,
                        "Barrett (Raw)": barrett_raw,
                        "Score Rank":    int(br["score_rank"]),
                        "Total Players": int(season_sizes.get(season, 0)),
                        "Salary":        float(br.get("salary", 0) or 0),
                    })
                if fast_rows:
                    result = pd.DataFrame(fast_rows)
                    result["_year"] = result["Season"].apply(lambda s: int(s.split("-")[0]))
                    return (result.sort_values("_year").drop(columns=["_year"])
                            .reset_index(drop=True))
    except Exception as e:
        logger.warning("full-career fast path failed for %s (%s) — using legacy loop",
                       player_name, e)

    rows: list[dict] = []
    for season in SEASONS:
        if not _raw_disk_exists(season, playoffs):
            continue
        try:
            stats = fetch_league_stats(season, season_type)
            # Pre-1996 fallback — pulls per-game stats from BBRef. Same path
            # works for both regular season and playoffs (different URLs).
            if stats.empty or "PLAYER_NAME" not in stats.columns:
                try:
                    stats = fetch_bref_player_stats(season, playoffs=playoffs)
                except Exception:
                    stats = pd.DataFrame()
            if stats.empty or "PLAYER_NAME" not in stats.columns:
                continue
            mask = stats["PLAYER_NAME"].apply(normalize) == name_norm
            if not mask.any():
                continue
            raw_row = stats[mask].iloc[0]

            ranked = apply_rankings(build_raw(season, playoffs))
            if ranked.empty:
                continue
            mask2 = ranked["Player"].apply(normalize) == name_norm
            if not mask2.any():
                continue
            br_row = ranked[mask2].iloc[0]

            # In v6+ the canonical barrett_score is already pace-adjusted.
            # Fall back gracefully if older caches lack barrett_score_raw.
            barrett_canonical = float(br_row["barrett_score"])
            barrett_raw = float(br_row.get("barrett_score_raw", barrett_canonical))
            rows.append({
                "Season":         season,
                "Team":           raw_row["TEAM_ABBREVIATION"],
                "GP":             int(raw_row["GP"]),
                "MPG":            float(raw_row["MIN"]),
                "PTS":            float(raw_row["PTS"]),
                "AST":            float(raw_row["AST"]),
                "OREB":           float(raw_row.get("OREB", 0)),
                "DREB":           float(raw_row.get("DREB", 0)),
                "REB":            float(raw_row.get("OREB", 0)) + float(raw_row.get("DREB", 0)),
                "STL":            float(raw_row["STL"]),
                "BLK":            float(raw_row["BLK"]),
                "TOV":            float(raw_row["TOV"]),
                "PF":             float(raw_row.get("PF", 0)),
                "TS%":            float(br_row.get("ts_pct", 0)) * 100,
                # Raw component pieces — used by the "Score Breakdown" view on Search.
                # All season-level (not per-game) so the breakdown matches the
                # base-score arithmetic exactly.
                "D-LEBRON":       float(br_row.get("d_lebron", 0) or 0),
                "EffAdj":         float(br_row.get("efficiency_adj", 0) or 0),
                "Avail":          float(br_row.get("avail_mult", 1.0) or 1.0),
                "Barrett Score":  barrett_canonical,  # pace-adjusted (v6+)
                "Barrett (Raw)":  barrett_raw,         # un-adjusted, for Search toggle
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
def fetch_season_component_distribution(season: str, playoffs: bool = False) -> pd.DataFrame:
    """Per-player six-component breakdown for one season.

    Returns one row per player who appears in the rankings pool, with
    columns:
      PLAYER_ID, Player, Scoring, Playmaking, Rebounding, Defense,
      Efficiency, Availability

    Each column is computed using the same formula as utils.base_score:
      Scoring     = PTS (per-game)
      Playmaking  = AST × 1.5 − TOV / 1.5
      Rebounding  = OREB / 2 + DREB / 3
      Defense     = BLK / 2 + STL / 1.5 − PF / 3 + D-LEBRON × 2
      Efficiency  = efficiency_adj × 2 (signed; TS%-based)
      Availability = avail_mult (0.30 → 1.00 multiplier)

    Used by the Score Breakdown section on Search to convert one
    player's components into percentile ranks against the rest of the
    league for that same season.
    """
    season_type = "Playoffs" if playoffs else "Regular Season"
    raw_stats = fetch_league_stats(season, season_type)
    if raw_stats.empty or "PLAYER_NAME" not in raw_stats.columns:
        # Pre-1996 fallback — BBRef per-game stats (NBA Stats API doesn't
        # cover these years for some endpoints).
        try:
            raw_stats = fetch_bref_player_stats(season, playoffs=playoffs)
        except Exception:
            raw_stats = pd.DataFrame()
    if raw_stats.empty or "PLAYER_NAME" not in raw_stats.columns:
        return pd.DataFrame()

    ranked = apply_rankings(build_raw(season, playoffs))
    if ranked.empty:
        return pd.DataFrame()

    # Need a name key on both sides. raw_stats has PLAYER_NAME; ranked has Player.
    raw_keep = raw_stats[["PLAYER_NAME", "PTS", "AST",
                          "OREB", "DREB", "BLK", "STL", "TOV", "PF"]].copy()
    raw_keep["_key"] = raw_keep["PLAYER_NAME"].apply(normalize)
    ranked2 = ranked.copy()
    ranked2["_key"] = ranked2["Player"].apply(normalize)

    merged = ranked2.merge(raw_keep, on="_key", how="inner", suffixes=("", "_raw"))
    if merged.empty:
        return pd.DataFrame()

    # All fields are per-game from fetch_league_stats. d_lebron, eff_adj,
    # avail_mult are season-level from build_raw.
    out = pd.DataFrame({
        "PLAYER_ID":   merged["PLAYER_ID"].values,
        "Player":      merged["Player"].values,
        "Scoring":     merged["PTS"].astype(float).values,
        "Playmaking":  (merged["AST"].astype(float) * 1.5
                        - merged["TOV"].astype(float) / 1.5).values,
        "Rebounding":  (merged["OREB"].astype(float) / 2
                        + merged["DREB"].astype(float) / 3).values,
        "Defense":     (merged["BLK"].astype(float) / 2
                        + merged["STL"].astype(float) / 1.5
                        - merged["PF"].astype(float) / 3
                        + merged.get("d_lebron", 0).astype(float) * 2).values,
        "Efficiency":  (merged.get("efficiency_adj", 0).astype(float) * 2).values,
        "Availability": merged.get("avail_mult", 1.0).astype(float).values,
    })
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_position_peer_distribution(
    season: str,
    position: str,
    n_seasons_back: int = 1,
    playoffs: bool = False,
) -> pd.DataFrame:
    """Component breakdown for a single primary position (PG/SG/SF/PF/C),
    pooled across the current season + N prior seasons for a smoother sample.

    Used by the Search page's "position peer" Score Breakdown view so users
    can see how a Guard's scoring stacks up against other Guards, not
    against the whole league (where Centers monopolize rebounds and Guards
    monopolize playmaking).

    `position` is a 5-bucket value (PG/SG/SF/PF/C). For each pooled season,
    we cross-reference fetch_player_positions_detailed and keep only
    players whose primary position matches. Players who appear in
    multiple pooled seasons appear multiple times (one row each) so the
    percentile reflects the full pool of qualifying player-seasons.

    Returns the same 6-component schema as
    fetch_season_component_distribution but filtered + pooled.
    """
    if position not in {"PG", "SG", "SF", "PF", "C"}:
        return pd.DataFrame()

    # Build the list of seasons to pool (current + N back), staying within SEASONS.
    if season not in SEASONS:
        return pd.DataFrame()
    base_idx = SEASONS.index(season)
    pool_seasons = SEASONS[base_idx : base_idx + 1 + n_seasons_back]

    frames: list[pd.DataFrame] = []
    for s in pool_seasons:
        dist = fetch_season_component_distribution(s, playoffs=playoffs)
        if dist.empty:
            continue
        # Pull the position lookup for this season; gracefully fall back
        # if the scrape failed for that year.
        try:
            pos_lookup = fetch_player_positions_detailed(s, cache_v=3)
        except Exception:
            pos_lookup = {}
        if not pos_lookup:
            continue

        # Filter to same-position players (primary position from BBRef).
        dist_pos = dist.copy()
        dist_pos["_pos"] = dist_pos["Player"].apply(
            lambda n: pos_lookup.get(normalize(n), "")
        )
        dist_pos = dist_pos[dist_pos["_pos"] == position]
        if not dist_pos.empty:
            frames.append(dist_pos)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_career_trend(player_id: int, num_seasons: int = 5,
                       playoffs: bool = False) -> pd.DataFrame:
    """Barrett Score per season pulled directly from build_raw, guaranteed to
    match the stat panel since both use the same LeagueDashPlayerStats source.

    Pass playoffs=True for postseason scores (uses the separate playoff
    parquet cache; seasons without playoff data on disk are skipped).

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
        if not _raw_disk_exists(season, playoffs):
            # Season not yet seeded — skip rather than trigger a fresh fetch
            # that would hang this request for tens of seconds.
            continue
        try:
            raw = build_raw(season, playoffs)
            if raw.empty:
                continue
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_player_career_with_rank(player_id: int, playoffs: bool = False) -> list[dict]:
    """Career trajectory with score + season-rank for SVG hover tooltips.

    Returns sorted list of {'season', 'score', 'rank', 'total'} for every
    season on disk where the player appears. Disk-only, never triggers a
    fresh API hit at view time. playoffs=True reads postseason caches.
    """
    info = nba_players_static.find_player_by_id(player_id)
    if not info:
        return []
    name_norm = normalize(info["full_name"])
    out: list[dict] = []
    for season in SEASONS:
        if not _raw_disk_exists(season, playoffs):
            continue
        try:
            ranked = apply_rankings(build_raw(season, playoffs))
            if ranked.empty:
                continue
            mask = ranked["Player"].apply(normalize) == name_norm
            if not mask.any():
                continue
            r = ranked[mask].iloc[0]
            out.append({
                "season": season,
                "score":  float(r["barrett_score"]),
                "rank":   int(r["score_rank"]),
                "total":  int(len(ranked)),
            })
        except Exception:
            continue
    out.sort(key=lambda d: int(d["season"].split("-")[0]))
    return out


def _player_season_splits_raw(player_id: int, season: str,
                               d_lebron_val: float = 0.0,
                               league_avg_ts: float = 0.57,
                               season_games: int = 82) -> pd.DataFrame:
    career = None
    delay = 1
    while career is None:
        try:
            career = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=15)
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

    # Vectorised per-stint pipeline (was three apply-axis-1 loops). Same
    # math as build_raw above; clipped at ±4 here vs ±6 in build_raw — that's
    # intentional, not a typo: stints use a tighter band since they're a
    # subset of a season's data.
    _eff = (0.15 * (rows["ts_pct"] - league_avg_ts) * 100).clip(-4, 4)
    _eligible = (rows["FGA"] >= 2.0) & rows["ts_pct"].notna()
    rows["efficiency_adj"] = _eff.where(_eligible, 0.0).astype(float)

    rows["total_min"] = (rows["MIN"] * rows["GP"]).round(0).astype(int)
    rows["base_score"] = (
        rows["PTS"]
        + rows["AST"] * 1.5
        + rows["OREB"] / 2
        + rows["DREB"] / 3
        + rows["BLK"] / 2
        + rows["STL"] / 1.5
        - rows["TOV"] / 1.5
        - rows["PF"] / 3
        + rows["d_lebron"] * 2
        + rows["efficiency_adj"] * 2
    )

    _min_cap = season_games * (2500 / 82)
    _ratio   = (rows["total_min"] / max(_min_cap, 1)).clip(0, 1)
    rows["avail_mult"] = 0.30 + 0.70 * np.sqrt(_ratio)
    rows["barrett_score"] = rows["base_score"] * rows["avail_mult"]

    return rows[["TEAM_ABBREVIATION", "GP", "MIN", "total_min",
                 "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF",
                 "d_lebron", "ts_pct", "efficiency_adj",
                 "base_score", "avail_mult", "barrett_score"]].rename(columns={
        "TEAM_ABBREVIATION": "Team",
        "MIN": "MPG",
    })


@st.cache_data(ttl=3600, show_spinner="Building splits table, loading once, fast for everyone after…")
def build_splits_data_live(season: str, salary_lookup: tuple) -> pd.DataFrame:
    sal_dict = dict(salary_lookup)

    TOTALS_COLS = ["GP", "MIN", "PTS", "AST", "OREB", "DREB", "BLK", "STL", "TOV", "PF"]

    def _fetch_team(team: dict):
        delay = 1
        for _attempt in range(3):     # bounded — a dead stats API must not hang the page
            try:
                ep = leaguedashplayerstats.LeagueDashPlayerStats(
                    season=season,
                    per_mode_detailed="Totals",
                    team_id_nullable=team["id"],
                    timeout=15,
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
    # Splits cache also contains barrett_score values so it's formula-dependent.
    # Same versioning rationale as the combined-seasons parquet above.
    path = CACHE_DIR / f"splits_{season.replace('-', '_')}_{FORMULA_VERSION}.pkl"
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
    except Exception as e:
        logger.warning("fetch_bref_positions(espn_year=%s) scrape failed: %s",
                       espn_year, e)
    _pkl_save(path, result)
    return result


def fetch_player_positions_detailed(season: str, cache_v: int = 3) -> dict:
    """Returns {normalized_name: "PG"|"SG"|"SF"|"PF"|"C"} from BBRef
    per-game stats. Much better coverage than the ESPN-salary-page scrape
    in fetch_bref_positions, and gives 5-bucket positions instead of 3.

    For hyphenated positions like "PG-SG" we take the primary (first) listing.
    Disk-cached for 1 day per season.

    cache_v=3: switched from the short "Mozilla/5.0" UA to _BREF_UA (the
    same UA the other working BBRef scrapers use). Short UA was getting
    blocked, returning empty responses → 0 players in the lookup → every
    player falling through to the coarse Guard/Forward/Center labels.
    """
    year = season_to_espn_year(season)
    path = _dc_path(f"positions_detailed_{year}_v{cache_v}.pkl")
    if _dc_fresh(path, ttl=86400):
        try:
            cached = _pkl_load(path)
            if cached:  # don't trust empty cache files
                return cached
        except Exception:
            pass

    result: dict = {}
    try:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
        time.sleep(0.6)  # be polite to BBRef
        r = requests.get(url, headers={"User-Agent": _BREF_UA}, timeout=15)
        if r.status_code == 429:
            time.sleep(15)
            r = requests.get(url, headers={"User-Agent": _BREF_UA}, timeout=15)
        # Force UTF-8 so accented names (Dončić, Jokić) decode properly
        # and normalize() can strip diacritics for the lookup key.
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")

        # Two layouts: BBRef sometimes serves the per-game table inline,
        # other times it wraps the table HTML inside a comment (anti-scraper
        # measure for current-season pages). Try inline first, then comments,
        # then any table with Player + Pos headers.
        table = soup.find("table", id="per_game_stats")

        if table is None:
            try:
                from bs4 import Comment
                for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
                    txt = str(c)
                    if 'id="per_game_stats"' in txt or 'id="per_game"' in txt:
                        inner_soup = BeautifulSoup(txt, "html.parser")
                        candidate = (
                            inner_soup.find("table", id="per_game_stats")
                            or inner_soup.find("table", id="per_game")
                        )
                        if candidate is not None:
                            table = candidate
                            break
            except Exception:
                pass

        if table is None:
            # Last-ditch fallback: any table with Player + Pos column headers.
            for t in soup.find_all("table"):
                headers = [th.get_text(strip=True) for th in t.find_all("th")]
                if "Player" in headers and "Pos" in headers:
                    table = t
                    break

        if table is not None:
            df = pd.read_html(io.StringIO(str(table)))[0]
            # Repeated header rows show up periodically in BBRef tables.
            if "Player" in df.columns:
                df = df[df["Player"].astype(str) != "Player"]
            if "Pos" not in df.columns or "Player" not in df.columns:
                raise RuntimeError("Missing Player/Pos columns")
            df = df[df["Pos"].notna() & df["Player"].notna()]
            for _, row in df.iterrows():
                name = str(row["Player"]).strip().rstrip("*").strip()
                pos_raw = str(row["Pos"]).strip().upper()
                primary = pos_raw.split("-")[0].strip()
                if primary in {"PG", "SG", "SF", "PF", "C"}:
                    # Multi-team players appear once per stint plus a TOT row.
                    # First occurrence wins; later identical entries are no-ops.
                    result.setdefault(normalize(name), primary)
    except Exception as e:
        logger.warning("fetch_player_positions_detailed(season=%s) scrape "
                       "failed (will fall back to ESPN coarse positions): %s",
                       season, e)

    if result:
        _pkl_save(path, result)
    return result


def position_to_bucket(detailed_pos: str) -> str:
    """PG/SG → Guard, SF/PF → Forward, C → Center. Used to roll up
    5-bucket BBRef positions to the 3-bucket scheme the contract
    multipliers were fit against."""
    if detailed_pos in ("PG", "SG"):
        return "Guard"
    if detailed_pos in ("SF", "PF"):
        return "Forward"
    if detailed_pos == "C":
        return "Center"
    return "Unknown"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_rookie_scale_players(season: str, cache_v: int = 3) -> set:
    """Returns a set of normalized names for first-round picks currently
    inside their rookie scale contract (years 1-4 post-draft).

    For season "2025-26": the most recent draft was Summer 2025 (rookies).
    Year-4 players were drafted Summer 2022. Range: 2022-2025.

    cache_v=2: fixes a year-range bug in v1 that missed year-4 rookies
    (e.g. 2022 draftees in the 2025-26 season, Jalen Duren, Tari Eason).
    cache_v=3: excludes draft-window players whose rookie deal ENDED EARLY —
    a rookie contract can't run past draft_year+4, so a current deal ending
    beyond that horizon means a NEW contract (e.g. Jake LaRavia: 2022 pick,
    option declined, re-signed with LAL through 2026-27 — not rookie scale).
    """
    path = _dc_path(f"rookie_scale_{season.replace('-','_')}_v{cache_v}.pkl")
    stale: set | None = None
    if path.exists():
        try:
            stale = _pkl_load(path)
        except Exception:
            stale = None
    if stale and _dc_fresh(path, ttl=86400):
        return stale
    try:
        end_year = int(season.split("-")[0]) + 1  # "2025-26" → 2026

        # Rookie scale spans years 1-4. For season ending in end_year,
        # year-4 players were drafted (end_year - 4), year-1 rookies drafted
        # (end_year - 1). Range: end_year-4 inclusive, end_year exclusive.
        rookie_draft_years = set(range(end_year - 4, end_year))

        idx = playerindex.PlayerIndex(season=season, timeout=15)
        df_idx = idx.get_data_frames()[0]

        draft_year_by: dict[str, int] = {}
        for _, row in df_idx.iterrows():
            try:
                draft_year  = int(row["DRAFT_YEAR"])
                draft_round = int(row["DRAFT_ROUND"])
            except (ValueError, TypeError):
                continue
            if draft_year in rookie_draft_years and draft_round == 1:
                full_name = f"{row['PLAYER_FIRST_NAME']} {row['PLAYER_LAST_NAME']}".strip()
                draft_year_by[normalize(full_name)] = draft_year

        # Contract-horizon filter: a rookie deal can't extend past draft_year+4, so a
        # current contract ending beyond that horizon means the player re-signed a NEW
        # deal (option declined then re-signed, or extended) and is NOT on rookie scale.
        contracts = fetch_contract_end_years()
        rookies: set = set()
        for nm, dy in draft_year_by.items():
            ci = contracts.get(nm)
            if ci and ci.get("end_season"):
                try:
                    end_year = int(str(ci["end_season"])[:4]) + 1   # "2026-27" -> 2027
                    if end_year > dy + 4:
                        continue                                    # new deal — not rookie scale
                except (ValueError, TypeError):
                    pass
            rookies.add(nm)
        # Sanity floor: 4 draft classes × 30 first-rounders ≈ 100+. A tiny result
        # means the API answered with partial data — treat it as a failure rather
        # than letting it poison the cache (an EMPTY set here made Wembanyama & co
        # show up as free agents and broke RFA classification page-wide).
        if len(rookies) < 50:
            raise ValueError(f"suspiciously small rookie-scale set ({len(rookies)})")
        _pkl_save(path, rookies)
        return rookies
    except Exception as e:
        if stale:
            logger.warning("rookie-scale refresh failed (%s) — serving stale pkl (%d players)", e, len(stale))
            return stale
        return set()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_next_year_contracts(espn_year: int, cache_v: int = 7) -> dict:
    """
    Returns {normalized_name: {"salary": float, "type": str}} for next season.
    type is one of: "guaranteed", "team_option", "player_option", "rfa".
    """
    path = _dc_path(f"next_contracts_{espn_year}_v{cache_v}.pkl")
    stale = None
    if path.exists():
        try:
            stale = _pkl_load(path)
        except Exception:
            stale = None
    if stale and _dc_fresh(path, ttl=86400):
        return stale
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

    # Sanity before caching: a failed ESPN scrape gives a near-empty dict, and a
    # failed Spotrac scrape strips EVERY option/RFA flag — both poisoned the cache
    # for a day and made the FA feed flap (options appearing/vanishing between
    # renders). A stale-but-complete dict always beats a fresh-but-partial one.
    has_flags = any(v.get("type") != "guaranteed" for v in contracts.values())
    stale_flags = any(v.get("type") != "guaranteed" for v in (stale or {}).values())
    if len(contracts) < 200 or (stale_flags and not has_flags):
        if stale:
            logger.warning("next-year contracts refresh looked partial (%d entries, flags=%s) — serving stale",
                           len(contracts), has_flags)
            return stale
        return contracts          # nothing better available; do NOT cache it
    _pkl_save(path, contracts)
    return contracts


@st.cache_data(ttl=3600, show_spinner="Fetching D-LEBRON defensive ratings...")
def fetch_dlebron_all() -> pd.DataFrame:
    """Fetches all D-LEBRON data in one call, every player, every season back to 2009-10."""
    path = _dc_path("dlebron_all.parquet")
    stale = None
    if path.exists():
        try:
            stale = pd.read_parquet(path)
        except Exception:
            stale = None
    if stale is not None and len(stale) and _dc_fresh(path, season=SEASONS[0]):
        return stale
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
            _atomic_to_parquet(df, path)
        except Exception:
            pass
        return df
    except Exception as e:
        if stale is not None and len(stale):
            logger.warning("D-LEBRON refresh failed (%s) — serving stale parquet", e)
            return stale
        return pd.DataFrame(columns=["nba_id", "Season", "D-LEBRON"])


def fetch_dlebron(season: str) -> dict:
    """Returns {player_id (int): d_lebron (float)} for a single season."""
    df = fetch_dlebron_all()
    if df.empty:
        return {}
    season_df = df[df["Season"] == season]
    return {int(row["nba_id"]): float(row["D-LEBRON"]) for _, row in season_df.iterrows()}


# ── D-LEBRON proxy for pre-2009 seasons ────────────────────────────────────
# Real D-LEBRON (from BBall-Index) only goes back to 2009-10. For older
# eras, the defensive component of Barrett Score would otherwise be 0 —
# heavily penalizing defensive specialists (Olajuwon, Mutombo, Pippen,
# Bruce Bowen, etc.) in pre-2009 contract validation.
#
# Coefficients fit via OLS on the 2009-2025 sample (n=9,649 player-seasons
# with MPG ≥ 10). Inputs are per-game stats from fetch_league_stats.
#
# Final fit: R² = 0.594, MAE = 0.52 (D-LEBRON units).
#
# Tried adding per-minute stats, MPG, position dummies, and position
# interactions — all gave ≤+0.05pp R² improvement at the cost of model
# complexity. The simple per-game form is the right tradeoff.
#
# Limitations:
#   - Per-game box stats explain ~60% of D-LEBRON variance. Real D-LEBRON
#     uses play-by-play impact data the box score genuinely can't see
#     (on/off splits, matchup difficulty, lineup effects).
#   - For pre-1973 (no STL/BLK), this proxy degenerates further.
#
# Used by build_raw when fetch_dlebron returns 0 for a player.

_DLEBRON_PROXY_COEFS = {
    "intercept": -1.355138,
    "STL":       +0.932316,
    "BLK":       +1.436954,
    "DREB":      +0.089358,
    "PF":        -0.164789,
}



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
#   v5: Availability multiplier rebalanced — floor dropped from 0.75 to 0.30,
#       drops the GP factor (only total_min/cap matters), so an 18-game
#       season multiplier goes from 0.80 → 0.62. Injured stars score lower.
#   v6: Pace adjustment is now CANONICAL — applied to barrett_score in
#       build_raw so every page (Rankings, Legacy, Trades, Search, all-time
#       lists) gets era-normalized scores by default. Old un-adjusted view
#       still available as `barrett_score_raw` for the Search toggle.
FORMULA_VERSION = "v7"  # v7: D-LEBRON proxy replaces hand-tuned fallback (R²=0.59)

# Separate version tag for playoff caches so playoff-formula tweaks invalidate
# only playoff parquets, leaving the regular-season caches valid.
#   p1: initial Stage 1 (team-max GP, min/total_min cap)
#   p2: league-max GP, min/total_min cap (depth-of-run signal)
#   p3: GP-only ratio (drops the min cap so playoff stars at 38-42 MPG
#       don't all max out the multiplier)
#   p4: round × engagement — uses BBRef's actual round outcomes (via
#       fetch_playoff_rounds) so Finals teams hit 1.00 regardless of GP,
#       and 4-game vs 7-game first-round series both count as 1 round.
PLAYOFF_VERSION = "p4"


def _raw_disk_path(season: str, playoffs: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if playoffs:
        return CACHE_DIR / f"raw_{season.replace('-', '_')}_playoff_{PLAYOFF_VERSION}_{FORMULA_VERSION}.parquet"
    return CACHE_DIR / f"raw_{season.replace('-', '_')}_{FORMULA_VERSION}.parquet"

def _raw_disk_fresh(season: str, playoffs: bool = False) -> bool:
    """True if the on-disk parquet exists AND is within its TTL.

    Used by **write-side** decisions: `build_raw` calls this to decide
    whether to load the cached parquet or rebuild from APIs. Returning
    False here triggers a network rebuild, which is expensive, so we
    only do it when the cache is genuinely stale.

    Use `_raw_disk_exists()` for view-time reads instead, those should
    serve cached data even when slightly stale (don't drop a season from
    a player's career arc just because the cache hasn't been touched in
    a few hours).
    """
    p = _raw_disk_path(season, playoffs)
    if not p.exists():
        return False
    # Historical seasons are IMMUTABLE — a finished season's stats never change, and
    # cache_v/FORMULA_VERSION filename bumps handle real invalidation. The old 30-day
    # TTL meant a long-lived /data disk "expired" all ~52 history parquets at once and
    # mass-refetched them (2026-07-02: the boot warm hit exactly that and deploys
    # ground for so long the old container never handed over). Current season:
    # hourly during the season, monthly in the offseason (stats are final in July).
    if season != SEASONS[0]:
        return True
    ttl = 3600 if time.localtime().tm_mon not in (7, 8, 9) else 30 * 86_400
    return (time.time() - p.stat().st_mtime) < ttl


def _raw_disk_exists(season: str, playoffs: bool = False) -> bool:
    """True if the on-disk parquet exists, regardless of TTL.

    Use this in view-time read paths (player career arcs, position-peer
    distributions, etc.) where serving slightly-stale cache is better
    than silently dropping the season entirely. The freshness check is
    for write-side decisions about when to refresh, not read-side gating.
    """
    return _raw_disk_path(season, playoffs).exists()

# ── Stale-while-revalidate plumbing for the raw-frame disk cache ──────────────
# At most one background rebuild per (season, playoffs) so a burst of visitors
# hitting a stale cache kicks off a single refresh, not one rebuild per request.
_raw_refresh_inflight = set()
_raw_refresh_lock = threading.Lock()


def _spawn_raw_refresh(season: str, playoffs: bool = False) -> None:
    """Rebuild a stale raw parquet in a background thread so the foreground
    request serves the cached copy without blocking on the ~20s rebuild."""
    key = (season, playoffs)
    with _raw_refresh_lock:
        if key in _raw_refresh_inflight:
            return
        _raw_refresh_inflight.add(key)

    def _job() -> None:
        try:
            _build_raw_live(season, playoffs)
        except Exception:
            logger.warning("background raw refresh failed for %s", key, exc_info=True)
        else:
            # Make the fresh parquet visible now: the in-memory memos still
            # hold the stale frame for up to another hour. Clearing them means
            # the next request re-reads the disk (~0.1s), not the old memo.
            # Full clear (not per-args): callers reach these with mixed call
            # shapes, and a re-read per season is cheap.
            try:
                build_raw.clear()
                build_ranked_projected.clear()
            except Exception:
                pass
        finally:
            with _raw_refresh_lock:
                _raw_refresh_inflight.discard(key)

    try:
        threading.Thread(target=_job, daemon=True).start()
    except Exception as e:
        # Thread exhaustion: undo the claim so a later request can retry,
        # and let the caller serve the stale frame as usual.
        with _raw_refresh_lock:
            _raw_refresh_inflight.discard(key)
        logger.warning("could not start raw refresh thread for %s: %s", key, e)


@st.cache_data(ttl=3600, show_spinner="Building rankings...")
def build_raw(season: str, playoffs: bool = False) -> pd.DataFrame:
    """Serve the raw Barrett-score frame, stale-while-revalidate.

    Inside the Streamlit server: a readable parquet is returned immediately;
    if it is past its TTL a single deduped background thread rebuilds it and
    then clears the in-memory memos, so the next request picks up the fresh
    file. In bare mode (seed_cache.py, scripts/build_*.py) a stale parquet
    rebuilds synchronously instead — publishing pipelines need fresh-or-block
    semantics, and daemon threads would be killed at script exit anyway.
    Only a missing or unreadable parquet rebuilds synchronously on the
    request path.
    """
    if _raw_disk_exists(season, playoffs):
        try:
            cached = pd.read_parquet(_raw_disk_path(season, playoffs))
            # Sanitize old caches that captured BBRef's Hall-of-Fame asterisk
            # in player names ("Michael Jordan*").
            if "Player" in cached.columns:
                cached["Player"] = cached["Player"].astype(str).str.rstrip("*").str.strip()
        except Exception as e:
            # Corrupted file — fall through to a live rebuild (which also
            # rewrites it). Spawning happens only after a successful read, so
            # a corrupt+stale file can't trigger two racing rebuilds.
            logger.warning("raw cache unreadable for %s (playoffs=%s), rebuilding: %s",
                           season, playoffs, e)
        else:
            if not _raw_disk_fresh(season, playoffs):
                from streamlit import runtime as _st_runtime
                if _st_runtime.exists():
                    _spawn_raw_refresh(season, playoffs)
                else:
                    return _build_raw_live(season, playoffs)
            return cached
    return _build_raw_live(season, playoffs)


def _build_raw_live(season: str, playoffs: bool = False) -> pd.DataFrame:
    """Fetch from the APIs, compute the Barrett-score frame, and persist it to
    disk. The expensive path; build_raw only calls this on a true cache miss or
    from a background refresh."""
    season_type = "Playoffs" if playoffs else "Regular Season"

    # Fetch stats + salaries in parallel — saves ~5s on cold cache misses.
    # Salaries are always regular-season (one annual salary applies to the
    # whole year, including the postseason).
    with ThreadPoolExecutor(max_workers=2) as _pool:
        _stats_f = _pool.submit(fetch_league_stats, season, season_type)
        _sal_f   = _pool.submit(fetch_salaries, season_to_espn_year(season))
        stats    = _stats_f.result().copy()
        salaries = _sal_f.result()

    # Pre-1996 fallback: NBA Stats API returns empty for those years, so
    # scrape per-game stats from BBRef instead. This unlocks Magic, Bird,
    # prime Jordan, and full Kareem. Both regular season AND playoffs are
    # supported (different BBRef URLs).
    if stats.empty:
        try:
            stats = fetch_bref_player_stats(season, playoffs=playoffs).copy()
        except Exception:
            stats = pd.DataFrame()
        if stats.empty:
            return pd.DataFrame()

    sal_lookup = ({normalize(n): s for n, s in zip(salaries["name"], salaries["salary"])}
                  if not salaries.empty and "name" in salaries.columns else {})
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

    # D-LEBRON: only available for regular-season. For playoffs we always
    # fall through to the box-score defense estimate (same formula already
    # used for pre-2009-10 seasons).
    if playoffs:
        dlebron = {}
        stats["d_lebron"] = 0.0
    else:
        dlebron = fetch_dlebron(season)
        stats["d_lebron"] = stats["PLAYER_ID"].map(dlebron).fillna(0)

    # ── D-LEBRON proxy fallback ──────────────────────────────────────────────
    # When fetch_dlebron returns no data (pre-2009-10 or playoffs), fall back
    # to the box-score-fit proxy. Replaces the earlier hand-tuned composite
    # (BLK*1.5 + STL*1.5 + DREB*0.15 - PF*0.4) with regression coefficients
    # fit on the 2009-2025 sample. R² = 0.594, MAE = 0.52 (D-LEBRON units).
    # See _DLEBRON_PROXY_COEFS above.
    if not dlebron:
        stats["d_lebron"] = (
            _DLEBRON_PROXY_COEFS["intercept"]
            + _DLEBRON_PROXY_COEFS["STL"]  * stats["STL"]
            + _DLEBRON_PROXY_COEFS["BLK"]  * stats["BLK"]
            + _DLEBRON_PROXY_COEFS["DREB"] * stats["DREB"]
            + _DLEBRON_PROXY_COEFS["PF"]   * stats["PF"]
        ).clip(-5, 6)

    stats["ts_pct"] = stats["PTS"] / (2 * (stats["FGA"] + 0.44 * stats["FTA"])).replace(0, float("nan"))
    league_avg_ts = (stats["ts_pct"] * stats["GP"]).sum() / stats["GP"].sum()
    K_EFF = 0.15
    MIN_FGA = 2.0

    # ── Vectorised efficiency adjustment ──────────────────────────────────────
    # Previously per-row apply (lambda row -> python conditional). Same logic
    # expressed as column arithmetic + np.where is ~50× faster on hundreds
    # of rows because pandas/numpy never drops into Python for the iteration.
    _eff = K_EFF * (stats["ts_pct"] - league_avg_ts) * 100
    _eff = _eff.clip(-6, 6)
    _eligible = (stats["FGA"] >= MIN_FGA) & stats["ts_pct"].notna()
    stats["efficiency_adj"] = _eff.where(_eligible, 0.0).astype(float)

    stats["total_min"] = (stats["MIN"] * stats["GP"]).round(0).astype(int)

    # ── Vectorised base_score ─────────────────────────────────────────────────
    # Was: stats.apply(base_score, axis=1) → per-row Python loop. The function
    # is just column arithmetic so column ops are equivalent.
    _d_lebron = stats["d_lebron"] if "d_lebron" in stats.columns else 0
    stats["base_score"] = (
        stats["PTS"]
        + stats["AST"] * 1.5
        + stats["OREB"] / 2
        + stats["DREB"] / 3
        + stats["BLK"] / 2
        + stats["STL"] / 1.5
        - stats["TOV"] / 1.5
        - stats["PF"] / 3
        + _d_lebron * 2
        + stats["efficiency_adj"] * 2
    )

    season_games = int(stats["GP"].max())
    if playoffs:
        # Playoff availability — round × engagement.
        #   round_credit = team's deepest round / 4   (R1=0.25, F=1.00)
        #   gp_factor    = player's GP / team's max GP (their share of the run)
        #   mult         = 0.30 + 0.70 × √(round_credit × gp_factor)
        # So Finals teams' iron-men land at 1.00, conf finals exits ~0.91,
        # conf semis ~0.78, R1 exits ~0.65 (regardless of whether the series
        # was a 4-game sweep or a 7-game battle). A player who missed games
        # within their team's run gets a proportional haircut on top.
        # Falls back to GP-only formula if BBRef rounds data is missing.
        rounds_lookup = fetch_playoff_rounds(season)
        team_max_gp = stats.groupby("TEAM_ABBREVIATION")["GP"].max()
        if rounds_lookup:
            stats["_team_rounds"] = (
                stats["TEAM_ABBREVIATION"].map(rounds_lookup).fillna(1).astype(int)
            )
            round_credit = stats["_team_rounds"] / 4.0
            denom        = stats["TEAM_ABBREVIATION"].map(team_max_gp).clip(lower=1)
            gp_factor    = (stats["GP"] / denom).clip(upper=1.0)
            engagement   = (round_credit * gp_factor).clip(lower=0, upper=1)
            stats["avail_mult"] = (0.30 + 0.70 * engagement.pow(0.5)).clip(0.30, 1.0)
            stats = stats.drop(columns=["_team_rounds"])
        else:
            # Heuristic fallback: GP-only ratio against league max
            gp_ratio = (stats["GP"] / max(season_games, 1)).clip(0, 1)
            stats["avail_mult"] = (0.30 + 0.70 * gp_ratio.pow(0.5)).clip(0.30, 1.0)
    else:
        # Regular season: GP × MPG via total_min / 2500-cap (default formula).
        # Vectorised — same math as availability_multiplier() applied to the
        # whole column at once. min_cap is constant across rows.
        _min_cap = season_games * (2500 / 82)
        _ratio = (stats["total_min"] / max(_min_cap, 1)).clip(0, 1)
        stats["avail_mult"] = 0.30 + 0.70 * np.sqrt(_ratio)
    # Canonical Barrett Score is now PACE-ADJUSTED — applies across the whole
    # site (Rankings, Legacy, Trades, Search, etc.) so cross-era comparisons
    # are honest by default. Volume stats (PTS, AST, REB, BLK, STL, TOV, PF)
    # get scaled by season pace; D-LEBRON and the TS% efficiency adjustment
    # are already era-relative so they're left alone.
    pf = pace_factor(season)
    volume_part = stats["base_score"] - stats["d_lebron"] * 2 - stats["efficiency_adj"] * 2
    stats["base_score_pace"] = volume_part * pf + stats["d_lebron"] * 2 + stats["efficiency_adj"] * 2
    stats["barrett_score"]     = stats["base_score_pace"] * stats["avail_mult"]
    stats["barrett_score_raw"] = stats["base_score"]      * stats["avail_mult"]

    result = stats[[
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "total_min",
        "base_score", "base_score_pace", "avail_mult",
        "barrett_score", "barrett_score_raw", "salary",
        "d_lebron", "ts_pct", "efficiency_adj",
    ]].rename(columns={
        "PLAYER_NAME": "Player",
        "TEAM_ABBREVIATION": "Team",
        "MIN": "MPG",
    })

    # ── Persist to disk so future cold-starts skip the API entirely ───────────
    try:
        _atomic_to_parquet(result, _raw_disk_path(season, playoffs))
    except Exception as e:
        # A failed persist means the parquet never freshens and the hourly
        # background rebuild keeps recomputing for nothing — surface it.
        logger.warning("raw cache write failed for %s (playoffs=%s): %s",
                       season, playoffs, e)

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
        # Eager historical warming is OFF by default. This background build_raw
        # barrage across 52 seasons contended badly with real page loads on
        # Render's 0.5-CPU box: a cold Contract Predictor view measured ~11s with
        # it off vs ~43s with it on (GIL + CPU contention with the page's own
        # build_raw calls). Season parquets now ship on disk, so each page warms
        # lazily from disk on first visit instead — no background thread fighting
        # the request. Set HOOPSVALUE_WARM=1 to opt back into eager warming.
        if os.environ.get("HOOPSVALUE_WARM") != "1":
            return
        time.sleep(float(os.environ.get("HOOPSVALUE_WARM_DELAY", "20")))
        historical = SEASONS[1:]  # current season already warm
        with ThreadPoolExecutor(max_workers=2) as pool:
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
def build_all_seasons_combined(min_threshold: int = DEFAULT_MIN_THRESHOLD,
                               playoffs: bool = False) -> pd.DataFrame:
    """Load every season, apply per-season rankings + projections, and concatenate.

    Rankings/projections are applied *within* each season so score_rank and
    value_diff are always comparable within a year.  The Season column is added
    so cross-season analysis can group/filter by year.

    playoffs=True builds a separate combined dataset from postseason caches.
    Different parquet on disk so the two modes don't clobber each other.

    NOTE: Uses @st.cache_resource (singleton, no copy on hit) instead of
    @st.cache_data. Callers MUST .copy() before mutating columns.
    """
    # Include FORMULA_VERSION + mode (and PLAYOFF_VERSION for playoff
    # variants) so formula bumps and mode switches both invalidate.
    if playoffs:
        path = _dc_path(f"all_seasons_{min_threshold}_playoff_{PLAYOFF_VERSION}_{FORMULA_VERSION}.parquet")
    else:
        path = _dc_path(f"all_seasons_{min_threshold}_{FORMULA_VERSION}.parquet")
    # Season-aware freshness: rebuilding means re-ranking all ~53 seasons — hourly
    # was pure churn (and in the offseason the data cannot change at all).
    if _dc_fresh(path, season=SEASONS[0]):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    frames: list[pd.DataFrame] = []
    for season in SEASONS:
        try:
            raw      = build_raw(season, playoffs)
            if raw.empty:
                continue
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
        _atomic_to_parquet(combined, path)
    except Exception:
        pass
    return combined


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_draft_classes() -> pd.DataFrame:
    """Draft history from the NBA: Player, draft_year (int), round, pick."""
    path = _dc_path("draft_history.parquet")
    stale = None
    if path.exists():
        try:
            stale = pd.read_parquet(path)
        except Exception:
            stale = None
    if stale is not None and len(stale) and _dc_fresh(path, ttl=86400):
        return stale
    try:
        from nba_api.stats.endpoints import drafthistory
        time.sleep(0.6)
        df = drafthistory.DraftHistory(timeout=15).get_data_frames()[0]
        keep = [c for c in ["PLAYER_NAME", "SEASON", "ROUND_NUMBER", "ROUND_PICK", "OVERALL_PICK"] if c in df.columns]
        df = df[keep].copy().rename(columns={"PLAYER_NAME": "Player", "SEASON": "draft_year"})
        df["draft_year"] = pd.to_numeric(df["draft_year"], errors="coerce")
        df = df.dropna(subset=["draft_year"])
        df["draft_year"] = df["draft_year"].astype(int)
        df["player_norm"] = df["Player"].apply(normalize)
        try:
            _atomic_to_parquet(df, path)
        except Exception:
            pass
        return df
    except Exception as e:
        # Stale-beats-empty: an empty frame here makes EVERY player "Undrafted" and
        # silently degrades the model's draft-tier features (July-1 outage lesson).
        if stale is not None and len(stale):
            logger.warning("draft classes refresh failed (%s) — serving stale parquet", e)
            return stale
        return pd.DataFrame(columns=["Player", "draft_year", "player_norm",
                                     "ROUND_NUMBER", "ROUND_PICK", "OVERALL_PICK"])


# ── Draft-tier classification ─────────────────────────────────────────────
# Used by the Contract Predictor to weight comparables by draft pedigree.
# Lottery picks earn on pedigree even when production drops; 2nd-round /
# undrafted developers stay on minimums until they prove out. Matching a
# late-round developer against a lottery-pick comp (or vice versa) biases
# the market view, so the comp-matching distance penalizes tier mismatch.
#
# Tiers are ordinal — adjacent tiers are penalized less than far ones.
DRAFT_TIERS = ["Lottery", "Mid-1st", "Late-1st", "2nd", "Undrafted"]
DRAFT_TIER_ORDINAL = {t: i for i, t in enumerate(DRAFT_TIERS)}


def _pick_to_tier(overall_pick, round_number=None) -> str:
    """Map an overall draft pick to one of five tiers. Returns 'Undrafted'
    for None / NaN / 0 / negative inputs. ROUND_NUMBER is a fallback when
    OVERALL_PICK is missing."""
    try:
        p = int(overall_pick) if overall_pick is not None else 0
    except (TypeError, ValueError):
        p = 0
    if p <= 0:
        # Fall back to round_number if overall pick is unknown but round is.
        try:
            r = int(round_number) if round_number is not None else 0
        except (TypeError, ValueError):
            r = 0
        if r == 1:
            return "Late-1st"   # conservative — no pick number, assume late
        if r == 2:
            return "2nd"
        return "Undrafted"
    if p <= 14:
        return "Lottery"
    if p <= 22:
        return "Mid-1st"
    if p <= 30:
        return "Late-1st"
    return "2nd"  # 31-60 (and any 61+ historical noise)


def get_player_draft_info(player_name: str) -> dict:
    """Return draft tier + overall pick for one player. Uses fetch_draft_classes
    which is cached for the day. Players with no draft record (undrafted FAs,
    international signings, etc.) return tier='Undrafted', pick=None."""
    try:
        df = fetch_draft_classes()
        if df.empty:
            return {"draft_tier": "Undrafted", "draft_pick": None, "draft_year": None}
        name_norm = normalize(player_name)
        mask = df["player_norm"] == name_norm
        if not mask.any():
            return {"draft_tier": "Undrafted", "draft_pick": None, "draft_year": None}
        row = df[mask].iloc[0]
        overall = row.get("OVERALL_PICK")
        rnd     = row.get("ROUND_NUMBER")
        try:
            pick_int = int(overall) if overall is not None and not pd.isna(overall) else None
        except (TypeError, ValueError):
            pick_int = None
        return {
            "draft_tier": _pick_to_tier(overall, rnd),
            "draft_pick": pick_int,
            "draft_year": int(row["draft_year"]) if not pd.isna(row.get("draft_year")) else None,
        }
    except Exception:
        return {"draft_tier": "Undrafted", "draft_pick": None, "draft_year": None}


def build_draft_tier_lookup() -> dict:
    """Bulk lookup: normalized name -> {tier, pick, year}. Used by
    load_historical_signings to avoid one fetch_draft_classes per row."""
    try:
        df = fetch_draft_classes()
        if df.empty:
            return {}
        out: dict = {}
        for _, r in df.iterrows():
            out[r["player_norm"]] = {
                "draft_tier": _pick_to_tier(r.get("OVERALL_PICK"), r.get("ROUND_NUMBER")),
                "draft_pick": int(r["OVERALL_PICK"]) if pd.notna(r.get("OVERALL_PICK")) else None,
                "draft_year": int(r["draft_year"]) if pd.notna(r.get("draft_year")) else None,
            }
        return out
    except Exception:
        return {}


# ── All-NBA selections ─────────────────────────────────────────────────────
# Used by the Contract Predictor for supermax / designated-vet eligibility.
# Scraped from BBRef awards page — one row per (season × 1st/2nd/3rd team)
# with 5 player columns each ending in a position letter (e.g. "Nikola JokićC").

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_all_nba_selections(cache_v: int = 1) -> dict:
    """Returns {normalized_name: [{"season": "YYYY-YY", "team": 1|2|3}, ...]}.

    Disk-cached for 1 day. On scrape failure returns whatever was cached
    previously, or an empty dict if no cache exists.
    """
    path = _dc_path(f"all_nba_selections_v{cache_v}.pkl")
    if _dc_fresh(path, ttl=86400):
        try:
            cached = _pkl_load(path)
            if cached:
                return cached
        except Exception:
            pass

    out: dict = {}
    try:
        url = "https://www.basketball-reference.com/awards/all_league.html"
        time.sleep(0.6)  # be polite to BBRef
        r = requests.get(url, headers={"User-Agent": _BREF_UA}, timeout=15)
        if r.status_code == 429:
            # BBRef rate limit — wait and retry once.
            time.sleep(15)
            r = requests.get(url, headers={"User-Agent": _BREF_UA}, timeout=15)
        # BBRef serves UTF-8 (Jokić, Dončić, etc.) but requests sometimes
        # auto-detects latin-1 from headers. Force UTF-8 so the accented
        # names decode correctly and our normalize() can strip diacritics.
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", id="awards_all_league")
        if table is None:
            # Fall back to comment-wrapped (BBRef pattern for some pages).
            from bs4 import Comment as _Comment
            for c in soup.find_all(string=lambda t: isinstance(t, _Comment)):
                if 'id="awards_all_league"' in str(c):
                    inner = BeautifulSoup(str(c), "html.parser")
                    table = inner.find("table", id="awards_all_league")
                    if table:
                        break

        if table is None:
            raise RuntimeError("awards_all_league table not found")

        team_map = {"1st": 1, "2nd": 2, "3rd": 3}
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if not cells or len(cells) < 5:
                continue
            # Skip header rows (Season cell is literal "Season").
            season_text = cells[0].get_text(strip=True)
            if not season_text or season_text == "Season":
                continue
            tm_text = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            team_level = team_map.get(tm_text)
            # Only collect All-NBA (NBA league) — skip ABA + All-Defensive.
            lg_text = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            if lg_text != "NBA" or team_level is None:
                continue
            # Player cells start at index 4 (after Season/Lg/Tm/Voting).
            for cell in cells[4:]:
                name_with_pos = cell.get_text(strip=True)
                if not name_with_pos:
                    continue
                # Trailing position letter (G/F/C). Strip it.
                if name_with_pos[-1] in ("G", "F", "C"):
                    name = name_with_pos[:-1]
                else:
                    name = name_with_pos
                nm = normalize(name)
                out.setdefault(nm, []).append({
                    "season": season_text,
                    "team": team_level,
                })

        _pkl_save(path, out)
        return out
    except Exception as e:
        logger.warning("fetch_all_nba_selections scrape failed: %s", e)
        # Try to return any older cache as fallback.
        try:
            return _pkl_load(path) or {}
        except Exception:
            return {}


def get_all_nba_in_window(player_name: str, current_season: str,
                          window_seasons: int = 3) -> list[dict]:
    """Return All-NBA selections for `player_name` within the most recent
    `window_seasons` seasons ending at `current_season`. Used by supermax
    eligibility checks (Designated Veteran Extension requires 1 All-NBA
    in immediately prior season OR 2 in past 3 seasons)."""
    selections = fetch_all_nba_selections().get(normalize(player_name), [])
    if not selections:
        return []
    # Build the set of allowed season strings.
    try:
        end_year = int(current_season.split("-")[0])
    except Exception:
        return selections
    allowed = {f"{end_year - i}-{str(end_year - i + 1)[-2:]}"
               for i in range(window_seasons)}
    return [s for s in selections if s["season"] in allowed]


# ── Contract end years (multi-year contracts) ──────────────────────────────
# Scraped from BBRef per-team contracts pages. Each player row has y1-y6
# columns showing salaries for next 6 seasons; empty cells indicate the
# contract has expired. CSS class flags:
#   class="right"           → guaranteed salary
#   class="right salary-pl" → player option (player can opt in/out)
#   class="right salary-tm" → team option
#   class="right salary-et" → early termination option
#   class="right iz"        → empty (contract has ended by this season)
#
# We use this to project a player's NEXT signing year (when does their
# current deal expire?). Critical for stars locked into supermax / max
# deals like Luka (signed through 2028-29 PO) — predicting their "next
# contract today" with current cap is misleading.

_BBREF_TEAMS = [
    "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]


@st.cache_data(ttl=86400, show_spinner="Loading contract data…")
def fetch_contract_end_years(cache_v: int = 1) -> dict:
    """Returns {normalized_player_name: contract_info} for every player
    with an active multi-year contract.

    contract_info = {
        "end_season":      "YYYY-YY",
        "last_year_type":  "guaranteed" | "player_option" | "team_option" | "et_option",
        "signing_season":  "YYYY-YY", projected season player signs NEXT deal,
        "years_remaining": int, how many seasons (including current) until end,
        "current_team":    "LAL",
    }

    For free agents this season (no future salaries), the player simply
    doesn't appear in the dict. The caller treats absence as "signing now."
    """
    path = _dc_path(f"contract_end_years_v{cache_v}.pkl")
    if _dc_fresh(path, ttl=86400):
        try:
            cached = _pkl_load(path)
            if cached:
                return cached
        except Exception:
            pass

    out: dict = {}
    base_url = "https://www.basketball-reference.com/contracts"

    # Pull the index page once to discover the season header → year mapping.
    # We don't strictly need this (BBRef columns are always y1=current),
    # but reading the season labels keeps the function robust if BBRef
    # ever rearranges columns.
    try:
        time.sleep(0.6)
        r = requests.get(f"{base_url}/", headers={"User-Agent": _BREF_UA}, timeout=15)
        if r.status_code == 429:
            time.sleep(15)
            r = requests.get(f"{base_url}/", headers={"User-Agent": _BREF_UA}, timeout=15)
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        # Map y1, y2, etc. to season strings from the column headers.
        year_map: dict[str, str] = {}
        for th in soup.find_all("th", attrs={"data-stat": True}):
            ds = th.get("data-stat", "")
            if ds.startswith("y") and ds[1:].isdigit():
                lbl = th.get("aria-label") or th.get_text(strip=True)
                if lbl and "-" in lbl and len(lbl) == 7:
                    year_map[ds] = lbl
    except Exception as e:
        logger.warning("fetch_contract_end_years index fetch failed: %s", e)
        year_map = {}

    if not year_map:
        # Fall back to a hard-coded assumption: y1 is current season.
        # Determine current season from latest known cap year.
        cur = SEASONS[0]
        cur_year = int(cur.split("-")[0])
        for i in range(1, 7):
            yr = cur_year + (i - 1)
            year_map[f"y{i}"] = f"{yr}-{str(yr + 1)[-2:]}"

    # Now iterate each team's contracts page and parse player rows.
    for team in _BBREF_TEAMS:
        url = f"{base_url}/{team}.html"
        try:
            time.sleep(0.6)
            r = requests.get(url, headers={"User-Agent": _BREF_UA}, timeout=15)
            if r.status_code == 429:
                time.sleep(15)
                r = requests.get(url, headers={"User-Agent": _BREF_UA}, timeout=15)
            r.encoding = "utf-8"
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            logger.warning("contracts fetch failed for %s: %s", team, e)
            continue

        # Player rows are <tr> with a <th data-stat="player"> child that
        # contains the player's link. Skip "notes" rows (which have
        # data-stat="notes" instead of y1-y6).
        for tr in soup.find_all("tr"):
            player_th = tr.find("th", attrs={"data-stat": "player"})
            if player_th is None:
                continue
            # Notes rows don't have csk attribute on player_th.
            if not player_th.get("csk"):
                continue
            link = player_th.find("a")
            if link is None:
                continue
            name = link.get_text(strip=True)
            if not name:
                continue

            # Walk y1 through y6 in order. Track the latest filled cell.
            last_filled_year: str | None = None
            last_filled_type = "guaranteed"
            for i in range(1, 7):
                cell = tr.find("td", attrs={"data-stat": f"y{i}"})
                if cell is None:
                    continue
                cls = cell.get("class", []) or []
                # Empty cell — "iz" class signals contract has ended.
                if "iz" in cls:
                    continue
                # This cell has a salary. Update tracker.
                year_label = year_map.get(f"y{i}", "")
                last_filled_year = year_label
                if "salary-pl" in cls:
                    last_filled_type = "player_option"
                elif "salary-tm" in cls:
                    last_filled_type = "team_option"
                elif "salary-et" in cls:
                    last_filled_type = "et_option"
                else:
                    last_filled_type = "guaranteed"

            if not last_filled_year:
                continue

            # Compute signing year. For guaranteed deals: signing season =
            # year after the last year. For options, assume exercised
            # (most options are taken when they're meaningful money — and
            # for non-meaningful, the player would still rather have an
            # extra year of guaranteed cap hit than be a FA at minimum).
            end_year_int = int(last_filled_year.split("-")[0])
            signing_year_int = end_year_int + 1
            signing_season = f"{signing_year_int}-{str(signing_year_int + 1)[-2:]}"
            years_remaining = end_year_int - int(SEASONS[0].split("-")[0]) + 1

            out[normalize(name)] = {
                "end_season":      last_filled_year,
                "last_year_type":  last_filled_type,
                "signing_season":  signing_season,
                "years_remaining": max(1, years_remaining),
                "current_team":    team,
            }

    _pkl_save(path, out)
    return out


def get_player_contract_info(player_name: str) -> dict | None:
    """Single-player lookup wrapping the bulk fetch. Returns None for
    free agents (no active multi-year contract on file)."""
    try:
        return fetch_contract_end_years().get(normalize(player_name))
    except Exception:
        return None


# project_contract_inputs / CAP_GROWTH_RATE were removed when we reverted
# the forward-projection experiment (see commit 76560d3). The predictor
# now answers "what would a GM pay TODAY" — no future cap / age / tenure
# projection. If we want to bring forward projection back as an opt-in
# toggle, the logic is recoverable from git history.


# ── Service years + team tenure ────────────────────────────────────────────
# Used by the Contract Predictor for max-contract tier eligibility (7+/10+
# years of service unlock higher max % under the CBA) and for Bird-rights
# proxy (consecutive seasons with current team).

def get_player_service_info(player_name: str) -> dict:
    """Derive service years + current-team tenure from career data.

    Returns:
        {
            "service_years": int, count of NBA seasons appeared in,
            "current_team": str, most recent season's team abbreviation,
            "team_tenure":  int, consecutive most-recent seasons on current team,
        }

    Note: "service years" here counts SEASONS in NBA data, which approximates
    the CBA definition. The CBA's "years of service" rule has nuances (a year
    on a two-way contract counts; rookie year counts even if mid-season call-up)
    that we don't try to model, close enough for a market signal.
    """
    try:
        career = fetch_player_full_career(player_name)
    except Exception:
        return {"service_years": 0, "current_team": "", "team_tenure": 0}
    if career.empty:
        return {"service_years": 0, "current_team": "", "team_tenure": 0}

    # Sort by season ascending (career data should already be sorted but be defensive).
    career = career.sort_values("Season").reset_index(drop=True)
    service = len(career)

    # Most recent season's team. If multi-team (TOT), use that as-is.
    last_team = str(career.iloc[-1]["Team"])

    # Tenure: walk backward, count consecutive seasons on last_team.
    tenure = 0
    for _, row in career.iloc[::-1].iterrows():
        if str(row["Team"]) == last_team:
            tenure += 1
        else:
            break

    return {
        "service_years": int(service),
        "current_team": last_team,
        "team_tenure": int(tenure),
    }


# ── Max-contract + supermax eligibility ────────────────────────────────────
# CBA-based max contract percentages by years of service, plus the
# "Designated" bumps that elite players unlock via All-NBA selections.

def get_max_contract_eligibility(player_name: str, current_season: str) -> dict:
    """Compute a player's CBA max-contract percentage and supermax eligibility.

    Standard max % by years of service:
        0-6 years:   25% of cap
        7-9 years:   30% of cap
        10+ years:   35% of cap

    Designated bumps (require qualifying All-NBA + tenure with current team):
        Designated Rookie Extension: 25% → 30% (for players entering year 5
            who've made All-NBA / DPOY / MVP recently AND are with the
            team that drafted them).
        Designated Veteran Extension (supermax): 30% → 35% (for vets with
            7+ years service who've made All-NBA recently AND are tenured
            with their team).

    Qualifying All-NBA = 1 selection in immediately preceding season
    OR 2 selections in the past 3 seasons.

    Returns:
        {
            "service_years":    int,
            "team_tenure":      int,
            "current_team":     str,
            "recent_all_nba":   list[dict] of selections in last 3 seasons,
            "qualifying":       bool, has qualifying All-NBA performance,
            "max_pct":          float, final % of cap they can earn,
            "supermax_tier":    str, "Designated Vet (35%)" / "Designated Rookie (30%)"
                                / "Max 35%" / "Max 30%" / "Max 25%",
        }
    """
    info = get_player_service_info(player_name)
    recent = get_all_nba_in_window(player_name, current_season, 3)
    immediate = get_all_nba_in_window(player_name, current_season, 1)
    qualifying = (len(immediate) >= 1) or (len(recent) >= 2)

    service = info["service_years"]
    tenure = info["team_tenure"]

    # Standard max % by service tier.
    if service <= 6:
        base_max = 0.25
        base_tier = "Max 25%"
    elif service <= 9:
        base_max = 0.30
        base_tier = "Max 30%"
    else:
        base_max = 0.35
        base_tier = "Max 35%"

    # Designated bumps require qualifying All-NBA + tenure with team.
    # Two cases to handle:
    #   - Young player still on rookie scale (Wemby year 3, tenure 3): only
    #     needs to have been with their drafting team since draft (tenure == service).
    #   - Older player (7+ yrs): needs tenure ≥ 4 as proxy for "tenured
    #     enough with current team" to qualify for Designated Vet.
    max_pct = base_max
    supermax_tier = base_tier
    if qualifying:
        # Designated Rookie path: under 7 yrs service AND with drafting team.
        # tenure ≥ min(service, 4) catches both "drafted-team rookies" and
        # "extension-year veterans (year 4-5) tenured with drafting team."
        if service <= 6 and tenure >= min(service, 4):
            max_pct = 0.30
            supermax_tier = "Designated Rookie (30%)"
        # Designated Vet path: 7+ yrs service AND tenured with current team.
        elif service >= 7 and tenure >= 4:
            max_pct = 0.35
            supermax_tier = "Designated Vet (35%)"

    return {
        "service_years":  service,
        "team_tenure":    tenure,
        "current_team":   info["current_team"],
        "recent_all_nba": recent,
        "qualifying":     qualifying,
        "max_pct":        max_pct,
        "supermax_tier":  supermax_tier,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_player_career_all_seasons(player_name: str, playoffs: bool = False) -> pd.DataFrame:
    """Return every season a player appears in raw data, regardless of minutes played.

    Uses per-season apply_rankings so score_rank reflects their true league rank
    that year.  No minutes threshold, injury years, cameo seasons all included.
    playoffs=True reads postseason caches instead of regular season.
    """
    name_norm = normalize(player_name)
    frames: list[pd.DataFrame] = []
    for season in SEASONS:
        try:
            raw  = build_raw(season, playoffs)
            if raw.empty:
                continue
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
def build_ranked_projected(season: str, playoffs: bool = False) -> pd.DataFrame:
    """Full pipeline, build_raw + apply_rankings + apply_projections, cached.

    Pass playoffs=True for postseason data (separate cache entry, separate
    on-disk parquet).

    NOTE: Uses @st.cache_resource (singleton, no copy on hit) instead of
    @st.cache_data. Callers MUST .copy() before mutating columns.
    """
    return apply_projections(apply_rankings(build_raw(season, playoffs)))


def apply_rankings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Empty input (e.g. playoff data missing for a season) — return early
    # with the expected columns so downstream code (apply_projections,
    # apply_rankings consumers) can handle gracefully without KeyErrors.
    if df.empty:
        for col in ("score_rank", "salary_rank", "rank_diff"):
            df[col] = pd.Series(dtype=int)
        return df
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
