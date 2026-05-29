"""Shared logic + UI helpers for the HoopsValue Streamlit app.

╔══════════════════════════════════════════════════════════════════════════╗
║  TABLE OF CONTENTS                                                       ║
║                                                                          ║
║  1.  Configuration constants                                             ║
║      - CACHE_DIR, SEASONS, DEFAULT_MIN_THRESHOLD                         ║
║      - SALARY_CAP_M, cap_dollars()                                       ║
║      - Contract calibration: AGE / POSITION multipliers, thresholds      ║
║      - age_bucket()                                                      ║
║      - LEAGUE_PACE, pace_factor(), pace_adjusted_barrett()               ║
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
║      - stat_card_html()                                                  ║
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
║      - build_raw() — raw per-season stats + Barrett Score                ║
║      - apply_rankings(), apply_projections()                             ║
║      - build_ranked_projected() — entry point used by every page         ║
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
║  a. @st.cache_data — in-memory per Streamlit session. Keyed by all       ║
║     function arguments. Use TTL for data that ages (current-season       ║
║     stats: 1h; historical: 24h+; truly static: no TTL).                  ║
║                                                                          ║
║  b. Disk cache (CACHE_DIR/*.{parquet,pkl}) — survives process restart.   ║
║     Filenames carry a version suffix to invalidate on schema changes.    ║
║     Conventions:                                                         ║
║       raw_<season>_<FORMULA_VERSION>.parquet     — main ranking output   ║
║       raw_<season>_playoff_<PLAYOFF_VERSION>_<FORMULA_VERSION>.parquet   ║
║       splits_<season>_<FORMULA_VERSION>.pkl                              ║
║       bref_positions_<year>_v<cache_v>.pkl       — per-scraper cache_v   ║
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
import plotly.express as px
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import leaguedashplayerstats, playercareerstats, playerindex
from nba_api.stats.static import players as nba_players_static
from nba_api.stats.static import teams as nba_teams_static


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

      ELITE     — Top 30 by career-weighted score, OR career_score ≥ 28.
                  These guys (LeBron, Curry, KD, Harden) hold their
                  contract value until ~37 then decline slowly (1.5% / yr).
                  Their body fails before the market does.

      ROTATION  — Top 100, OR career_score ≥ 18. Real NBA starters who
                  take moderate age discounts (~3% / yr past 28).

      DEPTH     — Everyone else. Bench / role guys who get steeply
                  discounted past 28 (6% / yr) — flooded out by younger
                  options at similar production.

    All tiers get an additional decline past age 35 (body fails). The
    floor is 0.40 — a 40+ year-old depth player isn't completely
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
    But the durability question is real for chronic cases — Embiid
    averaging 30 GP/yr over 3 years signals teams can't bank on him.
    That deserves its own multiplier separate from production.

    Computed on trailing N seasons (default 3). All seasons count toward
    the GP sum, including ones below the GP ≥ 40 "healthy" filter — we
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
        Rui after the 2023 WCF run — all got paid off ONE playoff
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

    /* Pin the global playoff-mode toggle to the right edge of the nav bar.
       Streamlit assigns the wrapping div class 'st-key-playoff_nav_toggle'
       when we use st.container(key='playoff_nav_toggle'). */
    .st-key-playoff_nav_toggle {
        position: fixed !important;
        top: 0.45rem !important;
        right: 1rem !important;
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
        color: #aaa !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    .st-key-playoff_nav_toggle:hover label p { color: #fff !important; }

    /* Make the toggle's wrapper hierarchy vanish from layout entirely.
       display:contents removes an element from the box tree (its children
       render in its place); since our only child is position:fixed, no
       flow space is claimed. Apply to every ancestor that might wrap the
       keyed container so it doesn't matter which one Streamlit creates. */
    .element-container:has(.st-key-playoff_nav_toggle),
    [data-testid="element-container"]:has(.st-key-playoff_nav_toggle),
    [data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-playoff_nav_toggle),
    [data-testid="stVerticalBlock"]:has(.st-key-playoff_nav_toggle):not(.st-key-playoff_nav_toggle) {
        display: contents !important;
    }

    /* Belt-and-suspenders: zero out any remaining wrapper dimensions in
       case display:contents isn't enough on some browser. */
    .element-container:has(.st-key-playoff_nav_toggle),
    [data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-playoff_nav_toggle),
    [data-testid="stVerticalBlock"]:has(.st-key-playoff_nav_toggle):not(.st-key-playoff_nav_toggle) {
        min-height: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
    }

    /* Page block-container top padding — fits the nav bar (3rem tall) plus
       comfortable breathing room before the page title. Goldilocks number:
       3.8rem was too generous (huge empty gap), 3.2rem was too tight
       (title mashed against the nav). 4.5rem leaves a clean ~1.5rem of
       space between nav and title, matching the bottom padding rhythm. */
    .main .block-container,
    section.main > .block-container,
    [data-testid="stMain"] .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-top: 4.5rem !important;
    }

    /* Same trick for the components.html hide-badge iframe — height=0 in
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
</style>
"""

_NAV_PAGES = [
    ("Current Rankings",   "/Rankings"),
    ("Search Player",      "/Search"),
    ("Legacy",             "/Legacy"),
    ("Team Analysis",      "/Team_Analysis"),
    # Trades tab removed — page lives at /Trades_disabled.py (kept for
    # easy revival) and a backup of the verdict-aware version is in
    # Trades_backup.py at repo root.
    ("Contract Predictor", "/Contract_Predictor"),
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


def render_page_chrome() -> None:
    """One-call page chrome: COMMON_CSS + the hide-badge MutationObserver.

    Call this once near the top of every page right after st.set_page_config.
    Replaces ~20 lines of identical components.html + st.markdown boilerplate
    that used to be copy-pasted into every page. Adding new pages? Just call
    render_page_chrome() and you're done — no need to remember which iframe
    script to copy.
    """
    import streamlit.components.v1 as _components
    st.markdown(COMMON_CSS, unsafe_allow_html=True)
    _components.html(_HIDE_BADGE_SCRIPT, height=0)


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

    # Playoff toggle — keyed container, pinned via CSS in COMMON_CSS
    with st.container(key="playoff_nav_toggle"):
        st.toggle(
            "Playoff mode",
            value=st.session_state.get("playoff_mode", False),
            key="playoff_mode",
            help=_PLAYOFF_HELP,
        )


def render_playoff_toggle() -> bool:
    """Shared playoff-mode toggle, backed by st.session_state.playoff_mode.

    Same key on every page — Streamlit allows reusing widget keys across
    different page renders since each page is its own script execution. The
    flag persists across page navigations because session_state.playoff_mode
    survives between scripts in the same multi-page app.
    """
    return st.toggle(
        "Playoff mode",
        value=st.session_state.get("playoff_mode", False),
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


def stat_card_html(label: str, value: str, sub: str, color: str) -> str:
    """Standard branded stat card. Returns HTML string — pass to st.markdown
    with unsafe_allow_html=True.

    Used by Track Record (accuracy summary tiles) and any page that wants a
    consistent "label + big number + subtitle" tile. Colors come from the
    site's accent palette (#e63946 red, #2ecc71 green, #16d4c1 teal,
    #f39c12 orange, #9b59b6 purple, etc.). Color is also used at 40% alpha
    for the border tint.
    """
    return (
        f'<div style="background:rgba(255,255,255,0.03); '
        f'border:1px solid {color}40; border-radius:10px; '
        f'padding:1.2rem 1.5rem; text-align:center;">'
        f'<div style="font-size:0.72rem; color:#888; letter-spacing:0.08em; '
        f'text-transform:uppercase; font-weight:600; margin-bottom:0.4rem;">'
        f'{label}</div>'
        f'<div style="font-size:2.4rem; font-weight:700; color:{color}; '
        f'line-height:1;">{value}</div>'
        f'<div style="font-size:0.78rem; color:#999; margin-top:0.4rem;">'
        f'{sub}</div>'
        f'</div>'
    )


# ── Name matching ──────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def season_to_espn_year(season: str) -> int:
    return int(season.split("-")[0]) + 1


# ── Buyout signings ─────────────────────────────────────────────────────────
# A bought-out player's new salary isn't a function of his stats: his money is
# already guaranteed by the old team's residual, so he joins a contender on a
# CBA exception-level deal (veteran minimum / taxpayer-MLE). Empirically, across
# every "came off a big deal, signed small" case in 2012+ (105 of them),
# predicting the veteran minimum lands within 5% of cap on 105/105. Because a
# buyout is a PUBLIC transaction known BEFORE the new signing, we flag these and
# predict the minimum — something a stats model can't anticipate.
#
# Veteran minimum as a fraction of cap by years of service (modern scale).
_CBA_MIN_PCT_BY_SVC = {0: 0.010, 1: 0.016, 2: 0.018, 3: 0.019, 4: 0.019,
                       5: 0.020, 6: 0.021, 7: 0.022, 8: 0.022, 9: 0.023}


def cba_min_pct(service_years: float) -> float:
    """Veteran minimum as % of cap by years of service (10+ ≈ 2.6%)."""
    return _CBA_MIN_PCT_BY_SVC.get(int(service_years or 0), 0.026)


# Verified true buyouts: player waived/bought out from an active contract (old
# team keeps paying a residual), then signed a NEW, smaller deal. Each is a
# documented public transaction. NOT included: contracts that merely expired
# (e.g. Otto Porter), or Russell Westbrook 2022-23 (finished a real $47M option
# that year — a mislabel handled separately, not a buyout-market signing).
KNOWN_BUYOUTS = {
    ("Josh Smith",        "2014-15"),  # Detroit stretch-waive (Dec '14) → Houston
    ("Andre Drummond",    "2020-21"),  # Cleveland buyout (Mar '21) → Lakers
    ("Blake Griffin",     "2020-21"),  # Detroit buyout (Mar '21) → Brooklyn
    ("Goran Dragic",      "2021-22"),  # Spurs buyout (Feb '22) → Brooklyn
    ("Kevin Love",        "2022-23"),  # Cleveland buyout (Feb '23) → Miami
    ("Kyle Lowry",        "2023-24"),  # Charlotte buyout (Feb '24) → 76ers ($2.8M)
    ("Spencer Dinwiddie", "2023-24"),  # Toronto waive (Feb '24) → Lakers
    ("Ben Simmons",       "2024-25"),  # Brooklyn buyout (Feb '25) → Clippers
    ("Deandre Ayton",     "2025-26"),  # Portland buyout (Jul '25) → Lakers ($8.1M)
    ("Bradley Beal",      "2025-26"),  # Phoenix buyout (Jul '25) → Clippers ($5.4M)
    ("Marcus Smart",      "2025-26"),  # Washington buyout (Jul '25) → Lakers
}
_BUYOUT_KEYS = {(normalize(p), s) for (p, s) in KNOWN_BUYOUTS}


def is_known_buyout(player: str, season: str) -> bool:
    """True if (player, season) is a verified buyout-market signing."""
    return (normalize(str(player)), season) in _BUYOUT_KEYS


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
    """Availability multiplier — punishes missed games + light minutes.

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
def fetch_league_stats(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Per-game player stats for one season.

    season_type controls regular season vs playoffs — accepts NBA Stats API's
    own values: "Regular Season" (default) or "Playoffs". Each variant gets
    its own disk cache so the two modes don't clobber each other.
    """
    suffix = "_playoff" if season_type == "Playoffs" else ""
    path = _dc_path(f"league_stats_{season.replace('-','_')}{suffix}.parquet")
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
                season_type_all_star=season_type,
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
    if _dc_fresh(path, season=season):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    from nba_api.stats.endpoints import leaguedashplayerstats as _ldps
    time.sleep(0.5)
    result = None
    delay = 1
    attempts = 0
    while result is None and attempts < 8:
        try:
            result = _ldps.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
                season_type_all_star=season_type,
                measure_type_detailed_defense="Advanced",
            )
        except Exception:
            attempts += 1
            time.sleep(delay)
            delay = min(delay * 2, 30)
    if result is None:
        return pd.DataFrame()
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
    page — used for pre-1996 playoff seasons since the NBA Stats API
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
    """Summarize a list of players in a given season — their Barrett Scores
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
        "side_a_picks": "Plus Marc Gasol's draft rights — became multi-time All-Star",
        "side_b_picks": "Lakers 2008 + 2010 first-round picks",
        "winner":      "side_a",
        "verdict":     "Lakers. Three Finals in three years, two titles, generational frontcourt. One of the most lopsided trades in NBA history at the time it was made.",
        "key_points": [
            "Lakers made the Finals 3 straight years (2008, 2009, 2010)",
            "Won championships in 2008-09 and 2009-10",
            "Memphis eventually got Marc Gasol back too — but won zero playoff series with the original return",
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
        "verdict":     "Toronto. Won the 2019 title — the only championship in franchise history — even though Kawhi left in free agency that summer. Worth it any day.",
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
            "Lakers won the 2019-20 championship — Bubble title",
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
            "Cavaliers made the 2018 Finals (lost to GSW) — final LeBron Finals run",
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
            "Iverson lasted 54 games in Detroit — fit was a disaster",
            "Pistons missed the playoffs the following year for the first time in 8 seasons",
        ],
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
def fetch_player_full_career(player_name: str, playoffs: bool = False) -> pd.DataFrame:
    """Full per-season career stats for one player: raw counting stats from
    fetch_league_stats joined with Barrett Score / rank from build_raw +
    apply_rankings. One row per season the player appeared in.

    playoffs=True uses postseason data — pulls from the playoff-cached
    league_stats and raw parquets. Pre-1996 playoff data isn't seeded yet
    (different BBRef URL), so those seasons just won't appear.

    Only reads seasons that are already on disk — view-time requests must
    NEVER trigger fresh BBRef scrapes. seed_cache.py populates the disk."""
    name_norm = normalize(player_name)
    season_type = "Playoffs" if playoffs else "Regular Season"
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
    """Barrett Score per season pulled directly from build_raw — guaranteed to
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
    season on disk where the player appears. Disk-only — never triggers a
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
def fetch_rookie_scale_players(season: str, cache_v: int = 2) -> set:
    """Returns a set of normalized names for first-round picks currently
    inside their rookie scale contract (years 1-4 post-draft).

    For season "2025-26": the most recent draft was Summer 2025 (rookies).
    Year-4 players were drafted Summer 2022. Range: 2022-2025.

    cache_v=2: fixes a year-range bug in v1 that missed year-4 rookies
    (e.g. 2022 draftees in the 2025-26 season — Jalen Duren, Tari Eason).
    """
    path = _dc_path(f"rookie_scale_{season.replace('-','_')}_v{cache_v}.pkl")
    if _dc_fresh(path, ttl=86400):
        try:
            return _pkl_load(path)
        except Exception:
            pass
    try:
        end_year = int(season.split("-")[0]) + 1  # "2025-26" → 2026

        # Rookie scale spans years 1-4. For season ending in end_year,
        # year-4 players were drafted (end_year - 4), year-1 rookies drafted
        # (end_year - 1). Range: end_year-4 inclusive, end_year exclusive.
        rookie_draft_years = set(range(end_year - 4, end_year))

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


# ── D-LEBRON proxy for pre-2009 seasons ────────────────────────────────────
# Real D-LEBRON (from BBall-Index) only goes back to 2009-10. For older
# eras, the defensive component of Barrett Score would otherwise be 0 —
# heavily penalizing defensive specialists (Olajuwon, Mutombo, Pippen,
# Bruce Bowen, etc.) in pre-2009 contract validation.
#
# Coefficients fit via OLS on the 2009-2025 sample (n=9,649 player-seasons
# with MPG ≥ 10, see scripts/experiment_dlebron_proxy.py). Inputs are
# per-game stats from fetch_league_stats (NBA Stats API default).
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


def dlebron_proxy(stl: float, blk: float, dreb: float, pf: float) -> float:
    """Estimate D-LEBRON from per-game box stats (pre-2009 fallback).

    Args:
        stl, blk, dreb, pf:  per-game defensive stats (as returned by
                             fetch_league_stats — already per-game)

    Returns:
        Proxy D-LEBRON value in roughly the real D-LEBRON scale
        (typically -2 to +3). Falls back to 0.0 on missing input.
    """
    if stl is None or blk is None or dreb is None or pf is None:
        return 0.0
    c = _DLEBRON_PROXY_COEFS
    return (
        c["intercept"]
        + c["STL"]  * stl
        + c["BLK"]  * blk
        + c["DREB"] * dreb
        + c["PF"]   * pf
    )


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
    False here triggers a network rebuild, which is expensive — so we
    only do it when the cache is genuinely stale.

    Use `_raw_disk_exists()` for view-time reads instead — those should
    serve cached data even when slightly stale (don't drop a season from
    a player's career arc just because the cache hasn't been touched in
    a few hours).
    """
    p = _raw_disk_path(season, playoffs)
    if not p.exists():
        return False
    # Current season refreshes every hour; historical seasons every 30 days
    ttl = 3600 if season == SEASONS[0] else 30 * 86_400
    return (time.time() - p.stat().st_mtime) < ttl


def _raw_disk_exists(season: str, playoffs: bool = False) -> bool:
    """True if the on-disk parquet exists, regardless of TTL.

    Use this in view-time read paths (player career arcs, position-peer
    distributions, etc.) where serving slightly-stale cache is better
    than silently dropping the season entirely. The freshness check is
    for write-side decisions about when to refresh, not read-side gating.
    """
    return _raw_disk_path(season, playoffs).exists()

@st.cache_data(ttl=3600, show_spinner="Building rankings...")
def build_raw(season: str, playoffs: bool = False) -> pd.DataFrame:
    # ── Disk cache hit: load parquet instead of hitting the APIs ──────────────
    if _raw_disk_fresh(season, playoffs):
        try:
            cached = pd.read_parquet(_raw_disk_path(season, playoffs))
            # Sanitize old caches that captured BBRef's Hall-of-Fame asterisk
            # in player names ("Michael Jordan*"). Newly-built parquets won't
            # have these, but on-disk pre-fix files do.
            if "Player" in cached.columns:
                cached["Player"] = (
                    cached["Player"].astype(str).str.rstrip("*").str.strip()
                )
            return cached
        except Exception:
            pass  # corrupted file — fall through to live fetch

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
    # See utils.dlebron_proxy + scripts/experiment_dlebron_proxy.py.
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
        result.to_parquet(_raw_disk_path(season, playoffs), index=False)
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
    if _dc_fresh(path, ttl=3600):
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
        "signing_season":  "YYYY-YY" — projected season player signs NEXT deal,
        "years_remaining": int — how many seasons (including current) until end,
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
            "service_years": int — count of NBA seasons appeared in,
            "current_team": str — most recent season's team abbreviation,
            "team_tenure":  int — consecutive most-recent seasons on current team,
        }

    Note: "service years" here counts SEASONS in NBA data, which approximates
    the CBA definition. The CBA's "years of service" rule has nuances (a year
    on a two-way contract counts; rookie year counts even if mid-season call-up)
    that we don't try to model — close enough for a market signal.
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
            "qualifying":       bool — has qualifying All-NBA performance,
            "max_pct":          float — final % of cap they can earn,
            "supermax_tier":    str — "Designated Vet (35%)" / "Designated Rookie (30%)"
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
    that year.  No minutes threshold — injury years, cameo seasons all included.
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
    """Full pipeline — build_raw + apply_rankings + apply_projections — cached.

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
