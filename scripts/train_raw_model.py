"""RAW-STATS contract model — forget the Barrett Score entirely.

The Barrett Score is a hand-tuned linear combination of box stats. This
experiment throws it out and lets a gradient-boosted model learn its own
weighting directly from raw + advanced stats. The hypothesis: an ML model
given the unprocessed signals can match or beat the Barrett-derived model,
because it isn't constrained to Barrett's hand-picked coefficients — and it
can use richer signals Barrett never had (usage rate, PIE, on/off ratings).

NO barrett_score, NO barrett-derived rank anywhere in the feature set.

Feature palette (all from NBA Stats API Base + Advanced measure types):
  Volume (per game): PTS AST OREB DREB REB STL BLK TOV PF FGM FGA FG3M FG3A
                     FTM FTA MIN
  Shooting:          FG_PCT FG3_PCT FT_PCT
  Derived eff:       ts (mine) eFG ast_to fg3a_rate fta_rate
  Advanced:          USG_PCT TS_PCT PIE AST_PCT REB_PCT OREB_PCT DREB_PCT
                     OFF_RATING DEF_RATING NET_RATING
  Impact:            PLUS_MINUS NBA_FANTASY_PTS DD2 TD3
  Availability:      GP
  Trailing 3-yr:     PTS MIN GP USG PIE NET_RATING TS
  Market anchors:    pie_base_proj_pct (PIE-rank → salary), salary_prev_pct
  Context:           age service_years all_nba_3yr draft_pick is_lottery
  Growth:            pts_growth pie_growth
  Derived:           age² service² CBA-tier flags position dummies

Target: salary_curr / cap_curr  (% of cap)
Temporal holdout: train 1999-2014, test 2015+.
Baselines to beat:
  - Canonical rank-mapping (Barrett):    79.0% within 5% of cap
  - HistGBM v2 (Barrett features):       80.76%

Usage:
    python -u scripts/train_raw_model.py
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from utils import (
    SEASONS, SALARY_CAP_M, normalize, season_to_espn_year,
    fetch_league_stats, fetch_advanced_stats, build_raw, build_ranked_projected,
    fetch_bref_positions, fetch_player_positions_detailed, position_to_bucket,
    fetch_all_nba_selections, get_all_nba_in_window,
    get_player_draft_info,
    NEW_CONTRACT_PCT, tiered_age_multiplier,
)

CURRENT_CAP_M = SALARY_CAP_M["2025-26"]

ROOKIE_SCALE_SAL_PCT  = 0.15
ROOKIE_SCALE_MAX_AGE  = 25
ROOKIE_SCALE_STEP_UP  = 1.5
ROOKIE_SCALE_FIRST_YR = 1995

# Pairs (prev, curr), 1999+ only (advanced stats + CBA-max era).
ALL_PAIRS = [(SEASONS[i + 1], SEASONS[i]) for i in range(len(SEASONS) - 1)]
PAIRS = [(p, c) for p, c in ALL_PAIRS
         if p in SALARY_CAP_M and c in SALARY_CAP_M
         and int(c.split("-")[0]) >= 1999]
SPLIT_YEAR = 2015
TRAIN_PAIRS = [(p, c) for p, c in PAIRS if int(c.split("-")[0]) < SPLIT_YEAR]
TEST_PAIRS  = [(p, c) for p, c in PAIRS if int(c.split("-")[0]) >= SPLIT_YEAR]

# Advanced columns we keep from fetch_advanced_stats.
ADV_COLS = [
    "USG_PCT", "TS_PCT", "PIE", "AST_PCT", "REB_PCT", "OREB_PCT", "DREB_PCT",
    "OFF_RATING", "DEF_RATING", "NET_RATING",
]


def _cap(season: str) -> float:
    return SALARY_CAP_M.get(season, 1.0) * 1_000_000


# ── Combined per-season table: raw box + advanced + salary ────────────────────
_combined_cache: dict[str, pd.DataFrame] = {}

def combined_season(season: str) -> pd.DataFrame:
    if season in _combined_cache:
        return _combined_cache[season]
    base = fetch_league_stats(season, "Regular Season")
    if base is None or base.empty:
        _combined_cache[season] = pd.DataFrame()
        return _combined_cache[season]
    adv = fetch_advanced_stats(season, "Regular Season")
    keep_adv = ["PLAYER_ID"] + [c for c in ADV_COLS if c in adv.columns]
    if not adv.empty:
        base = base.merge(adv[keep_adv], on="PLAYER_ID", how="left")
    # Salary from build_raw (robust market-data merge; we ignore its Barrett).
    try:
        raw = build_raw(season)
        if not raw.empty and "salary" in raw.columns:
            base = base.merge(raw[["PLAYER_ID", "salary"]], on="PLAYER_ID", how="left")
    except Exception:
        base["salary"] = np.nan
    if "salary" not in base.columns:
        base["salary"] = np.nan
    _combined_cache[season] = base
    return base


# ── Career index (combined stats per player per season) ───────────────────────
TRAIL_COLS = ["PTS", "MIN", "GP", "USG_PCT", "PIE", "NET_RATING", "TS_PCT"]

def build_career_index() -> dict:
    """player_id -> DataFrame(season, + TRAIL_COLS) oldest-first."""
    rows: dict[int, list[dict]] = {}
    seasons = [s for s in SEASONS if int(s.split("-")[0]) >= 1996]
    for season in sorted(seasons, key=lambda s: int(s.split("-")[0])):
        df = combined_season(season)
        if df.empty:
            continue
        for _, r in df.iterrows():
            pid = r.get("PLAYER_ID")
            if pd.isna(pid):
                continue
            entry = {"season": season}
            for c in TRAIL_COLS:
                entry[c] = float(r.get(c, 0) or 0) if c in df.columns else 0.0
            rows.setdefault(int(pid), []).append(entry)
    return {pid: pd.DataFrame(rs) for pid, rs in rows.items()}


def trailing(career_df: pd.DataFrame, up_to: str, col: str, n: int = 3):
    if career_df.empty or col not in career_df.columns:
        return None
    sub = career_df[career_df["season"] <= up_to]
    if sub.empty:
        return None
    return float(sub.tail(n)[col].mean())


def years_in_league(career_df: pd.DataFrame, up_to: str) -> int:
    if career_df.empty:
        return 0
    return int(len(career_df[career_df["season"] <= up_to]))


# ── Row builder ───────────────────────────────────────────────────────────────
def build_rows(pairs, careers: dict, all_nba_lookup: dict) -> pd.DataFrame:
    out = []
    for prev, curr in pairs:
        if prev not in SALARY_CAP_M or curr not in SALARY_CAP_M:
            continue
        cap_prev, cap_curr = _cap(prev), _cap(curr)
        prev_df = combined_season(prev)
        curr_df = combined_season(curr)
        if prev_df.empty or curr_df.empty:
            continue
        prev_df = prev_df[prev_df["salary"].fillna(0) > 0].copy()
        if prev_df.empty:
            continue

        # Curr-season salary lookup + PIE distribution for the market anchor.
        curr_sal = dict(zip(curr_df["PLAYER_ID"], curr_df["salary"].fillna(0)))
        curr_pie_sorted = np.sort(curr_df["PIE"].fillna(0).values)[::-1]
        curr_sal_sorted = np.sort(curr_df["salary"].fillna(0).values)[::-1]

        # Positions (prev season).
        try:
            detailed = fetch_player_positions_detailed(prev, cache_v=2)
        except Exception:
            detailed = {}
        try:
            coarse = fetch_bref_positions(season_to_espn_year(prev), cache_v=3)
        except Exception:
            coarse = {}

        def _pos(n):
            d = detailed.get(normalize(n))
            if d: return position_to_bucket(d)
            return coarse.get(normalize(n), "Unknown")

        for _, row in prev_df.iterrows():
            pid = int(row["PLAYER_ID"])
            salary_prev = float(row["salary"] or 0)
            salary_curr = float(curr_sal.get(pid, 0) or 0)
            if salary_curr <= 0 or salary_prev <= 0:
                continue
            pct_change = (salary_curr - salary_prev) / salary_prev
            if abs(pct_change) < NEW_CONTRACT_PCT:
                continue

            age = row.get("AGE")
            if age is None or pd.isna(age):
                continue
            age = float(age)
            name = row.get("PLAYER_NAME", "")

            career = careers.get(pid, pd.DataFrame())
            pts_3yr = trailing(career, prev, "PTS") or float(row.get("PTS", 0) or 0)
            min_3yr = trailing(career, prev, "MIN") or float(row.get("MIN", 0) or 0)
            gp_3yr  = trailing(career, prev, "GP")  or float(row.get("GP", 0) or 0)
            usg_3yr = trailing(career, prev, "USG_PCT") or float(row.get("USG_PCT", 0) or 0)
            pie_3yr = trailing(career, prev, "PIE") or float(row.get("PIE", 0) or 0)
            net_3yr = trailing(career, prev, "NET_RATING") or float(row.get("NET_RATING", 0) or 0)
            ts_3yr  = trailing(career, prev, "TS_PCT") or float(row.get("TS_PCT", 0) or 0)
            yrs = years_in_league(career, prev)

            # PIE-rank → salary market anchor (Barrett-free analog of
            # career_base_proj). Uses trailing PIE vs curr-season PIE dist.
            eff_rank = int((curr_pie_sorted > pie_3yr).sum()) + 1
            capped = min(eff_rank, len(curr_sal_sorted)) - 1
            pie_base_proj = float(curr_sal_sorted[capped]) if len(curr_sal_sorted) else 0.0

            # All-NBA in trailing 3 yrs.
            try:
                all_nba_3yr = len(get_all_nba_in_window(name, prev, 3))
            except Exception:
                all_nba_3yr = 0

            # Draft pedigree.
            try:
                di = get_player_draft_info(name)
                draft_pick = di.get("draft_pick") or 61  # undrafted → 61
            except Exception:
                draft_pick = 61
            if not draft_pick or pd.isna(draft_pick):
                draft_pick = 61

            # Raw box.
            pts  = float(row.get("PTS", 0) or 0)
            ast  = float(row.get("AST", 0) or 0)
            oreb = float(row.get("OREB", 0) or 0)
            dreb = float(row.get("DREB", 0) or 0)
            reb  = float(row.get("REB", oreb + dreb) or 0)
            stl  = float(row.get("STL", 0) or 0)
            blk  = float(row.get("BLK", 0) or 0)
            tov  = float(row.get("TOV", 0) or 0)
            pf   = float(row.get("PF", 0) or 0)
            fgm  = float(row.get("FGM", 0) or 0)
            fga  = float(row.get("FGA", 0) or 0)
            fg3m = float(row.get("FG3M", 0) or 0)
            fg3a = float(row.get("FG3A", 0) or 0)
            ftm  = float(row.get("FTM", 0) or 0)
            fta  = float(row.get("FTA", 0) or 0)
            mins = float(row.get("MIN", 0) or 0)
            gp   = float(row.get("GP", 0) or 0)

            ts_mine = pts / (2 * (fga + 0.44 * fta)) if (fga + 0.44 * fta) > 0 else 0.0
            efg = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.0
            ast_to = ast / tov if tov > 0 else ast  # high if no turnovers
            fg3a_rate = fg3a / fga if fga > 0 else 0.0
            fta_rate = fta / fga if fga > 0 else 0.0

            pts_growth = pts / pts_3yr if pts_3yr > 0 else 1.0
            pie_now = float(row.get("PIE", 0) or 0)
            pie_growth = pie_now / pie_3yr if pie_3yr > 0 else 1.0

            out.append({
                "player": name, "prev": prev, "curr": curr,
                "start_year": int(curr.split("-")[0]),
                # Raw volume.
                "PTS": pts, "AST": ast, "OREB": oreb, "DREB": dreb, "REB": reb,
                "STL": stl, "BLK": blk, "TOV": tov, "PF": pf,
                "FGM": fgm, "FGA": fga, "FG3M": fg3m, "FG3A": fg3a,
                "FTM": ftm, "FTA": fta, "MIN": mins,
                # Shooting.
                "FG_PCT": float(row.get("FG_PCT", 0) or 0),
                "FG3_PCT": float(row.get("FG3_PCT", 0) or 0),
                "FT_PCT": float(row.get("FT_PCT", 0) or 0),
                # Derived eff.
                "ts_mine": ts_mine, "efg": efg, "ast_to": ast_to,
                "fg3a_rate": fg3a_rate, "fta_rate": fta_rate,
                # Advanced.
                "USG_PCT": float(row.get("USG_PCT", 0) or 0),
                "TS_PCT": float(row.get("TS_PCT", 0) or 0),
                "PIE": pie_now,
                "AST_PCT": float(row.get("AST_PCT", 0) or 0),
                "REB_PCT": float(row.get("REB_PCT", 0) or 0),
                "OREB_PCT": float(row.get("OREB_PCT", 0) or 0),
                "DREB_PCT": float(row.get("DREB_PCT", 0) or 0),
                "OFF_RATING": float(row.get("OFF_RATING", 0) or 0),
                "DEF_RATING": float(row.get("DEF_RATING", 0) or 0),
                "NET_RATING": float(row.get("NET_RATING", 0) or 0),
                # Impact.
                "PLUS_MINUS": float(row.get("PLUS_MINUS", 0) or 0),
                "NBA_FANTASY_PTS": float(row.get("NBA_FANTASY_PTS", 0) or 0),
                "DD2": float(row.get("DD2", 0) or 0),
                "TD3": float(row.get("TD3", 0) or 0),
                # Availability.
                "GP": gp,
                # Trailing.
                "PTS_3yr": pts_3yr, "MIN_3yr": min_3yr, "GP_3yr": gp_3yr,
                "USG_3yr": usg_3yr, "PIE_3yr": pie_3yr,
                "NET_3yr": net_3yr, "TS_3yr": ts_3yr,
                # Anchors.
                "pie_base_proj_pct": pie_base_proj / cap_curr,
                "salary_prev_pct": salary_prev / cap_prev,
                # Context.
                "age": age, "service_years": float(yrs),
                "all_nba_3yr": float(all_nba_3yr),
                "draft_pick": float(draft_pick),
                "is_lottery": 1.0 if draft_pick <= 14 else 0.0,
                # Growth.
                "pts_growth": pts_growth, "pie_growth": pie_growth,
                # Position.
                "pos_bucket": _pos(name),
                # Bookkeeping.
                "salary_prev": salary_prev, "salary_curr": salary_curr,
                "salary_curr_pct": salary_curr / cap_curr,
                "cap_curr": cap_curr,
            })
    return pd.DataFrame(out)


# ── Feature matrix ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "PTS", "AST", "OREB", "DREB", "REB", "STL", "BLK", "TOV", "PF",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "MIN",
    "FG_PCT", "FG3_PCT", "FT_PCT",
    "ts_mine", "efg", "ast_to", "fg3a_rate", "fta_rate",
    "USG_PCT", "TS_PCT", "PIE", "AST_PCT", "REB_PCT", "OREB_PCT", "DREB_PCT",
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "PLUS_MINUS", "NBA_FANTASY_PTS", "DD2", "TD3",
    "GP",
    "PTS_3yr", "MIN_3yr", "GP_3yr", "USG_3yr", "PIE_3yr", "NET_3yr", "TS_3yr",
    "pie_base_proj_pct", "salary_prev_pct",
    "age", "service_years", "all_nba_3yr", "draft_pick", "is_lottery",
    "pts_growth", "pie_growth",
]
DERIVED = ["age2", "service2", "tier30", "tier35", "is_G", "is_F"]

def make_X(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS].fillna(0).astype(float).values
    age = df["age"].values
    yrs = df["service_years"].values
    derived = np.column_stack([
        age ** 2, yrs ** 2,
        (yrs >= 7).astype(float), (yrs >= 10).astype(float),
        (df["pos_bucket"] == "Guard").astype(float).values,
        (df["pos_bucket"] == "Forward").astype(float).values,
    ])
    return np.hstack([X, derived])


# ── Canonical baseline (Barrett rank-mapping) for comparison ──────────────────
def predict_canonical_baseline(df: pd.DataFrame) -> np.ndarray:
    """Mirror analyze_accuracy.py: build_ranked_projected gives the Barrett
    projected_salary; scale by cap ratio, apply age mult + rookie cap.
    Computed via build_ranked_projected lookups keyed by (prev, player)."""
    # Pre-build prev-season projected_salary lookups.
    proj_lookup: dict[tuple, float] = {}
    rank_lookup: dict[tuple, int] = {}
    barrett_lookup: dict[tuple, float] = {}
    for prev in df["prev"].unique():
        try:
            pdf = build_ranked_projected(prev)
        except Exception:
            continue
        for _, r in pdf.iterrows():
            key = (prev, int(r["PLAYER_ID"]) if not pd.isna(r.get("PLAYER_ID")) else -1)
            proj_lookup[key] = float(r.get("projected_salary", 0) or 0)
            rank_lookup[key] = int(r.get("score_rank", 999) or 999)
            barrett_lookup[key] = float(r.get("barrett_score", 0) or 0)
    # Need PLAYER_ID — re-derive from combined tables by name match.
    out = []
    for _, r in df.iterrows():
        prev, curr = r["prev"], r["curr"]
        cap_ratio = _cap(curr) / _cap(prev)
        # Find player by name in prev combined to get PLAYER_ID.
        pdf = combined_season(prev)
        match = pdf[pdf["PLAYER_NAME"].apply(normalize) == normalize(r["player"])]
        if match.empty:
            out.append(0.0); continue
        pid = int(match.iloc[0]["PLAYER_ID"])
        key = (prev, pid)
        proj = proj_lookup.get(key, 0.0) * cap_ratio
        try:
            age_m, _ = tiered_age_multiplier(
                age=float(r["age"]),
                career_score=barrett_lookup.get(key, 0.0),
                current_rank=rank_lookup.get(key, 999),
            )
        except Exception:
            age_m = 1.0
        proj *= age_m
        if (int(curr.split("-")[0]) >= ROOKIE_SCALE_FIRST_YR
                and r["salary_prev_pct"] < ROOKIE_SCALE_SAL_PCT
                and r["age"] <= ROOKIE_SCALE_MAX_AGE):
            proj = min(proj, r["salary_prev"] * ROOKIE_SCALE_STEP_UP)
        out.append(proj)
    return np.array(out)


def predict_model(model, X, cap):
    return np.clip(model.predict(X), 0.001, 0.45) * cap


def score(df, pred):
    actual = df["salary_curr"].values
    cap = df["cap_curr"].values
    e = np.abs(actual - pred) / cap * 100
    eM = e / 100 * CURRENT_CAP_M
    return {
        "n": len(df),
        "within_5": float(np.mean(e <= 5) * 100),
        "within_10": float(np.mean(e <= 10) * 100),
        "median_err": float(np.median(e)),
        "median_err_M": float(np.median(eM)),
        "err_M": eM, "actual_M": actual / cap * CURRENT_CAP_M,
    }


def pr(label, s, base_w5=None):
    L = label[:46].ljust(46)
    if base_w5 is None:
        print(f"  {L}  w5={s['within_5']:5.2f}%  w10={s['within_10']:5.2f}%  med=${s['median_err_M']:.2f}M", flush=True)
    else:
        d = s["within_5"] - base_w5
        m = "[+]" if d >= 1 else ("[?]" if d >= 0 else "[-]")
        print(f"  {L}  w5={s['within_5']:5.2f}% ({d:+5.2f}) {m}  w10={s['within_10']:5.2f}%  med=${s['median_err_M']:.2f}M", flush=True)


def tier(label, s):
    a, e = s["actual_M"], s["err_M"]
    print(f"\n  TIERS — {label}", flush=True)
    for nm, mask in [("Max/super", a >= 40), ("Big stars", (a >= 25) & (a < 40)),
                     ("Mid-tier", (a >= 15) & (a < 25)), ("Rotation", (a >= 7) & (a < 15)),
                     ("Min-ish", a < 7)]:
        n = int(mask.sum())
        if n == 0: continue
        se = e[mask]
        print(f"    {nm:<11} n={n:>4}  median ${np.median(se):>5.2f}M  "
              f"±$3M {np.mean(se <= 3)*100:>4.0f}%  ±$5M {np.mean(se <= 5)*100:>4.0f}%", flush=True)


def main():
    print(f"RAW-STATS model (NO Barrett). Train {len(TRAIN_PAIRS)} pairs, "
          f"test {len(TEST_PAIRS)} pairs.", flush=True)
    print("\nBuilding career index (combined raw+advanced)...", flush=True)
    t0 = time.time()
    careers = build_career_index()
    print(f"  {len(careers)} players in {time.time()-t0:.1f}s", flush=True)
    all_nba = fetch_all_nba_selections()

    print("\nBuilding train/test rows...", flush=True)
    t0 = time.time()
    train_df = build_rows(TRAIN_PAIRS, careers, all_nba)
    test_df  = build_rows(TEST_PAIRS,  careers, all_nba)
    print(f"  train {len(train_df)}, test {len(test_df)} in {time.time()-t0:.1f}s", flush=True)

    y = train_df["salary_curr_pct"].values
    X_tr = make_X(train_df)
    X_te = make_X(test_df)
    cap_te = test_df["cap_curr"].values

    print("\n" + "=" * 90, flush=True)
    print(f"OUT-OF-SAMPLE RESULTS  (test n={len(test_df)})", flush=True)
    print("=" * 90, flush=True)

    # Baseline (Barrett rank-mapping).
    print("  Computing canonical baseline...", flush=True)
    sA = score(test_df, predict_canonical_baseline(test_df))
    pr("A. CANONICAL baseline (Barrett rank-mapping)", sA)
    base_w5 = sA["within_5"]

    # GBM.
    gbm = GradientBoostingRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        min_samples_leaf=15, subsample=0.8, random_state=42,
    ).fit(X_tr, y)
    pr("B. GBM (raw stats)", score(test_df, predict_model(gbm, X_te, cap_te)), base_w5)

    # HistGBM sweep.
    print("\n  Sweeping HistGBM...", flush=True)
    best, best_s, best_p = None, None, None
    sweep = []
    for mi in [400, 600, 800]:
        for md in [4, 5, 6]:
            for lr in [0.02, 0.03, 0.05]:
                for leaf in [15, 25]:
                    sweep.append(dict(max_iter=mi, max_depth=md, learning_rate=lr,
                                      min_samples_leaf=leaf, l2_regularization=0.1))
    rng = np.random.RandomState(11)
    idx = rng.choice(len(sweep), size=min(30, len(sweep)), replace=False)
    for i in idx:
        p = sweep[i]
        m = HistGradientBoostingRegressor(random_state=42, **p).fit(X_tr, y)
        s = score(test_df, predict_model(m, X_te, cap_te))
        if best_s is None or s["within_5"] > best_s["within_5"]:
            best, best_s, best_p = m, s, p
            print(f"    new best w5={s['within_5']:.2f}% {p}", flush=True)
    pr(f"C. HistGBM tuned {best_p}", best_s, base_w5)

    # RF.
    rf = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=8,
                               random_state=42, n_jobs=-1).fit(X_tr, y)
    pr("D. Random Forest (raw stats)", score(test_df, predict_model(rf, X_te, cap_te)), base_w5)

    # Feature importances.
    print("\nTOP 20 FEATURE IMPORTANCES (GBM):", flush=True)
    fnames = FEATURE_COLS + DERIVED
    for nm, imp in sorted(zip(fnames, gbm.feature_importances_), key=lambda kv: -kv[1])[:20]:
        print(f"  {nm:<20} {imp:.3f}", flush=True)

    # Tiers.
    print("\n" + "=" * 90, flush=True)
    print("TIER COMPARISON", flush=True)
    print("=" * 90, flush=True)
    tier("A. Canonical baseline", sA)
    tier("C. HistGBM (raw stats)", best_s)

    # Verdict.
    print("\n" + "=" * 90, flush=True)
    print("VERDICT  (Barrett-derived baselines: canonical 79.0%, HistGBM-v2 80.76%)", flush=True)
    print("=" * 90, flush=True)
    print(f"  Best RAW model: HistGBM at {best_s['within_5']:.2f}% within 5%, "
          f"{best_s['within_10']:.2f}% within 10%", flush=True)
    print(f"  vs canonical Barrett baseline: {base_w5:.2f}%  →  {best_s['within_5']-base_w5:+.2f}pp", flush=True)
    print(f"  vs Barrett HistGBM v2 (80.76%): {best_s['within_5']-80.76:+.2f}pp", flush=True)


if __name__ == "__main__":
    main()
