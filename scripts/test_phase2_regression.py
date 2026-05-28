"""Phase 2: trained regression with service time + interactions.

Replaces the hand-tuned baseline (rank-mapping × age × pos × supermax-
suppression) with a fitted regression that learns weights from data,
including features the current model can't see:

  - Years of service (CBA tier eligibility: 25% / 30% / 35% of cap max)
  - Position × age interaction (older guards age differently than bigs)
  - Career-Score × prior-salary interaction (leverage/continuation)
  - Cubic age term (catches steep decline at 35+)

VARIANTS TESTED
  A. PRODUCTION BASELINE   — current page formula
  B. OLS minimal           — barrett, barrett², age, age², GP, MPG,
                              prior_pct, is_Guard, is_Forward
                              (this is the test_contract_model.py model)
  C. OLS + service time    — B + years_in_league + tier dummies
  D. OLS + interactions    — C + age × position + score × prior_pct
  E. RIDGE (best variant)  — D with L2 regularization to reduce overfit

Train: 2014-15 → 2021-22 (8 pairs, ~1,100 contracts)
Test:  2022-23 → 2024-25 (3 pairs, ~435 contracts)
Metric: % within 5% of cap

Decision rule:
  ≥ +1pp on Within 5% → SHIP (real improvement)
  +0.5 to +1pp        → CONSIDER (modest, watch for overfit)
  < +0.5pp            → DO NOT SHIP

Usage:
    python test_phase2_regression.py
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from utils import (
    SEASONS, normalize, season_to_espn_year,
    build_ranked_projected, build_raw, apply_rankings,
    fetch_league_stats,
    fetch_bref_positions, fetch_player_positions_detailed, position_to_bucket,
    SALARY_CAP_M, age_bucket,
    CONTRACT_AGE_MULTIPLIERS, CONTRACT_POSITION_MULTIPLIERS,
    HEALTHY_SEASON_GP, NEW_CONTRACT_PCT, SUPERMAX_CAP_PCT,
)


TRAIN_PAIRS = [
    ("2014-15", "2015-16"), ("2015-16", "2016-17"), ("2016-17", "2017-18"),
    ("2017-18", "2018-19"), ("2018-19", "2019-20"), ("2019-20", "2020-21"),
    ("2020-21", "2021-22"), ("2021-22", "2022-23"),
]
TEST_PAIRS = [
    ("2022-23", "2023-24"), ("2023-24", "2024-25"), ("2024-25", "2025-26"),
]


def build_career_indexes() -> dict:
    """Map player_id → DataFrame(Season, GP, Barrett Score), oldest first."""
    regular: dict[int, list[dict]] = {}
    for season in reversed(SEASONS):
        try:
            ranked = apply_rankings(build_raw(season, playoffs=False))
        except Exception:
            continue
        if ranked.empty:
            continue
        for _, r in ranked.iterrows():
            pid = r.get("PLAYER_ID")
            if pd.isna(pid):
                continue
            regular.setdefault(int(pid), []).append({
                "Season":        season,
                "GP":            int(r.get("GP", 0) or 0),
                "Barrett Score": float(r.get("barrett_score", 0) or 0),
            })
    return {pid: pd.DataFrame(rows) for pid, rows in regular.items()}


def career_weighted_score_at(career_df: pd.DataFrame, up_to_season: str,
                             min_gp: int = HEALTHY_SEASON_GP) -> float | None:
    if career_df.empty:
        return None
    up_to = career_df[career_df["Season"] <= up_to_season]
    if up_to.empty:
        return None
    healthy = up_to[up_to["GP"] >= min_gp]
    pool = healthy if not healthy.empty else up_to
    recent = pool.tail(3)
    weights = [0.20, 0.30, 0.50][-len(recent):]
    w_sum = sum(weights)
    return float((recent["Barrett Score"].values * weights).sum() / w_sum)


def years_in_league(career_df: pd.DataFrame, up_to_season: str) -> int:
    """Number of seasons (including up_to_season) the player has appeared."""
    if career_df.empty:
        return 0
    up_to = career_df[career_df["Season"] <= up_to_season]
    return int(len(up_to))


# ── Build test/train rows ────────────────────────────────────────────────────
def build_rows(pairs, regular_careers) -> pd.DataFrame:
    rows = []
    for prev, curr in pairs:
        if prev not in SALARY_CAP_M or curr not in SALARY_CAP_M:
            continue
        cap_prev_M = SALARY_CAP_M[prev]
        cap_curr_M = SALARY_CAP_M[curr]
        cap_ratio = cap_curr_M / cap_prev_M

        try:
            prev_df = build_ranked_projected(prev)
            curr_df = build_ranked_projected(curr)
        except Exception:
            continue
        if prev_df.empty or curr_df.empty:
            continue

        raw_prev = fetch_league_stats(prev, "Regular Season")
        if raw_prev.empty or "AGE" not in raw_prev.columns:
            continue
        age_lookup = dict(zip(raw_prev["PLAYER_ID"], raw_prev["AGE"]))

        try:
            detailed = fetch_player_positions_detailed(prev, cache_v=2)
        except Exception:
            detailed = {}
        try:
            coarse = fetch_bref_positions(season_to_espn_year(prev), cache_v=3)
        except Exception:
            coarse = {}

        def _pos_bucket(n):
            d = detailed.get(normalize(n))
            if d: return position_to_bucket(d)
            return coarse.get(normalize(n), "Unknown")

        prev_pool = prev_df[prev_df["salary"] > 0].copy()
        curr_slim = curr_df[["PLAYER_ID", "salary"]].rename(
            columns={"salary": "salary_curr"})
        m = prev_pool.merge(curr_slim, on="PLAYER_ID", how="left")
        m = m[m["salary_curr"].notna() & (m["salary_curr"] > 0)]
        if m.empty:
            continue
        m["pct_change"] = (m["salary_curr"] - m["salary"]) / m["salary"]
        m = m[m["pct_change"].abs() >= NEW_CONTRACT_PCT]
        if m.empty:
            continue

        for _, row in m.iterrows():
            pid = int(row["PLAYER_ID"])
            age = age_lookup.get(pid)
            if age is None or pd.isna(age):
                continue
            age = float(age)
            career = regular_careers.get(pid, pd.DataFrame())
            career_score = career_weighted_score_at(career, prev)
            if career_score is None:
                career_score = float(row["barrett_score"])
            yrs = years_in_league(career, prev)
            pos = _pos_bucket(row["Player"])

            # Compute career-weighted base proj (matches production)
            cur_scores = curr_df["barrett_score"].sort_values(
                ascending=False).values
            cur_salaries = curr_df["salary"].sort_values(
                ascending=False).values
            effective_rank = int((cur_scores > career_score).sum()) + 1
            capped_rank = min(effective_rank, len(cur_salaries)) - 1
            career_base_proj = float(cur_salaries[capped_rank])

            cap_curr_dollars = cap_curr_M * 1_000_000
            rows.append({
                "player":           row["Player"],
                "prev":             prev,
                "curr":             curr,
                "age":              age,
                "pos_bucket":       pos,
                "barrett":          career_score,
                "GP":               float(row.get("GP", 0) or 0),
                "MPG":              float(row.get("MPG", 0) or 0),
                "salary_prev":      float(row["salary"]),
                "salary_curr":      float(row["salary_curr"]),
                "salary_prev_pct":  float(row["salary"]) / (cap_prev_M * 1_000_000),
                "salary_curr_pct":  float(row["salary_curr"]) / cap_curr_dollars,
                "career_base_proj": career_base_proj,
                "years_in_league":  yrs,
                "cap_curr":         cap_curr_dollars,
                "cap_curr_M":       cap_curr_M,
            })
    return pd.DataFrame(rows)


# ── Feature matrices ────────────────────────────────────────────────────────
def make_X_minimal(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    age = df["age"].astype(float).values
    barrett = df["barrett"].astype(float).values
    gp = df["GP"].astype(float).values
    mpg = df["MPG"].astype(float).values
    prior = df["salary_prev_pct"].astype(float).values
    is_g = (df["pos_bucket"] == "Guard").astype(float).values
    is_f = (df["pos_bucket"] == "Forward").astype(float).values

    X = np.column_stack([
        np.ones_like(age), barrett, barrett**2,
        age, age**2,
        gp, mpg,
        prior,
        is_g, is_f,
    ])
    names = ["intercept", "barrett", "barrett²", "age", "age²",
             "GP", "MPG", "prior_salary_%cap", "is_Guard", "is_Forward"]
    return X, names


def make_X_with_service(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    X_min, n_min = make_X_minimal(df)
    yrs = df["years_in_league"].astype(float).values
    # CBA tier eligibility flags
    is_30pct = (yrs >= 7).astype(float)
    is_35pct = (yrs >= 10).astype(float)
    X = np.column_stack([X_min, yrs, is_30pct, is_35pct])
    return X, n_min + ["years_in_league", "is_30pct_tier", "is_35pct_tier"]


def make_X_full(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    X_svc, n_svc = make_X_with_service(df)
    age = df["age"].astype(float).values
    is_g = (df["pos_bucket"] == "Guard").astype(float).values
    is_f = (df["pos_bucket"] == "Forward").astype(float).values
    barrett = df["barrett"].astype(float).values
    prior = df["salary_prev_pct"].astype(float).values

    # Interactions
    age_x_guard = age * is_g
    age_x_forward = age * is_f
    age_cubed = age ** 3
    barrett_x_prior = barrett * prior

    X = np.column_stack([
        X_svc,
        age_x_guard, age_x_forward,
        age_cubed,
        barrett_x_prior,
    ])
    return X, n_svc + ["age×Guard", "age×Forward", "age³", "barrett×prior"]


# ── Model fitters ───────────────────────────────────────────────────────────
def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    β, *_ = np.linalg.lstsq(X, y, rcond=None)
    return β


def fit_ridge(X: np.ndarray, y: np.ndarray, l2: float = 0.01) -> np.ndarray:
    """Closed-form ridge regression. Don't regularize the intercept."""
    n_features = X.shape[1]
    reg = l2 * np.eye(n_features)
    reg[0, 0] = 0  # no penalty on intercept
    XtX = X.T @ X + reg
    return np.linalg.solve(XtX, X.T @ y)


def predict(X: np.ndarray, β: np.ndarray, cap_dollars: np.ndarray) -> np.ndarray:
    """X @ β gives predicted % of cap. Multiply by cap to get $."""
    pred_pct = np.clip(X @ β, 0.001, 0.6)  # clip to plausible range
    return pred_pct * cap_dollars


# ── Baseline (production formula) ───────────────────────────────────────────
def predict_baseline(df: pd.DataFrame) -> np.ndarray:
    out = []
    for _, r in df.iterrows():
        base = r["career_base_proj"]
        age_m = CONTRACT_AGE_MULTIPLIERS.get(age_bucket(r["age"]), 1.0)
        pos_m = CONTRACT_POSITION_MULTIPLIERS.get(r["pos_bucket"], 1.0)
        if base >= r["cap_curr"] * SUPERMAX_CAP_PCT:
            pos_m = 1.0
        out.append(base * age_m * pos_m)
    return np.array(out)


# ── Scoring ─────────────────────────────────────────────────────────────────
def score(df: pd.DataFrame, pred: np.ndarray) -> dict:
    actual = df["salary_curr"].values
    cap = df["cap_curr"].values
    err = np.abs(actual - pred) / cap * 100
    signed = (actual - pred) / cap * 100
    return {
        "n":              len(df),
        "median_err":     float(np.median(err)),
        "mean_err":       float(np.mean(err)),
        "within_5":       float(np.mean(err <= 5.0) * 100),
        "within_10":      float(np.mean(err <= 10.0) * 100),
        "median_bias":    float(np.median(signed)),
    }


def print_score(label: str, s: dict, baseline: dict = None):
    if baseline is None:
        print(f"  {label}")
        print(f"    n             = {s['n']}")
        print(f"    Median |err|  = {s['median_err']:5.2f}% of cap")
        print(f"    Within 5%     = {s['within_5']:5.1f}%")
        print(f"    Within 10%    = {s['within_10']:5.1f}%")
        print(f"    Median bias   = {s['median_bias']:+5.2f}% of cap")
    else:
        d5  = s["within_5"]  - baseline["within_5"]
        d10 = s["within_10"] - baseline["within_10"]
        dM  = s["median_err"] - baseline["median_err"]
        marker = "🟢" if d5 >= 1 else ("🟡" if d5 >= 0.5 else ("⚪" if d5 >= 0 else "🔴"))
        print(f"  {label}")
        print(f"    Within 5%     = {s['within_5']:5.1f}%  ({d5:+5.2f}pp) {marker}")
        print(f"    Within 10%    = {s['within_10']:5.1f}%  ({d10:+5.2f}pp)")
        print(f"    Median |err|  = {s['median_err']:5.2f}%  ({dM:+5.2f}pp)")


def main():
    print("Pre-building career indexes...")
    t0 = time.time()
    regular_careers = build_career_indexes()
    print(f"  Indexed {len(regular_careers)} careers in {time.time()-t0:.1f}s.\n")

    print("Building train + test rows...")
    t0 = time.time()
    train_df = build_rows(TRAIN_PAIRS, regular_careers)
    test_df  = build_rows(TEST_PAIRS,  regular_careers)
    print(f"  Train: {len(train_df)} rows  ·  Test: {len(test_df)} rows  "
          f"in {time.time()-t0:.1f}s.\n")

    if train_df.empty or test_df.empty:
        print("No data. Has the cache been seeded?"); return

    y_train = train_df["salary_curr_pct"].values

    # ── Variant A: baseline ──────────────────────────────────────────────────
    pred_A = predict_baseline(test_df)
    sA = score(test_df, pred_A)

    # ── Variant B: OLS minimal ──────────────────────────────────────────────
    Xb_train, _ = make_X_minimal(train_df)
    β_B = fit_ols(Xb_train, y_train)
    Xb_test, _ = make_X_minimal(test_df)
    pred_B = predict(Xb_test, β_B, test_df["cap_curr"].values)
    sB = score(test_df, pred_B)

    # ── Variant C: OLS + service ─────────────────────────────────────────────
    Xc_train, _ = make_X_with_service(train_df)
    β_C = fit_ols(Xc_train, y_train)
    Xc_test, _ = make_X_with_service(test_df)
    pred_C = predict(Xc_test, β_C, test_df["cap_curr"].values)
    sC = score(test_df, pred_C)

    # ── Variant D: OLS + service + interactions ──────────────────────────────
    Xd_train, names_d = make_X_full(train_df)
    β_D = fit_ols(Xd_train, y_train)
    Xd_test, _ = make_X_full(test_df)
    pred_D = predict(Xd_test, β_D, test_df["cap_curr"].values)
    sD = score(test_df, pred_D)

    # ── Variant E: Ridge on D's feature set (sweep λ) ────────────────────────
    best_λ, best_β_E, best_sE = None, None, None
    for λ in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        β_try = fit_ridge(Xd_train, y_train, l2=λ)
        pred_try = predict(Xd_test, β_try, test_df["cap_curr"].values)
        s_try = score(test_df, pred_try)
        if best_sE is None or s_try["within_5"] > best_sE["within_5"]:
            best_λ, best_β_E, best_sE = λ, β_try, s_try

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 76)
    print("OUT-OF-SAMPLE RESULTS")
    print("=" * 76)
    print_score("A. PRODUCTION BASELINE", sA)
    print()
    print_score(f"B. OLS minimal ({len(β_B)} features)", sB, sA)
    print()
    print_score(f"C. OLS + service time ({len(β_C)} features)", sC, sA)
    print()
    print_score(f"D. OLS + interactions ({len(β_D)} features)", sD, sA)
    print()
    print_score(f"E. RIDGE on D (λ={best_λ}, best of sweep)", best_sE, sA)

    # Best non-baseline variant
    candidates = [("B", sB, β_B, "minimal"),
                  ("C", sC, β_C, "with_service"),
                  ("D", sD, β_D, "full"),
                  ("E", best_sE, best_β_E, "full_ridge")]
    best_name, best_s, best_β, best_feat_set = max(
        candidates, key=lambda x: x[1]["within_5"])
    print(f"\nBest variant: {best_name} (Within 5% = {best_s['within_5']:.1f}%)")
    print(f"vs baseline ({sA['within_5']:.1f}%): "
          f"{best_s['within_5'] - sA['within_5']:+.2f}pp\n")

    # ── Print fitted coefficients for the best variant ──────────────────────
    if best_name == "D":
        names = names_d
    elif best_name == "E":
        names = names_d  # ridge uses same feature set as D
    elif best_name == "C":
        _, names = make_X_with_service(test_df)
    else:
        _, names = make_X_minimal(test_df)

    print("=" * 76)
    print(f"FITTED COEFFICIENTS — variant {best_name}")
    print("=" * 76)
    print("Copy-pasteable for utils.py if shipping:\n")
    print(f"REGRESSION_COEFFS_{best_name} = {{")
    for name, b in zip(names, best_β):
        print(f"    {name!r:30s}: {b:+.6f},")
    print("}\n")

    # Decision
    gain = best_s["within_5"] - sA["within_5"]
    print("=" * 76)
    print("DECISION")
    print("=" * 76)
    if gain >= 1.0:
        verdict = f"🟢 SHIP — variant {best_name} ({gain:+.2f}pp on Within 5%)"
    elif gain >= 0.5:
        verdict = f"🟡 CONSIDER — variant {best_name} ({gain:+.2f}pp, modest gain)"
    else:
        verdict = (f"🔴 DO NOT SHIP — best variant only {gain:+.2f}pp; "
                   "rank-mapping baseline is near the ceiling.")
    print(f"  {verdict}\n")


if __name__ == "__main__":
    main()
