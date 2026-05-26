"""Contract Prediction model — out-of-sample validation, no site changes.

Builds a proper salary prediction model on top of the Barrett Score and
compares it to the current rank-mapping baseline. Pure experiment — does
NOT modify utils.py, the model weights, or any live page.

Architecture (what we're testing):
    Barrett Score (production)   ← unchanged
            ↓
    Predicted contract  =  rank-base + age + position + role + cap
            ↓
    Out-of-sample accuracy on 2022-25 contracts

Three model variants compared on the same held-out test set:

  A. BASELINE — current Barrett rank → salary mapping, cap-scaled
  B. MULTIPLIER — A × age_bucket × position_bucket (from training set)
  C. OLS REGRESSION — fit salary_pct_cap ~ Barrett + age + age² + GP +
     MPG + position + prior_salary_pct_cap on training set; apply to test

Train: 2014-15 → 2021-22 (8 season pairs)
Test:  2022-23 → 2024-25 (3 season pairs)
Pool:  filtered to ≥25% YoY salary change (real new contracts only)

Usage:
    python test_contract_model.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from utils import (
    SEASONS, normalize, season_to_espn_year,
    build_ranked_projected, fetch_league_stats, fetch_bref_positions,
)


SALARY_CAP_M = {
    "1984-85": 3.6,  "1985-86": 4.2,  "1986-87": 4.9,  "1987-88": 6.2,
    "1988-89": 7.2,  "1989-90": 9.8,  "1990-91": 11.9, "1991-92": 12.5,
    "1992-93": 14.0, "1993-94": 15.2, "1994-95": 15.9, "1995-96": 23.0,
    "1996-97": 24.4, "1997-98": 26.9, "1998-99": 30.0, "1999-00": 34.0,
    "2000-01": 35.5, "2001-02": 42.5, "2002-03": 40.3, "2003-04": 43.8,
    "2004-05": 43.9, "2005-06": 49.5, "2006-07": 53.1, "2007-08": 55.6,
    "2008-09": 58.7, "2009-10": 57.7, "2010-11": 58.0, "2011-12": 58.0,
    "2012-13": 58.0, "2013-14": 58.7, "2014-15": 63.1, "2015-16": 70.0,
    "2016-17": 94.1, "2017-18": 99.1, "2018-19": 101.9, "2019-20": 109.1,
    "2020-21": 109.1, "2021-22": 112.4, "2022-23": 123.7, "2023-24": 136.0,
    "2024-25": 140.6, "2025-26": 154.6,
}

TRAIN_PAIRS = [
    ("2014-15", "2015-16"), ("2015-16", "2016-17"), ("2016-17", "2017-18"),
    ("2017-18", "2018-19"), ("2018-19", "2019-20"), ("2019-20", "2020-21"),
    ("2020-21", "2021-22"), ("2021-22", "2022-23"),
]
TEST_PAIRS = [
    ("2022-23", "2023-24"), ("2023-24", "2024-25"), ("2024-25", "2025-26"),
]

NEW_DEAL_THRESHOLD = 0.25


def _age_bucket(age) -> str:
    if pd.isna(age):
        return "UNK"
    age = int(age)
    if age <= 22: return "≤22"
    if age <= 25: return "23-25"
    if age <= 28: return "26-28"
    if age <= 31: return "29-31"
    if age <= 34: return "32-34"
    return "35+"


def build_pair_dataset(prev: str, curr: str) -> pd.DataFrame | None:
    """One row per qualifying player for a (prev, curr) pair, with all
    features the model needs."""
    if prev not in SALARY_CAP_M or curr not in SALARY_CAP_M:
        return None
    cap_prev_M = SALARY_CAP_M[prev]
    cap_curr_M = SALARY_CAP_M[curr]
    cap_ratio = cap_curr_M / cap_prev_M

    try:
        prev_df = build_ranked_projected(prev)
        curr_df = build_ranked_projected(curr)
    except Exception:
        return None
    if prev_df.empty or curr_df.empty:
        return None

    raw_prev = fetch_league_stats(prev, "Regular Season")
    if raw_prev.empty or "AGE" not in raw_prev.columns:
        return None
    age_lookup = dict(zip(raw_prev["PLAYER_ID"], raw_prev["AGE"]))

    # fetch_bref_positions returns "Guard"/"Forward"/"Center" keyed by
    # normalized name (NOT PG/SG/SF/PF/C — the previous script's bug).
    try:
        pos_lookup = fetch_bref_positions(season_to_espn_year(prev), cache_v=3)
    except Exception as e:
        print(f"      ⚠ position fetch failed for {prev}: {e}")
        pos_lookup = {}

    prev_df = prev_df[prev_df["salary"] > 0].copy()
    curr_slim = curr_df[["PLAYER_ID", "salary"]].rename(columns={"salary": "salary_curr"})
    m = prev_df.merge(curr_slim, on="PLAYER_ID", how="left")

    m["age"] = m["PLAYER_ID"].map(age_lookup)
    m["pos"] = m["Player"].map(lambda n: pos_lookup.get(normalize(n), "Unknown"))
    m["age_bucket"] = m["age"].apply(_age_bucket)

    # Cap-scaled current-dollar projection (matches analyze_accuracy.py).
    m["proj_capadj"] = m["projected_salary"] * cap_ratio
    m["still"] = m["salary_curr"].fillna(0) > 0
    m["pct_change"] = ((m["salary_curr"].fillna(0) - m["salary"]) / m["salary"]).where(m["salary"] > 0)

    # Era-fair target: next-season salary as fraction of next-season cap.
    m["salary_pct_cap_curr"] = m["salary_curr"] / (cap_curr_M * 1_000_000)
    m["salary_pct_cap_prev"] = m["salary"]      / (cap_prev_M * 1_000_000)
    m["cap_curr"] = cap_curr_M * 1_000_000
    m["cap_curr_M"] = cap_curr_M

    return m[[
        "PLAYER_ID", "Player", "age", "age_bucket", "pos",
        "barrett_score", "GP", "MPG",
        "salary", "projected_salary", "proj_capadj",
        "salary_curr", "still", "pct_change",
        "salary_pct_cap_curr", "salary_pct_cap_prev",
        "cap_curr", "cap_curr_M",
    ]]


def _filter_pool(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["still"]
        & (df["proj_capadj"] > 0)
        & (df["pct_change"].abs() >= NEW_DEAL_THRESHOLD)
        & df["age"].notna()
        & df["salary_pct_cap_curr"].notna()
        & (df["salary_pct_cap_curr"] > 0)
    ].copy()


# ── Models ────────────────────────────────────────────────────────────────────
def fit_multipliers(train: pd.DataFrame) -> tuple[dict, dict]:
    """median(actual / baseline) per age bucket and per position bucket."""
    train = train[train["proj_capadj"] > 0].copy()
    train["ratio"] = train["salary_curr"] / train["proj_capadj"]
    age_m = train.groupby("age_bucket")["ratio"].median().to_dict()
    pos_m = train.groupby("pos")["ratio"].median().to_dict()
    return age_m, pos_m


def predict_multiplier(df: pd.DataFrame, age_m: dict, pos_m: dict) -> np.ndarray:
    am = df["age_bucket"].map(age_m).fillna(1.0).values
    pm = df["pos"].map(pos_m).fillna(1.0).values
    return df["proj_capadj"].values * am * pm


def _make_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix for OLS regression. Target is salary_pct_cap_curr."""
    age = df["age"].astype(float).values
    barrett = df["barrett_score"].astype(float).values
    gp  = df["GP"].astype(float).values
    mpg = df["MPG"].astype(float).values
    prior_pct = df["salary_pct_cap_prev"].astype(float).values

    is_guard   = (df["pos"] == "Guard").astype(float).values
    is_forward = (df["pos"] == "Forward").astype(float).values
    # Center is the reference (omitted to avoid multicollinearity).

    X = np.column_stack([
        np.ones_like(age),       # intercept
        barrett,                 # production
        barrett ** 2,            # convexity at top
        age,                     # age main effect
        age ** 2,                # age curve
        gp,                      # availability proxy 1
        mpg,                     # role/usage proxy
        prior_pct,               # leverage / continuation signal
        is_guard,
        is_forward,
    ])
    names = ["intercept", "barrett", "barrett²", "age", "age²", "GP", "MPG",
             "prior_salary_%cap", "is_Guard", "is_Forward"]
    return X, names


def fit_ols(train: pd.DataFrame, l2: float = 0.0) -> tuple[np.ndarray, list[str]]:
    """Closed-form OLS (or ridge if l2 > 0). Returns (β, feature_names)."""
    X, names = _make_feature_matrix(train)
    y = train["salary_pct_cap_curr"].astype(float).values
    if l2 > 0:
        XtX = X.T @ X + l2 * np.eye(X.shape[1])
        # Don't regularize the intercept.
        XtX[0, 0] -= l2
        β = np.linalg.solve(XtX, X.T @ y)
    else:
        β, *_ = np.linalg.lstsq(X, y, rcond=None)
    return β, names


def predict_ols(df: pd.DataFrame, β: np.ndarray) -> np.ndarray:
    """Returns predicted salary in $ (not % of cap)."""
    X, _ = _make_feature_matrix(df)
    pred_pct = X @ β
    pred_pct = np.clip(pred_pct, 0.001, 0.6)  # floor + ceiling on % of cap
    return pred_pct * df["cap_curr"].values


# ── Scoring ───────────────────────────────────────────────────────────────────
def score(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """Accuracy metrics for one prediction series."""
    pred = np.asarray(pred, dtype=float)
    actual = df["salary_curr"].values
    cap = df["cap_curr"].values
    err_pct = np.abs(actual - pred) / cap * 100
    signed_pct = (actual - pred) / cap * 100
    return {
        "n":              len(df),
        "median_err_cap": float(np.median(err_pct)),
        "mean_err_cap":   float(np.mean(err_pct)),
        "within_5":       float(np.mean(err_pct <= 5.0)  * 100),
        "within_10":      float(np.mean(err_pct <= 10.0) * 100),
        "median_bias":    float(np.median(signed_pct)),
    }


def _print_score(label: str, s: dict, indent: str = "  ") -> None:
    print(f"{indent}{label}")
    print(f"{indent}  n             = {s['n']}")
    print(f"{indent}  Median |err|  = {s['median_err_cap']:5.2f}% of cap")
    print(f"{indent}  Mean   |err|  = {s['mean_err_cap']:5.2f}% of cap")
    print(f"{indent}  Within 5%     = {s['within_5']:5.1f}%")
    print(f"{indent}  Within 10%    = {s['within_10']:5.1f}%")
    print(f"{indent}  Median bias   = {s['median_bias']:+5.2f}% of cap")


def main() -> None:
    print("Loading data for train + test pairs...")
    train_dfs, test_dfs = [], []
    for prev, curr in TRAIN_PAIRS:
        d = build_pair_dataset(prev, curr)
        if d is None or d.empty:
            print(f"  skip TRAIN {prev} → {curr}")
            continue
        train_dfs.append(d)
        print(f"  load TRAIN {prev} → {curr}: {len(d)} rows")
    for prev, curr in TEST_PAIRS:
        d = build_pair_dataset(prev, curr)
        if d is None or d.empty:
            print(f"  skip TEST  {prev} → {curr}")
            continue
        test_dfs.append(d)
        print(f"  load TEST  {prev} → {curr}: {len(d)} rows")

    if not test_dfs:
        print("\nNo test data.")
        return

    train_all = pd.concat(train_dfs, ignore_index=True)
    test_all  = pd.concat(test_dfs,  ignore_index=True)
    train_pool = _filter_pool(train_all)
    test_pool  = _filter_pool(test_all)

    print(f"\nFiltered new-contract pools:")
    print(f"  Train: {len(train_pool)} predictions")
    print(f"  Test:  {len(test_pool)} predictions")

    # Position coverage diagnostic — was the big bug in test_age_position.
    pos_counts_train = train_pool["pos"].value_counts().to_dict()
    pos_counts_test  = test_pool["pos"].value_counts().to_dict()
    print(f"\nPosition coverage in training pool: {pos_counts_train}")
    print(f"Position coverage in test pool:     {pos_counts_test}")

    # ── Model A: BASELINE ────────────────────────────────────────────────────
    pred_baseline = test_pool["proj_capadj"].values
    s_a = score(test_pool, pred_baseline)

    # ── Model B: MULTIPLIER (age × position) ─────────────────────────────────
    age_m, pos_m = fit_multipliers(train_pool)
    pred_mult = predict_multiplier(test_pool, age_m, pos_m)
    s_b = score(test_pool, pred_mult)

    # ── Model C: OLS REGRESSION ──────────────────────────────────────────────
    β, feat_names = fit_ols(train_pool)
    pred_ols = predict_ols(test_pool, β)
    s_c = score(test_pool, pred_ols)

    # ── Model D: RIDGE REGRESSION (mild L2) ──────────────────────────────────
    β_r, _ = fit_ols(train_pool, l2=0.01)
    pred_r = predict_ols(test_pool, β_r)
    s_d = score(test_pool, pred_r)

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("LEARNED PARAMETERS")
    print("=" * 78)
    print(f"\nAge multipliers (B):")
    for b in ["≤22", "23-25", "26-28", "29-31", "32-34", "35+", "UNK"]:
        if b in age_m:
            print(f"  {b:7s}: ×{age_m[b]:.3f}")
    print(f"\nPosition multipliers (B):")
    for p in sorted(pos_m, key=lambda k: -pos_m[k]):
        print(f"  {p:8s}: ×{pos_m[p]:.3f}")

    print(f"\nOLS coefficients (C):")
    for name, b_ in zip(feat_names, β):
        print(f"  {name:20s}: {b_:+.5f}")

    print("\n" + "=" * 78)
    print("OUT-OF-SAMPLE RESULTS · test pool (n={})".format(len(test_pool)))
    print("=" * 78 + "\n")
    _print_score("A. BASELINE (current model)",         s_a); print()
    _print_score("B. MULTIPLIER (age × position)",      s_b); print()
    _print_score("C. OLS REGRESSION (all features)",    s_c); print()
    _print_score("D. RIDGE (OLS + L2 = 0.01)",          s_d)

    print("\n" + "=" * 78)
    print("IMPROVEMENT vs. BASELINE")
    print("=" * 78)
    header = f"{'Model':<28s} {'Within 5%':>12s} {'Within 10%':>13s} {'Median |err|':>14s}"
    print(header)
    print("-" * len(header))
    for name, s in [
        ("B. Multipliers",        s_b),
        ("C. OLS regression",     s_c),
        ("D. Ridge regression",   s_d),
    ]:
        d5  = s["within_5"]       - s_a["within_5"]
        d10 = s["within_10"]      - s_a["within_10"]
        de  = s["median_err_cap"] - s_a["median_err_cap"]
        print(f"  {name:<26s} {d5:+10.2f}pp {d10:+11.2f}pp {de:+13.2f}pp")

    # ── Top remaining errors — diagnostic ────────────────────────────────────
    test_pool = test_pool.copy()
    test_pool["pred_best"] = pred_ols if s_c["within_5"] >= s_d["within_5"] else pred_r
    test_pool["abs_err_M"] = (test_pool["salary_curr"] - test_pool["pred_best"]).abs() / 1_000_000
    test_pool["err_pct_cap"] = (test_pool["salary_curr"] - test_pool["pred_best"]).abs() / test_pool["cap_curr"] * 100
    worst = test_pool.nlargest(10, "abs_err_M")[
        ["Player", "age", "pos", "barrett_score",
         "salary_curr", "pred_best", "abs_err_M", "err_pct_cap"]
    ]
    print("\n" + "=" * 78)
    print("LARGEST REMAINING ERRORS (top 10, model C/D)")
    print("=" * 78)
    print(f"{'Player':<22s} {'Age':>3s} {'Pos':>8s} {'Barrett':>8s} "
          f"{'Actual$M':>9s} {'Pred$M':>8s} {'|Err|$M':>8s} {'%cap':>6s}")
    for _, r in worst.iterrows():
        print(f"{r['Player'][:21]:<22s} {int(r['age']):>3d} {str(r['pos'])[:8]:>8s} "
              f"{r['barrett_score']:>8.1f} "
              f"${r['salary_curr']/1e6:>7.1f}M ${r['pred_best']/1e6:>6.1f}M "
              f"${r['abs_err_M']:>6.1f}M {r['err_pct_cap']:>5.1f}%")

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print("- ≥2pp gain on Within 5% = real, deploy-worthy improvement.")
    print("- ≥1pp gain = meaningful, probably worth wiring in.")
    print("- <0.5pp = noise; don't change anything.")
    print("- The largest remaining errors are the hardest contracts to predict")
    print("  — usually max-extension stars, unusual situations, or roster")
    print("  fillers. They tell you where the model's natural ceiling is.")


if __name__ == "__main__":
    main()
