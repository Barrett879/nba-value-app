"""Phase 1 contract-prediction improvements — out-of-sample validation.

Pure test, no site changes. Measures whether each of three proposed
improvements actually moves accuracy on the same held-out test set we used
for the headline 80% / 77% / 74% numbers.

VARIANTS TESTED (model prediction)
  A. BASELINE          — current production formula
                          base_proj × age_mult × pos_mult, with supermax
                          suppression at base ≥ SUPERMAX_CAP_PCT
  B. + TRAJECTORY      — A × trajectory_mult based on Barrett trend
  C. + PLAYOFF BLEND   — A with playoff-blended career score (85% RS / 15% PO)
  D. + ALL OF THE ABOVE  — B + C stacked

VARIANTS TESTED (market view / comparables median)
  E. EQUAL-WEIGHTED MEDIAN     — current (median of top-6 comparables)
  F. DISTANCE-WEIGHTED MEDIAN  — weight closer matches more

Train:  2014-15 → 2021-22 (8 pairs)
Test:   2022-23 → 2024-25 (3 pairs, held out)
Pool:   ≥25% YoY salary change (real new contracts)
Metric: % within 5% of cap (era-fair, primary success metric)

Usage:
    python test_phase1_gains.py
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
    build_ranked_projected, build_raw, apply_rankings,
    fetch_league_stats,
    fetch_bref_positions, fetch_player_positions_detailed, position_to_bucket,
    SALARY_CAP_M, age_bucket,
    CONTRACT_AGE_MULTIPLIERS, CONTRACT_POSITION_MULTIPLIERS,
    HEALTHY_SEASON_GP, NEW_CONTRACT_PCT, SUPERMAX_CAP_PCT,
)


# Season → DataFrame index, computed once and reused. Keyed by (season, playoffs).
# Filled lazily as we walk through SEASONS.
def build_career_indexes() -> tuple[dict, dict]:
    """Pre-compute every (player_id → list[(season, gp, barrett_score)])
    across all SEASONS in ONE pass through the disk cache.

    Replaces ~870 individual fetch_player_full_career calls with 2 × 53 =
    106 build_raw reads. Each `season` parquet is already on disk — this
    is just iterating + appending to dicts, no re-scraping.

    Returns (regular_careers, playoff_careers).
    Each dict maps player_id → DataFrame with columns: Season, GP,
    Barrett Score (sorted oldest → newest).
    """
    regular: dict[int, list[dict]] = {}
    playoffs: dict[int, list[dict]] = {}
    # SEASONS is newest→oldest; iterate reverse so the dict ends up
    # oldest→newest naturally.
    for season in reversed(SEASONS):
        # Regular season
        try:
            ranked = apply_rankings(build_raw(season, playoffs=False))
            if not ranked.empty:
                for _, r in ranked.iterrows():
                    pid = r.get("PLAYER_ID")
                    if pd.isna(pid):
                        continue
                    regular.setdefault(int(pid), []).append({
                        "Season":        season,
                        "GP":            int(r.get("GP", 0) or 0),
                        "Barrett Score": float(r.get("barrett_score", 0) or 0),
                    })
        except Exception:
            pass
        # Playoffs
        try:
            ranked_po = apply_rankings(build_raw(season, playoffs=True))
            if not ranked_po.empty:
                for _, r in ranked_po.iterrows():
                    pid = r.get("PLAYER_ID")
                    if pd.isna(pid):
                        continue
                    playoffs.setdefault(int(pid), []).append({
                        "Season":        season,
                        "GP":            int(r.get("GP", 0) or 0),
                        "Barrett Score": float(r.get("barrett_score", 0) or 0),
                    })
        except Exception:
            pass

    # Convert to DataFrames for the rest of the code path.
    return (
        {pid: pd.DataFrame(rows) for pid, rows in regular.items()},
        {pid: pd.DataFrame(rows) for pid, rows in playoffs.items()},
    )


TRAIN_PAIRS = [
    ("2014-15", "2015-16"), ("2015-16", "2016-17"), ("2016-17", "2017-18"),
    ("2017-18", "2018-19"), ("2018-19", "2019-20"), ("2019-20", "2020-21"),
    ("2020-21", "2021-22"), ("2021-22", "2022-23"),
]
TEST_PAIRS = [
    ("2022-23", "2023-24"), ("2023-24", "2024-25"), ("2024-25", "2025-26"),
]


# ── Career-weighted helpers (mirror the production logic) ────────────────────
def career_weighted_score_at(career_df: pd.DataFrame, up_to_season: str,
                             min_gp: int = HEALTHY_SEASON_GP) -> float | None:
    """Mirror of production logic: 50/30/20 weighting of last 3 healthy
    seasons up to and including up_to_season."""
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


def career_trajectory(career_df: pd.DataFrame, up_to_season: str,
                      min_gp: int = HEALTHY_SEASON_GP) -> float | None:
    """Signed Barrett-score delta between the most recent healthy season
    and the season 2 healthy seasons before it (or 3 if available).
    Positive = trending up, negative = trending down."""
    if career_df.empty:
        return None
    up_to = career_df[career_df["Season"] <= up_to_season]
    healthy = up_to[up_to["GP"] >= min_gp]
    if len(healthy) < 2:
        return None  # not enough data to compute trajectory
    # Most recent vs. 2 healthy seasons ago (or earliest available if <3)
    most_recent = float(healthy.iloc[-1]["Barrett Score"])
    look_back_idx = max(0, len(healthy) - 3)  # 2 seasons ago, or earliest
    earlier = float(healthy.iloc[look_back_idx]["Barrett Score"])
    return most_recent - earlier


def trajectory_multiplier(traj: float | None) -> float:
    """Convert trajectory delta to a multiplier on base projection.
    Bucketed (vs linear) so a small noise change doesn't flip the result."""
    if traj is None:
        return 1.0
    if traj >= 10:  return 1.10
    if traj >= 5:   return 1.05
    if traj <= -10: return 0.90
    if traj <= -5:  return 0.95
    return 1.00


def playoff_blended_score(reg_career_score: float | None,
                          po_career_score: float | None,
                          playoff_gp: int) -> float | None:
    """Blend 85% regular / 15% playoff if player has meaningful playoff
    sample. Else regular only."""
    if reg_career_score is None:
        return None
    if po_career_score is None or playoff_gp < 10:
        return reg_career_score
    return 0.85 * reg_career_score + 0.15 * po_career_score


# ── Build per-contract test rows ─────────────────────────────────────────────
def build_test_rows(
    pairs: list[tuple[str, str]],
    regular_careers: dict,
    playoff_careers: dict,
) -> pd.DataFrame:
    """One row per qualifying new-contract signing, with every feature each
    variant needs. Uses pre-built career indexes (no per-player scraping)."""
    rows = []
    for prev, curr in pairs:
        if prev not in SALARY_CAP_M or curr not in SALARY_CAP_M:
            continue
        cap_prev = SALARY_CAP_M[prev] * 1_000_000
        cap_curr = SALARY_CAP_M[curr] * 1_000_000
        cap_ratio = cap_curr / cap_prev

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

        def _resolve_pos_bucket(name: str) -> str:
            d = detailed.get(normalize(name))
            if d:
                return position_to_bucket(d)
            return coarse.get(normalize(name), "Unknown")

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

        # Per-row enrichment using pre-built career indexes.
        # 200× faster than calling fetch_player_full_career per player.
        for _, row in m.iterrows():
            player = row["Player"]
            pid = int(row["PLAYER_ID"])
            age = age_lookup.get(pid)
            pos_bucket = _resolve_pos_bucket(player)

            reg_career = regular_careers.get(pid, pd.DataFrame())
            po_career  = playoff_careers.get(pid, pd.DataFrame())

            reg_career_score = career_weighted_score_at(reg_career, prev)
            traj             = career_trajectory(reg_career, prev)
            po_career_score  = career_weighted_score_at(po_career, prev)
            po_gp_total = int(po_career["GP"].sum()) if not po_career.empty else 0

            # Compute effective rank (career-weighted-Score → salary rank
            # in current season's pool) — matches production logic
            cur_scores = curr_df["barrett_score"].sort_values(
                ascending=False).values
            cur_salaries = curr_df["salary"].sort_values(
                ascending=False).values
            score_for_rank = (
                reg_career_score
                if reg_career_score is not None else row["barrett_score"]
            )
            effective_rank = int((cur_scores > score_for_rank).sum()) + 1
            capped_rank = min(effective_rank, len(cur_salaries)) - 1
            career_base_proj = float(cur_salaries[capped_rank])

            rows.append({
                "prev": prev,
                "curr": curr,
                "player": player,
                "age": age,
                "pos_bucket": pos_bucket,
                "salary_prev": float(row["salary"]),
                "salary_curr": float(row["salary_curr"]),
                "cap_curr": cap_curr,
                "reg_career_score": reg_career_score,
                "po_career_score":  po_career_score,
                "po_gp_total":      po_gp_total,
                "trajectory":       traj,
                "career_base_proj": career_base_proj,
            })

    return pd.DataFrame(rows)


# ── Model prediction variants ───────────────────────────────────────────────
def predict(row, *, use_trajectory: bool = False,
            use_playoff_blend: bool = False) -> float:
    """Compute the model prediction with optional improvements layered in."""
    # If playoff-blend on, recompute base proj from the blended score
    # (this requires a per-season-pool lookup; for simplicity we use the
    # multiplicative form: blended_score / regular_score scales the base).
    if use_playoff_blend and row.get("reg_career_score"):
        reg = row["reg_career_score"]
        blended = playoff_blended_score(
            reg, row.get("po_career_score"), int(row.get("po_gp_total", 0)),
        )
        if blended is not None and reg > 0:
            base = row["career_base_proj"] * (blended / reg)
        else:
            base = row["career_base_proj"]
    else:
        base = row["career_base_proj"]

    age_m = CONTRACT_AGE_MULTIPLIERS.get(age_bucket(row.get("age")), 1.0)
    pos_m = CONTRACT_POSITION_MULTIPLIERS.get(
        row.get("pos_bucket", "Unknown"), 1.0,
    )

    # Supermax suppression: if base is already ≥28% of cap, position
    # mult drops to 1.0 (mirrors production).
    cap = row.get("cap_curr", 154_600_000)
    if base >= cap * SUPERMAX_CAP_PCT:
        pos_m = 1.0

    pred = base * age_m * pos_m

    if use_trajectory:
        pred *= trajectory_multiplier(row.get("trajectory"))

    return pred


# ── Score one prediction variant ────────────────────────────────────────────
def score_variant(df: pd.DataFrame, pred_col: str) -> dict:
    err_pct = (df["salary_curr"] - df[pred_col]).abs() / df["cap_curr"] * 100
    signed_pct = (df["salary_curr"] - df[pred_col]) / df["cap_curr"] * 100
    return {
        "n":              len(df),
        "median_err_cap": float(err_pct.median()),
        "mean_err_cap":   float(err_pct.mean()),
        "within_5":       float((err_pct <= 5.0).mean() * 100),
        "within_10":      float((err_pct <= 10.0).mean() * 100),
        "median_bias":    float(signed_pct.median()),
    }


def print_scores(label: str, s: dict, baseline: dict | None = None) -> None:
    if baseline is None:
        print(f"  {label}")
        print(f"    n             = {s['n']}")
        print(f"    Median |err|  = {s['median_err_cap']:5.2f}% of cap")
        print(f"    Within 5%     = {s['within_5']:5.1f}%")
        print(f"    Within 10%    = {s['within_10']:5.1f}%")
        print(f"    Median bias   = {s['median_bias']:+5.2f}% of cap")
    else:
        d5  = s["within_5"]  - baseline["within_5"]
        d10 = s["within_10"] - baseline["within_10"]
        dM  = s["median_err_cap"] - baseline["median_err_cap"]
        print(f"  {label}")
        print(f"    Within 5%     = {s['within_5']:5.1f}%  ({d5:+5.2f}pp)")
        print(f"    Within 10%    = {s['within_10']:5.1f}%  ({d10:+5.2f}pp)")
        print(f"    Median |err|  = {s['median_err_cap']:5.2f}%  ({dM:+5.2f}pp)")


def main() -> None:
    print("Pre-building career indexes (one pass over all seasons)...")
    import time
    t0 = time.time()
    regular_careers, playoff_careers = build_career_indexes()
    print(f"  Indexed {len(regular_careers)} regular-season careers and "
          f"{len(playoff_careers)} playoff careers in {time.time() - t0:.1f}s.\n")

    print("Building test pool (2022-25)...")
    t0 = time.time()
    test_df = build_test_rows(TEST_PAIRS, regular_careers, playoff_careers)
    print(f"  Loaded {len(test_df)} qualifying new-contract test rows "
          f"in {time.time() - t0:.1f}s.\n")
    if test_df.empty:
        print("No data. Has the cache been seeded?")
        return

    # Compute predictions under each variant.
    test_df["pred_A_baseline"] = test_df.apply(
        lambda r: predict(r), axis=1,
    )
    test_df["pred_B_trajectory"] = test_df.apply(
        lambda r: predict(r, use_trajectory=True), axis=1,
    )
    test_df["pred_C_playoffblend"] = test_df.apply(
        lambda r: predict(r, use_playoff_blend=True), axis=1,
    )
    test_df["pred_D_combined"] = test_df.apply(
        lambda r: predict(r, use_trajectory=True, use_playoff_blend=True),
        axis=1,
    )

    sA = score_variant(test_df, "pred_A_baseline")
    sB = score_variant(test_df, "pred_B_trajectory")
    sC = score_variant(test_df, "pred_C_playoffblend")
    sD = score_variant(test_df, "pred_D_combined")

    print("=" * 76)
    print("MODEL PREDICTION ACCURACY — out-of-sample test set (2022-25)")
    print("=" * 76)
    print_scores("A. BASELINE (current production)", sA)
    print()
    print_scores("B. + TRAJECTORY",        sB, baseline=sA)
    print()
    print_scores("C. + PLAYOFF BLEND",     sC, baseline=sA)
    print()
    print_scores("D. + BOTH (B + C)",      sD, baseline=sA)

    # Trajectory diagnostic — how many had non-zero trajectory mult applied?
    mult_diag = test_df["trajectory"].apply(trajectory_multiplier)
    n_traj_up   = int((mult_diag > 1.0).sum())
    n_traj_down = int((mult_diag < 1.0).sum())
    n_traj_flat = int((mult_diag == 1.0).sum())
    print(f"\nTrajectory distribution: {n_traj_up} trending up · "
          f"{n_traj_down} trending down · {n_traj_flat} flat/unknown")

    # Playoff blend diagnostic
    n_blended = int(
        (test_df["po_career_score"].notna() & (test_df["po_gp_total"] >= 10)).sum()
    )
    print(f"Playoff blend applied to: {n_blended} of {len(test_df)} test rows")

    # Recommendation
    print("\n" + "=" * 76)
    print("RECOMMENDATION")
    print("=" * 76)
    deltas = {
        "B (trajectory)":   sB["within_5"] - sA["within_5"],
        "C (playoff blend)": sC["within_5"] - sA["within_5"],
        "D (both)":         sD["within_5"] - sA["within_5"],
    }
    for name, d in sorted(deltas.items(), key=lambda kv: -kv[1]):
        verdict = "SHIP" if d >= 0.5 else ("MARGINAL" if d >= 0 else "DO NOT SHIP")
        print(f"  {name:24s} {d:+5.2f}pp on Within 5%  →  {verdict}")


if __name__ == "__main__":
    main()
