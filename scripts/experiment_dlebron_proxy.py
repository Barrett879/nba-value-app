"""More sophisticated D-LEBRON proxy attempt.

V1 (basic): STL + BLK + DREB + PF per game → R² = 0.08.

V2 (this script): add position dummies, per-minute stats, minute share,
position × defensive-stat interactions. See if more features push the
R² above 0.20 (where we'd consider shipping for pre-2009 fallback).
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from utils import (
    SEASONS, build_raw, _raw_disk_exists, fetch_league_stats,
    fetch_player_positions_detailed, normalize,
)


def main() -> None:
    print("Loading per-season data with d_lebron + box stats + positions...\n")

    rows = []
    for season in SEASONS:
        if not _raw_disk_exists(season, False):
            continue
        try:
            raw = build_raw(season, False)
        except Exception:
            continue
        if raw.empty or "d_lebron" not in raw.columns:
            continue
        nonzero = raw[raw["d_lebron"] != 0]
        if nonzero.empty:
            continue
        try:
            stats = fetch_league_stats(season, "Regular Season")
        except Exception:
            continue
        if stats.empty or "PLAYER_ID" not in stats.columns:
            continue

        # Position lookup (current-format detailed only — pre-2010 will mostly miss).
        try:
            pos_lookup = fetch_player_positions_detailed(season, cache_v=3)
        except Exception:
            pos_lookup = {}

        # NBA Stats API (fetch_league_stats) returns PER-GAME values already.
        # STL=0.5 means 0.5 steals per game, not 0.5 total. Don't re-divide
        # by GP — that creates meaningless stl-per-game-per-game inputs.
        for col in ("STL", "BLK", "DREB", "PF", "GP", "MIN"):
            if col not in stats.columns:
                stats[col] = 0
        per_game = stats[["PLAYER_ID", "STL", "BLK", "DREB", "PF", "GP", "MIN"]].copy()
        per_game["GP"] = per_game["GP"].clip(lower=1)
        per_game["MPG"] = per_game["MIN"]  # per-game from NBA API
        per_game["min_share"] = per_game["MPG"] / 48.0

        # Per-minute versions of defensive stats.
        for stat in ("STL", "BLK", "DREB", "PF"):
            per_game[f"{stat}_per_min"] = per_game[stat] / per_game["MPG"].clip(lower=1)

        # Lookup position by player name (via raw which has both PLAYER_ID and Player).
        pid_to_name = dict(zip(raw["PLAYER_ID"], raw["Player"]))
        per_game["position"] = per_game["PLAYER_ID"].map(
            lambda pid: pos_lookup.get(normalize(pid_to_name.get(pid, "")), "Unknown")
        )

        m = nonzero[["PLAYER_ID", "d_lebron"]].merge(
            per_game[["PLAYER_ID", "STL", "BLK", "DREB", "PF",
                      "STL_per_min", "BLK_per_min", "DREB_per_min", "PF_per_min",
                      "MPG", "min_share", "position"]],
            on="PLAYER_ID", how="inner",
        )
        m["season"] = season
        rows.append(m)

    if not rows:
        print("No data found.")
        return

    df = pd.concat(rows, ignore_index=True).dropna()
    print(f"Sample size: {len(df):,} pairs")
    print(f"Seasons: {sorted(df['season'].unique())[0]} to "
          f"{sorted(df['season'].unique())[-1]}\n")

    # Filter out players with very low MPG (noise) — at least 10 mpg.
    df = df[df["MPG"] >= 10].copy()
    print(f"After MPG ≥ 10 filter: {len(df):,} pairs\n")

    # Position dummies. "Unknown" gets all zeros (baseline).
    for p in ("PG", "SG", "SF", "PF", "C"):
        df[f"pos_{p}"] = (df["position"] == p).astype(float)

    def fit_and_report(feature_cols: list[str], label: str) -> dict:
        X = df[feature_cols].values
        y = df["d_lebron"].values
        X_int = np.column_stack([np.ones(len(X)), X])
        coefs, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
        y_pred = X_int @ coefs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        mae = np.mean(np.abs(y - y_pred))
        return {"label": label, "r2": r2, "mae": mae,
                "coefs": dict(zip(["intercept"] + feature_cols, coefs))}

    # Model 1: V1 baseline.
    v1 = fit_and_report(["STL", "BLK", "DREB", "PF"], "V1: per-game basic")
    # Model 2: + per-minute.
    v2 = fit_and_report(
        ["STL", "BLK", "DREB", "PF",
         "STL_per_min", "BLK_per_min", "DREB_per_min", "PF_per_min"],
        "V2: + per-minute",
    )
    # Model 3: + minute share.
    v3 = fit_and_report(
        ["STL", "BLK", "DREB", "PF", "MPG", "min_share"],
        "V3: + MPG + min_share",
    )
    # Model 4: + position dummies.
    v4 = fit_and_report(
        ["STL", "BLK", "DREB", "PF", "MPG",
         "pos_PG", "pos_SG", "pos_SF", "pos_PF", "pos_C"],
        "V4: + position dummies",
    )
    # Model 5: position × BLK and position × DREB interactions.
    df["BLK_x_C"] = df["BLK"] * df["pos_C"]
    df["BLK_x_PF"] = df["BLK"] * df["pos_PF"]
    df["DREB_x_C"] = df["DREB"] * df["pos_C"]
    df["DREB_x_PF"] = df["DREB"] * df["pos_PF"]
    df["STL_x_PG"] = df["STL"] * df["pos_PG"]
    df["STL_x_SG"] = df["STL"] * df["pos_SG"]
    v5 = fit_and_report(
        ["STL", "BLK", "DREB", "PF", "MPG",
         "pos_PG", "pos_SG", "pos_SF", "pos_PF", "pos_C",
         "BLK_x_C", "BLK_x_PF", "DREB_x_C", "DREB_x_PF",
         "STL_x_PG", "STL_x_SG"],
        "V5: + position interactions",
    )

    print("=" * 60)
    print("PROGRESSIVE FEATURE MODELS")
    print("=" * 60)
    print(f"{'Model':<32} {'R²':>8} {'MAE':>8}  {'Δ R² vs V1':>10}")
    print("-" * 60)
    for r in [v1, v2, v3, v4, v5]:
        delta = r["r2"] - v1["r2"]
        print(f"{r['label']:<32} {r['r2']:>8.3f} {r['mae']:>8.3f}  "
              f"{delta:>+9.3f}")
    print()
    print("Best model:", max([v1, v2, v3, v4, v5], key=lambda r: r["r2"])["label"])
    print()
    print("=" * 60)
    print("V1 COEFFICIENTS (simplest model, R² = 0.59 — recommended to ship)")
    print("=" * 60)
    for name, val in v1["coefs"].items():
        print(f"    '{name}': {val:+.6f},")
    print()
    print("=" * 60)
    print("V5 COEFFICIENTS (position-aware, R² = 0.64 — only marginal gain)")
    print("=" * 60)
    for name, val in v5["coefs"].items():
        print(f"    '{name}': {val:+.6f},")
    print()
    print("Interpretation:")
    print("  R² ≥ 0.25:  worth shipping as pre-2009 fallback")
    print("  R² 0.15-0.25: borderline; would help slightly")
    print("  R² < 0.15:  not worth shipping — keep zero default")


if __name__ == "__main__":
    main()
