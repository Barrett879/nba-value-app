"""Show the 'math' behind specific predicted salaries.

The model is a HistGradientBoostingRegressor (≈800 trees) — no closed-form
formula. The honest decomposition is SHAP: each feature's additive push
(in % of cap) on top of the model's baseline, summing to the raw prediction.
Then the pipeline: raw % → × season cap → CBA floor/cap → final $.

For each (player, season) we train EXACTLY as the validation does — on prior
seasons only ([2012, season)) — so the number matches the miss list.

Usage:
    python -u scripts/explain_predictions.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import shap
from sklearn.ensemble import HistGradientBoostingRegressor

from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
from train_ml_model_v3 import PRUNED_FEATURES
from build_production_histgbm import (
    make_X_augmented, HISTGBM_PARAMS, TRAINING_START_YEAR, ADV_COLS,
    _is_bad_data, apply_cba_postprocess,
)
from utils import SALARY_CAP_M, normalize as _norm

FEATURE_NAMES = (PRUNED_FEATURES
                 + ["age^2", "barrett^2", "log(rank+1)", "service^2",
                    "tier30", "tier35", "is_Guard", "is_Forward"]
                 + ADV_COLS)

# (player, season)
TARGETS = [
    ("Michael Porter Jr.", "2022-23"),
    ("Jordan Clarkson",    "2023-24"),
    ("Gary Trent Jr.",     "2021-22"),
    ("Montrezl Harrell",   "2022-23"),
    ("Jordan Clarkson",    "2024-25"),
    ("Spencer Dinwiddie",  "2024-25"),
    ("Paul Reed",          "2024-25"),
    ("Brook Lopez",        "2025-26"),
    ("Brook Lopez",        "2023-24"),
    ("Jalen Brunson",      "2022-23"),
    ("Deandre Ayton",      "2025-26"),
    ("James Harden",       "2022-23"),
]


def main():
    print("Building pool...", flush=True)
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    X = make_X_augmented(df)
    sy = df["start_year"].values

    # group targets by season → one model per season (matches validation).
    by_season = {}
    for p, s in TARGETS:
        by_season.setdefault(s, []).append(p)

    for season, players in by_season.items():
        yr = int(season.split("-")[0])
        trm = (sy >= TRAINING_START_YEAR) & (sy < yr)  # match list_big_misses training
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        explainer = shap.TreeExplainer(reg)
        base_pct = float(explainer.expected_value)
        cap = SALARY_CAP_M[season] * 1_000_000

        for player in players:
            mask = (df["player"].map(lambda v: _norm(str(v))) == _norm(player)) & (df["curr"] == season)
            if not mask.any():
                print(f"\n{player} {season}: not found"); continue
            i = np.where(mask.values)[0][0]
            xrow = X[i:i+1]
            sv = explainer.shap_values(xrow)[0]
            raw_pct = base_pct + sv.sum()
            raw_dollars = raw_pct * cap
            final_pct = float(apply_cba_postprocess(np.array([raw_pct]), df.iloc[[i]])[0])
            final_dollars = final_pct * cap
            actual = df.iloc[i]["salary_curr"]

            print("\n" + "=" * 78, flush=True)
            print(f"{player}  —  {season}", flush=True)
            print("=" * 78, flush=True)
            # key raw inputs
            row = df.iloc[i]
            print(f"  inputs:  barrett(3yr-wt) {row['barrett']:.1f} · score_rank "
                  f"#{int(row['score_rank'])} · age {row['age']:.0f} · "
                  f"{int(row['years_in_league'])} yrs svc · All-NBA(3y) {int(row['all_nba_3yr'])} · "
                  f"prior salary {row['salary_prev_pct']*100:.1f}% of cap", flush=True)
            # SHAP decomposition (in % of cap, ×cap shown in $M)
            print(f"\n  baseline (avg prediction):      {base_pct*100:5.1f}% of cap "
                  f"= ${base_pct*cap/1e6:5.1f}M", flush=True)
            order = np.argsort(-np.abs(sv))
            shown = 0
            for j in order:
                if shown >= 8 or abs(sv[j] * cap) < 0.2e6:
                    break
                print(f"    {'+' if sv[j]>=0 else '-'} {FEATURE_NAMES[j]:<20} "
                      f"{sv[j]*100:+5.2f}pp  ({sv[j]*cap/1e6:+5.1f}M)", flush=True)
                shown += 1
            print(f"  ----------------------------------------------------------------", flush=True)
            print(f"  raw model output:               {raw_pct*100:5.1f}% of cap "
                  f"= ${raw_dollars/1e6:5.1f}M", flush=True)
            if abs(final_pct - raw_pct) > 1e-6:
                print(f"  after CBA floor/cap:            {final_pct*100:5.1f}% of cap "
                      f"= ${final_dollars/1e6:5.1f}M", flush=True)
            print(f"  PREDICTED:  ${final_dollars/1e6:.1f}M     ACTUAL: ${actual/1e6:.1f}M  "
                  f"(miss {abs(final_dollars-actual)/cap*100:+.0f}% of cap)", flush=True)


if __name__ == "__main__":
    main()
