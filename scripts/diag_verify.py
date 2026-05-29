"""Dump the exact dataset rows for the flagged contracts so each 'actual' can
be checked against the real-world deal. Shows salary_prev -> salary_curr, the
pct jump, age/service/All-NBA, and the temporal-CV model prediction, plus the
data-quality flags (rookie-lock / buyout-artifact / known-bad / gradeable).

Usage:
    python -u scripts/diag_verify.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
from build_production_histgbm import (
    make_X_augmented, HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask,
    apply_cba_postprocess, _is_rookie_lock, _is_buyout_artifact, _is_bad_data,
)
from utils import normalize as _norm

FLAGGED = [
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
    careers = build_career_indexes(playoffs=False)
    df = build_rows(PAIRS, careers, {}, fetch_all_nba_selections()).reset_index(drop=True)
    X = make_X_augmented(df)
    sy = df["start_year"].values

    rl = _is_rookie_lock(df).values
    bo = _is_buyout_artifact(df).values
    bd = _is_bad_data(df).values
    gok = gradeable_mask(df).values

    # cache one temporal-CV model per needed test year
    models = {}
    def model_for(ty):
        if ty not in models:
            trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
            models[ty] = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
                X[trm], df.loc[trm, "salary_curr_pct"].values)
        return models[ty]

    print("=" * 100)
    print(f"{'Player':<20}{'Season':<8}{'prev$':>8}{'curr$':>8}{'jump':>7}"
          f"{'cap$':>8}{'age':>4}{'svc':>4}{'aNBA':>5}{'pred$':>8}  flags")
    print("=" * 100)
    for name, season in FLAGGED:
        m = (df["player"].map(lambda v: _norm(str(v))) == _norm(name)) & (df["curr"] == season)
        idx = np.where(m.values)[0]
        if len(idx) == 0:
            print(f"{name:<20}{season:<8}  *** NOT IN DATASET (no qualifying salary jump) ***")
            continue
        j = idx[0]
        r = df.iloc[j]
        cap = r["cap_curr"]; ty = int(season.split("-")[0])
        pred = float(apply_cba_postprocess(model_for(ty).predict(X[j:j+1]), df.iloc[j:j+1])[0]) * cap
        jump = (r["salary_curr"] - r["salary_prev"]) / r["salary_prev"] * 100 if r["salary_prev"] else float("nan")
        flags = []
        if rl[j]: flags.append("ROOKIE-LOCK")
        if bo[j]: flags.append("BUYOUT-ARTIFACT")
        if bd[j]: flags.append("BAD-DATA")
        if not gok[j]: flags.append("(EXCLUDED from grading)")
        if not flags: flags.append("gradeable")
        print(f"{name:<20}{season:<8}{r['salary_prev']/1e6:>7.1f}M{r['salary_curr']/1e6:>7.1f}M"
              f"{jump:>+6.0f}%{cap/1e6:>7.1f}M{r['age']:>4.0f}{r.get('years_in_league',0):>4.0f}"
              f"{int(r.get('all_nba_3yr',0) or 0):>5}{pred/1e6:>7.1f}M  {', '.join(flags)}")


if __name__ == "__main__":
    main()
