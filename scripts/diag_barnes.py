"""Why does the LIVE APP say Scottie Barnes -> $38.6M (near-exact) but the
temporal-CV miss-list says $27.9M (a 6.9% undershoot)?

Isolate the cause by running THREE numbers on Barnes's identical feature row:
  1. PRODUCTION model (models/contract_histgbm_v2.joblib, trained on ALL
     2012-2025 gradeable data) -- this is what the app loads.
  2. TEMPORAL-CV model (HistGBM trained ONLY on 2012-2024, never saw 2025-26)
     -- this is what prove_misses.py / the headline accuracy uses.
For each: raw model %, raw $, then after CBA post-processing (cap + floor).
Also print Barnes's CBA max tier so we can see if the cap is doing the work.

Usage:
    python -u scripts/diag_barnes.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
from build_production_histgbm import (
    make_X_augmented, HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask,
    apply_cba_postprocess, cba_max_pct,
)
from utils import normalize as _norm

NAME, SEASON, START = "Scottie Barnes", "2025-26", 2025


def main():
    careers = build_career_indexes(playoffs=False)
    df = build_rows(PAIRS, careers, {}, fetch_all_nba_selections()).reset_index(drop=True)
    X = make_X_augmented(df)
    sy = df["start_year"].values

    m = df["player"].map(lambda v: _norm(str(v))) == _norm(NAME)
    m &= (df["curr"] == SEASON)
    idx = np.where(m.values)[0]
    if len(idx) == 0:
        print("Barnes row not found"); return
    j = idx[0]
    row = df.iloc[j]
    cap = row["cap_curr"]
    actual = row["salary_curr"]
    svc = row.get("years_in_league"); ann = row.get("all_nba_3yr", 0); age = row.get("age")
    xrow = X[j:j+1]

    print("=" * 78)
    print(f"{NAME} {SEASON}  —  age {age:.0f}, service {svc}, All-NBA(3yr) {int(ann or 0)}")
    print(f"cap ${cap/1e6:.1f}M   actual ${actual/1e6:.2f}M ({actual/cap*100:.1f}% of cap)")
    print(f"CBA max tier for him: {cba_max_pct(svc, ann)*100:.0f}% = ${cba_max_pct(svc, ann)*cap/1e6:.1f}M")
    print("=" * 78)

    # 1. PRODUCTION model (full data) — what the app loads
    bundle = joblib.load("models/contract_histgbm_v2.joblib")
    prod = bundle["model"] if isinstance(bundle, dict) else bundle
    print(f"\n[production artifact: trained_on={bundle.get('trained_on')}, "
          f"n_train_rows={bundle.get('n_train_rows')}]")
    p_raw = float(prod.predict(xrow)[0])
    p_post = float(apply_cba_postprocess(np.array([p_raw]), df.iloc[j:j+1])[0])
    print("\n1. PRODUCTION model (trained on ALL 2012-2025 — what the LIVE APP uses):")
    print(f"     raw model:  {p_raw*100:5.1f}% of cap  = ${p_raw*cap/1e6:5.1f}M")
    print(f"     after cap/floor: {p_post*100:5.1f}% = ${p_post*cap/1e6:5.1f}M")

    # 2. TEMPORAL-CV model (train only on <2025) — what the miss-list uses
    trm = (sy >= TRAINING_START_YEAR) & (sy < START)
    cv = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
        X[trm], df.loc[trm, "salary_curr_pct"].values)
    c_raw = float(cv.predict(xrow)[0])
    c_post = float(apply_cba_postprocess(np.array([c_raw]), df.iloc[j:j+1])[0])
    print(f"\n2. TEMPORAL-CV model (trained ONLY on 2012-2024, n={int(trm.sum())} — what the MISS-LIST uses):")
    print(f"     raw model:  {c_raw*100:5.1f}% of cap  = ${c_raw*cap/1e6:5.1f}M")
    print(f"     after cap/floor: {c_post*100:5.1f}% = ${c_post*cap/1e6:5.1f}M")

    # Is Barnes's own contract in the production training set?
    in_prod = bool(((sy >= TRAINING_START_YEAR) & gradeable_mask(df).values & m.values).any())
    print("\n" + "=" * 78)
    print(f"Is Barnes 2025-26 inside the PRODUCTION training data?  {in_prod}")
    print(f"Gap: production ${p_post*cap/1e6:.1f}M  vs  temporal-CV ${c_post*cap/1e6:.1f}M  "
          f"(actual ${actual/1e6:.1f}M)")
    print("=" * 78)


if __name__ == "__main__":
    main()
