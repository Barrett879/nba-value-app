"""Refine the max-tier floor: does lowering the trigger to 0.20 (to catch
Davis/Beal/Gobert, who sit just under 0.22) + adding an age gate (to stop
flooring aging discounters like Chris Paul 36) help predict FUTURE contracts
better than the current floor (0.22, no age gate)?

Same honest test: expanding-window temporal CV, per-year. Ship a variant only
if it beats the current floor across years (wins most, hurts ~none).

Usage:
    python -u scripts/test_floor_v2.py
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
    make_X_augmented, HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask, cba_max_pct,
    PRED_FLOOR_PCT, PRED_CEIL_PCT,
)

TEST_YEARS = list(range(2018, 2026))

# (label, trigger, age_cap)
CONFIGS = [
    ("current (0.22, no age)", 0.22, 99),
    ("0.20 + age<=33",         0.20, 33),
    ("0.20 + age<=31",         0.20, 31),
    ("0.20 + no age",          0.20, 99),
    ("0.22 + age<=33",         0.22, 33),
]


def apply_floor(pred_pct, sub, thr, age_cap):
    out = np.clip(pred_pct, PRED_FLOOR_PCT, PRED_CEIL_PCT).astype(float)
    svc = sub["years_in_league"].values
    ann = sub["all_nba_3yr"].values
    age = sub["age"].values
    for i in range(len(out)):
        a = age[i]
        age_ok = (a is None) or (isinstance(a, float) and np.isnan(a)) or (a <= age_cap)
        if out[i] >= thr and (ann[i] or 0) >= 1 and age_ok:
            out[i] = max(out[i], cba_max_pct(svc[i], ann[i]))
    return out


def main():
    print("Building pool...", flush=True)
    careers = build_career_indexes(playoffs=False)
    df = build_rows(PAIRS, careers, {}, fetch_all_nba_selections()).reset_index(drop=True)
    X = make_X_augmented(df)
    sy = df["start_year"].values
    grade_ok = gradeable_mask(df).values

    # Pre-train one model per test year, cache predictions.
    folds = []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade_ok
        if trm.sum() < 150 or tem.sum() < 10:
            continue
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem]
        rawpred = reg.predict(X[tem])
        actual = sub["salary_curr"].values / sub["cap_curr"].values
        folds.append((ty, sub, rawpred, actual))

    def w5_for(thr, age_cap):
        per_year, ns = [], []
        for ty, sub, rawpred, actual in folds:
            flo = apply_floor(rawpred, sub, thr, age_cap)
            per_year.append(np.mean(np.abs(flo - actual) * 100 <= 5) * 100)
            ns.append(len(sub))
        return np.array(per_year), np.array(ns)

    # Reference: current config per-year.
    cur_py, ns = w5_for(0.22, 99)

    print("\n" + "=" * 74, flush=True)
    print("FORWARD CV — weighted within-5%, and per-year vs CURRENT floor", flush=True)
    print("=" * 74, flush=True)
    print(f"  {'config':<24}{'wt w5':>8}{'Δ vs cur':>10}{'wins/loses (yr)':>18}", flush=True)
    for label, thr, age_cap in CONFIGS:
        py, _ = w5_for(thr, age_cap)
        wt = np.average(py, weights=ns)
        d = py - cur_py
        wins = int((d > 0.01).sum()); loses = int((d < -0.01).sum())
        cur_wt = np.average(cur_py, weights=ns)
        print(f"  {label:<24}{wt:>7.2f}%{wt-cur_wt:>+9.2f}{f'{wins}/{loses}':>18}", flush=True)

    # Specific players: are Davis/Beal/Gobert caught and CP3/Harden spared?
    print("\n" + "=" * 74, flush=True)
    print("KEY PLAYERS — current (0.22) vs candidate (0.20 + age<=33)", flush=True)
    print("=" * 74, flush=True)
    from utils import normalize as _norm, SALARY_CAP_M
    checks = [("Anthony Davis","2025-26"),("Bradley Beal","2022-23"),("Rudy Gobert","2021-22"),
              ("Chris Paul","2021-22"),("James Harden","2022-23"),("Jalen Brunson","2025-26")]
    for ty, sub, rawpred, actual in folds:
        for name, season in checks:
            if int(season.split("-")[0]) != ty:
                continue
            m = sub["player"].map(lambda v: _norm(str(v))) == _norm(name)
            if not m.any():
                continue
            j = np.where(m.values)[0][0]
            cap = SALARY_CAP_M[season] * 1e6
            cur = apply_floor(rawpred, sub, 0.22, 99)[j] * cap
            cand = apply_floor(rawpred, sub, 0.20, 33)[j] * cap
            act = sub.iloc[j]["salary_curr"]
            print(f"  {name:<16}{season}:  cur ${cur/1e6:5.1f}M  cand ${cand/1e6:5.1f}M  "
                  f"(actual ${act/1e6:.1f}M)", flush=True)


if __name__ == "__main__":
    main()
