"""R&D: tighten the CBA max-floor / tier over-credit.

Probe found the real, fixable error: apply_cba_postprocess OVER-assigns the max
tier. It (1) bumps a player a full max tier for just 1 recent All-NBA in
cba_max_pct, and (2) snaps the prediction UP to that tier whenever model >= 20%
+ >=1 All-NBA. Result: Harden pred 32% vs actual 26.7%, Morant 29.3 vs 25, etc.

This sweeps stricter variants of that logic against the shipped baseline on the
same 2021-2025 walk-forward OOS, tier-broken-down. Keep only what reduces the
max/star over-projection without hurting other tiers.

Usage:  python -u scripts/experiment_maxfloor.py
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
import build_production_histgbm as B
from build_production_histgbm import (
    make_X_augmented, gradeable_mask, HISTGBM_PARAMS, TRAINING_START_YEAR,
    PRED_FLOOR_PCT, PRED_CEIL_PCT,
)

CURRENT_CAP_M = 165.0
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def cba_max_pct_v(svc, ann, elite_thresh):
    """max % with a configurable All-NBA count needed to bump a tier."""
    elite = (ann or 0) >= elite_thresh
    s = svc or 0
    if s <= 6:  return 0.30 if elite else 0.25
    if s <= 9:  return 0.35 if elite else 0.30
    return 0.35


def postprocess_v(pred_pct, df, trigger, elite_thresh, discount, floor_elite_thresh):
    """Configurable variant of apply_cba_postprocess."""
    out = np.clip(pred_pct, PRED_FLOOR_PCT, PRED_CEIL_PCT).astype(float)
    svc = df["years_in_league"].values
    ann = df["all_nba_3yr"].values
    age = df["age"].values if "age" in df.columns else np.full(len(out), 30.0)
    for i in range(len(out)):
        a = age[i]
        age_ok = a is None or (isinstance(a, float) and np.isnan(a)) or a <= B.MAX_FLOOR_AGE_CAP
        if out[i] >= trigger and (ann[i] or 0) >= floor_elite_thresh and age_ok:
            target = max(cba_max_pct_v(svc[i], ann[i], elite_thresh) - discount, trigger)
            out[i] = max(out[i], target)
    return out


def collect(df, X, pp):
    sy = df["start_year"].values; grade_ok = gradeable_mask(df).values
    acts, preds, caps = [], [], []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty); tem = (sy == ty) & grade_ok
        if trm.sum() < 100 or tem.sum() < 5: continue
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X[trm], df.loc[trm,"salary_curr_pct"].values)
        sub = df[tem]; cap = sub["cap_curr"].values
        pred = pp(reg.predict(X[tem]), sub) * cap
        acts.append(sub["salary_curr"].values); preds.append(pred); caps.append(cap)
    return np.concatenate(acts), np.concatenate(preds), np.concatenate(caps)


def tiers_of(actual, pred, cap):
    err_pct = np.abs(actual-pred)/cap*100
    a_M = actual/cap*CURRENT_CAP_M; p_M = pred/cap*CURRENT_CAP_M
    e_M = np.abs(a_M-p_M); b_M = p_M-a_M
    w5 = np.mean(err_pct<=5)*100
    defs = [("Max",a_M>=40),("BigStar",(a_M>=25)&(a_M<40)),("Mid",(a_M>=15)&(a_M<25)),
            ("Rot",(a_M>=7)&(a_M<15)),("Min",a_M<7)]
    out = {}
    for name,m in defs:
        if m.sum()==0: continue
        out[name] = (np.median(b_M[m]), np.mean(e_M[m]<=3)*100)
    return w5, out


def main():
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    X = make_X_augmented(df)
    print(f"{len(df)} contracts\n", flush=True)

    variants = {
        "baseline (ship)": B.apply_cba_postprocess,
        "elite>=2 tier":   lambda p,d: postprocess_v(p,d, B.MAX_FLOOR_TRIGGER, 2, B.MAX_FLOOR_DISCOUNT, 1),
        "floor elite>=2":  lambda p,d: postprocess_v(p,d, B.MAX_FLOOR_TRIGGER, 1, B.MAX_FLOOR_DISCOUNT, 2),
        "both elite>=2":   lambda p,d: postprocess_v(p,d, B.MAX_FLOOR_TRIGGER, 2, B.MAX_FLOOR_DISCOUNT, 2),
        "discount .05":    lambda p,d: postprocess_v(p,d, B.MAX_FLOOR_TRIGGER, 1, 0.05, 1),
        "both+disc.05":    lambda p,d: postprocess_v(p,d, B.MAX_FLOOR_TRIGGER, 2, 0.05, 2),
        "no floor":        lambda p,d: postprocess_v(p,d, 1.0, 1, B.MAX_FLOOR_DISCOUNT, 1),  # trigger=1 disables
    }
    rows = {}
    for name, pp in variants.items():
        a,p,c = collect(df, X, pp)
        w5, t = tiers_of(a,p,c)
        rows[name] = (w5, t)
        print(f"{name:<16} within5%={w5:.1f}%   "
              + "  ".join(f"{k}:bias${t[k][0]:+.1f}M/±3M{t[k][1]:.0f}%" for k in ("Max","BigStar","Mid","Rot")),
              flush=True)


if __name__ == "__main__":
    main()
