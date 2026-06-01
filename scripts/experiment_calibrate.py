"""De-bias experiment: fold-honest calibration of the contract model.

The shipped model systematically UNDER-projects (negative signed bias that
worsens with salary) — the classic GBM regression-to-the-mean shrinkage. This
fits a monotonic calibrator (isotonic) mapping raw predicted % of cap -> actual
% of cap, learned ONLY on each fold's training data (no leakage), applied to the
raw model output BEFORE apply_cba_postprocess.

Compares baseline vs calibrated on the same 2021-2025 walk-forward OOS pool,
broken down by real-dollar salary tier. Decision rule: keep calibration only if
it moves signed bias toward 0 without hurting the ±$3M hit-rate.

Usage:  python -u scripts/experiment_calibrate.py
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
from build_production_histgbm import (
    make_X_augmented, apply_cba_postprocess, gradeable_mask,
    HISTGBM_PARAMS, TRAINING_START_YEAR,
)

CURRENT_CAP_M = 165.0
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def fit_calibrator(X_tr, y_tr):
    """Learn raw-pred -> actual mapping via 5-fold OOF on the TRAINING data, so
    the calibrator sees the model's out-of-sample error pattern (not its
    in-sample, near-perfect fit). Returns a fitted IsotonicRegression."""
    oof = np.zeros(len(y_tr))
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for itr, iva in kf.split(X_tr):
        m = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X_tr[itr], y_tr[itr])
        oof[iva] = m.predict(X_tr[iva])
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y_tr)
    return iso


def collect(df, X, calibrate):
    sy = df["start_year"].values
    grade_ok = gradeable_mask(df).values
    acts, preds, caps = [], [], []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade_ok
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        y_tr = df.loc[trm, "salary_curr_pct"].values
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X[trm], y_tr)
        raw = reg.predict(X[tem])
        if calibrate:
            iso = fit_calibrator(X[trm], y_tr)
            raw = iso.predict(raw)
        sub = df[tem]
        cap = sub["cap_curr"].values
        pred = apply_cba_postprocess(raw, sub) * cap
        acts.append(sub["salary_curr"].values)
        preds.append(pred)
        caps.append(cap)
    return np.concatenate(acts), np.concatenate(preds), np.concatenate(caps)


def summarize(tag, actual, pred, cap):
    err_pct = np.abs(actual - pred) / cap * 100
    a_M = actual / cap * CURRENT_CAP_M
    p_M = pred / cap * CURRENT_CAP_M
    e_M = np.abs(a_M - p_M)
    b_M = p_M - a_M
    print(f"\n{tag}: n={len(actual)}  within5%={np.mean(err_pct<=5)*100:.1f}%  "
          f"within10%={np.mean(err_pct<=10)*100:.1f}%  "
          f"med|err|=${np.median(e_M):.2f}M  medbias=${np.median(b_M):+.2f}M", flush=True)
    tiers = [
        ("Max/super", a_M >= 40), ("Big star", (a_M >= 25) & (a_M < 40)),
        ("Mid-tier", (a_M >= 15) & (a_M < 25)), ("Rotation", (a_M >= 7) & (a_M < 15)),
        ("Min-ish", a_M < 7),
    ]
    rows = {}
    for name, m in tiers:
        k = int(m.sum())
        if k == 0:
            continue
        rows[name] = (k, np.median(e_M[m]), np.median(b_M[m]),
                      np.mean(e_M[m] <= 3) * 100, np.mean(e_M[m] <= 5) * 100)
    return rows


def main():
    print("Building 2012+ pool...", flush=True)
    t0 = time.time()
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    X = make_X_augmented(df)
    print(f"  {len(df)} contracts in {time.time()-t0:.1f}s", flush=True)

    a0, p0, c0 = collect(df, X, calibrate=False)
    a1, p1, c1 = collect(df, X, calibrate=True)
    r0 = summarize("BASELINE   ", a0, p0, c0)
    r1 = summarize("CALIBRATED ", a1, p1, c1)

    print("\n" + "=" * 78, flush=True)
    print("TIER COMPARISON  (med bias → toward $0 is better; ±$3M → higher is better)", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'tier':<11} {'n':>4}  {'bias base':>9} {'bias cal':>9}  "
          f"{'±3M base':>8} {'±3M cal':>8}", flush=True)
    for name in r0:
        n, _, b0_, w0_, _ = r0[name]
        _, _, b1_, w1_, _ = r1.get(name, (0, 0, 0, 0, 0))
        print(f"  {name:<11} {n:>4}  ${b0_:>+7.2f}M ${b1_:>+7.2f}M  "
              f"{w0_:>7.0f}% {w1_:>7.0f}%", flush=True)


if __name__ == "__main__":
    main()
