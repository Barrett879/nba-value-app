"""Will the max-tier floor help predict FUTURE contracts — or did it just
happen to net positive on one test window?

Honest forward test: expanding-window temporal CV. For each season Y from
2018-2025, train ONLY on prior seasons [2012, Y) and predict Y. Score each
year WITH the max-floor and WITHOUT (paired — same predictions, floor toggled).
If the floor helps in MOST years, it's a real forward improvement. If the
per-year deltas bounce (some +, some -), it's noise and we don't ship it.

Also sweeps the snap threshold (0.20 / 0.22 / 0.25) so the gain isn't an
artifact of one tuned cutoff.

Usage:
    python -u scripts/test_floor_forward.py
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
    make_X_augmented, HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask, apply_cba_postprocess,
)

TEST_YEARS = list(range(2018, 2026))   # 8 forward seasons


def cba_max_pct(service, all_nba):
    elite = (all_nba or 0) >= 1
    s = service or 0
    if s <= 6:  return 0.30 if elite else 0.25
    if s <= 9:  return 0.35 if elite else 0.30
    return 0.35


def floored(pred_pct, sub, thr):
    out = pred_pct.copy()
    svc = sub["years_in_league"].values
    ann = sub["all_nba_3yr"].values
    for i in range(len(out)):
        if out[i] >= thr and (ann[i] or 0) >= 1:
            out[i] = max(out[i], cba_max_pct(svc[i], ann[i]))
    return out


def main():
    print("Building pool...", flush=True)
    careers = build_career_indexes(playoffs=False)
    df = build_rows(PAIRS, careers, {}, fetch_all_nba_selections()).reset_index(drop=True)
    X = make_X_augmented(df)
    sy = df["start_year"].values
    grade_ok = gradeable_mask(df).values

    print("\n" + "=" * 72, flush=True)
    print("EXPANDING-WINDOW TEMPORAL CV — per-year, raw vs +max-floor (thr=0.22)", flush=True)
    print("=" * 72, flush=True)
    print(f"  {'year':<6}{'n':>5}{'raw w5':>9}{'floor w5':>10}{'Δ':>8}", flush=True)
    rows = []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade_ok
        if trm.sum() < 150 or tem.sum() < 10:
            continue
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem]
        a = sub["salary_curr"].values / sub["cap_curr"].values
        raw = apply_cba_postprocess(reg.predict(X[tem]), sub)
        flo = floored(raw, sub, 0.22)
        w5_raw = np.mean(np.abs(raw - a) * 100 <= 5) * 100
        w5_flo = np.mean(np.abs(flo - a) * 100 <= 5) * 100
        rows.append((ty, len(sub), w5_raw, w5_flo))
        print(f"  {ty:<6}{len(sub):>5}{w5_raw:>8.1f}%{w5_flo:>9.1f}%{w5_flo-w5_raw:>+7.1f}", flush=True)

    deltas = np.array([r[3] - r[2] for r in rows])
    ns = np.array([r[1] for r in rows])
    wins = int((deltas > 0).sum()); losses = int((deltas < 0).sum()); ties = int((deltas == 0).sum())
    wmean = float(np.average(deltas, weights=ns))
    print("  " + "-" * 40, flush=True)
    print(f"  floor wins {wins} yrs, loses {losses}, ties {ties}  ·  weighted Δ {wmean:+.2f}pp", flush=True)

    # threshold robustness (weighted-mean Δ across all years)
    print("\n  threshold robustness (weighted-mean Δ across years):", flush=True)
    for thr in [0.20, 0.22, 0.25]:
        ds, nsx = [], []
        for ty in TEST_YEARS:
            trm = (sy >= TRAINING_START_YEAR) & (sy < ty); tem = (sy == ty) & grade_ok
            if trm.sum() < 150 or tem.sum() < 10:
                continue
            reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
                X[trm], df.loc[trm, "salary_curr_pct"].values)
            sub = df[tem]; a = sub["salary_curr"].values / sub["cap_curr"].values
            raw = apply_cba_postprocess(reg.predict(X[tem]), sub)
            flo = floored(raw, sub, thr)
            ds.append(np.mean(np.abs(flo-a)*100<=5)*100 - np.mean(np.abs(raw-a)*100<=5)*100)
            nsx.append(len(sub))
        print(f"    thr {thr}: {np.average(ds, weights=nsx):+.2f}pp", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("VERDICT", flush=True)
    print("=" * 72, flush=True)
    if wins >= losses + 3 and wmean > 0.3:
        print(f"  HELPS FORWARD — floor wins {wins}/{len(rows)} years, +{wmean:.2f}pp. Ship it.", flush=True)
    elif abs(wmean) <= 0.3 or wins <= losses + 1:
        print(f"  NOISE — wins {wins} loses {losses}, Δ {wmean:+.2f}pp. Not a reliable forward gain; don't ship.", flush=True)
    else:
        print(f"  MIXED — wins {wins} loses {losses}, Δ {wmean:+.2f}pp. Borderline.", flush=True)


if __name__ == "__main__":
    main()
