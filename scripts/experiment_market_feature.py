"""R&D: does a market-comp feature tighten the contract model?

Idea: the model predicts from a player's own production. The one thing it can't
see is "what teams actually pay players like this" — the negotiation/role/timing
context. Add a LEAK-FREE market feature: for each contract, the (distance-
weighted) median salary-%-of-cap of SIMILAR PRIOR signings (strictly earlier
start_year, so no leakage), matched on trailing Barrett + age + position.

Compares baseline (28 feat) vs +market (29 feat) on the same 2021-2025
walk-forward OOS, tier-broken-down. Ship only if it genuinely improves.

Usage:  python -u scripts/experiment_market_feature.py
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
from build_production_histgbm import (
    make_X_augmented, apply_cba_postprocess, gradeable_mask,
    HISTGBM_PARAMS, TRAINING_START_YEAR,
)

CURRENT_CAP_M = 165.0
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def market_feature(df):
    """For every row, distance-weighted median salary_curr_pct of the K nearest
    PRIOR signings (start_year strictly earlier) by trailing-Barrett + age +
    position. Leak-free: only earlier seasons inform each row. Falls back to the
    prior-seasons global median when too few neighbours exist."""
    bar = df["barrett"].values.astype(float)
    age = df["age"].values.astype(float)
    pos = df["pos_bucket"].astype(str).values
    sal = df["salary_curr_pct"].values.astype(float)
    yr  = df["start_year"].values.astype(int)
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        prior = yr < yr[i]
        if prior.sum() < 20:
            continue
        # distance in standardized Barrett+age, big penalty for different position
        db = (bar[prior] - bar[i]) / 6.0
        da = (age[prior] - age[i]) / 3.0
        dpos = np.where(pos[prior] == pos[i], 0.0, 1.5)
        dist = np.sqrt(db * db + da * da) + dpos
        order = np.argsort(dist)[:40]
        w = 1.0 / (dist[order] + 0.5)
        s = sal[prior][order]
        # weighted median
        idx = np.argsort(s)
        sw = s[idx]; ww = w[idx]
        c = np.cumsum(ww)
        out[i] = sw[np.searchsorted(c, c[-1] * 0.5)]
    # fill any remaining NaN with the global prior median (or overall median)
    med = np.nanmedian(out)
    out = np.where(np.isnan(out), med, out)
    return out.reshape(-1, 1)


def collect(df, X):
    sy = df["start_year"].values
    grade_ok = gradeable_mask(df).values
    acts, preds, caps = [], [], []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade_ok
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem]
        cap = sub["cap_curr"].values
        pred = apply_cba_postprocess(reg.predict(X[tem]), sub) * cap
        acts.append(sub["salary_curr"].values); preds.append(pred); caps.append(cap)
    return np.concatenate(acts), np.concatenate(preds), np.concatenate(caps)


def summarize(tag, actual, pred, cap):
    err_pct = np.abs(actual - pred) / cap * 100
    a_M = actual / cap * CURRENT_CAP_M; p_M = pred / cap * CURRENT_CAP_M
    e_M = np.abs(a_M - p_M); b_M = p_M - a_M
    print(f"\n{tag}: n={len(actual)}  within5%={np.mean(err_pct<=5)*100:.1f}%  "
          f"within10%={np.mean(err_pct<=10)*100:.1f}%  med|err|=${np.median(e_M):.2f}M  "
          f"medbias=${np.median(b_M):+.2f}M", flush=True)
    tiers = [("Max/super", a_M>=40), ("Big star", (a_M>=25)&(a_M<40)),
             ("Mid-tier", (a_M>=15)&(a_M<25)), ("Rotation", (a_M>=7)&(a_M<15)),
             ("Min-ish", a_M<7)]
    rows = {}
    for name, m in tiers:
        k = int(m.sum())
        if k == 0: continue
        rows[name] = (k, np.median(b_M[m]), np.mean(e_M[m]<=3)*100, np.mean(e_M[m]<=5)*100)
    return rows, np.mean(err_pct<=5)*100


def main():
    print("Building 2012+ pool...", flush=True)
    t0 = time.time()
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    X_base = make_X_augmented(df)
    print(f"  {len(df)} contracts in {time.time()-t0:.1f}s", flush=True)
    print("Building leak-free market feature...", flush=True)
    mkt = market_feature(df)
    X_mkt = np.hstack([X_base, mkt])

    r0, w5_0 = summarize("BASELINE (28)", *collect(df, X_base))
    r1, w5_1 = summarize("+MARKET  (29)", *collect(df, X_mkt))

    print("\n" + "=" * 74, flush=True)
    print("TIER COMPARISON  (bias→0 better, ±3M higher better)", flush=True)
    print("=" * 74, flush=True)
    print(f"  {'tier':<11} {'n':>4}  {'bias base':>9} {'bias +mkt':>9}  "
          f"{'±3M base':>8} {'±3M +mkt':>8}", flush=True)
    for name in r0:
        n, b0, w0, _ = r0[name]
        _, b1, w1, _ = r1.get(name, (0, 0, 0, 0))
        print(f"  {name:<11} {n:>4}  ${b0:>+7.2f}M ${b1:>+7.2f}M  "
              f"{w0:>7.0f}% {w1:>7.0f}%", flush=True)
    print(f"\n  Overall within-5%:  base {w5_0:.1f}%  →  +market {w5_1:.1f}%  "
          f"({w5_1-w5_0:+.1f}pp)", flush=True)


if __name__ == "__main__":
    main()
