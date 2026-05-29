"""Does dropping REB_PCT help? It adds a counterintuitive negative signal
(docks elite bigs for being bigs — Jokić/AD undershoot). Test rigorously:
paired 5-fold CV + temporal CV, with vs without REB_PCT on the same folds.

REB_PCT is the LAST column of make_X_augmented (ADV_COLS ends with it), so
"without" = drop the last column.

Decision: drop it if accuracy holds-or-improves AND the big-man undershoot
eases. Keep it if it clearly hurts.

Usage:
    python -u scripts/test_drop_rebpct.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
from build_production_histgbm import (
    make_X_augmented, HISTGBM_PARAMS, TRAINING_START_YEAR, ADV_COLS,
    gradeable_mask, apply_cba_postprocess, _is_bad_data,
)
from utils import SALARY_CAP_M, normalize as _norm

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
RNG = np.random.RandomState(7)


def w5_w10(actual, pred, cap):
    e = np.abs(actual - pred) / cap * 100
    return float(np.mean(e <= 5) * 100), float(np.mean(e <= 10) * 100)


def temporal(df, X, sy, grade_ok):
    w5s, w10s, ns = [], [], []
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
        a, b = w5_w10(sub["salary_curr"].values, pred, cap)
        w5s.append(a); w10s.append(b); ns.append(int(tem.sum()))
    ns = np.array(ns)
    return float(np.average(w5s, weights=ns)), float(np.average(w10s, weights=ns)), int(ns.sum())


def paired_5fold(df, Xw, Xwo, grade_ok):
    g = df[grade_ok]
    Xw_g, Xwo_g = Xw[grade_ok], Xwo[grade_ok]
    y = g["salary_curr_pct"].values
    cap = g["cap_curr"].values
    actual = g["salary_curr"].values
    d5, w_wins = [], 0
    for seed in range(1, 6):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tri, tei in kf.split(Xw_g):
            pw = apply_cba_postprocess(
                HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(Xw_g[tri], y[tri]).predict(Xw_g[tei]),
                g.iloc[tei]) * cap[tei]
            po = apply_cba_postprocess(
                HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(Xwo_g[tri], y[tri]).predict(Xwo_g[tei]),
                g.iloc[tei]) * cap[tei]
            a_w, _ = w5_w10(actual[tei], pw, cap[tei])
            a_o, _ = w5_w10(actual[tei], po, cap[tei])
            d5.append(a_o - a_w)            # +ve => without-REB is better
            w_wins += (a_o > a_w)
    d5 = np.array(d5)
    se = d5.std() / np.sqrt(len(d5))
    return float(d5.mean()), float(se), int(w_wins), len(d5)


def main():
    print("Building pool...", flush=True)
    careers = build_career_indexes(playoffs=False)
    df = build_rows(PAIRS, careers, {}, fetch_all_nba_selections()).reset_index(drop=True)
    Xw = make_X_augmented(df)            # REB_PCT is the last column
    Xwo = Xw[:, :-1]                     # drop REB_PCT
    sy = df["start_year"].values
    grade_ok = gradeable_mask(df).values
    assert ADV_COLS[-1] == "REB_PCT", f"expected REB_PCT last, got {ADV_COLS[-1]}"
    print(f"  {len(df)} rows · with {Xw.shape[1]} feats · without {Xwo.shape[1]} feats", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("TEMPORAL CV (recent seasons 2021-2025)", flush=True)
    print("=" * 70, flush=True)
    w5_w, w10_w, n = temporal(df, Xw, sy, grade_ok)
    w5_o, w10_o, _ = temporal(df, Xwo, sy, grade_ok)
    print(f"  WITH    REB_PCT:  within-5% {w5_w:.1f}%   within-10% {w10_w:.1f}%", flush=True)
    print(f"  WITHOUT REB_PCT:  within-5% {w5_o:.1f}%   within-10% {w10_o:.1f}%", flush=True)
    print(f"  Δ (without − with): {w5_o-w5_w:+.1f}pp w5, {w10_o-w10_w:+.1f}pp w10  (n={n})", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("PAIRED 5-FOLD CV (5 seeds × 5 folds = 25 folds)", flush=True)
    print("=" * 70, flush=True)
    md, se, wins, k = paired_5fold(df, Xw, Xwo, grade_ok)
    t = md / se if se else 0
    print(f"  within-5% delta (without − with): {md:+.2f}pp  (SE {se:.2f}, t={t:.2f})", flush=True)
    print(f"  without-REB wins {wins}/{k} folds", flush=True)

    # Big-man check: Jokić 23-24, AD 25-26.
    print("\n" + "=" * 70, flush=True)
    print("BIG-MAN CHECK (do the undershoots ease?)", flush=True)
    print("=" * 70, flush=True)
    for player, season in [("Nikola Jokić", "2023-24"), ("Anthony Davis", "2025-26"),
                            ("Giannis Antetokounmpo", "2021-22")]:
        yr = int(season.split("-")[0])
        trm = (sy >= TRAINING_START_YEAR) & (sy < yr)
        m = (df["player"].map(lambda v: _norm(str(v))) == _norm(player)) & (df["curr"] == season)
        if not m.any():
            continue
        i = np.where(m.values)[0][0]
        cap = SALARY_CAP_M[season] * 1e6
        rw = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(Xw[trm], df.loc[trm,"salary_curr_pct"].values)
        ro = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(Xwo[trm], df.loc[trm,"salary_curr_pct"].values)
        pw = float(apply_cba_postprocess(rw.predict(Xw[i:i+1]), df.iloc[[i]])[0]) * cap
        po = float(apply_cba_postprocess(ro.predict(Xwo[i:i+1]), df.iloc[[i]])[0]) * cap
        act = df.iloc[i]["salary_curr"]
        print(f"  {player:<24} {season}:  with ${pw/1e6:.1f}M → without ${po/1e6:.1f}M   "
              f"(actual ${act/1e6:.1f}M)", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)
    real = md > 0 and wins >= 0.6 * k and t > 1.5
    if real:
        print(f"  DROP REB_PCT — without is better (+{md:.2f}pp, {wins}/{k} folds, t={t:.1f}).", flush=True)
    elif abs(md) < 2 * se:
        print(f"  WASH — within noise (Δ {md:+.2f}pp, t={t:.1f}). Drop it for the cleaner")
        print(f"  attribution (removes the counterintuitive big-man penalty) at no cost.", flush=True)
    else:
        print(f"  KEEP REB_PCT — dropping it hurts ({md:+.2f}pp, t={t:.1f}).", flush=True)


if __name__ == "__main__":
    main()
