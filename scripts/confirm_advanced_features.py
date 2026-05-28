"""Rigorously confirm whether ADDING advanced stats (USG/PIE/NET/TS/AST%/REB%)
to the production feature set is a real accuracy gain or noise.

Method: PAIRED repeated cross-validation. For 5 seeds × 5 folds = 25 splits,
train the current-features model AND the +advanced model on the EXACT SAME
train fold, evaluate on the same test fold. Compare per-fold (paired), which
cancels fold-to-fold difficulty and is far more sensitive than comparing two
independent ±SE numbers.

If +advanced wins a clear majority of the 25 paired folds with a positive
mean delta, it's a real improvement → integrate into production.

Usage:
    python -u scripts/confirm_advanced_features.py
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
from sklearn.model_selection import KFold

from train_ml_model_v2 import (
    build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS,
)
from train_ml_model_v3 import make_X_pruned
from train_raw_model import combined_season
from utils import normalize as _norm

HP = dict(max_iter=800, max_depth=5, learning_rate=0.02,
          min_samples_leaf=25, l2_regularization=0.1)
TRAINING_START_YEAR = 2012
ADV_COLS = ["USG_PCT", "PIE", "NET_RATING", "TS_PCT", "AST_PCT", "REB_PCT"]


def build_augmented(df):
    """Return advanced-stat columns aligned to df rows (player+prev season)."""
    adv_lookup = {}
    for prev in df["prev"].unique():
        cs = combined_season(prev)
        if cs.empty:
            continue
        for _, r in cs.iterrows():
            key = (_norm(str(r.get("PLAYER_NAME", ""))), prev)
            adv_lookup[key] = {c: float(r.get(c, 0) or 0) for c in ADV_COLS}
    rows = [adv_lookup.get((_norm(str(p)), pv), {c: 0.0 for c in ADV_COLS})
            for p, pv in zip(df["player"], df["prev"])]
    return pd.DataFrame(rows)[ADV_COLS].fillna(0).values


def fold_w5_w10(Xtr, ytr, Xte, cap, actual):
    reg = HistGradientBoostingRegressor(random_state=42, **HP).fit(Xtr, ytr)
    pred = np.clip(reg.predict(Xte), 0.001, 0.45) * cap
    e = np.abs(actual - pred) / cap * 100
    return float(np.mean(e <= 5) * 100), float(np.mean(e <= 10) * 100)


def main():
    print("Building 2012+ pool + advanced features...", flush=True)
    t0 = time.time()
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba)
    df = df[df["start_year"] >= TRAINING_START_YEAR].reset_index(drop=True)
    X_base = make_X_pruned(df)
    X_aug = np.hstack([X_base, build_augmented(df)])
    y = df["salary_curr_pct"].values
    cap = df["cap_curr"].values
    actual = df["salary_curr"].values
    print(f"  {len(df)} contracts · base {X_base.shape[1]} feats · "
          f"aug {X_aug.shape[1]} feats · {time.time()-t0:.1f}s", flush=True)

    print("\nPaired repeated CV (5 seeds × 5 folds = 25 paired folds)...", flush=True)
    d5, d10, base5s, aug5s = [], [], [], []
    for seed in range(1, 6):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tri, tei in kf.split(X_base):
            b5, b10 = fold_w5_w10(X_base[tri], y[tri], X_base[tei], cap[tei], actual[tei])
            a5, a10 = fold_w5_w10(X_aug[tri], y[tri], X_aug[tei], cap[tei], actual[tei])
            d5.append(a5 - b5); d10.append(a10 - b10)
            base5s.append(b5); aug5s.append(a5)
    d5 = np.array(d5); d10 = np.array(d10)

    wins5 = int((d5 > 0).sum()); ties5 = int((d5 == 0).sum())
    wins10 = int((d10 > 0).sum())
    mean_d5, se_d5 = float(d5.mean()), float(d5.std() / np.sqrt(len(d5)))
    t_stat = mean_d5 / se_d5 if se_d5 > 0 else 0.0

    print("\n" + "=" * 70, flush=True)
    print("RESULT — does +advanced stats beat current features?", flush=True)
    print("=" * 70, flush=True)
    print(f"  current features:  within-5% mean {np.mean(base5s):.2f}%", flush=True)
    print(f"  + advanced stats:  within-5% mean {np.mean(aug5s):.2f}%", flush=True)
    print(f"\n  within-5%  delta: {mean_d5:+.2f}pp  (paired SE {se_d5:.2f}, t={t_stat:.2f})", flush=True)
    print(f"  within-10% delta: {d10.mean():+.2f}pp", flush=True)
    print(f"  +advanced wins {wins5}/{len(d5)} folds on within-5%, "
          f"{wins10}/{len(d10)} on within-10%", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("DECISION", flush=True)
    print("=" * 70, flush=True)
    # Real if: positive mean, wins clear majority, t-stat > 2 (≈95% paired).
    real = mean_d5 > 0 and wins5 >= 0.7 * len(d5) and t_stat > 2.0
    if real:
        print(f"  SHIP +advanced stats: +{mean_d5:.2f}pp within-5%, wins "
              f"{wins5}/{len(d5)} folds, t={t_stat:.1f} (statistically real).", flush=True)
    else:
        print(f"  Gain is NOT robust enough to ship (wins {wins5}/{len(d5)}, "
              f"t={t_stat:.1f}). Need wins ≥{int(0.7*len(d5))} and t>2.", flush=True)


if __name__ == "__main__":
    main()
