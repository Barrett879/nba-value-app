"""Squeeze maximum accuracy: CV-based hyperparameter search + feature test.

Pure accuracy push. Everything scored by 5-fold cross-validation on the
2012+ pool (mean within-5% of cap across folds), so any gain is real and not
test-set overfitting.

  Part A: random-search HistGBM hyperparameters (40 configs) on the current
          production feature set.
  Part B: test whether ADDING advanced stats (USG/PIE/NET_RATING/TS) to the
          feature set helps under CV.

Ships whatever genuinely beats the current config.

Usage:
    python -u scripts/hp_optimize.py
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
from train_ml_model_v3 import make_X_pruned, PRUNED_FEATURES

# Current production config.
CURRENT_HP = dict(max_iter=800, max_depth=5, learning_rate=0.02,
                  min_samples_leaf=25, l2_regularization=0.1)
TRAINING_START_YEAR = 2012
RNG = np.random.RandomState(12)


def cv_score(X, df, hp, n_splits=5):
    """5-fold CV mean within-5% and within-10% of cap."""
    y = df["salary_curr_pct"].values
    cap = df["cap_curr"].values
    actual = df["salary_curr"].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    w5s, w10s = [], []
    for tri, tei in kf.split(X):
        reg = HistGradientBoostingRegressor(random_state=42, **hp).fit(X[tri], y[tri])
        pred = np.clip(reg.predict(X[tei]), 0.001, 0.45) * cap[tei]
        e = np.abs(actual[tei] - pred) / cap[tei] * 100
        w5s.append(np.mean(e <= 5) * 100)
        w10s.append(np.mean(e <= 10) * 100)
    return float(np.mean(w5s)), float(np.std(w5s)), float(np.mean(w10s))


def sample_hp():
    return dict(
        learning_rate=float(10 ** RNG.uniform(-2.0, -1.0)),       # 0.01-0.1
        max_iter=int(RNG.choice([400, 600, 800, 1000, 1200, 1500])),
        max_leaf_nodes=int(RNG.choice([15, 21, 31, 47, 63])),
        min_samples_leaf=int(RNG.choice([10, 15, 20, 25, 30, 40])),
        l2_regularization=float(RNG.choice([0.0, 0.1, 0.2, 0.5, 1.0])),
        max_features=float(RNG.choice([0.6, 0.8, 1.0])),
    )


def main():
    print("Building 2012+ Barrett pool...", flush=True)
    t0 = time.time()
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba)
    df = df[df["start_year"] >= TRAINING_START_YEAR].reset_index(drop=True)
    X = make_X_pruned(df)
    print(f"  {len(df)} contracts, {X.shape[1]} features, in {time.time()-t0:.1f}s", flush=True)

    # Baseline.
    b5, b5sd, b10 = cv_score(X, df, CURRENT_HP)
    print("\n" + "=" * 78, flush=True)
    print("PART A — hyperparameter search (5-fold CV on 2012+)", flush=True)
    print("=" * 78, flush=True)
    print(f"  CURRENT config:  within-5% {b5:.2f}% ± {b5sd:.2f}   within-10% {b10:.2f}%", flush=True)
    print("  Searching 40 configs...", flush=True)

    results = [("CURRENT", CURRENT_HP, b5, b5sd, b10)]
    best = ("CURRENT", CURRENT_HP, b5, b5sd, b10)
    for i in range(40):
        hp = sample_hp()
        w5, sd, w10 = cv_score(X, df, hp)
        results.append((f"cfg{i}", hp, w5, sd, w10))
        if w5 > best[2]:
            best = (f"cfg{i}", hp, w5, sd, w10)
            print(f"    [{i:02d}] new best within-5% {w5:.2f}% ± {sd:.2f}  (w10 {w10:.2f}%)", flush=True)

    print(f"\n  Top 5 configs by CV within-5%:", flush=True)
    for name, hp, w5, sd, w10 in sorted(results, key=lambda r: -r[2])[:5]:
        tag = " (current)" if name == "CURRENT" else ""
        print(f"    {w5:.2f}% ± {sd:.2f}  w10 {w10:.2f}%  {name}{tag}", flush=True)

    gain = best[2] - b5
    print(f"\n  Best: {best[0]}  within-5% {best[2]:.2f}% vs current {b5:.2f}%  ({gain:+.2f}pp)", flush=True)
    # Is the gain bigger than the noise band? (averaged-fold SE ~ sd/sqrt(5))
    se = b5sd / np.sqrt(5)
    real = gain > 2 * se
    print(f"  Noise band (2·SE) ≈ ±{2*se:.2f}pp → gain is "
          f"{'REAL' if real else 'within noise'}", flush=True)
    if not best[0] == "CURRENT":
        print(f"  Best config: {best[1]}", flush=True)

    # ── Part B: does adding advanced stats help? ─────────────────────────────
    print("\n" + "=" * 78, flush=True)
    print("PART B — add advanced stats (USG/PIE/NET/TS) to features?", flush=True)
    print("=" * 78, flush=True)
    try:
        from train_raw_model import combined_season
        from utils import normalize as _norm
        # Merge advanced stats for each row by matching player+prev season.
        adv_cols = ["USG_PCT", "PIE", "NET_RATING", "TS_PCT", "AST_PCT", "REB_PCT"]
        # Build a (normalized name, prev season) -> adv dict from combined tables.
        adv_lookup = {}
        for prev in df["prev"].unique():
            cs = combined_season(prev)
            if cs.empty:
                continue
            for _, r in cs.iterrows():
                key = (_norm(str(r.get("PLAYER_NAME", ""))), prev)
                adv_lookup[key] = {c: float(r.get(c, 0) or 0) for c in adv_cols}
        add = pd.DataFrame([
            adv_lookup.get((_norm(str(p)), pv), {c: 0.0 for c in adv_cols})
            for p, pv in zip(df["player"], df["prev"])
        ])
        X_aug = np.hstack([X, add[adv_cols].fillna(0).values])
        print(f"  Augmented feature count: {X_aug.shape[1]} (+{len(adv_cols)} advanced)", flush=True)
        a5, a5sd, a10 = cv_score(X_aug, df, best[1])
        print(f"  With advanced stats: within-5% {a5:.2f}% ± {a5sd:.2f}  within-10% {a10:.2f}%", flush=True)
        print(f"  vs best base config ({best[2]:.2f}%): {a5-best[2]:+.2f}pp "
              f"({'REAL' if a5-best[2] > 2*se else 'within noise'})", flush=True)
    except Exception as e:
        print(f"  (advanced-stats test skipped: {e})", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("DECISION", flush=True)
    print("=" * 78, flush=True)
    if real:
        print(f"  SHIP new HP config (+{gain:.2f}pp, beats noise band).", flush=True)
        print(f"  {best[1]}", flush=True)
    else:
        print(f"  KEEP current config — no HP config beats it outside noise.", flush=True)
        print(f"  We're at the tuning ceiling; accuracy is feature/data-bound, not HP-bound.", flush=True)


if __name__ == "__main__":
    main()
