"""Refine the draft-pedigree feature: improve the name match (strip Jr/Sr/III
suffixes) and compare encodings to find the most ROBUST forward signal — or
confirm it's a wash. Encodings tested:
  - raw pick (1-60, undrafted/unmatched = 61)
  - draft tier ordinal (lottery=0 ... undrafted=4)
  - is_lottery (pick <= 14)
  - lottery pedigree curve: max(0, 15 - pick)  [emphasizes top picks, 0 otherwise]

Expanding-window temporal CV 2021-2025, fresh models per fold, eval gradeable.
Loads the cached pool.

Usage:
    python -u scripts/test_draft_encodings.py
"""
import sys, warnings, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from build_production_histgbm import (
    HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask, apply_cba_postprocess,
)
from utils import normalize, build_draft_tier_lookup

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv)\b\.?")


def _norm2(n):
    return _SUFFIX.sub("", normalize(str(n))).strip()


def main():
    df = pd.read_pickle("/tmp/pool_df.pkl").reset_index(drop=True)
    X = np.load("/tmp/pool_X.npy")
    sy = df["start_year"].values
    grade = gradeable_mask(df).values

    lookup = build_draft_tier_lookup()
    # second index keyed by suffix-stripped name to recover Jr/Sr mismatches
    lookup2 = {_norm2(k): v for k, v in lookup.items()}

    def pick_of(n):
        e = lookup.get(normalize(str(n))) or lookup2.get(_norm2(n))
        return (e or {}).get("draft_pick")

    raw_pick = df["player"].map(pick_of)
    matched = raw_pick.notna().mean() * 100
    print(f"draft match (with suffix recovery): {matched:.0f}%")
    pick = raw_pick.fillna(61).astype(float).values

    def tier_ord(p):
        if p <= 14: return 0
        if p <= 22: return 1
        if p <= 30: return 2
        if p <= 60: return 3
        return 4
    tier = np.array([tier_ord(p) for p in pick], dtype=float)
    is_lot = (pick <= 14).astype(float)
    curve = np.maximum(0.0, 15.0 - pick)

    encodings = {
        "raw pick":          pick,
        "tier ordinal":      tier,
        "is_lottery":        is_lot,
        "lottery curve":     curve,
    }

    # precompute folds
    folds = []
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        sub = df[tem]
        a = sub["salary_curr"].values / sub["cap_curr"].values
        y = df.loc[trm, "salary_curr_pct"].values
        folds.append((ty, trm, tem, sub, a, y))

    # baseline once
    base_w5 = {}
    for ty, trm, tem, sub, a, y in folds:
        m = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X[trm], y)
        p = apply_cba_postprocess(m.predict(X[tem]), sub)
        base_w5[ty] = np.mean(np.abs(p - a) * 100 <= 5) * 100

    print("\n" + "=" * 70)
    print("FORWARD CV — draft-feature encodings vs baseline (within-5%)")
    print("=" * 70)
    print(f"  {'encoding':<16}{'wt Δ5':>8}{'years up/down':>16}{'per-year Δ':>24}")
    for name, feat in encodings.items():
        Xa = np.hstack([X, feat.reshape(-1, 1)])
        deltas, ns = [], []
        for ty, trm, tem, sub, a, y in folds:
            m = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(Xa[trm], y)
            p = apply_cba_postprocess(m.predict(Xa[tem]), sub)
            w5 = np.mean(np.abs(p - a) * 100 <= 5) * 100
            deltas.append(w5 - base_w5[ty]); ns.append(len(sub))
        deltas = np.array(deltas); ns = np.array(ns)
        wm = np.average(deltas, weights=ns)
        up = int((deltas > 0.01).sum()); dn = int((deltas < -0.01).sum())
        per = " ".join(f"{d:+.1f}" for d in deltas)
        print(f"  {name:<16}{wm:>+7.2f}{f'{up}/{dn}':>16}{per:>24}")


if __name__ == "__main__":
    main()
