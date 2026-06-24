"""Validate the COMP-BLEND (the displayed projected_contract_value) vs the RAW
model. The page shows the raw HistGBM blended toward the median of comparable
signings (35-65% weight, only when raw prediction is $7-25M and they disagree
>25%). My accuracy work validated the RAW model; this checks whether the blend
users actually SEE helps or hurts.

Leakage-safe: in each temporal fold, the comp median for a test contract is the
inverse-distance-weighted median actual salary of the 6 nearest TRAIN contracts
(same position bucket, distance = |Barrett| + 1.5*|age|), exactly the
find_comparables + blend_toward_market mechanism, restricted to prior years.

Usage:  python -u scripts/exp_compblend.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
from build_production_histgbm import make_X_augmented, HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask, apply_cba_postprocess

LO_M, HI_M = 7.0, 25.0          # blend only in this raw-prediction tier (Contract_Predictor)


def wmedian(v, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def comp_median_pct(tb, ta, tbk, trb, tra, trbk, try_, K=6):
    """Inverse-distance weighted median salary_pct of the K nearest train comps
    (same pos bucket; distance = |Barrett|+1.5|age|)."""
    same = trbk == tbk
    if same.sum() < K:
        same = np.ones(len(trb), bool)              # fall back to all if a bucket is thin
    d = np.abs(trb[same] - tb) + 1.5 * np.abs(tra[same] - ta)
    idx = np.argsort(d)[:K]
    w = 1.0 / (d[idx] + 1.0)
    return wmedian(try_[same][idx], w)


def blend(raw_pct, mkt_pct, cap_M):
    raw_M = raw_pct * cap_M
    out = raw_pct.copy()
    for i in range(len(raw_pct)):
        if not (LO_M <= raw_M[i] <= HI_M):
            continue
        rM, mM = raw_M[i], mkt_pct[i] * cap_M[i]
        hi = max(rM, mM)
        gap = abs(rM - mM) / hi if hi > 0 else 0
        if gap <= 0.25 or mM <= 0:
            continue
        w = min(0.65, 0.35 + 0.30 * (gap - 0.25) / 0.35)
        out[i] = ((1 - w) * rM + w * mM) / cap_M[i]
    return out


def metrics(a, p, cap):
    return 100 * (np.abs(a - p) * 100 <= 5).mean(), 100 * (np.abs(a - p) * cap / 1e6 <= 4).mean()


def main():
    careers = build_career_indexes(playoffs=False); allnba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, allnba).reset_index(drop=True)
    X = make_X_augmented(df); sy = df["start_year"].values; grade = gradeable_mask(df).values
    y = df["salary_curr_pct"].values
    barr = df["barrett"].values; age = df["age"].values
    bk = df["pos_bucket"].astype(str).values

    RA, BL, AC, CAP, FIRED = [], [], [], [], []
    for ty in range(2016, 2026):
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty); tem = (sy == ty) & grade
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        m = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X[trm], y[trm])
        raw = np.clip(apply_cba_postprocess(m.predict(X[tem]), df[tem]), 0.001, 0.45)
        ti = np.where(tem)[0]
        mkt = np.array([comp_median_pct(barr[j], age[j], bk[j], barr[trm], age[trm], bk[trm], y[trm]) for j in ti])
        cap = df["cap_curr"].values[tem] / 1e6
        bl = blend(raw, mkt, cap)
        RA.append(raw); BL.append(bl); AC.append(y[tem]); CAP.append(df["cap_curr"].values[tem])
        FIRED.append(np.abs(bl - raw) > 1e-9)
    raw = np.concatenate(RA); bl = np.concatenate(BL); a = np.concatenate(AC); cap = np.concatenate(CAP)
    fired = np.concatenate(FIRED); actM = a * cap / 1e6

    print(f"\n{'':22s} {'within-5%cap':>13} {'within-$4M':>11}", flush=True)
    r5, r4 = metrics(a, raw, cap); b5, b4 = metrics(a, bl, cap)
    print(f"  {'RAW model':20s} {r5:12.2f}% {r4:10.2f}%", flush=True)
    print(f"  {'BLENDED (displayed)':20s} {b5:12.2f}% {b4:10.2f}%   Δ {b5-r5:+.2f}/{b4-r4:+.2f}pp", flush=True)
    print(f"\n  blend fired on {fired.sum()} of {len(raw)} contracts ({100*fired.mean():.0f}%)", flush=True)
    if fired.sum():
        fr5, fr4 = metrics(a[fired], raw[fired], cap[fired]); fb5, fb4 = metrics(a[fired], bl[fired], cap[fired])
        print(f"  ON THE FIRED SUBSET (where the page differs from the raw model):", flush=True)
        print(f"    RAW     within-5%cap {fr5:.1f}%  within-$4M {fr4:.1f}%", flush=True)
        print(f"    BLENDED within-5%cap {fb5:.1f}%  within-$4M {fb4:.1f}%   Δ {fb5-fr5:+.1f}/{fb4-fr4:+.1f}pp", flush=True)
        sM = (np.abs(a[fired]-bl[fired]) - np.abs(a[fired]-raw[fired])) * cap[fired]/1e6
        print(f"    mean |error| change from blending: {sM.mean():+.2f}M  (negative = blend helps)", flush=True)


if __name__ == "__main__":
    main()
