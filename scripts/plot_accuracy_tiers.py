"""Before/after accuracy by salary tier. Grades the FULL blended headline on
real signings 2020-2025 (actual >= $0.5M) under:
  BEFORE = trailing-Barrett comp match + vet-min EXCLUDED (threshold 0) + flat
           1.5% floor  (the shipped behavior before this session's fixes)
  AFTER  = current-form comp match + vet-min INCLUDED (<5) + service-min floor
Buckets by actual salary (today's $) and plots before vs after error-CDFs per
tier. Saves contract_accuracy_tiers.png.

Usage:  python -u scripts/plot_accuracy_tiers.py
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
from build_production_histgbm import (
    make_X_augmented, apply_cba_postprocess, gradeable_mask, HISTGBM_PARAMS, TRAINING_START_YEAR)

SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "p", "__file__": str(ROOT / "pages" / "Contract_Predictor.py")}
exec(compile("".join(SRC[:cut]), "p", "exec"), ns)
fc, lh, wm, idw = ns["find_comparables"], ns["load_historical_signings"], ns["_weighted_median"], ns["_inverse_distance_weights"]
min_salary_pct, SALCAP, normalize = ns["min_salary_pct"], ns["SALARY_CAP_M"], ns["normalize"]

CAPM = 165.0
TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def market_d(feats, hist, cap, vetmin):
    ns["VETMIN_COMP_TARGET_MAX"] = vetmin
    comps = fc(feats, hist, n=6)
    if comps.empty:
        return None
    w = idw(comps["distance"].astype(float).values) if "distance" in comps else np.ones(len(comps))
    return float(wm(comps["sal_pct"].astype(float).values, w)) * cap


def blend_d(model_d, mkt_d, floor_d, blend_floor):
    model = max(model_d, floor_d)
    if mkt_d is None:
        return model
    hi = max(model, mkt_d); gap = abs(model - mkt_d) / hi if hi > 0 else 0.0
    if gap > 0.25 and mkt_d > 0:
        w = min(0.65, 0.35 + 0.30 * (gap - 0.25) / 0.35)
        bl = (1 - w) * model + w * mkt_d
        if blend_floor:
            bl = max(bl, floor_d)
        if abs(bl - model) > 0.05e6:
            return bl
    return model


# model walk-forward prediction lookup: (norm name, year) -> (model_pct, svc)
careers = build_career_indexes(playoffs=False)
df = build_rows(PAIRS, careers, {}, fetch_all_nba_selections()).reset_index(drop=True)
Xm = make_X_augmented(df)
sy = df["start_year"].values; grade = gradeable_mask(df).values
mlut = {}
for ty in TEST_YEARS:
    trm = (sy >= TRAINING_START_YEAR) & (sy < ty); tem = (sy == ty) & grade
    if trm.sum() < 100 or tem.sum() < 5:
        continue
    reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(Xm[trm], df.loc[trm, "salary_curr_pct"].values)
    mp = apply_cba_postprocess(reg.predict(Xm[tem]), df[tem])
    sub = df[tem].reset_index(drop=True)
    for j in range(len(sub)):
        mlut[(normalize(sub.iloc[j]["player"]), int(sub.iloc[j]["start_year"]))] = (
            float(mp[j]), float(sub.iloc[j]["years_in_league"]))

pool = lh(n_recent_pairs=8).copy()
pool["year"] = pool["signed_in"].str[:4].astype(int)
pool["sal_pct"] = pool.apply(lambda r: float(r["salary_curr"]) / (SALCAP.get(r["signed_in"], 154.6) * 1e6), axis=1)

rows = []  # (actual_today_M, err_before_M, err_after_M)
for ty in TEST_YEARS:
    hist = pool[pool["year"] < ty]
    tg = pool[pool["year"] == ty]
    for _, r in tg.iterrows():
        key = (normalize(r["Player"]), ty)
        if key not in mlut:
            continue
        actual = float(r["salary_curr"])
        if actual < 0.5e6:
            continue
        model_pct, svc = mlut[key]
        cap = SALCAP.get(r["signed_in"], 154.6) * 1e6
        model_d = model_pct * cap
        floor_flat, floor_svc = 0.015 * cap, min_salary_pct(svc) * cap
        base = {"name": r["Player"], "position": r["pos"], "age": r["age"], "draft_tier": r.get("draft_tier", "Undrafted")}
        mb = market_d({**base, "barrett_score": float(r["career_weighted_barrett"])}, hist, cap, 0.0)
        ma = market_d({**base, "barrett_score": float(r["barrett_score"])}, hist, cap, 5.0)
        fb = blend_d(model_d, mb, floor_flat, False)
        fa = blend_d(max(model_d, floor_svc), ma, floor_svc, True)
        rows.append((actual / cap * CAPM, abs(fb - actual) / 1e6, abs(fa - actual) / 1e6))

rows = np.array(rows)
TIERS = [("Minimum (<$7M)", rows[:, 0] < 7),
         ("Rotation ($7–15M)", (rows[:, 0] >= 7) & (rows[:, 0] < 15)),
         ("Mid ($15–25M)", (rows[:, 0] >= 15) & (rows[:, 0] < 25)),
         ("Star / Max ($25M+)", rows[:, 0] >= 25)]
xs = np.linspace(0, 20, 400)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, (name, m) in zip(axes.ravel(), TIERS):
    eb = rows[m, 1]; ea = rows[m, 2]; n = int(m.sum())
    ax.plot(xs, [(eb <= x).mean() * 100 for x in xs], color="#c0392b", lw=2.4, ls="--", label=f"before  (med ${np.median(eb):.1f}M)")
    ax.plot(xs, [(ea <= x).mean() * 100 for x in xs], color="#1f6fb4", lw=2.8, label=f"after  (med ${np.median(ea):.1f}M)")
    ax.fill_between(xs, [(ea <= x).mean() * 100 for x in xs], color="#1f6fb4", alpha=0.10)
    ax.set_title(f"{name}   ·   n={n}", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 20); ax.set_ylim(0, 100); ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlabel("error  |headline − actual|  ($M)", fontsize=10)
    ax.set_ylabel("% within", fontsize=10)
    print(f"{name:<22} n={n:>4}  before median ${np.median(eb):.1f}M  ->  after ${np.median(ea):.1f}M", flush=True)
fig.suptitle("Contract Predictor accuracy — BEFORE vs AFTER fixes, by salary tier (2020–2025)", fontsize=15, fontweight="bold")
out = ROOT / "contract_accuracy_tiers.png"
fig.tight_layout(); fig.savefig(out, dpi=110)
print(f"\nsaved -> {out}", flush=True)
