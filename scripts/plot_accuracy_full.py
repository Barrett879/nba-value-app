"""Updated (END-TO-END) incremental-$ accuracy: grade the FINAL blended headline
the page shows (model -> comps -> market blend, with today's current-form match
fix + service-min floor) against actual signings, walk-forward OOS 2020-2025,
real contracts only (actual >= $0.5M). Overlays the model-only line for
reference. Saves contract_accuracy_full.png.

Usage:  python -u scripts/plot_accuracy_full.py
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

# comp/blend machinery from the live page
SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("selected = st_searchbox("))
ns = {"__name__": "p", "__file__": str(ROOT / "pages" / "Contract_Predictor.py")}
exec(compile("".join(SRC[:cut]), "p", "exec"), ns)
fc, lh, wm, idw = ns["find_comparables"], ns["load_historical_signings"], ns["_weighted_median"], ns["_inverse_distance_weights"]
min_salary_pct, SALCAP = ns["min_salary_pct"], ns["SALARY_CAP_M"]

TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def blend_d(model_d, market_d, min_d):
    model = max(model_d, min_d)
    if market_d is None:
        return model
    hi = max(model, market_d); gap = abs(model - market_d) / hi if hi > 0 else 0.0
    if gap > 0.25 and market_d > 0:
        w = min(0.65, 0.35 + 0.30 * (gap - 0.25) / 0.35)
        bl = max((1 - w) * model + w * market_d, min_d)
        if abs(bl - model) > 0.05e6:
            return bl
    return model


careers = build_career_indexes(playoffs=False)
df = build_rows(PAIRS, careers, {}, fetch_all_nba_selections()).reset_index(drop=True)
Xm = make_X_augmented(df)
sy = df["start_year"].values; grade = gradeable_mask(df).values
pool = lh(n_recent_pairs=8).copy()
pool["year"] = pool["signed_in"].str[:4].astype(int)
pool["sal_pct"] = pool.apply(lambda r: float(r["salary_curr"]) / (SALCAP.get(r["signed_in"], 154.6) * 1e6), axis=1)

err_full, err_model, empty = [], [], 0
for ty in TEST_YEARS:
    trm = (sy >= TRAINING_START_YEAR) & (sy < ty); tem = (sy == ty) & grade
    if trm.sum() < 100 or tem.sum() < 5:
        continue
    reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(Xm[trm], df.loc[trm, "salary_curr_pct"].values)
    model_pct = apply_cba_postprocess(reg.predict(Xm[tem]), df[tem])
    sub = df[tem].reset_index(drop=True)
    hist = pool[pool["year"] < ty]
    for j in range(len(sub)):
        r = sub.iloc[j]
        actual = float(r["salary_curr"])
        if actual < 0.5e6:
            continue
        cap = float(r["cap_curr"]); min_d = min_salary_pct(r["years_in_league"]) * cap
        model_d = max(float(model_pct[j]) * cap, min_d)
        feats = {"name": r["player"], "position": r["pos_bucket"], "age": r["age"],
                 "barrett_score": float(r["barrett"]), "draft_tier": "Undrafted",
                 "trailing_barrett": float(r["barrett"])}
        comps = fc(feats, hist, n=6)
        if comps.empty:
            market_d = None; empty += 1
        else:
            w = idw(comps["distance"].astype(float).values) if "distance" in comps else np.ones(len(comps))
            market_d = float(wm(comps["sal_pct"].astype(float).values, w)) * cap
        final_d = blend_d(model_d, market_d, min_d)
        err_full.append(abs(final_d - actual) / 1e6)
        err_model.append(abs(model_d - actual) / 1e6)

ef = np.sort(np.array(err_full)); em = np.sort(np.array(err_model)); n = len(ef)
p50, p80, p90 = (np.percentile(ef, q) for q in (50, 80, 90))
w10 = (ef <= 10).mean() * 100
print(f"FULL  n={n} median=${p50:.1f}M 80%<=${p80:.1f}M 90%<=${p90:.1f}M {w10:.0f}% w/in $10M  (comps empty for {empty})", flush=True)
print(f"MODEL n={len(em)} median=${np.percentile(em,50):.1f}M 90%<=${np.percentile(em,90):.1f}M", flush=True)

xs = np.linspace(0, 20, 500)
cdf = np.array([(ef <= x).mean() * 100 for x in xs])
cdf_m = np.array([(em <= x).mean() * 100 for x in xs])
fig, ax = plt.subplots(figsize=(13, 8))
ax.plot(xs, cdf_m, color="#9aa0aa", lw=2, ls="--", zorder=2, label="model only")
ax.plot(xs, cdf, color="#1f6fb4", lw=3, zorder=3, label="full pipeline (model + comps + blend)")
ax.fill_between(xs, cdf, color="#1f6fb4", alpha=0.12, zorder=1)
for x, y, c in [(p50, 50, "#e8743b"), (p80, 80, "#2ca02c"), (p90, 90, "#7b5ea7")]:
    ax.plot([x, x], [0, y], ls="--", color=c, lw=1.5, zorder=2)
    ax.plot([0, x], [y, y], ls="--", color=c, lw=1.2, alpha=0.7, zorder=2)
    ax.plot(x, y, "o", color=c, ms=10, zorder=4)
    ax.annotate(f"{y}% within ${x:.1f}M", (x, y), xytext=(x + 0.4, y - 9), color=c, fontsize=13, fontweight="bold")
ax.set_xlim(0, 20); ax.set_ylim(0, 100)
ax.set_xlabel("Prediction error  |final headline − actual|  ($M)", fontsize=13)
ax.set_ylabel("% of contracts predicted within", fontsize=13)
ax.set_title(f"Contract Predictor — incremental-$ accuracy, FULL pipeline (2020–2025, n={n})", fontsize=15, fontweight="bold")
ax.grid(True, alpha=0.25); ax.legend(loc="center right", fontsize=12)
ax.text(0.985, 0.04, f"median error  ${p50:.1f}M\n{w10:.0f}% within $10M",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
        bbox=dict(boxstyle="round", fc="#eef4fb", ec="#9bbad6"))
out = ROOT / "contract_accuracy_full.png"
fig.tight_layout(); fig.savefig(out, dpi=110)
print(f"saved -> {out}", flush=True)
