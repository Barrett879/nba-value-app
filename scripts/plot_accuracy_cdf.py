"""Regenerate the Contract Predictor incremental-$ accuracy CDF:
% of real signings predicted within a given $ error, walk-forward OOS
2020-2025, real contracts only (actual >= $0.5M). Saves contract_accuracy_cdf.png.

Usage:  python -u scripts/plot_accuracy_cdf.py
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

CAPM = 165.0
TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

careers = build_career_indexes(playoffs=False)
df = build_rows(PAIRS, careers, {}, fetch_all_nba_selections()).reset_index(drop=True)
X = make_X_augmented(df)
sy = df["start_year"].values
grade = gradeable_mask(df).values

errs = []
for ty in TEST_YEARS:
    trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
    tem = (sy == ty) & grade
    if trm.sum() < 100 or tem.sum() < 5:
        continue
    reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(X[trm], df.loc[trm, "salary_curr_pct"].values)
    sub = df[tem]; cap = sub["cap_curr"].values
    pred = apply_cba_postprocess(reg.predict(X[tem]), sub) * cap
    actual = sub["salary_curr"].values
    keep = actual >= 0.5e6                                   # real contracts only
    errs.extend((np.abs(pred - actual)[keep] / 1e6).tolist())  # nominal signing-year $M

errs = np.sort(np.array(errs)); n = len(errs)
p50, p80, p90 = (np.percentile(errs, q) for q in (50, 80, 90))
w10 = (errs <= 10).mean() * 100
xs = np.linspace(0, 20, 500)
cdf = np.array([(errs <= x).mean() * 100 for x in xs])
print(f"n={n}  median=${p50:.1f}M  80%<=${p80:.1f}M  90%<=${p90:.1f}M  96%? {(errs<=10).mean()*100:.0f}% within $10M", flush=True)

fig, ax = plt.subplots(figsize=(13, 8))
ax.plot(xs, cdf, color="#1f6fb4", lw=3, zorder=3)
ax.fill_between(xs, cdf, color="#1f6fb4", alpha=0.12, zorder=1)
marks = [(p50, 50, "#e8743b", f"50% within ${p50:.1f}M"),
         (p80, 80, "#2ca02c", f"80% within ${p80:.1f}M"),
         (p90, 90, "#7b5ea7", f"90% within ${p90:.1f}M")]
for x, y, c, lab in marks:
    ax.plot([x, x], [0, y], ls="--", color=c, lw=1.5, zorder=2)
    ax.plot([0, x], [y, y], ls="--", color=c, lw=1.2, alpha=0.7, zorder=2)
    ax.plot(x, y, "o", color=c, ms=10, zorder=4)
    ax.annotate(lab, (x, y), xytext=(x + 0.4, y - 9), color=c, fontsize=13, fontweight="bold")
ax.set_xlim(0, 20); ax.set_ylim(0, 100)
ax.set_xlabel("Prediction error  |predicted − actual|  ($M)", fontsize=13)
ax.set_ylabel("% of contracts predicted within", fontsize=13)
ax.set_title(f"Contract Predictor — incremental-$ accuracy (2020–2025, n={n})", fontsize=16, fontweight="bold")
ax.grid(True, alpha=0.25)
ax.text(0.985, 0.04, f"median error  ${p50:.1f}M\n{w10:.0f}% within $10M",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
        bbox=dict(boxstyle="round", fc="#eef4fb", ec="#9bbad6"))
out = ROOT / "contract_accuracy_cdf.png"
fig.tight_layout(); fig.savefig(out, dpi=110)
print(f"saved -> {out}", flush=True)
