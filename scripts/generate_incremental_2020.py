"""Single polished incremental-$ accuracy curve for the 2020-2025 window
(n=668) — the analyst-facing chart. % of contract predictions whose dollar
error |pred - actual| falls within a given threshold, swept from $0 up.

Loads the cached pool. Saves /tmp/incremental_2020_2025.png

Usage:
    python -u scripts/generate_incremental_2020.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

from build_production_histgbm import (
    HISTGBM_PARAMS, TRAINING_START_YEAR, gradeable_mask, apply_cba_postprocess,
)

OUT = "/tmp/incremental_2020_2025.png"
THRESH = np.arange(0.0, 20.01, 0.25)


def main():
    df = pd.read_pickle("/tmp/pool_df.pkl").reset_index(drop=True)
    X = np.load("/tmp/pool_X.npy")
    sy = df["start_year"].values
    grade = gradeable_mask(df).values

    errs = []
    for ty in range(2020, 2026):
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty) & grade
        m = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem]
        pred = apply_cba_postprocess(m.predict(X[tem]), sub) * sub["cap_curr"].values
        errs.extend(np.abs(pred - sub["salary_curr"].values) / 1e6)
    e = np.array(errs)
    cum = np.array([(e <= t).mean() * 100 for t in THRESH])
    med = np.median(e)

    def dollar_at(pct):
        idx = np.argmax(cum >= pct)
        return THRESH[idx] if cum[idx] >= pct else None

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(THRESH, cum, color="#2c7fb8", alpha=0.18)
    ax.plot(THRESH, cum, color="#1f6fae", lw=3)

    for pct, c in [(50, "#e6550d"), (80, "#31a354"), (90, "#6a51a3")]:
        d = dollar_at(pct)
        if d is None:
            continue
        ax.plot([d, d], [0, pct], color=c, ls="--", lw=1.4)
        ax.plot([0, d], [pct, pct], color=c, ls="--", lw=1.0, alpha=0.5)
        ax.scatter([d], [pct], color=c, s=45, zorder=5)
        ax.annotate(f"{pct}% within \\${d:.1f}M", xy=(d, pct),
                    xytext=(d + 0.5, pct - 7.5), fontsize=11, color=c, weight="bold")

    ax.set_title("Contract Predictor — incremental-$ accuracy (2020-2025, n=668)",
                 fontsize=13.5, weight="bold")
    ax.set_xlabel("Prediction error  |predicted - actual|  (\\$M)", fontsize=11)
    ax.set_ylabel("% of contracts predicted within", fontsize=11)
    ax.set_xlim(0, 20); ax.set_ylim(0, 101)
    ax.grid(alpha=0.25)
    ax.text(0.985, 0.06, f"median error  \\${med:.1f}M\n96% within \\$10M",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", fc="#f0f4f8", ec="#a8c3da"))
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"saved {OUT}")
    print(f"median ${med:.2f}M")
    for t in [0.5, 1, 2, 3, 5, 7.5, 10]:
        print(f"  within ${t}M: {(e<=t).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
