"""Plot the dollar-tolerance confidence curve for the SHIPPED Barrett model.

"If I predict a player's next contract, how confident am I in landing within
$X of their actual signing?" — answered at $0.5M increments.

Uses honest out-of-sample predictions from expanding-window temporal CV
(each contract predicted by a model trained only on PRIOR seasons), so the
curve reflects true forecasting performance, not in-sample optimism. Errors
are expressed in 2025-26 cap-equivalent dollars (a $5M miss in 2005 is
normalized to its 2025-26 purchasing power).

Outputs: /tmp/dollar_confidence.png

Usage:
    python -u scripts/plot_dollar_confidence.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

from train_ml_model_v2 import (
    build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS,
)
from train_ml_model_v3 import make_X_pruned

CURRENT_CAP_M = 154.6
REG_HP = dict(max_iter=800, max_depth=5, learning_rate=0.02,
              min_samples_leaf=25, l2_regularization=0.1)


def main():
    print("Building Barrett rows...", flush=True)
    t0 = time.time()
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    X = make_X_pruned(df)
    print(f"  {len(df)} contracts in {time.time()-t0:.1f}s", flush=True)

    # Out-of-sample predictions via expanding-window temporal CV, matching the
    # SHIPPED model: train only on the modern era (2012+), test recent seasons.
    TRAINING_START_YEAR = 2012
    print("Collecting out-of-sample predictions (temporal CV, 2012+ window)...", flush=True)
    err_today_M = []
    for ty in sorted(df["start_year"].unique()):
        if ty < 2021:  # recent seasons — the model's actual use case
            continue
        trm = ((df["start_year"] >= TRAINING_START_YEAR) & (df["start_year"] < ty)).values
        tem = (df["start_year"] == ty).values
        if trm.sum() < 200 or tem.sum() < 5:
            continue
        reg = HistGradientBoostingRegressor(random_state=42, **REG_HP).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        cap = df.loc[tem, "cap_curr"].values
        pred = np.clip(reg.predict(X[tem]), 0.001, 0.45) * cap
        actual = df.loc[tem, "salary_curr"].values
        e_pct_cap = np.abs(actual - pred) / cap            # error as fraction of cap
        err_today_M.extend((e_pct_cap * CURRENT_CAP_M).tolist())
    err = np.array(err_today_M)
    n = len(err)
    print(f"  {n} out-of-sample predictions", flush=True)

    # Cumulative confidence at $0.5M increments.
    thresholds = np.arange(0.5, 20.01, 0.5)
    conf = np.array([(err <= t).mean() * 100 for t in thresholds])
    median = float(np.median(err))

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117",
        "savefig.facecolor": "#0e1117", "text.color": "#e6e6ea",
        "axes.labelcolor": "#e6e6ea", "xtick.color": "#b9b9c3",
        "ytick.color": "#b9b9c3", "axes.edgecolor": "#3a3a45",
        "font.size": 11,
    })
    fig, ax = plt.subplots(figsize=(11, 6.2))

    teal = "#16d4c1"
    ax.plot(thresholds, conf, color=teal, lw=2.6, zorder=3)
    ax.fill_between(thresholds, conf, color=teal, alpha=0.10, zorder=1)
    ax.scatter(thresholds, conf, color=teal, s=16, zorder=4)

    # Reference markers.
    def mark(x, label, color="#f5a623"):
        y = float(np.interp(x, thresholds, conf))
        ax.axvline(x, color=color, ls="--", lw=1.0, alpha=0.55, zorder=2)
        ax.scatter([x], [y], color=color, s=70, zorder=5, edgecolor="#0e1117", linewidth=1)
        ax.annotate(f"{label}\n{y:.0f}%", xy=(x, y), xytext=(x + 0.5, y - 9),
                    color=color, fontsize=10, fontweight="bold")

    mark(5.0, "±$5M")
    mark(8.0, "±$8M (5% of cap)")
    # Median line (horizontal at 50%).
    ax.axhline(50, color="#8a8a96", ls=":", lw=1.0, alpha=0.5)
    ax.annotate(f"median miss ≈ ${median:.1f}M  (50% land closer than this)",
                xy=(median, 50), xytext=(median + 0.4, 53.5),
                color="#cfcfd6", fontsize=9.5)
    ax.scatter([median], [50], color="#ffffff", s=45, zorder=6,
               edgecolor="#0e1117", linewidth=1)

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 21, 2))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xlabel("Dollar tolerance  —  prediction within ±$X of actual signing "
                  "(2025-26 cap-equivalent)", fontsize=11.5)
    ax.set_ylabel("Confidence  —  % of contracts within tolerance", fontsize=11.5)
    ax.set_title("Contract Prediction Confidence by Dollar Tolerance",
                 fontsize=14.5, fontweight="bold", pad=14, color="#ffffff")
    ax.text(0.0, 1.012,
            f"Shipped HistGBM (Barrett, 2012+ era) · {n:,} out-of-sample "
            f"predictions · temporal CV on recent seasons 2021-2025",
            transform=ax.transAxes, fontsize=9.5, color="#9a9aa6")
    ax.grid(True, alpha=0.12, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    out = "/tmp/dollar_confidence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}", flush=True)

    # Print the table too.
    print("\n  ±$ tolerance   confidence", flush=True)
    for t in [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        c = float(np.interp(t, thresholds, conf))
        print(f"   ±${t:<5.1f}M     {c:5.1f}%", flush=True)
    print(f"\n  median |error| ≈ ${median:.2f}M", flush=True)


if __name__ == "__main__":
    main()
