"""List the ~2.5% of predictions that miss by >10% of cap (the tail).

Uses the SHIPPED model (2012+, Barrett + advanced) with expanding-window
temporal CV on recent seasons (2021-2025): train only on prior seasons,
predict each season, then show every contract the model misses by more than
10% of that season's cap. Characterizes overshoot (model too high — paycut /
ring chase) vs undershoot (model too low — breakout / surprise raise).

Usage:
    python -u scripts/list_big_misses.py
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from train_ml_model_v2 import build_career_indexes, build_rows, fetch_all_nba_selections, PAIRS
from build_production_histgbm import make_X_augmented, HISTGBM_PARAMS, TRAINING_START_YEAR

TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def main():
    print("Building 2012+ pool...", flush=True)
    t0 = time.time()
    careers = build_career_indexes(playoffs=False)
    all_nba = fetch_all_nba_selections()
    df = build_rows(PAIRS, careers, {}, all_nba).reset_index(drop=True)
    X = make_X_augmented(df)
    sy = df["start_year"].values
    print(f"  {len(df)} contracts in {time.time()-t0:.1f}s", flush=True)

    misses = []
    n_total = 0
    for ty in TEST_YEARS:
        trm = (sy >= TRAINING_START_YEAR) & (sy < ty)
        tem = (sy == ty)
        if trm.sum() < 100 or tem.sum() < 5:
            continue
        reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(
            X[trm], df.loc[trm, "salary_curr_pct"].values)
        sub = df[tem].copy()
        cap = sub["cap_curr"].values
        pred = np.clip(reg.predict(X[tem]), 0.001, 0.45) * cap
        actual = sub["salary_curr"].values
        err_pct = (actual - pred) / cap * 100          # signed: + = model too LOW
        n_total += len(sub)
        for i, (_, r) in enumerate(sub.iterrows()):
            if abs(err_pct[i]) > 10:
                misses.append({
                    "player": r["player"], "season": r["curr"],
                    "pred_M": pred[i] / 1e6, "actual_M": actual[i] / 1e6,
                    "prev_M": r["salary_prev"] / 1e6,
                    "err_pct": err_pct[i],
                    "age": r.get("age"), "all_nba": r.get("all_nba_3yr", 0),
                })

    print("\n" + "=" * 92, flush=True)
    print(f"BIG MISSES (>10% of cap)  —  {len(misses)} of {n_total} predictions "
          f"({len(misses)/n_total*100:.1f}%)", flush=True)
    print("=" * 92, flush=True)

    over = sorted([m for m in misses if m["err_pct"] < 0], key=lambda m: m["err_pct"])
    under = sorted([m for m in misses if m["err_pct"] > 0], key=lambda m: -m["err_pct"])

    def show(group, title):
        print(f"\n{title}", flush=True)
        print(f"  {'Player':<24}{'Season':<9}{'Pred':>7}{'Actual':>8}{'Prev':>7}"
              f"{'Err':>7}  context", flush=True)
        for m in group:
            age = f"{m['age']:.0f}" if m['age'] is not None and not np.isnan(m['age']) else "?"
            ctx = []
            if m["actual_M"] < m["prev_M"] * 0.7:
                ctx.append("PAYCUT")
            if m["all_nba"] >= 1:
                ctx.append(f"{int(m['all_nba'])}x All-NBA")
            ctx.append(f"age {age}")
            print(f"  {m['player'][:23]:<24}{m['season']:<9}"
                  f"{m['pred_M']:>6.1f}M{m['actual_M']:>7.1f}M{m['prev_M']:>6.1f}M"
                  f"{m['err_pct']:>+6.0f}%  {', '.join(ctx)}", flush=True)

    show(over, "OVERSHOOTS — model too HIGH (actual came in well below prediction):")
    show(under, "UNDERSHOOTS — model too LOW (player got paid well above prediction):")


if __name__ == "__main__":
    main()
