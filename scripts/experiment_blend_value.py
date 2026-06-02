"""Does the market BLEND earn its place? Model-only vs full (model+blend) error,
BY salary tier, walk-forward OOS 2020-2025 (actual >= $0.5M), with today's
current-form comp fix in place. Shows exactly which tiers the blend helps vs
hurts -> informs drop / de-weight / keep (per tier).

Usage:  python -u scripts/experiment_blend_value.py
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
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


def blend_d(model_d, mkt_d, floor_d):
    model = max(model_d, floor_d)
    if mkt_d is None:
        return model
    hi = max(model, mkt_d); gap = abs(model - mkt_d) / hi if hi > 0 else 0.0
    if gap > 0.25 and mkt_d > 0:
        w = min(0.65, 0.35 + 0.30 * (gap - 0.25) / 0.35)
        bl = max((1 - w) * model + w * mkt_d, floor_d)
        if abs(bl - model) > 0.05e6:
            return bl
    return model


careers = build_career_indexes(playoffs=False)
df = build_rows(PAIRS, careers, {}, fetch_all_nba_selections()).reset_index(drop=True)
Xm = make_X_augmented(df); sy = df["start_year"].values; grade = gradeable_mask(df).values
pool = lh(n_recent_pairs=8).copy()
pool["year"] = pool["signed_in"].str[:4].astype(int)
pool["sal_pct"] = pool.apply(lambda r: float(r["salary_curr"]) / (SALCAP.get(r["signed_in"], 154.6) * 1e6), axis=1)
plut = {(normalize(r["Player"]), int(r["year"])): r for _, r in pool.iterrows()}

rows = []  # (actual_today_M, err_model, err_full, blended_flag, signed_full)
for ty in TEST_YEARS:
    trm = (sy >= TRAINING_START_YEAR) & (sy < ty); tem = (sy == ty) & grade
    if trm.sum() < 100 or tem.sum() < 5:
        continue
    reg = HistGradientBoostingRegressor(**HISTGBM_PARAMS).fit(Xm[trm], df.loc[trm, "salary_curr_pct"].values)
    mp = apply_cba_postprocess(reg.predict(Xm[tem]), df[tem])
    sub = df[tem].reset_index(drop=True)
    hist = pool[pool["year"] < ty]
    for j in range(len(sub)):
        r = sub.iloc[j]; actual = float(r["salary_curr"])
        if actual < 0.5e6:
            continue
        cap = float(r["cap_curr"]); floor_d = min_salary_pct(r["years_in_league"]) * cap
        model_d = max(float(mp[j]) * cap, floor_d)
        feats = {"name": r["player"], "position": r["pos_bucket"], "age": r["age"],
                 "draft_tier": "Undrafted", "barrett_score": float(r["barrett"])}
        comps = fc(feats, hist, n=6)
        if comps.empty:
            mkt_d = None
        else:
            w = idw(comps["distance"].astype(float).values) if "distance" in comps else np.ones(len(comps))
            mkt_d = float(wm(comps["sal_pct"].astype(float).values, w)) * cap
        full_d = blend_d(model_d, mkt_d, floor_d)
        model_today = float(mp[j]) * CAPM
        gated_d = full_d if 7.0 <= model_today <= 25.0 else max(model_d, floor_d)  # blend only in the middle
        rows.append((actual / cap * CAPM, abs(model_d - actual) / 1e6,
                     abs(full_d - actual) / 1e6, abs(gated_d - actual) / 1e6))

rows = np.array(rows)
TIERS = [("ALL", np.ones(len(rows), bool)),
         ("Minimum (<$7M)", rows[:, 0] < 7),
         ("Rotation ($7-15M)", (rows[:, 0] >= 7) & (rows[:, 0] < 15)),
         ("Mid ($15-25M)", (rows[:, 0] >= 15) & (rows[:, 0] < 25)),
         ("Star/Max ($25M+)", rows[:, 0] >= 25)]
print(f"{'tier':<20}{'n':>5}{'MODEL med':>11}{'FULL med':>10}{'GATED med':>11}{'  best':>8}", flush=True)
for name, m in TIERS:
    sub = rows[m]; n = len(sub)
    if n == 0:
        continue
    mm, mf, mg = np.median(sub[:, 1]), np.median(sub[:, 2]), np.median(sub[:, 3])
    best = ["MODEL", "FULL", "GATED"][int(np.argmin([mm, mf, mg]))]
    print(f"{name:<20}{n:>5}{mm:>10.1f}M{mf:>9.1f}M{mg:>10.1f}M{best:>8}", flush=True)
print("\nGATED = blend only when the model projects $7-25M (today's $); model alone elsewhere.", flush=True)
