"""Which comp-match score is most accurate: trailing (prime), current form,
a blend, or the lower of the two? Leak-free walk-forward on real signings
2021-25 (MINIMUMS INCLUDED), grading the comp-market median against the actual
salary. Reports overall + two key segments: declined players (trailing >>
current) and minimum signings.

Variants override the target's match score (features['trailing_barrett'],
which drives BOTH comp distance and the vet-min inclusion threshold):
  trailing = career-weighted (last 3 healthy yrs) — the shipped behavior
  current  = sign-year Barrett (recent form)
  blend    = 0.6*current + 0.4*trailing
  lower    = min(current, trailing)

Usage:  python -u scripts/experiment_match_score.py
"""
import sys, warnings
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np

SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "p", "__file__": str(ROOT / "pages" / "Contract_Predictor.py")}
exec(compile("".join(SRC[:cut]), "p", "exec"), ns)
fc, lh = ns["find_comparables"], ns["load_historical_signings"]
wm, idw, SALCAP = ns["_weighted_median"], ns["_inverse_distance_weights"], ns["SALARY_CAP_M"]

CAPM = 165.0
TEST_YEARS = {2021, 2022, 2023, 2024, 2025}
VARIANTS = ["trailing", "current", "blend", "lower"]


def vscore(cur, tr, v):
    if v == "current":  return cur
    if v == "blend":    return 0.6 * cur + 0.4 * tr
    if v == "lower":    return min(cur, tr)
    return tr  # trailing


def main():
    pool = lh(n_recent_pairs=8).copy()
    pool["year"] = pool["signed_in"].str[:4].astype(int)
    pool["sal_pct"] = pool.apply(
        lambda r: float(r["salary_curr"]) / (SALCAP.get(r["signed_in"], 154.6) * 1e6), axis=1)
    print(f"pool: {len(pool)} signings, years {pool['year'].min()}-{pool['year'].max()}\n", flush=True)

    recs = {v: [] for v in VARIANTS}   # (actual_pct, market_pct, cur, tr)
    targets = pool[pool["year"].isin(TEST_YEARS)]
    for _, t in targets.iterrows():
        hist = pool[pool["year"] < t["year"]]
        if len(hist) < 30:
            continue
        cur, tr = float(t["barrett_score"]), float(t["career_weighted_barrett"])
        for v in VARIANTS:
            feats = {"name": t["Player"], "position": t["pos"], "age": t["age"],
                     "barrett_score": cur, "draft_tier": t.get("draft_tier", "Undrafted"),
                     "trailing_barrett": vscore(cur, tr, v)}
            comps = fc(feats, hist, n=6)
            mkt = None
            if not comps.empty:
                w = idw(comps["distance"].astype(float).values) if "distance" in comps else np.ones(len(comps))
                mkt = float(wm(comps["sal_pct"].astype(float).values, w))
            recs[v].append((float(t["sal_pct"]), mkt, cur, tr))

    def seg(rows, mask):
        rows = [r for r, m in zip(rows, mask) if m and r[1] is not None]
        if not rows:
            return None
        a = np.array([r[0] for r in rows]); m = np.array([r[1] for r in rows])
        e = np.abs(a - m) * CAPM
        return len(rows), np.median(e), np.mean(e <= 3) * 100, np.mean(e <= 5) * 100

    base = recs["trailing"]
    cur_arr = np.array([r[2] for r in base]); tr_arr = np.array([r[3] for r in base])
    act_arr = np.array([r[0] for r in base])
    masks = {
        "ALL":        np.ones(len(base), bool),
        "declined":   (tr_arr - cur_arr) >= 5,        # trail >> current
        "improving":  (cur_arr - tr_arr) >= 5,         # current >> trail (breakouts)
        "minimum":    act_arr < 0.03,                  # actually signed a minimum
        "stable":     np.abs(tr_arr - cur_arr) < 3,    # control: trail ~ current
    }
    for segname, mask in masks.items():
        print(f"=== {segname}  (n={int(mask.sum())}) ===", flush=True)
        print(f"  {'variant':<10}{'n':>5}{'med|err|':>10}{'within$3M':>10}{'within$5M':>10}", flush=True)
        for v in VARIANTS:
            s = seg(recs[v], mask)
            if s:
                n, med, w3, w5 = s
                print(f"  {v:<10}{n:>5}{med:>9.1f}M{w3:>9.0f}%{w5:>9.0f}%", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
