"""Full audit: every active player, sorted by Barrett score, with the numbers
that drive the projection — so the over-projections can be inspected at a glance.

Columns: rank, player, age, pos, svc, score (displayed Barrett), trailing
(trailing-weighted Barrett — what comps/threshold actually use), model$ (the
HistGBM + CBA prediction), market$ (comp median), final$ (blended headline the
page shows). Writes player_contract_audit.csv and prints the low-Barrett tail
(where the problems live).

Usage:  python -u scripts/audit_all_contracts.py
"""
import sys, csv, time, warnings
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np

SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "p", "__file__": str(ROOT / "pages" / "Contract_Predictor.py")}
exec(compile("".join(SRC[:cut]), "p", "exec"), ns)

fc, lh, gf, predict = ns["find_comparables"], ns["load_historical_signings"], ns["get_player_features"], ns["predict_contract"]
cd, wm, idw, ctx = ns["_comp_salaries_in_contract_dollars"], ns["_weighted_median"], ns["_inverse_distance_weights"], ns["_classify_context"]
SUPERMAX, SALCAP = ns["SUPERMAX_CAP_PCT"], ns["SALARY_CAP_M"]
CUR, CON, names = ns["CURRENT_SEASON"], ns["CONTRACT_SEASON"], ns["active_names"]
CAP = SALCAP.get(CON, 165.0)


def market_median(f, comps):
    if comps.empty:
        return None
    comps = comps.copy()
    comps["context"] = comps.apply(ctx, axis=1)
    used, suppressed = comps, False
    cur_pct = float(f.get("salary", 0) or 0) / (SALCAP.get(CUR, 154.6) * 1e6)
    if cur_pct >= SUPERMAX:
        nonpc = comps[comps["context"] != "Paycut"]
        if len(nonpc) >= 3:
            used = nonpc
        else:
            return None
    if used.empty:
        return None
    sal = cd(used)
    w = idw(used["distance"].astype(float).values) if "distance" in used else np.ones_like(sal)
    return wm(sal, w) / 1e6


def blend(pred, market):
    min_floor = pred.get("min_floor_dollars", 0.015 * CAP * 1e6) / 1e6
    model_M = pred["predicted"] / 1e6
    # Tier-gate: blend only in the mid-tier ($7-25M model projection); model
    # alone at the extremes (matches pages/Contract_Predictor.py).
    if (market is None or pred.get("cba_cap_applied") or pred.get("cba_floor_applied")
            or not (7.0 <= model_M <= 25.0)):
        return model_M
    hi = max(model_M, market)
    gap = abs(model_M - market) / hi if hi > 0 else 0.0
    if gap > 0.25 and market > 0:
        w_mkt = min(0.65, 0.35 + 0.30 * (gap - 0.25) / 0.35)
        bl = max((1 - w_mkt) * model_M + w_mkt * market, min_floor)
        if abs(bl - model_M) > 0.05:
            return bl
    return model_M


def main():
    hist = lh(n_recent_pairs=3)
    t0 = time.time()
    rows = []
    for i, name in enumerate(names):
        try:
            f = gf(name)
            if not f:
                continue
            pred = predict(f)
            comps = fc(f, hist, n=6)
            mkt = market_median(f, comps)
            final = blend(pred, mkt)
            rows.append([name, f.get("age"), f.get("position"), f.get("service_years"),
                         round(float(f.get("barrett_score") or 0), 1),
                         round(float(f.get("trailing_barrett", f.get("barrett_score")) or 0), 1),
                         round(pred["predicted"] / 1e6, 1),
                         round(mkt, 1) if mkt is not None else "",
                         round(final, 1)])
        except Exception as e:
            rows.append([name, "", "", "", "", "", "ERR", str(e)[:40], ""])
        if i % 40 == 0:
            print(f"  {i}/{len(names)}  ({time.time()-t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: (r[4] if isinstance(r[4], (int, float)) else -999), reverse=True)
    for rank, r in enumerate(rows, 1):
        r.insert(0, rank)

    out = ROOT / "player_contract_audit.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "player", "age", "pos", "svc", "score", "trailing",
                    "model_$M", "market_$M", "final_$M"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} players -> {out}\n", flush=True)

    hdr = f"{'#':>4} {'player':<24}{'age':>4}{'pos':>4}{'score':>7}{'trail':>7}{'model':>7}{'mkt':>7}{'FINAL':>7}"
    def fmt(r):  # r = [rank, name, age, pos, svc, score, trailing, model, market, final]
        return (f"{r[0]:>4} {str(r[1])[:23]:<24}{str(r[2]):>4}{str(r[3]):>4}"
                f"{str(r[5]):>7}{str(r[6]):>7}{str(r[7]):>7}{str(r[8]):>7}{str(r[9]):>7}")
    print("TOP 15 BY BARRETT:\n" + hdr, flush=True)
    for r in rows[:15]:
        print(fmt(r), flush=True)
    print("\nLOWEST 45 BY BARRETT (where over-projection shows up):\n" + hdr, flush=True)
    for r in rows[-45:]:
        print(fmt(r), flush=True)


if __name__ == "__main__":
    main()
