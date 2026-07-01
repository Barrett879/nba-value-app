"""Build cache/accuracy_tracker_v1.json: the model's projection vs the ACTUAL deal
for every real 2026 signing (data/real_signings_2026.csv). The live Accuracy
Tracker page reads this cache. Run after adding a signing or changing the model.

Usage:  python -u scripts/build_accuracy_tracker.py
"""
import csv
import json
import socket
from pathlib import Path

socket.setdefaulttimeout(45)   # bound every network read so a throttled scrape can't hang forever

ROOT = Path(__file__).parent.parent
SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "cp", "__file__": str((ROOT / "pages" / "Contract_Predictor.py").resolve())}
exec(compile("".join(SRC[:cut]), "cp", "exec"), ns)
gpf = ns["get_player_features"]
pcv = ns["projected_contract_value"]          # displayed projection (raw model now that the blend is off)
band_fn = ns["_relative_band_dollars"]
CAP_M = 165.0
ALIASES = {"CJ McCollum": ["CJ McCollum", "C.J. McCollum"]}
OUT = ROOT / "cache" / "accuracy_tracker_v1.json"


def proj(name):
    for n in ALIASES.get(name, [name]):
        f = gpf(n)
        if f:
            v = pcv(f) / 1e6
            b = band_fn(v * 1e6) / 1e6
            return round(v, 1), round(max(0.0, v - b), 1), round(v + b, 1)
    return None, None, None


def main():
    rows = []
    with open(ROOT / "data" / "real_signings_2026.csv") as fh:
        for r in csv.DictReader(l for l in fh if l.strip() and not l.lstrip().startswith("#")):
            if r.get("player"):
                rows.append(r)

    out, errs = [], []
    for r in rows:
        actual = float(r["yr1_M"])
        v, lo, hi = proj(r["player"])
        rec = {"player": r["player"], "team": r.get("team", ""),
               "deal": f"{r['years']}yr/${r['total_M']}M", "years": int(r["years"]),
               "total_M": float(r["total_M"]), "actual_M": round(actual, 1),
               "type": r.get("type", ""), "date": r.get("signed_date", "")}
        if v is None:
            rec.update(model_M=None)
        else:
            d = round(v - actual, 1)
            rec.update(model_M=v, low_M=lo, high_M=hi, delta_M=d,
                       in4=bool(abs(d) <= 4), in5cap=bool(abs(d) <= 0.05 * CAP_M))
            errs.append((abs(d), d, rec["in4"], rec["in5cap"]))
        out.append(rec)
    out.sort(key=lambda x: (x.get("date") or ""), reverse=True)

    n = len(errs)
    score = None
    if n:
        import statistics as st
        score = {"n": n,
                 "within_4M": round(100 * sum(e[2] for e in errs) / n),
                 "within_5cap": round(100 * sum(e[3] for e in errs) / n),
                 "median_err_M": round(st.median(e[0] for e in errs), 1),
                 "bias_M": round(sum(e[1] for e in errs) / n, 1)}
    OUT.write_text(json.dumps({"season": "2026", "scorecard": score, "signings": out}, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(out)} signings)", flush=True)
    if score:
        print(f"  {score['within_4M']}% within $4M | {score['within_5cap']}% within 5% cap | "
              f"median ${score['median_err_M']}M | bias ${score['bias_M']:+}M", flush=True)


if __name__ == "__main__":
    main()
