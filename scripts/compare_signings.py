"""Score the model against REAL signings. Reads data/real_signings_2026.csv (the
actual reported deals) and, for each player, pulls the model's OWN projection
(the pure prediction — no manual override) so we can see projected vs actual and
a running accuracy scorecard as free agency unfolds.

Usage:  python scripts/compare_signings.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
cut = next(i for i, l in enumerate(SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
ns = {"__name__": "cp", "__file__": str((ROOT / "pages" / "Contract_Predictor.py").resolve())}
exec(compile("".join(SRC[:cut]), "cp", "exec"), ns)
gpf = ns["get_player_features"]
pcv = ns["projected_contract_value"]          # the DISPLAYED projection (blended w/ comps) — what the page + board show
band_fn = ns["_relative_band_dollars"]        # calibrated ~80% half-width around the displayed value
CAP_M = 165.0

ALIASES = {"CJ McCollum": ["CJ McCollum", "C.J. McCollum"]}


def model_proj_M(name):
    """Return the projection EXACTLY as the site shows it (projected_contract_value,
    blended toward comparable signings), plus the displayed ~80% range."""
    for n in ALIASES.get(name, [name]):
        f = gpf(n)
        if f:
            v = pcv(f) / 1e6
            b = band_fn(v * 1e6) / 1e6
            return v, (max(0.0, v - b), v + b)
    return None, None


def main():
    rows = []
    with open(ROOT / "data" / "real_signings_2026.csv") as fh:
        for r in csv.DictReader(l for l in fh if l.strip() and not l.lstrip().startswith("#")):
            if r.get("player"):
                rows.append(r)

    print(f"\n{'player':22s} {'deal':>12} {'actual y1':>10} {'model':>8} {'model range':>15} {'Δ':>8} {'in $4M':>7} {'in 5%cap':>9}", flush=True)
    print("-" * 100, flush=True)
    errs = []
    for r in rows:
        actual = float(r["yr1_M"])
        proj, band = model_proj_M(r["player"])
        deal = f"{r['years']}yr/${r['total_M']}M"
        if proj is None:
            print(f"{r['player']:22s} {deal:>12} {actual:>9.1f}M {'  n/a':>8}", flush=True)
            continue
        d = proj - actual
        in4 = abs(d) <= 4
        in5 = abs(d) <= 0.05 * CAP_M
        errs.append((r["player"], proj, actual, d, in4, in5))
        rng = f"${band[0]:.0f}-{band[1]:.0f}M" if band else ""
        print(f"{r['player']:22s} {deal:>12} {actual:>9.1f}M {proj:>7.1f}M {rng:>15} {d:>+7.1f}M {('yes' if in4 else 'no'):>7} {('yes' if in5 else 'no'):>9}", flush=True)

    if errs:
        n = len(errs)
        import statistics as st
        ae = [abs(e[3]) for e in errs]
        print("-" * 100, flush=True)
        print(f"\nSCORECARD ({n} real signings):", flush=True)
        print(f"  within $4M:        {100*sum(e[4] for e in errs)/n:.0f}%", flush=True)
        print(f"  within 5% of cap:  {100*sum(e[5] for e in errs)/n:.0f}%", flush=True)
        print(f"  median |error|:    ${st.median(ae):.1f}M", flush=True)
        print(f"  mean signed error: ${sum(e[3] for e in errs)/n:+.1f}M  (+ = model projects HIGH)", flush=True)


if __name__ == "__main__":
    main()
