"""Invariant sweep of the pre-computed Front Office board (cache/fa_board_v1.json):
re-sign cap plan, best-fits, pursue, and text hygiene across all 30 teams. Pure
JSON checks, so it's fast and re-runnable.

Usage:  python -u scripts/audit_board.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
d = json.loads((ROOT / "cache" / "fa_board_v1.json").read_text())
T = d["teams"]

K = ["MISSING_FIELD", "RESIGN_RUNNING", "RESIGN_ZONE", "RESIGN_KEEP", "RESIGN_ALLIN",
     "RESIGN_COST", "RESIGN_ORDER", "BESTFIT_GRADE", "BESTFIT_DUP_POS", "BESTFIT_DUP_NAME",
     "PURSUE_OVER_VALUE", "PURSUE_DUP", "EMDASH", "COMMITTED_LOW"]
issues = {k: [] for k in K}


def primary(pos):
    return str(pos).split("/")[0]


for ab, b in T.items():
    for f in ("name", "timeline", "needs", "best_fits", "resign", "pursue"):
        if f not in b:
            issues["MISSING_FIELD"].append(f"{ab}: missing {f}")

    # ── best fits: valid grade, deduped by primary position + by name ───────────
    seen_pos, seen_name = set(), set()
    for f in b.get("best_fits", []):
        if f["grade"] not in ("A+", "A", "A-", "B+", "B"):
            issues["BESTFIT_GRADE"].append(f"{ab}: {f['name']} grade '{f['grade']}'")
        pp = primary(f["pos"])
        if pp in seen_pos:
            issues["BESTFIT_DUP_POS"].append(f"{ab}: two best fits at {pp} ({f['name']})")
        if f["name"] in seen_name:
            issues["BESTFIT_DUP_NAME"].append(f"{ab}: {f['name']} listed twice")
        seen_pos.add(pp)
        seen_name.add(f["name"])

    # ── pursue: offer never exceeds market value; no dup players ────────────────
    pnames = [x["name"] for x in b["pursue"]]
    if len(pnames) != len(set(pnames)):
        issues["PURSUE_DUP"].append(f"{ab}: duplicate in pursue")
    for x in b["pursue"]:
        if x["offer_M"] > x["value_M"] + 0.6:
            issues["PURSUE_OVER_VALUE"].append(
                f"{ab}: {x['name']} offer ${x['offer_M']}M > value ${x['value_M']}M")

    # ── re-sign: cost == market value (Bird rights pay full value) ──────────────
    for x in b["resign"]:
        if abs(x["offer_M"] - x["value_M"]) > 0.6:
            issues["RESIGN_COST"].append(
                f"{ab}: {x['name']} keep ${x['offer_M']}M != value ${x['value_M']}M")

    # ── re-sign cap plan: running total, zones, keep flag, ordering ─────────────
    p = b.get("resign_plan")
    if p:
        if p["committed_M"] < 50:
            issues["COMMITTED_LOW"].append(f"{ab}: committed ${p['committed_M']}M (<50, should've been skipped)")
        run = p["committed_M"]
        last_barrett = None
        names = {x["name"]: x for x in b["resign"]}
        for k in p["keeps"]:
            run += k["cost_M"]
            if abs(run - k["running_M"]) > 1.1:
                issues["RESIGN_RUNNING"].append(f"{ab}: {k['name']} running ${k['running_M']}M != ${round(run)}M")
            ez = "ok" if k["running_M"] < p["tax_M"] else "tax" if k["running_M"] < p["apron2_M"] else "over"
            if k["zone"] != ez:
                issues["RESIGN_ZONE"].append(f"{ab}: {k['name']} zone '{k['zone']}' != '{ez}'")
            if k["keep"] != (k["running_M"] < p["apron2_M"]):
                issues["RESIGN_KEEP"].append(f"{ab}: {k['name']} keep={k['keep']} but running ${k['running_M']}M vs apron ${p['apron2_M']}M")
            br = names.get(k["name"], {}).get("barrett")
            if last_barrett is not None and br is not None and br > last_barrett + 1e-6:
                issues["RESIGN_ORDER"].append(f"{ab}: {k['name']} (barrett {br}) ranked below {last_barrett}")
            last_barrett = br if br is not None else last_barrett
        if abs(run - p["all_in_M"]) > 1.1:
            issues["RESIGN_ALLIN"].append(f"{ab}: all_in ${p['all_in_M']}M != ${round(run)}M")

    if "—" in json.dumps(b, ensure_ascii=False):
        issues["EMDASH"].append(f"{ab}: em dash in displayed text")

total = sum(len(v) for v in issues.values())
print(f"board audit · {len(T)} teams · {total} issues\n")
for k, v in issues.items():
    flag = "" if not v else "  <-- "
    print(f"=== {k}: {len(v)} ==={flag}")
    for line in v[:15]:
        print("  ", line)
sys.exit(1 if total else 0)
