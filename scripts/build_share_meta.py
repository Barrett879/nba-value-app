"""Precompute per-player Open Graph share metadata (title + description).

A shared /?player=X link should unfurl in iMessage/Slack/X with THAT player's
Barrett Score, value gap, and predicted contract -- not the generic homepage
card. serve.py's root handler reads cache/share_meta.json and injects the tags
per request (no per-request model work). Re-run after the board/model updates.

Output: cache/share_meta.json = { normalized_name: {"t": title, "d": desc} }
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import utils  # noqa: E402

OUT = ROOT / "cache" / "share_meta.json"


def _m(x) -> str:
    try:
        return f"${float(x) / 1e6:.1f}M"
    except Exception:
        return "$0M"


def main() -> None:
    df = utils.build_ranked_projected(utils.SEASONS[0])
    # predicted next contract per player (normalized-name -> pcv $M)
    pcv = {}
    try:
        players = json.loads((ROOT / "cache" / "player_hub_pcv_v2.json").read_text()).get("players", {})
        src = players.values() if isinstance(players, dict) else players
        for rec in src:
            nm = rec.get("player")
            if nm and rec.get("pcv_M") is not None:
                pcv[utils.normalize(str(nm))] = float(rec["pcv_M"])
    except Exception as e:
        print("warn: no pcv cache:", e)

    meta = {}
    for _, r in df.iterrows():
        name = str(r["Player"])
        team = str(r["Team"])
        score = float(r["barrett_score"])
        rank = int(r["score_rank"])
        paid = _m(r["salary"])
        worth = _m(r["projected_salary"])
        vd = float(r["value_diff"])  # salary - market value; >0 overpaid, <0 underpaid
        gap = ""
        if vd <= -2e6:
            gap = f", underpaid by {_m(-vd)}"
        elif vd >= 2e6:
            gap = f", overpaid by {_m(vd)}"
        n = utils.normalize(name)
        p = pcv.get(n)
        tail = (f" HoopsValue projects his next contract at {_m(p * 1e6)}."
                if p is not None else " See the full value breakdown on HoopsValue.")
        desc = (f"{name} ({team}): {score:.1f} Barrett Score, ranked #{rank} in the NBA by value. "
                f"Paid {paid}, worth {worth}{gap}.{tail}")
        title = f"{name} - NBA Value & Contract Prediction | HoopsValue"
        meta[n] = {"t": title, "d": desc}

    OUT.write_text(json.dumps(meta, separators=(",", ":")))
    print(f"wrote {OUT.name}  ({len(meta)} players, {OUT.stat().st_size // 1024}KB)")
    for probe in ("nikola jokic", "lebron james", "victor wembanyama"):
        if probe in meta:
            print(f"  {probe}: {meta[probe]['d'][:120]}")


if __name__ == "__main__":
    main()
