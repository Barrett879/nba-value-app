"""Precompute per-player SEO + share metadata (title, description, numbers).

A /?player=X URL is a real landing page: it has to unfurl in iMessage/Slack/X
with THAT player's Barrett Score, and it has to give Google a title, a
description, a crawlable body and Person structured data that all agree with
each other. serve.py / seo_selfheal.py read cache/share_meta.json and inject
per request (no model work on the request path).

Output: cache/share_meta.json = {normalized_name: record}, where a record is
    n   display name (accents intact) -- the CANONICAL ?player= value
    tm  team code
    s   barrett score            r   value rank
    pd  salary $M                w   market value $M
    vd  value_diff $M (>0 overpaid, <0 underpaid)
    pcv predicted next contract $M, or absent
    t   <title>                  d   meta description
plus a "_meta" key with the build stamp and the population it was built from.

The structured fields exist so seo_selfheal can render a body and JSON-LD
without re-deriving anything, and so the sitemap can emit the display-name URL
(the normalized keys cannot be turned back into "Nikola Vucevic").

WHY THIS IS IN THE NIGHTLY (added 2026-11-12): it was not, and the committed
file sat at its 2026-07-14 build for four months while the board moved under
it. Measured the day it was found: 388 of 388 players' numbers disagreed with
the live board, 73 board players had no entry at all, and 136 entries were for
players no longer on the board. Vucevic's public page advertised "BOS, 19.7
Barrett Score, ranked #80, paid $21.5M" against a board that said
"ORL, -1.1, #423, $2.4M". Every one of those pages is what Google reads.

NOT subject to the Render /data seeding trap: seo_selfheal reads
Path(__file__).parent/"cache"/..., the REPO cache in the deploy image, not
utils.CACHE_DIR (/data/cache on Render). A re-commit under the same filename
does take effect here, unlike a utils cache artifact.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import utils  # noqa: E402

OUT = ROOT / "cache" / "share_meta.json"
# Refuse to publish a file that covers less of the board than this. A thin
# write is worse than a stale one: a player with no record falls back to the
# generic homepage card, and seo_selfheal has no way to tell "not a player"
# from "the build dropped him".
MIN_COVERAGE = 0.98


def _f(x, default=0.0):
    """$M at ONE decimal, the same precision the prose prints.

    Stored at two decimals, $2,449,421 became 2.45 in the record and "$2.5M" in
    the crawlable body while the meta description, formatted from the raw
    number, said "$2.4M". One page, one number, two values. Rounding once here
    and deriving both from it makes that impossible rather than unlikely."""
    try:
        return round(float(x) / 1e6, 1)
    except Exception:
        return default


def main() -> None:
    # build_ranked_BLENDED, not build_ranked_projected, because the hub panel a
    # /?player= URL actually opens is built from the blended frame (app.py:850).
    # The projected frame holds 461 current-season rows; the blended frame holds
    # those plus 134 players whose score falls back to the previous season, so
    # ranks are computed over a different pool: the same Vucevic was #423 in one
    # and #552 in the other. Titling a page with a rank the page itself does not
    # show is the stale-metadata bug again, just faster. Same scores, superset
    # population, matching ranks, and 134 more real landing pages.
    season = utils.SEASONS[0]
    df = utils.build_ranked_blended(season)
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
        # One rounded value per number, then BOTH the prose and the stored
        # field derive from it, so the description and the page body can never
        # print the same dollar figure two ways.
        pdm = _f(r["salary"])
        wm = _f(r["projected_salary"])
        vdm = _f(r["value_diff"])  # salary - market value; >0 overpaid, <0 underpaid
        paid = f"${pdm:.1f}M"
        worth = f"${wm:.1f}M"
        gap = ""
        if vdm <= -2:
            gap = f", underpaid by ${-vdm:.1f}M"
        elif vdm >= 2:
            gap = f", overpaid by ${vdm:.1f}M"
        # A fallback row's score is LAST season's. The hub tags that rather
        # than showing it as this year's, so the description has to as well:
        # otherwise 134 pages state a number for a season the player has not
        # played. Blank for a current-season row.
        syr = str(r.get("score_season") or "")
        when = f" in {syr}, his most recent season," if syr and syr != season else ""
        n = utils.normalize(name)
        p = pcv.get(n)
        pcvm = round(float(p), 1) if p is not None else None
        tail = (f" HoopsValue projects his next contract at ${pcvm:.1f}M."
                if pcvm is not None else " See the full value breakdown on HoopsValue.")
        desc = (f"{name} ({team}): {score:.1f} Barrett Score{when} ranked #{rank} "
                f"in the NBA by value. Paid {paid}, worth {worth}{gap}.{tail}"
                if when else
                f"{name} ({team}): {score:.1f} Barrett Score, ranked #{rank} in the NBA by value. "
                f"Paid {paid}, worth {worth}{gap}.{tail}")
        title = f"{name} - NBA Value & Contract Prediction | HoopsValue"
        rec = {
            "n": name, "tm": team,
            "s": round(score, 1), "r": rank,
            "pd": pdm, "w": wm, "vd": vdm,
            "t": title, "d": desc,
        }
        if syr and syr != season:
            rec["sy"] = syr   # the season this score came from, when not the current one
        if pcvm is not None:
            rec["pcv"] = pcvm
        meta[n] = rec

    # ---- guards: a partial write is a published defect, not a smaller file ---
    board = {utils.normalize(str(p)) for p in df["Player"]}
    covered = len(board & set(meta))
    frac = covered / max(len(board), 1)
    print(f"board {len(board)} players, records {len(meta)}, "
          f"coverage {covered}/{len(board)} = {frac:.3%}")
    if frac < MIN_COVERAGE:
        missing = sorted(board - set(meta))[:10]
        raise SystemExit(f"FAIL: coverage {frac:.1%} < {MIN_COVERAGE:.0%}; "
                         f"refusing to write. missing e.g. {missing}")
    bad = [k for k, v in meta.items()
           if not v.get("n") or not v.get("t") or not v.get("d")
           or v.get("r") is None or v.get("s") is None]
    if bad:
        raise SystemExit(f"FAIL: {len(bad)} record(s) missing a required field, "
                         f"e.g. {bad[:5]}; refusing to write.")

    meta["_meta"] = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "season": season,
        "players": len(board),
        "formula_version": getattr(utils, "FORMULA_VERSION", ""),
    }
    OUT.write_text(json.dumps(meta, separators=(",", ":")))
    print(f"wrote {OUT.name}  ({len(board)} players, {OUT.stat().st_size // 1024}KB)")
    for probe in ("nikola jokic", "lebron james", "victor wembanyama"):
        if probe in meta:
            print(f"  {probe}: {meta[probe]['d'][:120]}")


if __name__ == "__main__":
    main()
