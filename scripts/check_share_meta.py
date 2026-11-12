#!/usr/bin/env python3
"""Guard: cache/share_meta.json must agree with the board it describes.

Every /?player= landing page takes its <title>, meta description, crawlable
body and Person structured data from this one file, and the sitemap takes the
URL list from it. So a stale or thin share_meta.json is not a cache miss, it is
several hundred public pages stating numbers the site itself contradicts.

That is exactly what happened. The file was not in the nightly and sat at its
2026-07-14 build for four months: 388 of 388 players' numbers disagreed with
the live board, 73 board players had no record at all, and the public page for
Vucevic advertised "BOS, 19.7 Barrett Score, ranked #80, paid $21.5M" against a
board reading "ORL, -1.1, #423, $2.4M". Nothing failed, because nothing looked.

Run:  python3 scripts/check_share_meta.py            (checks the real file)
      python3 scripts/check_share_meta.py --selftest  (proves the checks fire)
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MIN_COVERAGE = 0.98     # share of board players that must have a record
MAX_DISAGREE = 0.02     # share whose score/rank may differ from the board
MAX_AGE_DAYS = 3        # the nightly rebuilds it, so older than this is a stall
PATH = ROOT / "cache" / "share_meta.json"
# Matches both forms: "12.3 Barrett Score, ranked #40" and the fallback
# "12.3 Barrett Score in 2025-26, his most recent season, ranked #40".
_NUM = re.compile(r"([-\d.]+) Barrett Score(?:[^,]*,)?[^#]*ranked #(\d+)")


def _load():
    return json.loads(PATH.read_text(encoding="utf-8"))


def audit(meta: dict, board: dict) -> list:
    """Return a list of failure strings. board: {normalized_name: (score, rank)}."""
    bad = []
    recs = {k: v for k, v in meta.items()
            if not k.startswith("_") and isinstance(v, dict)}

    covered = len(set(board) & set(recs))
    frac = covered / max(len(board), 1)
    print(f"  coverage      {covered}/{len(board)} = {frac:.2%}")
    if frac < MIN_COVERAGE:
        missing = sorted(set(board) - set(recs))[:8]
        bad.append(f"coverage {frac:.1%} < {MIN_COVERAGE:.0%} (missing e.g. {missing})")

    no_name = [k for k, v in recs.items() if not v.get("n")]
    print(f"  display names {len(recs) - len(no_name)}/{len(recs)} records carry 'n'")
    if no_name:
        bad.append(f"{len(no_name)} record(s) have no display name, so the sitemap "
                   f"cannot emit their URL (e.g. {no_name[:5]})")

    checked = disagree = 0
    worst = []
    for k, v in recs.items():
        if k not in board:
            continue
        b_score, b_rank = board[k]
        m = _NUM.search(v.get("d", ""))
        if not m:
            continue
        checked += 1
        if abs(float(m.group(1)) - b_score) > 0.05 or int(m.group(2)) != b_rank:
            disagree += 1
            worst.append((abs(int(m.group(2)) - b_rank), k,
                          m.group(1), m.group(2), f"{b_score:.1f}", b_rank))
    dfrac = disagree / max(checked, 1)
    print(f"  numbers       {checked - disagree}/{checked} agree with the board "
          f"({dfrac:.2%} disagree)")
    for _, k, os_, orr, ns, nr in sorted(worst, reverse=True)[:5]:
        print(f"      {k:26s} page says {os_:>6s} #{orr:<5s} board says {ns:>6s} #{nr}")
    if dfrac > MAX_DISAGREE:
        bad.append(f"{disagree} of {checked} players ({dfrac:.1%}) are published with "
                   f"numbers the board contradicts")

    stamp = str(meta.get("_meta", {}).get("built_at", ""))
    if not stamp:
        bad.append("no _meta.built_at stamp, so staleness cannot be measured "
                   "(rebuild with scripts/build_share_meta.py)")
        print("  built_at      MISSING")
    else:
        try:
            age = (time.time() - time.mktime(time.strptime(stamp[:19],
                   "%Y-%m-%dT%H:%M:%S"))) / 86400
            print(f"  built_at      {stamp}  ({age:.1f} days old)")
            if age > MAX_AGE_DAYS:
                bad.append(f"built {age:.0f} days ago; the nightly rebuilds this "
                           f"file, so anything over {MAX_AGE_DAYS} days means the "
                           f"step is not running")
        except Exception as e:
            bad.append(f"unparseable built_at {stamp!r}: {e}")

    orphan = sorted(set(recs) - set(board))
    print(f"  orphans       {len(orphan)} record(s) for players off the board "
          f"(harmless: an old shared link still resolves)")
    return bad


def _tracked() -> list:
    """A file on disk that git does not have is a file production never sees.

    This is the 347-vs-379 lesson wearing a new name: that bug ran for five
    months because a cache file existed locally and nowhere else. seo_selfheal
    reads share_meta from the REPO cache in the deploy image, so an uncommitted
    rebuild changes nothing on the live site however good it looks here."""
    import subprocess
    rel = str(PATH.relative_to(ROOT))
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                           cwd=ROOT, capture_output=True, text=True)
    except Exception as e:
        print(f"  tracked       could not ask git ({e})")
        return []
    if r.returncode != 0:
        return [f"{rel} is NOT tracked by git, so the deploy image will not "
                f"carry it and every /?player= page falls back to the generic "
                f"homepage card"]
    d = subprocess.run(["git", "diff", "--quiet", "--", rel], cwd=ROOT)
    print(f"  tracked       yes" + ("  (uncommitted changes pending)"
                                    if d.returncode else "  (committed)"))
    return []


def _selftest() -> int:
    """Positive control. A check that cannot fail proves nothing, so break the
    inputs on purpose and require each failure to be reported."""
    board = {"a player": (10.0, 5), "b player": (8.0, 9), "c player": (6.0, 20)}
    def rec(n, s, r):
        return {"n": n, "s": s, "r": r,
                "d": f"{n}: {s:.1f} Barrett Score, ranked #{r} in the NBA by value.",
                "t": f"{n} | HoopsValue"}
    good = {"a player": rec("A Player", 10.0, 5),
            "b player": rec("B Player", 8.0, 9),
            "c player": rec("C Player", 6.0, 20),
            "_meta": {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    cases = []
    print("control: a correct file")
    cases.append(("clean file reports nothing", audit(dict(good), board) == []))

    print("\ncontrol: stale numbers")
    stale = {k: (dict(v) if k != "_meta" else v) for k, v in good.items()}
    stale["a player"] = rec("A Player", 99.0, 400)
    stale["b player"] = rec("B Player", 50.0, 300)
    f = audit(stale, board)
    cases.append(("stale numbers are caught", any("contradicts" in x for x in f)))

    print("\ncontrol: a dropped player")
    thin = {k: v for k, v in good.items() if k != "c player"}
    f = audit(thin, board)
    cases.append(("missing coverage is caught", any("coverage" in x for x in f)))

    print("\ncontrol: no display name")
    nameless = {k: (dict(v) if k != "_meta" else v) for k, v in good.items()}
    nameless["a player"].pop("n")
    f = audit(nameless, board)
    cases.append(("a record with no display name is caught",
                  any("display name" in x for x in f)))

    print("\ncontrol: an old build stamp")
    old = dict(good)
    old["_meta"] = {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                              time.localtime(time.time() - 40 * 86400))}
    f = audit(old, board)
    cases.append(("a stalled nightly is caught", any("days ago" in x for x in f)))

    print()
    ok = True
    for label, passed in cases:
        print(("  PASS  " if passed else "  FAIL  ") + label)
        ok = ok and passed
    print("\n[PASS] selftest: every check fires" if ok else "\n[FAIL] selftest")
    return 0 if ok else 1


def main() -> int:
    if "--selftest" in sys.argv:
        return _selftest()
    if not PATH.exists():
        print(f"[FAIL] {PATH.relative_to(ROOT)} does not exist. Every /?player= page "
              f"falls back to the generic homepage card.")
        return 1
    import utils
    # Same frame build_share_meta uses, and the same one app.py's hub panel
    # uses. Checking against build_ranked_projected would report 134 "orphan"
    # records and a rank disagreement on every fallback player.
    df = utils.build_ranked_blended(utils.SEASONS[0])
    board = {utils.normalize(str(r["Player"])):
             (float(r["barrett_score"]), int(r["score_rank"]))
             for _, r in df.iterrows()}
    print(f"share_meta vs the {utils.SEASONS[0]} board")
    bad = audit(_load(), board) + _tracked()
    print()
    if bad:
        for b in bad:
            print(f"[FAIL] {b}")
        print("\nFix: python3 scripts/build_share_meta.py  (step 5d of the nightly)")
        return 1
    print("[PASS] share_meta.json agrees with the board it describes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
