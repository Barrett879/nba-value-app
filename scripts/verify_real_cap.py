"""Prototype: compute real per-team cap space for 2026-27 from actual committed
salaries and compare it to the hand-typed data/team_landscape_2026.csv.

Team map = cached current-season league stats (PLAYER_NAME -> TEAM_ABBREVIATION,
the app's own abbreviations). Salaries = cached next_contracts pkl (the ESPN
2026-27 salaries fetch_next_year_contracts builds). No network if the cache is
warm. Prints committed payroll, computed cap room + tool, vs the CSV values, so
we can eyeball whether capped-out teams read as capped and rebuilders as flush.

Usage:  python -u scripts/verify_real_cap.py
"""
import sys, glob, pickle, warnings
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import pandas as pd
import team_suitors as ts

CAP = 165.0  # 2026-27 projected cap (utils.SALARY_CAP_M["2026-27"])
CACHE = ROOT / "cache"


def load_team_map() -> pd.DataFrame:
    """[team, player] from cached current-season league stats, one row per player
    (their max-minutes team if traded)."""
    f = CACHE / "league_stats_2025_26.parquet"
    df = pd.read_parquet(f)
    name = next(c for c in df.columns if c.upper() in ("PLAYER_NAME", "PLAYER"))
    team = next(c for c in df.columns if c.upper() in ("TEAM_ABBREVIATION", "TEAM", "TM"))
    mins = next((c for c in df.columns if c.upper() in ("MIN", "GP")), None)
    df = df[[name, team] + ([mins] if mins else [])].copy()
    df.columns = ["player", "team"] + (["min"] if mins else [])
    if "min" in df.columns:
        df = df.sort_values("min", ascending=False)
    df = df.drop_duplicates(subset="player", keep="first")
    return df[["team", "player"]]


def load_next_contracts() -> dict:
    """{norm_name: {salary, type}} from the cached next_contracts pkl (espn 2026)."""
    cands = sorted(glob.glob(str(CACHE / "next_contracts_2026_v*.pkl")))
    if not cands:
        print("!! no next_contracts_2026_v*.pkl in cache — run the app once to warm it")
        return {}
    with open(cands[-1], "rb") as fh:
        return pickle.load(fh)


def main():
    rosters = load_team_map()
    contracts = load_next_contracts()
    n_sal = sum(1 for v in contracts.values() if (v or {}).get("salary"))
    print(f"team map: {len(rosters)} players, {rosters['team'].nunique()} teams")
    print(f"contracts: {len(contracts)} players, {n_sal} with a 2026-27 salary\n")

    cap_table = ts.compute_cap_space(rosters, contracts, CAP)
    land = ts.load_team_landscape()
    hand = {r["team"]: r for _, r in land.iterrows()}

    print(f"{'tm':<4}{'committed':>11}{'COMPUTED room':>16}{'tool':>10}   | "
          f"{'HAND room':>10}{'hand tool':>10}")
    rows = sorted(cap_table.items(), key=lambda kv: -kv[1]["cap_space_M"])
    for tm, c in rows:
        h = hand.get(tm, {})
        print(f"{tm:<4}{c['committed_M']:>10.1f}M{c['cap_space_M']:>15.1f}M{c['top_exception']:>10}   | "
              f"{float(h.get('cap_space_M', 0)):>9.1f}M{str(h.get('top_exception','')):>10}")

    miss = [tm for tm in hand if tm not in cap_table]
    if miss:
        print(f"\nno computed payroll (kept hand value): {', '.join(miss)}")
    # sanity: how many teams look capped vs flush
    capped = sum(1 for _, c in cap_table.items() if c["cap_space_M"] < 1)
    print(f"\n{capped}/{len(cap_table)} teams over the cap (cap room 0); "
          f"{len(cap_table)-capped} with room.")


if __name__ == "__main__":
    main()
