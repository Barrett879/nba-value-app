"""Step 1 of the "which teams would pay him?" feature.

Turns the model's predicted price into a shortlist of realistic suitor teams.
Two gates:

  1. MONEY (hand-curated, data/team_landscape_2026.csv): can the team afford it?
     - Expensive players ($25M+): only real CAP SPACE pays.
     - Mid-level (~$10-15M): nearly everyone has the ~$14M mid-level exception.

  2. NEED (computed live from rosters — no hand-curation): would he be an UPGRADE?
     Each team's depth chart at his position is ranked by Barrett. Drop the player
     in by score: if he out-Barretts their STARTER at the spot, that team would
     start him (strongest pull); if he only beats the BACKUP, it's a rotation
     upgrade; if he doesn't crack the top 3 there, no need. The bigger the gap to
     the guy he displaces, the more motivated the team.

A team is a suitor when it can AFFORD the price AND (he upgrades the position OR
it has cap room to burn). Ranked by how high he slots × the upgrade gap.

Rosters come from the app's live per-player table (Player / Team / pos / Barrett,
from build_ranked_projected) — pass it in as `rosters`. Positions come from NBA
2K26's primary/secondary designations (data/player_positions_2k.csv) with user
overrides (player_positions_override.csv) layered on top; a player with no 2K row
falls back to a position-group default. The CSV only carries the money/context
side. No Streamlit / no network: `python team_suitors.py` to demo.
"""
from __future__ import annotations
from pathlib import Path
import unicodedata
import pandas as pd

_DATA = Path(__file__).parent / "data"
CSV_PATH = _DATA / "team_landscape_2026.csv"
POS_2K_PATH = _DATA / "player_positions_2k.csv"             # NBA 2K26 primary/secondary
POS_OVERRIDE_PATH = _DATA / "player_positions_override.csv"  # user corrections (win over 2K)
DEFAULT_NT_MLE = 15.05  # 2026-27 non-taxpayer mid-level exception, $M (Spotrac)
ROTATION_DEPTH = 3      # he must crack the top-3 at his position to count as a "need"


SPECTRUM = ["PG", "SG", "SF", "PF", "C"]
_POS_IDX = {p: i for i, p in enumerate(SPECTRUM)}


def _normalize(name: str) -> str:
    """Match utils.normalize (NFKD strip-combining, lower, strip) WITHOUT importing
    utils (which pulls in Streamlit). Keys must align with the page's normalize()."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _norm5(pos: str) -> str:
    """Normalize any position label to one of PG / SG / SF / PF / C."""
    p = (pos or "").strip().upper().replace("-", "/").split("/")[0]
    if p in _POS_IDX:                  return p
    if p in ("G", "GUARD"):           return "SG"
    if p in ("F", "FORWARD", "WING"): return "SF"
    if p in ("C", "CENTER"):          return "C"
    return "SF"                        # unknown -> generic forward


_GROUP_OF = {"PG": "G", "SG": "G", "SF": "F", "PF": "F", "C": "C"}
_GROUP_POS = {"G": {"PG", "SG"}, "F": {"SF", "PF"}, "C": {"C"}}


def group_flex(pos: str) -> str:
    """Data-free fallback for a player with NO 2K/override entry: expand a single
    primary to its position GROUP — guards->PG/SG, forwards->SF/PF, center->C.
    Returns a resolved '/'-joined string so it parses like a 2K designation."""
    g = _GROUP_POS[_GROUP_OF[_norm5(pos)]]
    return "/".join(sorted(g, key=lambda p: _POS_IDX[p]))


def _eligible_positions(pos: str) -> set:
    """Parse a RESOLVED position string ('SG/SF', 'PG', 'PF/C') into the set of
    spectrum spots {PG,SG,SF,PF,C} the player covers. The string is already final:
    2K designations are honored exactly (a 2K 'PG' stays PG-only) and players with
    no 2K row are pre-expanded by group_flex() before they reach here — so this is
    a pure parser, applying no flex of its own."""
    parts = {_norm5(p) for p in str(pos).replace("|", "/").split("/") if p.strip()}
    return parts or {"SF"}


def load_player_positions(pos_path: Path | str = POS_2K_PATH,
                          override_path: Path | str = POS_OVERRIDE_PATH) -> dict:
    """Per-player position map {normalized_name: 'PG/SG'} from NBA 2K26's 478
    primary/secondary designations, with manual overrides layered on top (override
    WINS). Each value is canonicalized to spectrum order. Missing/blank files just
    contribute nothing, so the caller falls back to group_flex(). These strings are
    authoritative — a value of 'PG' means PG-only, no flex."""
    out: dict = {}
    for path in (pos_path, override_path):      # 2K first, override second -> override wins
        try:
            df = pd.read_csv(path, comment="#", skip_blank_lines=True)
        except Exception:
            continue
        if "name" not in df.columns or "positions" not in df.columns:
            continue
        for nm, pos in zip(df["name"], df["positions"]):
            nm, pos = str(nm).strip(), str(pos).strip()
            if not nm or not pos or pos.lower() == "nan":
                continue
            elig = _eligible_positions(pos)
            out[_normalize(nm)] = "/".join(sorted(elig, key=lambda p: _POS_IDX[p]))
    return out


def resolve_position(name: str, fallback_primary: str = "",
                     pos_map: dict | None = None) -> str:
    """A player's final position string. The NBA 2K / override designation if we
    have one (used EXACTLY — a 2K PG-only stays PG-only); otherwise group_flex() of
    the BBRef primary (the data-free default, which can't tell PG-only from PG/SG)."""
    if pos_map:
        s = pos_map.get(_normalize(name))
        if s:
            return s
    return group_flex(fallback_primary)


def load_team_landscape(path: Path | str = CSV_PATH) -> pd.DataFrame:
    """Load the hand-curated money/context table, filling defaults for blanks."""
    df = pd.read_csv(path)
    df["cap_space_M"] = pd.to_numeric(df["cap_space_M"], errors="coerce").fillna(0.0)
    df["exception_M"] = pd.to_numeric(df["exception_M"], errors="coerce").fillna(DEFAULT_NT_MLE)
    df["timeline"] = df.get("timeline", "").fillna("").astype(str)
    return df


def roster_need(target_score: float, target_pos: str, team_roster: pd.DataFrame) -> dict:
    """How much does `target` upgrade ONE team at his position?

    team_roster: that team's players, columns [player, pos, barrett].
    Returns {slot, displaces, displaces_score, gap, need_score}:
      slot 0 = would start (better than their best at the spot)
      slot 1 = backup upgrade ... slot >= ROTATION_DEPTH = no need.
    """
    elig = _eligible_positions(target_pos)              # the spots HE can play
    # An incumbent competes with him when their eligible sets OVERLAP — so a 2K
    # SG/SF wing blocks an SG or an SF target, but a pure PG never blocks a center.
    inc = (team_roster[team_roster["pos"].map(lambda p: bool(_eligible_positions(p) & elig))]
           .sort_values("barrett", ascending=False).reset_index(drop=True))
    scores = inc["barrett"].astype(float).tolist()
    slot = sum(1 for s in scores if s > target_score)   # how many incumbents are better

    if len(scores) == 0:                                # nobody mans the spot -> gaping hole
        return {"slot": 0, "displaces": None, "displaces_score": None,
                "gap": target_score, "need_score": 3.5}
    if slot >= ROTATION_DEPTH:                          # doesn't crack the top 3 -> no need
        return {"slot": slot, "displaces": None, "displaces_score": None,
                "gap": 0.0, "need_score": 0.0}
    if slot < len(scores):                             # he leapfrogs the incumbent at this slot
        displaced = inc.iloc[slot]
        gap = target_score - float(displaced["barrett"])
        need = (ROTATION_DEPTH - slot) + min(max(gap, 0.0) / 5.0, 1.0)   # slot0≈3-4, slot1≈2-3, slot2≈1
        return {"slot": slot, "displaces": displaced["player"],
                "displaces_score": float(displaced["barrett"]), "gap": gap, "need_score": need}
    # slot == len(scores) < ROTATION_DEPTH: thin at the spot — he fills an open rotation slot
    return {"slot": slot, "displaces": None, "displaces_score": None,
            "gap": 0.0, "need_score": float(ROTATION_DEPTH - slot)}


def rank_suitors(price_M: float, target_barrett: float, target_pos: str,
                 rosters: pd.DataFrame, landscape: pd.DataFrame | None = None,
                 n: int = 5) -> list[dict]:
    """Top-n suitor teams for a player at `price_M` / `target_barrett` / `target_pos`.

    rosters: live per-player table, columns [team, player, pos, barrett].
    landscape: the money/context table (defaults to the CSV).
    """
    if landscape is None:
        landscape = load_team_landscape()
    out = []
    for _, t in landscape.iterrows():
        afford = max(float(t["cap_space_M"]), float(t["exception_M"]))
        if afford + 1e-6 < price_M:
            continue                                    # literally can't pay it
        need = roster_need(target_barrett, target_pos, rosters[rosters["team"] == t["team"]])
        has_room = float(t["cap_space_M"]) + 1e-6 >= price_M
        if need["need_score"] <= 0:
            continue                                    # he doesn't upgrade their spot -> not a suitor
        # Rank: the slot TIER dominates (would-start > backup-upgrade > depth), so
        # all starter-upgrades sit above all backup-upgrades, etc. Within a tier,
        # the size of the upgrade breaks ties — he beats a LOWER-rated incumbent =>
        # bigger gap => higher rank, so the weakest backups surface first when no
        # starter is beatable. Cap room / rebuild are small final tiebreakers.
        tl = str(t.get("timeline", "")).strip().lower()
        score = ((ROTATION_DEPTH - need["slot"]) * 10.0
                 + min(max(need["gap"], 0.0), 9.9)
                 + (1.0 if has_room else 0.0)
                 + (0.5 if tl == "rebuild" else 0.0))
        out.append({
            "team":        t["team"],
            "team_name":   t.get("team_name", t["team"]),
            "score":       round(score, 2),
            "slot":        need["slot"],
            "tool":        "cap room" if has_room else f"${float(t['exception_M']):.1f}M exception",
            "reason":      _reason(need, has_room, tl, target_barrett),
        })
    out.sort(key=lambda d: -d["score"])
    return out[:n]


def _reason(need: dict, has_room: bool, timeline: str, target_barrett: float) -> str:
    if need["displaces"] is not None:
        role = {0: "would start over", 1: "upgrades over", 2: "adds depth over"}.get(
            need["slot"], "upgrades over")
        fit = f"{role} {need['displaces']} (Barrett {need['displaces_score']:.1f} vs {target_barrett:.1f})"
    elif need["need_score"] >= 3.0:
        fit = "fills an empty spot at the position"
    elif need["need_score"] > 0:
        fit = "adds depth at a thin spot"
    else:
        fit = "no upgrade at the position"
    money = "cap room" if has_room else "via the exception"
    return " · ".join(filter(None, [fit, money, timeline]))


def build_rosters(ranked: pd.DataFrame) -> pd.DataFrame:
    """Build the live [team, player, pos, barrett] roster table from the app's
    per-player ranked table (build_ranked_projected). Robust to column-name
    variants; returns an empty frame if the needed columns aren't present so the
    caller can degrade gracefully."""
    empty = pd.DataFrame(columns=["team", "player", "pos", "barrett"])
    if ranked is None or len(ranked) == 0:
        return empty
    pick = lambda *names: next((c for c in names if c in ranked.columns), None)
    tcol = pick("Team", "TEAM_ABBREVIATION", "team")
    pcol = pick("Player", "PLAYER_NAME", "name")
    poscol = pick("position_detailed", "pos", "position", "POSITION", "Pos")
    bcol = pick("barrett_score", "barrett", "Barrett")
    if not all([tcol, pcol, poscol, bcol]):
        return empty
    out = ranked[[tcol, pcol, poscol, bcol]].copy()
    out.columns = ["team", "player", "pos", "barrett"]
    out["barrett"] = pd.to_numeric(out["barrett"], errors="coerce")
    return out.dropna(subset=["team", "pos", "barrett"])


def landscape_is_filled(landscape: pd.DataFrame) -> bool:
    """True once the money table has real data entered (not the all-default blank
    scaffold) — used to gate the live UI so it never shows misleading suitors."""
    try:
        cap = pd.to_numeric(landscape["cap_space_M"], errors="coerce").fillna(0.0)
        tl = landscape.get("timeline", pd.Series(dtype=str)).fillna("").astype(str)
        return bool((cap > 0).any() or (tl.str.strip() != "").any())
    except Exception:
        return False


if __name__ == "__main__":
    # Position source check — NBA 2K26 (+ overrides), per-player primary/secondary.
    pos_map = load_player_positions()
    print(f"Loaded {len(pos_map)} player positions from NBA 2K26 (+ overrides):")
    for nm in ["Tyrese Maxey", "Jalen Brunson", "Mikal Bridges", "Chet Holmgren",
               "Rudy Gobert", "Austin Reaves"]:
        print(f"  {nm:18} -> {resolve_position(nm, '', pos_map)}")
    print("  (no 2K row falls back to group_flex: unknown PF ->",
          f"{resolve_position('Nobody At All', 'PF', pos_map)})\n")

    # Suitor demo with INLINE mock data (real rosters come from build_ranked_projected).
    landscape = pd.DataFrame([
        ["BKN", "Brooklyn Nets",   38, 38,   "rebuild"],
        ["UTA", "Utah Jazz",       25, 25,   "rebuild"],
        ["DET", "Detroit Pistons", 12, 12,   "middle"],
        ["MIA", "Miami Heat",       0, 14.1, "contender"],
        ["BOS", "Boston Celtics",   0, 5.7,  "contender"],
    ], columns=["team", "team_name", "cap_space_M", "exception_M", "timeline"])

    rosters = pd.DataFrame([
        # team, player, pos (resolved 2K-style strings), barrett
        ["BKN", "Weak SF",         "SF/PF", 7.2], ["BKN", "Bench PF",     "PF",    5.1],
        ["UTA", "Lauri Markkanen", "PF/SF", 18.0],["UTA", "John Collins", "PF/C",  12.0],
        ["UTA", "Taylor Hendricks","SF/PF", 6.0],
        ["DET", "Tobias Harris",   "PF/SF", 9.0], ["DET", "Ausar Thompson","SG/SF", 8.5],
        ["MIA", "Mid Wing",        "SF",    8.0], ["MIA", "Backup F",     "PF",    6.5],
        ["BOS", "Jayson Tatum",    "SF/PF", 25.0],["BOS", "Jaylen Brown", "SG/SF", 20.0],
    ], columns=["team", "player", "pos", "barrett"])

    laravia_pos = resolve_position("Jake LaRavia", "PF", pos_map)   # 2K if known, else SF/PF
    print(f"LaRavia — model $10.3M, Barrett 9.5, {laravia_pos}:\n")
    for s in rank_suitors(10.3, 9.5, laravia_pos, rosters, landscape, n=5):
        print(f"  {s['team']:>3}  score {s['score']:>4}  slot {s['slot']}  — {s['reason']}")
    print("\nMax wing — $45M, Barrett 22, SF/PF (money gates hard):\n")
    res = rank_suitors(45.0, 22.0, "SF/PF", rosters, landscape, n=5)
    print("  " + ("\n  ".join(f"{s['team']} {s['reason']}" for s in res) if res else "(no team has $45M in room)"))
