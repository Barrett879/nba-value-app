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
ROTATION_DEPTH = 3      # starter/rotation boundary: slot 0=starter, 1=key sub, 2=depth (labels)
INTEREST_DEPTH = 5      # in his market if he'd be ~top-5 at the position (own team gets +2 leash)

# Fit-scaled offers: a team pays toward full value for a starter, less for a depth
# add. Indexed by slot (how many of their incumbents out-rate him at the spot).
_FIT_FACTOR = {0: 1.00, 1: 0.90, 2: 0.78, 3: 0.62, 4: 0.50}

# Apron-implied largest exception a team can use, $M (2026-27 CBA). Derived from
# the CSV's `top_exception` so the file stays principled: set the TOOL, not the $.
_EXC_BY_TOOL = {
    "nt_mle":   15.05,   # under the first apron -> full non-taxpayer MLE
    "room_exc":  8.78,   # cap-room team's post-room exception
    "tp_mle":    6.07,   # over the first apron -> taxpayer MLE only
    "bae":       5.85,   # bi-annual exception
    "min":       2.30,   # over the SECOND apron -> minimum signings only
    "cap_room": 15.05,   # uses cap space; nominal exception
}


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


def _read_as_of(path: Path | str) -> str:
    """Pull the `# as_of:` freshness stamp from the top of the cap CSV, if present."""
    try:
        with open(path) as fh:
            for line in fh:
                if line.lower().lstrip().startswith("# as_of:"):
                    return line.split(":", 1)[1].strip()
                if not line.lstrip().startswith("#"):
                    break
    except Exception:
        pass
    return ""


def load_team_landscape(path: Path | str = CSV_PATH) -> pd.DataFrame:
    """Load the cap/context table. The largest exception each team can use is
    derived from `top_exception` (apron-implied: nt_mle $15.05M under the first
    apron, tp_mle $6.07M over it, min $2.3M over the second apron) so the file
    stays principled and easy to keep current — set the TOOL, not the dollar. An
    explicit `exception_M` still overrides. A `# as_of:` line at the top stamps
    the data's age (surfaced in the UI)."""
    df = pd.read_csv(path, comment="#")
    df["cap_space_M"] = pd.to_numeric(df["cap_space_M"], errors="coerce").fillna(0.0)
    tool = (df.get("top_exception", pd.Series(["nt_mle"] * len(df)))
            .fillna("nt_mle").astype(str).str.strip().str.lower())
    derived = tool.map(_EXC_BY_TOOL).fillna(DEFAULT_NT_MLE)
    if "exception_M" in df.columns:
        df["exception_M"] = pd.to_numeric(df["exception_M"], errors="coerce").fillna(derived)
    else:
        df["exception_M"] = derived
    df["timeline"] = df.get("timeline", "").fillna("").astype(str)
    df.attrs["as_of"] = _read_as_of(path)
    return df


def roster_need(target_score: float, target_pos: str, team_roster: pd.DataFrame) -> dict:
    """Where `target` slots into ONE team's depth chart at his position.

    team_roster: that team's players, columns [player, pos, barrett].
    Returns {slot, displaces, displaces_score, gap, depth_here}:
      slot 0 = better than their best at the spot (would start)
      slot 1 = beats their 2nd ... slot = how many incumbents out-rate him.
    No affordability or interest cutoff here — rank_suitors decides who's a
    realistic suitor and at what price."""
    elig = _eligible_positions(target_pos)              # the spots HE can play
    # An incumbent competes with him when their eligible sets OVERLAP — so a 2K
    # SG/SF wing blocks an SG or an SF target, but a pure PG never blocks a center.
    inc = (team_roster[team_roster["pos"].map(lambda p: bool(_eligible_positions(p) & elig))]
           .sort_values("barrett", ascending=False).reset_index(drop=True))
    scores = inc["barrett"].astype(float).tolist()
    slot = sum(1 for s in scores if s > target_score)   # how many incumbents are better
    if slot < len(scores):                              # he leapfrogs the incumbent at this slot
        d = inc.iloc[slot]
        return {"slot": slot, "displaces": d["player"], "displaces_score": float(d["barrett"]),
                "gap": target_score - float(d["barrett"]), "depth_here": len(scores)}
    return {"slot": slot, "displaces": None, "displaces_score": None,   # fills an open spot
            "gap": 0.0, "depth_here": len(scores)}


# Desire = how keen a team of each TIMELINE is to PURSUE a player of each ARCHETYPE
# (0–1), applied to the ranking (not the offer). DERIVED FROM DATA, not hand-tuned:
# 917 "new-team" free-agent signings (2013–2025), each tiered by its signing team's
# actual win% that season and cap-controlled via archetype-mix lift (a rebuilder's
# cap-driven signing *volume* divides out, leaving preference). Rebuilt by
# scripts/learn_desire_weights.py. What the data says:
#   - Vets favor title teams HARD (1.00 vs ~0.5 elsewhere — "contenders vs the field").
#   - Youth flows down the ladder, monotonically (rebuild 1.00 -> title 0.57).
#   - Prime role players have ~no tier preference (they sign everywhere, ~0.85-1.0).
#   - Star row is hand-smoothed — only 71 star signings, too thin to trust raw.
_DESIRE = {
    #            star   young  prime   vet
    "title":   {"star": 0.95, "young": 0.57, "prime": 0.87, "vet": 1.00},
    "playoff": {"star": 1.00, "young": 0.74, "prime": 1.00, "vet": 0.51},
    "bye":     {"star": 0.85, "young": 0.95, "prime": 0.89, "vet": 0.53},
    "rebuild": {"star": 0.80, "young": 1.00, "prime": 0.85, "vet": 0.59},
}
_DESIRE_DEFAULT = "playoff"   # an unclassified team -> neutral win-now-ish


def _timeline_key(timeline: str) -> str:
    """Map a CSV timeline label (incl. legacy contender/middle) to a desire tier."""
    tl = (timeline or "").lower()
    if "title" in tl or "conten" in tl:   return "title"
    if "playoff" in tl or "middle" in tl: return "playoff"
    if "bye" in tl:                        return "bye"
    if "rebuild" in tl:                    return "rebuild"
    return _DESIRE_DEFAULT


def desire_weight(timeline: str, age, value_M: float) -> float:
    """How badly a team of this TIMELINE would PURSUE a player of this age/value —
    the 'do they want him', separate from 'can they pay'. Returns a 0–1 multiplier
    on the ranking. Title teams chase win-now production; rebuilders chase youth and
    pass on aging vets; bye-year teams are selective; playoff teams are pragmatic."""
    a = float(age) if age is not None else 27.0
    if   value_M >= 22.0: arche = "star"    # a difference-maker — most teams want him
    elif a < 25:          arche = "young"   # youth / upside
    elif a > 31:          arche = "vet"      # aging
    else:                 arche = "prime"    # prime role player
    return _DESIRE[_timeline_key(timeline)][arche]


def rank_suitors(price_M: float, target_barrett: float, target_pos: str,
                 rosters: pd.DataFrame, landscape: pd.DataFrame | None = None,
                 n: int = 6, incumbent_team: str | None = None,
                 age: float | None = None, is_rfa: bool = False,
                 skill_fit: dict | None = None) -> list[dict]:
    """His projected free-agent market: the teams most likely to pursue him, each
    at the price THEY would realistically offer.

    Offer  = model value, scaled by FIT (a starter-level need pays toward full
             value, a depth add pays less) and capped by the team's biggest tool
             (cap room / apron-implied exception). His current team can match any
             number via Bird rights, and — if he's an RFA — can match ANY offer.
    Rank   = offer × DESIRE, where desire models whether a team of that timeline
             would actually pursue a player of his age/value (so a cap-rich tanker
             doesn't outrank a motivated win-now team for a prime role vet).

    rosters: live per-player table [team, player, pos, barrett].
    incumbent_team / age / is_rfa: his current team, age, restricted-FA flag.
    skill_fit: optional {team: {fit, need}} from skill_fit_scores() — nudges teams
        toward an FA who supplies a skill (shooting / rebounding / ...) they lack.
    """
    if landscape is None:
        landscape = load_team_landscape()
    inc_norm = _normalize(incumbent_team or "")
    out = []
    for _, t in landscape.iterrows():
        is_inc = bool(inc_norm) and _normalize(str(t["team"])) == inc_norm
        need = roster_need(target_barrett, target_pos, rosters[rosters["team"] == t["team"]])
        # In the market only if he'd be a rotation-level fit (own team gets +2 leash).
        if need["slot"] >= INTEREST_DEPTH + (2 if is_inc else 0):
            continue
        cap = float(t["cap_space_M"]); exc = float(t["exception_M"])
        tool = max(cap, exc, price_M if is_inc else 0.0)    # incumbent: Bird rights match any $
        fit = 1.0 if is_inc else _FIT_FACTOR.get(need["slot"], 0.45)
        offer = min(price_M, tool, price_M * fit)           # fair value × fit, capped by tool
        if offer < 1.0:
            continue
        has_room = cap + 1e-6 >= offer
        if   has_room:      tool_label = "cap room"
        elif is_inc:        tool_label = "Bird rights"
        elif exc >= 15.0:   tool_label = "mid-level exception"
        elif exc >= 6.0:    tool_label = "taxpayer MLE"
        else:               tool_label = "minimum"
        tl = str(t.get("timeline", "")).strip().lower()
        des = 1.0 if is_inc else desire_weight(tl, age, price_M)
        sf = (skill_fit or {}).get(t["team"]) or {"fit": 0.5, "need": None}
        # skill fit nudges rank x(1-W/2)..x(1+W/2) — favours teams who lack his skill
        rank = offer * des * ((1.0 - SKILL_WEIGHT / 2.0) + SKILL_WEIGHT * float(sf["fit"]))
        if is_inc and is_rfa:
            rank = 1e9                                       # RFA: his team can match anything
        out.append({
            "team":         t["team"],
            "team_name":    t.get("team_name", t["team"]),
            "offer_M":      round(offer, 1),
            "slot":         need["slot"],
            "is_incumbent": is_inc,
            "tool":         tool_label,
            "reason":       _reason(need, tl, target_barrett, is_inc, is_rfa, sf.get("need")),
            "_rank":        rank,
        })
    out.sort(key=lambda d: (-d["_rank"], d["slot"]))
    for d in out:
        d.pop("_rank", None)
    return out[:n]


def _reason(need: dict, timeline: str, target_barrett: float, is_incumbent: bool = False,
            is_rfa: bool = False, skill_need: str | None = None) -> str:
    if is_incumbent:
        fit = "can match any offer (restricted FA)" if is_rfa else "could re-sign him (Bird rights)"
    elif need["displaces"] is not None:
        role = {0: "would start over", 1: "upgrades over", 2: "rotation piece over"}.get(
            need["slot"], "depth piece over")
        fit = f"{role} {need['displaces']} ({need['displaces_score']:.1f} vs {target_barrett:.1f})"
    else:
        fit = "fills an open spot at the position"
    extra = f"fills their {skill_need} need" if skill_need else ""
    return " · ".join(filter(None, [fit, extra, timeline]))


# ── Skill-fit layer ─────────────────────────────────────────────────────────
# Beyond "needs a forward" — does a team need a forward who can SHOOT / REBOUND /
# PLAYMAKE / DEFEND? Built from pace- and volume-robust rates, so a chucking tank
# team doesn't read as a good shooting team (verified: such teams correctly flag as
# NEEDING shooting, and Jokic's Denver lands #1 in playmaking).
SKILL_CATS = ("shooting", "rebounding", "playmaking", "defense")
SKILL_WEIGHT = 0.5          # rank multiplier swings x(1-W/2)..x(1+W/2) by skill fit
_SKILL_MIN_GP = 30          # rotation filter for the percentile pools


def build_team_skills(box: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    """Per-team skill PERCENTILE (0 = league-worst at it = biggest need, 1 = best),
    from rates: shooting = team 3P%, rebounding = minutes-weighted REB%, playmaking =
    AST/TO, defense = minutes-weighted DEF_RATING (inverted so higher = better)."""
    gb = box.groupby("TEAM_ABBREVIATION")
    shooting = gb["FG3M"].sum() / gb["FG3A"].sum().replace(0, pd.NA)
    playmaking = gb["AST"].sum() / gb["TOV"].sum().replace(0, pd.NA)
    a = adv if "TEAM_ABBREVIATION" in adv.columns else adv.merge(
        box[["PLAYER_ID", "TEAM_ABBREVIATION"]], on="PLAYER_ID", how="left")

    def _wmean(col):
        d = a.dropna(subset=[col, "MIN"]).copy()
        d["_n"] = d[col] * d["MIN"]
        g = d.groupby("TEAM_ABBREVIATION")
        return g["_n"].sum() / g["MIN"].sum()

    team = pd.DataFrame({"shooting": shooting, "rebounding": _wmean("REB_PCT"),
                         "playmaking": playmaking, "defense": -_wmean("DEF_RATING")})
    return team.rank(pct=True)


def player_skills(player_id, box: pd.DataFrame, adv: pd.DataFrame) -> dict:
    """The player's STRENGTH percentile (0-1) per skill among rotation players. A
    non-shooter (low 3PA volume) gets a low shooting score, not a flattering %.
    box/adv are PER-GAME league tables (MIN = minutes per game)."""
    b = box[box["GP"] >= _SKILL_MIN_GP].copy()
    vol = b[b["FG3A"] >= 2.0].copy()                        # 2+ three-attempts per game
    vol["_sk"] = (vol["FG3M"] / vol["FG3A"]).rank(pct=True)
    shoot = dict(zip(vol["PLAYER_ID"], vol["_sk"]))
    a = adv[adv["MIN"] >= 15.0].copy()                      # rotation minutes (per game)
    reb = dict(zip(a["PLAYER_ID"], a["REB_PCT"].rank(pct=True)))
    ast = dict(zip(a["PLAYER_ID"], a["AST_PCT"].rank(pct=True)))
    dfn = dict(zip(a["PLAYER_ID"], (-a["DEF_RATING"]).rank(pct=True)))
    return {"shooting":   float(shoot.get(player_id, 0.25)),
            "rebounding": float(reb.get(player_id, 0.50)),
            "playmaking": float(ast.get(player_id, 0.50)),
            "defense":    float(dfn.get(player_id, 0.50))}


def skill_fit_scores(player_sk: dict, team_skills: pd.DataFrame) -> dict:
    """{team: {'fit': 0..1, 'need': category|None}} — how well the player fills each
    team's deficits: fit = sum(player_strength x team_deficit), deficit = 1 - percentile,
    min-maxed 0..1 across teams. 'need' = the category he most helps, flagged only when
    he's genuinely strong there (>= 0.6) and the team genuinely weak (deficit >= 0.45)."""
    raw, need = {}, {}
    for tm, row in team_skills.iterrows():
        contribs = []
        for c in SKILL_CATS:
            v = row.get(c)
            if v is None or v != v:                        # NaN guard
                continue
            deficit = 1.0 - float(v)
            contribs.append((player_sk.get(c, 0.5) * deficit, c, deficit))
        if not contribs:
            raw[tm], need[tm] = 0.0, None
            continue
        raw[tm] = sum(x[0] for x in contribs)
        best = max(contribs, key=lambda x: x[0])
        need[tm] = best[1] if (player_sk.get(best[1], 0) >= 0.6 and best[2] >= 0.45) else None
    vals = list(raw.values()) or [0.0]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {tm: {"fit": (raw[tm] - lo) / span, "need": need[tm]} for tm in raw}


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
        ["UTA", "Utah Jazz",       25, 25,   "bye"],
        ["DET", "Detroit Pistons", 12, 12,   "title"],
        ["MIA", "Miami Heat",       0, 14.1, "playoff"],
        ["BOS", "Boston Celtics",   0, 5.7,  "title"],
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
    print(f"LaRavia — value $10.3M, Barrett 9.5, {laravia_pos}, age 24, UFA (incumbent MIA):\n")
    for s in rank_suitors(10.3, 9.5, laravia_pos, rosters, landscape, n=6,
                          incumbent_team="MIA", age=24, is_rfa=False):
        tag = " *re-sign*" if s["is_incumbent"] else ""
        print(f"  {s['team']:>3}  ${s['offer_M']:>4}M  slot {s['slot']}  [{s['tool']:<19}] {s['reason']}{tag}")

    print("\nSame guy as a 23-yo RFA on BKN — his team can match ANY offer (jumps to #1):\n")
    for s in rank_suitors(10.3, 9.5, laravia_pos, rosters, landscape, n=4,
                          incumbent_team="BKN", age=23, is_rfa=True):
        tag = " *MATCH*" if s["is_incumbent"] else ""
        print(f"  {s['team']:>3}  ${s['offer_M']:>4}M  [{s['tool']:<19}] {s['reason']}{tag}")

    print("\nAging vet — value $10.3M, age 33: rebuilders (BKN/UTA) cool off, win-now stays keen:\n")
    for s in rank_suitors(10.3, 9.5, laravia_pos, rosters, landscape, n=6, age=33):
        print(f"  {s['team']:>3}  ${s['offer_M']:>4}M  [{s['tool']:<19}] {s['reason']}")
