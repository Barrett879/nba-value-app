"""Smoke test the HistGBM v2 integration end-to-end on real players.

Calls the same helpers Contract_Predictor.py uses, without running the
full Streamlit page. Verifies that:
  - The model loads from cache/contract_histgbm_v2.joblib
  - Feature extraction works for current-season players
  - Predictions land in plausible ranges
  - CBA cap/floor post-processing fires correctly
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import joblib

from utils import (
    SEASONS, SALARY_CAP_M, fetch_league_stats, build_ranked_projected,
    fetch_player_full_career, fetch_player_positions_detailed,
    position_to_bucket, normalize, HEALTHY_SEASON_GP,
    get_player_draft_info, get_max_contract_eligibility,
    fetch_all_nba_selections, get_all_nba_in_window,
    fetch_rookie_scale_players,
)

CURRENT_SEASON = SEASONS[0]
MODEL_PATH = Path(__file__).parent.parent / "models" / "contract_histgbm_v2.joblib"


def get_features(player_name: str, season: str = CURRENT_SEASON) -> dict | None:
    ranked = build_ranked_projected(season)
    if ranked.empty:
        return None
    name_norm = normalize(player_name)
    mask = ranked["Player"].apply(normalize) == name_norm
    if not mask.any():
        return None
    row = ranked[mask].iloc[0]

    raw = fetch_league_stats(season, "Regular Season")
    age = None
    if not raw.empty and "AGE" in raw.columns:
        age_lookup = dict(zip(raw["PLAYER_ID"], raw["AGE"]))
        age = age_lookup.get(int(row["PLAYER_ID"]))

    # Career trailing stats.
    barrett_3yr_simple = None
    gp_3yr_simple = None
    career_barrett = None
    try:
        career = fetch_player_full_career(player_name)
        if not career.empty:
            healthy = career[career["GP"] >= HEALTHY_SEASON_GP]
            pool = healthy if not healthy.empty else career
            recent = pool.tail(3)
            barrett_3yr_simple = float(recent["Barrett Score"].mean())
            gp_3yr_simple = float(recent["GP"].mean())
            weights_full = [0.20, 0.30, 0.50]
            weights = weights_full[-len(recent):]
            career_barrett = float(
                (recent["Barrett Score"].values * weights).sum() / sum(weights)
            )
    except Exception:
        pass
    if career_barrett is None:
        career_barrett = float(row["barrett_score"])

    # Effective rank.
    cur_rate = ranked["barrett_score"].sort_values(ascending=False).values
    cur_sal = ranked["salary"].sort_values(ascending=False).values
    effective_rank = int((cur_rate > career_barrett).sum()) + 1
    capped_rank = min(effective_rank, len(cur_sal)) - 1
    career_base_proj = float(cur_sal[capped_rank])

    # All-NBA.
    try:
        all_nba_3yr_count = len(get_all_nba_in_window(player_name, season, 3))
    except Exception:
        all_nba_3yr_count = 0

    # Position.
    pos_bucket = "Unknown"
    try:
        detailed = fetch_player_positions_detailed(season, cache_v=3)
        d = detailed.get(name_norm, "Unknown")
        if d != "Unknown":
            pos_bucket = position_to_bucket(d)
    except Exception:
        pass

    # CBA eligibility.
    try:
        elig = get_max_contract_eligibility(player_name, season)
    except Exception:
        elig = {"service_years": 0, "max_pct": 0.35,
                "qualifying": False, "supermax_tier": "",
                "current_team": "", "team_tenure": 0, "recent_all_nba": []}

    return {
        "name": row["Player"],
        "age": age,
        "position": pos_bucket,
        "barrett_score": float(row["barrett_score"]),
        "career_barrett": career_barrett,
        "score_rank": int(row["score_rank"]),
        "effective_rank": effective_rank,
        "career_base_proj": career_base_proj,
        "barrett_3yr_simple": barrett_3yr_simple,
        "gp_3yr_simple": gp_3yr_simple,
        "eff_adj": float(row.get("efficiency_adj", 0) or 0),
        "d_lebron": float(row.get("d_lebron", 0) or 0),
        "all_nba_3yr": all_nba_3yr_count,
        "gp": int(row.get("GP", 0) or 0),
        "mpg": float(row.get("MPG", 0) or 0),
        "salary": float(row.get("salary", 0) or 0),
        "service_years": elig["service_years"],
        "max_pct": elig["max_pct"],
        "supermax_eligible": elig["qualifying"] and elig["supermax_tier"] in
                              ("Designated Vet (35%)", "Designated Rookie (30%)"),
        "supermax_tier": elig["supermax_tier"],
    }


def predict_histgbm(features: dict, target_season: str = CURRENT_SEASON) -> dict:
    art = joblib.load(MODEL_PATH)
    model = art["model"]
    cap = SALARY_CAP_M.get(target_season, 154.6) * 1_000_000

    barrett = float(features.get("career_barrett") or 0)
    barrett_single = float(features.get("barrett_score") or 0)
    barrett_3yr = float(features.get("barrett_3yr_simple") or barrett or barrett_single)
    score_rank = float(features.get("score_rank") or 999)
    eff_adj = float(features.get("eff_adj") or 0)
    d_lebron = float(features.get("d_lebron") or 0)
    gp = float(features.get("gp") or 0)
    gp_3yr = float(features.get("gp_3yr_simple") or gp)
    age = float(features.get("age") or 25)
    salary_prev_pct = float(features.get("salary") or 0) / cap
    career_base_proj_pct = float(features.get("career_base_proj") or 0) / cap
    yrs = float(features.get("service_years") or 0)
    all_nba_3yr = float(features.get("all_nba_3yr") or 0)
    growth = (barrett_single / barrett_3yr) if barrett_3yr > 0 else 1.0
    pos = features.get("position", "Unknown")

    X = np.array([[
        barrett, barrett_single, barrett_3yr, score_rank,
        eff_adj, d_lebron, gp, gp_3yr, age,
        salary_prev_pct, career_base_proj_pct, yrs,
        all_nba_3yr, growth,
        age ** 2, barrett ** 2, np.log1p(score_rank), yrs ** 2,
        1.0 if yrs >= 7 else 0.0,
        1.0 if yrs >= 10 else 0.0,
        1.0 if pos == "Guard" else 0.0,
        1.0 if pos == "Forward" else 0.0,
    ]])
    pred_pct = float(np.clip(model.predict(X)[0], 0.001, 0.45))
    raw = pred_pct * cap

    # CBA cap.
    max_pct = float(features.get("max_pct") or 0.35)
    cba_max = max_pct * cap
    final = min(raw, cba_max)
    cap_applied = final < raw
    floor_applied = False
    if features.get("supermax_eligible") and age <= 32 and final < cba_max:
        final = cba_max
        floor_applied = True

    return {"raw": raw, "final": final, "pred_pct": pred_pct,
            "cap_applied": cap_applied, "floor_applied": floor_applied,
            "cba_max": cba_max}


def smoke_test_player(name: str) -> None:
    f = get_features(name)
    if f is None:
        print(f"{name}: NOT FOUND\n")
        return
    p = predict_histgbm(f)
    print(f"{name}  (age {f['age']}, {f['position']}, rank #{f['score_rank']}, {f['service_years']}yrs)")
    print(f"  Inputs: career_barrett={f['career_barrett']:.1f}, prior_sal=${f['salary']/1e6:.1f}M, "
          f"all_NBA_3yr={f['all_nba_3yr']}")
    print(f"  → ML raw:  ${p['raw']/1e6:.1f}M ({p['pred_pct']*100:.1f}% of cap)")
    print(f"  → Final:   ${p['final']/1e6:.1f}M  "
          f"(cap_applied={p['cap_applied']}, floor_applied={p['floor_applied']})")
    print()


if __name__ == "__main__":
    for name in ["Luka Doncic", "LeBron James", "Devin Booker",
                 "Tyrese Haliburton", "Jalen Brunson", "Bam Adebayo",
                 "Brandon Ingram", "Mikal Bridges", "Anfernee Simons"]:
        smoke_test_player(name)
