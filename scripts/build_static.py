"""Build the static HoopsValue site: run the Barrett engine ONCE, headless, and
emit plain JSON + static HTML. No Streamlit, no websocket, no Python at request
time — the output is just files a CDN serves in tens of milliseconds.

Phase 1: site/data/home.json (hero cards) + site/data/players.json
(rankings/search) + a generated site/index.html with the hero data inlined.

Usage:  python -u scripts/build_static.py
"""
import sys
import json
import re
import shutil
import unicodedata
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from utils import build_ranked_projected, SEASONS, DEFAULT_MIN_THRESHOLD  # noqa: E402
import team_suitors as _ts  # noqa: E402

SITE = ROOT / "site"
DATA = SITE / "data"
DATA.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


CUR = SEASONS[0]
df = build_ranked_projected(CUR)
df = df[df["total_min"] >= DEFAULT_MIN_THRESHOLD].copy()
df_paid = df[df["salary"] > 0] if "salary" in df.columns else df

_pos_map = _ts.load_player_positions()


def _pos(name: str) -> str:
    return _ts._primary_position(_ts.resolve_position(name, "", _pos_map))


# ── Hero cards (best value / biggest steal / most overpaid) ───────────────────
# value_diff = salary - projected_salary (dollars): negative = underpaid (steal).
b = df.nsmallest(1, "score_rank").iloc[0]
s = df_paid.nsmallest(1, "value_diff").iloc[0]
o = df_paid.nlargest(1, "value_diff").iloc[0]

home = {
    "season": CUR,
    "n_players": int(len(df)),
    "best": {"name": str(b["Player"]), "slug": slugify(str(b["Player"])),
             "team": str(b["Team"]), "score": round(float(b["barrett_score"]), 1)},
    "steal": {"name": str(s["Player"]), "slug": slugify(str(s["Player"])),
              "team": str(s["Team"]), "below_m": round(abs(float(s["value_diff"])) / 1e6, 1)},
    "overpaid": {"name": str(o["Player"]), "slug": slugify(str(o["Player"])),
                 "team": str(o["Team"]), "above_m": round(float(o["value_diff"]) / 1e6, 1)},
}

# ── Players list (drives rankings + instant client-side search) ───────────────
players = []
for _, r in df.sort_values("score_rank").iterrows():
    sal = float(r["salary"]) if r.get("salary") else 0.0
    players.append({
        "name": str(r["Player"]),
        "slug": slugify(str(r["Player"])),
        "team": str(r["Team"]),
        "pos": _pos(str(r["Player"])),
        "score": round(float(r["barrett_score"]), 1),
        "rank": int(r["score_rank"]),
        "salary_m": round(sal / 1e6, 1),
        "value_m": round(float(r["value_diff"]) / 1e6, 1),
    })

(DATA / "home.json").write_text(json.dumps(home, indent=0))
(DATA / "players.json").write_text(json.dumps(players, separators=(",", ":")))


# ── Generate index.html with hero data inlined (instant first paint, SEO) ─────
_bm, _sm, _om = home["best"], home["steal"], home["overpaid"]
INDEX = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HoopsValue — NBA Contract Value, Every Player Ranked</title>
<meta name="description" content="Every NBA player since 1973 ranked by the Barrett Score — on-court production sized up against every paycheck. Find the steals, expose the overpays, predict any contract.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="/assets/style.css">
</head>
<body>
<header>
  <div class="wordmark"><span class="h">HO</span><span class="ball"></span><span class="h">PS</span><span class="v">VALUE</span></div>
  <div class="eyebrow">NBA Contract Value</div>
</header>
<div class="wrap">
  <p class="tagline">Every NBA player since 1973, ranked by the <b>Barrett Score</b>. On-court production sized up against every paycheck — find the steals, expose the overpays, settle the GOAT debate.</p>

  <div class="search">
    <input id="search" type="text" autocomplete="off" spellcheck="false"
           placeholder="Search any player — LeBron, Jokić, Wembanyama…">
    <div id="results" class="results"></div>
  </div>

  <div class="heroes">
    <a class="card best" href="/player/{_bm['slug']}.html">
      <div class="label">Best Player Right Now</div>
      <div class="name">{_bm['name']}</div>
      <div class="sub">{_bm['team']} · Barrett Score <b>{_bm['score']}</b></div>
    </a>
    <a class="card steal" href="/player/{_sm['slug']}.html">
      <div class="label">Biggest Steal</div>
      <div class="name">{_sm['name']}</div>
      <div class="sub">{_sm['team']} · <span class="pos">${_sm['below_m']}M</span> below market</div>
    </a>
    <a class="card over" href="/player/{_om['slug']}.html">
      <div class="label">Most Overpaid</div>
      <div class="name">{_om['name']}</div>
      <div class="sub">{_om['team']} · <span class="red">${_om['above_m']}M</span> above market</div>
    </a>
  </div>

  <div class="explore-label">Explore deeper</div>
  <div class="nav">
    <a class="navcard" href="/rankings.html" style="--accent:#d13b46"><div><h3>Current Rankings</h3><p>All {home['n_players']} qualified players ranked by Barrett Score this season.</p></div></a>
    <div class="navcard soon" style="--accent:#c79a3a"><div><h3>Legacy</h3><p>53 seasons of history: all-time greats, era leaderboards, draft classes.</p></div></div>
    <div class="navcard soon" style="--accent:#3b82c7"><div><h3>Team Analysis</h3><p>Which front offices get the most for their money? Payroll efficiency by team.</p></div></div>
    <div class="navcard soon" style="--accent:#3d6f52"><div><h3>Free Agents</h3><p>Everyone hitting the market this offseason — UFAs, RFAs, options.</p></div></div>
  </div>
</div>
<footer>hoopsvalue.com · Barrett Score · {home['season']} · data from NBA Stats API</footer>
<script src="/assets/app.js"></script>
</body>
</html>
"""
(SITE / "index.html").write_text(INDEX)

print(f"season   : {CUR}")
print(f"best     : {home['best']['name']:<24} score {home['best']['score']}")
print(f"steal    : {home['steal']['name']:<24} ${home['steal']['below_m']}M below market")
print(f"overpaid : {home['overpaid']['name']:<24} ${home['overpaid']['above_m']}M above market")
print(f"players  : {len(players)} -> {DATA / 'players.json'}")
