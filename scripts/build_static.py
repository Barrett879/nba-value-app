"""Build the static HoopsValue site: run the Barrett engine ONCE, headless, and
emit plain JSON + static HTML. No Streamlit, no websocket, no Python at request
time — the output is just files a CDN serves in tens of milliseconds.

Phase 1: site/data/home.json (hero cards) + site/data/players.json
(rankings/search) + a generated site/index.html with the hero data inlined.

Usage:  python -u scripts/build_static.py
"""
import sys
import os
import json
import re
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


# ── Phase 2: one static HTML per player (contract prediction + comps) ─────────
# Reuse the Contract Predictor page's OWN functions (headless, via prefix-exec)
# so the numbers are identical to the app — no reimplementation, no drift.
import html as _html  # noqa: E402

_SRC = (ROOT / "pages" / "Contract_Predictor.py").read_text().splitlines(keepends=True)
_cut = next(i for i, l in enumerate(_SRC) if l.startswith("_sb_col, _fa_col = st.columns("))
_ns = {"__name__": "cp", "__file__": str((ROOT / "pages" / "Contract_Predictor.py").resolve())}
exec(compile("".join(_SRC[:_cut]), "cp", "exec"), _ns)

_gpf = _ns["get_player_features"]
_pc = _ns["predict_contract"]
_fc = _ns["find_comparables"]
_scout = _ns["_scouting_take"]
_hist = _ns["load_historical_signings"](n_recent_pairs=3)
CONTRACT = _ns.get("CONTRACT_SEASON", CUR)
_MODEL = "HistGBM v2"

PLAYER_DIR = SITE / "player"
PLAYER_DIR.mkdir(parents=True, exist_ok=True)


def _m(x: float) -> str:
    return f"${x / 1e6:.1f}M"


def render_player(p: dict) -> bool:
    name = p["name"]
    try:
        f = _gpf(name, CUR)
        if not f:
            return False
        pred = _pc(f)
        comps = _fc(f, _hist, n=6) if not _hist.empty else None
    except Exception:
        return False

    predicted = float(pred["predicted"])
    cur_sal = float(f.get("salary") or 0)
    if cur_sal > 0:
        d = predicted - cur_sal
        pct = d / cur_sal * 100
        cls = "up" if d >= 0 else "down"
        vs = (f'<span class="{cls}">{"+" if d >= 0 else "−"}{_m(abs(d))}/yr</span>'
              f'<span class="vs-pct">{"+" if pct >= 0 else ""}{pct:.0f}% vs current deal ({_m(cur_sal)})</span>')
        cur_deal = f'<div class="cur-deal">Current deal: {_m(cur_sal)}</div>'
    else:
        vs = '<span class="vs-pct">No current salary on file</span>'
        cur_deal = ""

    floor = pred.get("supermax_tier_label") if pred.get("cba_floor_applied") else None
    floor_html = f'<div class="pred-floor">{_html.escape(str(floor))} floor</div>' if floor else ""

    median_html, comps_html = "", ""
    if comps is not None and not comps.empty:
        take = _scout(f, comps)
        median_html = (f'<div class="market">Market second opinion '
                       f'<span class="muted">(weighted median of comparable signings)</span> '
                       f'<b>{_m(float(take["median"]))}</b></div>')
        rows = "".join(
            f'<tr><td class="cn"><a href="/player/{slugify(str(r["Player"]))}.html">'
            f'{_html.escape(str(r["Player"]))}</a></td>'
            f'<td>{_html.escape(str(r.get("pos_primary", "")))}</td>'
            f'<td>{_html.escape(str(r.get("signed_in", "")))}</td>'
            f'<td class="cd">{_m(float(r["salary_curr"]))}</td></tr>'
            for _, r in comps.iterrows())
        comps_html = (f'<section class="comps"><h2>Comparable signings</h2>'
                      f'<p class="comps-sub">Closest matches on trailing Barrett + age + position.</p>'
                      f'<table class="comps-table"><thead><tr><th>Player</th><th>Pos</th>'
                      f'<th>Signed</th><th>Deal</th></tr></thead><tbody>{rows}</tbody></table></section>')

    pos = _pos(name)
    draft = ""
    if f.get("draft_pick"):
        draft = f' · Pick #{int(f["draft_pick"])}'
        if f.get("draft_year"):
            draft += f' ({int(f["draft_year"])})'
    age = f' · Age {int(f["age"])}' if f.get("age") else ""
    nm = _html.escape(name)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{nm} — Next Contract Prediction · HoopsValue</title>
<meta name="description" content="{nm}'s projected {CONTRACT} contract: {_m(predicted)}/yr at next season's cap, with comparable signings and market value.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="/assets/style.css">
</head>
<body>
<header class="subhead">
  <a class="brand" href="/"><span class="h">HO</span><span class="ball"></span><span class="h">PS</span><span class="v">VALUE</span></a>
  <nav class="topnav"><a href="/rankings.html">Rankings</a></nav>
</header>
<main class="wrap player">
  <div class="eyebrow">Projected {CONTRACT} contract</div>
  <h1 class="pname">{nm}<span class="chip">Score {p['score']} · #{p['rank']}</span></h1>
  <div class="pmeta">{_html.escape(str(f.get('current_team') or p['team']))} · {CUR}{age} · {pos}{draft}</div>
  {cur_deal}
  <div class="pred-number">{_m(predicted)}<span class="yr">/yr</span></div>
  {floor_html}
  <div class="pred-vs">{vs}</div>
  {median_html}
  {comps_html}
  <p class="disclaimer">Model: {_MODEL}. Projection is for the player's next contract priced at the {CONTRACT} cap. Informational only — not financial advice.</p>
</main>
<footer>hoopsvalue.com · Barrett Score · {CUR}</footer>
</body>
</html>
"""
    (PLAYER_DIR / f"{p['slug']}.html").write_text(page)
    return True


if os.environ.get("SKIP_PLAYERS") == "1":
    print("player pages: SKIPPED (SKIP_PLAYERS=1)")
else:
    _built = sum(render_player(p) for p in players)
    print(f"player pages: {_built}/{len(players)} -> {PLAYER_DIR}")


# ── Phase 3: rankings page (pre-rendered rows + client sort/filter) ───────────
def _valcell(v: float) -> str:
    cls, sign = ("over", "+") if v > 0 else (("steal", "−") if v < 0 else ("", ""))
    return f'<td class="num val {cls}" data-v="{v}">{sign}${abs(v):.1f}M</td>'


_rrows = "".join(
    "<tr>"
    f'<td class="rk num" data-v="{p["rank"]}">{p["rank"]}</td>'
    f'<td class="pn"><a href="/player/{p["slug"]}.html">{_html.escape(p["name"])}</a></td>'
    f'<td class="hide-sm" data-t="{_html.escape(p["team"])}">{_html.escape(p["team"])}</td>'
    f'<td data-t="{p["pos"]}">{p["pos"]}</td>'
    f'<td class="sc num" data-v="{p["score"]}">{p["score"]}</td>'
    f'<td class="num hide-sm" data-v="{p["salary_m"]}">${p["salary_m"]:.1f}M</td>'
    f"{_valcell(p['value_m'])}"
    "</tr>"
    for p in players)

RANK = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Current NBA Rankings by Barrett Score · HoopsValue</title>
<meta name="description" content="Every qualified NBA player this season ranked by the Barrett Score — production vs pay, the steals and the overpays.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="/assets/style.css">
</head>
<body>
<header class="subhead">
  <a class="brand" href="/"><span class="h">HO</span><span class="ball"></span><span class="h">PS</span><span class="v">VALUE</span></a>
  <nav class="topnav"><a href="/">Home</a></nav>
</header>
<main class="wrap rankings">
  <h1>Current Rankings</h1>
  <p class="sub">All {len(players)} qualified players, {CUR} — ranked by Barrett Score. Tap a column to sort.</p>
  <input id="rankfilter" class="rank-filter" type="text" placeholder="Filter players…" autocomplete="off" spellcheck="false">
  <table class="rank-table" id="ranktable">
    <thead><tr>
      <th class="num" data-k="num">#</th>
      <th data-k="text">Player</th>
      <th class="hide-sm" data-k="text">Team</th>
      <th data-k="text">Pos</th>
      <th class="num" data-k="num">Score</th>
      <th class="num hide-sm" data-k="num">Salary</th>
      <th class="num" data-k="num">Value vs Pay</th>
    </tr></thead>
    <tbody>{_rrows}</tbody>
  </table>
</main>
<footer>hoopsvalue.com · Barrett Score · {CUR}</footer>
<script src="/assets/rankings.js"></script>
</body>
</html>
"""
(SITE / "rankings.html").write_text(RANK)
print(f"rankings : {len(players)} rows -> rankings.html")

print(f"season   : {CUR}")
print(f"best     : {home['best']['name']:<24} score {home['best']['score']}")
print(f"steal    : {home['steal']['name']:<24} ${home['steal']['below_m']}M below market")
print(f"overpaid : {home['overpaid']['name']:<24} ${home['overpaid']['above_m']}M above market")
print(f"players  : {len(players)} -> {DATA / 'players.json'}")
