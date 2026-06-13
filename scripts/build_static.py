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

import pandas as pd  # noqa: E402
from utils import build_ranked_projected, SEASONS, DEFAULT_MIN_THRESHOLD, normalize  # noqa: E402
import team_suitors as _ts  # noqa: E402

ONLY = os.environ.get("ONLY", "")  # build just this slug (fast template iteration)

SITE = ROOT / "site"
DATA = SITE / "data"
DATA.mkdir(parents=True, exist_ok=True)

# The interactive tools (predictor, team analysis, free agents, legacy) can't be
# static — they stay on the Streamlit app. When this static site becomes the
# hoopsvalue.com front door, these cards point at the app's own subdomain.
# Override with APP_BASE=… if you host the Streamlit app somewhere else.
APP_BASE = os.environ.get("APP_BASE", "https://app.hoopsvalue.com")


# ── Shared chrome: top nav, theme toggle, footer (port of the app's design) ───
# Plain strings and helpers (NOT f-string templates) so inner braces in the
# JS/SVG never collide with template interpolation.
_THEME_BOOT = (
    '<script>(function(){try{if(localStorage.getItem("hv-theme")==="dark")'
    'document.documentElement.setAttribute("data-theme","dark");}catch(e){}})();</script>'
)

_MOON_SVG = ('<svg class="ico-moon" viewBox="0 0 24 24" aria-hidden="true">'
             '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>')
_SUN_SVG = ('<svg class="ico-sun" viewBox="0 0 24 24" aria-hidden="true">'
            '<circle cx="12" cy="12" r="5"/>'
            '<path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4'
            'M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" stroke="currentColor" stroke-width="2" '
            'fill="none" stroke-linecap="round"/></svg>')

# Same pages, same order as the app's _NAV_PAGES; static pages link locally,
# interactive ones go to the app.
_NAV_LINKS = [
    ("Current Rankings",    "/rankings.html"),
    ("Search Player",       APP_BASE + "/Search"),
    ("Legacy",              APP_BASE + "/Legacy"),
    ("Team Analysis",       APP_BASE + "/Team_Analysis"),
    ("Contract Predictor",  APP_BASE + "/Contract_Predictor"),
    ("Current Free Agents", APP_BASE + "/Free_Agent_Class"),
]


def _nav(active: str = "") -> str:
    links = '<a class="home-link" href="/">Home</a><span class="divider">|</span>'
    for label, url in _NAV_LINKS:
        cls = ' class="active"' if label == active else ""
        links += f'<a{cls} href="{url}">{label}</a>'
    return (f'<div class="top-nav">{links}</div>'
            f'<button class="theme-btn" onclick="hvToggleTheme()" aria-label="Toggle dark mode">'
            f'{_MOON_SVG}{_SUN_SVG}</button>')


_X_PATH = ("M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83"
           "L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z")


def _footer() -> str:
    return (
        '<div class="hv-footer">'
        '<div class="hv-foot-top">Enjoying HoopsValue? <a href="mailto:contact@hoopsvalue.com">Share feedback</a></div>'
        '<div class="hv-foot-disc">HoopsValue.com is an independent project and is not affiliated with, '
        'endorsed by, or sponsored by the National Basketball Association.</div>'
        '<div class="hv-foot-rule"></div>'
        '<div class="hv-foot-bottom">'
        '<div class="hv-foot-left">© 2026 HoopsValue.com. All rights reserved.</div>'
        '<div class="hv-foot-right">'
        '<a href="mailto:contact@hoopsvalue.com">contact@hoopsvalue.com</a>'
        '<span class="sep">|</span>'
        '<a href="https://x.com/HoopsValue" target="_blank" rel="noopener">@HoopsValue</a>'
        '<a class="hv-foot-ico" href="https://x.com/HoopsValue" target="_blank" rel="noopener" '
        f'aria-label="HoopsValue on X"><svg viewBox="0 0 24 24"><path d="{_X_PATH}"/></svg></a>'
        '</div></div></div>'
    )


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def _relativize(s: str, depth: int) -> str:
    """Rewrite root-absolute refs ("/assets/…", "/player/…") to relative so the
    site works at any base path — a GitHub Pages project subpath now, the domain
    root (hoopsvalue.com) later. depth = directory levels below the site root."""
    pre = "../" * depth
    s = s.replace('href="/"', f'href="{pre or "./"}"')
    for seg in ("assets/", "player/", "rankings.html"):
        s = s.replace(f'"/{seg}', f'"{pre}{seg}')
    return s


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
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Source+Sans+3:wght@400;600;700&display=swap">
<link rel="stylesheet" href="/assets/style.css">
{_THEME_BOOT}
</head>
<body>
{_nav()}
<header class="hero-head">
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
    <a class="navcard" href="{APP_BASE}/Contract_Predictor" style="--accent:#16b8a6"><div><h3>Contract Predictor</h3><p>What any player would command on a new deal today — or build out a team's offseason from the front office chair.</p></div></a>
    <a class="navcard" href="{APP_BASE}/Team_Analysis" style="--accent:#3b82c7"><div><h3>Team Analysis</h3><p>Which front offices get the most for their money? Payroll efficiency by team.</p></div></a>
    <a class="navcard" href="{APP_BASE}/Free_Agent_Class" style="--accent:#3d6f52"><div><h3>Free Agents</h3><p>Everyone hitting the market this offseason — UFAs, RFAs, options.</p></div></a>
    <a class="navcard" href="{APP_BASE}/Legacy" style="--accent:#c79a3a"><div><h3>Legacy</h3><p>53 seasons of history: all-time greats, era leaderboards, draft classes.</p></div></a>
  </div>
</div>
{_footer()}
<script src="/assets/theme.js"></script>
<script src="/assets/app.js"></script>
</body>
</html>
"""
(SITE / "index.html").write_text(_relativize(INDEX, 0))

# Custom domain for GitHub Pages — written ONLY on a production deploy
# (PRODUCTION=1). Writing a CNAME makes GitHub 301-redirect the github.io
# preview URL to hoopsvalue.com, so we keep it off for normal deploys (preview
# stays live at github.io) and emit it only at cutover. Cutover is: point
# hoopsvalue.com's DNS at GitHub Pages, then run
#   PRODUCTION=1 bash scripts/deploy_ghpages.sh
if os.environ.get("PRODUCTION") == "1":
    (SITE / "CNAME").write_text("hoopsvalue.com\n")


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
_detect = _ns["detect_caveats"]

# Likely-suitors landscape (computed ONCE): real committed-salary cap room +
# league rosters + RFA/option tags — the same pipeline the Contract Predictor
# uses, so the suitor list matches the app.
_full = _ns["build_ranked_projected"](CUR).copy()
# build_rosters needs a position column on the ranked frame; add the curated one.
_full["pos"] = _full["Player"].map(lambda n: _ts.resolve_position(str(n), "", _pos_map))
try:
    _next_c = _ns["fetch_next_year_contracts"](_ns["season_to_espn_year"](CUR), cache_v=7)
    _payroll = pd.DataFrame({"team": _full["Team"].astype(str).values,
                             "player": _full["Player"].astype(str).values})
    _LAND = _ts.apply_real_cap(
        _ts.load_team_landscape(),
        _ts.compute_cap_space(_payroll, _next_c, _ns["SALARY_CAP_M"].get(CONTRACT, 165.0)))
    _ROST = _ts.build_rosters(_full)
    _fa_tags = {"rfa": "RFA", "player_option": "player option", "team_option": "team option"}
    _status_map = {nm: _fa_tags[(v or {}).get("type")] for nm, v in _next_c.items()
                   if (v or {}).get("type") in _fa_tags}
    _SUIT = not _ROST.empty
except Exception:
    _SUIT, _LAND, _ROST, _status_map = False, None, None, {}

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

    # Confidence band (low–high) over [min floor … max].
    lo, hi = float(pred.get("low", predicted)), float(pred.get("high", predicted))
    fmin = float(pred.get("min_floor_dollars", 0.0))
    fmax = max(hi, float(pred.get("cba_max_dollars", hi)), predicted)
    _span = max(fmax - fmin, 1.0)
    _lp = max(0.0, min(100.0, (lo - fmin) / _span * 100))
    _hp = max(0.0, min(100.0, (hi - fmin) / _span * 100))
    _pp = max(0.0, min(100.0, (predicted - fmin) / _span * 100))
    band_html = (
        f'<div class="band"><div class="band-head"><span>Likely range</span>'
        f'<b>{_m(lo)} – {_m(hi)}</b></div>'
        f'<div class="band-track"><div class="band-fill" style="left:{_lp:.1f}%;width:{max(_hp - _lp, 1.5):.1f}%"></div>'
        f'<div class="band-dot" style="left:{_pp:.1f}%"></div></div>'
        f'<div class="band-ends"><span>{_m(fmin)} min</span><span>{_m(fmax)} max</span></div></div>')

    # Caveats (rookie-scale lock, supermax eligibility, …).
    try:
        _cav = _detect(f) or []
    except Exception:
        _cav = []
    caveats_html = ('<ul class="caveats">'
                    + "".join(f'<li>{_html.escape(str(c))}</li>' for c in _cav)
                    + '</ul>') if _cav else ""

    # Scouting take + comparable signings.
    median_html, scout_html, comps_html = "", "", ""
    if comps is not None and not comps.empty:
        take = _scout(f, comps)
        median_html = (f'<div class="market">Market second opinion '
                       f'<span class="muted">(weighted median of comparable signings)</span> '
                       f'<b>{_m(float(take["median"]))}</b></div>')
        _xf = _html.escape(str(take.get("x_factor", "")))
        _rng = f'{_m(float(take["q25"]))} – {_m(float(take["q75"]))}'
        _top3 = _html.escape(", ".join(take.get("top3", []) or []))
        scout_html = (
            f'<section class="scout"><h2>Scouting take</h2>'
            f'<p class="scout-x">{_xf}</p>'
            f'<div class="scout-grid">'
            f'<div><div class="sk">Market middle 50%</div><div class="sv">{_rng}</div></div>'
            f'<div><div class="sk">Closest comps</div><div class="sv">{_top3}</div></div>'
            f'</div></section>')
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

    # Likely suitors — same engine as the app (real cap room + roster fit).
    suitors_html = ""
    if _SUIT:
        try:
            _rost = _ROST[_ROST["player"].map(normalize) != normalize(name)]
            _pos_ts = _ts.resolve_position(
                name, f.get("position_detailed") or f.get("position") or "", _pos_map)
            _sui = _ts.rank_suitors(
                predicted / 1e6, float(f["barrett_score"]), _pos_ts, _rost, _LAND, n=6,
                incumbent_team=f.get("current_team"), age=f.get("age"),
                is_rfa=(_status_map.get(normalize(name)) == "RFA"),
                skill_fit=None, fa_status=_status_map)
        except Exception:
            _sui = []
        if _sui:
            _sr = "".join(
                f'<tr><td class="st">{_html.escape(str(s["team"]))}</td>'
                f'<td class="so">${float(s["offer_M"]):.0f}M</td>'
                f'<td class="sr">{_html.escape(str(s.get("reason", "")))}</td>'
                f'<td class="stool">{_html.escape(str(s.get("tool", "")))}</td></tr>'
                for s in _sui)
            suitors_html = (
                f'<section class="suitors"><h2>Likely suitors</h2>'
                f'<p class="comps-sub">Teams most likely to pursue him — each at the price they\'d realistically offer.</p>'
                f'<table class="suitors-table"><tbody>{_sr}</tbody></table></section>')

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
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Source+Sans+3:wght@400;600;700&display=swap">
<link rel="stylesheet" href="/assets/style.css">
{_THEME_BOOT}
</head>
<body>
{_nav()}
<header class="subhead">
  <a class="brand" href="/"><span class="h">HO</span><span class="ball"></span><span class="h">PS</span><span class="v">VALUE</span></a>
</header>
<main class="wrap player">
  <div class="phead">
    <div class="eyebrow">Projected {CONTRACT} contract</div>
    <h1 class="pname">{nm}<span class="chip">Score {p['score']} · #{p['rank']}</span></h1>
    <div class="pmeta">{_html.escape(str(f.get('current_team') or p['team']))} · {CUR}{age} · {pos}{draft}</div>
    {cur_deal}
  </div>
  <div class="hero-grid">
    <div class="hero-main">
      <div class="pred-number">{_m(predicted)}<span class="yr">/yr</span></div>
      {floor_html}
      <div class="pred-vs">{vs}</div>
      {band_html}
    </div>
    <div class="hero-side">
      {median_html}
      {caveats_html}
    </div>
  </div>
  {scout_html}
  <div class="two-col">
    {comps_html}
    {suitors_html}
  </div>
  <p class="disclaimer">Model: {_MODEL}. Projection is for the player's next contract priced at the {CONTRACT} cap. Informational only — not financial advice.</p>
</main>
{_footer()}
<script src="/assets/theme.js"></script>
</body>
</html>
"""
    (PLAYER_DIR / f"{p['slug']}.html").write_text(_relativize(page, 1))
    return True


if os.environ.get("SKIP_PLAYERS") == "1":
    print("player pages: SKIPPED (SKIP_PLAYERS=1)")
else:
    _pl = [p for p in players if not ONLY or p["slug"] == ONLY]
    _built = sum(render_player(p) for p in _pl)
    print(f"player pages: {_built}/{len(_pl)} -> {PLAYER_DIR}")


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
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Source+Sans+3:wght@400;600;700&display=swap">
<link rel="stylesheet" href="/assets/style.css">
{_THEME_BOOT}
</head>
<body>
{_nav("Current Rankings")}
<header class="subhead">
  <a class="brand" href="/"><span class="h">HO</span><span class="ball"></span><span class="h">PS</span><span class="v">VALUE</span></a>
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
{_footer()}
<script src="/assets/theme.js"></script>
<script src="/assets/rankings.js"></script>
</body>
</html>
"""
(SITE / "rankings.html").write_text(_relativize(RANK, 0))
print(f"rankings : {len(players)} rows -> rankings.html")

print(f"season   : {CUR}")
print(f"best     : {home['best']['name']:<24} score {home['best']['score']}")
print(f"steal    : {home['steal']['name']:<24} ${home['steal']['below_m']}M below market")
print(f"overpaid : {home['overpaid']['name']:<24} ${home['overpaid']['above_m']}M above market")
print(f"players  : {len(players)} -> {DATA / 'players.json'}")
