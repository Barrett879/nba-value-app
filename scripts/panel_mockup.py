"""Render design options for the Rankings row panel, with real data.

    /opt/anaconda3/bin/python3 -u scripts/panel_mockup.py

Writes exports/panel_options.html: five directions for the expandable panel,
each drawn with the same three players so they are compared on substance and
not on whose numbers flatter them. Light and dark both, since the site follows
the system theme and a panel that reads well in one can disappear in the other.

Offline. Writes nothing the site reads.
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)
import pandas as pd, utils
logging.disable(logging.NOTSET)

WHO = ["Walker Kessler", "Domantas Sabonis", "Nikola Jokić"]
TEAM = {"Walker Kessler": "LAL", "Domantas Sabonis": "SAC", "Nikola Jokić": "DEN"}


def pct(v):
    return "-" if v is None or v != v else f"{v * 100:.1f}%"


def one(v, d=1):
    return "-" if v is None or v != v else f"{v:.{d}f}"


def rk(v):
    if v is None or v != v:
        return ""
    v = int(v)
    suf = "th" if 10 <= v % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(v % 10, "th")
    return f"{v}{suf}"


def build():
    src = utils.fetch_league_stats(utils.SEASONS[0])
    src = utils._merge_provisional_box_scores(src, utils.SEASONS[0])
    n = len(src)
    rows = {}
    for w in WHO:
        r = src[src.PLAYER_NAME == w]
        if len(r):
            rows[w] = r.iloc[0].to_dict()
    return rows, n


def variant_a(d):
    """Box score line: the shorthand every fan already reads."""
    return (
        f'<div class="vA">'
        f'<div class="vA-main"><b>{one(d.get("PTS"))}</b><span>/</span>'
        f'<b>{one(d.get("REB"))}</b><span>/</span><b>{one(d.get("AST"))}</b>'
        f'<em>pts · reb · ast</em></div>'
        f'<div class="vA-sub">'
        f'{one(d.get("FGM"))}-{one(d.get("FGA"))} FG'
        f' · {one(d.get("FG3M"))}-{one(d.get("FG3A"))} 3P'
        f' · {one(d.get("FTM"))}-{one(d.get("FTA"))} FT'
        f' · {one(d.get("MIN"))} min · {one(d.get("STL"))} stl'
        f' · {one(d.get("BLK"))} blk · {one(d.get("TOV"))} tov</div></div>')


def variant_b(d, n):
    """Ranked chips: the number plus where it sits in the league."""
    out = []
    for key, lab, rank in (("PTS", "PPG", "PTS_RANK"), ("REB", "RPG", "REB_RANK"),
                           ("AST", "APG", "AST_RANK"), ("STL", "SPG", "STL_RANK"),
                           ("BLK", "BPG", "BLK_RANK"), ("MIN", "MPG", "MIN_RANK")):
        r = d.get(rank)
        top = "top" if (r and r == r and int(r) <= 10) else ""
        out.append(f'<span class="chip {top}"><i>{lab}</i><b>{one(d.get(key))}</b>'
                   f'<u>{rk(r)}</u></span>')
    return f'<div class="vB">{"".join(out)}</div>'


def variant_c(d):
    """Grouped by Barrett Score component, in the Search breakdown's colours."""
    groups = [("Scoring", "#e67e22", [("PPG", one(d.get("PTS"))), ("FG%", pct(d.get("FG_PCT"))),
                                      ("3P%", pct(d.get("FG3_PCT"))), ("FT%", pct(d.get("FT_PCT")))]),
              ("Playmaking", "#9b59b6", [("APG", one(d.get("AST"))), ("TOV", one(d.get("TOV")))]),
              ("Rebounding", "#27ae60", [("RPG", one(d.get("REB"))), ("OFF", one(d.get("OREB"))),
                                         ("DEF", one(d.get("DREB")))]),
              ("Defense", "#3498db", [("SPG", one(d.get("STL"))), ("BPG", one(d.get("BLK"))),
                                      ("PF", one(d.get("PF")))])]
    out = []
    for name, col, items in groups:
        cells = "".join(f'<span><i>{k}</i><b>{v}</b></span>' for k, v in items)
        out.append(f'<div class="grp" style="--gc:{col}"><h6>{name}</h6>'
                   f'<div class="grp-row">{cells}</div></div>')
    return f'<div class="vC">{"".join(out)}</div>'


def variant_d(d, n):
    """Percentile bars: position in the league, drawn not stated."""
    out = []
    for key, lab, rank in (("PTS", "PPG", "PTS_RANK"), ("REB", "RPG", "REB_RANK"),
                           ("AST", "APG", "AST_RANK"), ("STL", "SPG", "STL_RANK"),
                           ("BLK", "BPG", "BLK_RANK")):
        r = d.get(rank)
        p = 0 if not r or r != r else max(2, 100 - (int(r) / n * 100))
        out.append(f'<div class="barrow"><i>{lab}</i><b>{one(d.get(key))}</b>'
                   f'<span class="track"><span class="fill" style="width:{p:.0f}%"></span></span>'
                   f'<u>{rk(r)}</u></div>')
    return f'<div class="vD">{"".join(out)}</div>'


def variant_e(d, team):
    """Team-accented card: splits as makes-attempts, ranks as quiet superscripts."""
    hex_ = utils.TEAM_HEX.get(team, "#888")
    prim = "".join(
        f'<span><b>{one(d.get(k))}</b><i>{lab}</i>'
        f'{f"<u>{rk(d.get(rk_))}</u>" if d.get(rk_) and d.get(rk_) == d.get(rk_) else ""}</span>'
        for k, lab, rk_ in (("PTS", "PPG", "PTS_RANK"), ("REB", "RPG", "REB_RANK"),
                            ("AST", "APG", "AST_RANK")))
    sec = (f'{one(d.get("FGM"))}-{one(d.get("FGA"))} FG · '
           f'{one(d.get("FG3M"))}-{one(d.get("FG3A"))} 3P · '
           f'{one(d.get("FTM"))}-{one(d.get("FTA"))} FT · '
           f'{one(d.get("MIN"))} min · {one(d.get("STL"))} stl · '
           f'{one(d.get("BLK"))} blk · {one(d.get("TOV"))} tov · '
           f'{"+" if (d.get("PLUS_MINUS") or 0) > 0 else ""}{one(d.get("PLUS_MINUS"))} +/-')
    return (f'<div class="vE" style="--tc:{hex_}">'
            f'<div class="vE-prim">{prim}</div><div class="vE-sec">{sec}</div></div>')


CSS = """
:root{--bg:#fff;--panel:#f7f8fa;--line:#e3e6ea;--fg:#1b1f24;--fg2:#5a6472;--fg3:#8a94a3;
      --accent:#e03131;--good:#2f9e44;}
@media (prefers-color-scheme: dark){
  :root{--bg:#14171c;--panel:#1c2026;--line:#2b313a;--fg:#e8ecf1;--fg2:#aab3c0;
        --fg3:#79838f;--accent:#ff6b6b;--good:#51cf66;}
}
*{box-sizing:border-box}
body{margin:0;padding:28px 22px 60px;background:var(--bg);color:var(--fg);
     font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;}
h1{font-size:19px;margin:0 0 4px}
.lede{color:var(--fg2);max-width:62ch;margin:0 0 26px;font-size:13px}
h2{font-size:14px;margin:34px 0 2px;display:flex;align-items:baseline;gap:9px}
h2 b{font-size:11px;font-weight:600;color:var(--accent);letter-spacing:.06em}
h2 em{font-style:normal;font-weight:400;color:var(--fg2);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:9px;
      border:1px solid var(--line);border-radius:9px;overflow:hidden}
th{background:var(--panel);color:var(--fg2);font-size:10.5px;letter-spacing:.07em;
   text-transform:uppercase;text-align:left;padding:8px 11px;font-weight:600}
td{padding:9px 11px;border-top:1px solid var(--line);vertical-align:top}
.num{text-align:right;font-variant-numeric:tabular-nums}
.nm{font-weight:600}
.mark{display:inline-block;width:0;height:0;margin-right:7px;vertical-align:1px;
      border-left:4px solid var(--fg3);border-top:3.5px solid transparent;
      border-bottom:3.5px solid transparent;transform:rotate(90deg)}

/* A */
.vA{margin:7px 0 2px}
.vA-main{display:flex;align-items:baseline;gap:5px;font-variant-numeric:tabular-nums}
.vA-main b{font-size:19px;font-weight:700;letter-spacing:-.02em}
.vA-main span{color:var(--fg3);font-size:15px}
.vA-main em{font-style:normal;margin-left:9px;color:var(--fg3);font-size:10.5px;
            letter-spacing:.05em;text-transform:uppercase}
.vA-sub{margin-top:4px;color:var(--fg2);font-size:12px;font-variant-numeric:tabular-nums}

/* B */
.vB{display:flex;flex-wrap:wrap;gap:6px;margin:7px 0 2px}
.chip{display:inline-flex;align-items:baseline;gap:5px;padding:3px 9px;
      border:1px solid var(--line);border-radius:14px;background:var(--panel);
      font-variant-numeric:tabular-nums}
.chip i{font-style:normal;color:var(--fg3);font-size:10px;letter-spacing:.05em;
        text-transform:uppercase}
.chip b{font-weight:700}
.chip u{text-decoration:none;color:var(--fg3);font-size:10px}
.chip.top{border-color:var(--good);background:color-mix(in srgb,var(--good) 11%,transparent)}
.chip.top u{color:var(--good);font-weight:600}

/* C */
.vC{display:flex;flex-wrap:wrap;gap:16px;margin:8px 0 3px}
.grp h6{margin:0 0 3px;font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;
        color:var(--gc);font-weight:700}
.grp-row{display:flex;gap:11px;padding-left:7px;border-left:2px solid var(--gc)}
.grp-row span{display:flex;flex-direction:column;font-variant-numeric:tabular-nums}
.grp-row i{font-style:normal;color:var(--fg3);font-size:9.5px;letter-spacing:.04em}
.grp-row b{font-weight:650;font-size:13px}

/* D */
.vD{margin:8px 0 3px;max-width:430px}
.barrow{display:grid;grid-template-columns:34px 46px 1fr 34px;align-items:center;
        gap:9px;padding:2.5px 0;font-variant-numeric:tabular-nums}
.barrow i{font-style:normal;color:var(--fg3);font-size:10px;letter-spacing:.05em;
          text-transform:uppercase}
.barrow b{font-weight:650;text-align:right}
.barrow u{text-decoration:none;color:var(--fg3);font-size:10px;text-align:right}
.track{height:5px;background:var(--line);border-radius:3px;overflow:hidden}
.fill{display:block;height:100%;background:var(--accent);border-radius:3px;opacity:.8}

/* E */
.vE{margin:7px 0 2px;padding:8px 12px;background:var(--panel);
    border-left:3px solid var(--tc);border-radius:0 6px 6px 0}
.vE-prim{display:flex;gap:22px}
.vE-prim span{display:flex;align-items:baseline;gap:4px;font-variant-numeric:tabular-nums}
.vE-prim b{font-size:17px;font-weight:700;letter-spacing:-.02em}
.vE-prim i{font-style:normal;color:var(--fg3);font-size:10px;letter-spacing:.05em;
           text-transform:uppercase}
.vE-prim u{text-decoration:none;color:var(--fg3);font-size:9.5px;align-self:flex-start}
.vE-sec{margin-top:5px;color:var(--fg2);font-size:11.5px;font-variant-numeric:tabular-nums}
.note{margin-top:7px;color:var(--fg3);font-size:11.5px;max-width:70ch}
"""


def table(rows, body_fn):
    out = ['<table><tr><th style="width:30px">#</th><th>Player</th>'
           '<th style="width:58px">Team</th><th class="num" style="width:92px">Barrett</th>'
           '<th class="num" style="width:92px">Salary</th></tr>']
    sal = {"Walker Kessler": "$30.11M", "Domantas Sabonis": "$45.47M", "Nikola Jokić": "$59.03M"}
    sc = {"Walker Kessler": "49.77", "Domantas Sabonis": "48.69", "Nikola Jokić": "40.39"}
    for i, w in enumerate(WHO, 1):
        d = rows.get(w)
        if d is None:
            continue
        out.append(
            f'<tr><td class="num">{i}</td><td><span class="mark"></span>'
            f'<span class="nm">{w}</span>{body_fn(d, w)}</td>'
            f'<td>{TEAM[w]}</td><td class="num">{sc[w]}</td>'
            f'<td class="num">{sal[w]}</td></tr>')
    out.append("</table>")
    return "".join(out)


def main():
    rows, n = build()
    secs = [
        ("A", "Box score line", "The shorthand a fan already reads. One big slash line, "
         "everything else demoted to a quiet second row.", lambda d, w: variant_a(d)),
        ("B", "Ranked chips", "Each number carries where it sits in the league. Top-ten "
         "values go green, so the eye finds what is exceptional without reading.",
         lambda d, w: variant_b(d, n)),
        ("C", "Score components", "Grouped the way the Barrett Score is built and coloured "
         "to match the Search breakdown, so the panel explains the score beside it.",
         lambda d, w: variant_c(d)),
        ("D", "Percentile bars", "Rank drawn rather than stated. Densest to scan down a "
         "column, and the only one that survives a tiny sample gracefully.",
         lambda d, w: variant_d(d, n)),
        ("E", "Team-accented card", "A stripe in the team's colour, shooting as makes-"
         "attempts instead of percentages, ranks as quiet superscripts.",
         lambda d, w: variant_e(d, TEAM[w])),
    ]
    body = [
        "<h1>Row panel: five directions</h1>",
        '<p class="lede">Same three players throughout, with live 2026-27 numbers, so these '
        'compete on design rather than on whose stat line flatters them. Every option is pure '
        'CSS: this table renders through st.markdown, which strips scripts, and a '
        'components.html iframe serves the app homepage on a Render cold start. Switch your '
        'system to dark mode and reload to see both themes.</p>',
    ]
    for key, name, blurb, fn in secs:
        body.append(f'<h2><b>{key}</b> {name} <em>{blurb}</em></h2>')
        body.append(table(rows, fn))
    body.append(
        '<p class="note">Small samples are the honest problem here. Kessler shoots 84.6% from '
        'the field on two games, and a percentage that extreme reads as a bug even when it is '
        'right. A and E answer it by showing makes-attempts; D answers it by drawing rank '
        'instead of rate. B and C still show the raw percentage.</p>')
    out = ROOT / "exports" / "panel_options.html"
    out.write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Row panel options</title><style>{CSS}</style></head>'
        f'<body>{"".join(body)}</body></html>')
    print(f"wrote {out}  ({len(rows)} players, {n} in the league pool)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
