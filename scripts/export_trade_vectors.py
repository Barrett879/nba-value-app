"""Export trade-rules test vectors for the in-browser JS engine.

The Python engine (trade_rules.py) is the verified source of truth. The Trade
Machine UI runs a JS port for instant drag-and-drop verdicts; this script
freezes a set of scenarios WITH the Python engine's verdicts so the JS port
can prove parity (the page runs every vector on load in dev mode and any
mismatch is loud). Re-run whenever trade_rules.py or the CBA config changes.

Output: cache/trade_vectors.json  [{name, trade, expected_codes, legal}]
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trade_rules import (CBAConfig, Trade, TradeException, TradePick,  # noqa: E402
                         TradePlayer, TradeTeam, validate)

CFG = CBAConfig.load("2026-27")
JULY = "2026-07-27"


def team(abbr, payroll, roster=15, firsts=None, **kw):
    return TradeTeam(abbr=abbr, payroll=payroll, roster_size=roster,
                     owned_future_firsts=firsts or [], **kw)


def swap(a, b, pa, pb):
    a.out_players, b.in_players = pa, pa
    b.out_players, a.in_players = pb, pb


def V(name, *teams, date=JULY):
    t = Trade(teams=list(teams), trade_date=date)
    r = validate(t, CFG)
    return {
        "name": name,
        "trade": {"trade_date": date, "teams": [asdict(x) for x in teams]},
        "legal": r.legal,
        "violation_codes": sorted({v.code for v in r.violations}),
        "note_codes": sorted({n.code for n in r.notes}),
    }


ALL7 = list(range(2027, 2034))
vectors = []

a, b = team("DAL", 180.0), team("BKN", 150.0)
swap(a, b, [TradePlayer("A", 20.0)], [TradePlayer("B", 29.0)])
vectors.append(V("below-apron matching legal", a, b))

a, b = team("DAL", 180.0), team("BKN", 150.0)
swap(a, b, [TradePlayer("A", 20.0)], [TradePlayer("B", 29.2)])
vectors.append(V("below-apron matching illegal", a, b))

a, b = team("BOS", 215.0), team("DET", 140.0)
swap(a, b, [TradePlayer("A", 30.0)], [TradePlayer("B", 30.1)])
vectors.append(V("apron1 flat 100 illegal", a, b))

a, b = team("PHX", 230.0), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 10.0), TradePlayer("A2", 8.0)], [TradePlayer("B", 17.0)])
vectors.append(V("apron2 aggregation ban", a, b))

a, b = team("PHX", 224.0), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 10.0), TradePlayer("A2", 8.0)], [TradePlayer("B", 12.0)])
vectors.append(V("post-trade apron evaluation legal + hardcap note", a, b))

a, b = team("DEN", 200.0), team("CHA", 130.0)
a.cash_out, b.cash_in = 1.0, 1.0
swap(a, b, [TradePlayer("A", 10.0)], [TradePlayer("B", 9.0)])
vectors.append(V("cash below apron2 legal + note", a, b))

a, b = team("MIA", 170.0), team("CHA", 130.0)
a.cash_out, b.cash_in, a.cash_sent_ytd = 3.0, 3.0, 6.0
swap(a, b, [TradePlayer("A", 10.0)], [TradePlayer("B", 9.0)])
vectors.append(V("annual cash limit", a, b))

a, b = team("UTA", 130.0), team("LAC", 190.0)
swap(a, b, [TradePlayer("A", 1.0)], [TradePlayer("B", 25.0)])
vectors.append(V("room team absorbs", a, b))

a, b = team("NYK", 210.0), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 2.0)], [TradePlayer("B", 2.4, signed_via="min")])
vectors.append(V("min exception exempt", a, b))

a, b = team("GSW", 207.0, hard_cap=CFG.apron1), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 5.0)], [TradePlayer("B", 8.0)])
vectors.append(V("existing hard cap absolute", a, b))

a, b = team("MEM", 160.0, firsts=list(ALL7)), team("CHA", 130.0, firsts=list(ALL7))
a.out_picks = [TradePick(2028, 1), TradePick(2029, 1)]
b.in_picks = list(a.out_picks)
vectors.append(V("stepien consecutive", a, b))

a, b = team("MEM", 160.0, firsts=[2031, 2032, 2033]), team("CHA", 130.0, firsts=list(ALL7))
a.out_picks = [TradePick(2033, 1)]
b.in_picks = list(a.out_picks)
vectors.append(V("stepien pre-existing gap not worsened", a, b))

a, b = team("ORL", 150.0), team("MIL", 209.0)
snt = TradePlayer("S", 25.0, signed_via="sign_and_trade")
swap(b, a, [TradePlayer("X", 24.0)], [snt])
vectors.append(V("snt over apron1 illegal", a, b))

a, b = team("LAL", 190.0), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 10.0, signed_date="2026-07-10")], [TradePlayer("B", 9.0)])
vectors.append(V("recently signed dec15", a, b))

a, b = team("HOU", 170.0, roster=21), team("CHA", 130.0, roster=14)
swap(a, b, [TradePlayer("A", 10.0)],
     [TradePlayer("B", 4.0), TradePlayer("C", 3.0), TradePlayer("D", 2.0)])
vectors.append(V("roster max offseason", a, b))

a, b = team("CHA", 175.0), team("LAL", 200.0)
a.tpes = [TradeException(40.77, "LaMelo Ball", prior_season=False)]
swap(b, a, [TradePlayer("Big", 30.0)], [])
a.in_players[0].via_tpe = True
vectors.append(V("tpe absorb same-season legal", a, b))

a, b = team("BOS", 195.0), team("LAL", 200.0)
a.tpes = [TradeException(27.68, "Anfernee Simons", prior_season=True)]
swap(b, a, [TradePlayer("Mid", 10.0)], [])
a.in_players[0].via_tpe = True
vectors.append(V("tpe prior-season hardcap note", a, b))

a, b = team("BOS", 205.0), team("LAL", 200.0)
a.tpes = [TradeException(27.68, "Anfernee Simons", prior_season=True)]
swap(b, a, [TradePlayer("Big", 10.0)], [])
a.in_players[0].via_tpe = True
vectors.append(V("tpe prior-season above apron1 illegal", a, b))

a, b = team("CLE", 170.0), team("LAL", 200.0)
a.tpes = [TradeException(10.0, "Lonzo Ball", prior_season=True)]
swap(b, a, [TradePlayer("Big", 10.3)], [])
a.in_players[0].via_tpe = True
vectors.append(V("tpe does not fit", a, b))

a, b = team("MIL", 170.0), team("LAL", 200.0)
a.tpes = [TradeException(25.46, "Giannis Antetokounmpo", prior_season=False)]
swap(a, b, [TradePlayer("Out", 15.0)],
     [TradePlayer("Matched", 16.0), TradePlayer("Absorbed", 20.0)])
a.in_players[1].via_tpe = True
vectors.append(V("tpe mixed with matching", a, b))

out = ROOT / "cache" / "trade_vectors.json"
out.write_text(json.dumps(vectors, separators=(",", ":")))
print(f"wrote {out.name}: {len(vectors)} vectors, "
      f"{sum(1 for v in vectors if not v['legal'])} illegal cases")
