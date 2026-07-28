"""Scenario tests for the trade-legality engine.

Every check gets a legal case, an illegal case, and an edge case. Matching
figures test the exact 2026-27 crossovers the verified spec pins
($8.846M and $35.384M). Run: python3 tests/test_trade_rules.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trade_rules import (CBAConfig, Trade, TradePlayer, TradePick, TradeTeam,
                         byc_outgoing, validate)

CFG = CBAConfig.load("2026-27")
JULY = "2026-07-20"
FEB = "2027-02-01"


def team(abbr, payroll, roster=15, **kw):
    return TradeTeam(abbr=abbr, payroll=payroll, roster_size=roster, **kw)


def two_team(a, b, date=JULY):
    return Trade(teams=[a, b], trade_date=date)


def swap(a, b, pa, pb):
    """Wire players so legs balance: a sends pa->b, b sends pb->a."""
    a.out_players, b.in_players = pa, pa
    b.out_players, a.in_players = pb, pb


def codes(verdict):
    return {v.code for v in verdict.violations}


def note_codes(verdict):
    return {x.code for x in verdict.notes}


# ── matching formula: exact crossovers from the verified spec ─────────────────
assert abs(CFG.matching_limit(8.846) - 17.942) < 1e-6, CFG.matching_limit(8.846)
assert abs(CFG.matching_limit(5.0) - 10.25) < 1e-6           # 200% branch
assert abs(CFG.matching_limit(20.0) - 29.096) < 1e-6         # flat-add branch
assert abs(CFG.matching_limit(35.384) - (35.384 + 9.096)) < 1e-6  # = 125% point
assert abs(CFG.matching_limit(40.0) - 50.25) < 1e-6          # 125% branch
print("ok  matching formula crossovers")

# ── below-apron matching: legal and illegal ───────────────────────────────────
a, b = team("DAL", 180.0), team("BKN", 150.0)
swap(a, b, [TradePlayer("A", 20.0)], [TradePlayer("B", 29.0)])
assert validate(two_team(a, b), CFG).legal          # 29.0 <= 29.096
a, b = team("DAL", 180.0), team("BKN", 150.0)
swap(a, b, [TradePlayer("A", 20.0)], [TradePlayer("B", 29.2)])
r = validate(two_team(a, b), CFG)
assert "MATCH_BELOW_APRON" in codes(r)              # 29.2 > 29.096, DAL at 189.2 over cap
print("ok  below-apron matching")

# ── expanded matching hard-caps at apron 1 (S9.4 conservative) ────────────────
a, b = team("DAL", 180.0), team("BKN", 150.0)
swap(a, b, [TradePlayer("A", 20.0)], [TradePlayer("B", 25.0)])
r = validate(two_team(a, b), CFG)
assert r.legal and "HARDCAP_APRON1" in note_codes(r)
print("ok  expanded-matching hard-cap note")

# ── apron-1 regime: 100% flat, no cushion, post-trade position ────────────────
a, b = team("BOS", 215.0), team("DET", 140.0)
swap(a, b, [TradePlayer("A", 30.0)], [TradePlayer("B", 30.1)])
r = validate(two_team(a, b), CFG)                   # BOS after = 215.1 > apron1
assert "MATCH_APRON" in codes(r)
a, b = team("BOS", 215.0), team("DET", 140.0)
swap(a, b, [TradePlayer("A", 30.0)], [TradePlayer("B", 29.9)])
assert validate(two_team(a, b), CFG).legal          # 100% respected
print("ok  apron-1 flat matching")

# ── second apron: aggregation ban, evaluated POST-trade (spec 1.4) ────────────
a, b = team("PHX", 230.0), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 10.0), TradePlayer("A2", 8.0)], [TradePlayer("B", 17.0)])
r = validate(two_team(a, b), CFG)                   # PHX after = 229.0, still > apron2
assert "APRON2_AGGREGATION" in codes(r)
a, b = team("PHX", 224.0), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 10.0), TradePlayer("A2", 8.0)], [TradePlayer("B", 12.0)])
r = validate(two_team(a, b), CFG)                   # PHX after = 218.0 < apron2: legal
assert "APRON2_AGGREGATION" not in codes(r)
assert "HARDCAP_APRON2_AGG" in note_codes(r)        # ...but hard-caps there (4b.2)
print("ok  second-apron aggregation + post-trade evaluation")

# ── second apron: cash ban / hard-cap note below it ───────────────────────────
a, b = team("PHX", 230.0), team("CHA", 130.0)
a.cash_out, b.cash_in = 1.0, 1.0
swap(a, b, [TradePlayer("A", 10.0)], [TradePlayer("B", 9.0)])
assert "APRON2_CASH" in codes(validate(two_team(a, b), CFG))
a, b = team("DEN", 200.0), team("CHA", 130.0)
a.cash_out, b.cash_in = 1.0, 1.0
swap(a, b, [TradePlayer("A", 10.0)], [TradePlayer("B", 9.0)])
r = validate(two_team(a, b), CFG)
assert r.legal and "HARDCAP_APRON2_CASH" in note_codes(r)
print("ok  second-apron cash rules")

# ── annual cash limit ─────────────────────────────────────────────────────────
a, b = team("MIA", 170.0), team("CHA", 130.0)
a.cash_out, b.cash_in, a.cash_sent_ytd = 3.0, 3.0, 6.0     # 9.0 > 8.495
swap(a, b, [TradePlayer("A", 10.0)], [TradePlayer("B", 9.0)])
assert "CASH_LIMIT" in codes(validate(two_team(a, b), CFG))
print("ok  annual cash limit")

# ── room team absorbs with no matching (spec 1.9) ─────────────────────────────
a, b = team("UTA", 130.0), team("LAC", 190.0)
swap(a, b, [TradePlayer("A", 1.0)], [TradePlayer("B", 25.0)])
r = validate(two_team(a, b), CFG)                   # UTA after = 154.0 < cap
assert r.legal
a, b = team("UTA", 160.0), team("LAC", 190.0)       # 4.961 room + buffer + 1 out < 25 in
swap(a, b, [TradePlayer("A", 1.0)], [TradePlayer("B", 25.0)])
assert "MATCH_BELOW_APRON" in codes(validate(two_team(a, b), CFG))
print("ok  room-team absorption")

# ── minimum-exception arrivals need no matching (spec 1.7) ────────────────────
a, b = team("NYK", 210.0), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 2.0)], [TradePlayer("B", 2.4, signed_via="min")])
assert validate(two_team(a, b), CFG).legal          # NYK above apron1, min-in exempt
print("ok  minimum-exception exemption")

# ── two-ways count zero (spec 1.8) ────────────────────────────────────────────
a, b = team("SAS", 214.0), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 0.7, two_way=True)], [TradePlayer("B", 0.7, two_way=True)])
assert validate(two_team(a, b), CFG).legal
print("ok  two-way zero matching")

# ── existing hard cap is absolute ─────────────────────────────────────────────
a, b = team("GSW", 207.0, hard_cap=CFG.apron1), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 5.0)], [TradePlayer("B", 8.0)])
assert "HARD_CAP_EXCEEDED" in codes(validate(two_team(a, b), CFG))
print("ok  existing hard cap")

# ── Stepien ───────────────────────────────────────────────────────────────────
own = list(range(2027, 2034))
a, b = team("MEM", 160.0, owned_future_firsts=list(own)), team("CHA", 130.0)
a.out_picks = [TradePick(2028, 1), TradePick(2030, 1)]
b.in_picks = list(a.out_picks)
assert validate(two_team(a, b), CFG).legal          # 2027/2029/2031+ still cover gaps
a, b = team("MEM", 160.0, owned_future_firsts=list(own)), team("CHA", 130.0)
a.out_picks = [TradePick(2028, 1), TradePick(2029, 1)]
b.in_picks = list(a.out_picks)
assert "STEPIEN" in codes(validate(two_team(a, b), CFG))
a, b = team("MEM", 160.0, owned_future_firsts=list(own)), team("CHA", 130.0)
a.out_picks = [TradePick(2028, 1, swap=True), TradePick(2029, 1, swap=True)]
b.in_picks = list(a.out_picks)
assert validate(two_team(a, b), CFG).legal          # swaps keep years covered (5.5)
# protected incoming does NOT cover a year (S9.15)
a = team("MEM", 160.0, owned_future_firsts=[2027, 2029, 2031, 2032, 2033])
b = team("CHA", 130.0, owned_future_firsts=list(own))
a.out_picks = [TradePick(2027, 1)]
b.in_picks = [TradePick(2027, 1)]
b.out_picks = [TradePick(2028, 1, protection="top-10")]
a.in_picks = [TradePick(2028, 1, protection="top-10")]
assert "STEPIEN" in codes(validate(two_team(a, b), CFG))   # 2027+2028 both uncovered
# horizon: 2034 pick is beyond the 7th draft (2033)
a, b = team("MEM", 160.0, owned_future_firsts=list(own)), team("CHA", 130.0)
a.out_picks = [TradePick(2034, 1)]
b.in_picks = list(a.out_picks)
assert "PICK_HORIZON" in codes(validate(two_team(a, b), CFG))
print("ok  Stepien + horizon + swaps + protections")

# ── sign-and-trade: apron-1 bar + hard cap + season deadline ──────────────────
a, b = team("ORL", 150.0), team("MIL", 208.0)
snt = TradePlayer("S", 25.0, signed_via="sign_and_trade")
swap(b, a, [TradePlayer("X", 24.0)], [snt])          # MIL receives S&T, after=209.0
r = validate(two_team(a, b), CFG)
assert "SNT_APRON1" not in codes(r) and "HARDCAP_APRON1_SNT" in note_codes(r)
a, b = team("ORL", 150.0), team("MIL", 209.0)
swap(b, a, [TradePlayer("X", 24.0)], [TradePlayer("S", 25.0, signed_via="sign_and_trade")])
assert "SNT_APRON1" in codes(validate(two_team(a, b), CFG))  # after=210.0 > apron1
a, b = team("ORL", 150.0), team("MIL", 180.0)
swap(b, a, [TradePlayer("X", 24.0)], [TradePlayer("S", 25.0, signed_via="sign_and_trade")])
assert "SNT_SEASON" in codes(validate(two_team(a, b, date=FEB), CFG))
print("ok  sign-and-trade rules")

# ── eligibility timing ────────────────────────────────────────────────────────
a, b = team("LAL", 190.0), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 10.0, signed_date="2026-07-10")], [TradePlayer("B", 9.0)])
r = validate(two_team(a, b), CFG)                   # signed 10 days ago: Dec 15 floor
assert "RECENTLY_SIGNED" in codes(r)
assert "2026-12-15" in [v.message for v in r.violations if v.code == "RECENTLY_SIGNED"][0]
a, b = team("LAL", 190.0), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 10.0, signed_date="2026-07-10", signed_via="bird_resign",
                        raise_pct=45.0)], [TradePlayer("B", 9.0)])
r = validate(two_team(a, b, date="2026-12-20"), CFG)
assert "RECENTLY_SIGNED" in codes(r)                # Jan 15 rule extends the freeze
a, b = team("LAL", 190.0), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 10.0, signed_date="2026-07-10")], [TradePlayer("B", 9.0)])
assert validate(two_team(a, b, date="2026-12-16"), CFG).legal
print("ok  Dec 15 / Jan 15 timing")

# ── aggregation wait (2 months, spec 1.11) ────────────────────────────────────
a, b = team("IND", 170.0), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 10.0, acquired_date="2026-07-01"),
            TradePlayer("A2", 5.0)], [TradePlayer("B", 12.0)])
assert "AGGREGATION_WAIT" in codes(validate(two_team(a, b, date="2026-08-15"), CFG))
print("ok  aggregation wait")

# ── consent note + roster bounds ──────────────────────────────────────────────
a, b = team("HOU", 170.0), team("CHA", 130.0)
swap(a, b, [TradePlayer("A", 10.0, consent_required=True)], [TradePlayer("B", 9.0)])
r = validate(two_team(a, b), CFG)
assert r.legal and "CONSENT" in note_codes(r)
a, b = team("HOU", 170.0, roster=21), team("CHA", 130.0, roster=14)
swap(a, b, [TradePlayer("A", 10.0)],
     [TradePlayer("B", 4.0), TradePlayer("C", 3.0), TradePlayer("D", 2.0)])
assert "ROSTER_MAX" in codes(validate(two_team(a, b), CFG))  # HOU 21->23 offseason
print("ok  consent + roster bounds")

# ── unbalanced legs caught ────────────────────────────────────────────────────
a, b = team("CHI", 170.0), team("CHA", 130.0)
a.out_players = [TradePlayer("Ghost", 10.0)]        # nobody receives him
assert "UNBALANCED_PLAYERS" in codes(validate(two_team(a, b), CFG))
print("ok  structural balance")

# ── Stepien is a DELTA rule: pre-existing gaps do not block unrelated trades ──
gappy = team("DEN2", 160.0, owned_future_firsts=[2031, 2032, 2033])  # 27-30 gap exists
b = team("CHA", 130.0, owned_future_firsts=list(range(2027, 2034)))
gappy.out_picks = [TradePick(2033, 1)]
b.in_picks = list(gappy.out_picks)
assert validate(two_team(gappy, b), CFG).legal      # 2033 out: no NEW pair (2032 covers)
gappy = team("DEN2", 160.0, owned_future_firsts=[2031, 2032, 2033])
b = team("CHA", 130.0, owned_future_firsts=list(range(2027, 2034)))
gappy.out_picks = [TradePick(2031, 1)]
b.in_picks = list(gappy.out_picks)
r = validate(two_team(gappy, b), CFG)               # creates NEW 2030+2031 pair
assert "STEPIEN" in codes(r)
gappy = team("DEN2", 160.0, owned_future_firsts=[2031, 2032, 2033])
b = team("CHA", 130.0, owned_future_firsts=list(range(2027, 2034)))
gappy.out_picks = [TradePick(2032, 1)]              # 2031+2033 still cover neighbors
b.in_picks = list(gappy.out_picks)
assert validate(two_team(gappy, b), CFG).legal
print("ok  Stepien delta semantics")

# ── integration: real ledger drives coverage ──────────────────────────────────
import json
ledger = json.loads((Path(__file__).resolve().parent.parent / "cache" / "pick_ledger.json").read_text())
okc = ledger["teams"]["OKC"]["covered"]
assert len(ledger["teams"]) == 30
assert all(2027 <= y <= 2033 for t in ledger["teams"].values() for y in t["covered"])
a = team("OKC", 170.0, owned_future_firsts=list(okc))
b = team("CHA", 130.0, owned_future_firsts=list(range(2027, 2034)))
r = validate(two_team(a, b), CFG)                   # no picks moved: always legal
assert r.legal
print(f"ok  ledger integration (OKC covered years: {okc})")

# ── TPE absorption: no matching needed, fit + apron semantics ─────────────────
from trade_rules import TradeException

# legal absorb into a same-season TPE: no matching, no hard-cap note
a, b = team("CHA", 175.0), team("LAL", 200.0)
a.tpes = [TradeException(40.77, "LaMelo Ball", prior_season=False)]
swap(b, a, [TradePlayer("Big", 30.0)], [])          # CHA takes Big for nothing
a.in_players[0].via_tpe = True
r = validate(two_team(a, b), CFG)
assert r.legal and "TPE_ABSORB" in note_codes(r)
assert "HARDCAP_APRON1_TPE" not in note_codes(r)
print("ok  TPE absorb, same-season, no matching")

# same trade WITHOUT the TPE flag dies on matching (CHA sends nothing out)
a, b = team("CHA", 175.0), team("LAL", 200.0)
a.tpes = [TradeException(40.77, "LaMelo Ball", prior_season=False)]
swap(b, a, [TradePlayer("Big", 30.0)], [])
assert "MATCH_BELOW_APRON" in codes(validate(two_team(a, b), CFG))
print("ok  same trade without the flag needs matching")

# prior-season TPE: legal but hard-caps at apron 1
a, b = team("BOS", 195.0), team("LAL", 200.0)
a.tpes = [TradeException(27.68, "Anfernee Simons", prior_season=True)]
swap(b, a, [TradePlayer("Mid", 10.0)], [])
a.in_players[0].via_tpe = True
r = validate(two_team(a, b), CFG)
assert r.legal and "HARDCAP_APRON1_TPE" in note_codes(r)
print("ok  prior-season TPE hard-caps at apron 1")

# prior-season TPE finishing above apron 1: illegal
a, b = team("BOS", 205.0), team("LAL", 200.0)
a.tpes = [TradeException(27.68, "Anfernee Simons", prior_season=True)]
swap(b, a, [TradePlayer("Big", 10.0)], [])
a.in_players[0].via_tpe = True
assert "TPE_PRIOR_APRON1" in codes(validate(two_team(a, b), CFG))
print("ok  prior-season TPE above apron 1 illegal")

# fit: player larger than every exception + allowance
a, b = team("CLE", 170.0), team("LAL", 200.0)
a.tpes = [TradeException(10.0, "Lonzo Ball", prior_season=True)]
swap(b, a, [TradePlayer("Big", 10.3)], [])
a.in_players[0].via_tpe = True
assert "TPE_FIT" in codes(validate(two_team(a, b), CFG))
# ...but 10.2 fits inside amount + the $0.25M allowance
a, b = team("CLE", 170.0), team("LAL", 200.0)
a.tpes = [TradeException(10.0, "Lonzo Ball", prior_season=True)]
swap(b, a, [TradePlayer("Big", 10.2)], [])
a.in_players[0].via_tpe = True
assert validate(two_team(a, b), CFG).legal
print("ok  TPE fit boundary (amount + allowance)")

# two players share one exception; a third does not fit; no TPE at all
a, b = team("CHA", 150.0), team("LAL", 200.0)
a.tpes = [TradeException(20.0, "X", prior_season=False)]
swap(b, a, [TradePlayer("P1", 12.0), TradePlayer("P2", 8.0),
            TradePlayer("P3", 5.0)], [])
for p in a.in_players:
    p.via_tpe = True
r = validate(two_team(a, b), CFG)
assert "TPE_FIT" in codes(r)                        # P3 has nowhere to go
a, b = team("NOP", 150.0), team("LAL", 200.0)
swap(b, a, [TradePlayer("P1", 5.0)], [])
a.in_players[0].via_tpe = True
assert "TPE_NONE" in codes(validate(two_team(a, b), CFG))
print("ok  shared exception + overflow + no-TPE cases")

# mixed structure: one player via TPE, another matched by outgoing salary
a, b = team("MIL", 170.0), team("LAL", 200.0)
a.tpes = [TradeException(25.46, "Giannis Antetokounmpo", prior_season=False)]
swap(a, b, [TradePlayer("Out", 15.0)],
     [TradePlayer("Matched", 16.0), TradePlayer("Absorbed", 20.0)])
a.in_players[1].via_tpe = True                      # Absorbed rides the TPE
r = validate(two_team(a, b), CFG)                   # after: 191, below apron 1
assert r.legal and "TPE_ABSORB" in note_codes(r)    # 16.0 matches vs 15.0 out
# the SAME structure at a higher payroll flips to the flat over-apron regime
# (the absorbed salary still counts toward post-trade payroll): illegal
a, b = team("MIL", 190.0), team("LAL", 200.0)
a.tpes = [TradeException(25.46, "Giannis Antetokounmpo", prior_season=False)]
swap(a, b, [TradePlayer("Out", 15.0)],
     [TradePlayer("Matched", 16.0), TradePlayer("Absorbed", 20.0)])
a.in_players[1].via_tpe = True
assert "MATCH_APRON" in codes(validate(two_team(a, b), CFG))
print("ok  mixed TPE + matching structure (both apron regimes)")

# ── S&T builder: new signing traded out, BYC matching, payroll semantics ──────
def snt_player(name, new, prior=None):
    return TradePlayer(name, new, signed_via="sign_and_trade",
                       snt_out=True, prior_salary=prior)

# under-cap sender: full credit, and the new deal never touches its payroll
a, b = team("DET", 130.0), team("CHA", 150.0)
swap(a, b, [snt_player("FA", 20.0, prior=5.0)], [TradePlayer("Back", 18.0)])
r = validate(two_team(a, b), CFG)
assert r.legal and "SNT_BUILDER" in note_codes(r)
imp = {i.abbr: i for i in r.impact}
assert abs(imp["DET"].payroll_after - 148.0) < 1e-6   # 130 + 18, NOT -20
assert imp["DET"].roster_after == 16                  # FA never left its roster
assert imp["CHA"].roster_after == 15                  # -1 Back +1 FA
assert "HARDCAP_APRON1_SNT" in note_codes(r)          # CHA receives an S&T player
print("ok  S&T builder payroll + roster semantics")

# over-cap sender: BYC halves the credit; the take-back stops matching
a, b = team("MIA", 170.0), team("CHA", 150.0)
swap(a, b, [snt_player("FA", 20.0, prior=5.0)], [TradePlayer("Back", 20.0)])
r = validate(two_team(a, b), CFG)                     # credit 10 -> limit 19.096
assert "MATCH_BELOW_APRON" in codes(r)
# same raise below 20% (no BYC): full credit, legal
a, b = team("MIA", 170.0), team("CHA", 150.0)
swap(a, b, [snt_player("FA", 20.0, prior=18.0)], [TradePlayer("Back", 20.0)])
r = validate(two_team(a, b), CFG)
assert r.legal
# unknown prior salary: conservative BYC applies
a, b = team("MIA", 170.0), team("CHA", 150.0)
swap(a, b, [snt_player("FA", 20.0)], [TradePlayer("Back", 20.0)])
assert "MATCH_BELOW_APRON" in codes(validate(two_team(a, b), CFG))
# the incoming take-back still counts toward post-trade payroll, so a big
# take-back can flip the sender into the over-apron flat matching regime
a, b = team("MIA", 190.0), team("CHA", 150.0)
swap(a, b, [snt_player("FA", 20.0, prior=5.0)], [TradePlayer("Back", 20.0)])
assert "MATCH_APRON" in codes(validate(two_team(a, b), CFG))   # after 210 > apron1
print("ok  BYC credit (over-cap, raise >20%, unknown prior, apron flip)")

# receiving team must finish below apron 1 (existing rule via the builder path)
a, b = team("UTA", 140.0), team("BOS", 205.0)
swap(a, b, [snt_player("FA", 20.0, prior=5.0)], [TradePlayer("Back", 10.0)])
assert "SNT_APRON1" in codes(validate(two_team(a, b), CFG))
print("ok  S&T builder receiver apron rule")

# ── BYC helper (Kessler example from the spec) ────────────────────────────────
assert abs(byc_outgoing(4.878938, 30.108821) - 15.0544105) < 1e-6
print("ok  BYC outgoing (Kessler example)")

print("\nALL TRADE-RULES SCENARIOS PASS")
