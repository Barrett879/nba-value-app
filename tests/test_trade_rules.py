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


# ── spec 3.1 and 3.4, added 2026-11-20 ──────────────────────────────────────

def _two(a_pay, b_pay, out_a, in_a, **akw):
    import trade_rules as tr
    a = tr.TradeTeam(abbr="AAA", payroll=a_pay, roster_size=15, **akw)
    b = tr.TradeTeam(abbr="BBB", payroll=b_pay, roster_size=15)
    a.out_players += out_a
    a.in_players += in_a
    b.out_players += in_a
    b.in_players += out_a
    return tr.Trade(teams=[a, b], trade_date="2026-11-20"), a


def test_one_to_one_above_the_second_apron_is_not_aggregation():
    """Spec 3.1, line 171, verbatim: "Multiple players in a deal each matched
    separately (or absorbed into separate exceptions) is NOT aggregation."
    Confidence: high.

    The ban was implemented as a HEAD COUNT -- len(counted_out()) >= 2 and
    matchable_in() > 0 -- so a two-for-two above the second apron was refused
    even when each incoming player was covered one-to-one by a single outgoing
    salary. Nothing in such a trade is aggregated. A ban that refuses legal
    trades is the loud failure direction, and it was about to be fed MORE
    trades by other work.
    """
    import trade_rules as tr
    C = tr.CBAConfig.load("2026-27")
    P = tr.TradePlayer
    trade, a = _two(230.0, 170.0,
                    [P(name="o1", salary=12.0), P(name="o2", salary=9.0)],
                    [P(name="i1", salary=11.0), P(name="i2", salary=8.5)])
    assert a.payroll_after() > C.apron2, "the premise needs AAA above the second apron"
    codes = [v.code for v in tr.validate(trade, C).violations if v.team == "AAA"]
    assert "APRON2_AGGREGATION" not in codes, (
        f"a one-to-one two-for-two was refused as aggregation: {codes}")

    # and the definitional floor: ONE outgoing salary is not a combination of
    # two or more, whatever else is wrong with the trade. Mutation testing
    # found nothing asserting this.
    trade1, a1 = _two(230.0, 170.0, [P(name="o", salary=20.0)],
                      [P(name="i", salary=19.0)])
    assert a1.payroll_after() > C.apron2
    codes1 = [v.code for v in tr.validate(trade1, C).violations if v.team == "AAA"]
    assert "APRON2_AGGREGATION" not in codes1, (
        f"a one-for-one was called aggregation: {codes1}")


def test_genuine_aggregation_above_the_second_apron_is_still_refused():
    """The positive control. Loosening the test above must not switch the ban
    off: one incoming player who needs BOTH outgoing salaries IS aggregation."""
    import trade_rules as tr
    C = tr.CBAConfig.load("2026-27")
    P = tr.TradePlayer
    trade, a = _two(230.0, 170.0,
                    [P(name="o1", salary=12.0), P(name="o2", salary=9.0)],
                    [P(name="big", salary=20.0)])
    assert a.payroll_after() > C.apron2
    codes = [v.code for v in tr.validate(trade, C).violations if v.team == "AAA"]
    assert "APRON2_AGGREGATION" in codes, (
        f"$20.0M taken back against $12.0M and $9.0M separately is aggregation "
        f"and must be refused above the second apron: {codes}")


def test_row_J_a_second_apron_team_cannot_use_an_snt_exception():
    """CBA row J / spec 3.4, which was in NEITHER engine: "row J" appeared zero
    times in trade_rules.py and in templates/trade_machine.html.

    Denver holds a $10.23M exception generated by signing and trading Peyton
    Watson away, and data/trade_exceptions_2026_27.csv already called it
    "UNUSABLE while DEN is over the second apron", sourced to three named
    reporters -- in free text no code read. Absorbing $9.0M into it finished
    Denver at $227.1M, $5.45M above the second apron on our own figures, and
    returned legal with no violation at all.
    """
    import trade_rules as tr
    C = tr.CBAConfig.load("2026-27")
    P, TE = tr.TradePlayer, tr.TradeException

    def verdict(payroll, snt):
        a = tr.TradeTeam(abbr="AAA", payroll=payroll, roster_size=14,
                         tpes=[TE(amount=10.23, label="Watson",
                                  prior_season=False, snt=snt)])
        a.in_players += [P(name="big", salary=9.0, via_tpe=True)]
        b = tr.TradeTeam(abbr="BBB", payroll=150.0, roster_size=15)
        b.out_players += [P(name="big", salary=9.0)]
        r = tr.validate(tr.Trade(teams=[a, b], trade_date="2026-11-20"), C)
        return ([v.code for v in r.violations if v.team == "AAA"],
                [n.code for n in r.notes if n.team == "AAA"])

    over, _ = verdict(218.137, snt=True)
    assert "APRON2_SNT_TPE" in over, (
        f"a second-apron team absorbed salary into an S&T-generated exception: {over}")

    plain, _ = verdict(218.137, snt=False)
    assert "APRON2_SNT_TPE" not in plain, (
        f"an ORDINARY trade exception was refused under row J, which would ban "
        f"every TPE absorption above the second apron: {plain}")

    under, notes = verdict(180.0, snt=True)
    assert "APRON2_SNT_TPE" not in under, under
    assert "HARDCAP_APRON2_SNT" in notes, (
        f"below the line the same action is legal and hard-caps at the second "
        f"apron (spec 4b.4), and that note is missing: {notes}")


def test_row_J_does_not_become_a_blanket_sign_and_trade_ban():
    """THE BOUNDARY, and it is written into the spec. Section 9 item 7: a PURE
    outbound S&T salary dump with nothing taken back is "legal per CBA text, no
    secondary confirmation; some media claim a blanket ban. Default: allow."

    Row J bans taking salary BACK against a signed-and-traded contract. Sending
    one away for nothing is the one S&T move an over-second-apron team keeps.
    """
    import trade_rules as tr
    C = tr.CBAConfig.load("2026-27")
    P = tr.TradePlayer
    a = tr.TradeTeam(abbr="AAA", payroll=230.0, roster_size=14)
    a.out_players += [P(name="snt", salary=12.0, snt_out=True, prior_salary=10.0)]
    b = tr.TradeTeam(abbr="BBB", payroll=150.0, roster_size=14)
    b.in_players += [P(name="snt", salary=12.0)]
    codes = [v.code for v in tr.validate(
        tr.Trade(teams=[a, b], trade_date="2026-11-20"), C).violations if v.team == "AAA"]
    assert not any(c.startswith("APRON2_SNT") for c in codes), (
        f"a pure outbound S&T dump was refused, which widens row J into the "
        f"blanket ban the spec explicitly declines to adopt: {codes}")


def test_the_js_port_mirrors_row_J_and_the_aggregation_fix():
    """Both engines or neither. The JS port renders the verdicts users see, and
    parity compares codes, so a rule added only in Python would leave the page
    approving what the suite refuses."""
    from pathlib import Path
    js = (Path(__file__).resolve().parent.parent
          / "templates" / "trade_machine.html").read_text(encoding="utf-8")
    for token in ("aggregationNeeded", "sntTpesUsed",
                  "APRON2_SNT_TPE", "APRON2_SNT_MATCH", "HARDCAP_APRON2_SNT"):
        assert token in js, f"the JS port is missing {token}"
    assert "countedOut(t).length>=2 && matchIn(t)>0" not in js, (
        "the JS port still uses the head-count proxy for aggregation")
    assert "snt:!!x.snt" in js, (
        "buildTrade does not carry the S&T flag into the engine, so row J "
        "can never fire on the live page")


def test_a_team_published_above_the_second_apron_is_not_told_it_will_be_hard_capped_there():
    """You cannot be hard-capped at a level you already exceed.

    Denver's publisher states "Operating over the second tax apron ($221.7MM)"
    and gives its apron-basis salary as $224.8M against a tax-basis $219.2M.
    HoopsValue prices on the tax basis, so the engine saw $218.1M, and told
    anyone building a Denver trade that aggregating "hard-caps DEN at the
    second apron for the rest of the season" -- a sentence that is false on
    sourced evidence, not merely incomplete.

    The fix is TEXT ONLY and deliberately so: same note code, same verdict,
    so the Python/JS parity vectors are untouched and no invented number can
    leak into a hard-cap comparison. data/apron_salaries_2026_27.csv explains
    at length why this field must never reach check_hard_caps.
    """
    import trade_rules as tr
    C = tr.CBAConfig.load("2026-27")
    P = tr.TradePlayer

    def note_for(apron_salary):
        a = tr.TradeTeam(abbr="DEN", payroll=218.137, roster_size=14,
                         apron_salary=apron_salary)
        a.out_players += [P(name="o1", salary=12.0), P(name="o2", salary=9.0)]
        a.in_players += [P(name="big", salary=20.0)]
        b = tr.TradeTeam(abbr="BBB", payroll=150.0, roster_size=15)
        b.out_players += a.in_players
        b.in_players += a.out_players
        r = tr.validate(tr.Trade(teams=[a, b], trade_date="2026-11-20"), C)
        return [n for n in r.notes if n.team == "DEN" and n.code == "HARDCAP_APRON2_AGG"]

    plain = note_for(None)
    assert plain, "the hard-cap note should still fire for a team below the line"
    assert "already above" not in plain[0].message

    told = note_for(224.8)
    assert told, "the note code must not change: parity compares codes"
    assert "already above the second apron" in told[0].message, (
        f"a team published above the second apron is still being told it would "
        f"be hard-capped there: {told[0].message}")
    assert "224.8" in told[0].message, "the published figure should be shown"


def test_the_published_apron_figures_never_reach_a_cap_comparison():
    """The guard on the file, not on the code. An apron figure folded into a
    hard-cap test would put Cleveland ($0.023M of room) and Toronto over the
    line on numbers we did not compute, which is the failure that cost this
    project four months on Golden State.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    src = (root / "trade_rules.py").read_text(encoding="utf-8")
    body = src.split("def check_hard_caps(", 1)[1].split("\ndef ", 1)[0]
    assert "apron_salary" not in body, (
        "check_hard_caps reads apron_salary; it must enforce on the same basis "
        "every other check does")

    uses = src.count("apron_salary")
    assert uses <= 4, (
        f"apron_salary is referenced {uses} times in the engine; it is meant to "
        f"have exactly one reader, the apron-2 note text")

    csvp = root / "data" / "apron_salaries_2026_27.csv"
    assert csvp.exists(), "the sourced apron figures are gone"
    head = csvp.read_text(encoding="utf-8")
    assert "DO NOT POINT check_hard_caps AT THIS FILE" in head, (
        "the warning that keeps the next reader from wiring this into a cap "
        "test has been removed")
    import csv as _csv
    rows = list(_csv.DictReader(
        l for l in head.splitlines() if l.strip() and not l.lstrip().startswith("#")))
    assert len(rows) >= 20, f"only {len(rows)} teams have a sourced apron figure"
    combined = [r for r in rows if (r.get("combined") or "").strip()]
    assert combined, (
        "no row is marked combined, so the file has lost the distinction "
        "between a gap of zero and nobody having looked")
