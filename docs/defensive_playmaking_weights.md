# Is the composite underrating defensive playmaking?

Barrett, 2026-11-02: "i feel like we are now underrating defensive playmaking
stats, like steals, blocks, charges taken, etc"

Answer: **not in the main branch, and badly in the fallback branch.** The useful
version of the intuition is narrower than the general claim and worth more.

## The test bed

`cache/pbpstats/player_<season>.json` exists for 2015-16 onward and joins to
`cache/dlebron_all.parquet` on `EntityId` = `nba_id` with **100% coverage and no
name matching**. Eleven seasons overlap with real D-LEBRON. DBPM comes from
`cache/bref_def/advanced_<season>.parquet`, matched by `utils.NameIndex` at 99.7%.

3,955 player-seasons; **3,211** qualify at the project's usual filter (40+ GP,
15+ mpg, DBPM present). That is 11 seasons against the 2 the shipped weights were
chosen on.

Caveat kept in view: real D-LEBRON shrinks RAPM toward a **box prior**, so it is
not a neutral referee for a box-built composite. It is the best one available.

## 1. Steals are already the largest box input we have

Regressing z(DBPM) on the standardised rates, 11 seasons, R2 = 0.562:

| | coefficient inside DBPM | implicit weight (x0.60) | explicit | **effective** |
|---|---|---|---|---|
| steals | +0.572 | +0.343 | 0 | **0.34** |
| blocks | +0.341 | +0.205 | 0.15 | **0.36** |
| def rebounds | +0.268 | +0.161 | 0.15 | **0.31** |
| charges drawn | +0.053 | +0.032 | 0 | **0.03** |

DBPM is 60% of the composite, so steals already carry about the same effective
weight as blocks. They are not missing; they arrive through DBPM instead of
through a channel with their name on it.

## 2. Adding them to the fast channel makes it worse

Fit on 2015-16..2021-22 (n=2030), test on 2022-23..2025-26 (n=1181), Spearman vs
real D-LEBRON averaged per season so no single season dominates:

| candidate | test rho | vs shipped |
|---|---|---|
| shipped: 0.67 z(DBPM) + 0.33 z(blk+dreb) | +0.8441 | |
| fast gains steals | +0.8326 | **-0.0116** |
| fast gains steals and charges | +0.8292 | **-0.0149** |
| steals as its own channel | +0.8437 | -0.0005 |
| charges as its own channel | +0.8465 | +0.0023 |
| playmaking (stl+blk+chg) as its own channel | +0.8477 | +0.0036 |
| blk, dreb, stl, charges all separate | +0.8454 | +0.0013 |

The fast channel is a SUM of z-scores, so a third term cuts blocks and rebounds
from half each to a third each. Their partial correlations with D-LEBRON after
DBPM are +0.456 and +0.459; steals' is about zero. The trade is strong signal for
noise. Nothing here beats the shipped formula by more than noise.

## 3. The negative steals coefficient is a POSITION artifact, not a finding

Raw, after DBPM, steals get beta **-0.146** and that is negative in **11 of 11
seasons**. It would be easy to read that as "steals are actively bad". It is not:

- corr(D-LEBRON, guard%) = **-0.392**
- corr(STL/36, guard%) = **+0.273**

Steals proxy for being a guard, and the referee rates guards lower. Control for
position and the coefficient flips:

| | beta on z(STL/36) |
|---|---|
| after DBPM | -0.146 |
| after DBPM + position | **+0.034** |
| after DBPM + position + blk + dreb | **+0.080** |

So steals carry a small positive marginal value *within position*, and the raw
negative is composition.

**We are not adding an anti-guard bias of our own**, which is the thing worth
checking before changing anything:

| correlated with guard% | |
|---|---|
| real D-LEBRON | -0.392 |
| our composite | -0.373 |
| residual (real minus ours) | **-0.035** |

We reproduce the referee's positional tilt almost exactly. That is calibration,
not vindication: if the Barrett Score should measure "best defender **for his
position**" rather than absolute defensive value, position adjustment is the
lever, and it is a question about the estimand, not about steals.

## 4. The fallback branch is the real hole

A player with no DBPM yet is scored `(0.75 fast + 0.25 onoff)`, renormalised. The
renormalisation replaces the *weight* DBPM carried but nothing replaces what it
*measured*, and what it mostly measured was steals. So for those players
playmaking has no route in at all, explicit or implicit.

Channel candidates, same train/test split, pure sums of z-scores so there is
nothing to overfit:

| fallback channel | train | **test** |
|---|---|---|
| shipped: blk + dreb | +0.6623 | **+0.6130** |
| **blk + dreb + stl** | +0.7528 | **+0.7350** |
| blk + dreb + stl + chg | +0.7329 | +0.6986 |
| blk + stl | +0.6460 | +0.6525 |
| steals alone | +0.2615 | +0.3010 |

**+0.122 on the test seasons, and better in 11 of 11 seasons** (+0.089 to +0.141).

The live board shows it in one line: **Bryce McGowens leads the league in steals
per 36 (5.71) and sits 181st of 353 on defence**, because he has no DBPM yet.
Dyson Daniels, 3.74 per 36 and DBPM present, is 3rd. The playmaking penalty is
entirely in the branch where DBPM is missing.

Effect of adding steals to the fallback channel only, on the live pool:

- 30 of 353 players affected; the 323 with DBPM move by at most **0.0014**
- endpoints unchanged at -4.00 .. +6.00; pool mean +0.4116 -> +0.4144, so this
  does not touch the unresolved mean-shift question
- Spearman 0.9945 against the shipped artifact
- McGowens 181 -> 96, Dean Wade 198 -> 100, Trendon Watford 194 -> 118, and it
  correctly demotes DeAndre Jordan (0.00 steals per 36) 81 -> 115

## 5. Charges: no, twice over

- **The obvious column is the wrong one.** `Charge Fouls` is charges COMMITTED:
  its 2025-26 leaders are Towns, Queen, Wembanyama, Zion, Jaylen Brown. The
  defensive stat is `Charge Fouls Drawn`, led by Brunson, Podziemski, Smart,
  Camara. Using the obvious one would reward offensive fouls.
- **The signal is near zero anyway.** Spearman +0.043 alone; after DBPM + blk +
  dreb, beta +0.058 for R2 +0.0007. Added to the fallback channel it makes it
  *worse* (+0.6986 against +0.7350).
- **And it is too sparse to use in-season.** 21 of 374 players have drawn one
  this season. Over a full season about 30% of qualified players draw none and
  the leader draws 28.

`DefensiveGoaltends` looked promising (+0.310 alone, +0.300 partial after DBPM)
and is not: it correlates +0.609 with blocks and its coefficient collapses to
+0.024 after blocks and -0.009 after blocks and rebounds. It is a proxy for being
a rim protector, which blocks already say.

## Recommendation

1. **Add steals to the fallback channel only.** +0.122 out of sample, 11 of 11
   seasons, 30 players, no rank change for anyone with a DBPM.
2. **Leave the main branch alone.** Steals are effectively weighted 0.34 there,
   about the same as blocks, and every variant tested is flat or worse.
3. **Do not add charges.** Wrong column available live, negligible signal, far too
   sparse.

Open question, Barrett's to settle: whether the metric should be position-adjusted
at all. It would change what the number means against a salary.

Reproduce: the scripts behind every figure here are in the session scratchpad
(`panel.py`, `diag.py`, `robust.py`, `oos.py`, `implicit.py`, `fallback.py`,
`live.py`).
