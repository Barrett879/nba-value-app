# What actually measures defence, using only the current season

**2026-11-01.** Written after Barrett asked why Trae Young scores +2.03 on defence while
being, by every public measure, one of the worst defenders in the league. Three parallel web
scours of the public record, plus direct tests against this repo's own data.

**The constraint:** current-season data only. No prior-season or career priors.

---

## 1. The constraint is narrower than it needs to be

There are two different things people mean by "uses prior seasons", and only one is
disqualifying:

- **Coefficients fit on multi-year data, then applied as constants.** DBPM, RAPTOR's box
  component and Dean Oliver's DRtg were all fit on multi-year RAPM, but once fit the
  coefficients are fixed numbers. Applying them needs only the season being rated.
- **A player-specific prior carried forward.** EPM is "powered by players' entire careers
  rather than single seasons" and starts from a preseason Expected EPM. DARKO weights every
  game since 1996-97 with half-lives up to ~8 seasons. These are genuinely out.

**This repo already accepts the first kind.** `utils._DLEBRON_PROXY_COEFS` is a regression
fit on the 2009-2025 sample and applied to whatever season is being scored. Accepting DBPM
is the same move, not a new one.

## 2. The on/off family is the worst-behaved defensive measurement there is

Independent confirmation of what the blend's +0.535 was telling us. Basketball Index,
2,895 player-season pairs, 2013-14 to 2025-26, 750+ minutes in consecutive seasons:

| measurement | year-over-year |
|---|---|
| box-prior-anchored and talent-anchored defence | **0.87** |
| BPM | 0.78 |
| defensive four-factor on/off impact | **0.21 - 0.47**, none reaching 0.54 |
| opponent eFG% impact, even with a FIVE-season window | **0.28** |
| defensive RAPM | 0.35 |

BPM's own author calls raw one-season on/off "basically useless". Basketball-Reference is
candid about its own metric too: *"Box Plus/Minus is a very good offensive metric, but it
struggles some with defense... take DBPM with a spoonful of salt."*

**Sample sizes required.** Ilardi and Barzilai put single-season adjusted plus-minus standard
errors "typically around 5.0 points per 100 possessions", margins of error near ±10, falling
~40% with two seasons and to ~1.1 with five. Medvedovsky puts five-man lineup **defensive**
rating stabilisation at ~850 possessions against ~550 for offence — and only **11 lineups
league-wide** reached 850 in a season. DARKO itself weights observed on/off at
`poss/(poss+6000)` on defence, and a full season as a starter is only ~5,000 possessions.

## 3. Signal arrives in rates and frequencies, last in percentages

DARKO publishes per-stat decay half-lives, which order the channels directly:

| channel | half-life |
|---|---|
| share of shots from three | **~13 games** |
| blocks | ~41 games |
| defensive rebounds | ~46 games |
| steal rate | ~75 games |
| three-point % allowed | ~1.5 seasons |
| corner-three % allowed | **~8 seasons** |

**Our own disjoint split-half on this season agrees closely** — first 3 games against the
next 4, n=64 rotation players:

| channel | measured here |
|---|---|
| DREB/36 | **+0.516** |
| BLK/36 | +0.339 |
| PF/36 | +0.217 |
| STL/36 | **+0.066** |

Blocks and defensive rebounds carry signal early; steals barely do. That is an independent
convergence between DARKO's published half-lives and this repo's own data, arrived at
separately.

## 4. What we measured directly, on this repo's data

Each on/off channel rebuilt from only the first *k* games of 2024-25, scored against that
season's real D-LEBRON (Spearman):

| channel | 5 games | 10 games | 18 games |
|---|---|---|---|
| `d_oreb` | +0.149 | +0.145 | +0.219 |
| `d_rim` | +0.073 | +0.190 | +0.202 |
| `d_pps` | +0.050 | +0.170 | +0.188 |
| `d_tov` | +0.134 | +0.075 | +0.142 |
| `d_cor` | +0.032 | **-0.039** | **-0.054** |

And whole-metric comparisons on the same players:

| estimator | Spearman vs real D-LEBRON |
|---|---|
| DBPM | **+0.801** |
| 0.8 DBPM + 0.2 on/off blend | **+0.831** |
| the shipped blend | +0.535 |
| the box proxy already in utils | +0.520 |
| on/off defensive rating (pts/100 off - on) | +0.173 |

**The shipped blend's +0.535 is its box half.** The on/off machinery, where all the
complexity lives, adds almost nothing on top of the box proxy's +0.520.

## 5. Where the research and our data DISAGREE

Worth recording rather than smoothing over.

A Northwestern zone study found that for corner threes, **allowed volume** beats allowed
percentage as a predictor of defensive rating — the reverse of the restricted area — so one
agent judged the corner-3 channel "defensible" and the rim-share channel the weak one.
Our `d_cor` **is** volume (corner-3 share allowed, not percentage), and DARKO's ~13-game
half-life for three-point share says it should stabilise fastest of anything we have.

But measured against actual D-LEBRON it scores **-0.05**. A channel can be *stable* and
still not be *valid*: shot-diet suppression is highly repeatable year to year, yet a
team-level replication found rim-suppression skill correlates **r = -0.01** with how often
opponents actually attack the rim, while opponent rim FG% correlates **+0.76** with opponent
points per possession. Stability is not value. We tested validity.

## 6. The honest caveat on our own benchmark

DBPM's +0.801 against D-LEBRON is **partly circular**. D-LEBRON is RAPM shrunk toward a box
prior and DBPM carries a team-adjustment constant, so the two share team-defence variance
that no play-by-play player channel can reproduce. For the blend's actual purpose -- standing
in for D-LEBRON until BBall Index publishes -- sharing its prior is a feature. As evidence
that DBPM measures defence *in the world*, it is weaker than the number looks.

No head-to-head accuracy study between defensive metrics exists anywhere in the public
record. 538's own comparison of defensive RAPTOR, DRPM, DEPM and D-LEBRON reported only that
they disagree about everyone except Rudy Gobert, with no coefficients.

## 7. What this implies for the build

1. **Drop or heavily downweight the on/off channels.** They are the noisiest defensive
   measurement in the published record and our own test puts every one of them under +0.22.
2. **Make DBPM the spine.** It is legal under the constraint, it is already computed in
   `def_lib.players()`, and it beats the current formula by 0.27 Spearman.
3. **Shrink hard toward a current-season league or positional mean while the sample is
   thin.** No published figure supports an individual defensive metric at seven games; the
   smallest sample in the literature with a measured reliability is a half-season of
   perimeter shot defence at split-half rho = 0.17. Shrinking toward a current-season mean
   uses no outside data, so it is legal.
4. **Weight the fast channels** -- blocks, defensive rebound rate, shot-location shares --
   over steal rate and anything percentage-allowed.
5. **If any on/off is kept, luck-adjust it first.** Substituting expected opponent 3P% and
   FT% for actual is the cheapest large noise reduction available: published examples move a
   defensive on/off figure by 7.3 to 10.1 points per 100 over a full season, and LA-RAPM
   recovers ~9 correlation points over raw RAPM.

## Sources

- Basketball-Reference, BPM 2.0 methodology: https://www.basketball-reference.com/about/bpm2.html
- Basketball Index, projecting impact (2,895 player pairs): https://www.bball-index.com/projecting-impact/
- CraftedNBA glossary (CraftedDPM component weights): https://craftednba.com/glossary
- FiveThirtyEight RAPTOR data (box and on/off published separately): https://github.com/fivethirtyeight/data/tree/master/nba-raptor
- Ilardi and Barzilai on adjusted plus-minus standard errors
- DARKO per-stat decay half-lives
- Medvedovsky on lineup-level stabilisation
