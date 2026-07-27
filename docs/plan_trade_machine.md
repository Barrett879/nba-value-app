# Trade Machine Hub - design plan

Goal: an interactive trade builder in the Fanspo/Spotrac mold, with HoopsValue's
twist: every trade is graded by the Barrett Score value model, not just legality.
Users pick 2-4 teams, move players/picks/cash between them, and get an instant
verdict: LEGAL (with cap math per team) or ILLEGAL (with the exact rule broken,
in plain English). The "why it's illegal" explanations are the credibility
feature; the value grade ("LAL wins this trade by +$18M of surplus value") is
the shareable feature nobody else has.

## Architecture

Three layers, strictly separated:

1. **`trade_rules.py` - the rules engine (pure, no Streamlit).**
   - `CBAConfig` dataclass: every dollar line and percentage in ONE place
     (cap, tax, apron 1, apron 2, matching bands, cash limit, ...), loaded
     per-season so 2027-28 is a config update, not a code change. Figures come
     from the verified rules spec below, dollar lines from the app's own data
     (`fa_sim_v1.json` carries cap/apron levels).
   - `Team` / `Player` / `Pick` / `TradeLeg` dataclasses; a `Trade` is a set of
     legs. Engine input is plain data - fully unit-testable.
   - `validate(trade, cfg) -> TradeVerdict`: runs an ordered pipeline of rule
     checks, each returning `Violation(code, team, message, rule_cite)` or a
     `Warning` (e.g. "this hard-caps DEN at the second apron for the rest of
     the year"). Messages are written for fans, not lawyers.
   - `cap_impact(trade, cfg) -> per-team before/after`: payroll, tax bill
     direction, apron room, roster count, hard-cap flags triggered.
   - `value_grade(trade) -> per-team surplus value delta` using the board's
     projected market values (the HoopsValue twist).

2. **Data plumbing (build-time, per the no-request-path-work doctrine).**
   - Rosters + salaries: `cache/team_rosters_2627.json` (already reality-first
     with real signings).
   - Contract shape (years left, option type, no-trade flags): extend the
     contract-end scraper output into the roster JSON at build time.
   - Signed dates for trade-eligibility timing: `data/real_signings_2026.csv`
     already carries `signed_date`.
   - **NEW: `data/pick_ledger.csv`** - future first/second ownership per team
     through the 7-year horizon, with protections and swap flags. This is the
     one dataset the repo lacks; curated from RealGM/Spotrac and verified, then
     maintained by hand as trades happen (same doctrine as real_signings).
   - Apron/tax lines: from `fa_sim_v1.json` (cap_M / apron1_M / apron2_M).

3. **`pages/Trade_Machine.py` - the UI (Streamlit).**
   - Team pickers (2-4 teams; pills). Per-team column: roster list with
     salaries (multiselect players out), pick picker from the ledger, cash
     input capped at the annual limit.
   - Live verdict panel on every rerun: one card per team - LEGAL/ILLEGAL
     badge, the violated rules with plain-English reasons, before/after
     payroll bar vs tax/apron 1/apron 2 lines, triggered hard-cap warnings.
   - Value grade card: surplus-value delta per team + a one-line take
     ("Utah adds $22M in surplus value and a first").
   - Shareable: the whole trade serializes into the URL query string, so a
     built trade is a link; per-trade OG text via the serve.py share plumbing
     later (same trick as `?player=`).
   - Explicitly NOT drag-and-drop (Streamlit can't); the interaction model is
     checklists + live verdicts, which is also how Spotrac's works.

## Build phases

- **Phase 1 (rules core):** `trade_rules.py` + `CBAConfig` + scenario test
  suite (each rule gets a legal case, an illegal case, and an edge case pulled
  from a real historical trade). No UI yet.
- **Phase 2 (pick ledger):** curate + verify `data/pick_ledger.csv`; Stepien
  checks come alive.
- **Phase 3 (UI):** the page, 2-team first, then multi-team, then cash/TPE.
- **Phase 4 (share loop):** URL-serialized trades + OG unfurls; maybe a
  "grade this trade" gallery later.

## Verified 2023 CBA rules spec

Research pass: 6 areas, 13 agents, adversarially verified against the official
2023 CBA PDF. salary-matching: 15 rules, 2 corrections · apron-restrictions: 15 rules, 3 corrections · hard-caps: 18 rules, 7 corrections · picks-stepien: 13 rules, 2 corrections · sign-and-trade: 8 rules, 4 corrections · eligibility-misc: 16 rules, 3 corrections.

# NBA Trade Legality Engine — 2023 CBA Rules Specification

Consolidated from five verified research areas (salary matching, apron restrictions, hard caps, picks/Stepien, sign-and-trade, eligibility/misc). All areas passed an adversarial verification pass against the official 2023 CBA PDF (ak-static.cms.nba.com) and/or NBA CBA-101; every correction from that pass is applied below. No area was left unverified, so no rules carry an area-level "verification pending" cap; individual single-sourced items are listed in Section 9.

**Sourcing doctrine:** Larry Coon's cbafaq.com was never updated for the 2023 CBA (latest edition = salarycap17.htm, 2017 CBA). Never cite it for apron-era rules. Primary source = official 2023 CBA PDF (Art. VII); best secondary = cbaguide.com, HoopsRumors, NBA CBA-101, Sports Business Classroom. Note: the NBA's own CBA-101 summary table uses shifted row letters (omits the transition row); cite only the actual CBA §2(e)(4) table letters (A–K).

---

## 0. Reference figures by season

| Season | Cap | Tax level | First apron | Second apron | Min team salary |
|---|---|---|---|---|---|
| 2023-24 | $136,021,000 | $165,294,000 | $172,346,000 | $182,794,000 | — |
| 2024-25 | $140,588,000 | — | $178,132,000 | $188,931,000 | — |
| 2025-26 | $154,647,000 | $187,895,000 | $195,945,000 | $207,824,000 | $139,182,000 |
| 2026-27 | $164,961,000 | $200,428,000 | $209,015,000 | $221,686,000 | $148,465,000 |

| Exception / amount | 2023-24 (base) | 2024-25 | 2025-26 | 2026-27 |
|---|---|---|---|---|
| Non-Taxpayer MLE | $12,405,000 | $12,822,000 | $14,104,000 | $15,044,000 |
| Taxpayer MLE | $5,000,000 | $5,168,000 | $5,685,000 | $6,064,000 |
| Bi-Annual Exception | $4,516,000 | $4,668,000 | $5,134,000 | $5,477,000 |
| Room exception | — | — | $8,781,000 | $9,366,000 |
| Expanded-TPE flat add | $7,500,000 | ~$7,752,000 (derived) | $8,527,000 | $9,096,000 |
| Trade cash limit (5.15% of cap) | — | — | $7,964,000 | $8,495,000 |
| Two-way salary | — | — | — | $678,882 |

**Indexing rule (corrected):** dollar exceptions grow from fixed 2023-24 base values at the cap-growth rate. The often-quoted percentages (TMLE 3.68%, BAE 3.32%) are descriptive of 2023-24 only — computing from them yields wrong numbers (e.g., 3.68% × 2026-27 cap = $6,071,000 vs the official $6,064,000). Aprons likewise index from the fixed 2023-24 anchors ($172,346,000 / $182,794,000), NOT "tax + $7M" (actual 2023-24 gap was $7,052,000; second-apron gap exactly $17,500,000). Confidence: high. Source: 2023 CBA Art. VII §2(a)(4); official values via pr.nba.com, HoopsRumors, Spotrac; arithmetic verification in review pass.

---

## 1. Salary matching

**1.1 Below-first-apron matching formula (Expanded TPE) — canonical implementation.**
Incoming ≤ **greater of**: (a) **lesser of** [200% of outgoing + $250,000] and [100% of outgoing + indexed flat add], and (b) [125% of outgoing + $250,000]. Implement the formula, not fixed tiers — the "200% band" is not literal (formulas cross at $7.25M in 2023-24 dollars; a $7.4M outgoing allows $14.9M via the add rule, not $15.05M).
- Figures: flat add by season per table above; $250K cushions and percentages fixed forever; 2026-27 explicit crossovers $8,846,000 / $35,384,000; 2023-24 crossovers $7,250,000 / $29,000,000.
- Confidence: high (formula); derived 2024-25 add and 2025-26 crossovers → Section 9.
- Source: 2023 CBA Art. VII §6(j)(1)(iv); HoopsRumors TPE glossary (hoopsrumors.com/2024/06/...traded-player-exception-5.html); CBA Guide (cbaguide.com/transactions/trades/tpe/); SBC trade-matching explainer.

**1.2 Over-first-apron matching.** Post-trade above first apron → incoming ≤ 100% of outgoing, flat, $0 cushion (every $250K cushion in the CBA zeroes out above the first apron, §6(j)(3)). Transition year 2023-24 only: 110% flat, also no cushion (corrected — the "110% + $250K" phrasing is wrong).
- Confidence: high. Source: CBA §6(j)(1)(i)-(iii), §6(j)(3); HoopsRumors tax-aprons glossary; HoopsRumors Sept 2023 matching article (110% flat).

**1.3 Over-second-apron matching.** 100% flat AND: no aggregation of 2+ outgoing salaries, no cash sent, no use of a signed-and-traded player's salary/TPE to take back salary. One outgoing player for ≤100% remains legal; multiple incoming players against one outgoing salary remain legal.
- Confidence: high. Source: CBA §6(j)(1)(ii), §2(e)(4) rows H–J; HoopsRumors tax-aprons glossary.

**1.4 Apron status is evaluated POST-trade.** Each transaction is tested on apron team salary "immediately following such transaction." A second-apron team whose trade drops it below the line in that same deal MAY aggregate/send cash in it. There is no standing "apron team" label for trade legality.
- Confidence: high. Source: CBA §2(e)(2)(i)(A); HoopsRumors; CBA Guide.

**1.5 Simultaneous vs non-simultaneous trades.** Simultaneous: 1+ outgoing players (aggregation allowed below second apron), completes instantly, no exception left behind; aggregated trades MUST be simultaneous and generate no TPE. Non-simultaneous: exactly ONE outgoing player; creates a Standard TPE = 100% of outgoing + $250,000 (cushion dropped if over first apron); 12 months to use on one or more incoming players; unused remainder expires. TPEs cannot be combined with each other, other exceptions, or outgoing salary; cannot sign FAs (except completing a sign-and-trade); cannot be traded; CAN claim a player off waivers. An under-cap team that would cross the cap via the trade cannot create a TPE from it.
- Confidence: high. Source: CBA Art. VII §6(j); HoopsRumors TPE glossary; CBA Guide; SBC.

**1.6 Prior-period TPE freeze for over-first-apron teams (CORRECTED, CBA-text verified).** A team over the first apron may not use a Standard TPE after the end of the regular season in which it arose; a TPE created in an offseason remains usable through the END of the following regular season (not merely to its start). Practical restatement: during a regular season, an over-apron team may use TPEs from the current RS or the immediately preceding offseason; during an offseason it may use none created before that offseason began. Frozen TPEs unfreeze if the team drops below the first apron before expiry. Using a prior-period TPE hard-caps at the first apron (see 4a.6). This is a FIRST-apron rule — media placing it at the second apron are wrong.
- Confidence: high. Source: 2023 CBA Art. VII §2(e)(4) row F(i)-(ii) (pp. 214-215); CBA Guide "Time-Crunch Caveat"; HoopsRumors.

**1.7 Minimum-salary players.** Incoming via the Minimum Salary Exception = $0 matching required, any apron status (contract 1-2 seasons at the minimum, never above minimum, no unwaived trade bonus). Outgoing minimums count at full guaranteed salary. Anti-stuffing: outside the Dec 15-to-trade-deadline window, an aggregation of 3+ outgoing contracts taking back FEWER players may include at most ONE minimum contract (exactly two minimums and nothing else remains legal); the rule is suspended between Dec 15 and the deadline (corrected/confirmed).
- Confidence: high. Source: CBA §6(j)(4)(ii) (verified verbatim); HoopsRumors min-exception + aggregation glossaries; CBA Guide trade-salary page.

**1.8 Two-way players.** Count $0 for matching in BOTH directions; no cap/apron charge; generate no TPE; tradable but not within 30 days of signing the two-way deal; acquiring team needs an open two-way slot (max 3).
- Figures: 2026-27 two-way salary $678,882.
- Confidence: medium ($0-both-ways explicit only in CBA Guide, uncontradicted). Source: CBA Guide trade-salary; HoopsRumors two-way glossary.

**1.9 Under-cap (room) teams.** No matching needed up to cap room plus a fixed $250,000 Room-TPE buffer over the cap; the Room TPE cannot be combined with another exception in one trade. Matching kicks in only when post-trade salary exceeds cap + $250K.
- Confidence: high. Source: CBA Guide Room TPE; HoopsRumors TPE glossary.

**1.10 What salary counts (matchable salary).** Outgoing = guaranteed salary only. Timing: offseason→RS start = guaranteed only; RS start–Jan 7 = guaranteed + earned to date; Jan 8–end of RS = full salary; post-season–Jun 30 = lesser of full current salary and next season's guarantee.
- **Poison pill (CORRECTED):** a player traded between signing a rookie-scale extension and its start counts INCOMING at the average of his final rookie-scale (current) year salary PLUS each season of the extended term (option years included) — not the AAV of the extension alone; outgoing/apron number stays at current salary. Example: Keegan Murray 2025-26 — $11,144,093 out for Sacramento, $25,190,682 in for the acquirer.
- **BYC:** signed-and-traded player under BYC counts OUTGOING at max(prior salary, 50% of new first-year salary); incoming = 100% of new salary (see 6.6).
- Confidence: high. Source: 2023 CBA Art. VII §8(g) (pp. 288-289), §6(j)(5); HoopsRumors non-guaranteed-salary + poison-pill glossaries; CBA Guide trade-salary.

**1.11 Re-aggregation freeze.** A player acquired via any exception (incl. TPE/salary-matching — not cap room) cannot have his salary AGGREGATED in a subsequent trade for 2 months from acquisition (solo re-trade fine immediately). Carve-out: acquired on/before ~Dec 16 → may be aggregated on deadline day or the day before, same cap year (Schroder→Butler example). Exact carve-out wording → Section 9.
- Confidence: high (2-month rule, CBA §6(j)(4)(i) verified verbatim); medium (carve-out edge dates). Source: CBA §6(j)(4)(i); CBA Guide; HoopsRumors aggregation glossary.

---

## 2. First-apron restrictions

A team whose POST-transaction apron team salary would exceed the first apron ($209,015,000 in 2026-27) may not:

| # | Restriction | Figures | Confidence | Source |
|---|---|---|---|---|
| 2.1 | Take back more than 100% of outgoing salary (no cushion) | 100% flat | High | CBA §6(j)(1)(i), §6(j)(3) |
| 2.2 | Acquire a player via sign-and-trade | — | High | CBA §2(e)(4) row C |
| 2.3 | Sign (during the RS) a player waived during that RS whose pre-waiver salary > NT-MLE; a reduced buyout doesn't restore eligibility; offseason waivers exempt | NT-MLE: $14,104,000 (25-26), $15,044,000 (26-27) | High | CBA §2(e)(4) row D; HoopsRumors |
| 2.4 | Use a prior-period Standard TPE (timing per rule 1.6) | — | High | CBA §2(e)(4) row F |
| 2.5 | Sign or acquire (trade/waiver) a player using the Non-Taxpayer MLE | $15,044,000 (26-27) | High | CBA §2(e)(4) rows A-B |
| 2.6 | Sign or acquire a player using the Bi-Annual Exception | $5,477,000 (26-27) | High | CBA §2(e)(4) rows A-B |

Being above the first apron without doing any of these imposes nothing by itself (see 4c.1). First-apron rows in the CBA table are A–G (G = the 2023-24-only Transition TPE).

---

## 3. Second-apron restrictions

A team whose POST-transaction apron team salary would exceed the second apron ($221,686,000 in 2026-27) is subject to everything in Section 2 PLUS:

**3.1 No salary aggregation.** May not combine 2+ outgoing salaries for matching in one trade. Effective 2024-25. Multiple players in a deal each matched separately (or absorbed into separate exceptions) is NOT aggregation. Confidence: high. Source: CBA §6(j)(1)(ii), §2(e)(4) row H.

**3.2 No cash sent in trades.** Any amount > $0. Effective 2024-25. Receiving cash: not restricted by any source, but never affirmatively confirmed → Section 9. Confidence: high (send ban). Source: CBA §2(e)(4) row I; HoopsRumors.

**3.3 No Taxpayer MLE signing (CORRECTED effective date).** TMLE use is second-apron-restricted from 2023-24, not 2024-25 — the §2(e)(5) transition exemption covered only rows F–J. What changed in 2024-25 is the follow-on bar after TMLE use: rows A–E (2023-24) vs rows A–F (2024-25 on). Note "all first-apron transactions" = rows A–G formally, but row G existed only in 2023-24, so A–F is operative. Figures: TMLE $6,064,000 (26-27), max 2 years. Confidence: high. Source: CBA §2(e)(2)(iii)(A)-(B), §2(e)(5), §2(e) Examples 1/3/5; CBA-101.

**3.4 No taking salary back against a signed-and-traded player.** May not use any TPE "in respect of a Player Contract signed and traded" — i.e., can't use the outgoing S&T salary for matching (simultaneously or later) or use an S&T-generated TPE. A pure outbound S&T salary dump (nothing back against it) is legal per CBA text but unconfirmed by a second source → Section 9. Effective 2024-25. Confidence: medium. Source: CBA §2(e)(4) row J; HoopsRumors.

**3.5 Frozen draft pick.** See rule 5.8 for full mechanics. The "Second Apron Team" designation for pick penalties is measured once per year: apron salary above the second apron **as of the start of the team's last regular-season game**. Confidence: high. Source: CBA §2(f)(1)(i) verbatim.

---

## 4. Hard-cap triggers

Core mechanic (the thing trade machines botch): being above an apron ≠ hard-capped. Each restricted transaction is (a) forbidden if the team would end above the relevant apron, and (b) if executed by a team ending below it, imposes a hard cap AT that apron for the rest of the cap year. A team over the second apron that does nothing on these lists is NOT hard-capped and can go higher (Bird re-signings, extensions, minimums). Confidence: high. Source: CBA §2(e)(2); HoopsRumors hard-cap glossary; Spotrac (Keith Smith).

### 4a. Triggers that hard-cap at the FIRST apron

| # | Trigger | Figures | Confidence | Source |
|---|---|---|---|---|
| 4a.1 | Using the NT-MLE to sign beyond Taxpayer-MLE terms (more dollars than TMLE OR more than 2 years — a 3-yr deal at TMLE dollars still triggers) | NT-MLE $15,044,000 / TMLE $6,064,000 (26-27) | High | CBA rows A-B; HoopsRumors; Spotrac |
| 4a.2 | Using any MLE portion to ACQUIRE a player via trade or waiver claim (TMLE is sign-only) | absorption ≤ NT-MLE | Medium | HoopsRumors ×2; CBA Guide |
| 4a.3 | Using any portion of the BAE (sign, trade, or waiver claim) | $5,477,000 (26-27) | High | CBA rows A-B; SBC; CBA Guide |
| 4a.4 | ACQUIRING a player via sign-and-trade (receiving team only) | — | High | CBA row C; HoopsRumors (Klay/DAL example) |
| 4a.5 | Taking back more than Standard-TPE matching (i.e., using the Expanded TPE) | trigger threshold >100%+$250K (exact $250K margin → §9); 2023-24 backtests: threshold was >110% flat | Medium | CBA row E; HoopsRumors; CBA Guide |
| 4a.6 | Using a prior-period Standard TPE (timing per rule 1.6, corrected) | — | High | CBA row F |
| 4a.7 | Signing an in-season-waived player whose pre-waiver salary > NT-MLE | $15,044,000 (26-27) | High | CBA row D; CBA Guide; Spotrac |

### 4b. Triggers that hard-cap at the SECOND apron

| # | Trigger | Effective | Confidence | Source |
|---|---|---|---|---|
| 4b.1 | Signing a player with the Taxpayer MLE (≤ TMLE dollars AND ≤2 yrs; exceed either → it's NT-MLE use → first-apron cap instead) | 2023-24 (corrected) | High | CBA row K, §2(e)(2)(iii) |
| 4b.2 | Aggregating 2+ outgoing salaries for matching | 2024-25 | High | CBA row H (Knicks/Bridges example) |
| 4b.3 | SENDING any amount of cash in a trade (even nominal, e.g., buying a 2nd-rounder) | 2024-25 | High | CBA row I (Warriors June 2024 example) |
| 4b.4 | Using the outgoing signed-and-traded player's salary as matching, or using an S&T-generated TPE | 2024-25 | Medium | CBA row J; HoopsRumors (Kyle Anderson example) |

### 4c. Hard-cap mechanics

**4c.1 Duration.** Trigger moment → June 30 of that cap year; cannot be waived; only shedding salary creates room under it. Confidence: high. Source: CBA §2(e)(2)(i)(B).

**4c.2 Offseason carry-forward.** For a trigger executed between the day after the RS ends and June 30, trade-related transactions (rows E–J) are ALSO tested against the following year's apron using mandated assumptions (all options exercised, no ETOs, no further transactions, rookie-scale higher max achieved, current apron levels carried forward), and the hard cap extends through the END of the following cap year. This is how draft-week deals hard-cap the upcoming season. Confidence: high. Source: CBA §2(e)(2)(ii), §2(e)(3); HoopsRumors; CBA Guide.

**4c.3 The ceiling tests Apron Team Salary (§2(e)(1)),** not regular team salary: cap holds excluded; certain excluded bonuses/unlikely-bonus items added back. Exact unlikely-bonus computation → Section 9. Confidence: medium (bonus details). Source: CBA §2(e)(1); CBA Guide.

**4c.4 Explicit NON-triggers** (engine must NOT hard-cap for these): Room-exception use; Bird/Early-Bird re-signings at any price; veteran extensions; minimum-salary signings/claims via the min exception; absorbing salary into genuine cap room; Standard-TPE matching within limits (below apron 1); receiving cash; outbound S&T with no salary back; being pushed over an apron passively by earned bonuses. Confidence: medium (assembled from exhaustive trigger lists). Source: HoopsRumors hard-cap glossary; CBA Guide.

**4c.5 Transition year (backtests only).** In 2023-24: over-apron matching was 110% flat; the below-apron hard-cap threshold was taking back >110% of outgoing (Thunder/Patty Mills example), not >100%+$250K; the aggregation/cash/S&T/prior-TPE triggers did not exist; the only second-apron trigger was TMLE use. A 2023-24 backtest using current thresholds produces false hard caps. Confidence: high (corrected against HoopsRumors Sept 2023 matching article + CBA §2(e)(5)).

---

## 5. Draft picks / Stepien

**5.1 Stepien rule (corrected provenance).** A team may not make any trade that could leave it without a first-round pick in two consecutive FUTURE drafts — a "possibility" standard (any scenario, however unlikely, violates). This is NOT in the CBA: it is Section 7 of the NBA Constitution and By-Laws ("No Member may... trade or exchange its right to select a player in the first round... if the result... may be to leave the Member without first-round picks in any two (2) consecutive future NBA Drafts"). Second-round picks are fully exempt. Confidence: high. Source: NBA By-Laws §7 (via stepienrules.com, RealGM); HoopsRumors Stepien glossary; CBA Guide.

**5.2 "Future" = drafts not yet held.** The moment a draft concludes, that year exits the window; the current year's upcoming pick counts until its draft is over (hence draft-night deals are executed as select-on-behalf handshakes finalized post-draft). Confidence: high. Source: Coon Q89 (mechanic unchanged); HoopsRumors.

**5.3 Any unconditional first satisfies Stepien.** The team needs A first in each gap year, not its OWN; a protected/conditional incoming pick does NOT count (possibility standard). Corrected example: the Feb 2025 Suns-Jazz deal — Phoenix acquired the least favorable of CLE/MIN/UTA firsts in 2025, 2027, and 2029 (not 2026); the 2027 leg restored Stepien flexibility. Confidence: high. Source: Coon Q89; CBA Guide; ESPN (Suns trade detail).

**5.4 Seven-year rule.** Picks (both rounds) tradable no further out than the 7th draft following the trade date; new eligibility opens as each draft concludes. Confidence: high. Source: Coon Q89; CBA Guide.

**5.5 Pick swaps.** Right to exchange firsts in one specified draft; the granting team still ends that draft with a first, so swaps satisfy Stepien; subject to the 7-draft limit; second-round swaps legal. Confidence: high. Source: HoopsRumors; SBC.

**5.6 Protections.** Max protection top-55; may step down year to year; must resolve within the 7-draft window (extinguish or convert to specified consideration, typically seconds with exact years written in); a protected pick cannot also carry a deferral option. Confidence: high. Source: Coon Q89; CBA Guide.

**5.7 Protections × Stepien arithmetic.** A team owing a protected first can't unconditionally trade a first that could become consecutive with the conveyance year; workaround = "second draft after the first pick conveys" language. Constraints: max 4-year protection when one subsequent conveyance is owed; max 2-year protection to run two; hard limit of 2 such rolling conveyances outstanding; cannot acquire an intermediate pick to accelerate an already-traded conditional pick's "first allowable draft." Confidence: medium (precise year caps single-sourced to Coon; 2-conveyance cap dual-sourced). Source: Coon Q89; CBA Guide; HoopsRumors.

**5.8 Second-apron frozen pick (2023 CBA, from 2024-25).** A team that is a "Second Apron Team" for a cap year (measured per rule 3.5) has its own first in the draft 7 years out frozen (untradable) from the following offseason. If it is a Second Apron Team in 2+ of the next 4 cap years, the frozen pick moves to pick 30 regardless of record (multiple demotions in one draft ordered by inverse prior-season win%); if fewer than 2, the pick unfreezes the day after the RS ends in the 3rd non-second-apron year of that 4-year window. First application: BOS/PHX/MIN 2024-25 → 2032 firsts frozen (earliest unfreeze 2028). Whether a demoted pick regains tradability → Section 9. Confidence: high. Source: CBA §2(f)(1)-(2); HoopsRumors; Forbes.

**5.9 Draft rights.** Rights to unsigned draftees trade freely and immediately (no 30-day wait); count $0 for matching. Unsigned FIRST-rounders carry a team-salary hold of 120% of rookie scale that transfers with the rights (120% carryover to 2023 CBA unconfirmed → §9); unsigned seconds = $0. Stashed rights: players from the 3 previous drafts auto-qualify as tradable "NBA Prospects" (single-sourced → §9). Confidence: medium. Source: Coon Q89; CBA Guide draft-picks page.

**5.10 30-day rule.** Once a draftee SIGNS, no trade for 30 days from signing (applies to late-signed firsts too, and to any two-way signing). A team cannot sign-and-trade its own draft pick, and cannot reacquire him the same season after trading him. Confidence: high. Source: Coon; CBA Guide; HoopsRumors trade-restrictions tracker.

**5.11 Draft-window blackouts.** Lottery picks untradable from 6:00 PM ET the day before the lottery until it completes (high confidence). Current-year picks untradable after 2:00 PM ET on draft day (2:00 PM time single-sourced to CBA Guide → §9). Confidence: medium overall. Source: Coon Q89; CBA Guide.

**5.12 One-time deferral.** A trade may grant a one-time option to defer a pick's conveyance by exactly one year; incompatible with protections; cannot breach the 7-draft limit; once per pick. Confidence: medium. Source: Coon Q89; CBA Guide.

**5.13 Other constraints (low confidence, Coon-only → §9).** No trading picks not yet owned; protection may be added to an acquired pick only if received unconditionally (splitting one unconditional pick between two teams via complementary protections is legal); every traded pick must specify exact conveyance years.

---

## 6. Sign-and-trade / BYC

**6.1 Eligibility.** Only a team's OWN veteran FA: signs with the prior team holding his Bird/Early Bird/Non-Bird rights, and must have finished the prior season on that team's roster. All three Bird tiers qualify (incl. previously renounced players); an RFA is eligible only until he signs an offer sheet. Salary capped by the rights tier (Non-Bird: greatest of 120% of prior salary, 120% of minimum, or QO amount). Cannot use the NT-MLE or Room exception to sign the S&T contract (§8(e)(1)(iii)); TMLE excluded in practice (can't produce 3 years). Confidence: high. Source: CBA §8(e)(1)(i),(iii); HoopsRumors S&T glossary.

**6.2 Contract structure (CORRECTED).** At least 3 seasons **excluding option years** but no more than 4 seasons **including option years** — a 2+1 fails the minimum, 3+1 is legal, 4+1 is illegal. Year 1 fully guaranteed; years 2-4 may be non-guaranteed. Raises max 5% (vs 5 yrs/8% staying put). Starting salary up to the player's normal max, EXCEPT a supermax-qualifying 5th-Year-Eligible player is capped at 25% of cap; Designated Veteran contracts cannot be signed-and-traded. Confidence: high. Source: CBA §8(e)(1)(ii),(iv),(vi), §5(a)(1)-(2) (verified verbatim); NBA CBA-101 2024-25 Trade Rules J(1).

**6.3 48-hour rule.** The trade must execute within 48 hours of the contract signing or the contract is void. Signing bonus may be paid by either team; any portion the signing team pays counts against its annual cash-in-trade limit. Confidence: high. Source: CBA Art. II §3(q)/Exhibit 8 (verified verbatim); §8(a).

**6.4 Deadline.** S&T contracts must be entered into BEFORE the first day of the regular season — no S&Ts once the season starts. There is NO mid-December S&T deadline; Dec 15 is only (a) trade eligibility for normal offseason signees and (b) earliest re-trade of the S&T player. Confidence: high. Source: CBA §8(e)(1)(v); Coon Q92; HoopsRumors.

**6.5 Acquiring team.** Barred if post-trade apron salary would exceed the FIRST apron; completing the acquisition hard-caps at the first apron for the cap year. Must have "Room" (cap space or an available exception, usually the TPE via matching) for year-1 salary + unlikely bonuses. A team that used the TMLE that cap year cannot acquire via S&T (row C is a First-Apron-Level transaction; TMLE users are barred from all First-Apron-Level rows — formally A–G, operatively A–F since row G was 2023-24 only; corrected enumeration). Confidence: high. Source: CBA §2(e)(2)(i),(iii)(B), §2(e)(4) row C, §8(e)(1)(vii); CBA Guide; HoopsRumors.

**6.6 Base Year Compensation (CORRECTED trigger wording).** Applies when ALL hold: player re-signs via Bird/Early Bird rights as part of the S&T; team salary is strictly **above** the cap immediately after the signing (CBA §6(j)(5)(y) — "at or above" is a paraphrase; a team exactly at the cap is not literally covered, measure-zero case); new salary above the minimum with a raise >20% (i.e., more than what Non-Bird rights could pay). Effect: SENDING team's outgoing matching number = greater of (a) last-season salary of the prior contract, (b) 50% of new first-year salary; ACQUIRING team counts 100% of the new salary incoming. Minimum contracts excluded; for a player off a 1-year minimum, the league-reimbursed portion counts in "previous salary." Applies only to the trade leg — the new team books his real salary. Example: Walker Kessler $4,878,938 prior / $30,108,821 new / $15,054,411 outgoing. Confidence: high. Source: CBA §6(j)(5) (verified verbatim); HoopsRumors BYC glossary.

**6.7 Aggregating the S&T salary.** No S&T-specific ban — the sending team may aggregate the newly signed (BYC-deemed) contract with other outgoing salaries, subject to standard limits (second-apron aggregation ban; minimum-contract anti-stuffing; the 2-month freeze in §6(j)(4)(i) covers only contracts ACQUIRED via exception, not newly signed ones). When aggregated, the BYC-deemed number enters the aggregate. Confidence: medium (permission implied by CBA structure, not stated affirmatively). Source: CBA §6(j)(4); HoopsRumors.

**6.8 Re-trading the S&T player.** The initial S&T trade is exempt from the new-signee freeze; the freeze applies in full to the SECOND trade: later of 3 months from signing or Dec 15 — extended to later of 3 months or JAN 15 if the BYC-style criteria are met (over-cap Bird/Early Bird re-signing, >120% raise, non-minimum). The acquiring team also cannot AGGREGATE him for 2 months. Confidence: medium (Jan 15 application to the second trade rests on reading §8(d)(ii)+(iii) with Coon's gloss → §9). Source: CBA §8(d)(ii)-(iii) (8(d)(ii) initial-trade exemption verified verbatim); Coon Q93/Q96.

**6.9 Sending team.** Signing-and-trading a player AWAY has no apron precondition and no hard cap by itself. But taking salary back against the S&T contract (using its TPE or its salary as matching, simultaneously or later) is a Second-Apron transaction (row J): prohibited above the second apron, hard-caps at the second apron if done below it. Net: an over-second-apron team can only S&T a player out for nothing. The TPE received is computed on the BYC-deemed outgoing amount; its $250K allowances zero out above the first apron. Confidence: high. Source: CBA §2(e)(4) row J, §6(j)(3),(5); HoopsRumors.

**6.10 Extend-and-trade (CORRECTED blackout start).** No extend-and-trade agreement from **the last day** of the last regular season covered by the contract (or any season that could be the last via option/ETO non-exercise) through the following June 30 — the blackout begins ON that day, not the day after. From 2024-25 signings: max 4 seasons from signing, 5% raises, 120% first-year limit. Confidence: high. Source: CBA §8(e)(2)(i)-(iii), §5(a)(4) (verified verbatim); HoopsRumors CBA-changes list.

---

## 7. Player eligibility / timing

**7.1 Newly signed free agents.** Untradable for 3 months from signing or until Dec 15, whichever is later (Sept 15 is the effective pivot: signed on/before it → eligible Dec 15). Applies to teams matching RFA offer sheets and to two-way conversions (3 months from conversion or Dec 15, later). Exemptions: draft picks (30 days, rule 5.10), the initial S&T trade (rule 6.8). Confidence: high. Source: CBA §8(d); Coon Q96; HoopsRumors trackers.

**7.2 January 15 rule.** Freeze extends to later of 3 months or Jan 15 when ALL: re-signed with prior team via Bird/Early Bird; team over the cap immediately after signing; salary above minimum; raise >20% (CBA phrasing "greater than 20%"; HoopsRumors says "at least" — only matters at exactly 20.0%). Distinct from the poison pill and from consent rules. Confidence: high. Source: CBA §8(d)(iii); Coon Q93; HoopsRumors special-eligibility-dates; Spotrac.

**7.3 Implicit veto (one-year Bird re-signees).** A player who re-signed with his prior team on a 1-year deal (or 2-year with option year) and will hold Bird/Early Bird rights at season's end cannot be traded without consent (includes QO acceptors). Consenting downgrades his rights to Non-Bird; consent required again for any later trade that season (re-consent detail Coon-only → §9). New in 2023 CBA: waivable at signing (Porter Jr./Trent Jr. 2025-26). Confidence: high. Source: Coon Q101; HoopsRumors veto list.

**7.4 Negotiated no-trade clauses.** Require 8+ years NBA service AND 4+ seasons with the signing team; new FA contract only (not an extension unless the prior deal had one); carries to the new team after a consented trade. ("Partial seasons round up" is unverified → §9.) Confidence: high. Source: Coon Q101; HoopsRumors.

**7.5 Matched offer sheet.** For 1 year after matching, the player's consent is required for ANY trade, and he can NEVER (even with consent) be traded to the offer-sheet team that year. Standard Dec-15 freeze also applies. Confidence: high. Source: Coon Q101; HoopsRumors.

**7.6 Reacquisition bans.** A team cannot reacquire by trade a player it traded away during the same season (RS day 1 through last day of Finals). Offseason trades: no reacquisition until the end of the following season (Coon-only wrinkle → §9). If the acquiring team WAIVES him, the original team cannot re-SIGN him until the earlier of the 1-year trade anniversary or the July 1 after his contract's last season (Ilgauskas rule; confirmed by CBA Guide). Bogut loophole: flipped to a third team and waived there → original team may re-sign. Draft-rights-only trades exempt. Confidence: high (core), medium (offseason wrinkle). Source: Coon Q101; CBA Guide; SI case reports.

**7.7 Fixed waiting periods.** (a) Signed draft picks: 30 days from signing. (b) Designated Veteran (supermax) contract/extension: 1 year from signing (SGA → July 7, 2026). (c) Veteran extension exceeding extend-and-trade limits (>4 total years, first-year raise >20%, or raises/decreases beyond 5%): 6 months (Booker/Doncic/Fox 2025-26 examples). (d) Waiver claims: 30 days in-season; offseason claims → 30th day of the following season. Confidence: high for (a)-(b); medium for (c)-(d) (exact 2023-CBA thresholds and waiver timing thinly sourced → §9). Source: Coon Q95/Q96/Q101; HoopsRumors trackers; SBC extensions article.

---

## 8. Cash, trade bonuses, roster

**8.1 Trade cash limits.** Maximum Annual Cash Limit = 5.15% of the cap per cap year (CBA-verbatim percentage): $7,964,000 (2025-26), $8,495,000 (2026-27 — corrected; not $8,496,000). Sending and receiving are SEPARATE per-direction pools tracked independently; per-trade amount limited only by the remaining annual allowance; indexed to cap growth. Confidence: high. Source: CBA §8(a); HoopsRumors cash tracker; SBC.

**8.2 Apron cash restriction.** Over-second-apron (post-transaction): cannot SEND any cash; sending it while below hard-caps at the second apron (4b.3). Receiving assumed unrestricted (→ §9). First apron does not restrict cash. Confidence: high (send)/medium (receive). Source: CBA §2(e)(4) row I; HoopsRumors; SBC.

**8.3 Trade bonuses (kickers) — CORRECTED consequence.** Max 15% of remaining guaranteed base compensation (options and incentives excluded; prorates down in-season). Paid by the SENDING team; earned only on the first trade. Matching asymmetry: receiving team counts the current-season bonus allocation in INCOMING salary; sending team's OUTGOING salary excludes it. Player may waive all or part; waiving triggers a 6-month RENEGOTIATION ban only (not an extension ban — the 6-month extension limit after trades is a separate general rule). Auto-reduced so salary + bonus ≤ max salary (or 120% of rookie scale). Allocation follows guaranteed-salary proportions; all-non-guaranteed future years → entire bonus hits the current season. Confidence: high. Source: CBA Guide trade-bonus page; Coon Q99/Q100; HoopsRumors.

**8.4 Roster limits.** In-season: 15 standard + up to 3 two-way (18). Offseason: 21. Minimum 14 standard; below 14 allowed max 2 weeks at a time / 28 total days per season. Under-15 rule: a team not carrying 15 standard contracts has its two-ways limited to 90 combined active games. (Coon's 2 two-way slots / 20 offseason are stale 2017 figures.) Confidence: high. Source: HoopsRumors roster-limits glossary; SLAM; HoopsRumors under-15 article.

**8.5 Roster space at trade (UPGRADED to high per verification).** A team acquiring more players than it sends must have open roster spots BEFORE executing the trade — even if it plans to waive an incoming player immediately. Workaround: waive first (the spot opens when waivers are requested, before clearing), then trade. Confidence: high. Source: CBA Guide general trade rules (explicit); Coon Q86/Q101.

---

## 9. Unresolved and verification-pending items

The engine needs conservative defaults for each. No area had a null verdict; these are the item-level gaps that survived verification.

**Figures derived, not officially published**
1. 2024-25 Expanded-TPE flat add: $7,752,000 is a derivation (cap-ratio math matches the league's rounding pattern); no source printed it. Default: use $7,752,000; only matters for 2024-25 backtests.
2. 2025-26 Expanded-TPE crossovers $8,277,000 / $33,108,000: mathematically entailed by the CBA formula + official $8,527,000 add, but never published. Default: compute crossovers from the formula (safe).
3. Rounding method (round each season independently vs compound rounded figures): unconfirmed, ≤ ~$200 effect. Default: round to nearest $1,000 from exact cap-ratio math.

**Rule-margin ambiguities**
4. Standard-TPE $250K cushion vs the hard-cap trigger: whether a below-apron team taking back exactly 100%+$250K is hard-capped (CBA Guide: no; HoopsRumors: trigger is ">100%"). Default (conservative): treat any take-back above 100% flat as triggering the first-apron hard cap; warn the user in the $0–$250K sliver.
5. Second-apron teams RECEIVING cash: inferred legal (all sources restrict only sending; the HR tracker treats received cash as routine), never affirmatively stated. Default: allow, with a low-severity warning flag.
6. Jan 15 rule governing the SECOND trade of an S&T contract: only construction that avoids absurdity, matches Coon's 2017 gloss, but no post-2023 source states it. Default (conservative): apply later-of-3-months-or-Jan-15 to the second trade when BYC-style criteria are met.
7. Pure outbound S&T salary dump by an over-second-apron team: legal per CBA row J text, no secondary confirmation; some media claim a blanket ban. Default: allow (CBA text controls), flag as contested.
8. Re-aggregation carve-out edges: "acquired before Dec 17" (CBA Guide) vs "on or before Dec 16" (HR), and deadline day vs also day-before. Default (conservative): require acquisition on/before Dec 16 AND aggregation only on deadline day or the day before.
9. Apron Team Salary unlikely-bonus accounting: directionally confirmed (less favorable than cap treatment), exact computation unverified. Default: include unlikely bonuses in apron salary for hard-cap ceiling tests (conservative overcount).
10. Jan-15-rule threshold "greater than 20%" (CBA) vs "at least 20%" (HR paraphrase). Default: trigger at >20.0% exactly per CBA wording; flag the exact-20% case.

**Picks / draft**
11. Draft-day pick-trade deadline of 2:00 PM ET: single-sourced (CBA Guide); mapping onto the two-night draft format unverified. Default: block current-year pick trades from 2:00 PM ET draft day; mark low confidence.
12. "NBA Prospect" 3-previous-drafts auto-qualification for stashed rights: single-sourced (CBA Guide). Default: use it; flag older-rights trades for manual review.
13. Protection-length arithmetic (max 4-year protection with one subsequent conveyance; max 2-year to run two; 2-conveyance cap): year caps verbatim in Coon (2017 CBA) only; 2-conveyance cap dual-sourced. Default: enforce as written.
14. Whether a frozen pick demoted to No. 30 ever regains tradability: unanswered anywhere. Default: treat as permanently untradable.
15. Whether a "least favorable of N teams" incoming pick with protected legs counts as unconditional for Stepien: unaddressed. Default (conservative): any protected component → does not satisfy Stepien.
16. 120% rookie-scale team-salary hold for unsigned firsts carrying into the 2023 CBA: not independently confirmed. Default: apply it.
17. Misc pick constraints in rule 5.13 (no not-yet-owned picks; protection-adding limits; exact-year specificity): Coon-only, low confidence. Default: enforce, flag low confidence.

**Eligibility / misc**
18. NTC detail "partial seasons round up" toward the 4-season requirement: unverified. Default: omit rounding; require 4 credited seasons.
19. Offseason-trade reacquisition wrinkle (banned until end of the FOLLOWING season): Coon-only; CBA Guide states only the same-season ban. Default (conservative): apply the longer ban.
20. Implicit-veto re-consent for each subsequent same-season trade: Coon-only. Default: require consent each time.
21. Oversized-extension 6-month-freeze thresholds (>4 yrs / >20% first-year raise / >5% subsequent): from HoopsRumors 2025 wording; not verified against CBA text. Default: use as stated.
22. Waiver-claim trade timing (30 days in-season; 30th day of next season for offseason claims): Coon-only among sources. Default: enforce.

**Historical / immaterial (backtest-only)**
23. Which rulebook governed June 2024 (post-draft, pre-July-1): new triggers appear to have applied to 2024-offseason transactions within the 2023-24 cap year (Warriors cash example), but effective-date mechanics were never explicitly sourced. Default: apply new-CBA trade triggers to transactions on/after the 2024 draft.
24. Official 2023-24 tax level ($165,294,000) exceeds 121.5% of cap by ~$28K, unreconciled (likely a true-up). Does not affect any apron figure; apron bases reproduce all later years exactly.
25. Two-way $0-both-directions and the $678,882 figure: explicit in one source each, uncontradicted. Default: use as stated (rule 1.8 already capped at medium).
