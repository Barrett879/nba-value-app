# Sources that rank defenders for the season in progress

Swept 2026-11-01 for 2026-27. 50 candidates found, 6 confirmed live by a second agent
that loaded the page or endpoint itself. Companion to `defensive_metrics_research.md`,
which covers METHOD; this one covers WHAT IS PUBLISHED AND REACHABLE.

## The dividing line

Barrett's constraint is "I just want it to use this season's stats". That splits the field
cleanly, and it is a sharper cut than free-versus-paid:

- **Carries a prior** (DARKO, EPM, DRIP, LEBRON, CraftedDPM): fine to eyeball the board
  against, defensible as a validation target with a caveat, DISQUALIFIED as an input.
- **This season only** (NBA.com tracking, pbpstats, BBRef, Cleaning the Glass, deep_rapm,
  the DPOY markets): the only family eligible to become a term in the composite.

The sharpest illustration of why it matters, verified in DARKO's own payload: it ranks
**Jonathan Isaac 5th-best defender in the NBA with ZERO games played this season**, and
Steven Adams 3rd on one game and 4.9 minutes. DARKO says so itself: "It is a forecast ...
It is not a summary of one season." Its defensive half-lives are 41 games for blocks, 46
for defensive rebounds, 75 for steal rate, so at 6 games the current season is a minority
of the signal.

## Confirmed live with 2026-27 data

### DARKO DPM

- https://www.darko.app/?sort=d_dpm
- season-only: **no** | access: free
- machine readable: Yes. CSV export button on the leaderboard; sort is URL-addressable (?sort=d_dpm). No public API documented. No robots.txt restriction found (darko.app 302s to www.darko.app). About page states no terms of use or licensing.
- method: Daily-updating Bayesian box-score projection: exponentially decayed game weights (beta^t, tuned per stat) combined with a modified Kalman filter, blended by a gradient boosted tree. Coverage from 1996-97. Separate OFF/DEF split.
- seen live: Loaded 2026-11-01. Header: "Daily Player Metrics for every NBA player. Ratings through October 31, 2026." Season selector top entries "Current / 2026-27 Season". Card: "TOP DEFENSE +3.9 Rudy Gobert MIN 0.7 ahead of Hartenstein". Sorted by DEF: 1 Rudy Gobert MIN +3.9, 2 Isaiah Hartenstein OKC +3.2, 3 Steven Adams HOU +3.2, 4 Clint Capela HOU +3.0, 5 Jonathan Isaac ORL +2.8, then Jakob Poeltl +2.5 and Victor Wembanyama +2.5, Alex Caruso +2.3, Chet Holmgren +2.3, Draymond Green +2.2. Prior-dominance tell: Adams shows 4.9 MPG and Capela 8.2 MPG yet rank 3rd/4th.
- use: (a) — best free live sanity check available today, and the only one with a populated defensive column plus CSV. Weak for (b) because its DEF is mostly prior at 6 games, so Spearman would measure prior-agreement rather than accuracy. Not (c): fails the current-season-only rule outright.

### Dunks & Threes — Expected EPM / DEPM

- https://dunksandthrees.com/epm?m=def
- season-only: **no** | access: free to view the web table; subscriptions are Basic $24.95/yr, Premium $49.95/yr, Premium + API $250.00/yr (REST API Access, Download Data (CSV))
- machine readable: HTML table, sort addressable via ?m=def. CSV download and REST API are paid. IMPORTANT: robots.txt is "User-agent: * / Disallow: /api/" — the JSON API path is explicitly disallowed to crawlers, though the /epm HTML page is not. A Terms of Use link exists in the footer (©2026 Snarr Data Science LLC, All rights reserved); read it before any automated collection. API note: "Version 1 of the API remai
- method: Two SPM models (a box-score model and a tracking model covering defense) each fit to multi-season RAPM, then used as the prior for a nightly exponentially decayed RAPM run over every possession since 2001-02.
- seen live: Loaded 2026-11-01. Page states "Expected impact and stats for Sunday, November 1, 2026 using preceding career data." Season banner 2026-27. Sorted by DEF: 1 Victor Wembanyama SAS +5.0, 2 Rudy Gobert MIN +4.7, 3 Ausar Thompson DET +4.7, 4 Alex Caruso OKC +4.3, 5 Chet Holmgren OKC +3.9, then Dereck Lively II +3.8, Isaiah Hartenstein +3.3, Jonathan Isaac +3.2, Walker Kessler +3.0, Franz Wagner +3.0.
- use: (a) only. It is the single best-known public DEPM and it is live, so it is useful to eyeball against — but it is a career-prior projection dated to today, not a measurement of this season, so treat disagreement with your board as expected. Disqualified from (c) by Barrett's rule.

### NBA.com official tracking defense (stats.nba.com via nba_api)

- https://www.nba.com/stats/players/defense-dash-overall
- season-only: **yes** | access: free
- machine readable: Fully. JSON endpoints, already installed via nba_api in the repo. Note the NBA's terms restrict commercial reuse; worth a read before anything public-facing.
- method: Optical/player tracking: shots defended as the closest defender, opponent FG% allowed versus that shot's expected FG%, plus matchup, hustle and rim-protection categories.
- seen live: Verified by running it from this machine on 2026-11-01, not by assumption. LeagueDashPtDefend(season='2026-27', defense_category='Overall') returned 427 rows with columns CLOSE_DEF_PERSON_ID, PLAYER_NAME, AGE, GP, G, FREQ, D_FGM, D_FGA, D_FG_PCT, NORMAL_FG_PCT, PCT_PLUSMINUS; GP values 3-5. Top 5 by opponent FG% allowed vs expected, min 25 D_FGA: Damian Lillard -0.223, Chet Holmgren -0.214, Derrick Jones Jr. -0.196, Justin Champagnie -0.179, LeBron James -0.160. Lillard topping a defensive list is a fair illustration of how noisy this is at 5 games.
- use: (c), and it is the strongest candidate for a new input because it is tracking-based and therefore independent of the DBPM that already carries 0.60 of your composite. Two caveats: it is components, not a ranking, so it cannot serve as (b); and stats.nba.com is blocked from the production host, so this has to be baked into a committed versioned artifact rather than read on a request path.

### PBP Stats

- https://www.pbpstats.com/on-off/nba/player
- season-only: **yes** | access: free (Patreon-supported)
- machine readable: Yes — a documented API exists (api.pbpstats.com) and a pbpstats Python client is published. Also offers WOWY, lineup subunits and possession finder.
- method: Derived directly from play-by-play: possession-accurate on/off splits, lineup and WOWY queries, opponent efficiency by shot zone.
- seen live: Loaded in a browser 2026-11-01. The on/off player tool's Season control is populated and defaults to "2026-27" with Season Type "Regular Season"; the only season string on the page is 2026-27. Site advertises "Live Stats During Games" and "NBA Stats going back to 2000-01". It is on/off and lineup data, so it produces no defender ranking and no top-5 list.
- use: (c) as a cleaner replacement or complement for the lineup on/off term that is currently 0.10 of your composite — its possession counts are more careful than the NBA's own (it excludes end-of-quarter heave possessions). Also (a) for spot-checking an individual player. Not (b): no ranking.

### Cleaning the Glass

- https://cleaningtheglass.com/stats/players?stat_category=defense_rebounding
- season-only: **yes** | access: paid — "Check out the stats free for one week, then just $5/month after that"; "Subscribe for $50/year" with "two months free when you subscribe for a full year"
- machine readable: Poorly. No CSV export or API found; screen-only behind login. Its usage statement is "use any stats from this site in articles, on podcasts, or on social media" with attribution — that permits citation, NOT bulk scraping or ingestion into a model. I would not treat it as a licence to automate.
- method: Play-by-play on/off with garbage time removed; opponent shooting frequency and accuracy splits by zone, halfcourt/transition/putback breakdowns. Component stats, not a single rating.
- seen live: Season selector runs from "2026-27" down to "2003-04" with Regular Season / Playoffs types, so the current season is offered. I did not sign in, so I could not read actual player values or produce a top-5 — the leaderboard body was not visible to me. Defensive categories listed: "Defense and Rebounding", "On/Off Opponent Shooting: Frequency", "On/Off Opponent Shooting: Accuracy", "On/Off Opponent Halfcourt & Putbacks", "On/Off Opponent Transition".
- use: (a) — a good human eyeball check, and the garbage-time filtering makes it unusually trustworthy this early. Not (b) or (c): there is no single defensive rating, no machine-readable export, and the attribution statement does not cover automated ingestion.

### Basketball-Reference 2026-27 Advanced (DBPM / DWS / DRtg)

- https://www.basketball-reference.com/leagues/NBA_2027_advanced.html
- season-only: **yes** | access: free
- machine readable: Yes — HTML table with CSV/share export, and widely mirrored. Sports Reference's terms restrict bulk automated scraping; their published guidance is a rate limit of roughly one request every few seconds.
- method: Box plus-minus: a regression of box-score production onto team performance, split into OBPM and DBPM, for the selected season only.
- seen live: Live and populated: title "2026-27 NBA Player Stats: Advanced", G column shows 7 for the top rows (Tyrese Maxey, Jabari Smith Jr., Brandon Miller, Amen Thompson, Kevin Durant). OBPM, DBPM and BPM are all columns. Top DBPM values are dominated by 1-2 game micro-samples (Mike Conley 15.2 in 1 G / 7 MP; Maliq Brown 13.3 in 1 G / 1 MP), which is itself a useful reminder of what unshrunk DBPM looks like at this point. Among rotation players the leaders are Cooper Flagg 4.6 and Victor Wembanyama 3.3.
- use: (a) partially, and you already have it. I am listing it mainly to rule it OUT for (b): DBPM is 0.60 of your own composite, so scoring Spearman against it would be largely circular and would mostly measure your own shrinkage and stretch. It cannot serve as an outside opinion.

### NBA.com Players Defense dashboard (DEF RTG, DEF WS, STL/BLK/DREB, opponent points splits)

- https://www.nba.com/stats/players/defense
- season-only: **yes** | access: free, no account
- machine readable: Yes via the JSON endpoint stats.nba.com/stats/leaguedashplayerstats. The HTML page is JS-rendered and returned no table to WebFetch. No anti-scraping clause in the Terms, but Section 9 restricts NBA Statistics to "legitimate news reporting or private, non-commercial purposes" and bars use "in connection with any website, product, or service that features a database (in any medium or format) of com
- method: Team points allowed per 100 possessions while the player is on the floor, plus his own counting stats (DREB/STL/BLK) and opponent points off turnovers / second chance / fast break / in paint. DEF_WS is an undocumented minutes-weighted allocation of that team rating.
- seen live: nba_api LeagueDashPlayerStats season="2026-27", measure_type_detailed_defense="Defense" returned 441 rows, max GP 7. DEF_WS leaders (Totals): Tyrese Maxey PHI 1.49, Jaylen Brown PHI 1.48, Jaden McDaniels MIN 1.44, Rudy Gobert MIN 1.44, Herbert Jones NOP 1.23, Wembanyama SAS 1.23. With measure_type="Advanced" the DEF_RATING leaders at MIN>=20 were Herbert Jones 89.0, Gobert 93.5, Zach Edey 94.9, Ausar Thompson 94.9, Kawhi Leonard 95.0. Web page itself rendered no table to WebFetch (JS-only).
- use: (a) only, and even then with care. NOT (b) and NOT (c): I measured DEF_WS vs -DEF_RATING at rho=+0.93 and its top 10 clusters by team (three PHI, three MIN, with Maxey #1 and LaMelo #9), so it is team defense times minutes, not individual defense. Using it as a Spearman target would reward your composite for tracking team context.

### NBA.com Defensive Dashboard - Defended Field Goal % and DIFF% (overall and by shot distance)

- https://www.nba.com/stats/players/defense-dash-overall
- season-only: **yes** | access: free, no account
- machine readable: Yes via stats.nba.com/stats/leaguedashptdefend (defense_category accepts Overall, 3 Pointers, 2 Pointers, Less Than 6Ft, Less Than 10Ft, Greater Than 15Ft). Same Terms Section 9 use restriction as above; no anti-scraping clause.
- method: Optical tracking: for every shot, the nearest defender is credited, and his opponents' FG% is compared against those same shooters' own season averages (DIFF%).
- seen live: nba_api LeagueDashPtDefend(league_id="00", season="2026-27", per_mode_simple="Totals", defense_category="Overall") returned 427 rows, max GP 7, median GP 5. Best DIFF% at D_FGA>=40: Derrick Jones Jr. -0.196 (.271 vs .466 normal), LeBron James -0.160, Trevon Brazile -0.139, Deni Avdija -0.137, Wembanyama -0.136 (106 D_FGA), Jamal Murray -0.127, Amen Thompson -0.125, Draymond Green -0.113. defense_category="Less Than 6Ft" returned 397 rows; best rim DIFF at FGA>=20: Peyton Watson -0.313, Donovan Clingan -0.308, Daniel Gafford -0.223, Jalen Smith -0.206, Luke Kornet -0.198, Towns -0.190, Gobert -0.173. Pages https://www.nba.com/stats/players/defense-dash-overall and .../defense-dash-lt6 both HT
- use: (c), the best candidate from this lens, and (a). It is a genuinely individual measurement, it is not in your composite, and I measured it as near-orthogonal to deflections (rho=-0.02), so it adds information rather than duplicating the BLK/DREB term. Caveat for now: median D_FGA among qualified players is only 52 and rim FGA only 12, so it needs a volume gate and heavy shrinkage at 6 games. Weak for (b) - rho vs your blend is only +0.307.

### NBA.com Defensive Impact (rim protection: BLK, STL, DREB, opponent FG% at the rim)

- https://www.nba.com/stats/players/defensive-impact
- season-only: **yes** | access: free, no account
- machine readable: Yes via stats.nba.com/stats/leaguedashptstats with pt_measure_type=Defense. Same Terms Section 9 use restriction; no anti-scraping clause.
- method: Optical tracking counts of shots at the rim taken against a player credited as the rim defender, expressed as opponent FG% allowed, alongside raw STL/BLK/DREB.
- seen live: nba_api LeagueDashPtStats(season="2026-27", player_or_team="Player", pt_measure_type="Defense") returned 441 rows with columns STL, BLK, DREB, DEF_RIM_FGM, DEF_RIM_FGA, DEF_RIM_FG_PCT. Unfiltered the leaderboard is pure noise (Damian Lillard and Mikel Brown Jr. at .000 on 0.2 and 0.0 rim FGA per game; Luka Doncic .267 on 3.0). Median DEF_RIM_FGA among my MIN>=100 cohort was just 12 total. Page HTTP 200.
- use: (c) for bigs specifically, and (a). Opponent rim FG% is a real individual signal not in your composite, and sign-flipped it hit rho=+0.322 against your blend. It is useless without a volume gate at this point in the season, and it says nothing about perimeter defenders, so it would belong as a position-conditioned term rather than a league-wide one.

### NBA.com Hustle stats (deflections, contested shots, charges drawn, loose balls, screen assists, box-outs)

- https://www.nba.com/stats/players/hustle
- season-only: **yes** | access: free, no account
- machine readable: Yes via stats.nba.com/stats/leaguehustlestatsplayer (per_mode_time PerGame or Totals). Same Terms Section 9 use restriction; no anti-scraping clause.
- method: Pure counting stats of defensive effort events from tracking and league statisticians. No model, no possession weighting, no opponent adjustment.
- seen live: nba_api LeagueHustleStatsPlayer(season="2026-27") returned 422 rows, max G 7, all 28 columns populated. Per-game deflection leaders at MIN>=20: Dyson Daniels ATL 6.00, Paul George 5.50, Kris Dunn 5.20, Kawhi Leonard 5.00, Reed Sheppard 5.00, Bam Adebayo 4.67. Totals population check: CONTESTED_SHOTS 6928 league-wide across 406 players; DEFLECTIONS 3141 across 383; DEF_BOXOUTS 901 across 270 (Wendell Carter Jr. 22, Moussa Diabate 12, Naz Reid 12, Wembanyama 11); SCREEN_ASSISTS 1288; but CHARGES_DRAWN only 75 league-wide across 57 players, max 3 (Davion Mitchell, Aaron Nesmith, Jalen Brunson, Draymond Green). Page HTTP 200; https://www.nba.com/stats/players/box-outs also HTTP 200.
- use: (a) and a qualified (c). Deflections/36 and contested shots/36 are the strongest perimeter-defense signals the NBA publishes and are absent from your composite; deflections is near-orthogonal to DIFF% (rho=-0.02), so it genuinely adds. Contested shots hit rho=+0.426 against your blend and deflections only +0.201. Do NOT use charges drawn (75 events league-wide, max 3) or screen assists (an offensive stat) as defensive inputs. Not usable for (b): no single hustle column is a defender ranking.

### NBA.com individual matchup data (who guarded whom, per matchup)

- https://stats.nba.com/stats/leagueseasonmatchups
- season-only: **yes** | access: free, no account
- machine readable: Yes, JSON only (no HTML page at all). Requires one call per defender if you filter by def_player_id, so a full-league pull is ~450 requests and needs rate limiting. Same Terms Section 9 use restriction.
- method: Raw per-possession assignment of defender to offensive player from tracking, with the shooting and turnover outcomes of each pairing.
- seen live: nba_api LeagueSeasonMatchups(season="2026-27", season_type_playoffs="Regular Season", def_player_id_nullable=1641705) returned 59 matchup rows for Wembanyama with 27 columns including MATCHUP_MIN, PARTIAL_POSS, PLAYER_PTS, MATCHUP_FG_PCT, MATCHUP_TOV, MATCHUP_BLK, HELP_BLK, SFL. Top rows: vs Joan Beringer 6:59 / 34.8 partial poss / 1-of-3 allowed; vs Domantas Sabonis 6:15 / 4-of-7; vs Alperen Sengun 6:15 / 4-of-9; vs Chet Holmgren 3:45 / 0-of-1. Note there is NO public web page: https://www.nba.com/stats/players/matchups, /stats/matchups, /stats/players/matchups-overall and /stats/teams/matchups all returned HTTP 404. API only.
- use: Raw material for (c) only, not a ranking in itself. This is the input you would need to build matchup-difficulty or opponent-quality adjustment yourself. At 6 games the per-pairing samples are tiny (Wembanyama's largest single matchup is 7 minutes), so it is a build for later in the season, not now. Worth knowing it exists and is API-reachable since there is no page for it.

### NBA.com Player Estimated Metrics (E_DEF_RATING)

- https://stats.nba.com/stats/playerestimatedmetrics
- season-only: **yes** | access: free, no account
- machine readable: Yes, JSON endpoint. No dedicated HTML page. Same Terms Section 9 use restriction.
- method: NBA's own estimated on-court ratings, derived within-season from box score and team context rather than from play-by-play possession counting.
- seen live: nba_api PlayerEstimatedMetrics(season="2026-27", season_type="Regular Season") returned 441 rows, max GP 7. At GP>=4 and MIN>=20 the E_DEF_RATING leaders were Herbert Jones 89.3, Rudy Gobert 92.9, Daniss Jenkins 94.8, Zach Edey 95.8, Ausar Thompson 96.2, Paul Reed 96.6. Note MIN in this endpoint is per-game, not total, which silently returns an empty frame if you filter as if it were totals.
- use: (a) as a cross-check on the Defense dashboard's DEF_RATING, nothing more. It shares DEF_RATING's fundamental problem of being team defense while on floor, and its leaderboard is nearly identical to it (Herbert Jones, Gobert, Edey, Ausar Thompson appear in both top 6). Not (b), not (c). Flagged mainly so you do not mistake it for an independent NBA impact metric.

### Basketball-Reference — Advanced (DBPM, DWS, BPM, VORP)

- https://www.basketball-reference.com/leagues/NBA_2027_advanced.html
- season-only: **yes** | access: free, no account
- machine readable: HTML table plus the in-page Share & Export "Get table as CSV" widget (JS, not a URL endpoint). No public API. robots.txt sets Crawl-delay: 3 and disallows */on-off/, */splits/, */lineups/, */shooting/, */gamelog/ — the league-wide season pages are allowed, per-player subpages are not.
- method: Single-season box-score regression with a team-quality adjustment; coefficients calibrated once against historical multi-year RAPM. DBPM is a residual (Total BPM minus OBPM), not a direct defensive measurement.
- seen live: Loaded 2026-11-01. H1 "2026-27 NBA Player Stats: Advanced", breadcrumb "2026-27 NBA". Max G = 7. Season index page states "Site Last Updated: Sunday, November 1, 7:42PM". Columns confirmed present: DBPM, OBPM, BPM, OWS, DWS, WS, WS/48, VORP, plus STL%, BLK%, DRB%. Top 5 by raw DBPM as listed: Mike Conley 15.2 (1 G), Pat Spencer 11.4 (1 G), Jalen Pickett 7.6 (2 G), Chaz Lanier 6.9 (2 G), DaRon Holmes 6.9 (2 G); Cason Wallace 6.8 (3 G) next. That leaderboard is pure small-sample noise — BBRef applies no minutes shrinkage.
- use: Already the 60% term, so (c) is in place. Genuine new value is as a documented caution: the source itself calls DBPM a residual and says to take it with a spoonful of salt, and its own all-time leaderboards impose a 1000-minute floor. Barrett's min/(min+150) shrinkage is doing more work than the weight implies, and any eyeball comparison must apply the same floor or it compares noise.

### Basketball-Reference — Play-by-Play page (On-Off per 100, position estimates, defensive fouls)

- https://www.basketball-reference.com/leagues/NBA_2027_play-by-play.html
- season-only: **yes** | access: free, no account
- machine readable: HTML table plus the in-page CSV export widget. Important: robots.txt disallows */on-off/ (the per-player on-off pages) but this league-wide play-by-play page is NOT disallowed, so it is the robots-compliant route to the same quantity. Crawl-delay: 3.
- method: Raw play-by-play aggregation for the single season: on-court net rating per 100, on-minus-off differential, foul and turnover type splits, and a minutes-weighted position estimate.
- seen live: Loaded 2026-11-01. Title "2026-27 NBA Player Stats: Play-by-Play", links to NBA_2027, max G = 7, page last updated November 1. Full column set confirmed: Rk, Player, Age, Team, Pos, G, GS, MP; Position Estimate PG%, SG%, SF%, PF%, C%; "+/- Per 100 Poss" OnCourt and On-Off; Turnovers BadPass, LostBall; Fouls Committed Shoot, Off.; Fouls Drawn Shoot, Off.; Misc. PGA, And1, Blkd.
- use: This is the main find in the BBRef half. (c) first: "+/- Per 100 Poss: On-Off" is a free, league-wide, one-page, robots-allowed cross-check or second source for the 10% lineup on/off term that currently comes only from pbpstats — a cheap way to de-risk a single-source input. The PG%/SG%/SF%/PF%/C% estimates are a second (c) candidate: they would let the z-scores be taken within position rather than league-wide, which is what Cleaning the Glass does and which matters a lot for a blend that mixes 

### pbpstats public API

- https://api.pbpstats.com/get-totals/nba?Season=2026-27&SeasonType=Regular%20Season&Type=Player
- season-only: **yes** | access: free, no account, no API key — the endpoints returned data unauthenticated
- machine readable: JSON, excellent. Self-describing via https://api.pbpstats.com/openapi.json. One call returns all 441 players. No robots.txt (404) and I found no /about or terms page, so I could not verify a stated scraping policy.
- method: Direct play-by-play aggregation for the requested season, with shot-location tagging. No model, no regression, no prior — raw counts and rates.
- seen live: Queried live 2026-11-01, HTTP 200, 1.66 MB, 441 player rows, max GamesPlayed = 7, 205 fields per row. Best OnDefRtg at >=120 min: Herbert Jones 90.3 (127 min), Rudy Gobert 95.5 (180), Daniss Jenkins 97.4 (148), Ausar Thompson 98.5 (133), Anthony Edwards 98.9 (135), Joel Embiid 99.2 (176). Block leaders: Wembanyama 25, Walker Kessler 17, Myles Turner 15. FromDate/ToDate windowing verified as a true subset: Gobert 180 min / 6 GP full season vs 50 min / 2 GP for 2026-10-27..2026-11-01. Type=Player, Team, Lineup and Opponent all return 200 for 2026-27. The public openapi.json lists 40 endpoints.
- use: (c), and the highest-value (c) in this lens. The repo uses pbpstats for lineups only and is leaving the richer defensive fields on the table: OnDefRtg, DefPoss, OpponentPoints; shot-location block rates (AtRimPctBlocked, Arc3PctBlocked, ShortMidRangePctBlocked, LongMidRangePctBlocked); location-split defensive rebounding (DefAtRimReboundPct, DefTwoPtReboundPct, DefThreePtReboundPct, DefCorner3ReboundPct and three more); DefensiveGoaltends; Charge Fouls; RecoveredBlocks and BlocksRecoveredPct; st

### NBA.com Stats — LeagueDashPtDefend (tracking matchup defense)

- https://www.nba.com/stats/players/defense-dash-overall
- season-only: **yes** | access: free, no account
- machine readable: JSON, and nba_api is already installed in the repo. Note NBA.com's terms restrict commercial redistribution of the raw feed.
- method: Player tracking. For each shot it assigns the closest defender, then compares the opponent's FG% against that defender (D_FG_PCT) to that shooter's own normal FG% (NORMAL_FG_PCT). Pure current-season measurement.
- seen live: Queried live 2026-11-01 through the repo's own nba_api for season 2026-27: 427 rows, columns CLOSE_DEF_PERSON_ID, PLAYER_NAME, PLAYER_POSITION, GP, G, FREQ, D_FGM, D_FGA, D_FG_PCT, NORMAL_FG_PCT, PCT_PLUSMINUS. Best PCT_PLUSMINUS at D_FGA>=25: Damian Lillard -.223 (5 GP, 6/28), Chet Holmgren -.214 (3 GP, 9/36), Derrick Jones Jr. -.196 (5 GP, 13/48), Justin Champagnie -.179 (4 GP), LeBron James -.160 (5 GP, 18/64), Nate Ament -.151 (4 GP).
- use: (c), and it is the only genuinely new KIND of information found anywhere free — every other source in this lens measures what happened while a player was on the floor, not what happened to the men he guarded. It is the single best candidate to replace or supplement the 10% on/off term. Also (a) and (b). HARD CAVEAT for (c): stats.nba.com is reachable from this Mac but blocked from the production host, so this must be built into a committed versioned artifact and read from disk, never fetched on 

### NBA.com Stats — LeagueHustleStatsPlayer (deflections, contested shots, charges)

- https://www.nba.com/stats/players/hustle
- season-only: **yes** | access: free, no account
- machine readable: JSON via nba_api.
- method: Human/tracking-logged hustle events for the single season. Raw counts, no model.
- seen live: Queried live 2026-11-01 via nba_api for 2026-27: 422 rows, max G = 7. Columns include CONTESTED_SHOTS, CONTESTED_SHOTS_2PT, CONTESTED_SHOTS_3PT, DEFLECTIONS, CHARGES_DRAWN, DEF_LOOSE_BALLS_RECOVERED, DEF_BOXOUTS, BOX_OUTS. Deflection leaders: Dyson Daniels 42 (7 G, 202 min), Paul George 33 (6 G), Kris Dunn 30 (6 G), Bam Adebayo 28 (6 G), Matisse Thybulle 26 (6 G), Zaccharie Risacher 25 (6 G).
- use: (c) as a perimeter-defense term, which is the composite's weakest spot: BLK/36 + DREB/36 is a big-man axis, so guards can only score well through DBPM and the on/off blend. Deflections and contested 3s give perimeter defenders a positive signal of their own, and the leaderboard passing the eye test (Dyson Daniels, Dunn, Thybulle) is evidence the stat tracks real reputation. Also (a). Same blocked-host caveat as above.

### NBA.com Stats — LeagueDashPlayerStats, measure type Defense (DEF_RATING, DEF_WS)

- https://www.nba.com/stats/players/defense
- season-only: **yes** | access: free, no account
- machine readable: JSON via nba_api. Notably it ships pre-computed rank columns, so a ranking comes back without computing one.
- method: Single-season aggregation of on-court defensive rating plus box rates and a points-allowed-by-type breakdown. DEF_WS here is the NBA's own defensive win shares, a different quantity from BBRef's DWS.
- seen live: Queried live 2026-11-01 via nba_api for 2026-27: 441 rows, max GP = 7. Columns DEF_RATING, DREB, DREB_PCT, PCT_DREB, STL, PCT_STL, BLK, PCT_BLK, OPP_PTS_OFF_TOV, OPP_PTS_2ND_CHANCE, OPP_PTS_FB, OPP_PTS_PAINT, DEF_WS, DEF_WS_RAW, plus a _RANK column for every one of those. Best DEF_RATING at MIN>=18: Alex Caruso 80.9 (4 GP), Alpha Diallo 86.4 (1 GP), Herbert Jones 89.0 (6 GP), Mo Bamba 89.7 (3 GP), Javonte Green 89.7 (6 GP), Rudy Gobert 93.5 (6 GP).
- use: (b) primarily — DEF_WS and the shipped _RANK columns give a ready-made machine-readable 2026-27 defensive ordering to score Spearman against, which is the thing that does not currently exist. Weak for (c): DEF_RATING is on-court and team-contaminated (Diallo ranks 2nd on one game). OPP_PTS_PAINT and OPP_PTS_FB are the parts worth considering as inputs.

### Cleaning the Glass (Insider stats)

- https://cleaningtheglass.com/stats/players?stat_category=defense_rebounding&season=2026&seasontype=regseason
- season-only: **yes** | access: paid. Verbatim from the site: "Check out the stats free for one week, then just $5/month after that:", "Subscribe for $5/month", "Or get two months free when you subscribe for a full year:", "Subscribe for $50/year". One-week free trial.
- machine readable: A "Download this table" control exists but is href="#" (JS) and subscriber-gated. No public API. The underlying rows are present in the page HTML behind a visual overlay — do not scrape around that, it would breach their terms; the honest route is the $5/month subscription.
- method: Play-by-play and box aggregation with garbage time removed, reported as per-season rates and as on/off splits. Percentiles are computed RELATIVE TO POSITION using CTG's own five buckets (Point, Combo, Wing, Forward, Big). No composite defensive metric — it publishes components, not a single defensive number.
- seen live: Loaded in the browser 2026-11-01. "2026-27" is the first and default season option (season=2026), and the table holds 212 player rows with real current-season minutes — Precious Achiuwa SAC 167 MIN / 28.0 MPG, Darius Acuff SAC 218 MIN, Bam Adebayo MIA 188 MIN, Nickeil Alexander-Walker ATL 210 MIN, Grayson Allen CHA 190 MIN. Defense and Rebounding columns confirmed: BLK%, STL%, FOUL%, fgOR%, fgDR%, ftOR%, ftDR%, each paired with a percentile. Fifteen leaderboard categories exist, including four On/Off Opponent ones. I did NOT read the defensive VALUES or top defenders — those sit behind the paywall.
- use: (a), and (b) for a subscriber willing to export by hand. The unique asset is the four On/Off Opponent categories — Opponent Shooting: Frequency, Opponent Shooting: Accuracy, Opponent Halfcourt & Putbacks, Opponent Transition — i.e. how the opponent's whole shot profile changes with the player on vs off, which is a defensive impact signal nothing else here offers in that form. Not (c): a hand-downloaded $5/month table is not a pipeline input, and automating past the gate is off the table. Worth o

### RealGM — Advanced Stats

- https://basketball.realgm.com/nba/stats/2027/Advanced_Stats/Qualified/points/All/desc/1/Regular_Season
- season-only: **yes** | access: free, no account
- machine readable: HTML table only; I found no CSV or API. It 403s plain automated fetches (both WebFetch and curl) and loaded only in a real browser, so treat it as screen-oriented.
- method: Box-score derived per-season rates and ratings, recomputed as games are played.
- seen live: Loaded in the browser 2026-11-01. Heading "2026-2027 NBA Regular Season Stats - Advanced Stats", season selector set to 2026-2027. It publishes a Defensive Rating leaderboard: Kawhi Leonard 81.8, Jayson Kent 87.2, Cason Wallace 89.4, Pat Spencer 96.2, Chet Holmgren 96.5. Columns: TS%, eFG%, Total S%, ORB%, DRB%, TRB%, AST%, TOV%, STL%, BLK%, USG%, PPR, PPS, ORtg, DRtg, eDiff, FIC, PER. No DBPM and no DWS.
- use: Weak (a). It carries the same STL%/BLK%/DRB% components Barrett already has and no DBPM, so there is little new. Two things are worth noting: its Stat Split selector offers Last 5 Games, Last 10 Games, individual months, Home/Away and Opponent Top 10 Defense, which BBRef does not offer free league-wide; and its DRtg disagrees sharply with other sites' DRtg (Kawhi 81.8 here vs 88.7 on StatRankings), so DRtg is not a shared quantity across sources and should never be pooled.

### StatRankings — NBA Defensive Rating

- https://statrankings.com/nba/players/advanced-efficiency/defensive-rating
- season-only: **yes** | access: free to view the table; a free account is offered for full player pages, and some features sit behind "Unlock statrankings+" (no price shown)
- machine readable: HTML table only. No NBA API or CSV found — its downloadable archive covers NFL, not NBA.
- method: Points allowed per 100 possessions, recomputed from play-by-play after each game.
- seen live: Loaded 2026-11-01. Labels the season "2026", figures "as of October 31, 2026", last updated 11/01/26, "Minutes per team game (35 so far)". Top 5 by Defensive Rating: Kawhi Leonard (TOR) 88.7, Hansen Yang (POR) 91.3, Cason Wallace (OKC) 93.1, Jalen Williams (OKC) 94.3, Chet Holmgren (OKC) 94.7. Covers top 25 of 331 players, 5 minutes-per-team-game minimum. It also displayed "Couldn't update, showing the last loaded results", so it may have been serving a cached copy.
- use: (a) at best, and only loosely. It ranks one metric, that metric is team-driven (three of its top five are Thunder teammates), and it has no export. Its one use is as a quick second opinion on DRtg — and the fact that its Kawhi figure (88.7) differs from RealGM's (81.8) for the same season is itself the useful signal: public DRtg is not standardized.

### Dunks & Threes — EPM (offense/defense split)

- https://dunksandthrees.com/epm
- season-only: **no** | access: free to view; the API is paid — "The same data is available through the API with an API subscription" (price not shown on the methodology page)
- machine readable: Paid API. No free CSV found.
- method: Two-stage: a box-plus-tracking SPM becomes the Bayesian prior for an exponentially-decayed multi-season RAPM run over every possession since 2001-02.
- seen live: Loaded 2026-11-01. The page is live for the current season — game results listed from October 28 through November 1, 2026 — and the methodology page states values "update nightly" with "two variables per player, offense and defense". I could NOT reach the leaderboard itself: both the EPM page and the methodology page are over 110,000 characters and scoreboard-dominated, so I never saw the D-EPM table or its top defenders.
- use: Ruled out for (c) by Barrett's constraint, in the source's own words. More important and easy to miss: it is also the WRONG validation target for (b) in November specifically. By its own description, "Early-season values lean on the expectation" — at 7 games, D-EPM is mostly last season's prior, so a Spearman run against it now would largely measure how well Barrett's composite reproduces 2025-26, not how well it reads 2026-27. It becomes a reasonable (b) target around the turn of the year and a

### deep_rapm (DRAPM leaderboard + pip package)

- https://github.com/aaronjdanielson/deep_rapm
- season-only: **partly** | access: free (MIT license, `pip install "deep-rapm[viz]"`)
- machine readable: Yes, but you generate the data: `collect-possessions`, `collect-players`, then `solve-rapm`/`rolling-rapm` write per-season parquet (one row per possession). The README leaderboard itself is a markdown table. data/ and outputs/ are gitignored, so no data file ships with the repo.
- method: Weighted ridge RAPM on individual possessions: each row has the ten players on court split into an offense block and a defense block, outcome is points on the possession; coefficients are reported per 100 possessions with the defensive sign flipped so higher is better. Weights are recency x garbage-time x preseason-context, solved exactly via incremental Gram updates.
- seen live: Loaded the raw README and queried the GitHub API directly. README badge reads 'seasons 2013-14 -> 2026-27'. Leaderboard block states the fit uses 'every regular-season, playoff and preseason possession through October 25, 2026'. Repo pushed_at 2026-11-02T02:07:54Z, created 2026-03-20, 18 commits, latest 'sweeping upgrades' 2026-11-02. Published 'Highest defense' table (DRAPM, pts/100): Victor Wembanyama +5.3, Alex Caruso +5.0, Rudy Gobert +4.3, Chet Holmgren +3.4, Daniss Jenkins +3.1, then Jalen Suggs +3.0, Shai Gilgeous-Alexander +3.0, Ajay Mitchell +2.7, Landry Shamet +2.6, Marcus Smart +2.5.
- use: (a), (b) AND (c) - the only source I verified that can be run current-season-only. For (c) it is also the only candidate that gives Barrett a true lineup-level defensive impact term with no multi-year prior, which is a strict upgrade on the 10% on/off blend he already has. Caveats that matter: 0 stars, 0 forks, single author, repo only 7 months old, so it has had essentially zero external review; the README itself warns the posterior intervals 'have not been checked against outcomes' and 'among 

### Kalshi - Pro Basketball Defensive Player of the Year, 2026-27 (series KXNBADPOY, event KXNBADPOY-27)

- https://kalshi.com/markets/kxnbadpoy/nba-defensive-player-of-the-year
- season-only: **yes** | access: free to read (no auth, no key, no rate-limit trouble); trading needs an account
- machine readable: Yes - open JSON, no auth. GET /trade-api/v2/events?series_ticker=KXNBADPOY&with_nested_markets=true. Gotcha: the integer `yes_bid`/`last_price` fields come back null; the real values are in the `*_dollars` string fields (`yes_bid_dollars`, `last_price_dollars`).
- method: Crowd/market forecast of the DPOY vote.
- seen live: Queried the public API directly (api.elections.kalshi.com/trade-api/v2). Event KXNBADPOY-27 'Pro Basketball Defensive Player of the Year / 2026-27 season' is active with 48 player contracts. Wembanyama: yes_bid $0.73, yes_ask $0.77, last $0.73, previous $0.76, open interest 13,716, total volume 17,571, volume_24h 20.86. Full ladder by last price: Wembanyama 0.73, Dyson Daniels 0.25, Donovan Clingan 0.16, Amen Thompson 0.12, LeBron James 0.08, Walker Kessler 0.07, Zach Edey 0.07, Anthony Davis 0.06, Bam Adebayo 0.03, Gobert 0.03, Scottie Barnes 0.03, Evan Mobley 0.02, Jaren Jackson Jr. 0.02, Giannis 0.02, Chet Holmgren 0.01, Ausar Thompson 0.01. IMPORTANT: only 2 of 48 contracts had any 24h v
- use: (a) only, and realistically only for the #1 slot. It is a real answer to 'who ranks defenders this season' for exactly one name: the market says Wembanyama ~73-77%. Below that it is too thin to read as a ranking - a 0.03 bid / 0.13 ask spread is not an opinion. Not usable for (b): 48 names, one liquid, no Spearman worth computing. Not (c).

### Polymarket - NBA: 2026-27 Defensive Player of the Year

- https://polymarket.com/event/nba-2026-27-defensive-player-of-the-year
- season-only: **yes** | access: free to read (no auth); trading needs an account and crypto funding
- machine readable: Yes - gamma-api.polymarket.com/events?slug=nba-2026-27-defensive-player-of-the-year, JSON, no auth. Prices are JSON-encoded strings inside the `outcomePrices` field. Note the DPOY event does not appear in the top ~100 NBA events by volume, so you have to page or query by slug.
- method: Crowd/market forecast of the DPOY vote.
- seen live: Found it by scanning 189 open NBA-tagged events through gamma-api.polymarket.com, then loaded the event by slug. Event id 1183256, updatedAt 2026-11-02T04:20:17Z - so it is genuinely live today. Wembanyama: price 0.732, bestBid 0.724, bestAsk 0.740, volume $949. Total event volume only $3,278. Every other outcome is an unfilled placeholder literally named 'Player Z', 'Player Y', 'Player X' ... at price 0.50 with volume 0, bestBid 0, bestAsk 1. So beyond Wembanyama there is no board at all. Note for contrast: Covers.com (loaded, stamped 'Odds as of 10-20', last updated 2026-10-19 11:50pm ET) reproduces a 10-name Polymarket board - Wembanyama 70%, Holmgren 11%, Ausar Thompson 5%, Dillon Brooks
- use: (a) only, one name. It independently corroborates Kalshi almost exactly (0.732 vs 0.73), which is worth something as a cross-check on the #1 slot and nothing else. Not (b), not (c).

### ESPN 'defensive impact' player stat leaders, 2026-27

- https://www.espn.com/nba/stats/player/_/stat/defensive-impact
- season-only: **yes** | access: free, no auth
- machine readable: Yes - open JSON at site.web.api.espn.com/apis/common/v3/sports/basketball/nba/statistics/byathlete, no key, undocumented. Note the season param is 2027 for the 2026-27 season.
- method: Raw box score. Steals and blocks only.
- seen live: Verified twice. The HTML page is titled 'NBA Player defensive-impact Stats 2026-27', says 'Statistics are updated nightly' with no as-of date, and shows GP 5-7 (Luka Doncic 5 GP at rank 1, Maxey 7, Curry 5, Giannis 6, Wembanyama 6). I also hit the undocumented JSON directly: GET site.web.api.espn.com/apis/common/v3/sports/basketball/nba/statistics/byathlete?season=2027&seasontype=2 returned HTTP 200, 162,884 bytes of live 2026-27 rows - Wembanyama 6 GP, 4.2 BLK/g, 1.3 STL/g, 25 blocks; Gobert 6 GP, 0.5 BLK, 0.5 STL. BUT THIS SOURCE IS A MISNOMER AND SHOULD BE DISCARDED: ESPN's own JSON defines the 'defensive' category as exactly four fields - Steals Per Game, Blocks Per Game, Steals, Blocks.
- use: None as a ranking of defenders - do not use it for (a), (b) or (c), and do not let the name fool anyone. One genuinely useful byproduct though: it is a free, no-auth, nightly-updating box-score JSON feed for 2026-27 that does NOT touch stats.nba.com, so it is reachable from the production host. That makes it a candidate fallback feed for the BLK/36 and DREB/36 leg of the composite if stats.nba.com is unreliable - which it was for me today.

### StatMuse 'defensive EPM leaders this season'

- https://www.statmuse.com/nba/ask/nba-defensive-epm-leaders-by-player-this-season
- season-only: **unknown** | access: free
- machine readable: No - natural-language answer cards, no export.
- method: Natural-language query layer over licensed stats; intended to surface EPM but does not here.
- seen live: Loaded it. It is on 2026-27 (latest news items dated 2 November 2026) but the answer is broken: it returns 25 players in ALPHABETICAL order with NO defensive EPM column at all - Steven Adams (HOU) 1 GP, Giannis (MIA) 6 GP, Harrison Barnes (SAS) 4 GP, Kentavious Caldwell-Pope (PHI) 7 GP, Mike Conley (BOS) 1 GP. No minutes or games filter either, so a 1-game player sits at rank 1. It does not serve a defensive EPM leaderboard.
- use: None. Including it because it looks like a cheap win in search results and is not one - the ordering is alphabetical, not by any defensive value, and it has no minutes floor. Anyone eyeballing this would be reading noise.

## Not reachable, or not live for 2026-27

- **Dunks & Threes — Actual EPM (season-to-date)** (no) -- (b) LATER IN THE SEASON only, and even then with the caveat that it is prior-anchored by design rather than a clean current-season measurement. Useless today because it is empty. Never (c). Worth a recheck in December/January — this is the thing most likely to
- **BBall Index — Daily LEBRON Dashboard (D-LEBRON)** (no) -- (b), via a route you may not have considered. It cannot score 2026-27. But because it publishes DAILY D-LEBRON snapshots back to October 2025 with CSV export, you can pull the early-November-2025 snapshot and recompute your composite on the matching 6-7 game w
- **BBall Index — NBA LEBRON Dashboard (full-season)** (no) -- Not (a)/(b)/(c) for the live season. Its value is that 2025-26 D-LEBRON is now final and CSV-exportable, so you can extend your existing finished-season validation series (currently 2024-25 and 2023-24) with a third year.
- **CraftedNBA — Best Defenders / CraftedDPM** (no) -- None, as things stand — and I would treat it as an active hazard. It is the most convenient-looking defensive comparison on the web and it will silently hand you last season under a current-season label. Do not wire it into anything without an independent seas
- **nbarapm.com — Current Impact** (no) -- None for the live season. If it starts carrying 2026-27 its defensive time-decay RAPM would still fail the current-season-only rule because of the 550-day half-life.
- **xRAPM.com** (no) -- None right now — it is stale by a full offseason. Worth rechecking later in the season, since a populated 1-year defensive RAPM would be one of the few genuinely current-season-only impact ratings.
- **ESPN RPM / DRPM** (no) -- None. Rule it out and do not let the "defensive-impact" page fool you — Luka Doncic at number one is the giveaway that it is sorting counting stats.
- **DRIP / Defensive DRIP (DDRIP)** (could-not-verify) -- Possibly (a) if you can reach a live 2026-27 table — worth two minutes in a browser since I could not confirm it. Not (c): it is self-described as a projection, so it fails the current-season-only rule.
- **HoopsJunkie — single-season RAPM** (could-not-verify) -- Potentially (b) and the best candidate I found for it — a true single-season defensive RAPM would be current-season-only by construction and independent of DBPM, which is what a validation target needs. Please check it yourself in your browser; if its 2026-27 
- **NBA.com DefenseHub (curated league defensive leaders)** (could-not-verify) -- none, as of today. Reporting it explicitly because it is the endpoint most likely to become the (a) panel you want if it comes back, and because a silent JSONDecodeError here is exactly the kind of dead feed your memory notes have bitten you with twice before.
- **nbarapm.com (databallr) — Time-Decay RAPM** (could-not-verify) -- None as it stands, for two reasons. First, liveness is unverified. Second, the 550-day half-life is the killer: at 7 games into 2026-27, essentially all of the weight is still on 2025-26 and earlier possessions, so despite the honest "zero prior" label this is
- **HoopsJunkie — RAPM leaderboard (ORAPM / DRAPM / RAPM)** (could-not-verify) -- Potentially the best (b) in this whole list, and the one thing most worth Barrett spending five minutes on himself. The path /stats/2026-27/players/rapm/rapm exists and the 2025-26 equivalent carries a DRAPM column; if that page is live and genuinely single-se
- **BALLDONTLIE NBA API (season_averages defense/tracking/hustle + v2 advanced matchup fields)** (could-not-verify) -- Potentially (b) and (c), and this is the single highest-value thing for Barrett to check himself. It is the only candidate that solves the production-host constraint: it is NOT stats.nba.com, so a Render dyno can reach it on a request path, which no NBA.com-de
- **Cleaning the Glass - player stats and on/off** (could-not-verify) -- (a) cheaply and well - $5 for a month is the lowest-friction way to eyeball the live board against an outside, garbage-time-filtered number. Weak for (b): screen-only, so Spearman means hand-transcribing. For (c) be careful: CTG on/off is the SAME construct as
- **NBA.com / stats.nba.com official tracking defense dashboards (leaguedashptdefend, defense-dash-2pt / 3pt / lt6)** (could-not-verify) -- (a) and (b) if it is reachable - it is current-season-only, free, machine-readable and genuinely independent of DBPM, so it is the most natural validation target to Spearman the live board against. Explicitly NOT (c) on the production path: the production host
- **Yahoo Sports - 'NBA best defender rankings 2026-27: The 15 best defenders in the league'** (no) -- (a) weakly, as a preseason prior to sanity-check the top of the board against - and it is useful precisely because it is honest about being a projection. Not (b) (unordered, prose, prior-season). Not (c). Worth noting it is not independent of Barrett's composi
- **Bleacher Report - 'Ranking the Top 30 NBA Defenders Ahead of 2026-27' ('Defensive Swing Rating')** (no) -- None for Barrett's purpose. It is preseason, prior-season, and - the bigger problem - it is on/off, the same family as the 10% term already in his composite, so it is not an independent opinion. Its one use is as a labelled-stale preseason baseline, and the de
- **ShotQuality / databallr 'Defensive Shot Quality' (D-SQ RAPM)** (could-not-verify) -- Possibly (a), unverified, and not worth paying $39.99/mo before someone opens it in a real browser and confirms there are 2026-27 rows and a stated method. Barrett should check this one by hand - it is the only paid product I found that claims a defender-level
- **Thinking Basketball / Backpicks 'Daily Board'** (could-not-verify) -- (a) at most, and only if the board actually has a defensive column - unverified. At $2-$7 for one month it is the cheapest unknown on this list, so it is worth Barrett spending $4 to look rather than me guessing. I would not plan on it for (b) (screen-only) or
- **Synergy Sports (Sportradar) - NBA film and scouting platform** (no) -- None. Ruling it out explicitly so it stops coming up: there is no path for one person to buy NBA Synergy, and even the college Deep Dives product would not help. Same verdict for Sportradar's own NBA API (quote-only enterprise; third-party estimates range from
- **SportsBettingDime 2027 NBA DPOY odds tracker** (no) -- None - superseded. If Barrett wants the betting-market view, Kalshi and Polymarket give the same information through open JSON APIs with no scraping and no stale 'auto-updating' claim. I checked this one because an auto-updating odds tracker would have been th
- **openrapm (bbstats) and Chase Analytics nba-model - public RAPM repos that fail the constraint** (no) -- None. Including them so the GitHub search is closed out rather than left open: of the public RAPM repos, deep_rapm is the only one that both reaches 2026-27 and exposes a single-season no-prior fit. The others either stop at 2025-26, bucket three seasons into 
- **Kaggle - 'NBA Dataset: Box Scores and Stats (1947 - Today)' (eoinamoore)** (could-not-verify) -- None directly - it is a feed, not a ranking of defenders, and it contains no defensive impact metric to sanity-check or Spearman against. Barrett already has box-score inputs. Listing it to close out the Kaggle search: I found no Kaggle dataset that publishes 
