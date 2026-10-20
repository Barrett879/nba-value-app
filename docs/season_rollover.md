# Season rollover and in-season operation

Written 2026-10-19, the eve of the 2026-27 opener, when the season being PLAYED
and the season being SCORED diverged for the first time.

## The one idea worth keeping

`utils.SEASONS[0]` is the season the site SCORES. It is not the season being
played. Everything else is derived from it:

| derived from `SEASONS[0]` | today | after a flip to 2026-27 |
|---|---|---|
| the player board and Barrett Score | 2025-26 | 2026-27 |
| `_NEXT_SEASON` (the "actual salary" column) | 2026-27 | 2027-28 |
| `_FA_YEAR` (the free-agent class) | 2026 | 2027 |
| which caches are immutable | everything but 2025-26 | everything but 2026-27 |
| which cap prices a max contract | `SALARY_CAP_M["2026-27"]` | `SALARY_CAP_M["2027-28"]` |

Nothing in that table raises when it is wrong. Each one silently publishes a
wrong number, which is why the rollover is a checklist and not an edit.

## Do not flip on opening night

The board should keep scoring the last COMPLETE season until the new one has
enough games to mean something. Per-game rates and the availability multiplier
are both unstable across a handful of games, so flipping early publishes noise
with a straight face. `scripts/preflight_season_rollover.py` holds at
`MIN_GAMES = 20`, about a quarter of the season, which lands near the start of
December.

The correct state on opening night is: board on 2025-26, salaries on 2026-27,
preflight reporting HOLD.

## Procedure

Run the preflight first. It is read-only and exits non-zero only when something
would ship a wrong number.

```bash
/opt/anaconda3/bin/python3 -u scripts/preflight_season_rollover.py
```

`PASS` is ready, `FAIL` must be fixed, `WAIT` means it is simply too early.

When it reports READY:

1. Prepend the new season to `utils.SEASONS`. Everything in the table above
   follows automatically; that is the point of deriving them.
2. Confirm `SALARY_CAP_M` has the NEW `_NEXT_SEASON` key. The preflight checks
   this, because `_amax_floor` and `_amax_cap` use
   `SALARY_CAP_M.get(_NEXT_SEASON, 165.0)` and a missing key does not raise, it
   just prices the max tier at a stale cap.
3. Create the new free-agent class files, `data/real_signings_<year>.csv` and
   `data/option_decisions_<year>.csv`. The loaders in `app.py` and
   `pages/Free_Agent_Class.py` name the 2026 files LITERALLY, so they do not
   follow `_FA_YEAR` and must be pointed at the new pair. The 2026 files then
   become a closed historical record, as does the accuracy scorecard built from
   them.
4. Decide what the forward-looking contract set is. `dead_money_2026_27.csv`,
   `hard_caps_2026_27.csv`, `trade_exceptions_2026_27.csv`,
   `contract_years_2026_27.csv`, `salary_supplement_2026_27.csv` and
   `cba_config_2026_27.json` do not move, but their MEANING changes from "next
   season's books" to "this season's books". The Trade Machine and Rosters want
   the current season, which is what they will now get.
5. Warm the new season before the first user does. `serve.py`'s boot warm and
   `_bootstrap_warm()` both reference `SEASONS[0]`, so they pick it up, but the
   first cold start after the flip has no disk cache to serve and the
   serve-then-refresh path has nothing to fall back on. Build the caches locally
   and commit them, or accept one slow cold start.
6. Re-run the preflight after deploying and confirm it still reports READY with
   the new `SEASONS[0]`.

## What changes about the verification rounds

Rounds so far have been offseason-shaped: signings, options, waivers. In season
the inputs change.

- **Stats move nightly.** `_dc_fresh` now keys on `season_is_live(season)`
  rather than on the calendar month, so a season in progress refreshes hourly
  and a finished season is immutable. Before that change, FINAL 2025-26 counted
  as mutable and queued a background refetch every hour forever.
- **Rosters keep moving**: 10-day contracts, two-way conversions, hardship
  signings, and from the trade deadline onward a wave of buyouts and waivers.
  Dead money grows all season rather than settling.
- **The accuracy tracker is closed.** Every 2026 signing has happened, so the
  scorecard is now a historical record. It still must be checked each round,
  because it is the top-priority salary source and the Signed flag, not a
  display statistic.

## Dates that matter in 2026-27

Confidence is marked, because some of these come from this project's own
verified data and some from reporting.

| date | what | source |
|---|---|---|
| 2026-10-20 | opening night | reported |
| 2026-10-31 | Walter Clayton Jr.'s 2027-28 club option was due | our own row, verified |
| 2026-12-15 | most July signings become trade-eligible; the Trade Machine already applies this | engine rule |
| 2027-01-10 | league-wide date non-guaranteed deals become guaranteed; Jamal Cain's row turns on it | our own row, verified |
| 2027-01-27 | Highsmith's and Bouyea's contract-specific guarantee date | reported 2026-10 |
| 2027-02 | trade deadline. CONFIRM THE EXACT DATE before relying on it; not verified here | unverified |
| 2027-03-01 | waived after this and a player is playoff-ineligible | reported, and it is why Cam Thomas could not play in 2026 |
| 2027-06-30 | cap year ends, so a 2026-27 dead-money charge expires. This is exactly how Cameron Thomas's $845K ended up wrongly sitting in a 2026-27 file | CBA |
| 2027-07-01 | free agency opens, and the 2027 class becomes live | CBA |

## Known gaps, carried forward

- `fetch_monthly_scores` still blocks: it has no disk cache to serve, so it
  cannot use the serve-then-refresh pattern the other fetchers use. It mattered
  less in the offseason because the numbers were static. It matters more now.
- A full 30-team dead-money reconciliation against Spotrac. The 2026-10-19 round
  did the six teams it touched and found two bad rows in them.
- `python3` on this machine resolves to Xcode's interpreter, which has no numpy.
  Use `/opt/anaconda3/bin/python3` for anything importing `utils`.
