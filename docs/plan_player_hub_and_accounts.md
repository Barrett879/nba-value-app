# Plan: Player-Hub Homepage + User Accounts & Comments

Status: PLANNED (build later). Written 2026-07-02. Two features, designed together
because they reinforce each other: the homepage becomes player-centric, and comments
attach to players.

---

## Part A — Homepage revamp: the Player Hub

### The idea (Barrett's sketch)
Replace the current homepage (hero + search + promo strips) with a **list of players**.
Selecting a player shows a **hub panel**: the stats and contract prediction that today
live across the various tabs, with buttons to jump to each tab for the full view.

### Design

1. **Top of page**: keep the HoopsValue hero + the existing search box (it already spans
   1973-today). The search now doubles as the list filter.
2. **The list**: a ranked player table (current season, Barrett Score order) — think a
   lighter Rankings grid: rank, name, team, pos, score, salary, projected contract.
   Paged or top-N with a "show more" (200 rows renders fine in the themed html_table).
3. **The hub panel** (renders above/beside the list when a player is selected):
   - Identity row: name, team, age, position (reviewed Excel source), draft pedigree.
   - Barrett Score + rank, with the score-vs-salary verdict (under/overpaid delta).
   - **Predicted contract** — precomputed, NOT a live model run. VERIFIED GAP: the
     existing board caches only cover ~160 of ~500 pool players, so
     `scripts/build_player_hub.py` must batch-compute pcv for the FULL pool offline
     (feasible now that the feature path is parquet-only after the 2026-07-01 fetch
     fixes; re-run alongside the accuracy tracker). Fallback for any player still
     missing: rank-based Proj. Value + "run full prediction ->" link.
   - FA context when applicable: Status, Outcome (opt in/out, signed), Signed $ / vs
     Model from the accuracy cache.
   - Career sparkline (score by season) from the all-seasons parquet.
   - **Jump buttons**: "Full Contract Prediction ->", "Player Profile ->", "Career /
     Legacy ->", each deep-linking to the tab pre-loaded on this player.
   - (Part B lands here): the player's comment thread at the bottom of the hub.

### Deep links (the glue)
Streamlit query params (`st.query_params`). Each target page learns to read
`?player=<normalized-name>` and preselect that player. Required edits:
- `pages/Contract_Predictor.py`: if `player` param present and matches
  `get_all_player_names()`, seed the selectbox state (respect the
  stable-options rule — set state by key BEFORE the widget instantiates,
  never reorder options; see the selectbox element-ID gotcha).
- `pages/Search.py`: same pattern.
- The hub's jump buttons emit `/Contract_Predictor?player=jake%20laravia` style links
  (plain `<a>` in the existing themed markup — no components.html iframes, per the
  Render cold-start iframe gotcha).

### Performance rules (hard requirements)
- The homepage must not get slower than today: **no live model runs, no network** in
  the hub. Everything reads from caches already committed/warmed: rankings frame,
  board-cache pcv, accuracy tracker, option decisions, all-seasons parquet.
- New build step `scripts/build_player_hub.py` -> `cache/player_hub_v1.parquet`:
  one row per current player (rank, score, salary, pcv, status, outcome, teams,
  sparkline series as a list column). One parquet read at page start; selection is a
  dict lookup. serve.py warms it at boot.
- Comments load LAST in the script (bottom of hub) with a bounded-timeout fail-soft
  fetch, so Supabase can never delay the player content.

### What happens to the current homepage content
The promo strips (Contract Predictor / charts) are replaced by the hub. The FA summary
cards can move into the hub context or be dropped (open question below). VERIFIED:
`render_strip` also lives in the root-level drafts `app_home_hybrid.py` and
`app_home_strips.py` — delete those alternates in the same change so no stale copies
linger (removal touches 3 files, not just app.py).

### Open questions for Barrett (answer before build)
1. List defaults: current-season Barrett Score order? How many rows before "show more"?
   Any filters on the list itself (team / position / FA-only), or is search enough?
2. Hub contents priority: is the list above roughly right? Anything missing you want
   surfaced (e.g. suitor prediction from the parked FA-sim engine)?
3. Do the homepage FA summary cards (Total FA / UFA / RFA / PO / TO) survive the revamp?
4. Search scope on the homepage: keep all-eras (1973+) search, or current-season only
   with Legacy search staying on its own tab?
5. Mobile: the list+panel layout — panel above list (stacked) is the safe default; OK?

---

## Part B — User accounts + comments (Supabase)

### Decisions already made (Barrett, 2026-07-02)
- Sign-in: **Both** Google one-click AND email magic links.
  Phasing recommendation: **Phase 1 = Google** (Streamlit-native `st.login`/`st.user`
  OIDC, cookie-backed session, least code), **Phase 2 = email magic links** (Supabase
  Auth/GoTrue; requires token round-trip + session handling we own). Shipping Google
  first gets comments live weeks earlier; magic links added without schema changes.
- Placement: **player hub threads** (one thread per player, `page_key` = normalized
  player name). A general FA-page thread can reuse the same table later.

### Supabase setup (Barrett's ~15 minutes, one time)
1. supabase.com -> New project (free tier). Region: **US West (Oregon)** to sit next to
   Render. Save the database password somewhere safe.
2. Project Settings -> API: copy `Project URL`, `anon` key, `service_role` key.
3. Google OAuth client (for st.login): console.cloud.google.com -> Credentials ->
   Create OAuth client (Web application). Authorized redirect URI:
   `https://hoopsvalue.com/oauth2callback`. Copy client id + secret.
4. Render dashboard -> barrett-score -> Environment: add
   `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`,
   `AUTH_COOKIE_SECRET` (any long random string), `ADMIN_EMAILS`
   (comma-separated; Barrett's gmail).

### Schema (run once in the Supabase SQL editor)
```sql
create table comments (
  id          uuid primary key default gen_random_uuid(),
  page_key    text not null,                 -- normalized player name or 'fa_class'
  user_email  text not null,
  display     text not null,                 -- shown name (from Google profile)
  body        text not null check (char_length(body) between 1 and 2000),
  created_at  timestamptz not null default now(),
  deleted     boolean not null default false
);
create index comments_page_idx on comments (page_key, created_at desc);

-- Lock the table: the anon key gets NOTHING. All reads/writes go through the app
-- server with the service key (which bypasses RLS). If/when the static site needs
-- direct browser reads, add a select-only policy for anon on deleted=false.
alter table comments enable row level security;
```

### App integration (`hv_comments.py`, new module)
- Plain HTTPS to Supabase PostgREST (`requests`, no new SDK dependency):
  GET `/rest/v1/comments?page_key=eq.<key>&deleted=is.false&order=created_at.desc
  &limit=50`, POST to insert, PATCH to soft-delete. Service key in headers,
  server-side only.
- **House rules (non-negotiable, learned 2026-07-01):** every call
  `timeout=5`, ONE retry, fail-soft — on failure render "Comments are taking a nap"
  and never block the page. No infinite loops. No empty-result caching.
- Auth gate: composer shows only for `st.user.is_logged_in`; otherwise a
  "Sign in with Google to comment" `st.login` button.
- Abuse guardrails: 1 comment / 30s / user (checked via the user's latest
  `created_at`); body HTML-escaped on render (themed markup, no raw HTML);
  max 2000 chars; owner + `ADMIN_EMAILS` see a delete button (soft-delete).
- Streamlit config: serve.py writes `.streamlit/secrets.toml` `[auth]` block from env
  vars at boot (redirect_uri, cookie_secret, Google client) — Render env is the source
  of truth, nothing secret in git. VERIFIED: no such writer exists yet in serve.py —
  this is net-new code in B1 (streamlit==1.51.0 already supports st.login natively;
  no extra deps for Phase 1).
- `page_key` design note (verified): `normalize()` keeps apostrophes/periods/hyphens
  ("d'angelo russell"). Fine in URLs (they encode) and as Postgres keys — but ALWAYS
  key comments through the app's own `normalize()`, never an ad-hoc slug, or threads
  will silently split (same class of mismatch that produced false "unresolved options"
  in the 2026-07-01 audit).

### Phase 2 — email magic links (after Google ships)
- Supabase Auth `signInWithOtp` via GoTrue REST: user enters email, gets a link with a
  token; landing back on hoopsvalue.com we verify the token server-side and mint our
  own session (signed cookie via the same AUTH_COOKIE_SECRET). This is the clunky part
  Streamlit doesn't help with — budget a real day for edge cases (expired tokens,
  refresh, sign-out).

### Explicitly rejected alternatives
- Disqus embed (ads/tracking), giscus (requires GitHub accounts — wrong audience),
  Postgres-on-Render (+$7/mo, no auth story, doesn't survive the static-site future).
- Putting the PUBLIC site's curated data (signings/options CSVs) into Supabase:
  rejected — read-heavy single-writer data is better in git (versioned, zero-latency,
  no runtime dependency). Supabase is for user-generated data only.

---

## Build order & estimates
| Step | What | Size |
|---|---|---|
| A1 | `build_player_hub.py` (incl. full-pool pcv batch) + hub cache + serve.py warm | 1 day |
| A2 | Homepage revamp (list + hub panel + jump buttons) | 1 day |
| A3 | Deep-link params on Contract Predictor / Search | half day |
| B1 | Supabase project + schema + env (Barrett 15 min + me wiring) | half day |
| B2 | Google sign-in (st.login) + comments UI on the hub | 1 day |
| B3 | Email magic links (Phase 2) | 1 day |

Sequence: A1→A2→A3 ships the new homepage alone; B1→B2 attaches accounts+comments to
it; B3 whenever. Nothing here touches the model or the accuracy tracker.
