# Deploying the static HoopsValue site

The site is **pre-built static files** under `site/` — no server, no build step
required at deploy time. Any static host serves them from a CDN edge in
~20–80ms. Paths are root-absolute (`/assets/...`, `/player/...`), so deploy at a
**domain root** (works for `hoopsvalue.com`, a `*.pages.dev`, or an
`*.onrender.com` subdomain — not a GitHub Pages *project* subpath).

## Rebuild the site (run after data changes)
```bash
python scripts/build_static.py        # full: home + rankings + all player pages (~3 min)
SKIP_PLAYERS=1 python scripts/build_static.py   # fast: skip per-player pages
```
Output goes to `site/`. Commit it; the host serves it as-is.

## Option A — Cloudflare Pages (recommended: free, root, fast)
1. Cloudflare dashboard → **Pages** → **Create** → **Connect to Git** → pick
   `Barrett879/nba-value-app`.
2. Build settings: **Framework preset = None**, **Build command = (blank)**,
   **Build output directory = `site`**.
3. **Save and Deploy** → live at `https://<project>.pages.dev` in ~1 min.
4. Add custom domain → `hoopsvalue.com` → follow the CNAME instructions.
   (This is the cutover from the Streamlit Render service.)

## Option B — Render Static Site
1. Render dashboard → **New** → **Static Site** → connect the same repo.
2. **Build command = (blank)**, **Publish directory = `site`**.
3. Deploy → live at `https://<name>.onrender.com`. Add `hoopsvalue.com` in
   Settings → Custom Domains.

## Option C — Netlify
Connect the repo; **Publish directory = `site`**, no build command.

## Hybrid: the Streamlit app keeps the interactive tools
This static site is the read-only front door — home, rankings, all 365 player
pages. The **interactive** tools (Contract Predictor / Front Office, Team
Analysis, Free Agents, Legacy) can't be static, so they stay on the Streamlit
app. The homepage nav links to them via `APP_BASE` in `build_static.py`
(default `https://app.hoopsvalue.com`). Cutover:

1. **Static → apex.** Point `hoopsvalue.com` at the CDN host (Cloudflare Pages /
   GitHub Pages / Render Static). For GitHub Pages, emit the custom-domain file:
   `PRODUCTION=1 bash scripts/deploy_ghpages.sh`.
2. **Streamlit → subdomain.** Add `app.hoopsvalue.com` as a custom domain on the
   Render service, and point that subdomain's DNS at Render.
3. Done — apex loads instantly; the nav hands off to the app for the live tools.

## Notes
- Live preview (no domain change): https://barrett879.github.io/nba-value-app/
- Local preview: `python -m http.server 8502 --directory site` → `localhost:8502`.
