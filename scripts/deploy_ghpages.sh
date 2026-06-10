#!/usr/bin/env bash
# Rebuild the static site and publish it to the gh-pages branch (GitHub Pages).
# Run this whenever you change the data or the design and want it live.
#
#   bash scripts/deploy_ghpages.sh            # full rebuild (~3 min) + deploy
#   SKIP_PLAYERS=1 bash scripts/deploy_ghpages.sh   # skip per-player pages (fast)
#   PRODUCTION=1 bash scripts/deploy_ghpages.sh     # cutover: publish the hoopsvalue.com CNAME
#
# First time only: enable Pages once in the repo —
#   Settings → Pages → Source: "Deploy from a branch" → Branch: gh-pages / (root)
# Then it's live at https://barrett879.github.io/nba-value-app/
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/build_static.py

# CNAME is declarative and STICKY: present iff this is a PRODUCTION deploy or
# the live gh-pages branch already carries one (the domain has been cut over).
# Without this, a routine deploy after cutover would publish a tree with no
# CNAME and silently unbind hoopsvalue.com: the subtree split only packages
# committed files, and a bare `git diff` cannot even see an untracked CNAME.
git fetch -q origin gh-pages 2>/dev/null || true
if [ "${PRODUCTION:-}" = "1" ] || git cat-file -e origin/gh-pages:CNAME 2>/dev/null; then
  printf 'hoopsvalue.com\n' > site/CNAME
  echo "CNAME: publishing hoopsvalue.com (production)"
else
  rm -f site/CNAME
fi

# Stage first so new and deleted files (the CNAME!) count as changes; gate on
# the index, then commit only site/ so unrelated staged work is left alone.
git add site/
if ! git diff --cached --quiet -- site/; then
  git commit -q -m "Rebuild static site" -- site/
  git push origin main
fi

git subtree split --prefix site -b _ghpages >/dev/null
git push origin _ghpages:gh-pages --force
git branch -D _ghpages >/dev/null

echo "Deployed → https://barrett879.github.io/nba-value-app/ (allow ~1 min on first enable)"
