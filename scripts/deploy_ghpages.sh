#!/usr/bin/env bash
# Rebuild the static site and publish it to the gh-pages branch (GitHub Pages).
# Run this whenever you change the data or the design and want it live.
#
#   bash scripts/deploy_ghpages.sh            # full rebuild (~3 min) + deploy
#   SKIP_PLAYERS=1 bash scripts/deploy_ghpages.sh   # skip per-player pages (fast)
#
# First time only: enable Pages once in the repo —
#   Settings → Pages → Source: "Deploy from a branch" → Branch: gh-pages / (root)
# Then it's live at https://barrett879.github.io/nba-value-app/
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/build_static.py

if ! git diff --quiet -- site/; then
  git add site/
  git commit -q -m "Rebuild static site"
  git push origin main
fi

git subtree split --prefix site -b _ghpages >/dev/null
git push origin _ghpages:gh-pages --force
git branch -D _ghpages >/dev/null

echo "Deployed → https://barrett879.github.io/nba-value-app/ (allow ~1 min on first enable)"
