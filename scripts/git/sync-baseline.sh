#!/usr/bin/env bash
# Keep the fork's `main` a clean mirror of the official upstream's `main`,
# WITHOUT checking out main -- your current feature branch and WIP are untouched.
#
#   upstream = sgl-project/sglang-omni   (official repo)
#   origin   = your fork                 (single source of truth for both clusters)
#
# Safe to run from any branch, on either cluster. Idempotent.
# Aborts (never force-pushes) if the fork's main carries commits upstream lacks.
set -euo pipefail

UPSTREAM_URL="https://github.com/sgl-project/sglang-omni.git"

# 1. ensure upstream remote exists (self-heal on a fresh clone)
if ! git config --get remote.upstream.url >/dev/null 2>&1; then
  echo ">> adding upstream remote -> ${UPSTREAM_URL}"
  git remote add upstream "$UPSTREAM_URL"
fi

# 2. fetch both remotes
echo ">> fetching upstream + origin..."
git fetch upstream
git fetch origin

before="$(git rev-parse origin/main 2>/dev/null || echo '')"

# 3. safety gate: does origin/main carry commits NOT in upstream/main?
unique="$(git log --oneline upstream/main..origin/main 2>/dev/null || true)"
if [ -n "$unique" ]; then
  echo "ABORT: origin/main has commit(s) not present in upstream/main:" >&2
  echo "$unique" | sed 's/^/  /' >&2
  echo "These would be discarded by a baseline reset. Resolve them first." >&2
  exit 1
fi

# 4. move local main to upstream/main without disturbing the working tree
if [ "$(git symbolic-ref --short -q HEAD || true)" = "main" ]; then
  echo ">> currently on main; fast-forwarding (ff-only)"
  git merge --ff-only upstream/main
else
  # Move the ref only; do NOT touch branch.main tracking (stays on origin/main).
  git update-ref refs/heads/main "$(git rev-parse upstream/main)"
fi

# 5. push to the fork, protected by a lease on the value we just observed
echo ">> pushing main to origin (force-with-lease)..."
if [ -n "$before" ]; then
  git push origin main:main --force-with-lease="main:${before}"
else
  git push origin main:main
fi

echo
echo "main baseline synced:"
echo "  before (origin/main): ${before:-(none)}"
echo "  after  (main):        $(git rev-parse main)"
echo "  upstream/main:        $(git rev-parse upstream/main)"
