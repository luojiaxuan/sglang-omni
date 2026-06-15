#!/usr/bin/env bash
# Pin this clone to one ref so every cluster runs identical code.
# Run with the SAME <ref> on H100 and B200, then diff the printed "commit:" lines:
# they must match for benchmark results to be comparable.
#
#   usage: scripts/git/checkout-pinned.sh <branch-name|commit-sha>
set -euo pipefail

ref="${1:-}"
if [ -z "$ref" ]; then
  echo "usage: $0 <branch-name|commit-sha>" >&2
  exit 2
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">> fetching origin..." >&2
git fetch origin >&2
# Best-effort fetch of the specific ref (covers a raw SHA the server lets us request).
git fetch origin "$ref" >&2 2>/dev/null || true

echo ">> checking out ${ref}..." >&2
git checkout "$ref"

echo
echo "=== provenance ==="
bash "$here/provenance.sh"
