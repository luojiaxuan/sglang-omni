#!/usr/bin/env bash
# Emit a provenance block identifying the exact code + hardware a run executed on.
# Paste this into benchmark/diagnostic records so results are reproducible and
# comparable across clusters (H100 <-> B200). Run from inside the repo.
set -euo pipefail

repo_url="$(git config --get remote.origin.url 2>/dev/null || echo 'unknown')"
# normalize "git@github.com:owner/repo.git" / "https://github.com/owner/repo.git" -> "owner/repo"
repo="$(printf '%s' "$repo_url" | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')"

branch="$(git symbolic-ref --short -q HEAD || echo 'DETACHED')"
commit="$(git rev-parse HEAD)"
host="$(hostname)"

if command -v nvidia-smi >/dev/null 2>&1; then
  # group identical GPU names -> e.g. "2x NVIDIA B200" (comma-joins distinct types)
  hardware="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
    | sort | uniq -c \
    | awk '{n=$1; $1=""; sub(/^[ \t]+/,""); printf "%s%dx %s", sep, n, $0; sep=", "} END{print ""}')"
  [ -z "$hardware" ] && hardware="unknown (nvidia-smi returned no GPUs)"
else
  hardware="unknown (no nvidia-smi)"
fi

cat <<EOF
repo:     ${repo}
branch:   ${branch}
commit:   ${commit}
hardware: ${hardware}
host:     ${host}
EOF
