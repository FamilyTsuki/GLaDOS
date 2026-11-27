#!/usr/bin/env bash
# Weekly quota burn on an always-on server (see .claude/commands/setup-quota-server.md).
# 1. pull the latest code, 2. generate until just before the weekly reset,
# 3. commit the new parts and push them to the dataset-auto branch.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MINUTES="${BURN_MINUTES:-175}"
WORKERS="${BURN_WORKERS:-3}"
MODEL="${BURN_MODEL:-opus}"
CODE_BRANCH="${CODE_BRANCH:-myGLaDOS}"
DATA_BRANCH="${DATA_BRANCH:-dataset-auto}"

cd "$REPO"
git fetch origin "$CODE_BRANCH"
git merge --no-edit "origin/$CODE_BRANCH"

python3 training/quota/burn_weekly_quota.py --force --minutes "$MINUTES" \
    --workers "$WORKERS" --model "$MODEL"

# training/parts/ is gitignored: force-add only the generated data, not the logs.
shopt -s nullglob
files=(training/parts/claude_*.jsonl)
if (( ${#files[@]} )); then
    git add -f "${files[@]}"
    git commit -m "data: weekly quota burn $(date +%F)" || true
    git push origin "HEAD:$DATA_BRANCH"
fi
