#!/usr/bin/env bash
# Push the local repo to the GPU VM (/workspace/lineloop). Code only: the top-level data/, runs/ and models/
# stay where they are on each side (anchored patterns, so lineloop/data/ is synced).
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -az --delete --exclude /.venv --exclude __pycache__ --exclude /data/ --exclude /runs/ --exclude /models/ \
  --exclude .pytest_cache --exclude .ruff_cache --exclude '*.egg-info' --exclude /.git/ \
  -e "ssh -o BatchMode=yes -o LogLevel=ERROR" ./ vast-ai:/workspace/lineloop/
