#!/usr/bin/env bash
# Pull a path from the VM repo into the local repo, e.g. tools/pull.sh runs/plan
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: tools/pull.sh <subpath>"; exit 2; }
cd "$(dirname "$0")/.."
sub="${1%/}"
mkdir -p "$(dirname "$sub")"
rsync -az -e "ssh -o BatchMode=yes -o LogLevel=ERROR" "vast-ai:/workspace/lineloop/$sub" "$(dirname "$sub")/"
