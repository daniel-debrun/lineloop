#!/usr/bin/env bash
# One-time setup of a GPU box (tested: vast.ai, RTX 5090, Ubuntu 24.04, driver 580): a uv venv with the pinned
# NVIDIA stack, then the repo in editable mode. Run from the repo root on the box: deploy/vm/setup.sh
set -euo pipefail
VENV=${VENV:-/workspace/venvs/ll}
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -q -p 3.12 "$VENV"
uv pip install -q -p "$VENV/bin/python" -r deploy/requirements-gpu.txt --extra-index-url https://pypi.nvidia.com
uv pip install -q -p "$VENV/bin/python" -e ".[dev]"
"$VENV/bin/python" -m lineloop doctor
