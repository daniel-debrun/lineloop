PY ?= .venv/bin/python
LINE ?= battery-module

.PHONY: install test lint demo sync gpu-test report figures

install:         ## CPU install (uv venv)
	uv venv -p 3.11 .venv && uv pip install -p $(PY) -e ".[dev]"

test:            ## everything that runs without a GPU (GPU tests skip)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check lineloop tests

demo:            ## the whole loop on a line (default: the simulated battery-module line), CPU is enough
	$(PY) -m lineloop demo --line $(LINE)

report:          ## results/SCOREBOARD.md from results/*.json
	$(PY) -m lineloop report

figures:         ## the README's figures (docs/img/*.svg) from results/*.json; needs the [docs] extra
	$(PY) tools/figures.py

sync:            ## code -> GPU VM
	tools/sync.sh

gpu-test: sync   ## full suite on the GPU VM, GPU tests included
	ssh -T -o BatchMode=yes vast-ai 'cd /workspace/lineloop && /workspace/venvs/ll/bin/python -m pytest -q' </dev/null
