#!/usr/bin/env bash
# The full Bosch study (lines/bosch.toml) on a GPU box, in order; every step writes JSON under runs/ and a log under runs/logs/.
#   tools/gpu_pipeline.sh [cache] [t1] [t2]
# t1 / t2: train before t1, calibrate the inspection policy on [t1, t2), hold out after t2 (defaults: the 70 % and
# 85 % quantiles of decision times at line-3 entry, from `python -m lineloop demo`'s _splits).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/workspace/venvs/ll/bin/python}
CACHE=${1:-/workspace/data/cache/bosch}
T1=${2:-1171.0}
T2=${3:-1410.3}
P='{"eta": 0.02}'
R=1300
L="$PY -W ignore -m lineloop"
mkdir -p runs/logs
export COLUMNS=150 PYTHONUNBUFFERED=1
# FROM=<step> resumes at that step (cv-causal cv-l3entry fwd-causal fwd-l3entry fit-l3 abl-id abl-order compile
# bench replay plan twin place solvers)
# re-resolve the line every time, so the cache carries the spec's current [edge] and [economics] tables
$L line ingest --line bosch --cache "$CACHE" --keep-parquet
FROM=${FROM:-cv-causal}
skip=1
run() {
  local name=$1; shift
  [ "$name" = "$FROM" ] && skip=0
  [ $skip = 1 ] && { echo "== $name: skipped (FROM=$FROM)"; return; }
  echo "== $name: $*"
  "$@" 2>&1 | grep -v "^Warp\|^   \|Module lineloop" | tee "runs/logs/$name.log"
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "== $name failed"; exit 1; }
}
run cv-causal   $L quality cv  --recipe causal   --cache "$CACHE" --params "$P" --rounds $R --out runs/quality/causal
run cv-l3entry  $L quality cv  --recipe l3-entry --cache "$CACHE" --params "$P" --rounds $R --out runs/quality/l3-entry
run fwd-causal  $L quality cv  --recipe causal   --cache "$CACHE" --params "$P" --rounds $R --split time \
                    --out runs/quality/causal-forward
run fwd-l3entry $L quality cv  --recipe l3-entry --cache "$CACHE" --params "$P" --rounds $R --split time \
                    --out runs/quality/l3-entry-forward
run fit-l3      $L quality fit --recipe l3-entry --cache "$CACHE" --params "$P" --rounds $R --until "$T1" --out runs/edge/l3-entry
run abl-id      $L quality fit --recipe l3-entry --cache "$CACHE" --params "$P" --rounds $R --until "$T1" \
                    --drop '_id$|^I_' --out runs/ablation/l3-entry-no-id
run abl-order   $L quality fit --recipe l3-entry --cache "$CACHE" --params "$P" --rounds $R --until "$T1" \
                    --drop '^[SEI]_|^start_rank$|^nbr_ysum$' --out runs/ablation/l3-entry-no-order
run compile     $L edge compile --model runs/edge/l3-entry
run bench       $L edge bench   --model runs/edge/l3-entry --out runs/edge/bench
run replay      $L edge replay  --model runs/edge/l3-entry --cache "$CACHE" --t-from "$T1" --backend trt --out runs/edge/replay
run plan        $L plan --scored runs/edge/replay/scored.parquet --cache "$CACHE" --split "$T1,$T2" \
                    --sweep-value 50,100,150,300,600 --out runs/plan
run twin        $L twin --scored runs/edge/replay/scored.parquet --cache "$CACHE" --plan runs/plan/plan.json \
                    --scenarios 8192 --out runs/twin
run place       $L place --model runs/edge/l3-entry --cache "$CACHE" --plan runs/plan/plan.json --out runs/place
run solvers     $L solvers --scored runs/edge/replay/scored.parquet --cache "$CACHE" --plan runs/plan/plan.json \
                    --out runs/solvers
