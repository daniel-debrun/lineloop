# Scoreboard

Rendered by `python -m lineloop report` from the JSON files in this folder.

## Bosch failure prediction against the Kaggle leaderboard (MCC)

| entry | MCC | 95 % CI | MCC, threshold from other folds | AUC |
|---|---|---|---|---|
| public leaderboard #1 | 0.5249 |  |  |  |
| public leaderboard #10 | 0.5116 |  |  |  |
| public leaderboard #12 (gold cut) | 0.5093 |  |  |  |
| public leaderboard #25 | 0.4951 |  |  |  |
| Line Loop, competition recipe (5-fold CV) | 0.4810 | [0.4697, 0.4922] | 0.4805 | 0.9209 |
| public leaderboard #50 | 0.4796 |  |  |  |
| public leaderboard #68 (silver cut) | 0.4711 |  |  |  |
| public leaderboard #137 (bronze cut) | 0.4397 |  |  |  |

Line Loop numbers are cross-validation on the 1,183,747 labelled parts, not a leaderboard submission (no test-set measurements were used or submitted). Leaderboard: public board, 1371 teams, downloaded 2026-10-08. Selection history: results/quality/runs.md.

## What going live costs: shuffled folds against forward in time

| recipe | folds | AUC | MCC (best threshold) | 95 % CI | MCC, threshold from other (shuffled) / earlier (forward) folds |
|---|---|---|---|---|---|
| competition: whole route, any neighbour part | shuffled, 5 folds | 0.9209 | 0.4810 | [0.4697, 0.4922] | 0.4805 |
| live, end of line: only what was known when the part finished | shuffled, 5 folds | 0.9155 | 0.4737 | [0.4637, 0.4846] | 0.4707 |
|   the same, forward in time | forward: 5 blocks | 0.8655 | 0.2628 | [0.2498, 0.2741] | 0.1689 |
| live, before line 3: own data up to line 3 + live context | shuffled, 5 folds | 0.9009 | 0.2770 | [0.2706, 0.2829] | 0.2254 |
|   the same, forward in time | forward: 5 blocks | 0.8619 | 0.2218 | [0.2170, 0.2262] | 0.2174 |

Shuffled folds train on parts made after the ones they score, which no live model can; forward folds score each block of time with a model trained on the blocks before it. Causal recipes are reproduced bit for bit by the streaming edge node (tests/test_edge_stream.py).

## Does the line-3 model read the process, or the part order? (forward in time)

| l3-entry model | later parts | AUC | change |
|---|---|---|---|
| all features | 354824 | 0.8644 |  |
| without the part-number features (_id, Id-neighbour I_) | 354824 | 0.8309 | -0.0334 |
| without any production-order or neighbour feature (S_, E_, I_, start_rank, nbr_ysum) | 354824 | 0.7989 | -0.0654 |

Trained on parts decided before t = 1171.0, scored on every later part; only the features differ.

## RAPIDS cuDF vs pandas: the same feature matrix, built on GPU and CPU

| recipe | cells compared | missing mismatches / max abs diff | pandas | cuDF | speed-up |
|---|---|---|---|---|---|
| kaggle | 1,379,065,255 (v1 features) | 0 / 0 | 61.4 s | 18.7 s | 3.3x |
| l3-entry | 1,148,234,590 | 0 / 0 | 114.1 s | 60.1 s | 1.9x |

Builds include sorting 2.37 M parts in three orders, group counts and the float32 matrix fill; the competition recipe was checked on its first (v1) feature set.

## Edge inference backends (the l3-entry model)

| backend | batch | p50 ms | p99 ms | parts/s | max |dp| vs XGBoost CPU |
|---|---|---|---|---|---|
| xgb-cpu | 1 | 0.235 | 0.337 | 4,263 | 0.0e+00 |
| xgb-cpu | 16 | 1.239 | 6.987 | 12,909 | 0.0e+00 |
| xgb-cpu | 256 | 3.700 | 23.000 | 69,192 | 0.0e+00 |
| xgb-cpu | 4096 | 17.483 | 35.871 | 234,278 | 0.0e+00 |
| xgb-cuda | 1 | 4.107 | 4.200 | 243 | 2.6e-06 |
| xgb-cuda | 16 | 6.223 | 6.318 | 2,571 | 2.6e-06 |
| xgb-cuda | 256 | 6.616 | 6.707 | 38,694 | 2.6e-06 |
| xgb-cuda | 4096 | 7.637 | 7.744 | 536,367 | 2.6e-06 |
| fil | 1 | 0.181 | 0.300 | 5,520 | 2.0e-06 |
| fil | 16 | 0.184 | 0.304 | 86,957 | 2.0e-06 |
| fil | 256 | 0.234 | 0.601 | 1,093,459 | 2.0e-06 |
| fil | 4096 | 1.131 | 2.339 | 3,622,349 | 2.0e-06 |
| trt | 1 | 0.052 | 0.063 | 19,216 | 2.0e-06 |
| trt | 16 | 0.065 | 0.069 | 245,234 | 2.0e-06 |
| trt | 256 | 0.142 | 0.157 | 1,808,687 | 2.0e-06 |
| trt | 4096 | 1.878 | 1.901 | 2,180,639 | 2.0e-06 |
| ort-cpu | 1 | 0.244 | 0.281 | 4,100 | 2.0e-06 |
| ort-cpu | 16 | 3.627 | 10.085 | 4,411 | 2.0e-06 |
| ort-cpu | 256 | 13.488 | 22.936 | 18,980 | 2.0e-06 |
| ort-cpu | 4096 | 201.630 | 205.593 | 20,314 | 2.0e-06 |

Model: 1300 trees of depth 8, 970 features, compiled to a 4.3 MB ONNX tensor program. Timings include host<->device copies. GPU: NVIDIA GeForce RTX 5090

## Edge node replaying the line, forward in time

| metric | value |
|---|---|
| events replayed (starts, QC results, decisions) | 1,739,702 |
| labelled parts scored live / failures | 354,824 / 1,412 |
| AUC / MCC (best threshold) on parts after the training cut-off | 0.8644 / 0.1931 |
| throughput | 1,658 events/s (trt) |
| decision latency per tick p50 / p99 | 52.64 / 118.08 ms |

A tick is 0.01 Bosch time units (about 6 min); the node spends milliseconds of it.

## Early inspection before line 3, hold-out weeks (per week)

| policy | net cost | inspector h | inspections | defects caught / findable / all | L3 visits saved | bypass |
|---|---|---|---|---|---|---|
| today (no early inspection) | 0 ± 0 | 0 | 0 | 0.0 / 36.5 / 36.5 | 0 | 0.0% |
| best constant policy (none) | 0 ± 0 | 0 | 0 | 0.0 / 36.5 / 36.5 | 0 | 0.0% |
| best per-family policy (greedy search in the twin) | 7,345 ± 22 | 168 | 53 | 1.9 / 36.5 / 36.5 | 13 | 4.8% |
| optimised (MILP + twin) | 0 ± 0 | 0 | 0 | 0.0 / 36.5 / 36.5 | 0 | 0.0% |

cuopt MILP (1932 variables, 350 rows) x Warp twin on cuda:0, 1 loop iteration(s). Economics (assumed): value_per_station 8.0, value_per_defect 150.0, handling 3.0, inspector_hour 45.0, wait_hour 1.0, svc_mean_h 0.25, svc_cv 0.5, detect 0.9, max_wait_h 2.0, shift_h 8.0

## Break-even: optimised policy by value of a defect caught early (hold-out weeks)

| value / defect | net cost / week | inspector h | defects caught / all |
|---|---|---|---|
| 50 | 0 ± 0 | 0 | 0.0 / 36.5 |
| 100 | 0 ± 0 | 0 | 0.0 / 36.5 |
| 150 | 0 ± 0 | 0 | 0.0 / 36.5 |
| 300 | 703 ± 13 | 16 | 0.1 / 36.5 |
| 600 | 3,828 ± 117 | 128 | 3.4 / 36.5 |

## Does the plan survive its assumptions? 8,192 drawn economics per policy (hold-out weeks)

| policy | saves money in | net / week, median | 5 % to 95 % | break-even value / defect | moved most by |
|---|---|---|---|---|---|
| best constant policy (none) | 0% | 0 | 0 to 0 | - | max_wait_h +0.01, svc_mean_h -0.01 |
| best per-family policy (greedy search in the twin) | 0% | 7,509 | 5,264 to 10,700 | 4,809 | inspector_hour +0.99, value_per_defect -0.10 |
| optimised (MILP + twin) | 0% | 0 | 0 to 0 | - | max_wait_h +0.01, svc_mean_h -0.01 |

Every assumption drawn log-uniformly from its range (lineloop.twin.robust.RANGES); the same scenarios and seeds for every policy.

## Line twin (NVIDIA Warp): one kernel launch, every scenario a thread

| device | scenarios | seconds | part-steps/s |
|---|---|---|---|
| GPU (cuda:0) | 1,442,339 | 1.185 | 215,811,592,210 |
| CPU, 1 core | 8,448 | 4.807 | 311,681,154 |
| CPU, 8 processes | 8,448 | 1.262 | 1,186,909,328 |

GPU speed-up: 692x against CPU, 1 core, 182x against CPU, 8 processes

## Where should the l3-entry model run? (lineloop place)

|  | server | edge node |
|---|---|---|
| data per part | 3.9 kB | 3.9 kB |
| decision delay p50 / p99 | 7.2 ms / 0.07 s | 0.24 ms / 0.25 ms |
| late decisions per month | 124.6 | 0.0 |
| cost per month (late decisions + node) | 0 | 150 |

Verdict: server: outages: 125 parts a month pass unchecked, 0 / month. Network, deadline and costs are assumptions in the spec's [edge] table.

## cuOpt vs HiGHS: the inspection MILP grown from one line to a plant sharing a crew

| lines | variables | cuopt | highs |
|---|---|---|---|
| 1 | 1,932 | 0.71 s, optimal | 0.15 s, optimal |
| 4 | 7,665 | 120.28 s, feasible, +13.11% | 85.62 s, optimal |
| 16 | 30,597 | 120.32 s, feasible, +26.62% | 120.05 s, feasible |
| 64 | 122,325 | 120.37 s, feasible, +7.95% | 120.15 s, feasible |
| 256 | 489,237 | 121.01 s, feasible | 382.78 s, feasible, +2.15% |

Same model, 120 s limit; +x% = objective worse than the best found at that size.
