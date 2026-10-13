# Results against the plan

Every check from [docs/PLAN.md](../docs/PLAN.md), with the run that decided it. Numbers: [SCOREBOARD.md](SCOREBOARD.md).

| id | check | threshold | result | verdict |
|---|---|---|---|---|
| Q1 | `kaggle` CV MCC, best threshold | ≥ 0.4711 (silver cut) | 0.4810 [0.4697, 0.4922]; 0.4805 with thresholds from other folds | **pass** |
| Q2 | `kaggle` CV MCC, best threshold | ≥ 0.5093 (gold cut), stretch | 0.4810 | **fail** |
| Q3 | a "beats #1" claim | only if both MCCs > 0.5249 | not made | — |
| Q4 | `causal` CV MCC | reported | 0.4737 [0.4637, 0.4846]; 0.4707 from other folds | reported |
| Q5 | `l3-entry` CV MCC > 0, CI excluding 0 | > 0 | 0.2770 [0.2706, 0.2829] (after the cuDF hash fix; 0.2775 before) | **pass** |
| Q6 | cuDF and pandas build the same matrix | 0 mismatches, max diff < 1e-4 | 0 / 0 on `kaggle` (v1 features) and `l3-entry` (final); every recipe on the synthetic line in `tests/test_gpu.py` | **pass** (first `l3-entry` check failed: 32-bit / null-row hashes, fixed) |
| E3 | live replay reproduces the batch forward AUC | equal to 4 decimals | first run 0.8622 vs 0.8625: **fail**, traced to the cuDF hash bug in the training features; after the fix 0.8644 vs 0.8644, scores within 6e-7 on all 354,824 parts | **pass after fix** |
| E4 | decision latency per busy tick, p99 | < 1 s | 120 ms | **pass** |
| P1 | optimised hold-out cost ≤ best constant policy | ≤ | 0 vs 0 (both: no inspection) | pass, trivially |
| P2 | optimised hold-out cost ≤ today | ≤ 0 | 0 | pass, trivially |
| P3 | calibration's predicted cost vs hold-out | reported | 0 vs 0 at default economics. Break-even sweep: at 300 and 600 per defect the calibrated policies lose 703 and 3,579 / week on the hold-out | reported: **the early-warning model is not precise enough for early inspection to pay on this line under these assumptions, and the calibration window overstates its later value** |
| P4 | hold-out bypass rate | ≤ 5 % | 0 % | pass, trivially |
| T2 | twin GPU vs CPU throughput | reported | 6.27 G vs 0.33 G part-steps/s: 19x (65,536 scenarios x 354,824 parts in 3.7 s). Note added 2026-10-09: the CPU figure was one core (Warp's CPU backend runs a launch on one thread); see R5 | reported |

## R. After the review ([plan § R](../docs/PLAN.md#r-bosch-after-the-review-written-2026-10-09-before-any-of-these-runs))

| id | check | threshold | result | verdict |
|---|---|---|---|---|
| R1 | `causal` and `l3-entry` CV forward in time | reported | `causal` MCC 0.2628 [0.2498, 0.2741], AUC 0.8655 (0.1689 with earlier blocks' thresholds), against 0.4737 shuffled; `l3-entry` MCC 0.2218 [0.2170, 0.2262], AUC 0.8619 (0.2174), against 0.2770 shuffled | reported: **"going live costs 0.007" is withdrawn**; forward in time it costs about 0.22 MCC |
| R2 | `l3-entry` forward AUC without the part-number features | reported | 0.8309 (11 features dropped), against 0.8644 | reported |
| R3 | the same without any order or neighbour feature | claim only if ≥ 0.75 | 0.7989 (30 features dropped) | **pass**: the line-3 model reads the process, not only the order |
| R4 | plan with the bypass-held baselines | P1, P2 | optimised 0 (never flags) against best constant 0 and today 0; the new greedy per-family baseline **loses 7,345 / week** on the hold-out | P1 and P2 pass trivially, as before; the greedy search is beaten |
| R5 | twin: policies under 8,192 drawn economics; GPU vs one core vs every core | reported | the optimised policy never flags (nothing to stress); greedy loses money in 100 % of scenarios. GPU 216 G part-steps/s, 692x one core, 182x the faster of 8 and 16 processes (8) on the same scenario mix | reported |
| R6 | `lineloop place` for `l3-entry` | reported | plant server: 7 ms median decision, 125 parts a month unchecked during outages, worth 0 while the policy never flags; an edge node costs 150 / month | reported |
| R7 | cuOpt vs HiGHS, 1 to 256 lines sharing a crew, 120 s | "faster" only where at least as good in less time | 1 line: both optimal, HiGHS 0.15 s vs 0.71 s; 4 lines: HiGHS optimal in 86 s, cuOpt +13 % at the limit; 16 and 64: HiGHS better; 256 lines: cuOpt better by 2.2 % in 121 s, HiGHS 383 s | **cuOpt is faster at 256 lines only** |

An earlier stage of the project kept a line-balancing benchmark (Scholl's SALBP instances) here. It used no NVIDIA
library and nothing in the framework depended on it, so it was removed after the review (2026-10-09).
