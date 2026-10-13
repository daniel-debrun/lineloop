# Plan: what each run must show, fixed before it runs

Each section is written before its run and not edited afterwards; results are reported against it in
[`results/`](../results). A check that fails stays failed, with the reason.

## Q. Failure prediction on Bosch (written 2026-10-08, before the first full-data run)

**Question.** How well does Line Loop predict which parts fail final QC, by the competition's metric, and how
much of that survives when the model may only use what an edge node knows live?

**Data.** Kaggle "Bosch Production Line Performance": 1,183,747 labelled parts (6,879 failures, 0.581 %),
968 measurements, 1,156 timestamps; the 1,183,748 unlabelled test parts' timestamps are used for line order only
(no test measurements, no test labels: local cross-validation only, no submissions).

**Protocol.** 5-fold stratified CV, seed 0, fixed rounds (no early stopping on the scored fold). MCC is reported
two ways: at the best threshold over pooled out-of-fold scores (how competitors reported CV), and with each
fold's threshold taken from the other four (the stricter number). 95 % bootstrap interval at the best threshold.

**Reference points.** Public leaderboard, 1,371 teams ([leaderboard.json](../results/quality/leaderboard.json)):
#1 0.52491, #10 0.51161, gold cut (#12) 0.50934, silver cut (#68) 0.47113, bronze cut (#137) 0.43972.

**Recipes.** `kaggle` (whole route, any neighbour part); `causal` (end of line, only what was known when the part
finished); `l3-entry` (causal, before the part enters line 3).

| id | check | threshold |
|---|---|---|
| Q1 | `kaggle` CV MCC (best threshold) | ≥ 0.4711 (silver cut) |
| Q2 | `kaggle` CV MCC (best threshold) | ≥ 0.5093 (gold cut), stretch |
| Q3 | a "beats #1" claim | made only if both MCCs > 0.5249, and stated as CV, not a leaderboard score |
| Q4 | `causal` CV MCC (best threshold) | reported; no threshold (it is the price of going live) |
| Q5 | `l3-entry` CV MCC | > 0 with a bootstrap interval excluding 0 |
| Q6 | cuDF and pandas build the same matrix | 0 missing-value mismatches, max abs difference < 1e-4 |

**What this would not show.** CV on train is not a leaderboard score; top teams' CV and LB tracked closely but
not exactly. Thresholds and features may be iterated on CV; every iteration is logged in
`results/quality/runs.md`, so the final number's selection is visible.

## E, P, T. Edge node, inspection policy, twin on Bosch (written 2026-10-08, while the Bosch replay was running and before its, the plan's or the twin's results were seen)

Already seen when this was written, so not tests: the compiled program's parity on boundary rows (ONNX Runtime
1.8e-6, TensorRT 2.0e-6) and the inference benchmark. Developed and tuned on the synthetic line only: the
expected-value MILP inputs (after the realised-count version lost money on the synthetic hold-out) and the
per-slot tightening loop.

**Split.** `l3-entry` model trained on parts decided before t = 1171.0 (70 % quantile of line-3 entry times);
policy calibrated on [1171.0, 1410.3); hold-out after 1410.3. Default economics (`ECONOMICS` in
`lineloop/optimize/inspection.py`), assumed, not measured.

| id | check | threshold |
|---|---|---|
| E3 | the live replay scores the same parts as the batch forward check and reproduces its AUC (0.8625) | equal to 4 decimals |
| E4 | decision latency per busy tick, p99 | < 1 s (the tick is ~6 min) |
| P1 | optimised policy's hold-out net cost vs the best constant policy (both chosen on calibration only) | ≤ |
| P2 | optimised policy's hold-out net cost vs today (no early inspection) | ≤ 0 |
| P3 | calibration's predicted net cost vs the hold-out's | reported |
| P4 | hold-out bypass rate of the optimised policy | ≤ 5 % |
| T2 | twin GPU vs CPU throughput (part-steps/s, same kernel) | reported |

## R. Bosch after the review (written 2026-10-09, before any of these runs)

A review of the project asked: is "going live costs 0.007 MCC" real when both numbers come from
shuffled folds that train on later parts? Does the line-3 model read the process or mostly the part numbering
(about 45 % of its top-30 gain is part-number and production-order features)? Is the twin's GPU speed-up fair
when Warp's CPU backend uses one core? Does cuOpt earn its place? These runs answer with the same data and models.
The l3-entry model, its features and the split (T1 = 1171.0, T2 = 1410.3) are unchanged.

| id | check | threshold |
|---|---|---|
| R1 | `causal` and `l3-entry` CV forward in time (5 blocks scored, each by a model trained on the blocks before it; threshold from earlier blocks) | reported, next to the shuffled CV; the README's "price of going live" becomes offline shuffled vs `causal` forward |
| R2 | `l3-entry` forward AUC (trained before T1) without the part-number features (`_id$` and the Id-neighbour `I_` features) | reported, next to 0.8644 |
| R3 | the same without any production-order or neighbour feature (`^[SEI]_`, `start_rank`, `nbr_ysum`) | the claim "the line-3 model reads the process, not only the order" is made only if this AUC is ≥ 0.75 |
| R4 | plan, with the best constant policy held to the same 2 % bypass target as the optimiser | P1 and P2 as before |
| R5 | twin: the plan's policies under 8,192 drawn economics on the hold-out weeks | reported; GPU timed against one CPU core and every core on the same scenarios, each run long enough to time |
| R6 | `lineloop place` for `l3-entry` | reported |
| R7 | cuOpt vs HiGHS on the inspection MILP grown to 1 to 256 lines sharing a crew (120 s limit) | "cuOpt is faster" is said only for sizes where it reaches an objective at least as good in less time |
