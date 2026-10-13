# The Bosch study: the framework on a real line

The framework's evidence on real data is the Bosch Production Line Performance competition (Kaggle, 2016). It
covers:
- 1.18 million labelled parts (and as many unlabelled test parts);
- 52 stations on four lines, L0 to L3;
- 968 measurements and 1,156 timestamps;
- a failure rate of 0.58 %.

The whole line is described by [`lines/bosch.toml`](../lines/bosch.toml):
- the clock: one unit is about 10 hours;
- one checkpoint, `l3-entry`, as parts enter line 3;
- route families before it;
- 8-hour shifts.

Every number below is program output, collected in [`results/`](../results). The checks were written down
before each run in [`PLAN.md`](PLAN.md), and every check has a verdict in
[`results/RESULTS.md`](../results/RESULTS.md). Every model iteration is logged, including the ones that made
things worse: [`results/quality/runs.md`](../results/quality/runs.md).

## Results

| | result | for scale |
|---|---|---|
| benchmark: the competition setting (`offline`), shuffled 5-fold CV | **MCC 0.4810** [0.4697, 0.4922], AUC 0.921 | public board of 1,371 teams: #1 0.5249, gold cut 0.5093, #50 0.4796, silver cut 0.4711 (inside the interval). Selected over 15 logged runs. |
| the same pipeline, live (`live`: only what was known when the part finished), shuffled CV | MCC 0.4737 [0.4637, 0.4846] | shuffled folds train on later parts: not the live number |
| `live`, **forward in time** (each block scored by a model trained on the blocks before it) | **MCC 0.2628** [0.2498, 0.2741], AUC 0.866; 0.1689 with each block's threshold from earlier blocks | the price of going live: about 0.22 MCC |
| early warning as parts enter line 3 (`l3-entry`) | AUC 0.901 (shuffled CV); AUC 0.862, MCC 0.2218 [0.2170, 0.2262] (forward CV; 0.2174 with earlier blocks' thresholds); **AUC 0.864 forward** on 354,824 parts after the training cut-off | |
| `l3-entry` without the part-number features (11 features: `_id`, Id-neighbours `I_`), forward | AUC 0.8309 | -0.034 |
| `l3-entry` without any production-order or neighbour feature (30 features), forward | AUC 0.7989 | -0.066; the pre-registered bar for "reads the process" was 0.75 |
| cuDF feature build vs pandas | identical matrices (1.15 billion cells, 0 differences), 1.9 to 3.3x faster | |
| edge inference, TensorRT | **0.052 ms** per call at batch 1, **1.81 M parts/s** at batch 256; within 2e-6 of XGBoost | cuML FIL: 3.62 M parts/s at batch 4,096; XGBoost on one CPU core: 0.235 ms at batch 1 |
| edge node replaying 1.74 M real events | features equal to the batch ones; p99 **118 ms** per 6-minute tick, almost all of it Python bookkeeping, not inference | |
| cuOpt inspection policy at line-3 entry, hold-out weeks | declines early inspection at the default costs; at 300 to 600 per defect caught, the policies it calibrates **lose** money on later weeks | a negative result, below |
| the same, against a greedy per-family search in the twin (held to the same bypass target) | the greedy policy, chosen on the calibration weeks, **loses 7,345 / week** on the hold-out | the optimiser's "never" stands |
| the policies under 8,192 drawn economics (Warp) | the optimiser's policy never flags, so there is nothing to stress; the greedy policy loses money in every scenario, and would break even only at about 4,800 per defect | |
| Warp twin speed, the same scenarios on each | 692x one CPU core, **182x the CPU's best** (8 processes on the VM's 8 cores) | earlier runs said "19x the CPU": that was one core, timed on another mix of scenarios |
| placement of the `l3-entry` model (`lineloop place`) | **plant server**: 7 ms median decision against a 6-minute deadline; outages would let 125 parts a month through unchecked, which costs nothing while the policy never flags | an edge node would add 150 / month for nothing |
| cuOpt vs HiGHS on the plan's MILP grown to a plant (120 s limit) | 1 line: both optimal, HiGHS in 0.15 s, cuOpt 0.7 s. 4 to 64 lines: HiGHS better. **256 lines (489,237 binaries): cuOpt's plan 2.2 % better, in 121 s against HiGHS's 383 s** | cuOpt earns its place at plant scale only |

**Not reached:** the gold cut (CV 0.4810 against 0.5093). No leaderboard submission was made, so these are
cross-validation numbers, not leaderboard scores, and the end-of-line recipes are a benchmark: at the end of the
line the final test already has the answer. The deployable model is the line-3 checkpoint's.

**The inspection result is negative.** On this line, the line-3 early warning is not precise enough for an
inspection cell to pay under the assumed costs. Policies calibrated on earlier weeks also overstate their value on
later ones. The 600-per-defect row is not stable to the unit: three runs of the same pipeline lost 3,579, 4,034 and
3,828 / week, choosing different policies. The cause was not isolated: the solver's near-ties, or tiny score
differences from retraining on the GPU. The ± reported per row covers only the twin's random seeds, not this
run-to-run variance. Every run loses money, so the conclusion stands.
The simulated battery-module line, where the warning signs are stronger, shows the same loop finding savings
([README](../README.md#what-it-found)).

## What is real and what is assumed

| real (from the data) | assumed (the spec's `[economics]`, `[edge]`, defaults) |
|---|---|
| every part's stations, timestamps, measurements and final result | the cost of an inspector-hour, of a station visit after line 3, of a defect found late |
| the line's order, batches and gaps (train and test parts interleave on the line) | the inspection cell: 15 min mean (lognormal), finds 90 % of defects, 2 h maximum wait |
| the live model's score for each hold-out part, forward in time | 1 Bosch time unit ≈ 10 h (inferred from the day and week cycles; Bosch did not document it) |
| | where defects arise: the data has no origins, so every failure counts as findable at line-3 entry, and `detect` has to cover defects made in line 3 itself |
| | the network at the line and what an edge node costs (placement) |

The test parts' measurements and labels were never available (no submission was made). The test parts are on
the simulated line and shape every neighbour feature, but they are not scored.

## Lessons the logs record

- **A leave-one-out leak.** Adding group sizes next to leave-one-out group failure counts dropped fold AUC from
  0.924 to 0.896. Fixed with the Id-class rule, which a test now guards
  ([How it works § leak controls](HOW_IT_WORKS.md#leak-controls)).
- **The optimiser's curse.** Choosing inspection cells by their realised failures (about 2 per cell) predicted a
  saving that vanished on hold-out weeks. Expected failures from the model's probabilities, with one calibration
  factor, transfer. The greedy baseline added after the review shows it again on real data: searching per-family
  thresholds by simulated cost on the calibration weeks found a policy that lost 7,345 / week on the hold-out.
- **cuDF row hashes.** The default hash is 32-bit, and the 64-bit one returns null for any row with a missing
  value. Both silently merged groups at the line-3 checkpoint. The parity check caught it, and a GPU test now
  guards it.
- **Not every idea helps.** Per-station previous / next parts lowered CV MCC from 0.4810 to 0.4701 and were
  removed; seed bagging made no difference.
- **Shuffled folds flatter a live model.** Shuffled CV put the price of going live at 0.007 MCC (0.4810 to
  0.4737). Forward in time the live model scores 0.2628, so the real price is about 0.22. Some of the gap is
  less training data in the early blocks (the first is scored by a model trained on a sixth of the history),
  but that is also what a newly deployed model faces. Every live number is now reported forward in time too.
- **Ask what the model reads.** About 45 % of the line-3 model's top-30 gain was production-order and
  part-number features, which could mean it mostly reads the part numbering. Ablated, forward in time: without
  the part-number features AUC drops from 0.864 to 0.831, without every order and neighbour feature to 0.799.
  Gain shares overstate what a feature family is worth when others can stand in for it; the ablation is the
  measure.

## The study after the framework refactor

The Bosch study was first run with Bosch-specific code. The framework now reads everything from
`lines/bosch.toml` instead:
- the 31 "group" stations come from the rule "stations at least 10 % of parts visit", which selects exactly the
  31 the old code listed by hand;
- the line-3 checkpoint, the route blocks and the 10-hour clock come from the spec.

The refactored code was checked on the real data. For all three recipes it builds the same feature matrices, cell
for cell, as the code that produced the results above, with the same labels, Ids and neighbour index. So the
numbers above are the framework's numbers.

## Reproducing it

On a machine with an NVIDIA GPU (setup: [`deploy/README.md`](../deploy/README.md)), with the Kaggle files in
`data/bosch`:

```bash
python -m lineloop line ingest --line bosch                     # Kaggle CSVs -> data/cache/bosch
python -m lineloop quality cv --line bosch --recipe offline --params '{"eta": 0.02}' --rounds 1300
tools/gpu_pipeline.sh data/cache/bosch                         # live recipes (shuffled and forward), ablations,
                                                               # edge, replay, plan, twin, placement, solvers
python -m lineloop report --collect                             # results/SCOREBOARD.md
```
