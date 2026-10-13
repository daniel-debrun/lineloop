# Quality runs: every cross-validation run, in order

Plan: [docs/PLAN.md § Q](../../docs/PLAN.md). All runs: 1,183,747 labelled parts, 5-fold stratified CV, seed 0,
RTX 5090 (XGBoost 3.4.1 `device=cuda`, cuDF 25.12), unless noted. MCC "best" = best threshold over pooled
out-of-fold scores; "honest" = each fold's threshold from the other folds. Listed in the order they were run, so
the selection that led to the final number is visible.

| run | recipe | change | features | AUC | MCC best [95 % CI] | MCC honest | train s |
|---|---|---|---|---|---|---|---|
| kaggle-v1 | kaggle | baseline: station times, route, order "magic" (start / decision / Id order), groups, all 968 measurements | 1165 | 0.9154 | 0.4340 [0.4228, 0.4437] | 0.4310 | 74.9 |
| kaggle-v2 | kaggle | + QC results of neighbouring parts (start, Id, decision order), hidden for the scored fold | 1179 | 0.9177 | 0.4596 [0.4486, 0.4697] | 0.4584 | 81.6 |
| kaggle-v3 | kaggle | + known failures and results among parts sharing the start tick, end tick, timestamp signature, measurement vector | 1187 | 0.9201 | 0.4669 [0.4568, 0.4790] | 0.4654 | 83.8 |
| sweep1 #0 | kaggle | v3 features, eta 0.02, 1300 rounds | 1187 | 0.9199 | 0.4704 | 0.4683 | 130.0 |
| sweep1 #1 | kaggle | v3, max_depth 10, min_child_weight 4 | 1187 | 0.9195 | 0.4674 | 0.4663 | 92.2 |
| sweep1 #2 | kaggle | v3, colsample_bytree 0.25 | 1187 | 0.9204 | 0.4693 | 0.4654 | 82.4 |
| sweep1 #3 | kaggle | v3, colsample_bytree 0.6, bylevel 1.0 | 1187 | 0.9204 | 0.4651 | 0.4628 | 80.1 |
| sweep1 #4 | kaggle | v3, min_child_weight 6, gamma 0.5 | 1187 | 0.9204 | 0.4677 | 0.4635 | 81.1 |
| sweep1 #5 | kaggle | v3, scale_pos_weight 3 | 1187 | 0.9192 | 0.4589 | 0.4556 | 80.2 |
| kaggle-v4 | kaggle | + route, hour, day and 7 station-time groups (counts and known failures); **stopped after fold 0** (AUC 0.8956, MCC 0.4145): leave-one-out group sums leak the part's own label through S − y. Fixed by counting only members of other Id classes (Id mod 5); a test now flips each part's own label and requires its features unchanged. v1–v3 had the weaker form of this leak in their four groups; it can only mis-train the model (scored-fold labels were never used), so their CV numbers stand | 1216 | — | — | — | — |
| v4b #0 | kaggle | v4 with the Id-class rule, default parameters | 1216 | 0.9212 | 0.4691 [0.4584, 0.4805] | 0.4682 | 95.8 |
| v4b #1 | kaggle | v4 with the Id-class rule, eta 0.02, 1300 rounds | 1216 | 0.9209 | 0.4741 [0.4616, 0.4852] | 0.4713 | 154.1 |
| v5 #0 | kaggle | + station-time groups for all 31 stations that at least 10 % of parts visit; eta 0.02, 1300 rounds | 1288 | 0.9209 | 0.4810 [0.4697, 0.4922] | 0.4805 | 148.9 |
| v5 #1 | kaggle | v5, 3 seeds per fold averaged | 1288 | 0.9217 | 0.4798 [0.4696, 0.4905] | 0.4794 | 444.2 |
| v6 | kaggle | + previous / next part through each of the 31 busy stations (Id and time gaps; competition recipe only). Worse in every fold: **removed**; v5 #0 is the competition model | 1412 | 0.9205 | 0.4701 [0.4603, 0.4812] | 0.4689 | 159.4 |

**Found after the runs above (2026-10-08):** with cuDF, the row hashes behind the `l3-entry` recipe's signature
and measurement groups were wrong: cuDF's default 32-bit hash collided, and its 64-bit xxhash returns null for
any row with a missing value. `sig_n` and `num_dup_n` differed from pandas by up to 1.2 million on that recipe.
Fixed (64-bit hash of null-filled rows), parity on `l3-entry` is now exact (1,148,234,590 cells, 0 differences),
and `tests/test_gpu.py` checks every recipe. The `kaggle` and `causal` recipes key these groups on the ingest's
64-bit hashes and were not affected; the `l3-entry` CV, model and everything downstream were re-run.
