"""Cross-validated XGBoost for final-QC failure, on the GPU (data stays in device memory) or the CPU.

  GPU: cuDF frame -> CuPy matrix -> xgboost.QuantileDMatrix on device="cuda"; nothing goes back to the host but
       the out-of-fold scores.
  CPU: pandas -> NumPy -> the same QuantileDMatrix with the hist method on CPU threads.

The number of boosting rounds is fixed (no early stopping on the scored fold), so out-of-fold scores are never
used to tune the model that produced them.

Two ways to cut folds:
  stratified   k shuffled folds with the same failure rate (how the Kaggle competitors reported CV). Each model
               trains on parts made before *and after* the ones it scores, so it is optimistic for a live model.
  time         forward chaining, in decision-time order: the parts are cut into k + 1 consecutive blocks, and
               block f + 1 is scored by a model trained on blocks 0..f only. Neighbour results are visible only
               from the training blocks, and the threshold of each scored block comes from earlier scored
               blocks. This is the number a deployed model can expect.
"""
import json
import time
from pathlib import Path

import numpy as np

from . import mcc

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "eta": 0.03,
    "max_depth": 8,
    "min_child_weight": 2,
    "subsample": 0.85,
    "colsample_bytree": 0.4,
    "colsample_bylevel": 0.8,
    "max_bin": 256,
    "base_score": 0.0058,   # the failure rate; starting there saves early rounds
    "seed": 0,
}
ROUNDS = 800


def matrix(X):
    """float32 2-D array on the frame's device: CuPy for cudf, NumPy for pandas."""
    if type(X).__module__.startswith("cudf"):
        return X.to_cupy(dtype="float32")
    return X.to_numpy(dtype=np.float32)


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def folds(y, k=5, seed=0):
    """Stratified fold number per row (deterministic)."""
    from sklearn.model_selection import StratifiedKFold
    f = np.empty(len(y), dtype=np.int8)
    for i, (_, va) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        f[va] = i
    return f


def time_folds(decide, k=5):
    """Forward chaining: block number 0..k per row, in decision-time order (block 0 only ever trains)."""
    order = np.argsort(np.asarray(decide, np.float64), kind="stable")
    f = np.empty(len(order), dtype=np.int8)
    for b, idx in enumerate(np.array_split(order, k + 1)):
        f[idx] = b
    return f


def _take(A, idx):
    if hasattr(A, "get"):
        import cupy as cp
        return A[cp.asarray(idx)]
    return A[idx]


def _fill_labels(A, names, nbr, y, known):
    """Write the neighbour-result features into their reserved columns of A (in place, on A's device)."""
    from .features import label_features
    lab = (nbr or {}).get("label_names")
    if not lab or not names or list(names[-len(lab):]) != list(lab):
        return
    L = label_features(nbr, y, known)
    if "label_keep" in nbr:                    # some label features were dropped (an ablation)
        L = L[:, nbr["label_keep"]]
    if hasattr(A, "get"):
        import cupy as cp
        L = cp.asarray(L)
    A[:, len(names) - len(lab):] = L


def cv(X, y, k=5, seed=0, device="cpu", params=None, rounds=ROUNDS, names=None, nbr=None, bags=1, log=print,
       split="stratified"):
    """Out-of-fold scores and metrics. X: frame or array; y: 0/1 labels; nbr: neighbour index (features.build),
    whose results are features only from the training rows. split: "stratified" or "time" (forward chaining on
    nbr["decide"]; the first block is never scored and its rows get NaN scores)."""
    import xgboost as xgb
    prm = {**PARAMS, **(params or {}), "device": device}
    A = matrix(X) if hasattr(X, "columns") else X
    names = names or (list(X.columns) if hasattr(X, "columns") else None)
    y = host(y.to_numpy() if hasattr(y, "to_numpy") else y).astype(np.float32)
    forward = split == "time"
    if forward:
        if nbr is None or "decide" not in nbr:
            raise ValueError("split='time' needs the neighbour index's decision times")
        fold = time_folds(nbr["decide"], k)
    else:
        fold = folds(y, k, seed)
    oof = np.full(len(y), np.nan) if forward else np.zeros(len(y), dtype=np.float64)
    gain = {}
    per = []
    for f in (range(1, k + 1) if forward else range(k)):
        trm = fold < f if forward else fold != f
        tr, va = np.where(trm)[0], np.where(fold == f)[0]
        _fill_labels(A, names, nbr, y, trm)
        t0 = time.perf_counter()
        dtr = xgb.QuantileDMatrix(_take(A, tr), label=y[tr], feature_names=names, max_bin=prm["max_bin"])
        dva = xgb.DMatrix(_take(A, va), feature_names=names)
        p = np.zeros(len(va))
        t_pred = 0.0
        for b in range(bags):                  # seed bagging: average models that differ only in their seed
            bst = xgb.train({**prm, "seed": prm["seed"] + 1000 * b}, dtr, rounds)
            tp = time.perf_counter()
            p += bst.predict(dva) / bags
            t_pred += time.perf_counter() - tp
            for kk, v in bst.get_score(importance_type="total_gain").items():
                gain[kk] = gain.get(kk, 0.0) + v
        oof[va] = p
        t2 = time.perf_counter()
        t1 = t2 - t_pred
        del dtr, dva
        m, t = mcc.best(y[va], oof[va])
        per.append({"fold": f, "n": int(len(va)), "pos": int(y[va].sum()), "auc": mcc.auc(y[va], oof[va]),
                    "mcc_best": m, "train_s": round(t1 - t0, 2), "predict_s": round(t2 - t1, 3)})
        log(f"  fold {f}: auc {per[-1]['auc']:.4f}  mcc {m:.4f}  train {t1 - t0:.1f} s")
    sc = ~np.isnan(oof)
    ys, ps, fs = y[sc], oof[sc], fold[sc]
    best_m, thr = mcc.best(ys, ps)
    honest_m, ths = mcc.forward_honest(ys, ps, fs) if forward else mcc.honest(ys, ps, fs)
    lo, hi = mcc.bootstrap(ys, ps, thr, n=200, seed=seed)
    total = sum(gain.values()) or 1.0
    top = sorted(gain.items(), key=lambda kv: -kv[1])[:30]
    summary = {
        "parts": int(len(y)), "failures": int(y.sum()), "features": int(A.shape[1]), "folds": k, "seed": seed,
        "split": split, "scored_parts": int(sc.sum()), "scored_failures": int(ys.sum()),
        "rounds": rounds, "bags": bags, "device": device, "params": prm,
        "auc": mcc.auc(ys, ps), "mcc_best": best_m, "threshold": thr, "mcc_best_ci95": [lo, hi],
        "mcc_honest": honest_m, "fold_thresholds": ths, "confusion": mcc.confusion(ys, ps, thr),
        "per_fold": per, "train_s_total": round(sum(p["train_s"] for p in per), 1),
        "importance_top30": [(n_, round(v / total, 4)) for n_, v in top],
    }
    return summary, oof, fold


def fit(X, y, device="cpu", params=None, rounds=ROUNDS, names=None, nbr=None, rows=None):
    """One model for deployment, on every labelled part or only `rows` (e.g. those decided before a cut-off)."""
    import xgboost as xgb
    prm = {**PARAMS, **(params or {}), "device": device}
    A = matrix(X) if hasattr(X, "columns") else X
    names = names or (list(X.columns) if hasattr(X, "columns") else None)
    y = host(y.to_numpy() if hasattr(y, "to_numpy") else y).astype(np.float32)
    _fill_labels(A, names, nbr, y, None)
    if rows is not None:
        A, y = _take(A, rows), y[rows]
    t0 = time.perf_counter()
    bst = xgb.train(prm, xgb.QuantileDMatrix(A, label=y, feature_names=names, max_bin=prm["max_bin"]), rounds)
    return bst, time.perf_counter() - t0


def save(out, summary, ids=None, y=None, oof=None, fold=None):
    """summary.json always; oof.parquet (per part: local only, never committed) when scores are given."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if oof is not None:
        import pandas as pd
        pd.DataFrame({"Id": host(ids), "y": y, "p": oof, "fold": fold}).to_parquet(out / "oof.parquet")
