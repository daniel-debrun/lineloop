"""Matthews correlation coefficient (the competition's metric) over every threshold at once.

Kaggle scored hard 0/1 predictions with MCC, so a probability model also needs a threshold. Two ways to report:
  best        the threshold that maximises MCC on the pooled out-of-fold predictions (what competitors quoted
              as "CV"); slightly optimistic, since the threshold is fitted on the scored data
  honest      each fold's threshold is chosen on the other folds' predictions, then applied to that fold
"""
import numpy as np


def curve(y, p):
    """(thresholds, mcc) for the rule `p >= t`, one point per distinct score, descending."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    o = np.argsort(-p, kind="stable")
    ps, ys = p[o], y[o]
    last = np.r_[ps[1:] != ps[:-1], True]        # cut only between distinct scores
    tp = np.cumsum(ys)[last]
    fp = np.cumsum(1 - ys)[last]
    P, N = ys.sum(), len(ys) - ys.sum()
    fn, tn = P - tp, N - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)
    return ps[last], m


def best(y, p):
    """(mcc, threshold) maximising MCC for `p >= threshold`."""
    t, m = curve(y, p)
    i = int(np.argmax(m))
    return float(m[i]), float(t[i])


def at(y, p, threshold):
    y = np.asarray(y).astype(bool)
    h = np.asarray(p) >= threshold
    tp, fp = float(np.sum(h & y)), float(np.sum(h & ~y))
    fn, tn = float(np.sum(~h & y)), float(np.sum(~h & ~y))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def confusion(y, p, threshold):
    y = np.asarray(y).astype(bool)
    h = np.asarray(p) >= threshold
    return {"tp": int(np.sum(h & y)), "fp": int(np.sum(h & ~y)), "fn": int(np.sum(~h & y)), "tn": int(np.sum(~h & ~y))}


def honest(y, p, fold):
    """MCC of pooled predictions where fold k's threshold comes from the other folds; (mcc, thresholds)."""
    y, p, fold = np.asarray(y), np.asarray(p), np.asarray(fold)
    hard = np.zeros(len(y), dtype=bool)
    ths = []
    for k in np.unique(fold):
        _, t = best(y[fold != k], p[fold != k])
        ths.append(t)
        hard[fold == k] = p[fold == k] >= t
    return at(y, hard.astype(float), 0.5), ths


def forward_honest(y, p, fold):
    """Forward chaining: each scored block's threshold is the best on the blocks scored before it (the first scored
    block has none and is left out); (mcc over the rest, thresholds)."""
    y, p, fold = np.asarray(y), np.asarray(p), np.asarray(fold)
    blocks = np.unique(fold)
    hard, keep, ths = np.zeros(len(y), bool), np.zeros(len(y), bool), []
    for i, k in enumerate(blocks[1:], 1):
        prev = np.isin(fold, blocks[:i])
        _, t = best(y[prev], p[prev])
        ths.append(t)
        m = fold == k
        hard[m], keep[m] = p[m] >= t, True
    return (at(y[keep], hard[keep].astype(float), 0.5) if keep.any() else float("nan")), ths


def bootstrap(y, p, threshold, n=200, seed=0):
    """(lo, hi) 95 % interval of MCC at a fixed threshold, resampling parts."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(bool)
    h = np.asarray(p) >= threshold
    m = len(y)
    vals = []
    for _ in range(n):
        i = rng.integers(0, m, m)
        vals.append(at(y[i], h[i].astype(float), 0.5))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def auc(y, p):
    """ROC AUC with ties averaged (Mann-Whitney)."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    r = rankdata(np.asarray(p, dtype=np.float64))
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
