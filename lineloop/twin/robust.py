"""Are the assumptions load-bearing? The chosen inspection policies, replayed under thousands of plausible economics.

The line's data gives the parts, their times, scores and outcomes; the inspection cell and the money are assumptions
(the spec's [economics]). So the policies the optimiser chose are run again with every assumption drawn from a
range around its value, all in one Warp launch:

  cell     svc_mean_h, svc_cv, detect, max_wait_h       change what the cell does: simulated, one thread each
  money    value_per_defect, value_per_station,         change only what it is worth: applied to the simulated
           handling, inspector_hour, wait_hour          counts afterwards (cost is linear in them)

Each scenario draws every factor log-uniformly from its range (the spec's [uncertainty] table: key = [low, high]
multipliers of the assumed value; detect is capped at 0.99). Every policy sees the same scenarios and seeds, so
their differences are not noise. Reported per policy: the share of scenarios in which it saves money against
today's line, the spread of its net cost, the value per defect at which it breaks even, and which assumption
moves its result most (rank correlation).
"""
import numpy as np

from ..optimize import inspection as I
from . import inspection as T

RANGES = {          # multipliers of the assumed value, drawn log-uniformly
    "value_per_defect": (0.33, 3.0), "value_per_station": (0.5, 2.0), "handling": (0.5, 2.0),
    "inspector_hour": (0.7, 1.5), "wait_hour": (0.5, 2.0),
    "svc_mean_h": (0.6, 2.0), "svc_cv": (0.5, 2.0), "detect": (0.67, 1.1), "max_wait_h": (0.5, 2.0),
}
CELL = ("svc_mean_h", "svc_cv", "detect", "max_wait_h")
MONEY = ("value_per_defect", "value_per_station", "handling", "inspector_hour", "wait_hour")


def draw(econ, n, ranges=None, seed=0):
    """n scenarios: {key: (n,) values} for every uncertain assumption."""
    r = {**RANGES, **(ranges or {})}
    unknown = set(r) - set(RANGES)
    if unknown:
        raise ValueError(f"unknown [uncertainty] keys {sorted(unknown)}; known: {list(RANGES)}")
    rng = np.random.default_rng(seed)
    out = {}
    for k, (lo, hi) in r.items():
        f = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        out[k] = econ[k] * f
    out["detect"] = np.minimum(out["detect"], 0.99)
    return out


def run(win, n_fam, policies, econ, n=4096, ranges=None, device="cpu", seed=0, simulate=None):
    """Every policy x n drawn scenarios over the window, in one launch. Returns (per-policy results, draws,
    kernel seconds). simulate: the function that runs the kernel (default lineloop.twin.inspection.simulate)."""
    sim = simulate or T.simulate
    shift_h = econ.get("shift_h", T.SHIFT_H)
    H = T.n_shifts(shift_h)
    names = list(policies)
    P = len(names)
    dr = draw(econ, n, ranges, seed)
    thr = np.concatenate([np.broadcast_to(_thr(policies[k]["thr"], n_fam, H), (n, n_fam, H)) for k in names])
    cap = np.concatenate([np.broadcast_to(np.broadcast_to(policies[k]["cap"], (H,)), (n, H)) for k in names])
    rep = {k: np.tile(dr[k], P) for k in dr}
    sc = T.scenarios(n_fam, thr, cap, rep["svc_mean_h"], rep["svc_cv"], rep["detect"], rep["max_wait_h"],
                     seed=np.tile(np.arange(n, dtype=np.int32) + seed, P), shift_h=shift_h)
    m, _, secs = sim(win, sc, device)
    weeks = (win["t_h"].max() - win["t_h"].min()) / T.WEEK_H
    e = {**econ, **{k: rep[k] for k in MONEY}}
    net = I.cost(m, cap, weeks, e)
    out = {}
    for i, k in enumerate(names):
        s = slice(i * n, (i + 1) * n)
        c, caught = net[s], m["caught"][s] / weeks
        other = c + dr["value_per_defect"] * caught            # net cost without the value of defects caught
        be = np.where(caught > 0, other / np.maximum(caught, 1e-12), np.inf)
        out[k] = {"scenarios": n, "saves_share": float(np.mean(c < -1e-9)), "net_median": float(np.median(c)),
                  "net_p05": float(np.quantile(c, 0.05)), "net_p95": float(np.quantile(c, 0.95)),
                  "break_even_value_per_defect_median": float(np.median(be)) if np.isfinite(be).any() else None,
                  "caught_per_week_median": float(np.median(caught)),
                  "sensitivity": {key: _spearman(dr[key], c) for key in dr}}
    return out, dr, secs


def _thr(thr, n_fam, H):
    t = np.asarray(thr, np.float32)
    return np.broadcast_to(t.reshape((n_fam, -1)) if t.ndim else t, (n_fam, H))


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def policies_from_plan(plan):
    """{name: {thr, cap}} from a plan.json (lineloop plan)."""
    return {k: {"thr": np.asarray(v["thr"], np.float32), "cap": np.asarray(v["cap"], int)}
            for k, v in plan["policies"].items()}
