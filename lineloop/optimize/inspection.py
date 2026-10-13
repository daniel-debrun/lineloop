"""Choose the inspection policy: a flag threshold per route family and shift slot, and inspectors per shift slot
(MILP), then check it in the Warp twin and tighten until the simulated cell keeps up (simulation-optimisation).

MILP, built on a calibration window of scored parts (the edge model's live scores, never its training data):

  x[f, h, j] in {0, 1}   in shift slot h, family f flags its top q_j share of parts (j = 0: never flags)
  c[h] in 0..cmax        inspectors on shift slot h (a week of the line's shifts: 21 slots of 8 h by default)

  minimise   sum x[f,h,j] * (handling * flags[f,h,j] - detect * (value_per_defect * defects[f,h,j]
                                                             + value_per_station * work[f,h,j])) / weeks
  where defects / work are expected values from the model's probabilities (one calibration factor for the
  window, and the share of failures whose defect already exists at the checkpoint, by score band), not the
  window's realised failures, which are too few per cell to choose on
             + inspector_hour * shift_h * sum_h c[h]
  subject to sum_j x[f,h,j] = 1                                                   for every family and slot
             sum_f,j x[f,h,j] * svc_mean * flags[f,h,j,w] <= rho * shift_h * c[h]   for every week w and slot h

The capacity row holds in *every* calibration week, not on average, so a busy week does not overrun the cell.
The MILP still sees only averages within a shift and a fixed service time; the twin then runs the chosen policy
on the same window with random service times and real arrival bursts. If too many flagged parts would be moved on
uninspected (bypass rate above target), rho is lowered and the MILP solved again.

Solver: NVIDIA cuOpt on a GPU, HiGHS on the CPU (lineloop.optimize.mip, same model for both).
"""
import numpy as np

from ..twin import inspection as T
from . import mip

SHARES = (0.0, 0.001, 0.002, 0.004, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15)
NEVER = 2.0                                     # a threshold no probability reaches

ECONOMICS = {      # illustrative defaults in currency units (the data has no costs); a line spec's [economics]
                   # table overrides them, and every one is also a CLI flag of `plan`
    "value_per_station": 8.0,                   # one station visit after the checkpoint not spent on a failing part
    "value_per_defect": 150.0,                  # reworking a defect at the checkpoint instead of scrapping /
                                                # diagnosing it after the final test
    "handling": 3.0,                            # per inspection, besides the inspector's time
    "inspector_hour": 45.0,                     # fully loaded labour
    "wait_hour": 1.0,                           # per part-hour waiting at the cell (WIP delay)
    "svc_mean_h": 0.25, "svc_cv": 0.5,          # inspection time: 15 min mean, lognormal
    "detect": 0.9, "max_wait_h": 2.0,           # inspector finds 90 % of real defects; a part waits at most 2 h
    "shift_h": 8.0,                             # shift length (the line spec's [shifts] hours)
}


def thresholds(win, n_fam, shares=SHARES):
    """thr[f, j]: the score at which family f flags its top shares[j] of parts in the window (j=0: never)."""
    thr = np.full((n_fam, len(shares)), NEVER, np.float32)
    for f in range(n_fam):
        s = np.sort(win["score"][win["fam"] == f])[::-1]
        for j, q in enumerate(shares):
            if q > 0 and len(s):
                k = max(int(round(q * len(s))), 1)
                thr[f, j] = s[min(k, len(s)) - 1]
    return thr


def _slots(t, shift_h):
    week = np.floor((t - t.min()) / T.WEEK_H).astype(int)
    how = t - np.floor(t / T.WEEK_H) * T.WEEK_H
    return week, np.minimum((how / shift_h).astype(int), T.n_shifts(shift_h) - 1)


def calibration(win):
    """alpha = observed failures / sum of predicted probabilities over the whole window (one number, so it is
    stable where per-cell counts are not)."""
    return float(np.clip(win["y"].sum() / max(float(win["score"].sum()), 1e-9), 0.2, 5.0))


def presence(win, bands=4):
    """Per part: the share of failures at its score band whose defect already exists at the checkpoint (from the
    calibration window's origins). An inspection cannot find a defect made later, and the model's score covers
    every failure, so the expected value of a flag is scaled by this share. All ones without origins."""
    pres = T.present(win).astype(bool)
    fail = win["y"] == 1
    if pres[fail].all() or not fail.any():
        return np.ones(len(win["y"]))
    edges = np.quantile(win["score"][fail], np.linspace(0, 1, bands + 1)[1:-1])
    b = np.searchsorted(edges, win["score"])
    share = np.array([pres[fail & (b == k)].mean() if (fail & (b == k)).any() else pres[fail].mean()
                      for k in range(bands)])
    return share[b]


def tables(win, thr, estimate="expected", shift_h=T.SHIFT_H):
    """Per (family, slot, share): flags per calibration week, and the defects flagged and the work after the
    checkpoint behind them.

    estimate="expected": defects = alpha * sum of the flagged parts' probabilities. A cell holds a handful of real
    failures, so choosing cells by realised counts picks lucky cells (the optimiser's curse: on the synthetic line
    it predicted a saving of 3,892 / week and lost 521 on the hold-out). "realised" keeps the counts for comparison.
    """
    week, sh = _slots(win["t_h"], shift_h)
    W = int(week.max()) + 1
    F, J = thr.shape
    H = T.n_shifts(shift_h)
    alpha = calibration(win)
    yv = (win["y"] * T.present(win) if estimate == "realised" else
          alpha * win["score"].astype(np.float64) * presence(win))
    flags = np.zeros((F, H, J, W))
    defects, work = np.zeros((F, H, J)), np.zeros((F, H, J))
    for f in range(F):
        m = win["fam"] == f
        for j in range(J):
            sel = m & (win["score"] >= thr[f, j])
            np.add.at(flags[f, :, j], (sh[sel], week[sel]), 1.0)
            np.add.at(defects[f, :, j], sh[sel], yv[sel])
            np.add.at(work[f, :, j], sh[sel], win["saved_work"][sel] * yv[sel])
    return {"flags": flags, "defects": defects, "work": work, "weeks": W, "alpha": alpha, "estimate": estimate}


def build(tab, econ, rho, cmax=8):
    """rho: one utilisation cap, or one per shift slot (the sim-opt loop lowers it where the twin overruns)."""
    flags, W = tab["flags"], tab["weeks"]
    F, H, J = flags.shape[:3]
    shift_h = econ.get("shift_h", T.SHIFT_H)
    rho = np.broadcast_to(np.asarray(rho, float), (H,))
    m = mip.MIP("inspection")
    x = {}
    for f in range(F):
        for h in range(H):
            for j in range(J):
                gain = econ["detect"] * (econ["value_per_defect"] * tab["defects"][f, h, j]
                                         + econ["value_per_station"] * tab["work"][f, h, j])
                # + 1e-6 * j: between equal-cost choices (a share that flags nothing in this slot), never flag
                x[f, h, j] = m.binary(cost=(econ["handling"] * flags[f, h, j].sum() - gain) / W + 1e-6 * j,
                                      name=f"x_{f}_{h}_{j}")
            m.row({x[f, h, j]: 1 for j in range(J)}, 1, 1)
    c = [m.var(0, cmax, econ["inspector_hour"] * shift_h, integer=True, name=f"c_{h}") for h in range(H)]
    for w in range(W):
        for h in range(H):
            coefs = {x[f, h, j]: econ["svc_mean_h"] * flags[f, h, j, w] for f in range(F) for j in range(J)
                     if flags[f, h, j, w] > 0}
            if coefs:
                coefs[c[h]] = -rho[h] * shift_h
                m.row(coefs, hi=0.0)
    return m, x, c


def solve(win, n_fam, econ, rho, solver="auto", cmax=8, time_limit=60.0, estimate="expected"):
    shift_h = econ.get("shift_h", T.SHIFT_H)
    thr = thresholds(win, n_fam)
    tab = tables(win, thr, estimate, shift_h)
    m, x, c = build(tab, econ, rho, cmax)
    r = mip.solve(m, solver, time_limit=time_limit, rel_gap=1e-6)
    if r["x"] is None:
        raise RuntimeError(f"inspection MILP: {r['status']}")
    sol = r["x"]
    F, J = thr.shape
    H = T.n_shifts(shift_h)
    choice = np.array([[max(range(J), key=lambda j: sol[x[f, h, j]]) for h in range(H)] for f in range(F)])
    cap = np.array([int(round(sol[k])) for k in c])
    load = np.zeros(H)                         # busiest calibration week's inspection hours per slot
    for h in range(H):
        load[h] = sum(tab["flags"][f, h, choice[f, h]] for f in range(F)).max() * econ["svc_mean_h"]
    pol = {"thr": thr[np.arange(F)[:, None], choice], "share": np.array(SHARES)[choice], "cap": cap,
           "rho": np.broadcast_to(np.asarray(rho, float), (H,)).copy(), "load_h": load}
    return pol, {"solver": r["solver"], "status": r["status"], "objective_per_week": r["objective"],
                 "seconds": r["time_s"], "size": r["size"], "alpha": tab["alpha"], "estimate": estimate}


def cost(m, cap, weeks, econ):
    """Net cost per week of a simulated policy, against today's line (no early inspection = 0). Negative = saving."""
    staffed = np.asarray(cap).reshape(len(m["flagged"]), -1).sum(axis=1) * econ.get("shift_h", T.SHIFT_H) * weeks
    total = (econ["inspector_hour"] * staffed + econ["handling"] * m["inspected"] + econ["wait_hour"] * m["wait_sum_h"]
             - econ["value_per_defect"] * m["caught"] - econ["value_per_station"] * m["work_saved"])
    return total / weeks


def evaluate(win, n_fam, policies, econ, reps=32, device="cpu", seed=0):
    """Simulate each policy `reps` times on the window; mean per-week metrics per policy."""
    names = list(policies)
    shift_h = econ.get("shift_h", T.SHIFT_H)
    H = T.n_shifts(shift_h)
    thr = np.stack([np.broadcast_to(np.asarray(policies[k]["thr"], np.float32).reshape(
        (n_fam, -1) if np.ndim(policies[k]["thr"]) else (1, 1)), (n_fam, H)) for k in names
        for _ in range(reps)])
    cap = np.vstack([np.broadcast_to(policies[k]["cap"], (H,)) for k in names for _ in range(reps)])
    sc = T.scenarios(n_fam, thr, cap, econ["svc_mean_h"], econ["svc_cv"], econ["detect"], econ["max_wait_h"],
                     seed=np.arange(len(thr), dtype=np.int32) + seed, shift_h=shift_h)
    m, hist, secs = T.simulate(win, sc, device)
    weeks = (win["t_h"].max() - win["t_h"].min()) / T.WEEK_H
    m["cost_per_week"] = cost(m, cap, weeks, econ)
    p95 = T.wait_quantile(hist, sc["max_wait"])
    out = {}
    for i, k in enumerate(names):
        s = slice(i * reps, (i + 1) * reps)
        r = {key: float(np.mean(v[s])) for key, v in m.items() if np.ndim(v) == 1}
        r["slot_flagged"] = m["slot_flagged"][s].mean(axis=0)
        r["slot_bypassed"] = m["slot_bypassed"][s].mean(axis=0)
        r["cost_sd"] = float(np.std(m["cost_per_week"][s]))
        r["bypass_rate"] = r["bypassed"] / max(r["flagged"], 1.0)
        r["wait_p95_h"] = float(np.mean(p95[s]))
        r["inspector_hours_per_week"] = float(np.sum(policies[k]["cap"]) * shift_h)
        r["defects"] = float(win["y"].sum())
        r["defects_present"] = float((win["y"] * T.present(win)).sum())     # those an inspection here could find
        r["weeks"] = float(weeks)
        out[k] = r
    return out, secs


def greedy(win, n_fam, econ, start, reps=16, device="cpu", target_bypass=0.02, passes=2, caps=(1, 2, 3, 4)):
    """A stronger baseline than one threshold for everyone: coordinate descent in the twin. Starting from `start`
    (a constant policy, one threshold), each family in turn tries every share (as its own threshold) with every
    constant crew,
    the others held, keeping the cheapest policy within the bypass target; `passes` rounds over the families.
    No MILP: what simulation and search alone find on the calibration window."""
    H = T.n_shifts(econ.get("shift_h", T.SHIFT_H))
    thr = np.full(n_fam, float(start["thr"]), np.float32)          # the constant policy: one threshold
    shares = [q for q in SHARES if q > 0]
    best = None
    for _ in range(passes):
        for f in range(n_fam):
            s = np.sort(win["score"][win["fam"] == f])[::-1]
            opts = [NEVER] + [float(s[min(max(int(round(q * len(s))), 1), len(s)) - 1]) for q in shares if len(s)]
            cand = {}
            for k, t in enumerate(opts):
                for c in caps:
                    th = thr.copy()
                    th[f] = t
                    cand[(k, c)] = {"thr": np.repeat(th[:, None], H, axis=1), "cap": np.full(H, c)}
            ev, _ = evaluate(win, n_fam, cand, econ, reps, device)
            ok = [k for k in ev if ev[k]["bypass_rate"] <= target_bypass] or list(ev)
            k = min(ok, key=lambda kk: ev[kk]["cost_per_week"])
            thr[f] = opts[k[0]]
            best = cand[k]
    return best


def optimise(win, n_fam, econ, solver="auto", device="cpu", rho=0.85, target_bypass=0.02, reps=32, max_iter=8,
             log=print, estimate="expected"):
    """MILP -> twin -> tighten where the twin overran -> MILP ... Returns (policy, history).

    A slot whose simulated bypass rate is above target gets its utilisation cap lowered just enough that its
    current load needs one more inspector; the MILP then either staffs it or flags less there."""
    shift_h = econ.get("shift_h", T.SHIFT_H)
    rho_h = np.full(T.n_shifts(shift_h), float(rho))
    hist = []
    for it in range(max_iter):
        pol, info = solve(win, n_fam, econ, rho_h, solver, estimate=estimate)
        ev, secs = evaluate(win, n_fam, {"p": pol}, econ, reps, device)
        e = ev["p"]
        slot_rate = e["slot_bypassed"] / np.maximum(e["slot_flagged"], 1e-9)
        bad = (slot_rate > target_bypass) & (e["slot_flagged"] > 0)
        hist.append({"iter": it, "rho_min": float(rho_h.min()), **info, "bypass_rate": e["bypass_rate"],
                     "wait_p95_h": e["wait_p95_h"], "cost_per_week": e["cost_per_week"], "twin_s": secs,
                     "inspector_hours_per_week": e["inspector_hours_per_week"], "slots_over": int(bad.sum())})
        log(f"  iter {it}: {info['solver']} {info['status']} in {info['seconds']:.2f} s  inspector-h/week "
            f"{e['inspector_hours_per_week']:.0f}  bypass {e['bypass_rate']:.1%}  p95 wait {e['wait_p95_h']:.2f} h"
            f"  slots over target {int(bad.sum())}  cost/week {e['cost_per_week']:.0f}")
        if not bad.any():
            break
        for h in np.where(bad)[0]:
            need = pol["load_h"][h] / (shift_h * (pol["cap"][h] + 1))       # cap at which load needs c+1
            rho_h[h] = max(min(rho_h[h], need * 0.999), 1e-3)
    return pol, hist
