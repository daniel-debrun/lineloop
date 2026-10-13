"""Does the GPU solver earn its place? The inspection MILP, grown from one line to a plant.

One line's policy MILP (lineloop.optimize.inspection) has a few hundred binaries: HiGHS on a CPU solves it in about
a second, as cuOpt does. A plant is where the model grows: L lines, each with its own families, shift slots and
shares, share one inspection crew per shift slot, so the lines' flags compete for the same inspectors and the
model no longer splits by line. Each extra line is the calibration window's line with its expected defects, work
and flags scaled by random factors (0.6 to 1.4, per family and slot), a plausible family of lines rather than
copies, so the problem does not decompose trivially.

Every solver gets the same model (lineloop.optimize.mip) and the same time limit; reported per size: wall time,
status, objective and its gap to the best objective any solver found at that size.
"""
import numpy as np

from . import inspection as I
from . import mip


def plant(tab, econ, lines, rho=0.85, seed=0, crew_max_per_line=4):
    """The plant MILP for `lines` lines sharing a crew. tab: inspection.tables() of one line's window."""
    rng = np.random.default_rng(seed)
    flags, W = tab["flags"], tab["weeks"]
    F, H, J = flags.shape[:3]
    shift_h = econ["shift_h"]
    m = mip.MIP(f"plant{lines}")
    load = {}                                     # (w, h) -> {var: inspection hours}
    for li in range(lines):
        a = rng.uniform(0.6, 1.4, (F, H, 1)) if li else np.ones((F, H, 1))
        b = rng.uniform(0.8, 1.2, (F, H, 1)) if li else np.ones((F, H, 1))
        for f in range(F):
            for h in range(H):
                xs = {}
                for j in range(J):
                    gain = econ["detect"] * (econ["value_per_defect"] * tab["defects"][f, h, j]
                                             + econ["value_per_station"] * tab["work"][f, h, j]) * a[f, h, 0]
                    fl = flags[f, h, j] * b[f, h, 0]
                    v = m.binary(cost=(econ["handling"] * fl.sum() - gain) / W + 1e-6 * j, name=f"x{li}_{f}_{h}_{j}")
                    xs[j] = v
                    for w in range(W):
                        if fl[w] > 0:
                            load.setdefault((w, h), {})[v] = econ["svc_mean_h"] * fl[w]
                m.row({v: 1 for v in xs.values()}, 1, 1)
    c = [m.var(0, crew_max_per_line * lines, econ["inspector_hour"] * shift_h, integer=True, name=f"c_{h}")
         for h in range(H)]
    for (w, h), coefs in load.items():
        m.row({**coefs, c[h]: -rho * shift_h}, hi=0.0)
    return m


def run(win, n_fam, econ, sizes=(1, 4, 16, 64), solvers=None, time_limit=60.0, log=print):
    thr = I.thresholds(win, n_fam)
    tab = I.tables(win, thr, "expected", econ["shift_h"])
    solvers = solvers or mip.available()
    out = []
    for L in sizes:
        m = plant(tab, econ, L)
        res = {}
        for s in solvers:
            r = mip.solve(m, s, time_limit=time_limit, rel_gap=1e-4)
            res[s] = {"status": r["status"], "objective": r["objective"], "seconds": r["time_s"]}
            log(f"  {L:>4} lines, {m.n_vars:,} variables: {s} {r['status']} in {r['time_s']:.2f} s, "
                f"objective {r['objective'] if r['objective'] is None else round(r['objective'], 1)}")
        objs = [v["objective"] for v in res.values() if v["objective"] is not None]
        best = min(objs) if objs else None
        for v in res.values():
            v["gap_to_best"] = (None if best is None or v["objective"] is None else
                                (v["objective"] - best) / max(abs(best), 1e-9))
        out.append({"lines": L, "variables": m.n_vars, "rows": len(m.rows), "solvers": res})
    return out
