"""A small mixed-integer model and two interchangeable solvers.

  cuopt   NVIDIA cuOpt (GPU MIP; `cuopt-cu13` / `cuopt-cu12` from pypi.nvidia.com)
  highs   HiGHS via `highspy` (CPU, open source), the fallback

The model is minimise c.x subject to lo <= A x <= hi, lb <= x <= ub, some x integer.
Every solver gets exactly the same model, so answers are comparable (`python -m lineloop solvers` compares them).
"""
import math
import time

import numpy as np

INF = math.inf
SOLVERS = ("cuopt", "highs")


class MIP:
    def __init__(self, name="mip"):
        self.name = name
        self.lb, self.ub, self.cost, self.integer, self.names = [], [], [], [], []
        self.rows = []  # (dict var -> coef, lo, hi)

    def var(self, lb=0.0, ub=INF, cost=0.0, integer=False, name=None):
        self.lb.append(float(lb))
        self.ub.append(float(ub))
        self.cost.append(float(cost))
        self.integer.append(bool(integer))
        self.names.append(name or f"x{len(self.names)}")
        return len(self.lb) - 1

    def binary(self, cost=0.0, name=None):
        return self.var(0, 1, cost, True, name)

    def row(self, coefs, lo=-INF, hi=INF):
        self.rows.append(({int(k): float(v) for k, v in coefs.items() if v != 0}, float(lo), float(hi)))

    @property
    def n_vars(self):
        return len(self.lb)

    def csr(self):
        """(values, indices, offsets, row_lo, row_hi) of A."""
        vals, idx, off = [], [], [0]
        for coefs, _, _ in self.rows:
            for k in sorted(coefs):
                idx.append(k)
                vals.append(coefs[k])
            off.append(len(idx))
        lo = np.array([r[1] for r in self.rows], dtype=float)
        hi = np.array([r[2] for r in self.rows], dtype=float)
        return (np.array(vals, dtype=float), np.array(idx, dtype=np.int32), np.array(off, dtype=np.int32), lo, hi)

    def size(self):
        return {"vars": self.n_vars, "int_vars": int(sum(self.integer)), "rows": len(self.rows),
                "nnz": sum(len(r[0]) for r in self.rows)}

    def check(self, x, tol=1e-6):
        """Max violation of bounds, rows and integrality for a solution vector."""
        x = np.asarray(x, dtype=float)
        v = max(0.0, float(np.max(np.array(self.lb) - x, initial=0)), float(np.max(x - np.array(self.ub), initial=0)))
        for coefs, lo, hi in self.rows:
            a = sum(c * x[k] for k, c in coefs.items())
            v = max(v, lo - a, a - hi)
        ints = x[np.array(self.integer, dtype=bool)]
        v = max(v, float(np.max(np.abs(ints - np.round(ints)), initial=0)))
        return v


def available():
    """Solvers importable here (cuopt also needs a visible CUDA GPU)."""
    from importlib.util import find_spec
    out = []
    if find_spec("cuopt") is not None:
        try:
            import ctypes
            ctypes.CDLL("libcuda.so.1")
            out.append("cuopt")
        except OSError:
            pass
    if find_spec("highspy") is not None:
        out.append("highs")
    return out


def solve(m, solver="auto", time_limit=60.0, rel_gap=1e-6, threads=None):
    """Solve and return {solver, status, objective, bound, x, time_s, solver_time_s, size}.

    status: optimal | feasible (stopped with a solution, not proven) | infeasible | no_solution | error.
    time_s is wall time of the whole call (model hand-over, GPU transfer, solve, read-back).
    solver="auto": cuopt if available, else highs.
    """
    if solver == "auto":
        avail = available()
        if not avail:
            raise RuntimeError("no MIP solver installed: pip install highspy (or cuopt-cu13)")
        solver = avail[0]
    fn = {"cuopt": _cuopt, "highs": _highs}[solver]
    t0 = time.perf_counter()
    r = fn(m, time_limit, rel_gap, threads)
    r["time_s"] = time.perf_counter() - t0
    r["solver"] = solver
    r["size"] = m.size()
    if r.get("x") is not None:
        x = np.asarray(r["x"], dtype=float)
        r["max_violation"] = m.check(x)
        ints = np.array(m.integer, dtype=bool)
        x[ints] = np.round(x[ints])  # cuOpt returns e.g. 1e-14 for a zero binary
        r["x"] = x
    return r


def _cuopt(m, time_limit, rel_gap, threads):
    from cuopt import linear_programming as lp
    from cuopt.linear_programming.solver.solver_parameters import (CUOPT_LOG_TO_CONSOLE, CUOPT_MIP_RELATIVE_GAP,
                                                                   CUOPT_NUM_CPU_THREADS, CUOPT_TIME_LIMIT)
    vals, idx, off, lo, hi = m.csr()
    dm = lp.DataModel()
    dm.set_csr_constraint_matrix(vals, idx, off)
    dm.set_constraint_lower_bounds(lo)
    dm.set_constraint_upper_bounds(hi)
    dm.set_objective_coefficients(np.array(m.cost, dtype=float))
    dm.set_variable_lower_bounds(np.array(m.lb, dtype=float))
    dm.set_variable_upper_bounds(np.array(m.ub, dtype=float))
    dm.set_variable_types(np.array(["I" if i else "C" for i in m.integer]))
    s = lp.SolverSettings()
    s.set_parameter(CUOPT_TIME_LIMIT, float(time_limit))
    s.set_parameter(CUOPT_MIP_RELATIVE_GAP, float(rel_gap))
    s.set_parameter(CUOPT_LOG_TO_CONSOLE, False)
    if threads:
        s.set_parameter(CUOPT_NUM_CPU_THREADS, int(threads))
    sol = lp.Solve(dm, s)
    reason = str(sol.get_termination_reason())
    x = np.asarray(sol.get_primal_solution(), dtype=float)
    status = {"Optimal": "optimal", "FeasibleFound": "feasible", "Infeasible": "infeasible"}.get(reason)
    if status is None:
        status = "feasible" if len(x) == m.n_vars and m.check(x) < 1e-5 else "no_solution"
    stats = sol.get_milp_stats() if any(m.integer) else {}
    has_x = status in ("optimal", "feasible")
    return {"status": status, "reason": reason, "objective": float(sol.get_primal_objective()) if has_x else None,
            "bound": float(stats["solution_bound"]) if stats.get("solution_bound") is not None else None,
            "x": x if has_x else None, "solver_time_s": float(sol.get_solve_time())}


def _highs(m, time_limit, rel_gap, threads):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(rel_gap))
    if threads:
        h.setOptionValue("threads", int(threads))
    n = m.n_vars
    inf = highspy.kHighsInf
    lb = np.array([max(v, -inf) for v in m.lb])
    ub = np.array([min(v, inf) for v in m.ub])
    h.addVars(n, lb, ub)
    cols = np.arange(n, dtype=np.int32)
    h.changeColsCost(n, cols, np.array(m.cost, dtype=float))
    types = np.array([highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous for i in m.integer])
    if any(m.integer):
        h.changeColsIntegrality(n, cols, types)
    vals, idx, off, lo, hi = m.csr()
    lo, hi = np.maximum(lo, -inf), np.minimum(hi, inf)
    h.addRows(len(m.rows), lo, hi, len(vals), off[:-1], idx, vals)
    t0 = time.perf_counter()
    h.run()
    st = h.getModelStatus()
    info = h.getInfo()
    name = h.modelStatusToString(st)
    has_x = info.primal_solution_status == 2  # kSolutionStatusFeasible
    if st == highspy.HighsModelStatus.kOptimal:
        status = "optimal"
    elif st == highspy.HighsModelStatus.kInfeasible:
        status, has_x = "infeasible", False
    else:
        status = "feasible" if has_x else "no_solution"
    return {"status": status, "reason": name, "objective": float(info.objective_function_value) if has_x else None,
            "bound": float(info.mip_dual_bound) if any(m.integer) else None,
            "x": list(h.getSolution().col_value) if has_x else None, "solver_time_s": time.perf_counter() - t0}

