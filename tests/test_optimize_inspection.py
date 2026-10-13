"""lineloop.optimize.inspection: the MILP picks sensible policies, and the sim-opt loop tightens until it keeps up."""
import numpy as np
import pytest

from lineloop.optimize import inspection as I
from lineloop.optimize import mip
from lineloop.twin import inspection as T

pytest.importorskip("warp")


def window(n=6000, seed=0, weeks=4):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, weeks * T.WEEK_H, n)).astype(np.float32)
    fam = rng.integers(0, 2, n).astype(np.int32)
    risk = np.where(fam == 0, 0.08, 0.005)                       # family 0 is far riskier
    y = (rng.random(n) < risk).astype(np.int32)
    score = np.clip(rng.normal(0.15 + 0.6 * y, 0.12), 0, 1).astype(np.float32)
    return {"t_h": t, "score": score, "y": y, "fam": fam, "saved_work": np.full(n, 9.0, np.float32)}


def solver():
    s = [x for x in ("highs", "cuopt") if x in mip.available()]
    if not s:
        pytest.skip("no MIP solver")
    return s[0]


def test_milp_prefers_the_risky_family_and_staffs_for_it():
    w = window(n=40000)
    pol, info = I.solve(w, 2, I.ECONOMICS, rho=0.85, solver=solver())
    assert info["status"] == "optimal"
    on = pol["cap"] > 0
    assert on.any() and np.all(pol["share"][0, on] > 0)                 # staffed slots inspect the risky family
    assert np.all(pol["share"][0] >= pol["share"][1])
    assert np.all(pol["share"][:, ~on] == 0)                            # nobody on shift: never flag


def test_expensive_inspectors_mean_no_inspection():
    econ = {**I.ECONOMICS, "inspector_hour": 1e6}
    pol, _ = I.solve(window(), 2, econ, rho=0.85, solver=solver())
    assert pol["cap"].sum() == 0 and np.all(pol["share"] == 0)


def test_capacity_row_holds_every_week():
    w = window()
    pol, _ = I.solve(w, 2, I.ECONOMICS, rho=0.5, solver=solver())
    tab = I.tables(w, I.thresholds(w, 2))
    for h in range(T.N_SHIFTS):
        j = [list(I.SHARES).index(pol["share"][f, h]) for f in range(2)]
        load = sum(tab["flags"][f, h, j[f]] for f in range(2)) * I.ECONOMICS["svc_mean_h"]
        assert np.all(load <= 0.5 * T.SHIFT_H * pol["cap"][h] + 1e-6)


def test_loop_tightens_until_bypass_target():
    w = window(n=12000, seed=3)
    econ = {**I.ECONOMICS, "svc_cv": 1.2, "max_wait_h": 0.5}
    pol, hist = I.optimise(w, 2, econ, solver=solver(), rho=0.95, target_bypass=0.02, reps=8, max_iter=4,
                           log=lambda *_: None)
    assert hist[0]["slots_over"] > 0                                            # the first MILP answer overran
    assert hist[-1]["slots_over"] == 0 or len(hist) == 4                       # fixed, or out of iterations
    assert hist[-1]["bypass_rate"] < hist[0]["bypass_rate"]
    assert [h["rho_min"] for h in hist] == sorted([h["rho_min"] for h in hist], reverse=True)
