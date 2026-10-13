"""lineloop.twin.inspection: the Warp kernel agrees with the line-by-line reference; queueing behaves."""
import numpy as np
import pytest

from lineloop.twin import inspection as T

wp = pytest.importorskip("warp")


def stream(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 3 * T.WEEK_H, n)).astype(np.float32)
    y = (rng.random(n) < 0.05).astype(np.int32)
    score = np.clip(rng.normal(0.2 + 0.5 * y, 0.15), 0, 1).astype(np.float32)
    return {"t_h": t, "score": score, "y": y, "fam": rng.integers(0, 3, n).astype(np.int32),
            "saved_work": rng.integers(5, 10, n).astype(np.float32)}


def test_kernel_matches_reference_when_deterministic():
    s = stream()
    thr = np.array([[0.3, 0.4, 0.5], [0.0, 0.0, 0.0], [0.6, 0.6, 2.0], [0.3, 0.3, 0.3]], np.float32)
    cap = np.array([[1] * 21, [2] * 21, [0] * 10 + [3] * 11, [1] * 21])
    sc = T.scenarios(3, thr, cap, svc_mean_h=0.5, svc_cv=0.0, detect=[1.0, 1.0, 1.0, 0.0],
                     max_wait_h=[1.0, 0.25, 4.0, 1.0])
    got, hist, _ = T.simulate(s, sc, "cpu")
    for k in range(len(thr)):
        ref = T.simulate_numpy(s, sc, k)
        for name in T.OUT:
            assert got[name][k] == pytest.approx(ref[name], rel=1e-5, abs=1e-3), (k, name)
        assert hist[k].sum() == ref["inspected"]


def test_conservation_and_monotonicity():
    s = stream(seed=1)
    thr = np.linspace(0.0, 1.0, 11)
    sc = T.scenarios(3, thr, np.full(11, 2), svc_cv=0.6, detect=0.85)
    m, _, _ = T.simulate(s, sc, "cpu")
    defects = s["y"].sum()
    np.testing.assert_allclose(m["caught"] + m["escaped"], defects)                 # every defect ends somewhere
    np.testing.assert_allclose(m["inspected"] + m["bypassed"], m["flagged"])
    assert np.all(np.diff(m["flagged"]) <= 0)                                       # higher threshold, fewer flags
    more = T.simulate(s, T.scenarios(3, thr, np.full(11, 6), svc_cv=0.6, detect=0.85), "cpu")[0]
    assert np.all(more["bypassed"] <= m["bypassed"])                                # more inspectors, fewer bypasses


def test_stochastic_service_mean_is_right():
    s = stream(n=20000, seed=2)
    sc = T.scenarios(3, np.zeros(1), np.array([16]), svc_mean_h=0.3, svc_cv=0.8, detect=1.0, max_wait_h=50.0)
    m, _, _ = T.simulate(s, sc, "cpu")
    assert m["busy_h"][0] / m["inspected"][0] == pytest.approx(0.3, rel=0.03)


def test_wait_quantile_from_histogram():
    h = np.zeros((1, T.WAIT_BINS), int)
    h[0, 0], h[0, 10] = 90, 10
    assert T.wait_quantile(h, 64.0, 0.95)[0] == pytest.approx(11.0)


def test_kernel_matches_reference_with_twelve_hour_shifts():
    s = stream(seed=4)
    H = T.n_shifts(12.0)
    thr = np.array([[0.3] * 3, [0.5, 0.2, 0.9]], np.float32)
    cap = np.array([[1, 0] * (H // 2), [2] * H])
    sc = T.scenarios(3, thr, cap, svc_mean_h=0.5, svc_cv=0.0, detect=1.0, max_wait_h=1.0, shift_h=12.0)
    assert H == 14 and sc["thr"].shape == (2, 3 * H)
    got, hist, _ = T.simulate(s, sc, "cpu")
    for k in range(len(thr)):
        ref = T.simulate_numpy(s, sc, k)
        for name in T.OUT:
            assert got[name][k] == pytest.approx(ref[name], rel=1e-5, abs=1e-3), (k, name)
    assert got["slot_flagged"].shape == (2, H)
    assert got["slot_flagged"][0, 1::2].sum() == got["slot_bypassed"][0, 1::2].sum() > 0   # unstaffed nights


def test_defects_made_after_the_checkpoint_are_never_caught():
    s = stream(seed=5)
    rng = np.random.default_rng(5)
    s["present"] = np.where(s["y"] == 1, rng.random(len(s["y"])) < 0.6, 1).astype(np.int32)
    sc = T.scenarios(3, np.zeros(2, np.float32), np.array([16, 16]), svc_mean_h=0.01, svc_cv=0.0,
                     detect=[1.0, 0.0], max_wait_h=50.0)
    got, _, _ = T.simulate(s, sc, "cpu")
    present_defects = int((s["y"] * s["present"]).sum())
    assert 0 < present_defects < s["y"].sum()
    assert got["caught"][0] == present_defects == T.simulate_numpy(s, sc, 0)["caught"]   # flag all, find all present
    assert got["caught"][1] == 0
    np.testing.assert_allclose(got["caught"] + got["escaped"], s["y"].sum())


def test_every_core_gives_the_same_answer_as_one():
    s = stream(seed=6)
    sc = T.scenarios(3, np.linspace(0.2, 0.8, 6).astype(np.float32), np.arange(1, 7), svc_cv=0.5, detect=0.8)
    one, h1, _ = T.simulate(s, sc, "cpu")
    many, h2, _, procs = T.simulate_cpu_cores(s, sc, procs=3)
    assert procs == 3
    for k in T.OUT:
        np.testing.assert_array_equal(one[k], many[k])
    np.testing.assert_array_equal(h1, h2)


def test_robustness_sweep_break_even_and_sensitivity():
    from lineloop.optimize import inspection as I
    from lineloop.twin import robust as R
    s = stream(n=4000, seed=7)
    econ = {**I.ECONOMICS, "shift_h": 8.0}
    H = T.n_shifts(8.0)
    pols = {"flag top": {"thr": np.float32(0.6), "cap": np.full(H, 2)},
            "flag none": {"thr": np.float32(I.NEVER), "cap": np.zeros(H, int)}}
    res, dr, _ = R.run(s, 3, pols, econ, n=256, device="cpu")
    assert res["flag none"]["saves_share"] == 0.0 and res["flag none"]["net_median"] == 0.0
    r = res["flag top"]
    assert 0.0 <= r["saves_share"] <= 1.0 and r["net_p05"] <= r["net_median"] <= r["net_p95"]
    assert r["sensitivity"]["value_per_defect"] < 0          # worth more per defect -> lower net cost
    assert r["sensitivity"]["inspector_hour"] > 0
    assert set(dr) == set(R.RANGES) and dr["detect"].max() <= 0.99
