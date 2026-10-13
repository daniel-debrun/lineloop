"""Raw in-station signals: the compiled signal stage equals the reference, a model with it scores raw blocks as the
trained model scores the history, and the placement check's verdict follows its costs."""
import numpy as np
import pandas as pd
import pytest

from lineloop.data import cache, simulate
from lineloop.edge import compile as EC
from lineloop.edge import place as PL
from lineloop.edge import scorers
from lineloop.edge import signals as G
from lineloop.quality import features as F
from lineloop.quality import train

ort = pytest.importorskip("onnxruntime")


def test_signal_stage_in_onnx_equals_the_reference(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    sg = G.Signal("weld", ["a"], ["power", "photodiode", "depth"], 20000, 120, 5)
    nodes, init = G.onnx_nodes(sg)
    g = helper.make_graph(nodes, "t", [helper.make_tensor_value_info("sig", TensorProto.FLOAT, ["b", *sg.shape])],
                          [helper.make_tensor_value_info("sigfeat", TensorProto.FLOAT, ["b", len(sg.feature_names())])],
                          init)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    raw = simulate.waveforms(sg, np.random.default_rng(1).normal(size=(9, 5)) * 2, np.arange(9))
    got = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"sig": raw})[0]
    np.testing.assert_allclose(got, G.features_numpy(sg, raw), atol=1e-6)


def test_waveforms_are_deterministic_and_carry_the_weld_quality():
    sg = G.Signal("weld", ["a"], ["power", "photodiode", "depth"], 20000, 300, 8)
    z = np.random.default_rng(2).normal(size=(400, 8)) * 1.5
    seeds = np.arange(400)
    a, b = simulate.waveforms(sg, z, seeds), simulate.waveforms(sg, z, seeds)
    np.testing.assert_array_equal(a, b)
    f = G.features_numpy(sg, a)
    names = sg.feature_names()
    depth_max = f[:, names.index("weld.depth.end_max")]
    assert np.corrcoef(z.max(axis=1), depth_max)[0, 1] > 0.3


def test_compiled_model_scores_raw_blocks_like_the_trained_model(battery, tmp_path):
    """The post-weld model sees the weld signal's features; compiled, it takes the raw blocks instead and computes
    them in the graph: same scores, to float32 rounding."""
    st, num, ln = cache.load(battery, pd)
    sg = ln.signal_specs()[0]
    ds = F.build(st, num, pd, ln, **F.recipe("post-weld", ln))
    X, y, ids, names = ds
    assert set(sg.columns()) <= set(names)
    assert not set(sg.columns()) & set(F.build(st, num, pd, ln, **F.recipe("pre-weld", ln))[3])
    bst, _ = train.fit(X, y, names=names, nbr=ds.nbr, rounds=40, params={"max_depth": 4})
    bst.save_model(tmp_path / "model.json")
    assert EC.write_signal_manifest(tmp_path, ln, names) is not None
    man = EC.compile_model(tmp_path / "model.json", tmp_path)
    assert man["signal"]["features"] == len(sg.columns())
    rows = np.arange(0, len(ids), max(len(ids) // 40, 1))
    raw = simulate.raw_source(ln, battery, sg, src=battery.parent / "src")(ids[rows])
    Xr = np.asarray(X, np.float32)[rows]
    np.testing.assert_array_equal(G.features_numpy(sg, raw), Xr[:, [names.index(c) for c in sg.columns()]])
    ref = bst.inplace_predict(Xr)
    blank = Xr.copy()
    blank[:, [names.index(c) for c in sg.columns()]] = np.nan          # the engine must not need them in x
    got = scorers.make("ort-cpu", tmp_path)(blank, raw)
    assert np.max(np.abs(got - ref)) < 1e-5


def test_placement_verdict_follows_the_costs():
    arr = np.sort(np.random.default_rng(0).uniform(0, 30 * 24 * 3600.0, 20000))
    comp = np.full(10, 0.002)
    calm = {**PL.DEFAULTS, "outages_per_month": 0.0, "deadline_s": 10.0}
    r = PL.assess(arr, calm, 150e3, 500, comp, comp, value_per_part=5.0, draws=4)
    assert r["placements"]["server"]["late_share"] == 0 and r["verdict"] == "server"
    stop = {**calm, "outages_per_month": 4.0, "outage_minutes": 60.0, "undecided": "stop",
            "downtime_per_hour": 5000.0}
    r = PL.assess(arr, stop, 150e3, 500, comp, comp, value_per_part=5.0, draws=4)
    assert r["placements"]["server"]["late_share"] > 0 and r["verdict"] == "edge"
    assert r["placements"]["edge"]["late_share"] == 0
    thin = {**calm, "uplink_mbps": 0.015}             # 150 kB every ~2 min through 15 kbit/s: 2/3 busy, it queues
    r = PL.assess(arr, thin, 150e3, 500, comp, comp, value_per_part=5.0, draws=2)
    assert r["placements"]["server"]["uplink_utilisation"] > 0.5
