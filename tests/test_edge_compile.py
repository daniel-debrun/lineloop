"""lineloop.edge.compile: the tensorised tree program takes exactly XGBoost's branches, NaN included."""
import numpy as np
import pytest

from lineloop.edge import compile as C

xgb = pytest.importorskip("xgboost")


def _model(tmp_path, depth=5, rounds=40, nan_rate=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3000, 12)).astype(np.float32)
    X[rng.random(X.shape) < nan_rate] = np.nan
    y = ((np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 3]) + np.isnan(X[:, 5])) > 0.8).astype(int)
    bst = xgb.train({"objective": "binary:logistic", "max_depth": depth, "eta": 0.3, "base_score": 0.2,
                     "tree_method": "hist"}, xgb.DMatrix(X, y), rounds)
    p = tmp_path / "model.json"
    bst.save_model(p)
    return bst, p, X


def test_numpy_program_matches_xgboost(tmp_path):
    bst, p, X = _model(tmp_path)
    trees, base, n_feat, _ = C.trees_from_xgboost(p)
    t = C.tensorize(trees)
    ref = bst.predict(xgb.DMatrix(X))
    got = C.predict_numpy(t, base, X)
    assert n_feat == 12 and t["depth"] <= 5
    assert np.max(np.abs(got - ref)) < 2e-6


def test_padding_shallow_leaves_and_lossguide(tmp_path):
    # lossguide trees are lopsided: leaves at many depths, so padding is exercised hard
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 6)).astype(np.float32)
    X[rng.random(X.shape) < 0.2] = np.nan
    y = (np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) > 0.1).astype(int)
    bst = xgb.train({"objective": "binary:logistic", "grow_policy": "lossguide", "max_leaves": 9, "max_depth": 0,
                     "tree_method": "hist"}, xgb.DMatrix(X, y), 25)
    bst.save_model(tmp_path / "m.json")
    trees, base, _, _ = C.trees_from_xgboost(tmp_path / "m.json")
    t = C.tensorize(trees)
    assert np.max(np.abs(C.predict_numpy(t, base, X) - bst.predict(xgb.DMatrix(X)))) < 2e-6


def test_onnx_runtime_matches_xgboost(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    bst, p, X = _model(tmp_path, depth=6, rounds=60)
    man = C.compile_model(p, tmp_path / "out")
    s = ort.InferenceSession(str(tmp_path / "out" / "model.onnx"), providers=["CPUExecutionProvider"])
    for B in (1, 7, 3000):
        got = s.run(["p"], {"x": X[:B]})[0]
        assert got.shape == (B,)
        assert np.max(np.abs(got - bst.predict(xgb.DMatrix(X[:B])))) < 2e-6
    assert man["trees"] == 60 and man["features"] == 12
