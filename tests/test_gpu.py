"""GPU path (skipped without the [gpu] extra): cuDF builds exactly the pandas matrix, for every recipe of both
simulated lines."""
import numpy as np
import pandas as pd
import pytest

from lineloop import line as L
from lineloop.data import cache as C
from lineloop.quality import features as F

from conftest import make_cache

cudf = pytest.importorskip("cudf")
pytestmark = pytest.mark.gpu
CASES = [(spec, r) for spec in ("bosch-synthetic", "battery-module") for r in F.recipes(L.read(spec))]


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    return {spec: make_cache(tmp_path_factory.mktemp(spec), spec, 20000, 5)
            for spec in ("bosch-synthetic", "battery-module")}


@pytest.mark.parametrize("spec,name", CASES)
def test_cudf_matches_pandas(caches, spec, name):
    """Regression: cuDF's default 32-bit row hash collided, and its xxhash64 returns null for rows with a null;
    both silently merged groups (sig_n, num_dup_n) at the line-3 checkpoint."""
    st, num, ln = C.load(caches[spec], pd)
    a = F.build(st, num, pd, ln, **F.recipe(name, ln))
    st, num, ln = C.load(caches[spec], cudf)
    b = F.build(st, num, cudf, ln, **F.recipe(name, ln))
    assert a[3] == b[3]
    A, B = a[0], b[0].get()
    np.testing.assert_array_equal(np.isnan(A), np.isnan(B))
    np.testing.assert_array_equal(np.nan_to_num(A), np.nan_to_num(B))
    np.testing.assert_array_equal(a.nbr["idx"], b.nbr["idx"])


def test_tensorrt_runs_the_signal_stage(caches, tmp_path):
    """The compiled raw-signal model on TensorRT: the same scores as XGBoost on the history's features."""
    pytest.importorskip("tensorrt")
    from lineloop.data import simulate
    from lineloop.edge import compile as EC
    from lineloop.edge import scorers
    from lineloop.quality import train
    st, num, ln = C.load(caches["battery-module"], pd)
    sg = ln.signal_specs()[0]
    ds = F.build(st, num, pd, ln, **F.recipe("post-weld", ln))
    X, y, ids, names = ds
    bst, _ = train.fit(X, y, names=names, nbr=ds.nbr, rounds=60)
    bst.save_model(tmp_path / "model.json")
    EC.write_signal_manifest(tmp_path, ln, names)
    EC.compile_model(tmp_path / "model.json", tmp_path)
    rows = np.arange(0, len(ids), len(ids) // 300)
    raw = simulate.raw_source(ln, caches["battery-module"], sg,
                              src=caches["battery-module"].parent / "src")(ids[rows])
    Xr = np.asarray(X, np.float32)[rows]
    d = np.abs(scorers.make("trt", tmp_path)(Xr, raw) - bst.inplace_predict(Xr))
    assert np.mean(d > 1e-5) <= 0.01            # float32 summation order can still move a value across a split
