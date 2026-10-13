"""lineloop.quality on simulated lines: ingest, recipes, causality, leaks, MCC. The Bosch-shaped line goes through
the Bosch adapter; the battery line (tests/test_line.py) through the generic one."""
import json

import numpy as np
import pandas as pd
import pytest

from lineloop.quality import features as F
from lineloop.quality import mcc, train


@pytest.fixture(scope="module")
def frames(bosch_frames):
    return bosch_frames


def test_ingest_station_times(frames, bosch_like):
    st, num, ln = frames
    meta = json.loads((bosch_like / "meta.json").read_text())
    assert meta["numeric"]["rows"] + meta["dates"]["rows"]["test_date"] == 6000
    assert set(st["is_train"].unique()) == {0, 1}
    assert len(num) == int((st["is_train"] == 1).sum())
    t0 = st[[c for c in st.columns if c.endswith("_t0")]]
    t1 = st[[c for c in st.columns if c.endswith("_t1")]].to_numpy()
    assert np.all(np.nan_to_num(t1, nan=0) >= np.nan_to_num(t0.to_numpy(), nan=0))
    assert num["num_hash"].dtype == np.int64


@pytest.mark.parametrize("name", ["kaggle", "causal", "l3-entry"])
def test_recipes_build(frames, name):
    st, num, ln = frames
    X, y, ids, names = ds = F.build(st, num, pd, ln, **F.recipe(name, ln))
    lab = F.label_feature_names(ln)
    assert X.shape == (len(num), len(names)) and X.dtype == np.float32
    assert np.array_equal(ids, np.sort(num["Id"].to_numpy())) and y.sum() > 20
    assert names[-len(lab):] == lab
    assert np.isnan(X[:, -len(lab):]).all()                        # filled per fold, never at build time
    if name == "l3-entry":
        assert not any(n.startswith("L3_") for n in names)         # nothing from line 3 before entering it
    if name != "kaggle":
        assert "sig_rank" not in names
        assert ds.nbr["ok"].mean() < F.build(st, num, pd, ln, **F.recipe("kaggle", ln)).nbr["ok"].mean()


def test_causal_masks_follow_the_decision_time(frames):
    """Brute force on the raw times: S_next_id is present exactly when the next starter started by D."""
    st, num, ln = frames
    X, y, ids, names = F.build(st, num, pd, ln, **F.recipe("causal", ln))
    P = F.part_table(st.reset_index(drop=True), F.stations(ln, st.columns), pd, ln)
    P = P.sort_values(["start", "Id"]).reset_index(drop=True)
    nxt_start = P["start"].shift(-1)
    expect = P.assign(has=(nxt_start <= P["decide"]))[["Id", "has", "is_train"]]
    expect = expect[expect["is_train"] == 1].set_index("Id")["has"]
    got = pd.Series(~np.isnan(X[:, names.index("S_next_id")]), index=ids)
    assert (got.loc[expect.index] == expect).all()


def test_label_features_hide_the_scored_fold(frames):
    st, num, ln = frames
    X, y, ids, names = ds = F.build(st, num, pd, ln, **F.recipe("kaggle", ln))
    fold = train.folds(y, 5, 0)
    known = fold != 0
    a = F.label_features(ds.nbr, y, known)
    y2 = y.copy()
    rng = np.random.default_rng(0)
    y2[fold == 0] = rng.permutation(y2[fold == 0])                  # scramble the scored fold's labels
    b = F.label_features(ds.nbr, y2, known)
    np.testing.assert_array_equal(a, b)
    assert np.nanmax(a[:, F.label_feature_names(ln).index("nbr_ysum")]) >= 1


def test_mcc_curve_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.1
    p = np.round(rng.random(500) * 0.5 + 0.4 * y, 2)                 # ties on purpose
    t, m = mcc.curve(y, p)
    for th, mm in zip(t[::7], m[::7]):
        assert mm == pytest.approx(mcc.at(y, p, th), abs=1e-12)
    best, th = mcc.best(y, p)
    assert best == pytest.approx(max(mcc.at(y, p, x) for x in np.unique(p)), abs=1e-12)
    assert mcc.at(y, y.astype(float), 0.5) == pytest.approx(1.0)


def test_cv_finds_signal_on_the_synthetic_line(frames):
    st, num, ln = frames
    ds = F.build(st, num, pd, ln, **F.recipe("kaggle", ln))
    X, y, ids, names = ds
    s, oof, fold = train.cv(X, y, k=3, device="cpu", rounds=60, names=names, nbr=ds.nbr, log=lambda *_: None,
                            params={"eta": 0.1, "max_depth": 4})
    assert s["auc"] > 0.7 and s["mcc_best"] > 0.1
    assert s["mcc_honest"] <= s["mcc_best"] + 1e-9


@pytest.mark.parametrize("name", ["kaggle", "causal"])
def test_label_features_never_depend_on_the_parts_own_label(frames, name):
    """Leave-one-out group sums leak (sum = S - y_self); flipping a part's own label must change none of its own
    label features, in any fold."""
    st, num, ln = frames
    ds = F.build(st, num, pd, ln, **F.recipe(name, ln))
    X, y, ids, names = ds
    known = train.folds(y, 5, 0) != 1
    a = F.label_features(ds.nbr, y, known)
    rng = np.random.default_rng(1)
    for i in rng.choice(np.where(known)[0], 25, replace=False):
        y2 = y.copy()
        y2[i] = 1 - y2[i]
        b = F.label_features(ds.nbr, y2, known)
        np.testing.assert_array_equal(a[i], b[i])
