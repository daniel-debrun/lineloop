"""lineloop.edge.stream: the live node reproduces the batch causal matrix exactly, from events alone, on any line."""
import numpy as np
import pandas as pd
import pytest

from lineloop.edge.stream import Stream
from lineloop.quality import features as F

CASES = [("bosch_frames", "live"), ("bosch_frames", "l3-entry"), ("battery_frames", "live"),
         ("battery_frames", "pre-weld"), ("battery_frames", "pre-test"), ("battery_frames", "post-weld")]


@pytest.mark.parametrize("frames,name", CASES)
def test_stream_matches_batch_causal_features(frames, name, request):
    st, num, ln = request.getfixturevalue(frames)
    ds = F.build(st, num, pd, ln, **F.recipe(name, ln))
    X, y, ids, names = ds
    X = X.copy()
    n_lab = len(F.label_feature_names(ln))
    X[:, -n_lab:] = F.label_features(ds.nbr, y, None)                    # live: every result that exists
    s = Stream(st, num, names, ln, F.recipe(name, ln)["checkpoint"])
    sid, t, M = s.replay()
    assert len(sid) > 0.9 * len(ids)                                     # every labelled part with dates
    row = {i: k for k, i in enumerate(ids)}
    B = X[[row[i] for i in sid]]
    bad = ~((M == B) | (np.isnan(M) & np.isnan(B)))
    cols = sorted({names[j] for j in np.where(bad.any(axis=0))[0]})
    assert not bad.any(), f"stream differs from batch in {cols}"


def test_offline_features_are_not_live(bosch_frames):
    st, num, ln = bosch_frames
    with pytest.raises(ValueError):
        Stream(st, num, ["x"], ln, None, causal=False)


@pytest.mark.parametrize("frames", ["bosch_frames", "battery_frames"])
def test_warm_start_gives_the_same_rows(frames, request):
    st, num, ln = request.getfixturevalue(frames)
    ds = F.build(st, num, pd, ln, **F.recipe("live", ln))
    X, y, ids, names = ds
    full = Stream(st, num, names, ln, None)
    fid, ft, FM = full.replay()
    t0 = float(np.quantile(ft, 0.6))
    warm = Stream(st, num, names, ln, None, t_from=t0)
    wid, wt, WM = warm.replay()
    assert len(wid) and wt.min() >= t0
    keep = ft >= t0
    np.testing.assert_array_equal(fid[keep], wid)
    np.testing.assert_array_equal(np.isnan(FM[keep]), np.isnan(WM))
    np.testing.assert_array_equal(np.nan_to_num(FM[keep]), np.nan_to_num(WM))


def test_rework_after_the_decision_is_cut_in_batch_and_live(battery_frames):
    """A module sent back to the glue station after reaching a welder: at pre-weld its glue visit counts only up to
    D (one time, no later last time) and its glue measurements, which may come from the second visit, are dropped;
    the live node agrees."""
    st, num, ln = battery_frames
    st = st.copy()
    P = F.part_table(st, F.stations(ln, st.columns, "busbar-weld-1"), pd, ln, "busbar-weld-1")
    lab = set(num["Id"])
    redo = P.loc[P["Id"].isin(lab) & st["glue-dispense_t0"].notna() & P["decide"].notna(), "Id"].to_numpy()[::50]
    rows = st["Id"].isin(redo)
    st.loc[rows, "glue-dispense_t1"] = P.loc[rows, "decide"] + 30.0                  # back at glue, 30 min later
    ds = F.build(st, num, pd, ln, **F.recipe("pre-weld", ln))
    X, y, ids, names = ds
    r = np.isin(ids, redo)
    assert r.sum() > 5
    assert np.all(X[r, names.index("glue-dispense_span")] == 0)
    glue = [j for j, n in enumerate(names) if n.startswith("glue-dispense.")]
    assert glue and np.isnan(X[np.ix_(r, glue)]).all() and not np.isnan(X[np.ix_(~r, glue)]).all()
    X = X.copy()
    X[:, -len(F.label_feature_names(ln)):] = F.label_features(ds.nbr, y, None)
    sid, _, M = Stream(st, num, names, ln, "busbar-weld-1").replay()
    B = X[[{i: k for k, i in enumerate(ids)}[i] for i in sid]]
    assert ((M == B) | (np.isnan(M) & np.isnan(B))).all()
