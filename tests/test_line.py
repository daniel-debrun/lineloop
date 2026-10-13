"""lineloop.line and the generic data path: specs, the events adapter, and a whole line that is not Bosch."""
import numpy as np
import pandas as pd
import pytest

from lineloop import line as L
from lineloop.data import cache, events
from lineloop.quality import features as F

SPEC = """
name = "tiny"
[[segment]]
name = "a"
stations = ["s1", "s2", "s3"]
[[segment]]
name = "b"
stations = ["s4", "s5"]
[checkpoints]
mid = "s4"
[time]
unit_hours = 1.0
resolution = 0.25
"""


def tiny(**over):
    d = L.tomllib.loads(SPEC)
    d.update(over)
    return L.from_dict(d)


def test_spec_reads_and_orders_stations():
    ln = tiny()
    assert ln.stations == ["s1", "s2", "s3", "s4", "s5"]
    assert ln.before("mid") == ["s1", "s2", "s3"] and ln.after("mid") == ["s4", "s5"]
    assert ln.before(None) == ln.stations and ln.segment_of("s5") == "b"
    assert list(F.recipes(ln)) == ["offline", "live", "mid"]
    assert F.recipe("kaggle", ln) == {"checkpoint": None, "causal": False, "numeric": True}
    assert ln.family_blocks("mid") == {"a": ["s1", "s2", "s3"]}          # blocks after the checkpoint dropped
    assert ln.day == 24 and ln.week == 168 and ln.ticks_per_unit == 4 and ln.n_shifts == 21


@pytest.mark.parametrize("over,msg", [
    ({"checkpoints": {"mid": "s9"}}, "no segment lists"),
    ({"checkpoints": {"live": "s4"}}, "reserved"),
    ({"shifts": {"hours": 10}}, "divide a week"),
    ({"families": {"x": ["s1"]}}, "first station, last station"),
    ({"colour": "red"}, "unknown keys"),
])
def test_spec_errors_are_specific(over, msg):
    with pytest.raises(ValueError, match=msg):
        tiny(**over)


def test_events_ingest_handles_datetimes_strings_and_unlabelled_parts(tmp_path):
    ln = tiny()
    t = pd.Timestamp("2026-03-02 06:00")                                  # a Monday morning
    visits = pd.DataFrame({"part": ["p-b", "p-b", "p-a", "p-a", "p-a", "p-c"],
                           "station": ["s1", "s4", "s1", "s2", "s2", "s1"],
                           "time": [t, t + pd.Timedelta("2h"), t - pd.Timedelta("1h"), t, t + pd.Timedelta("7min"),
                                    t + pd.Timedelta("3h")]})
    results = pd.DataFrame({"part": ["p-a", "p-b"], "failed": [1, 0]})
    meas = pd.DataFrame({"part": ["p-a", "p-a", "p-b"], "station": ["s2", "s1", "s1"], "name": ["w", "w", "w"],
                         "value": [1.5, 0.5, 0.7]})
    events.write({"visits": visits, "results": results, "measurements": meas}, tmp_path / "src", fmt="csv")
    events.ingest(ln, tmp_path / "src", tmp_path / "cache", log=lambda *_: None)
    st, num, got = cache.load(tmp_path / "cache", pd)
    assert got.unit_hours == pytest.approx(1 / 60)                        # date-times -> minutes since a Monday
    assert list(st["Id"]) == [1, 2, 3]                                    # p-a first seen, then p-b, then p-c
    assert list(st["is_train"]) == [1, 1, 0]
    a = st.iloc[0]
    assert a["s2_t1"] - a["s2_t0"] == pytest.approx(7)                    # first and last visit, in minutes
    assert (a["s1_t0"] / 60) % 24 == pytest.approx(5)                     # 05:00: time of day survives
    assert list(num["Response"]) == [1, 0]
    assert got.measurements == {"s1.w": "s1", "s2.w": "s2"}               # flow order, then name
    assert np.isnan(num.loc[1, "s2.w"]) and num.loc[0, "s2.w"] == 1.5


def test_integer_part_numbers_keep_their_order_and_become_contiguous():
    first = pd.Series([3.0, 1.0, 2.0], index=[30, 10, 20])
    ids, renumbered = events._ids(pd.Series([30, 10, 20, 10]), first)
    assert renumbered and ids.loc[10] == 1 and ids.loc[20] == 2 and ids.loc[30] == 3
    ids, renumbered = events._ids(pd.Series([5, 6, 7]), pd.Series([0.0, 1, 2], index=[5, 6, 7]))
    assert not renumbered and ids.loc[6] == 6


def test_battery_line_end_to_end(battery_frames):
    """A line with nothing in common with Bosch builds every recipe, and the checkpoint model sees no station at or
    after its checkpoint."""
    st, num, ln = battery_frames
    assert ln.unit_hours == pytest.approx(1 / 60) and ln.n_shifts == 14
    assert set(ln.measurements.values()) <= set(ln.stations)
    for name in F.recipes(ln):
        X, y, ids, names = F.build(st, num, pd, ln, **F.recipe(name, ln))
        assert X.shape[0] == len(num) and 0 < y.mean() < 0.2
        if name == "pre-weld":
            later = set(ln.after("pre-weld"))
            assert not any(n.split("_")[0] in later or n.split(".")[0] in later for n in names if "_grp_" not in n)
            assert "cell-prep_start" in names and "welding_start" not in names


def test_resolve_rejects_stations_the_spec_does_not_list(tmp_path):
    ln = tiny()
    visits = pd.DataFrame({"part": [1, 1], "station": ["s1", "zz"], "time": [0.0, 1.0]})
    events.write({"visits": visits, "results": pd.DataFrame({"part": [1], "failed": [0]})}, tmp_path / "src")
    with pytest.raises(ValueError, match="zz"):
        events.ingest(ln, tmp_path / "src", tmp_path / "cache", log=lambda *_: None)


@pytest.mark.parametrize("fixture", ["battery", "bosch_like"])
def test_simulated_failures_carry_their_origin_through_ingest(fixture, request):
    """Both adapters keep each failure's origin station, and on both lines some defects arise only after the first
    checkpoint, which an inspection there cannot find."""
    from lineloop.data import cache
    st, num, ln = cache.load(request.getfixturevalue(fixture), pd)
    org = cache.origins(request.getfixturevalue(fixture))
    failed = set(num.loc[num["Response"] == 1, "Id"])
    assert org and set(org) == failed
    cut = ln.position(ln.checkpoint_station({"battery": "pre-weld", "bosch_like": "l3-entry"}[fixture]))
    later = np.mean([ln.position(o) >= cut for o in org.values()])
    assert 0.05 < later < 0.8


def test_simulated_fail_rate_is_the_expected_rate():
    from lineloop.data import simulate
    ln = L.read("battery-module")
    r = simulate.generate(ln, parts=4000, seed=2)["results"]
    assert r["failed"].mean() == pytest.approx(ln.simulate["fail_rate"], abs=0.006)
    assert r.loc[r["failed"] == 1, "origin"].notna().all() and r.loc[r["failed"] == 0, "origin"].isna().all()
