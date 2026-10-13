"""Shared fixtures: a cache built from a simulated history of a line spec in lines/, through that line's adapter."""
import pytest

from lineloop import line as L
from lineloop.data import bosch, cache, events, simulate


def make_cache(tmp, spec, parts, seed):
    """Simulate `parts` parts of the spec's line, ingest them with its own adapter; (stations, numeric, line)."""
    ln = L.read(spec)
    src, out = tmp / "src", tmp / "cache"
    simulate.write(ln, src, parts=parts, seed=seed, log=lambda *_: None)
    if ln.data["format"] == "bosch":
        bosch.ingest(ln, src, out, log=lambda *_: None)
    else:
        events.ingest(ln, src, out, log=lambda *_: None)
    return out


@pytest.fixture(scope="session")
def bosch_like(tmp_path_factory):
    """The Bosch-shaped line, written as Kaggle CSVs and read back by the Bosch adapter."""
    return make_cache(tmp_path_factory.mktemp("bosch_like"), "bosch-synthetic", 6000, 3)


@pytest.fixture(scope="session")
def battery(tmp_path_factory):
    """A line that shares nothing with Bosch: events tables, string serials, a minute clock, 12-hour shifts."""
    return make_cache(tmp_path_factory.mktemp("battery"), "battery-module", 5000, 1)


@pytest.fixture(scope="session")
def bosch_frames(bosch_like):
    import pandas as pd
    return cache.load(bosch_like, pd)


@pytest.fixture(scope="session")
def battery_frames(battery):
    import pandas as pd
    return cache.load(battery, pd)
