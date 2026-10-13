"""The ingested form of a line's history: two parquet files and the resolved line, read by every later step.

  stations.parquet  one row per part (labelled or not): Id, is_train (1 = its final result is known), then per
                    station <station>_t0 / _t1 (first / last time the part was there, missing if never),
                    date_hash (hash of the part's raw timestamps), date_nnz (how many it has)
  numeric.parquet   one row per labelled part, in Id order: Id, Response (1 = failed the final test),
                    the measurement columns, num_hash (hash of the part's measurements), num_nnz
  origins.parquet   optional, failed parts only: Id, origin (the station where the defect arose, as failure
                    analysis records it). Without it, every failure counts as present at every checkpoint.
  line.json         the line spec resolved against this data (lineloop.line)

Every adapter (lineloop.data.events, lineloop.data.bosch) writes exactly this, so the features, the edge node,
the twin and the optimiser never see a source format.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .. import line as L

STATIONS, NUMERIC, ORIGINS = "stations.parquet", "numeric.parquet", "origins.parquet"


def row_hash(df):
    """uint64 hash per row over all columns; equal rows (NaN included) hash equal."""
    return pd.util.hash_pandas_object(df, index=False).to_numpy()


def resolve(ln, cache, log=print):
    """Fill in what the data decides (stations seen, measurement columns, group stations), check it against the
    spec, and write line.json next to the parquet files."""
    import pyarrow.parquet as pq
    cache = Path(cache)
    cols = pq.read_schema(cache / STATIONS).names
    seen = [c[:-3] for c in cols if c.endswith("_t0")]
    if ln.segments:
        unknown = [s for s in seen if s not in set(ln.stations)]
        if unknown:
            raise ValueError(f"stations in the data but not in the spec: {unknown[:10]}")
    else:
        raise ValueError("the line has no segments; the format adapter must derive them before resolve")
    missing = [s for s in ln.stations if s not in set(seen)]
    if missing:
        log(f"note: {len(missing)} station(s) in the spec never appear in the data: {missing[:8]}")
    if ln.group_stations is None:
        st = pq.read_table(cache / STATIONS, columns=[f"{s}_t0" for s in seen])
        n = max(st.num_rows, 1)
        share = {s: float(np.mean(~np.isnan(st[f"{s}_t0"].to_numpy(zero_copy_only=False).astype(np.float64))))
                 for s in seen}
        ln.group_stations = [s for s in ln.stations if share.get(s, 0.0) >= ln.group_min_share]
        log(f"group stations: {len(ln.group_stations)} visited by >= {ln.group_min_share:.0%} of {n:,} parts")
    for name, s in ln.checkpoints.items():
        if s not in ln.stations:
            raise ValueError(f"checkpoint {name!r}: station {s!r} is not on the line")
    L.check(ln)
    ln.save(cache)
    return ln


def load(cache, xp=pd, numeric_cols=None):
    """(stations, numeric, line) from a cache, as cudf or pandas frames. numeric_cols=[] skips the numeric file."""
    cache = Path(cache)
    if not (cache / STATIONS).exists():
        raise FileNotFoundError(f"no ingested data in {cache}; run `python -m lineloop line ingest --line <spec>`")
    ln = L.load(cache)
    st = xp.read_parquet(cache / STATIONS)
    if numeric_cols == []:
        return st, None, ln
    cols = None if numeric_cols is None else ["Id", "Response", "num_hash", "num_nnz", *numeric_cols]
    return st, xp.read_parquet(cache / NUMERIC, columns=cols), ln


def write_origins(out, ids, origin):
    """origins.parquet from failed parts' Ids and origin stations (missing origins are dropped)."""
    o = pd.DataFrame({"Id": np.asarray(ids, np.int64), "origin": pd.Series(origin, dtype=object).to_numpy()})
    o = o[o["origin"].notna() & (o["origin"].astype(str) != "")].astype({"origin": str})
    if len(o):
        o.sort_values("Id").to_parquet(Path(out) / ORIGINS, index=False)
    return len(o)


def origins(cache):
    """{Id: origin station} for failed parts with a known origin; None if the line's data has none."""
    p = Path(cache) / ORIGINS
    if not p.exists():
        return None
    o = pd.read_parquet(p)
    return dict(zip(o["Id"].to_numpy(), o["origin"].to_numpy()))


def station_columns(cache):
    import pyarrow.parquet as pq
    return pq.read_schema(Path(cache) / STATIONS).names
