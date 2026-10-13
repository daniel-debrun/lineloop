"""Bosch Production Line Performance (Kaggle, 2016): the "bosch" data format, streamed once into the cache.

The data is used under the competition's rules: it stays on the local machine (data/ is git-ignored) and nothing
at per-part granularity is committed. Only code and aggregate scores are.

Source files (Kaggle `bosch-production-line-performance`), zipped or extracted:
  train_numeric.csv   1,183,747 parts x 968 measurements + Response (1 = failed final QC, 0.58 %)
  train_date.csv      the same parts x 1,156 timestamps (unit 0.01 ~ 6 min, inferred)
  test_date.csv       1,183,748 unlabelled parts x 1,156 timestamps; interleaved in time with train, so the
                      line's order (which part followed which) needs both

Column names carry the layout: L{line}_S{station}_F{n} is a measurement and L{line}_S{station}_D{n} a timestamp,
and station numbers increase along the line. So a "bosch" spec need not list its stations: they are read from the
header, in station-number order, one segment per line (L0..L3), unless the spec lists segments itself.

Outputs in `out` (the cache layout of lineloop.data.cache):
  numeric.parquet   Id, Response, 968 float32 measurements, num_hash (row hash), num_nnz
  stations.parquet  Id, is_train, then per station L{l}_S{s}: _t0 / _t1 (first / last timestamp at that
                    station), date_hash (hash of all 1,156 timestamps), date_nnz. 98 % of station visits are a
                    single instant, so t0/t1 keep almost everything the 1,156 columns carry.
  origins.parquet   only if the source folder has origins.csv (Id, origin): a simulated line's defect origins.
                    The Kaggle data has none.
  line.json         the spec resolved against the header
  meta.json         row and column counts, station list, source sizes, seconds per step
"""
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from . import cache as C

FILES = ("train_numeric", "train_date", "test_date")
STATION = re.compile(r"^(L(\d+)_S(\d+))_[FD]\d+$")


def station_of(col):
    """'L3_S32_F3850' -> ('L3_S32', 3, 32); None for Id / Response."""
    m = STATION.match(col)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None


def stations_in(cols):
    """Station names in line order (station numbers are globally increasing along the line)."""
    seen = {}
    for c in cols:
        s = station_of(c)
        if s:
            seen[s[0]] = s[2]
    return sorted(seen, key=seen.get)


def _source(src, name):
    """An open binary stream of <name>.csv, from the zip if present."""
    src = Path(src)
    if (src / f"{name}.csv").exists():
        return open(src / f"{name}.csv", "rb"), (src / f"{name}.csv").stat().st_size
    z = src / f"{name}.csv.zip"
    if not z.exists():
        raise FileNotFoundError(f"{name}.csv(.zip) not in {src}; get it with "
                                f"`kaggle competitions download -c bosch-production-line-performance`")
    zf = zipfile.ZipFile(z)
    return zf.open(f"{name}.csv"), z.stat().st_size


def _header(src, name):
    f, _ = _source(src, name)
    with f:
        return f.readline().decode().strip().split(",")


def _reader(src, name, cols, block_mb=64):
    types = {c: pa.float32() for c in cols}
    types["Id"] = pa.int32()
    if "Response" in cols:
        types["Response"] = pa.int8()
    f, size = _source(src, name)
    r = pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=block_mb << 20),
                       convert_options=pacsv.ConvertOptions(column_types=types))
    return f, size, r


def _numeric(src, out, limit=None):
    cols = _header(src, "train_numeric")
    feats = [c for c in cols if c not in ("Id", "Response")]
    f, size, r = _reader(src, "train_numeric", cols)
    writer, n = None, 0
    with f:
        for batch in r:
            df = batch.to_pandas()
            x = df[feats]
            df["num_hash"] = C.row_hash(x).view(np.int64)
            df["num_nnz"] = x.notna().sum(axis=1).astype(np.int16)
            t = pa.Table.from_pandas(df, preserve_index=False)
            writer = writer or pq.ParquetWriter(out / "numeric.parquet", t.schema, compression="zstd")
            writer.write_table(t)
            n += len(df)
            if limit and n >= limit:
                break
    writer.close()
    return {"rows": n, "features": len(feats), "source_bytes": size}


def _dates(src, out, limit=None):
    cols = _header(src, "train_date")
    sts = stations_in(cols)
    idx = {s: [c for c in cols if (m := station_of(c)) and m[0] == s] for s in sts}
    writer, n, sizes = None, {}, {}
    for name, is_train in (("train_date", 1), ("test_date", 0)):
        if _header(src, name) != cols:
            raise ValueError(f"{name} columns differ from train_date")
        f, sizes[name], r = _reader(src, name, cols)
        n[name] = 0
        with f:
            for batch in r:
                df = batch.to_pandas()
                x = df[cols[1:]]
                o = {"Id": df["Id"].to_numpy(), "is_train": np.full(len(df), is_train, np.int8)}
                for s in sts:
                    v = df[idx[s]].to_numpy()
                    o[f"{s}_t0"] = np.fmin.reduce(v, axis=1)   # fmin skips NaN; an all-NaN row stays NaN
                    o[f"{s}_t1"] = np.fmax.reduce(v, axis=1)
                o["date_hash"] = C.row_hash(x).view(np.int64)
                o["date_nnz"] = x.notna().sum(axis=1).to_numpy().astype(np.int16)
                t = pa.table(o)
                writer = writer or pq.ParquetWriter(out / "stations.parquet", t.schema, compression="zstd")
                writer.write_table(t)
                n[name] += len(df)
                if limit and n[name] >= limit:
                    break
    writer.close()
    return {"rows": n, "stations": sts, "date_columns": len(cols) - 1, "source_bytes": sizes}


def segments_from_header(cols):
    """{L0: [L0_S0, ...], ...}: stations in station-number order, one segment per line."""
    segs = {}
    for s in stations_in(cols):
        segs.setdefault(s.split("_")[0], []).append(s)
    return segs


def ingest(ln, src, out, limit=None, log=print, keep_parquet=False):
    """Write numeric.parquet, stations.parquet, line.json and meta.json; `limit` stops each file early (tests,
    smoke runs). keep_parquet: the parquet files are already there, only resolve the line again."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not keep_parquet:
        meta = {"source": str(src), "format": "bosch", "limit": limit}
        t = time.perf_counter()
        meta["numeric"] = _numeric(src, out, limit)
        meta["numeric"]["seconds"] = round(time.perf_counter() - t, 1)
        log(f"numeric: {meta['numeric']['rows']:,} parts x {meta['numeric']['features']} in "
            f"{meta['numeric']['seconds']} s")
        t = time.perf_counter()
        meta["dates"] = _dates(src, out, limit)
        meta["dates"]["seconds"] = round(time.perf_counter() - t, 1)
        log(f"dates: {meta['dates']['rows']} parts, {len(meta['dates']['stations'])} stations "
            f"in {meta['dates']['seconds']} s")
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
    else:
        meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    if (Path(src) / "origins.csv").exists():
        import pandas as pd
        o = pd.read_csv(Path(src) / "origins.csv")
        C.write_origins(out, o["Id"], o["origin"])
    num_cols = pq.read_schema(out / C.NUMERIC).names
    st_cols = [c[:-3] + "_D0" for c in pq.read_schema(out / C.STATIONS).names if c.endswith("_t0")]
    if not ln.segments:
        ln.segments = segments_from_header(st_cols)
    ln.measurements = {c: station_of(c)[0] for c in num_cols if station_of(c)}
    C.resolve(ln, out, log)
    return meta


def write(tables, out):
    """Generic tables (lineloop.data.events) -> train_numeric / train_date / test_date .csv.zip, laid out like the
    Kaggle download (one date column per station; measurement names must already read F<n>). Used to run the
    Bosch adapter on a simulated line."""
    import io

    import pandas as pd
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    v, r, m = tables["visits"], tables["results"], tables.get("measurements")
    sts = list(dict.fromkeys(v["station"]))
    order = {s: station_of(f"{s}_D0")[2] for s in sts}
    sts = sorted(sts, key=order.get)
    dates = v.pivot_table(index="part", columns="station", values="time", aggfunc="min").reindex(columns=sts)
    dates.columns = [f"{s}_D{order[s] * 10 + 1}" for s in sts]
    lab = r.set_index("part")["failed"]
    if m is not None and len(m):
        m = m.assign(col=m["station"] + "_" + m["name"])
        num = m.pivot_table(index="part", columns="col", values="value", aggfunc="last")
        num = num.reindex(columns=sorted(num.columns, key=lambda c: (order[station_of(c)[0]], int(c.split("_F")[1]))))
    else:
        num = pd.DataFrame(index=lab.index)
    ids = dates.index
    train = ids[ids.isin(lab.index)]
    files = {"train_numeric": num.reindex(train).assign(Response=lab.loc[train].astype(int).to_numpy()),
             "train_date": dates.loc[train], "test_date": dates.loc[ids[~ids.isin(lab.index)]]}
    if "origin" in r.columns:
        o = r[r["failed"] == 1].dropna(subset=["origin"])
        pd.DataFrame({"Id": o["part"], "origin": o["origin"]}).to_csv(out / "origins.csv", index=False)
    for name, df in files.items():
        df = df.rename_axis("Id").reset_index()
        buf = io.StringIO()
        df.to_csv(buf, index=False, float_format="%.3f")
        with zipfile.ZipFile(out / f"{name}.csv.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{name}.csv", buf.getvalue().replace(".000,", ",").replace(".000\n", "\n"))
    return {k: len(v) for k, v in files.items()}
