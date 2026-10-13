"""Generic line data: three plain tables any MES, historian or test-stand export can produce.

  visits         part, station, time            one row per timestamp a part logged at a station (several per
                                                station are fine: the first and last are kept)
  results        part, failed [, origin]        the final test result, 1 = failed; parts without a row are on the
                                                line but unlabelled (not yet tested, or the result was not kept).
                                                Optional `origin`: the station where a failed part's defect arose
                                                (failure analysis), so an inspection is only credited with defects
                                                that already exist where it stands
  measurements   part, station, name, value     optional; one row per value a station measured on a part

Each is a .parquet or .csv file in the line's data folder (names and column names can be changed in the spec's
[data] table). `time` is a number in the spec's time unit, or a date-time, which is converted to minutes since
Monday 1970-01-05 (the line is then resolved with unit_hours = 1/60, so time-of-day and day-of-week features keep
their meaning). Part identifiers may be integers (their order is kept: production order, ideally) or strings
(numbered in order of first appearance).

ingest() turns them into the cache every later step reads (lineloop.data.cache).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache as C

TABLES = {"visits": ("part", "station", "time"), "results": ("part", "failed"),
          "measurements": ("part", "station", "name", "value")}
OPTIONAL = {"results": ("origin",)}
EPOCH_MONDAY = pd.Timestamp("1970-01-05")


def _file(src, name, required=True):
    for ext in (".parquet", ".csv", ".csv.gz", ""):
        p = Path(src) / f"{name}{ext}"
        if p.is_file():
            return p
    if required:
        raise FileNotFoundError(f"no {name}.parquet or {name}.csv in {src}")
    return None


def _read(src, ln, table, required=True):
    name = ln.data.get(table, table)
    p = _file(src, name, required)
    if p is None:
        return None
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    ren = {v: k for k, v in ln.data.get("columns", {}).items() if v in df.columns}
    df = df.rename(columns=ren)
    need = TABLES[table]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{p.name}: missing column(s) {miss}; has {list(df.columns)[:10]} (rename them in the "
                         f"spec's [data.columns] table, e.g. part = \"serial\")")
    return df[list(need) + [c for c in OPTIONAL.get(table, ()) if c in df.columns]]


def _clock(t, ln, log):
    """Times as float64 in data units; date-times become minutes since a Monday."""
    if pd.api.types.is_datetime64_any_dtype(t) or not pd.api.types.is_numeric_dtype(t):
        dt = pd.to_datetime(t, utc=False)
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert(None)
        if ln.unit_hours != 1 / 60:
            log("time column holds date-times: using minutes since Monday 1970-01-05 (unit_hours = 1/60)")
        ln.unit_hours = 1 / 60
        return ((dt - EPOCH_MONDAY) / pd.Timedelta(minutes=1)).to_numpy(np.float64)
    return t.to_numpy(np.float64)


def _ids(parts, first_t):
    """Integer Id per distinct part, contiguous (the live node finds a part's numbering neighbours as Id - 1 and
    Id + 1). Integer part numbers keep their order: as they are if already contiguous, else renumbered 1..n by
    rank. Other identifiers are numbered 1..n in order of first appearance on the line."""
    u = pd.Index(parts.unique())
    if pd.api.types.is_integer_dtype(u.dtype):
        v = np.sort(u.to_numpy(np.int64))
        if len(v) and v[0] >= 0 and v[-1] - v[0] + 1 == len(v):
            return pd.Series(u.to_numpy(np.int64), index=u), False
        return pd.Series(np.arange(1, len(v) + 1, dtype=np.int64), index=pd.Index(v)), True
    order = first_t.loc[u].sort_values(kind="stable").index
    return pd.Series(np.arange(1, len(order) + 1, dtype=np.int64), index=order), True


def ingest(ln, src, out, limit=None, log=print):
    """visits / results / measurements in `src` -> stations.parquet, numeric.parquet, ids.parquet, line.json."""
    t_all = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    v = _read(src, ln, "visits")
    r = _read(src, ln, "results")
    m = _read(src, ln, "measurements", required=False)
    if limit:
        keep = v["part"].drop_duplicates().iloc[:limit]
        v, r = v[v["part"].isin(keep)], r[r["part"].isin(keep)]
        m = m[m["part"].isin(keep)] if m is not None else None
    v = v.assign(time=_clock(v["time"], ln, log)).dropna(subset=["time"])
    unknown = sorted(set(v["station"].astype(str).unique()) - set(ln.stations))
    if unknown:
        raise ValueError(f"visits name stations the spec does not list: {unknown[:10]}")
    v["station"] = v["station"].astype(str)
    q = ln.resolution
    v["time"] = np.round(v["time"].to_numpy() / q) * q                  # on the clock's grid
    first_t = v.groupby("part")["time"].min()
    idmap, renumbered = _ids(v["part"], first_t)
    v["Id"] = idmap.loc[v["part"]].to_numpy()

    # stations: first / last time per (part, station), wide, in flow order
    g = v.groupby(["Id", "station"])["time"].agg(["min", "max"])
    seen = [s for s in ln.stations if s in set(g.index.get_level_values(1))]
    t0 = g["min"].unstack("station").reindex(columns=seen)
    t1 = g["max"].unstack("station").reindex(columns=seen)
    st = pd.DataFrame({"Id": t0.index.to_numpy(np.int64)})
    lab = r.drop_duplicates("part", keep="last")
    lab = lab[lab["part"].isin(idmap.index)]
    if len(lab) < len(r.drop_duplicates("part")):
        log(f"note: {len(r.drop_duplicates('part')) - len(lab):,} result(s) for parts with no visits were dropped")
    y = pd.Series(lab["failed"].astype(int).clip(0, 1).to_numpy(np.int8), index=idmap.loc[lab["part"]].to_numpy())
    st["is_train"] = st["Id"].isin(y.index).astype(np.int8).to_numpy()
    n_origin = 0
    if "origin" in lab.columns:
        o = lab[lab["failed"].astype(int) == 1]
        bad = sorted(set(o["origin"].dropna().astype(str)) - set(ln.stations) - {""})
        if bad:
            raise ValueError(f"results name origin stations the spec does not list: {bad[:10]}")
        n_origin = C.write_origins(out, idmap.loc[o["part"]].to_numpy(), o["origin"].to_numpy())
    cols = {}
    for s in seen:
        cols[f"{s}_t0"] = t0[s].to_numpy(np.float64)
        cols[f"{s}_t1"] = t1[s].to_numpy(np.float64)
    st = pd.concat([st, pd.DataFrame(cols)], axis=1)
    st["date_hash"] = C.row_hash(st[list(cols)]).view(np.int64)
    st["date_nnz"] = v.groupby("Id").size().reindex(st["Id"]).to_numpy().astype(np.int16)
    st.to_parquet(out / C.STATIONS, index=False, compression="zstd")

    # numeric: labelled parts in Id order, one column per (station, measurement)
    num = pd.DataFrame({"Id": np.sort(y.index.to_numpy())})
    num["Response"] = y.loc[num["Id"]].to_numpy(np.int8)
    ln.measurements = {}
    if m is not None and len(m):
        m = m[m["part"].isin(idmap.index)].copy()
        m["station"] = m["station"].astype(str)
        bad = sorted(set(m["station"].unique()) - set(ln.stations))
        if bad:
            raise ValueError(f"measurements name stations the spec does not list: {bad[:10]}")
        m["col"] = m["station"] + "." + m["name"].astype(str)
        m["Id"] = idmap.loc[m["part"]].to_numpy()
        wide = m.pivot_table(index="Id", columns="col", values="value", aggfunc="last")
        pos = {s: i for i, s in enumerate(ln.stations)}
        meta = m.drop_duplicates("col").set_index("col")["station"]
        order = sorted(wide.columns, key=lambda c: (pos[meta[c]], c))
        wide = wide.reindex(index=num["Id"], columns=order).astype(np.float32)
        num = pd.concat([num, wide.reset_index(drop=True)], axis=1)
        ln.measurements = {c: meta[c] for c in order}
    feats = list(ln.measurements)
    num["num_hash"] = C.row_hash(num[feats]).view(np.int64) if feats else np.zeros(len(num), np.int64)
    num["num_nnz"] = (num[feats].notna().sum(axis=1).astype(np.int16) if feats else np.zeros(len(num), np.int16))
    num.to_parquet(out / C.NUMERIC, index=False, compression="zstd")
    if renumbered:
        pd.DataFrame({"Id": idmap.to_numpy(), "part": idmap.index.astype(str)}).to_parquet(out / "ids.parquet",
                                                                                            index=False)
    C.resolve(ln, out, log)
    meta = {"source": str(src), "format": "events", "parts": int(len(st)), "labelled": int(len(num)),
            "failures": int(num["Response"].sum()), "stations": len(seen), "measurements": len(feats),
            "visits": int(len(v)), "failure_origins": n_origin, "renumbered_ids": renumbered,
            "seconds": round(time.perf_counter() - t_all, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"{ln.name}: {meta['parts']:,} parts ({meta['labelled']:,} labelled, {meta['failures']:,} failed), "
        f"{meta['stations']} stations, {meta['measurements']} measurements, {meta['visits']:,} visits "
        f"in {meta['seconds']} s")
    return meta


def write(tables, out, fmt="parquet"):
    """Write visits / results / measurements tables (DataFrames) to `out`, as the events format expects."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = {}
    for name, df in tables.items():
        if df is None:
            continue
        if fmt == "parquet":
            df.to_parquet(out / f"{name}.parquet", index=False, compression="zstd")
        else:
            df.to_csv(out / f"{name}.csv", index=False)
        n[name] = len(df)
    return n
