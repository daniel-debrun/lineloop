"""From the edge node's scored parts to the twin's input: route families, work after the checkpoint, hours.

  family      which blocks of the line (line.family_blocks: the spec's [families], or its segments) a part
              passed before the checkpoint, e.g. "L0a+L2"; the k most common patterns become families 0..k-1,
              the rest family k. Routes differ in failure rate, so the optimiser may set a threshold for each.
  saved_work  how many stations at or after the checkpoint the part really went on to visit: the work saved if it
              is removed at the checkpoint.
  t_h         decision time in hours (data units x line.unit_hours).
  present     1 if a failing part's defect already exists at the checkpoint: its origin (lineloop.data.cache
              origins) is a station before it. Without origins in the data, every failure counts as present,
              and the inspection cell's `detect` share has to cover defects made later.
"""
import numpy as np
import pandas as pd


def route_blocks(st, line, checkpoint):
    """Per part (st order): the block pattern before the checkpoint, e.g. 'L0a+L2' (or 'none')."""
    have = {c[:-3] for c in st.columns if c.endswith("_t0")}
    parts = []
    for name, sts in line.family_blocks(checkpoint).items():
        cols = [f"{s}_t0" for s in sts if s in have]
        parts.append(np.where(st[cols].notna().any(axis=1).to_numpy(), name, "") if cols
                     else np.full(len(st), "", dtype=object))
    pat = ["+".join(x for x in row if x) or "none" for row in zip(*parts)] if parts else ["none"] * len(st)
    return np.array(pat)


def work_after(st, line, checkpoint):
    have = {c[:-3] for c in st.columns if c.endswith("_t0")}
    cols = [f"{s}_t0" for s in line.after(checkpoint, have)]
    return st[cols].notna().sum(axis=1).to_numpy().astype(np.float32)


def present_at(ids, y, line, checkpoint, origins):
    """Per part: 1 unless it failed with a known origin at or after the checkpoint's station."""
    out = np.ones(len(ids), np.int32)
    if origins is None:
        return out
    cut = line.position(line.checkpoint_station(checkpoint))
    for i, (pid, yy) in enumerate(zip(ids, y)):
        o = origins.get(pid) if yy == 1 else None
        if o is not None and line.position(o) >= cut:
            out[i] = 0
    return out


def prepare(scored, st, line, checkpoint, k=6, fit_on=None, origins=None):
    """Twin input from scored parts (frame: Id, t, p, y). Families are the k most common patterns among `fit_on`
    rows (default: all), so calibration and hold-out windows share one coding. origins: {Id: station} of failed
    parts (lineloop.data.cache.origins), or None."""
    s = st.set_index("Id")
    ids = scored["Id"].to_numpy()
    sub = s.loc[ids].reset_index()
    pat = route_blocks(sub, line, checkpoint)
    base = pat if fit_on is None else pat[fit_on]
    top = pd.Series(base).value_counts().index[:k].tolist()
    fam = np.array([top.index(p) if p in top else k for p in pat], np.int32)
    o = np.argsort(scored["t"].to_numpy(), kind="stable")
    pres = present_at(ids, scored["y"].to_numpy(), line, checkpoint, origins)
    return {"present": pres[o], "Id": ids[o], "t_h": (scored["t"].to_numpy()[o] * line.unit_hours).astype(np.float32),
            "score": scored["p"].to_numpy()[o].astype(np.float32), "y": scored["y"].to_numpy()[o].astype(np.int32),
            "fam": fam[o], "saved_work": work_after(sub, line, checkpoint)[o]}, top + ["other"]


def window(d, t_from_h, t_to_h):
    m = (d["t_h"] >= t_from_h) & (d["t_h"] < t_to_h)
    return {k: v[m] for k, v in d.items()}
