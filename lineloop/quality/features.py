"""Features for predicting final-test failure of a part from its line telemetry, on cuDF or pandas.

One code path for both dataframe libraries: `xp` is the `cudf` or `pandas` module (lineloop.accel.frame_lib),
and only operations both implement the same way are used (`python -m lineloop quality parity` checks that the two
build the same matrix).

Everything about the line (its stations and segments, checkpoints, clock, group stations) comes from a
lineloop.line.Line; nothing here names a station.

Each part is scored at a decision time D:
  checkpoint None   D = the part's last timestamp (end of line)
  checkpoint c      D = when the part reaches checkpoint c's station or any later one in flow order; a part that
                    never does is scored after its last earlier station. Its own times, measurements and route
                    are cut at the checkpoint, and at D: a part sent back to an earlier station after D (rework)
                    keeps only that station's first time, and loses its measurements there, which may come from
                    the later visit.

  causal False      anything in the data. The strongest such features compare a part with the parts produced just
                    before *and after* it (Faron's "magic" order features).
  causal True       a neighbour feature is kept only if that neighbour's information existed at D (its start for
                    start-order features, its end for end-order ones, its route once it passed the checkpoint);
                    otherwise it is missing. Running counts include only parts already decided by D. This is
                    exactly what an edge node can compute live, and lineloop.edge.stream recomputes it online from
                    the event stream (tests check the two agree).

Times are in the line's data units; two times are "the same" when they fall on the same step of its clock
(line.resolution). Day and week features use line.day and line.week.

Recipes: "offline" (end of line, not causal: the competition setting), "live" (end of line, causal) and one per
checkpoint in the spec, named after it (causal, cut at the checkpoint).
"""
import numpy as np

ALIASES = {"kaggle": "offline", "causal": "live"}       # the Bosch study's names for the two end-of-line recipes


def recipes(line):
    """{name: (checkpoint station or None, causal, numeric)} for this line."""
    r = {"offline": (None, False, True), "live": (None, True, True)}
    for name, st in line.checkpoints.items():
        r[name] = (st, True, True)
    return r


def stations(line, st_cols, checkpoint=None, after=False):
    """Stations present in the stations frame, in flow order: before the checkpoint, or (after=True) at or after
    it. checkpoint: a station (or checkpoint name), None = the whole line."""
    present = {c[:-3] for c in st_cols if c.endswith("_t0")}
    if checkpoint is None:
        return [] if after else line.before(None, present)
    return line.after(checkpoint, present) if after else line.before(checkpoint, present)


def _hash_rows(df, xp):
    """int64 per row, equal rows equal. Values differ between cudf and pandas, so they are only grouped on."""
    if xp.__name__ == "cudf":
        # 64-bit (cuDF's default 32-bit murmur3 collides hundreds of times on 2.4M rows), and missing values filled
        # first: cuDF's xxhash64 returns null for any row with a null, which would put all such rows in one group
        return df.fillna(-1e30).hash_values(method="xxhash64").astype("int64")
    import pandas as pd
    return pd.Series(pd.util.hash_pandas_object(df, index=False).to_numpy().view(np.int64), index=df.index)


def part_table(st, sts, xp, line, checkpoint=None):
    """Per part (labelled or not): start, end, decision time, per-segment and per-station timing, route keys.
    At a checkpoint, `_late_<station>` columns (only where some part has one) flag a station the part was at again
    after D; its last time there becomes its first, and its measurements there must be dropped
    (cut_measurements)."""
    late = {}
    if checkpoint is not None:
        later = stations(line, st.columns, checkpoint, after=True)
        all_t1 = st[[f"{s}_t1" for s in sts]].max(axis=1)
        dec = st[[f"{s}_t0" for s in later]].min(axis=1).fillna(all_t1)
        st = st.copy()
        for s in sts:
            gone = st[f"{s}_t0"] > dec                    # first came here after D: not visited by D
            back = (st[f"{s}_t1"] > dec) & ~gone             # here before D, and again after it
            if bool(gone.any()) or bool(back.any()):
                st[f"{s}_t1"] = st[f"{s}_t1"].where(~back, st[f"{s}_t0"]).mask(gone)
                st[f"{s}_t0"] = st[f"{s}_t0"].mask(gone)
                late[f"_late_{s}"] = gone | back
    t0 = st[[f"{s}_t0" for s in sts]]
    t1 = st[[f"{s}_t1" for s in sts]]
    P = st[["Id", "is_train"]].copy()
    P["end"] = t1.max(axis=1)
    # when the part's final-QC result exists: its last timestamp anywhere on the line
    P["qc_time"] = st[[f"{s}_t1" for s in stations(line, st.columns)]].max(axis=1)
    if checkpoint is None:
        P["decide"] = P["end"]
        P["start"] = t0.min(axis=1)
    else:
        later = stations(line, st.columns, checkpoint, after=True)
        P["decide"] = st[[f"{s}_t0" for s in later]].min(axis=1).fillna(P["end"])
        P["start"] = t0.min(axis=1).fillna(P["decide"])
    P["dur"] = P["end"] - P["start"]
    P["n_st"] = t0.notna().sum(axis=1).astype("int16")
    for g in line.segments:
        cols = [s for s in sts if line.segment_of(s) == g]
        if not cols:
            continue
        a, b = st[[f"{s}_t0" for s in cols]], st[[f"{s}_t1" for s in cols]]
        P[f"{g}_start"] = a.min(axis=1)
        P[f"{g}_dur"] = b.max(axis=1) - P[f"{g}_start"]
        P[f"{g}_n"] = a.notna().sum(axis=1).astype("int16")
    P["tod"] = P["start"] % line.day
    P["tow"] = P["start"] % line.week
    per = {}
    for s in sts:
        per[f"{s}_t0"] = st[f"{s}_t0"]
        per[f"{s}_rel"] = st[f"{s}_t0"] - P["start"]
        per[f"{s}_span"] = st[f"{s}_t1"] - st[f"{s}_t0"]
    P = xp.concat([P, xp.DataFrame(per)], axis=1)
    P["_path"] = _hash_rows(t0.notna().astype("int8"), xp)
    # signature = identical timestamps everywhere. At the end of line the ingest's hash of all the part's raw
    # timestamps is used (finer than first/last per station); at a checkpoint only the stations before it count.
    P["_sig"] = st["date_hash"] if checkpoint is None else _hash_rows(xp.concat([t0, t1], axis=1).fillna(-1.0), xp)
    for k, v in late.items():
        P[k] = v
    return P


def late_columns(P):
    return [c for c in P.columns if c.startswith("_late_")]


def cut_measurements(num, P, line):
    """num with the measurements of stations a part revisited after D set missing (P: part_table rows of the same
    parts in the same order). Returns num itself when no part was reworked after its decision."""
    lc = late_columns(P)
    if not lc:
        return num
    num = num.copy()
    for c in lc:
        flag = P[c].reset_index(drop=True)
        flag.index = num.index
        for m in [m for m, s_ in line.measurements.items() if s_ == c[len("_late_"):] and m in num.columns]:
            num[m] = num[m].mask(flag)
    return num


def _known_count(df, key, causal):
    """Parts sharing `key`: all of them, or (causal) those already decided at this part's decision time."""
    if not causal:
        return df.groupby(key)["Id"].transform("count").astype("float32")
    return df.groupby(key)["decide"].rank(method="max").astype("float32")


def order_features(P, causal):
    """Neighbour and group features over all parts, as new columns of P (row order kept).

    What is known about another part q at this part's decision time D: its start once q.start <= D; its route,
    end and signature only once q is itself decided (q.decide <= D), because before that it may still visit
    stations that change them. Masks below follow exactly that rule when causal.
    """
    K = P[["Id", "start", "end", "decide", "_path", "_sig"]]
    new = {}

    def keep(v, ok):
        return v.where(ok) if causal else v

    def same(a, b):
        return (a == b).fillna(False).astype("float32")

    # start order: the previous / next part to enter the line
    o = K.sort_values(["start", "Id"])
    p, n = o.shift(1), o.shift(-1)
    new["S_prev_id"] = o["Id"] - p["Id"]
    new["S_next_id"] = keep(n["Id"] - o["Id"], n["start"] <= o["decide"])
    new["S_prev_gap"] = o["start"] - p["start"]
    new["S_next_gap"] = keep(n["start"] - o["start"], n["start"] <= o["decide"])
    new["S_prev_same_path"] = keep(same(o["_path"], p["_path"]), p["decide"] <= o["decide"])
    new["S_next_same_path"] = keep(same(o["_path"], n["_path"]), n["decide"] <= o["decide"])
    new["S_prev_same_end"] = keep(same(o["end"], p["end"]), p["decide"] <= o["decide"])
    new["start_rank"] = o.groupby("start").cumcount().astype("float32")
    if not causal:
        new["sig_rank"] = o.groupby("_sig").cumcount().astype("float32")
    # decision order: the previous / next part to be decided (end of line, or arriving at the checkpoint)
    o = K.sort_values(["decide", "Id"])
    p, n = o.shift(1), o.shift(-1)
    new["E_prev_id"] = o["Id"] - p["Id"]
    new["E_next_id"] = keep(n["Id"] - o["Id"], n["decide"] <= o["decide"])
    new["E_prev_gap"] = o["decide"] - p["decide"]
    new["E_next_gap"] = keep(n["decide"] - o["decide"], n["decide"] <= o["decide"])
    # Id order: the line's own numbering of parts
    o = K.sort_values("Id")
    p, n = o.shift(1), o.shift(-1)
    new["I_prev_gap"] = keep(o["start"] - p["start"], p["start"] <= o["decide"])
    new["I_next_gap"] = keep(n["start"] - o["start"], n["start"] <= o["decide"])
    new["I_prev_end_gap"] = keep(o["end"] - p["end"], p["decide"] <= o["decide"])
    new["I_next_end_gap"] = keep(n["end"] - o["end"], n["decide"] <= o["decide"])
    # groups. Every part sharing this start has started by D, so start_n is known in full even live
    new["start_n"] = K.groupby("start")["Id"].transform("count").astype("float32")
    for key, name in (("end", "end_n"), ("_sig", "sig_n"), ("_path", "path_n")):
        new[name] = _known_count(K, key, causal)
    return _join(P, {k: v.sort_index().astype("float32") for k, v in new.items()})


def _join(P, cols):
    xp = __import__(type(P).__module__.split(".")[0])
    return xp.concat([P, xp.DataFrame(cols, index=P.index)], axis=1)


def measurement_columns(line, columns, sts):
    """The numeric file's measurement columns taken at the allowed stations, in file order."""
    keep = set(sts)
    return [c for c in columns if line.measurements.get(c) in keep]


def numeric_features(num, sts, P, xp, line, checkpoint, causal):
    """Measurement columns of the allowed stations, and per-part summaries: (extra frame aligned to num, cols)."""
    cols = measurement_columns(line, num.columns, sts)
    E = num[["Id"]].copy()
    if checkpoint is None:
        E["num_nnz"] = num["num_nnz"].astype("float32")
    elif cols:
        E["num_nnz"] = num[cols].notna().sum(axis=1).astype("float32")
    if cols:
        # how many parts share exactly these measurements (known only once those parts are decided, when causal)
        E["_num"] = num["num_hash"] if checkpoint is None else _hash_rows(num[cols], xp)
        E = E.merge(P[["Id", "decide"]], on="Id", how="left").sort_values("Id").reset_index(drop=True)
        E["num_dup_n"] = _known_count(E, "_num", causal)
        E = E.drop(columns=["decide"])
    return E, cols


def _alloc(xp, shape):
    if xp.__name__ == "cudf":
        import cupy as cp
        return cp.empty(shape, dtype=cp.float32)
    return np.empty(shape, dtype=np.float32)


def _block(df, xp):
    return df.to_cupy(dtype="float32") if xp.__name__ == "cudf" else df.to_numpy(dtype=np.float32)


def build(st, num, xp, line, checkpoint=None, causal=False, numeric=True, chunk=100_000):
    """(X, y, ids, names) for the labelled parts in Id order.

    X is one float32 matrix (CuPy with cudf, NumPy with pandas), allocated once and filled in row chunks: the
    measurement columns are never copied as a whole frame, which keeps peak memory near data + matrix.
    """
    st = st.reset_index(drop=True)
    checkpoint = line.checkpoint_station(checkpoint)
    sts = stations(line, st.columns, checkpoint)
    P = order_features(part_table(st, sts, xp, line, checkpoint), causal)
    keys = group_keys(P, sts, line)
    P = _join(P, group_counts(keys, _host(P["decide"]).astype(np.float64), causal, line))
    nbr = neighbours(P, causal)
    lab = nbr.pop("lab_pos")
    gkeys = {g: (k[lab], v[lab]) for g, (k, v) in keys.items()}
    P = P[P["is_train"] == 1].drop(columns=["is_train", "_path", "_sig"]).sort_values("Id").reset_index(drop=True)
    if not bool((num["Id"].diff().fillna(1) > 0).all()):
        num = num.sort_values("Id").reset_index(drop=True)
    if len(P) != len(num) or not bool((P["Id"].reset_index(drop=True) == num["Id"].reset_index(drop=True)).all()):
        raise ValueError("labelled parts in the stations and numeric caches differ; re-run the ingest")
    num = cut_measurements(num, P, line)
    P = P.drop(columns=late_columns(P))
    E, cols = numeric_features(num, sts, P, xp, line, checkpoint, causal) if numeric else (None, [])
    if E is not None and "_num" in E.columns:
        gkeys["num"] = (_host(E["_num"]).astype(np.int64), np.ones(len(E), bool))
        E = E.drop(columns=["_num"])
    nbr["decide"] = _host(P["decide"]).astype(np.float64)
    nbr["groups"] = label_groups(gkeys, _host(P["qc_time"]).astype(np.float64), nbr["decide"], causal, line)
    nbr["group_names"] = groups(line)
    label_names = label_feature_names(line)
    nbr["label_names"] = label_names
    pcols = [c for c in P.columns if c not in ("Id", "decide", "qc_time")]
    ecols = [c for c in E.columns if c != "Id"] if E is not None else []
    names = pcols + ecols + cols + label_names
    n = len(P)
    X = _alloc(xp, (n, len(names)))
    X[:, len(names) - len(label_names):] = np.nan           # filled per fold by label_features()
    a, b = len(pcols), len(pcols) + len(ecols)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        X[i:j, :a] = _block(P.iloc[i:j][pcols], xp)
        if ecols:
            X[i:j, a:b] = _block(E.iloc[i:j][ecols], xp)
        if cols:
            X[i:j, b:b + len(cols)] = _block(num.iloc[i:j][cols], xp)
    y = num["Response"].to_numpy().astype(np.int8)
    ids = num["Id"].to_numpy().astype(np.int32)
    if not np.array_equal(ids, nbr["ids"]):
        raise ValueError("neighbour index is not aligned with the labelled parts")
    return Dataset(X, y, ids, names, nbr)


# ------------------------------------------------------------------ neighbour labels
# A part's QC result is known once it has left the line (qc_time). Parts that fail come in clusters (bad batches,
# a drifting station), so the results of the parts produced around this one are strong evidence. They are
# features only where the result exists: in cross-validation never for parts in the scored fold, never for the
# unlabelled test parts, and when causal only if the neighbour's QC was done by this part's decision time.
K_START, K_ID, K_DECIDE = 5, 2, 3
_SLOTS = ([("S", d) for d in range(-K_START, 0)] + [("S", d) for d in range(1, K_START + 1)]
          + [("I", d) for d in (-2, -1, 1, 2)] + [("E", d) for d in range(-K_DECIDE, 0)]
          + [("E", d) for d in range(1, K_DECIDE + 1)])
# groups of parts that share something; per group: how many are known (decided by D when causal) and how many of
# them are known failures. Station groups: the same timestamp at a high-traffic station (line.group_stations:
# by default those at least 10 % of parts visit; missing where a checkpoint excludes the station, so the names do
# not depend on the recipe).
NEIGHBOUR_LABELS = ["S_prev1_y", "S_next1_y", "S_prev_ysum", "S_prev_known", "S_next_ysum", "S_next_known",
                    "I_prev1_y", "I_next1_y", "I_ysum", "E_prev1_y", "E_next1_y", "E_ysum", "E_known", "nbr_ysum"]


def count_groups(line):
    return ("hour", "day") + tuple(f"st_{s}" for s in line.group_stations)


def groups(line):
    return ("start", "end", "sig", "num", "path") + count_groups(line)


def label_feature_names(line):
    return NEIGHBOUR_LABELS + [f"{g}_grp_{x}" for g in groups(line) for x in ("ysum", "known")]


class Dataset(tuple):
    """(X, y, ids, names) plus .nbr, the neighbour index the label features are computed from."""
    def __new__(cls, X, y, ids, names, nbr):
        t = super().__new__(cls, (X, y, ids, names))
        t.nbr = nbr
        return t


def neighbours(P, causal):
    """For every labelled part (Id order): the labelled-row index of its neighbours in start, Id and decision
    order (-1 for unlabelled parts or none) and whether that neighbour's QC result is usable at all."""
    host = {c: (P[c].to_numpy() if hasattr(P[c], "to_numpy") else P[c]) for c in
            ("Id", "start", "decide", "qc_time", "is_train")}
    if hasattr(host["Id"], "get"):
        host = {k: v.get() for k, v in host.items()}
    Id, start, decide = host["Id"].astype(np.int64), host["start"], host["decide"]
    qc, train = host["qc_time"], host["is_train"] == 1
    order = np.argsort(Id[train], kind="stable")
    tids = Id[train][order]
    row = np.full(len(Id), -1, np.int64)
    row[np.where(train)[0][order]] = np.arange(train.sum())
    orders = {"S": np.lexsort((Id, start)), "I": np.argsort(Id, kind="stable"), "E": np.lexsort((Id, decide))}
    lab = np.where(train)[0][order]                       # global position of each labelled part, Id order
    n, m = len(lab), len(_SLOTS)
    idx = np.full((n, m), -1, np.int32)
    ok = np.zeros((n, m), bool)
    for key, o in orders.items():
        rank = np.empty(len(o), np.int64)
        rank[o] = np.arange(len(o))
        r = rank[lab]
        for j, (k, d) in enumerate(_SLOTS):
            if k != key:
                continue
            rr = r + d
            inside = (rr >= 0) & (rr < len(o))
            q = np.where(inside, o[np.clip(rr, 0, len(o) - 1)], 0)
            idx[:, j] = np.where(inside, row[q], -1)
            avail = inside & (row[q] >= 0)
            if causal:
                with np.errstate(invalid="ignore"):
                    avail &= qc[q] <= decide[lab]
            ok[:, j] = avail
    return {"ids": tids.astype(np.int32), "idx": idx, "ok": ok, "lab_pos": lab}


def subset_neighbours(nbr, rows):
    """The neighbour index restricted to labelled rows `rows` (neighbours outside them become unlabelled)."""
    remap = np.full(len(nbr["ids"]), -1, np.int64)
    remap[rows] = np.arange(len(rows))
    idx = nbr["idx"][rows]
    new = np.where(idx >= 0, remap[np.clip(idx, 0, None)], -1).astype(np.int32)
    return {"ids": nbr["ids"][rows], "idx": new, "ok": nbr["ok"][rows] & (new >= 0),       # group sums dropped
            **({"decide": nbr["decide"][rows]} if "decide" in nbr else {}),
            **{k: nbr[k] for k in ("group_names", "label_names", "label_keep") if k in nbr}}


def drop(X, names, nbr, pattern):
    """An ablation: X, names and the neighbour index without the features whose name matches the regex."""
    import re
    rx = re.compile(pattern)
    keep = [n for n in names if not rx.search(n)]
    if len(keep) == len(names):
        raise ValueError(f"--drop {pattern!r} matches no feature")
    nbr = dict(nbr)
    lab = list(nbr.get("label_names") or [])
    if lab:
        full = list(nbr.get("label_names_all", lab))
        kept = [n for n in lab if n in keep]
        nbr["label_names_all"] = full
        nbr["label_names"] = kept
        nbr["label_keep"] = [full.index(n) for n in kept]
        keep = [n for n in keep if n not in set(lab)] + kept          # label features stay last
    X = X[keep] if hasattr(X, "columns") else X[:, [names.index(n) for n in keep]]
    return X, keep, nbr, [n for n in names if n not in set(keep)]


def _host(v):
    a = v.to_numpy() if hasattr(v, "to_numpy") else v
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _tick_key(v, tpu):
    """int64 key of a time: its step on the line's clock (tpu = steps per data unit). Missing -> -1 (no group)."""
    v = np.asarray(v, np.float64)
    return np.where(np.isnan(v), -1, np.round(np.nan_to_num(v) * tpu)).astype(np.int64), ~np.isnan(v)


def group_keys(P, sts, line):
    """(int64 key, valid) per group for every row of P (part_table output: all parts). Shared by the batch build
    and the live node, so both group parts identically."""
    tpu = line.ticks_per_unit
    start = _host(P["start"]).astype(np.float64)
    ok = ~np.isnan(start)
    ones = np.ones(len(start), bool)
    keys = {"start": _tick_key(start, tpu), "end": _tick_key(_host(P["end"]), tpu),
            "sig": (_host(P["_sig"]).astype(np.int64), ones), "path": (_host(P["_path"]).astype(np.int64), ones),
            "hour": (np.floor(np.nan_to_num(start) * line.unit_hours).astype(np.int64), ok),
            "day": (np.floor(np.nan_to_num(start) / line.day).astype(np.int64), ok)}
    for s in line.group_stations:
        keys[f"st_{s}"] = _tick_key(_host(P[f"{s}_t0"]), tpu) if s in sts else (np.full(len(start), -1, np.int64),
                                                                                np.zeros(len(start), bool))
    return keys


def _slices(key, valid, avail, query, causal, tpu):
    """For every row: [lo, hi) of its group's members (sorted by `avail` time) with avail <= its `query` time
    (all members if not causal), and whether the row itself is inside."""
    a = np.round(np.nan_to_num(avail) * tpu)
    big = max(10 ** 7, int(np.max(a, initial=0)) + 2)        # a missing time sorts after every real one
    a100 = np.where(np.isnan(avail), big, a).astype(np.int64)
    q100 = np.where(np.isnan(query), -1, np.round(np.nan_to_num(query) * tpu)).astype(np.int64)
    o = np.lexsort((a100, key))
    ks = key[o]
    _, dense = np.unique(ks, return_inverse=True)
    dense = dense.reshape(-1).astype(np.int64)
    comp = dense * (big + 1) + a100[o]
    r = np.empty(len(o), np.int64)
    r[o] = np.arange(len(o))
    lo = np.searchsorted(ks, ks, "left")[r]
    hi_all = np.searchsorted(ks, ks, "right")[r]
    if causal:
        hi = np.minimum(np.maximum(np.searchsorted(comp, dense[r] * (big + 1) + q100, "right"), lo), hi_all)
        self_in = a100 <= q100
    else:
        hi, self_in = hi_all, np.ones(len(o), bool)
    return {"order": o, "lo": lo, "hi": hi, "self_in": self_in, "valid": valid}


def group_counts(keys, decide, causal, line):
    """'<group>_n' for the count groups over all parts: members known (decided by D when causal), self included."""
    out = {}
    for g in count_groups(line):
        key, valid = keys[g]
        sl = _slices(key, valid, decide, decide, causal, line.ticks_per_unit)
        out[f"{g}_n"] = np.where(valid, sl["hi"] - sl["lo"], np.nan).astype(np.float32)
    return out


def label_groups(keys, qc, decide, causal, line):
    """Per group, for every labelled part: the slice of members whose QC result can count (QC done by this part's
    decision time when causal), and whether the part itself is in it (so it can be left out)."""
    return {g: _slices(k, v, qc, decide, causal, line.ticks_per_unit) for g, (k, v) in keys.items()}


ID_CLASSES = 5    # group results count only from members in another Id class (Id mod 5): see group_label_features


def group_label_features(groups, y, known=None, ids=None, names=None):
    """(n, 2 * len(names)): known failures and known results among the group's members in *other* Id classes.

    Leaving out only the part itself (leave-one-out) leaks: within a group the sum is S - y_self, so failures sit
    one lower than passes and a model that can infer S (from the group-size features) reads the label back.
    Excluding every member whose Id mod ID_CLASSES equals the part's own makes the feature independent of the
    part's label, and the rule needs only the part's Id, so it is the same in CV, in the final fit and live."""
    y = np.asarray(y, np.float64)
    kn = np.ones(len(y), bool) if known is None else np.asarray(known, bool)
    cls = (np.asarray(ids, np.int64) % ID_CLASSES) if ids is not None else np.zeros(len(y), np.int64)
    cols = []
    for g in (names if names is not None else list(groups)):
        if g not in groups:
            cols += [np.full(len(y), np.nan), np.full(len(y), np.nan)]
            continue
        G = groups[g]
        o, lo, hi = G["order"], G["lo"], G["hi"]
        ys, ks = np.zeros(len(y)), np.zeros(len(y))
        for c in range(ID_CLASSES):
            w = kn & (cls != c)                      # what a part of class c may count
            cy = np.r_[0.0, np.cumsum((y * w)[o])]
            ck = np.r_[0.0, np.cumsum(w[o].astype(np.float64))]
            m = cls == c
            ys[m] = cy[hi[m]] - cy[lo[m]]
            ks[m] = ck[hi[m]] - ck[lo[m]]
        cols += [np.where(G["valid"], ys, np.nan), np.where(G["valid"], ks, np.nan)]
    return np.column_stack(cols)


def label_features(nbr, y, known=None):
    """(n, len(label_feature_names(line))) float32 from neighbour results. known[i]: may part i's result be used (e.g.
    not in the scored fold); None = every labelled result."""
    idx, ok = nbr["idx"], nbr["ok"].copy()
    if known is not None:
        ok &= np.where(idx >= 0, np.asarray(known)[np.clip(idx, 0, None)], False)
    L = np.where(ok, np.asarray(y, np.float32)[np.clip(idx, 0, None)], np.nan)
    sl = {s: j for j, s in enumerate(_SLOTS)}
    S_prev = [sl[("S", d)] for d in range(-K_START, 0)]
    S_next = [sl[("S", d)] for d in range(1, K_START + 1)]
    I_all = [sl[("I", d)] for d in (-2, -1, 1, 2)]
    E_all = [sl[("E", d)] for d in range(-K_DECIDE, 0)] + [sl[("E", d)] for d in range(1, K_DECIDE + 1)]

    def ysum(cols):
        return np.nansum(L[:, cols], axis=1)

    def known_n(cols):
        return ok[:, cols].sum(axis=1)

    out = np.column_stack([
        L[:, sl[("S", -1)]], L[:, sl[("S", 1)]], ysum(S_prev), known_n(S_prev), ysum(S_next), known_n(S_next),
        L[:, sl[("I", -1)]], L[:, sl[("I", 1)]], ysum(I_all), L[:, sl[("E", -1)]], L[:, sl[("E", 1)]],
        ysum(E_all), known_n(E_all), np.nansum(L, axis=1)])
    gn = nbr.get("group_names", ())
    grp = (group_label_features(nbr["groups"], y, known, nbr["ids"], gn) if "groups" in nbr
           else np.full((len(L), 2 * len(gn)), np.nan))
    return np.column_stack([out, grp]).astype(np.float32)


def recipe(name, line):
    """{checkpoint, causal, numeric} for a recipe name ("offline", "live" or a checkpoint of the line)."""
    r = recipes(line)
    name = ALIASES.get(name, name)
    if name not in r:
        raise ValueError(f"unknown recipe {name!r} for line {line.name!r}; choose from {list(r)}")
    c, causal, numeric = r[name]
    return {"checkpoint": c, "causal": causal, "numeric": numeric}
