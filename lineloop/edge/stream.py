"""The edge node: consume the line's event stream and score each part at its decision time, with no look-ahead.

The node sees the line only through events, processed one step of the line's clock at a time:

  start     a part entered the line (its Id and start time)
  qc        a part left the final test; its result is now known (labelled parts only)
  decide    a part reached the decision point (end of line, or the checkpoint): its own station times and
            measurements up to there arrive with it, and it is scored

Within a tick, starts, then decisions, then QC results are registered before any part of that tick is scored, so
"known by time D" means "<= D", exactly as in lineloop.quality.features. What the node knows about other parts is
only what these events told it (`self.start_of`, `self.decided`, `self.label_of`, ...); the data rows are never
consulted for another part. That is what makes the parity test meaningful: if the node reproduces the batch
causal matrix bit for bit, the batch features use nothing a live node could not know.

Each part's own features (station times, route, measurements) come from features.part_table, which is a function of
that part's record alone; everything that involves other parts is recomputed here online.
"""
import bisect
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..quality import features as F


@dataclass
class Event:
    t: float
    kind: int        # 0 start, 1 qc, 2 decide
    id: int


@dataclass
class Node:
    """Online state of one edge node watching one line."""
    sts: list
    checkpoint: str | None
    numeric_cols: list
    labelled_ids: np.ndarray                   # all labelled Ids, sorted (row index for neighbour results)
    group_names: tuple = ()                    # features.groups(line): the batch column order
    count_names: tuple = ()                    # features.count_groups(line)
    starts: list = field(default_factory=list)          # sorted (start, Id)
    start_of: dict = field(default_factory=dict)
    start_n: Counter = field(default_factory=Counter)
    decided: list = field(default_factory=list)         # (decide, Id) in decision order
    known: dict = field(default_factory=dict)           # Id -> (decide, end, path, sig) once decided
    group_n: dict = field(default_factory=lambda: {k: Counter() for k in ("end", "sig", "path", "num")})
    label_of: dict = field(default_factory=dict)        # Id -> 0/1 once its QC result arrived
    num_of: dict = field(default_factory=dict)          # Id -> measurement hash (labelled parts)
    g_y: dict = field(default_factory=dict)             # known failures per group key
    g_n: dict = field(default_factory=dict)             # known results per group key
    g_dec: dict = field(default_factory=dict)           # decided parts per group key (count groups)
    keys_fn: object = None      # Id -> {group: key or None}; called only for parts already decided (own record)

    def __post_init__(self):
        self.g_y = {g: Counter() for g in self.group_names}
        self.g_n = {g: Counter() for g in self.group_names}
        self.g_dec = {g: Counter() for g in self.count_names}

    # ------------------------------------------------------------- events
    def on_start(self, i, t):
        bisect.insort(self.starts, (t, i))
        self.start_of[i] = t
        self.start_n[t] += 1

    def group_keys(self, i):
        """Group keys of a decided part, exactly as features.group_keys computes them."""
        return {**self.keys_fn(i), "num": self.num_of.get(i)}

    def on_qc(self, i, label):
        self.label_of[i] = label
        c = i % F.ID_CLASSES
        for g, k in self.group_keys(i).items():
            if k is not None:
                self.g_y[g][k] += label                  # all classes
                self.g_n[g][k] += 1
                self.g_y[g][(k, c)] += label             # this Id class, to subtract (see group_label_features)
                self.g_n[g][(k, c)] += 1

    def group_row(self, i, y_self=None):
        """Known failures / results in the part's groups, from members of other Id classes only."""
        out = []
        c = i % F.ID_CLASSES
        keys = self.group_keys(i)
        for g in self.group_names:               # the batch column order
            k = keys.get(g)
            if k is None:
                out += [np.nan, np.nan]
            else:
                out += [self.g_y[g][k] - self.g_y[g][(k, c)], self.g_n[g][k] - self.g_n[g][(k, c)]]
        return out

    def on_decide(self, i, decide, end, path, sig, num_key):
        self.decided.append((decide, i))
        self.known[i] = (decide, end, path, sig)
        self.group_n["end"][end] += 1
        self.group_n["sig"][sig] += 1
        self.group_n["path"][path] += 1
        if num_key is not None:
            self.group_n["num"][num_key] += 1
            self.num_of[i] = int(num_key)
        for g, k in self.keys_fn(i).items():
            if k is not None and g in self.g_dec:
                self.g_dec[g][k] += 1

    # ------------------------------------------------------------- features of one part at its decision
    def order_row(self, i, start, end, decide, path, sig, pos_decided):
        nan = np.nan
        r = {}
        k = bisect.bisect_left(self.starts, (start, i))
        prev = self.starts[k - 1] if k > 0 else None
        nxt = self.starts[k + 1] if k + 1 < len(self.starts) else None
        r["S_prev_id"] = i - prev[1] if prev else nan
        r["S_next_id"] = nxt[1] - i if nxt and nxt[0] <= decide else nan
        r["S_prev_gap"] = start - prev[0] if prev else nan
        r["S_next_gap"] = nxt[0] - start if nxt and nxt[0] <= decide else nan
        pk = self.known.get(prev[1]) if prev else None
        nk = self.known.get(nxt[1]) if nxt else None
        r["S_prev_same_path"] = float(pk[2] == path) if pk and pk[0] <= decide else nan
        r["S_next_same_path"] = float(nk[2] == path) if nk and nk[0] <= decide else nan
        r["S_prev_same_end"] = float(pk[1] == end) if pk and pk[0] <= decide else nan
        r["start_rank"] = float(k - bisect.bisect_left(self.starts, (start, -1)))
        p = self.decided[pos_decided - 1] if pos_decided > 0 else None
        n = self.decided[pos_decided + 1] if pos_decided + 1 < len(self.decided) else None
        r["E_prev_id"] = i - p[1] if p else nan
        r["E_next_id"] = n[1] - i if n and n[0] <= decide else nan
        r["E_prev_gap"] = decide - p[0] if p else nan
        r["E_next_gap"] = n[0] - decide if n and n[0] <= decide else nan
        for d, tag in ((-1, "prev"), (1, "next")):
            q = i + d                                   # Ids are one contiguous range: Id +- 1 always exists
            qs = self.start_of.get(q)
            r[f"I_{tag}_gap"] = ((start - qs) if d < 0 else (qs - start)) if qs is not None and qs <= decide else nan
            qk = self.known.get(q)
            ok = qk is not None and qk[0] <= decide
            r[f"I_{tag}_end_gap"] = ((end - qk[1]) if d < 0 else (qk[1] - end)) if ok else nan
        r["start_n"] = float(self.start_n[start])
        for g, k in self.keys_fn(i).items():
            if g in self.g_dec:
                r[f"{g}_n"] = float(self.g_dec[g][k]) if k is not None else nan
        r["end_n"] = float(self.group_n["end"][end]) if end == end else nan
        r["sig_n"] = float(self.group_n["sig"][sig])
        r["path_n"] = float(self.group_n["path"][path])
        return r

    def neighbour_slots(self, i, start, decide, pos_decided):
        """(labelled row index, result known) per F._SLOTS, from what the node has seen."""
        idx = np.full(len(F._SLOTS), -1, np.int32)
        ok = np.zeros(len(F._SLOTS), bool)
        k = bisect.bisect_left(self.starts, (start, i))
        for j, (key, d) in enumerate(F._SLOTS):
            if key == "S":
                kk = k + d
                q = self.starts[kk][1] if 0 <= kk < len(self.starts) else None
            elif key == "I":
                q = i + d
            else:
                kk = pos_decided + d
                q = self.decided[kk][1] if 0 <= kk < len(self.decided) else None
            if q is None:
                continue
            row = int(np.searchsorted(self.labelled_ids, q))
            if row < len(self.labelled_ids) and self.labelled_ids[row] == q:
                idx[j] = row
                ok[j] = q in self.label_of
        return idx, ok


def events_from(st, line, checkpoint, t_from=-np.inf):
    """(part table, events after t_from) from the stations frame: start, qc and decide times per part."""
    sts = F.stations(line, st.columns, checkpoint)
    P = F.part_table(st.reset_index(drop=True), sts, pd, line, checkpoint)
    ok = P["decide"].notna().to_numpy()
    ids = P["Id"].to_numpy()
    ev = []
    for col, kind in (("start", 0), ("qc_time", 1), ("decide", 2)):
        t = P[col].to_numpy()
        m = ok & (t >= t_from)
        if kind == 1:
            m &= P["is_train"].to_numpy() == 1
        ev.append(np.rec.fromarrays([t[m], np.full(m.sum(), kind), ids[m]], names="t,kind,id"))
    ev = np.concatenate(ev)
    ev.sort(order=["t", "kind", "id"])
    return P, ev


class Stream:
    """Replay a line's events through a Node and a scorer; collects one row per decided labelled part.

    t_from: events before it are loaded into the node in bulk (the state a node that had been running would
    hold); only events from t_from on are replayed one tick at a time. The tests check this gives the same rows
    as replaying everything.

    Speed: each part's own features and measurements sit in NumPy matrices indexed by Id (they are functions of
    that part's record alone); per tick the node fills one preallocated block, so the hot loop is the online
    neighbour state, not dataframe bookkeeping."""

    def __init__(self, st, num, names, line, checkpoint, causal=True, labels=None, t_from=None, log=None,
                 raw_fn=None):
        if not causal:
            raise ValueError("a live node can only compute causal features")
        clock = [time.perf_counter()]

        def lap(what):
            if log:
                now = time.perf_counter()
                log(f"  {what}: {now - clock[0]:.1f} s")
                clock[0] = now

        self.names = list(names)
        self.raw_fn = raw_fn              # Ids -> raw signal blocks, for a model with a signal stage
        col = {n: j for j, n in enumerate(self.names)}
        self.line = line
        checkpoint = line.checkpoint_station(checkpoint)
        self.checkpoint = checkpoint
        self.sts = F.stations(line, st.columns, checkpoint)
        self.P, self.events = events_from(st, line, checkpoint)
        P = self.P
        lap(f"part table and {len(self.events):,} events")
        ids_all = P["Id"].to_numpy().astype(np.int64)
        self.row_of = np.full(int(ids_all.max()) + 2, -1, np.int64)
        self.row_of[ids_all] = np.arange(len(ids_all))
        self.own_cols = [n for n in self.names if n in P.columns and n != "Id"]
        self.O = P[self.own_cols].to_numpy(np.float32)
        self.own_idx = np.array([col[n] for n in self.own_cols], np.int64)
        self.c = {k: P[k].to_numpy() for k in ("start", "end", "decide")}
        self.c["_path"] = P["_path"].to_numpy().astype(np.int64)
        self.c["_sig"] = P["_sig"].to_numpy().astype(np.int64)
        # labelled parts, in Id order
        self.num = None
        if num is not None:
            nm = num.sort_values("Id").reset_index(drop=True)
            if F.late_columns(P):                       # reworked after D: the same cut as the batch features
                Pl = P[P["Id"].isin(nm["Id"])].sort_values("Id").reset_index(drop=True)
                nm = F.cut_measurements(nm, Pl, line)
            lab = nm["Id"].to_numpy()
            self.labels = labels if labels is not None else nm["Response"].to_numpy()
            self.num_cols = [n for n in self.names if n in nm.columns and n not in ("num_nnz", "Id", "Response")]
            self.Nm = nm[self.num_cols].to_numpy(np.float32)
            self.num_idx = np.array([col[n] for n in self.num_cols], np.int64)
            cols = F.measurement_columns(line, nm.columns, self.sts)
            if checkpoint is None:
                self.nkey, self.nnz = nm["num_hash"].to_numpy(), nm["num_nnz"].to_numpy().astype(np.float32)
            elif cols:
                self.nkey = F._hash_rows(nm[cols], pd).to_numpy()
                self.nnz = nm[cols].notna().sum(axis=1).to_numpy().astype(np.float32)
            else:
                self.nkey = self.nnz = None
            self.num = True
        else:
            lab = np.array([], np.int64)
            self.labels = labels
        self.labrow = np.full(len(self.row_of), -1, np.int64)
        self.labrow[lab] = np.arange(len(lab))
        self.order_idx = {n: col[n] for n in self.names}
        self.group_names, self.label_names = F.groups(line), F.label_feature_names(line)
        self.label_idx = np.array([col[n] for n in self.label_names], np.int64) if self.label_names[0] in col \
            else None
        lap("own-feature and measurement matrices")
        self.gk = F.group_keys(P, self.sts, line)
        self.node = Node(self.sts, checkpoint, [], np.asarray(lab), self.group_names, F.count_groups(line))
        self.node.keys_fn = self._keys
        lap("group keys")
        if t_from is not None:
            hist = self.events[self.events["t"] < t_from]
            self.events = self.events[self.events["t"] >= t_from]
            self._warm(hist, lap)

    def _keys(self, i):
        r = self.row_of[i]
        return {g: (int(k[r]) if v[r] else None) for g, (k, v) in self.gk.items()}

    def _nk(self, i):
        r = self.labrow[i]
        return None if r < 0 or self.nkey is None else int(self.nkey[r])

    def _warm(self, hist, lap=lambda *_: None):
        node, c = self.node, self.c
        s = hist[hist["kind"] == 0]
        st_t, st_i = s["t"], s["id"].astype(np.int64)
        o = np.lexsort((st_i, st_t))
        node.starts = list(zip(st_t[o].tolist(), st_i[o].tolist()))
        node.start_of = dict(zip(st_i.tolist(), st_t.tolist()))
        node.start_n = Counter(st_t.tolist())
        lap(f"warm: {len(st_i):,} starts")
        d_i = hist[hist["kind"] == 2]["id"].astype(np.int64)
        r = self.row_of[d_i]
        node.decided = list(zip(c["decide"][r].astype(np.float64).tolist(), d_i.tolist()))
        dec, end = c["decide"][r].astype(np.float64).tolist(), c["end"][r].astype(np.float64).tolist()
        path, sig = c["_path"][r].tolist(), c["_sig"][r].tolist()
        node.known = {i: (a, b, p_, g) for i, a, b, p_, g in zip(d_i.tolist(), dec, end, path, sig)}
        node.group_n["end"] = Counter(end)
        node.group_n["sig"] = Counter(sig)
        node.group_n["path"] = Counter(path)
        if self.num and self.nkey is not None:
            lr = self.labrow[d_i]
            m = lr >= 0
            keys = self.nkey[lr[m]].astype(np.int64).tolist()
            node.num_of = dict(zip(d_i[m].tolist(), keys))
            node.group_n["num"] = Counter(keys)
        lap(f"warm: {len(d_i):,} decisions")
        for g in F.count_groups(self.line):
            k, v = self.gk[g]
            node.g_dec[g] = Counter(k[r][v[r]].tolist())
        lap("warm: group counts")
        q = hist[hist["kind"] == 1]["id"].astype(np.int64)
        yq = self.labels[self.labrow[q]].astype(int)
        node.label_of = dict(zip(q.tolist(), yq.tolist()))
        rq = self.row_of[q]
        cq = q % F.ID_CLASSES

        def load(g, keys, ys, cs):
            tot = pd.Series(ys).groupby(keys).agg(["sum", "count"])
            per = pd.DataFrame({"k": keys, "c": cs, "y": ys}).groupby(["k", "c"])["y"].agg(["sum", "count"])
            node.g_y[g] = Counter({**dict(zip(tot.index.tolist(), tot["sum"].tolist())),
                                   **dict(zip(per.index.tolist(), per["sum"].tolist()))})
            node.g_n[g] = Counter({**dict(zip(tot.index.tolist(), tot["count"].tolist())),
                                   **dict(zip(per.index.tolist(), per["count"].tolist()))})

        for g, (k, v) in self.gk.items():
            m = v[rq]
            load(g, k[rq][m], yq[m], cq[m])
        if self.num and self.nkey is not None:
            load("num", self.nkey[self.labrow[q]].astype(np.int64), yq, cq)
        lap(f"warm: {len(q):,} QC results into {len(self.gk) + 1} group kinds")

    def replay(self, scorer=None, threshold=None, t_until=np.inf, log=None):
        """Process every event up to t_until. Returns (Ids, decision times, rows [features (+ score)])."""
        node, ev = self.node, self.events
        tt, kk, ii = ev["t"], ev["kind"], ev["id"].astype(np.int64)
        cut = np.searchsorted(tt, t_until, "right") if np.isfinite(t_until) else len(tt)
        bounds = np.r_[0, np.where(np.diff(tt[:cut]) != 0)[0] + 1, cut] if cut else np.array([0])
        out_ids, out_rows, out_t, lat = [], [], [], []
        self.raw_s = 0.0
        step, next_log, t_start = max(cut // 10, 1), max(cut // 10, 1), time.perf_counter()
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            if log and b0 >= next_log:
                el = time.perf_counter() - t_start
                log(f"  {b0:,} / {cut:,} events, {el:.0f} s, {b0 / max(el, 1e-9):,.0f} events/s")
                next_log += step
            t = float(tt[b0])
            kind, ids = kk[b0:b1], ii[b0:b1]
            t0 = time.perf_counter()
            for i in ids[kind == 0].tolist():
                node.on_start(i, t)
            deciding = ids[kind == 2]
            if len(deciding):
                self._register(deciding)
            for q in ids[kind == 1].tolist():
                node.on_qc(q, int(self.labels[self.labrow[q]]))
            if len(deciding):
                scored, X = self._score(deciding)
                t_raw = 0.0
                if len(scored):
                    S = None
                    if scorer is not None and self.raw_fn is not None:
                        tr = time.perf_counter()          # the welder hands the node its raw block: simulated here,
                        S = self.raw_fn(scored)           # so its making is not part of the node's latency
                        t_raw = time.perf_counter() - tr
                        self.raw_s += t_raw
                    out_rows.append(np.column_stack([X, scorer(X, S)]) if scorer is not None else X)
                    out_ids.append(scored)
                    out_t.append(np.full(len(scored), t))
                lat.append((time.perf_counter() - t0 - t_raw, len(deciding), len(scored)))
        self.ticks = np.array(lat, dtype=[("seconds", "f8"), ("decided", "i4"), ("scored", "i4")])
        width = len(self.names) + (1 if scorer is not None else 0)
        M = np.vstack(out_rows) if out_rows else np.zeros((0, width), np.float32)
        return (np.concatenate(out_ids) if out_ids else np.zeros(0, np.int64),
                np.concatenate(out_t) if out_t else np.zeros(0), M)

    def _register(self, ids):
        """Register every part deciding in this tick ("known by D" includes the same tick)."""
        node, c = self.node, self.c
        self._base = len(node.decided)
        r = self.row_of[ids]
        for i, de, en, pa, sg in zip(ids.tolist(), c["decide"][r].astype(np.float64).tolist(),
                                     c["end"][r].astype(np.float64).tolist(), c["_path"][r].tolist(),
                                     c["_sig"][r].tolist()):
            node.on_decide(i, de, en, pa, sg, self._nk(i))

    def _score(self, ids):
        """Feature rows of the labelled parts deciding in this tick. Unlabelled parts are on the line and shape
        every neighbour feature, but have no result to score against (in Bosch, the withheld test set)."""
        node, c = self.node, self.c
        lr = self.labrow[ids]
        keep = np.where(lr >= 0)[0]
        if not len(keep):
            return np.zeros(0, np.int64), np.zeros((0, len(self.names)), np.float32)
        X = np.full((len(keep), len(self.names)), np.nan, np.float32)
        r = self.row_of[ids[keep]]
        X[:, self.own_idx] = self.O[r]
        if self.num:
            X[:, self.num_idx] = self.Nm[lr[keep]]
        oi = self.order_idx
        s_idx, s_ok, grp = [], [], []
        for row, k in enumerate(keep.tolist()):
            i = int(ids[k])
            rr = r[row]
            st_, en, de = float(c["start"][rr]), float(c["end"][rr]), float(c["decide"][rr])
            o = node.order_row(i, st_, en, de, int(c["_path"][rr]), int(c["_sig"][rr]), self._base + k)
            if self.num and self.nkey is not None:
                o["num_dup_n"] = float(node.group_n["num"][int(self.nkey[lr[k]])])
            if self.num and self.nnz is not None:
                o["num_nnz"] = float(self.nnz[lr[k]])
            for name, v in o.items():
                j = oi.get(name)
                if j is not None:
                    X[row, j] = v
            si, so = node.neighbour_slots(i, st_, de, self._base + k)
            s_idx.append(si)
            s_ok.append(so)
            grp.append(node.group_row(i))
        if self.label_idx is not None:
            L = F.label_features({"idx": np.array(s_idx), "ok": np.array(s_ok), "group_names": self.group_names},
                                 self.labels, None)
            L[:, L.shape[1] - 2 * len(self.group_names):] = np.array(grp, dtype=np.float32)
            X[:, self.label_idx] = L
        return ids[keep], X
