"""A synthetic history for any line spec, so the whole loop runs on a line before (or without) its real data.

The spec's [simulate] table says how parts move and why some fail; the line's segments, stations, clock and
data format come from the spec itself, so the output is exactly what that line's ingest expects.

How a part is made:
  release   parts enter in batches (sizes `batch`), with a day / week rhythm: full rate on weekdays, `weekend`
            times the rate from Saturday to Monday, and a daily swing. About `parts_per_week` parts a week.
  route     each part follows one route (a list of segments, chosen by `share`); in a segment it visits every
            station, except that of a group of `parallel` stations it visits one, at random.
  timing    at each station the part spends no time or, half the time, an exponential delay of mean
            `step_hours`; times are rounded to the spec's clock.
  measure   each station measures `measurements_per_station` values: noise, shifted for parts released in a
            "bad" window (`bad_share` of `bad_window_hours` windows run a degraded process). A share
            `missing_share` of values is not recorded.
  drift     a `drift` entry moves one measurement's mean by `per_week` every week, so a model trained on
            early weeks meets a slightly different line later (as real lines do).
  fail      a defect arises from independent mechanisms, each at a station (its origin):
              - every `driver`: a defect at the driver's station, log-odds `offset + weight x measured value`;
              - a bad window: a defect at the part's first station, log-odds `offset + bad_risk`;
              - background: a defect at a random station on the part's route, log-odds `offset + the route's
                risk`.
            A part fails the final test if any mechanism fires; its origin is the earliest station among them.
            The offset is solved so that `fail_rate` is the expected share of parts that fail.
  signals   for a [[signal]] the line records (lineloop.edge.signals), a `signal_driver` gives every repeat (each
            weld) a latent quality z ~ N(0, 1) (+ `bad_shift` in a bad window), and each repeat is its own defect
            mechanism at the signal's station, log-odds `offset + weight x (z - above)`. The repeat's waveforms
            carry z through noisy signatures (a level that sags, spikes, a final position), together with
            nuisance variation, so z can be estimated from them but not read off. The history keeps only the
            per-part signal features (measurements of the signal's last station) and, per part, z and a seed
            (signal_<name> table), from which `waveforms` re-creates the raw block exactly when the edge node
            needs it.
  origin    the results carry each failure's origin station. An inspection at a checkpoint can only find a defect
            that already exists there; the twin credits catches by it (lineloop.twin.data).
  label     a share `labelled_share` of parts keep their result; the rest are on the line but unlabelled.

Nothing here is fitted to a real line: it is a fixture with the structure the framework looks for (failures that
cluster in time, defects that arise before and after each checkpoint), not a twin of any plant.
"""
import numpy as np
import pandas as pd

DEFAULTS = {
    "parts": 60000, "parts_per_week": 20000, "batch": [1, 8], "weekend": 0.2, "step_hours": 0.5,
    "measurements_per_station": 3, "missing_share": 0.05, "noise": 0.1, "bad_share": 0.03,
    "bad_window_hours": 0.1, "bad_shift": 0.08, "bad_risk": 4.0, "fail_rate": 0.012, "labelled_share": 1.0,
    "id_jitter_hours": 3.0, "id_prefix": None, "measurement_names": "m{j}", "seed": 0,
}


def config(ln, **over):
    if not ln.simulate:
        raise ValueError(f"line {ln.name!r} has no [simulate] table")
    c = {**DEFAULTS, **ln.simulate, **{k: v for k, v in over.items() if v is not None}}
    unknown = set(c) - set(DEFAULTS) - {"route", "parallel", "driver", "drift", "signal_driver"}
    if unknown:
        raise ValueError(f"unknown [simulate] keys: {sorted(unknown)}")
    if not c.get("route"):
        c["route"] = [{"share": 1.0, "segments": list(ln.segments)}]
    for r in c["route"]:
        bad = [g for g in r["segments"] if g not in ln.segments]
        if bad:
            raise ValueError(f"route names unknown segments {bad}")
    for grp in c.get("parallel", []):
        if any(s not in ln.stations for s in grp):
            raise ValueError(f"parallel group {grp} names stations not on the line")
    names = {sg.name for sg in ln.signal_specs()}
    for d in c.get("signal_driver", []):
        if d.get("signal") not in names:
            raise ValueError(f"signal_driver {d} must name a [[signal]] of the line: {sorted(names)}")
    for kind in ("driver", "drift"):
        for d in c.get(kind, []):
            if d["station"] not in ln.stations or not 0 <= d.get("measurement", 0) < c["measurements_per_station"]:
                raise ValueError(f"{kind} {d} must name a station and a measurement index < measurements_per_station")
    return c


def _templates(S):
    t = np.arange(S, dtype=np.float64) / S
    trap = np.clip(np.minimum(t / 0.1, (1.0 - t) / 0.1), 0.0, 1.0)
    rise = (1.0 - np.exp(-t / 0.2)) / (1.0 - np.exp(-5.0))
    return trap, rise


def waveforms(sig, z, seeds):
    """Raw blocks [n, per_part, channels, samples] float32 for parts with repeat qualities z [n, per_part] and
    seeds [n]; deterministic in (z, seed). Channel k behaves like a power (k % 3 == 0: its level sags as z rises),
    a photodiode (1: spatter spikes, more of them as z rises) or a depth (2: the final penetration grows)."""
    W, C, S = sig.shape
    trap, rise = (t.astype(np.float32) for t in _templates(S))
    (m0, m1), _ = sig.windows()
    out = np.empty((len(z), W, C, S), np.float32)
    w_idx = np.arange(W)[:, None]
    for i, (zi, sd) in enumerate(zip(np.asarray(z, np.float32), np.asarray(seeds))):
        rng = np.random.default_rng(int(sd))
        nu = rng.standard_normal((W, C), dtype=np.float32)
        x = out[i]
        rng.standard_normal((W, C, S), dtype=np.float32, out=x)
        x *= np.float32(0.02)
        for k in range(C):
            kind = k % 3
            if kind == 0:
                x[:, k] += (1.0 - 0.03 * zi + 0.03 * nu[:, k])[:, None] * trap
            elif kind == 1:
                x[:, k] += (0.8 + 0.02 * nu[:, k])[:, None] * trap
                cnt = np.minimum(rng.poisson(np.exp(0.9 * zi - 1.8)), 8)
                pos = rng.integers(m0, m1 - 3, size=(W, 8))
                amp = (rng.uniform(0.3, 0.6, size=(W, 8)) * (np.arange(8) < cnt[:, None])).astype(np.float32)
                for d, h in ((0, 0.5), (1, 1.0), (2, 0.5)):
                    np.add.at(x[:, k], (np.broadcast_to(w_idx, pos.shape), pos + d), np.float32(h) * amp)
            else:
                x[:, k] += (1.0 + 0.06 * zi + 0.06 * nu[:, k])[:, None] * rise
    return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _offset(logits, rate):
    """The offset c at which the expected share of parts with at least one defect, 1 - prod(1 - sigmoid(c + L)),
    equals `rate` (bisection; a logit of -inf is a mechanism that cannot fire for that part)."""
    lo, hi = -40.0, 20.0
    for _ in range(80):
        c = (lo + hi) / 2
        p_none = np.prod([1.0 - _sigmoid(c + L) for L in logits], axis=0)
        lo, hi = (c, hi) if 1.0 - p_none.mean() < rate else (lo, c)
    return (lo + hi) / 2


def _path(ln, segments, alt_of, rng):
    out = []
    for g in segments:
        for s in ln.segments[g]:
            grp = alt_of.get(s)
            if grp is None:
                out.append(s)
            elif s == grp[0]:
                out.append(grp[rng.integers(len(grp))])
    return out


def generate(ln, parts=None, seed=None):
    """{visits, results, measurements} DataFrames for the line (times in the line's data units)."""
    c = config(ln, parts=parts, seed=seed)
    n = int(c["parts"])
    rng = np.random.default_rng(int(c["seed"]))
    lo, hi = c["batch"]
    mean_batch = (lo + hi) / 2
    weekday = 120 / 168
    intensity = 0.75 * (weekday + (1 - weekday) * c["weekend"])           # mean of the release rhythm
    gap_h = 168 * intensity * mean_batch / c["parts_per_week"]
    t, starts = 0.0, []
    while len(starts) < n:
        day = 0.5 + 0.5 * np.sin(2 * np.pi * t / 24) ** 2
        week = c["weekend"] if (t % 168) >= 120 else 1.0                   # Saturday and Sunday (from Monday 0 h)
        t += rng.exponential(gap_h / (day * week))
        starts += [t] * int(rng.integers(lo, hi + 1))
    start_h = np.array(starts[:n])
    shares = np.array([r["share"] for r in c["route"]], float)
    route = rng.choice(len(shares), size=n, p=shares / shares.sum())
    win = np.floor(start_h / c["bad_window_hours"]).astype(np.int64)
    bad_win = rng.random(int(win.max()) + 2) < c["bad_share"]
    bad = bad_win[win]
    alt_of = {s: grp for grp in c.get("parallel", []) for s in grp}
    drv = c.get("driver", [])
    drivers = {}
    for di, d in enumerate(drv):
        drivers.setdefault(d["station"], []).append((di, int(d.get("measurement", 0)), float(d["weight"])))
    drifts = {}
    for d in c.get("drift", []):
        drifts.setdefault(d["station"], []).append((int(d.get("measurement", 0)), float(d["per_week"])))
    k = int(c["measurements_per_station"])
    names = [c["measurement_names"].format(j=j) for j in range(k)]
    vp, vs, vt = [], [], []
    mp, ms, mn, mv = [], [], [], []
    x_drv = np.full((n, len(drv)), -np.inf)                 # a driver's mechanism exists only where it is visited
    sigs = {sg.name: sg for sg in ln.signal_specs()}
    sdrv = [(d, sigs[d["signal"]]) for d in c.get("signal_driver", [])]
    sig_at = np.full((n, len(sdrv)), -1, np.int64)          # position of the signal station the part visited
    first_pos, any_pos = np.zeros(n, np.int64), np.zeros(n, np.int64)
    pos = {s: i for i, s in enumerate(ln.stations)}
    for i in range(n):
        tt = start_h[i]
        path = _path(ln, c["route"][route[i]]["segments"], alt_of, rng)
        first_pos[i], any_pos[i] = pos[path[0]], pos[path[rng.integers(len(path))]]
        for k_, (_, sg) in enumerate(sdrv):
            at = [s for s in path if s in sg.stations]
            if at:
                sig_at[i, k_] = pos[at[0]]
        for s in path:
            if rng.random() < 0.5:
                tt += rng.exponential(c["step_hours"])
            vp.append(i)
            vs.append(s)
            vt.append(tt)
            val = rng.normal(0.0, c["noise"], k) + (c["bad_shift"] if bad[i] else 0.0)
            for j, per_week in drifts.get(s, ()):
                val[j] += per_week * start_h[i] / 168.0
            for di, j, w in drivers.get(s, ()):
                x_drv[i, di] = w * val[j]
            keep = rng.random(k) >= c["missing_share"]
            for j in np.where(keep)[0]:
                mp.append(i)
                ms.append(s)
                mn.append(names[j])
                mv.append(round(float(val[j]), 3))
    # independent defect mechanisms, each with its origin station; the part fails if any fires
    logits = [x_drv[:, di] for di in range(len(drv))]
    origins = [np.full(n, pos[d["station"]]) for d in drv]
    logits.append(np.where(bad, c["bad_risk"], -np.inf))
    origins.append(first_pos)
    logits.append(np.array([r.get("risk", 0.0) for r in c["route"]])[route])
    origins.append(any_pos)
    sig_z = []
    for k_, (d, sg) in enumerate(sdrv):                     # one mechanism per repeat (weld) of the signal
        z = rng.normal(size=(n, sg.per_part)) + float(d.get("bad_shift", 0.0)) * bad[:, None]
        seen = sig_at[:, k_] >= 0
        for j in range(sg.per_part):
            logits.append(np.where(seen, float(d["weight"]) * (z[:, j] - float(d.get("above", 2.0))), -np.inf))
            origins.append(np.where(seen, sig_at[:, k_], 0))
        sig_z.append((sg, z, seen, rng.integers(0, 2 ** 62, n)))
    off = _offset(logits, c["fail_rate"])
    fired = np.stack([rng.random(n) < _sigmoid(off + L) for L in logits])
    failed = fired.any(axis=0).astype(np.int8)
    first = np.where(fired, np.stack(origins), len(ln.stations)).min(axis=0)
    origin = np.where(failed == 1, np.array(ln.stations + [""], dtype=object)[first], None)
    # part numbers: roughly (not exactly) in release order, like a line that numbers parts as it plans them
    num = np.argsort(np.argsort(start_h + rng.normal(0, c["id_jitter_hours"], n))) + 1
    pid = (np.array([f"{c['id_prefix']}{x:07d}" for x in num]) if c["id_prefix"] else num.astype(np.int64))
    from ..edge.signals import features_numpy
    side = {}
    for sg, z, seen, seeds in sig_z:                        # the edge node's per-part features, as uploaded
        rows = np.where(seen)[0]
        fn = sg.feature_names()
        for a in range(0, len(rows), 256):
            r = rows[a:a + 256]
            f = features_numpy(sg, waveforms(sg, z[r], seeds[r]))
            mp += np.repeat(r, len(fn)).tolist()
            ms += [sg.station] * (len(r) * len(fn))
            mn += fn * len(r)
            mv += f.ravel().astype(np.float64).tolist()
        side[f"signal_{sg.name}"] = pd.DataFrame({"part": pid[rows], "seed": seeds[rows],
                                                   **{f"z{j}": z[rows, j] for j in range(sg.per_part)}})
    q = ln.resolution
    to_units = np.round(np.array(vt) / ln.unit_hours / q) * q
    visits = pd.DataFrame({"part": pid[np.array(vp, np.int64)], "station": vs, "time": to_units})
    labelled = rng.random(n) < c["labelled_share"]
    results = pd.DataFrame({"part": pid[labelled], "failed": failed[labelled], "origin": origin[labelled]})
    meas = pd.DataFrame({"part": pid[np.array(mp, np.int64)], "station": ms, "name": mn, "value": mv})
    return {"visits": visits, "results": results, "measurements": meas, **side}


def raw_source(ln, cache, signal, src=None):
    """Ids -> raw signal blocks for a simulated line: each part's z and seed from the source's signal_<name>
    table, re-made by `waveforms` (the stand-in for the cell's own signal buffer)."""
    from pathlib import Path
    src = pd.read_parquet(Path(src or ln.source) / f"signal_{signal.name}.parquet")
    ids_p = Path(cache) / "ids.parquet"
    if ids_p.exists():
        m = pd.read_parquet(ids_p)
        to_part = dict(zip(m["Id"].to_numpy(), m["part"].astype(str).to_numpy()))
        key = src["part"].astype(str)
    else:
        to_part, key = None, src["part"]
    row = pd.Series(np.arange(len(src)), index=key.to_numpy())
    Z = src[[f"z{j}" for j in range(signal.per_part)]].to_numpy()
    seeds = src["seed"].to_numpy()

    def fn(ids):
        r = row.loc[[to_part[i] for i in ids] if to_part else list(ids)].to_numpy()
        return waveforms(signal, Z[r], seeds[r])
    return fn


def fingerprint(ln, parts=None, seed=None):
    """A hash of everything a simulated history depends on: the spec's [simulate], [[signal]] and stations."""
    import hashlib
    import json
    d = {"simulate": config(ln, parts=parts, seed=seed), "signals": ln.signals, "segments": ln.segments,
         "unit_hours": ln.unit_hours, "resolution": ln.resolution, "version": 2}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def is_current(ln, src, parts=None):
    """True if `src` holds a history simulated from this spec as it is now."""
    from pathlib import Path
    f = Path(src) / "simulated.txt"
    return f.exists() and f.read_text().strip() == fingerprint(ln, parts)


def write(ln, out=None, parts=None, seed=None, log=print):
    """Generate the line's history and write it where (and how) the spec says its data lives."""
    from pathlib import Path
    out = out or ln.source
    fmt = ln.data.get("format", "events")
    if fmt == "bosch" and ln.signals:
        raise ValueError("the bosch format has no place for raw signals; use the events format")
    tables = generate(ln, parts, seed)
    if fmt == "bosch":
        from . import bosch
        n = bosch.write(tables, out)
    else:
        from . import events
        n = events.write(tables, out)
    (Path(out) / "simulated.txt").write_text(fingerprint(ln, parts, seed) + "\n")
    log(f"simulated {ln.name}: {len(tables['results']):,} labelled of {tables['visits']['part'].nunique():,} parts, "
        f"{int(tables['results']['failed'].sum()):,} failed -> {out} ({fmt}: {n})")
    return n
