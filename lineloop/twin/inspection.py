"""Line twin: an early-inspection cell at a checkpoint, simulated over the real part stream with NVIDIA Warp.

What is real and what is assumed:
  real      every part's arrival at the checkpoint (its recorded time), its live risk score (the edge node's
            checkpoint model, replayed forward in time), whether it really failed the final test and, where the
            data records it, whether its defect already existed at the checkpoint (its origin is upstream)
  assumed   the inspection cell: how long an inspection takes (lognormal, mean and CV), the chance it finds a real
            defect, how many inspectors work each shift, and how long a part may wait before the line moves it on

Policy: a part whose score is at or above the threshold for its route family in the current shift slot (a week cut
into slots of the line's shift length: 21 of 8 h, 14 of 12 h; "never" for a slot with nobody on shift) is pulled
for inspection. It waits for the first free inspector on shift; if that wait would exceed `max_wait_h`, it
continues uninspected (bypass). A real defect present at the checkpoint and found at inspection is removed there
(the work after it is saved); one not flagged, bypassed, missed by the inspector, or not yet made (its origin is
at or after the checkpoint) is found at the final test, as today.

One Warp thread simulates one scenario (policy x staffing x economics x random seed) over the whole stream, so
thousands of what-ifs run in one launch. The same kernel runs on `cuda:0` or, compiled by LLVM, on the CPU, where
Warp runs a launch on one thread: `simulate_cpu_cores` splits the scenarios over every core for a fair CPU
baseline. `simulate_numpy` is the line-by-line reference the tests compare it with.
"""
import math

import numpy as np

SHIFT_H, WEEK_H = 8.0, 168.0           # default shift length; a week is always 168 h


def n_shifts(shift_h=SHIFT_H):
    """Shift slots in a week (21 for 8 h shifts)."""
    return int(round(WEEK_H / shift_h))


N_SHIFTS = n_shifts()
MAX_SERVERS = 16
WAIT_BINS = 64                          # wait histogram: bin k covers [k, k+1) * max_wait_h / WAIT_BINS
_KERNEL = None


def _kernel():
    """Build the Warp kernel on first use (importing warp costs a second; the CPU tests may not need it)."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import warp as wp
    wp.config.quiet = True

    @wp.kernel
    def run(t_h: wp.array(dtype=wp.float32), score: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.int32),
            present: wp.array(dtype=wp.int32), fam: wp.array(dtype=wp.int32), saved_work: wp.array(dtype=wp.float32),
            thr: wp.array2d(dtype=wp.float32), cap: wp.array2d(dtype=wp.int32),
            svc_mean: wp.array(dtype=wp.float32), svc_cv: wp.array(dtype=wp.float32),
            detect: wp.array(dtype=wp.float32), max_wait: wp.array(dtype=wp.float32), seed: wp.array(dtype=wp.int32),
            free: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32), hist: wp.array2d(dtype=wp.int32),
            slot_flag: wp.array2d(dtype=wp.int32), slot_byp: wp.array2d(dtype=wp.int32), shift_h: float,
            n_sh: int):
        s = wp.tid()
        n = t_h.shape[0]
        sig2 = wp.log(1.0 + svc_cv[s] * svc_cv[s])
        mu = wp.log(svc_mean[s]) - 0.5 * sig2
        sig = wp.sqrt(sig2)
        flagged = float(0.0)
        inspected = float(0.0)
        bypassed = float(0.0)
        caught = float(0.0)
        escaped = float(0.0)
        false_alarm = float(0.0)
        busy = float(0.0)
        wait_sum = float(0.0)
        wait_max = float(0.0)
        saved = float(0.0)
        for i in range(n):
            how = t_h[i] - wp.floor(t_h[i] / 168.0) * 168.0
            sh = wp.min(int(how / shift_h), n_sh - 1)
            if score[i] < thr[s, fam[i] * n_sh + sh]:
                if y[i] == 1:
                    escaped += 1.0
                continue
            flagged += 1.0
            slot_flag[s, sh] = slot_flag[s, sh] + 1
            c = cap[s, sh]
            if c <= 0:
                slot_byp[s, sh] = slot_byp[s, sh] + 1
                bypassed += 1.0
                if y[i] == 1:
                    escaped += 1.0
                continue
            k = int(0)
            best = free[s, 0]
            for j in range(1, c):
                if free[s, j] < best:
                    best = free[s, j]
                    k = j
            start = wp.max(t_h[i], best)
            wait = start - t_h[i]
            if wait > max_wait[s]:
                slot_byp[s, sh] = slot_byp[s, sh] + 1
                bypassed += 1.0
                if y[i] == 1:
                    escaped += 1.0
                continue
            st = wp.rand_init(seed[s], 2 * i)
            svc = svc_mean[s]
            if svc_cv[s] > 0.0:
                svc = wp.exp(mu + sig * wp.randn(st))
            free[s, k] = start + svc
            inspected += 1.0
            busy += svc
            wait_sum += wait
            wait_max = wp.max(wait_max, wait)
            b = wp.min(int(wait / max_wait[s] * 64.0), 63)
            hist[s, b] = hist[s, b] + 1
            if y[i] == 1:
                st2 = wp.rand_init(seed[s], 2 * i + 1)
                if present[i] == 1 and wp.randf(st2) < detect[s]:
                    caught += 1.0
                    saved += saved_work[i]
                else:
                    escaped += 1.0
            else:
                false_alarm += 1.0
        out[s, 0] = flagged
        out[s, 1] = inspected
        out[s, 2] = bypassed
        out[s, 3] = caught
        out[s, 4] = escaped
        out[s, 5] = false_alarm
        out[s, 6] = busy
        out[s, 7] = wait_sum
        out[s, 8] = wait_max
        out[s, 9] = saved

    _KERNEL = run
    return run


OUT = ("flagged", "inspected", "bypassed", "caught", "escaped", "false_alarms", "busy_h", "wait_sum_h",
       "wait_max_h", "work_saved")


def scenarios(n_fam, thresholds, capacity, svc_mean_h=0.25, svc_cv=0.5, detect=0.9, max_wait_h=2.0, seed=0,
              shift_h=SHIFT_H):
    """Scenario arrays. thresholds: (S,) one for all, (S, n_fam) per family, or (S, n_fam, H) per family and
    shift slot; capacity: (S,) inspectors on every shift or (S, H) per shift slot (H = n_shifts(shift_h))."""
    H = n_shifts(shift_h)
    thr = np.asarray(thresholds, np.float32)
    if thr.ndim == 1:
        thr = np.repeat(thr[:, None], n_fam, axis=1)
    if thr.ndim == 2:
        thr = np.repeat(thr[:, :, None], H, axis=2)
    S = len(thr)
    thr = thr.reshape(S, n_fam * H)
    cap = np.asarray(capacity, np.int32)
    cap = np.repeat(cap[:, None], H, axis=1) if cap.ndim == 1 else cap
    if cap.shape != (S, H) or cap.max(initial=0) > MAX_SERVERS:
        raise ValueError(f"capacity must be (S, {H}) with at most {MAX_SERVERS} inspectors")

    def col(v, dt=np.float32):
        v = np.asarray(v, dt)
        return np.full(S, v, dt) if v.ndim == 0 else v

    return {"thr": thr, "cap": cap, "svc_mean": col(svc_mean_h), "svc_cv": col(svc_cv), "detect": col(detect),
            "max_wait": col(max_wait_h), "seed": col(seed, np.int32) if np.ndim(seed) else
            (np.arange(S, dtype=np.int32) + int(seed)), "shift_h": float(shift_h)}


def simulate(stream, sc, device="cpu"):
    """Run every scenario over the stream on a Warp device. stream: dict t_h, score, y, fam, saved_work and
    optionally present (1 = the part's defect exists at the checkpoint; default all), sorted by t_h.
    Returns (metrics: OUT names -> (S,), slot_flagged / slot_bypassed -> (S, shift slots);
    wait histogram (S, WAIT_BINS); kernel seconds)."""
    import time

    import warp as wp
    k = _kernel()
    S = len(sc["thr"])
    H = sc["cap"].shape[1]
    with wp.ScopedDevice(device):
        a = {key: wp.array(np.ascontiguousarray(stream[key]), dtype=dt) for key, dt in
             (("t_h", wp.float32), ("score", wp.float32), ("y", wp.int32), ("fam", wp.int32),
              ("saved_work", wp.float32))}
        a["present"] = wp.array(np.ascontiguousarray(present(stream)), dtype=wp.int32)
        p = {"thr": wp.array(sc["thr"], dtype=wp.float32), "cap": wp.array(sc["cap"], dtype=wp.int32)}
        for key in ("svc_mean", "svc_cv", "detect", "max_wait"):
            p[key] = wp.array(sc[key], dtype=wp.float32)
        p["seed"] = wp.array(sc["seed"], dtype=wp.int32)
        free = wp.zeros((S, MAX_SERVERS), dtype=wp.float32)
        out = wp.zeros((S, len(OUT)), dtype=wp.float32)
        hist = wp.zeros((S, WAIT_BINS), dtype=wp.int32)
        sf = wp.zeros((S, H), dtype=wp.int32)
        sb = wp.zeros((S, H), dtype=wp.int32)
        wp.synchronize_device()
        t0 = time.perf_counter()
        wp.launch(k, dim=S, inputs=[a["t_h"], a["score"], a["y"], a["present"], a["fam"], a["saved_work"], p["thr"],
                                    p["cap"],
                                    p["svc_mean"], p["svc_cv"], p["detect"], p["max_wait"], p["seed"], free, out,
                                    hist, sf, sb, float(sc.get("shift_h", SHIFT_H)), int(H)])
        wp.synchronize_device()
        secs = time.perf_counter() - t0
        o, h = out.numpy(), hist.numpy()
        m = {name: o[:, j].astype(np.float64) for j, name in enumerate(OUT)}
        m["slot_flagged"], m["slot_bypassed"] = sf.numpy(), sb.numpy()
    return m, h, secs


_STREAM = None


def _worker_init(stream):
    global _STREAM
    _STREAM = stream
    simulate(stream, _one_scenario(stream), "cpu")                      # load the compiled kernel once


def _one_scenario(stream):
    return scenarios(int(np.max(stream["fam"])) + 1, np.zeros(1, np.float32), np.ones(1, int))


def _first(sc, n=1):
    return {k: (v[:n] if np.ndim(v) else v) for k, v in sc.items()}


def _worker_run(sc):
    return simulate(_STREAM, sc, "cpu")


def simulate_cpu_cores(stream, sc, procs=None):
    """The kernel on every CPU core: one process per core, each running its share of the scenarios on Warp's CPU
    backend. Returns (metrics, wait histogram, wall seconds of the parallel run, processes); start-up and kernel
    loading are not timed, as on the GPU."""
    import multiprocessing as mp
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor
    procs = procs or os.cpu_count() or 1
    S = len(sc["thr"])
    cuts = np.linspace(0, S, procs + 1).astype(int)
    chunks = [{k: (v[a:b] if np.ndim(v) else v) for k, v in sc.items()} for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    with ProcessPoolExecutor(len(chunks), mp_context=mp.get_context("spawn"), initializer=_worker_init,
                             initargs=(stream,)) as ex:
        list(ex.map(_worker_run, [_first(c) for c in chunks]))              # every worker warm
        t0 = time.perf_counter()
        res = list(ex.map(_worker_run, chunks))
        secs = time.perf_counter() - t0
    m = {k: np.concatenate([r[0][k] for r in res]) for k in res[0][0]}
    return m, np.concatenate([r[1] for r in res]), secs, len(chunks)


def simulate_numpy(stream, sc, s=0):
    """Scenario s, line by line in Python: the reference for the kernel (deterministic when svc_cv = 0 and
    detect is 0 or 1, the only cases where two RNGs need not agree)."""
    free = np.zeros(MAX_SERVERS)
    m = dict.fromkeys(OUT, 0.0)
    pres = present(stream)
    shift_h, H = sc.get("shift_h", SHIFT_H), sc["cap"].shape[1]
    for i in range(len(stream["t_h"])):
        t, y = float(stream["t_h"][i]), int(stream["y"][i])
        sh = min(int((t - math.floor(t / WEEK_H) * WEEK_H) / shift_h), H - 1)
        if stream["score"][i] < sc["thr"][s, stream["fam"][i] * H + sh]:
            m["escaped"] += y
            continue
        m["flagged"] += 1
        c = int(sc["cap"][s, sh])
        if c <= 0:
            m["bypassed"] += 1
            m["escaped"] += y
            continue
        k = int(np.argmin(free[:c]))
        start = max(t, free[k])
        wait = start - t
        if wait > sc["max_wait"][s]:
            m["bypassed"] += 1
            m["escaped"] += y
            continue
        svc = float(sc["svc_mean"][s])
        free[k] = start + svc
        m["inspected"] += 1
        m["busy_h"] += svc
        m["wait_sum_h"] += wait
        m["wait_max_h"] = max(m["wait_max_h"], wait)
        if y:
            if pres[i] and sc["detect"][s] >= 1.0:
                m["caught"] += 1
                m["work_saved"] += float(stream["saved_work"][i])
            else:
                m["escaped"] += 1
        else:
            m["false_alarms"] += 1
    return m


def present(stream):
    """1 where a failing part's defect exists at the checkpoint; all ones if the data does not say."""
    p = stream.get("present")
    return np.ones(len(stream["y"]), np.int32) if p is None else np.asarray(p, np.int32)


def wait_quantile(hist, max_wait_h, q=0.95):
    """Per-scenario wait quantile (hours) from the kernel's histogram (bin upper edges)."""
    c = np.cumsum(hist, axis=1)
    tot = c[:, -1:]
    k = np.argmax(c >= q * np.maximum(tot, 1), axis=1)
    return np.where(tot[:, 0] > 0, (k + 1) * np.asarray(max_wait_h) / WAIT_BINS, 0.0)        # nothing inspected: 0
