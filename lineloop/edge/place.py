"""Where should a checkpoint's model run: on a plant server, or on an edge node at the cell?

Inference speed rarely decides it (a server's GPU is faster than any edge box). What decides it is what the
decision needs from the cell and what happens when the network does not deliver. For each placement the line's
real stream of decisions at the checkpoint (its hold-out weeks, from the edge replay) is played against:

  payload      what must travel to decide: for a server, the part's record and, if the model uses a signal,
               its raw block; an edge node only sends its features and decision upstream, when it can
  network      the cell's uplink (Mbit/s, first-come first-served), a round trip drawn from a log-normal with the
               spec's median and 99th percentile, and outages (Poisson, exponential length)
  compute      measured here: the compiled model scoring one part at a time on the server's backend and on
               the edge node's (and on one CPU core, the smallest edge box that would do)
  deadline     the time a decision has to reach the line (the spec's deadline_s)

A decision that misses its deadline costs, per the spec's [edge.<checkpoint>]:
  undecided = "pass"   the part goes on unchecked, as it would today: the policy's value per part is lost
                       (from the plan: its hold-out saving per part)
  undecided = "stop"   the line waits for a decision (an interlocked gate): downtime_per_hour while it waits
and an edge node costs edge_cost_per_month. The verdict is the cheaper placement per month; every figure it
used is printed with it. They are assumptions about a plant, so the verdict is a worked example of the
trade-off, and it flips when they change.
"""
import time

import numpy as np

DEFAULTS = {"deadline_s": 60.0, "uplink_mbps": 10.0, "rtt_ms": [4.0, 60.0], "outages_per_month": 2.0,
            "outage_minutes": 30.0, "undecided": "pass", "downtime_per_hour": 0.0, "edge_cost_per_month": 150.0,
            "record_bytes": None}
MONTH_S = 30 * 24 * 3600.0


def config(ln, checkpoint):
    c = {**DEFAULTS, **ln.edge.get(checkpoint, {})}
    unknown = set(c) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"[edge.{checkpoint}]: unknown keys {sorted(unknown)}; known: {list(DEFAULTS)}")
    if c["undecided"] not in ("pass", "stop"):
        raise ValueError(f"[edge.{checkpoint}].undecided must be 'pass' or 'stop'")
    return c


def latency(scorer, X, S=None, calls=300):
    """Seconds per decision, one part per call (as parts arrive): (samples, p50, p99)."""
    for k in range(5):
        scorer(X[k:k + 1], None if S is None else S[k:k + 1])
    out = []
    for k in range(calls):
        j = k % len(X)
        t0 = time.perf_counter()
        scorer(X[j:j + 1], None if S is None else S[j:j + 1])
        out.append(time.perf_counter() - t0)
    out = np.array(out)
    return out, float(np.median(out)), float(np.percentile(out, 99))


def _rtt(rng, n, med_ms, p99_ms):
    sig = max(np.log(p99_ms / med_ms) / 2.326, 1e-6)
    return rng.lognormal(np.log(med_ms / 1e3), sig, n)


def _outages(rng, t0, t1, per_month, minutes):
    k = rng.poisson(per_month * (t1 - t0) / MONTH_S)
    s = np.sort(rng.uniform(t0, t1, k))
    return s, s + rng.exponential(minutes * 60.0, k)


def simulate(arr_s, c, payload_bytes, compute_s, rng, server=True):
    """Decision delay per part (seconds after its arrival) for one draw of the network. Edge: compute only."""
    n = len(arr_s)
    comp = rng.choice(compute_s, n)
    if not server:
        return comp, 0
    tx = payload_bytes * 8 / (c["uplink_mbps"] * 1e6)
    o0, o1 = _outages(rng, arr_s[0], arr_s[-1], c["outages_per_month"], c["outage_minutes"])
    ready = arr_s.copy()
    if len(o0):                                  # a part made during an outage waits for the network to return
        k = np.searchsorted(o0, arr_s, "right") - 1
        inside = (k >= 0) & (arr_s < o1[np.clip(k, 0, None)])
        ready[inside] = o1[k[inside]]
    sent = np.empty(n)
    free = -np.inf
    for i in range(n):                           # the uplink: first come, first served
        free = max(free, ready[i]) + tx
        sent[i] = free
    done = sent + _rtt(rng, n, *c["rtt_ms"]) + comp
    return done - arr_s, len(o0)


def assess(arr_s, c, server_bytes, edge_up_bytes, server_compute, edge_compute, value_per_part, draws=64, seed=0):
    """Both placements over `draws` network histories: per placement, latency percentiles, late decisions and
    their cost per month, bytes upstream; and the verdict."""
    rng = np.random.default_rng(seed)
    span = arr_s[-1] - arr_s[0]
    months = span / MONTH_S
    busy_h = len(np.unique(np.floor(arr_s / 3600.0)))       # hours in which the line made parts
    run_share = busy_h * 3600.0 / max(span, 1.0)
    out = {}
    for name, server, payload, comp in (("server", True, server_bytes, server_compute),
                                        ("edge", False, 0.0, edge_compute)):
        lat, late, outs = [], [], []
        for _ in range(draws if server else 1):
            d, k = simulate(arr_s, c, payload, comp, rng, server)
            lat.append(d)
            late.append(np.mean(d > c["deadline_s"]))
            outs.append(k)
        lat = np.concatenate(lat)
        late_share = float(np.mean(late))
        late_month = late_share * len(arr_s) / months
        if c["undecided"] == "pass":
            cost = late_month * value_per_part
        else:                                     # the line waits out every outage it is running through
            down_h = (c["outages_per_month"] * c["outage_minutes"] / 60.0 * run_share) if server else 0.0
            cost = down_h * c["downtime_per_hour"]
        up = (payload if server else edge_up_bytes) * len(arr_s) / (span / 86400.0)
        out[name] = {"p50_s": float(np.median(lat)), "p99_s": float(np.percentile(lat, 99)),
                     "max_s": float(lat.max()), "late_share": late_share, "late_per_month": late_month,
                     "miss_cost_per_month": float(cost), "upstream_mb_per_day": up / 1e6,
                     "uplink_utilisation": float(server * payload * 8 * len(arr_s) / span / (c["uplink_mbps"] * 1e6))}
    out["edge"]["cost_per_month"] = out["edge"]["miss_cost_per_month"] + c["edge_cost_per_month"]
    out["server"]["cost_per_month"] = out["server"]["miss_cost_per_month"]
    better = "edge" if out["edge"]["cost_per_month"] < out["server"]["cost_per_month"] else "server"
    reasons = []
    s = out["server"]
    if s["p99_s"] > c["deadline_s"]:
        reasons.append(f"the server's p99 decision ({s['p99_s']:.1f} s) is past the {c['deadline_s']:g} s deadline")
    if s["uplink_utilisation"] > 0.5:
        reasons.append(f"the raw data would take {s['uplink_utilisation']:.0%} of the cell's uplink")
    if s["late_per_month"] > 0:
        what = (f"{s['late_per_month']:,.0f} parts a month pass unchecked" if c["undecided"] == "pass" else
                "the line stops while the network is down")
        reasons.append(f"outages: {what}, {s['miss_cost_per_month']:,.0f} / month")
    return {"verdict": better, "reasons": reasons, "placements": out, "months": months, "parts": int(len(arr_s)),
            "run_share": run_share, "value_per_part": value_per_part, "config": c}
