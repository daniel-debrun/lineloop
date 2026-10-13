"""python -m lineloop <command>: the framework's terminal front end. The stages, in run order:

  0 Load        doctor                       which NVIDIA accelerators this machine has, and each step's fallback
                line show | simulate | ingest   a line spec: what Line Loop understood; a synthetic history of it;
                                             its data (events tables or the Bosch CSVs) -> the cache
  1 Predict     quality cv | fit | parity    XGBoost failure model: cross-validation (shuffled or forward in time),
                        | sweep              the deployable model (trained before t1, checked after), cuDF parity
  2 Score live  edge compile | bench         the model (+ any signal stage) -> one ONNX program -> TensorRT; latency
                     | replay                the edge node scoring the line's event stream live, after t1
  3 Decide      plan                         MILP (cuOpt | HiGHS) checked by the Warp twin; judged on the hold-out
  4 Stress-test twin                         the plan under thousands of drawn economics; GPU vs CPU timing
  5 Place       place                        server or edge node for the checkpoint model, as a monthly cost
                solvers                      cuOpt vs HiGHS on the plan's MILP grown to a plant
                demo --line <spec>           all of the above, in order, with a summary scoreboard
                report [--collect]           results/SCOREBOARD.md from the run summaries

Data commands take --line <spec or name> (its cache is the spec's data.cache) or --cache <folder>.
Every command prints a scoreboard and writes JSON next to it (--out).
See docs/HOW_IT_WORKS.md for every stage in depth.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console(highlight=False)
REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"


def _cache(a):
    """The cache folder a command works on: --cache, or the cache of --line's spec."""
    if getattr(a, "cache", None):
        return Path(a.cache)
    if getattr(a, "line", None):
        from . import line as L
        return L.read(a.line).cache
    raise SystemExit("give --line <spec or name> (see lines/) or --cache <folder>")


def table(title, cols, rows, caption=None):
    t = Table(title=title, caption=caption, title_justify="left", caption_justify="left")
    for c in cols:
        t.add_column(c, justify="left" if c == cols[0] else "right")
    for r in rows:
        t.add_row(*[str(x) for x in r])
    console.print(t)


# ---------------------------------------------------------------- doctor
def cmd_doctor(a):
    from . import accel
    table("accelerators", ["component", "library", "version", "runs on"], accel.report(),
          caption="'gpu' = the NVIDIA path runs here; otherwise the CPU fallback is used")


# ---------------------------------------------------------------- line
def cmd_line(a):
    from . import line as L
    if a.action == "show":
        if a.line:
            ln = L.read(a.line)
            if (ln.cache / L.LINE_FILE).exists() and not a.spec_only:
                ln = L.load(ln.cache)
        else:
            ln = L.load(_cache(a))
        return _show(ln)
    ln = L.read(a.line) if a.line else None
    if ln is None:
        raise SystemExit("give --line <spec or name>")
    if a.action == "simulate":
        from .data import simulate
        simulate.write(ln, a.out, parts=a.parts, seed=a.seed, log=console.print)
    elif a.action == "ingest":
        cache = Path(a.cache) if a.cache else ln.cache
        _ingest(ln, Path(a.src) if a.src else ln.source, cache, a.limit, a.keep_parquet)
        _show(L.load(cache))


def _ingest(ln, src, cache, limit=None, keep_parquet=False):
    """The line's data, through its format's adapter, into the cache (with the resolved line.json)."""
    if ln.data["format"] == "bosch":
        from .data import bosch
        bosch.ingest(ln, src, cache, limit=limit, log=console.print, keep_parquet=keep_parquet)
    else:
        from .data import events
        events.ingest(ln, src, cache, limit=limit, log=console.print)


def _show(ln):
    from .quality import features as F
    rel = lambda p: str(Path(p).relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p)  # noqa: E731
    rows = [("description", ln.description or "-"), ("data", f"{ln.data.get('format')} format, {rel(ln.source)}"),
            ("clock", f"1 unit = {ln.unit_hours:g} h, steps of {ln.resolution:g} "
                      f"({ln.resolution * ln.unit_hours * 60:g} min)"),
            ("shifts", f"{ln.shift_hours:g} h: {ln.n_shifts} slots a week")]
    for g, sts in ln.segments.items():
        rows.append((f"segment {g}", f"{len(sts)} station{'s' * (len(sts) != 1)}: {', '.join(sts[:6])}"
                     f"{' ...' if len(sts) > 6 else ''}"))
    if not ln.segments:
        rows.append(("stations", "read from the data at ingest"))
    for name, st in ln.checkpoints.items():
        rows.append((f"checkpoint {name}", f"at {st}: {len(ln.before(name))} stations before it" if ln.stations
                     else f"at {st}"))
    if ln.stations:
        rows.append(("recipes", ", ".join(F.recipes(ln))))
        for name in ln.checkpoints:
            fam = ln.family_blocks(name)
            rows.append((f"families at {name}", ", ".join(f"{k} ({len(v)})" for k, v in fam.items()) or "-"))
    if ln.group_stations is not None:
        rows.append(("group stations", f"{len(ln.group_stations)} (visited by >= {ln.group_min_share:.0%} of parts)"))
    if ln.measurements:
        rows.append(("measurements", f"{len(ln.measurements)} columns"))
    if ln.economics:
        rows.append(("economics", ", ".join(f"{k} {v:g}" for k, v in ln.economics.items() if not isinstance(v, dict))))
        for cp, over in ((k, v) for k, v in ln.economics.items() if isinstance(v, dict)):
            rows.append((f"economics at {cp}", ", ".join(f"{k} {v:g}" for k, v in over.items())))
    for sg in ln.signal_specs():
        rows.append((f"signal {sg.name}", f"at {', '.join(sg.stations)}: {len(sg.channels)} channels "
                     f"({', '.join(sg.channels)}) x {sg.samples} samples x {sg.per_part} per part = "
                     f"{sg.bytes_per_part / 1e3:,.0f} kB raw; {len(sg.feature_names())} features"))
    for cp, e in ln.edge.items():
        rows.append((f"edge at {cp}", ", ".join(f"{k} {v}" for k, v in e.items())))
    table(f"line: {ln.name}", ["", ""], rows, caption=f"spec: {rel(ln.spec)}" if ln.spec else None)


# ---------------------------------------------------------------- quality
def _features(a, xp):
    from .data import cache as C
    from .quality import features as F
    t = time.perf_counter()
    st, num, ln = C.load(_cache(a), xp)
    r = F.recipe(a.recipe, ln)
    t_load = time.perf_counter() - t
    t = time.perf_counter()
    ds = F.build(st, num, xp, ln, **r)
    X, y, ids, names = ds
    nbr = ds.nbr
    del st, num
    t_feat = time.perf_counter() - t
    dropped = []
    if getattr(a, "drop", None):
        X, names, nbr, dropped = F.drop(X, names, nbr, a.drop)
        console.print(f"ablation: dropped {len(dropped)} features matching {a.drop!r}")
    if a.sample < 1.0:
        import numpy as np
        from .quality.train import _take
        idx = np.where(np.random.default_rng(a.seed).random(len(y)) < a.sample)[0]
        X, y, ids, nbr = _take(X, idx), y[idx], ids[idx], F.subset_neighbours(nbr, idx)
    console.print(f"[bold]{a.recipe}[/]: {X.shape[0]:,} parts x {X.shape[1]} features "
                  f"({xp.__name__}: load {t_load:.1f} s, features {t_feat:.1f} s)")
    return X, y, ids, names, nbr, {"load_s": round(t_load, 2), "features_s": round(t_feat, 2),
                                   "frames": xp.__name__, "line": ln.name, **r,
                                   **({"drop": a.drop, "dropped": dropped} if dropped else {})}


def cmd_quality(a):
    from . import accel
    from .quality import train
    xp = accel.frame_lib(a.frames)
    device = accel.xgb_device(a.device)
    out = Path(a.out or RUNS / "quality" / a.recipe)
    from .quality.features import ALIASES
    a.recipe = ALIASES.get(a.recipe, a.recipe)
    params = json.loads(a.params) if a.params else None
    if a.action == "parity":
        return _parity(a)
    X, y, ids, names, nbr, info = _features(a, xp)
    if a.action == "sweep":
        grid = json.loads(Path(a.grid).read_text())
        rows = []
        for k, g in enumerate(grid):
            prm = {kk: v for kk, v in g.items() if kk not in ("rounds", "bags")}
            rounds = g.get("rounds", a.rounds)
            console.print(f"[bold]config {k}[/]: {g}")
            s, oof, fold = train.cv(X, y, k=a.folds, seed=a.seed, device=device, params=prm, rounds=rounds,
                                    names=names, nbr=nbr, bags=g.get("bags", a.bags), log=console.print,
                                    split=a.split)
            s.update({"recipe": a.recipe, "config": g, **info, "versions": _versions()})
            train.save(out / f"config{k}", s, ids, y, oof, fold)
            rows.append((k, json.dumps(g), f"{s['auc']:.4f}", f"{s['mcc_best']:.4f}", f"{s['mcc_honest']:.4f}",
                         s["train_s_total"]))
        table(f"sweep: {a.recipe}", ["#", "config", "AUC", "MCC best", "MCC honest", "train s"], rows)
        return
    if a.action == "cv":
        s, oof, fold = train.cv(X, y, k=a.folds, seed=a.seed, device=device, params=params, rounds=a.rounds,
                                names=names, nbr=nbr, bags=a.bags, log=console.print, split=a.split)
        s.update({"recipe": a.recipe, "sample": a.sample, **info, "versions": _versions()})
        train.save(out, s, ids, y, oof, fold)
        lo, hi = s["mcc_best_ci95"]
        fwd = a.split == "time"
        table(f"cross-validated failure prediction: {a.recipe}" + (" (forward in time)" if fwd else ""),
              ["metric", "value"], [
            ("parts / failures" + (" scored" if fwd else ""), f"{s['scored_parts']:,} / {s['scored_failures']:,}"),
            ("features", s["features"]),
            ("ROC AUC", f"{s['auc']:.4f}"),
            ("MCC, best threshold (pooled out-of-fold)", f"{s['mcc_best']:.4f}  [{lo:.4f}, {hi:.4f}]"),
            ("MCC, threshold from " + ("earlier blocks" if fwd else "other folds"), f"{s['mcc_honest']:.4f}"),
            ("threshold", f"{s['threshold']:.4f}"),
            ("training time (all folds)", f"{s['train_s_total']} s on {device}"),
            ("feature build", f"{info['features_s']} s on {info['frames']}"),
        ], caption=f"written to {out}/summary.json")
    elif a.action == "fit":
        rows = None
        if a.until is not None:
            import numpy as np
            rows = np.where(nbr["decide"] < a.until)[0]
            info["until"] = a.until
            info["train_parts"] = int(len(rows))
            console.print(f"training on the {len(rows):,} parts decided before t={a.until}")
        bst, secs = train.fit(X, y, device=device, params=params, rounds=a.rounds, names=names, nbr=nbr, rows=rows)
        out.mkdir(parents=True, exist_ok=True)
        bst.save_model(out / "model.json")
        from . import line as L
        from .edge.compile import write_signal_manifest
        write_signal_manifest(out, L.load(_cache(a)), names)
        cvs = out / "summary.json"
        thr = json.loads(cvs.read_text())["threshold"] if cvs.exists() else None
        card = {"recipe": a.recipe, **info, "features": names, "rounds": a.rounds, "device": device,
                "fit_s": round(secs, 1), "threshold": thr, "versions": _versions()}
        if rows is not None:                     # forward check: parts decided after the cut-off, never trained on
            import numpy as np
            import xgboost as xgb
            from .quality import mcc
            later = np.where(nbr["decide"] >= a.until)[0]
            p = bst.predict(xgb.DMatrix(train._take(X, later), feature_names=names))
            m, t = mcc.best(y[later], p)
            card["forward"] = {"parts": int(len(later)), "failures": int(y[later].sum()), "auc": mcc.auc(y[later], p),
                               "mcc_best": m, "threshold_best": t}
            table(f"forward check: {a.recipe}, trained before t={a.until}", ["metric", "value"], [
                ("parts after the cut-off / failures", f"{len(later):,} / {int(y[later].sum()):,}"),
                ("ROC AUC", f"{card['forward']['auc']:.4f}"),
                ("MCC at its best threshold (optimistic)", f"{m:.4f} at {t:.3f}")])
        (out / "model_card.json").write_text(json.dumps(card, indent=1))
        console.print(f"model written to {out}/model.json ({secs:.1f} s on {device}); threshold {thr}")


def _parity(a):
    """Build the same recipe with pandas and cudf; report time and the largest difference."""
    import numpy as np
    import pandas as pd
    from . import accel
    if not accel.has("cudf"):
        console.print("[red]cuDF is not available here; parity needs a GPU[/]")
        return 1
    import cudf
    res = {}
    for xp in (cudf, pd):
        X, y, ids, names, nbr, info = _features(a, xp)
        res[xp.__name__] = (X.get() if xp is cudf else X, names, info)
        del X
    (A, na, ip), (B, nb, ic) = res["pandas"], res["cudf"]
    same_cols = na == nb
    col_diff, col_nan = np.zeros(A.shape[1]), np.zeros(A.shape[1], dtype=np.int64)
    for j0 in range(0, A.shape[1], 64):                       # column chunks: no full-size temporaries
        ca, cb = A[:, j0:j0 + 64], B[:, j0:j0 + 64]
        col_nan[j0:j0 + 64] = (np.isnan(ca) != np.isnan(cb)).sum(axis=0)
        with np.errstate(invalid="ignore"):
            col_diff[j0:j0 + 64] = np.nan_to_num(np.nanmax(np.abs(ca - cb), axis=0, initial=0.0))
    nan_mismatch, diff = int(col_nan.sum()), float(col_diff.max(initial=0.0))
    worst = [f"{na[j]} ({col_diff[j]:.3g})" for j in np.argsort(-col_diff)[:5] if col_diff[j] > 0]
    bad_nan = [f"{na[j]} ({col_nan[j]})" for j in np.argsort(-col_nan)[:5] if col_nan[j] > 0]
    rows = [("columns identical", same_cols), ("cells", f"{A.size:,}"), ("missing-value mismatches", nan_mismatch),
            ("max abs difference", f"{diff:.3g}"), ("pandas feature build", f"{ip['features_s']} s"),
            ("cuDF feature build", f"{ic['features_s']} s"),
            ("speed-up", f"{ip['features_s'] / max(ic['features_s'], 1e-9):.1f}x"),
            ("largest differences in", ", ".join(worst) or "-"), ("missing mismatches in", ", ".join(bad_nan) or "-")]
    table(f"cuDF vs pandas feature parity: {a.recipe}", ["check", "value"], rows)
    out = Path(a.out or RUNS / "quality" / f"parity-{a.recipe}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "parity.json").write_text(json.dumps({"recipe": a.recipe, "same_columns": same_cols, "cells": int(A.size),
                                                 "nan_mismatch": nan_mismatch, "max_abs_diff": diff,
                                                 "worst_columns": worst, "nan_mismatch_columns": bad_nan,
                                                 "pandas": ip, "cudf": ic, "versions": _versions()}, indent=1))
    return 0 if same_cols and nan_mismatch == 0 and diff < 1e-4 else 1


def _versions():
    from . import accel
    out = {}
    for d in ("xgboost", "cudf", "pandas", "numpy", "cuopt", "warp-lang", "tensorrt", "highspy"):
        v = accel._ver(d)
        if v:
            out[d] = v
    g = accel.gpu()
    out["gpu"] = g["name"] if g else None
    return out


# ---------------------------------------------------------------- edge
def _probe_rows(model_dir, n=4096, seed=0):
    """Rows that exercise the trees: each value is one of that feature's split thresholds (+-), or missing."""
    import numpy as np
    from .edge import compile as C
    trees, _, n_feat, _ = C.trees_from_xgboost(Path(model_dir) / "model.json")
    rng = np.random.default_rng(seed)
    cuts = [[] for _ in range(n_feat)]
    for left, _, sidx, cond, _, _ in trees:
        for k in np.where(left != -1)[0]:
            cuts[sidx[k]].append(cond[k])
    X = rng.normal(size=(n, n_feat)).astype(np.float32)
    for j, c in enumerate(cuts):
        if c:
            c = np.array(c, np.float32)
            X[:, j] = rng.choice(c, n) * np.where(rng.random(n) < 0.5, 1.0005, 0.9995)
    X[rng.random(X.shape) < 0.3] = np.nan
    return X


def _probe_signals(model_dir, n, seed=0):
    """Raw signal blocks for n probe rows (random weld qualities, simulated), or None if the model has no signal
    stage."""
    import numpy as np
    from .data import simulate
    from .edge.compile import model_signal
    ms = model_signal(model_dir)
    if ms is None:
        return None
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, ms[0].per_part)) + rng.random((n, 1)) * 2.0
    return simulate.waveforms(ms[0], z, rng.integers(0, 2 ** 62, n))


def cmd_edge(a):
    import numpy as np
    from .edge import compile as C
    from .edge import scorers
    md = Path(a.model)
    if a.action == "compile":
        man = C.compile_model(md / "model.json", md)
        sig = man["signal"]
        X = _probe_rows(md, 512 if sig else 4096)
        S = _probe_signals(md, len(X))
        ref = scorers.make("xgb-cpu", md)(X, S)
        rows = [("trees x depth", f"{man['trees']} x {man['depth']}"), ("features", man["features"]),
                ("model.onnx", f"{man['onnx_bytes'] / 1e6:.1f} MB")]
        if sig:
            rows.append(("signal stage in the graph", f"{sig['name']}: raw {sig['shape']} "
                         f"({sig['bytes_per_part'] / 1e3:,.0f} kB per part) -> {sig['features']} features"))
        for b in [x for x in ("ort-cpu", "trt") if x in scorers.available()]:
            p = scorers.make(b, md)(X, S)
            d = float(np.max(np.abs(p - ref)))
            man[f"max_abs_diff_{b}"] = d
            man[f"share_above_1e-4_{b}"] = float(np.mean(np.abs(p - ref) > 1e-4))
            rows.append((f"{b} vs XGBoost, max |dp| on {len(X)} boundary rows", f"{d:.2e}"
                         + (f" ({man[f'share_above_1e-4_{b}']:.1%} of rows above 1e-4)" if sig else "")))
        (md / "compile.json").write_text(json.dumps(man, indent=1))
        table("edge compile: XGBoost -> ONNX tensor program", ["item", "value"], rows)
    elif a.action == "bench":
        has_sig = C.model_signal(md) is not None
        X = _probe_rows(md, 1024 if has_sig else 8192)
        S = _probe_signals(md, len(X))
        backs = scorers.available() if a.backends == "all" else a.backends.split(",")
        ref = scorers.make("xgb-cpu", md, a.threads)(X[:1024], None if S is None else S[:1024])
        res, rows = {}, []
        for b in backs:
            sc = scorers.make(b, md, a.threads) if b in ("xgb-cpu", "ort-cpu") else scorers.make(b, md)
            if hasattr(sc, "tune"):
                sc.tune(X)
            diff = float(np.max(np.abs(sc(X[:1024], None if S is None else S[:1024]) - ref)))
            r = scorers.bench(sc, X, batch_sizes=tuple(int(x) for x in a.batches.split(",")), seconds=a.seconds,
                              S=S)
            res[b] = {"max_abs_diff_vs_xgb_cpu": diff, "runs": r}
            for x in r:
                rows.append((b, x["batch"], f"{x['p50_ms']:.3f}", f"{x['p99_ms']:.3f}", f"{x['parts_per_s']:,.0f}",
                             f"{diff:.1e}"))
        table("edge inference: latency per call and throughput", ["backend", "batch", "p50 ms", "p99 ms",
                                                                    "parts/s", "max |dp|"], rows,
              caption="host in, host out (copies included); max |dp| against XGBoost on the CPU"
                      + ("; every call includes the raw signal block and its features" if has_sig else ""))
        out = Path(a.out or RUNS / "edge" / "bench")
        out.mkdir(parents=True, exist_ok=True)
        (out / "bench.json").write_text(json.dumps({"model": str(md), "backends": res, "versions": _versions(),
                                                    "cpu_threads": a.threads, "signal": has_sig}, indent=1))
    elif a.action == "replay":
        return _replay(a, md)


def _replay(a, md):
    import time

    import numpy as np
    import pandas as pd
    from .data import cache as C
    from .edge import scorers
    from .edge.stream import Stream
    from .quality import features as F
    from .quality import mcc
    card = json.loads((md / "model_card.json").read_text())
    st, num, ln = C.load(_cache(a), pd)
    if card.get("line", ln.name) != ln.name:
        raise SystemExit(f"the model was trained on line {card['line']!r}, the cache holds {ln.name!r}")
    r = F.recipe(card["recipe"], ln)
    if not r["causal"]:
        raise SystemExit(f"recipe {card['recipe']!r} uses the future; a live node needs 'live' or a checkpoint")
    from .edge.compile import model_signal
    ms = model_signal(md)
    raw_fn = None
    if ms is not None:
        from .data import simulate
        raw_fn = simulate.raw_source(ln, _cache(a), ms[0])
    t0 = time.perf_counter()
    s = Stream(st, num, card["features"], ln, r["checkpoint"], t_from=a.t_from, log=console.print, raw_fn=raw_fn)
    t_warm = time.perf_counter() - t0
    sc = scorers.make(a.backend, md)
    console.print(f"edge node: {card['recipe']} model, {a.backend}; warm state before t={a.t_from} loaded in "
                  f"{t_warm:.1f} s; replaying {len(s.events):,} events")
    t0 = time.perf_counter()
    ids, t, M = s.replay(scorer=sc, t_until=a.t_until if a.t_until is not None else np.inf, log=console.print)
    wall = time.perf_counter() - t0
    p = M[:, -1]
    y = s.labels[np.searchsorted(s.node.labelled_ids, ids)]
    sig_info = {}
    if ms is not None:                      # the signal features the node computed from raw vs the history's
        import xgboost as xgb
        ref = xgb.Booster(model_file=str(md / "model.json")).inplace_predict(M[:, :-1].astype(np.float32))
        dp = np.abs(ref - p)
        sig_info = {"signal": ms[0].name, "raw_bytes_per_part": ms[0].bytes_per_part,
                    "max_abs_diff_vs_history_features": float(dp.max()),
                    "share_above_1e-4": float(np.mean(dp > 1e-4)), "raw_make_s": s.raw_s}
    out = Path(a.out or RUNS / "edge" / f"replay-{card['recipe']}")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": ids, "t": t, "p": p, "y": y}).to_parquet(out / "scored.parquet")
    ticks = s.ticks
    busy = ticks[ticks["decided"] > 0]
    m, thr = mcc.best(y, p)
    summ = {"model": str(md), "line": ln.name, "recipe": card["recipe"], "backend": a.backend, "t_from": a.t_from,
            "events": int(len(s.events)), "scored_parts": int(len(ids)), "failures": int(y.sum()),
            "auc": mcc.auc(y, p), "mcc_best": m, "threshold_best": thr, "warm_s": t_warm, "wall_s": wall,
            "ticks": int(len(busy)), "tick_ms_p50": float(np.median(busy["seconds"]) * 1e3),
            "tick_ms_p99": float(np.percentile(busy["seconds"], 99) * 1e3),
            "parts_per_tick_p50": float(np.median(busy["decided"])),
            "events_per_s": len(s.events) / (wall - s.raw_s), **sig_info, "versions": _versions()}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    table(f"edge replay: {card['recipe']} from t={a.t_from} ({a.backend})", ["metric", "value"], [
        ("events replayed", f"{summ['events']:,} ({summ['events_per_s']:,.0f} events/s)"),
        ("parts scored / failures", f"{len(ids):,} / {int(y.sum()):,}"),
        ("ROC AUC (live, forward in time)", f"{summ['auc']:.4f}"),
        ("MCC at its best threshold (optimistic)", f"{m:.4f}"),
        ("decision latency per tick, p50 / p99", f"{summ['tick_ms_p50']:.2f} / {summ['tick_ms_p99']:.2f} ms"),
        ("parts decided per busy tick (p50)", f"{summ['parts_per_tick_p50']:.0f}")] + ([
        ("raw signal per part (computed on the node)", f"{ms[0].bytes_per_part / 1e3:,.0f} kB of {ms[0].name}"),
        ("score vs the model on the history's signal features",
         f"max |dp| {sig_info['max_abs_diff_vs_history_features']:.1e}, "
         f"{sig_info['share_above_1e-4']:.2%} above 1e-4")] if ms else []),
        caption=f"a tick is one step of the line's clock ({ln.resolution * ln.unit_hours * 60:g} min): the "
                f"node's budget per tick is that long, it uses ms")
    return 0


# ---------------------------------------------------------------- plan (optimise + twin)
def cmd_plan(a):
    import numpy as np
    import pandas as pd
    from . import accel
    from .optimize import inspection as I
    from .twin import data as D
    from .twin import inspection as T
    cache = _cache(a)
    ln, cp_name, cp = _checkpoint_of(a, cache)
    econ = _economics(a, ln, cp_name)
    solver, device = accel.mip_solver(a.solver), accel.warp_device(a.device)
    scored = pd.read_parquet(a.scored)
    st = _station_starts(cache)
    split = a.split or ",".join(str(x) for x in _splits(cache, cp))
    t1, t2 = (float(x) * ln.unit_hours for x in split.split(","))
    fit_on = (scored["t"].to_numpy() * ln.unit_hours >= t1) & (scored["t"].to_numpy() * ln.unit_hours < t2)
    from .data import cache as C
    d, fams = D.prepare(scored, st, ln, cp, k=a.families, fit_on=fit_on, origins=C.origins(cache))
    nf = len(fams)
    cal, hold = D.window(d, t1, t2), D.window(d, t2, np.inf)
    console.print(f"calibration: {len(cal['y']):,} parts, {cal['y'].sum()} failures; hold-out: {len(hold['y']):,} "
                  f"parts, {hold['y'].sum()} failures; families: {', '.join(fams)}")
    if "present" in d and C.origins(cache) is not None:
        f = d["y"] == 1
        console.print(f"defects already present at {cp_name}: {d['present'][f].mean():.0%} of failures (from the "
                      f"data's origins); an inspection there cannot find the rest")
    else:
        console.print("[dim]no defect origins in the data: every failure counts as findable at the checkpoint[/]")
    console.print(f"[bold]simulation-optimisation[/] ({solver} MILP + Warp twin on {device}):")
    pol, hist = I.optimise(cal, nf, econ, solver=solver, device=device, rho=a.rho, target_bypass=a.target_bypass,
                           reps=a.reps, log=console.print, estimate=a.estimate)
    # baseline: the best single threshold + constant staffing, found by brute force in the twin on calibration
    H = T.n_shifts(econ["shift_h"])
    shares = [q for q in I.SHARES if q > 0]
    cand = {"none": {"thr": np.float32(I.NEVER), "cap": np.zeros(H, int)}}
    for q in shares:
        thr = float(np.quantile(cal["score"], 1 - q))
        for c in range(1, 5):
            cand[f"top {q:.1%}, {c}/shift"] = {"thr": np.float32(thr), "cap": np.full(H, c)}
    ev_c, secs_c = I.evaluate(cal, nf, cand, econ, a.reps, device)
    # the same rule the optimiser obeys: a policy whose cell moves on more than target_bypass of its flagged parts
    # uninspected does not count (else the baseline buys savings with a queue the line would not accept)
    ok = [k for k in ev_c if ev_c[k]["bypass_rate"] <= a.target_bypass] or list(ev_c)
    best_name = min(ok, key=lambda k: ev_c[k]["cost_per_week"])
    greedy = I.greedy(cal, nf, econ, cand[best_name], reps=max(a.reps // 2, 8), device=device,
                      target_bypass=a.target_bypass)
    policies = {"today (no early inspection)": {"thr": np.float32(I.NEVER), "cap": np.zeros(H, int)},
                f"best constant policy ({best_name})": cand[best_name],
                "best per-family policy (greedy search in the twin)": greedy,
                "optimised (MILP + twin)": pol}
    ev, secs_h = I.evaluate(hold, nf, policies, econ, a.reps, device)
    rows = []
    for k, e in ev.items():
        rows.append((k, f"{e['cost_per_week']:,.0f} ± {e['cost_sd']:,.0f}", f"{e['inspector_hours_per_week']:.0f}",
                     f"{e['inspected'] / e['weeks']:,.0f}", f"{e['caught'] / e['weeks']:.1f} / "
                     f"{e['defects_present'] / e['weeks']:.1f} / {e['defects'] / e['weeks']:.1f}",
                     f"{e['work_saved'] / e['weeks']:,.0f}",
                     f"{e['bypass_rate']:.1%}", f"{e['wait_p95_h'] * 60:.0f}"))
    table(f"hold-out weeks: early inspection at {cp_name} (per week, mean ± sd over seeds)",
          ["policy", "net cost", "inspector h", "inspections", "defects caught / present / all", "visits saved",
           "bypass", "p95 wait min"], rows,
          caption="net cost against today's line; negative = saving. Economics are assumptions (flags), the parts, "
                  "times, scores and outcomes are real.")
    n_sc = len(cand) * a.reps
    timing = {"candidates_scenarios": n_sc, "parts": int(len(cal["y"])), "seconds": secs_c, "device": device}
    sweep = []
    if a.sweep_value:
        console.print("[bold]break-even[/]: the optimised policy for each value of a defect caught early "
                      "(calibrated, then run on the hold-out weeks)")
        for v in (float(x) for x in a.sweep_value.split(",")):
            e2 = {**econ, "value_per_defect": v}
            p2, h2 = I.optimise(cal, nf, e2, solver=solver, device=device, rho=a.rho, target_bypass=a.target_bypass,
                                reps=a.reps, log=lambda *_: None, estimate=a.estimate)
            r2 = I.evaluate(hold, nf, {"p": p2}, e2, a.reps, device)[0]["p"]
            sweep.append({"value_per_defect": v, "iterations": len(h2), **{k: r2[k] for k in (
                "cost_per_week", "cost_sd", "inspector_hours_per_week", "inspected", "caught", "defects", "weeks",
                "work_saved", "bypass_rate")}})
        table("break-even: optimised policy on the hold-out weeks, by value of a defect caught early",
              ["value / defect", "net cost / week", "inspector h", "inspections", "defects caught / all",
               "visits saved", "bypass"],
              [(f"{r['value_per_defect']:,.0f}", f"{r['cost_per_week']:,.0f} ± {r['cost_sd']:,.0f}",
                f"{r['inspector_hours_per_week']:.0f}", f"{r['inspected'] / r['weeks']:,.0f}",
                f"{r['caught'] / r['weeks']:.1f} / {r['defects'] / r['weeks']:.1f}",
                f"{r['work_saved'] / r['weeks']:,.0f}", f"{r['bypass_rate']:.1%}") for r in sweep])
    out = Path(a.out or RUNS / "plan")
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps({
        "line": ln.name, "checkpoint": cp_name, "economics": econ, "solver": solver, "device": device,
        "families": fams, "split_ticks": split,
        "loop": hist, "policy": {k: np.asarray(v).tolist() for k, v in pol.items()},
        "policies": {k: {"thr": np.asarray(v["thr"]).tolist(), "cap": np.asarray(v["cap"]).tolist()}
                     for k, v in policies.items()},
        "holdout": {k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in e.items()}
                    for k, e in ev.items()},
        "brute_force": timing, "break_even": sweep, "versions": _versions()}, indent=1, default=float))
    console.print(f"twin: {n_sc:,} scenarios x {len(cal['y']):,} parts in {secs_c:.2f} s on {device}")
    return 0


def _station_starts(cache):
    import pandas as pd
    from .data import cache as C
    return pd.read_parquet(Path(cache) / C.STATIONS,
                           columns=["Id"] + [c for c in C.station_columns(cache) if c.endswith("_t0")])


def _checkpoint_of(a, cache):
    """(line, checkpoint name, station) for plan / twin: --checkpoint, else the recipe of the replay that wrote
    --scored, else the line's first checkpoint."""
    from . import line as L
    ln = L.load(cache)
    name = getattr(a, "checkpoint", None)
    summ = Path(a.scored).parent / "summary.json"
    if name is None and summ.exists():
        name = json.loads(summ.read_text()).get("recipe")
    if name is None and ln.checkpoints:
        name = next(iter(ln.checkpoints))
    if name not in ln.checkpoints:
        raise SystemExit(f"plan and twin need a checkpoint model (one of {list(ln.checkpoints)}), not {name!r}")
    return ln, name, ln.checkpoints[name]


def _economics(a, ln, checkpoint=None):
    """ECONOMICS defaults <- the line spec's [economics] <- its [economics.<checkpoint>] and the shift length <-
    command-line flags."""
    from .optimize import inspection as I
    flat = {k: v for k, v in ln.economics.items() if not isinstance(v, dict)}
    over = ln.economics.get(checkpoint, {}) if checkpoint else {}
    sub = {k for k, v in ln.economics.items() if isinstance(v, dict)}
    unknown = (set(flat) | set(over)) - set(I.ECONOMICS)
    if unknown or sub - set(ln.checkpoints):
        raise SystemExit(f"unknown [economics] keys in the line spec: {sorted(unknown | (sub - set(ln.checkpoints)))};"
                         f" known: {list(I.ECONOMICS)} and [economics.<checkpoint>] tables")
    econ = {**I.ECONOMICS, **{k: float(v) for k, v in {**flat, **over}.items()}, "shift_h": ln.shift_hours}
    for k in econ:
        v = getattr(a, k, None)
        if v is not None:
            econ[k] = v
    return econ


# ---------------------------------------------------------------- solvers
def cmd_solvers(a):
    """cuOpt against HiGHS on the inspection MILP grown from one line to a plant sharing one inspection crew."""
    import pandas as pd
    from .data import cache as C
    from .optimize import scale
    from .twin import data as D
    cache = _cache(a)
    ln, cp_name, cp = _checkpoint_of(a, cache)
    econ = _economics(a, ln, cp_name)
    plan = json.loads(Path(a.plan).read_text())
    scored = pd.read_parquet(a.scored)
    t = scored["t"].to_numpy() * ln.unit_hours
    t1, t2 = (float(x) * ln.unit_hours for x in plan["split_ticks"].split(","))
    d, fams = D.prepare(scored, _station_starts(cache), ln, cp, k=len(plan["families"]) - 1,
                        fit_on=(t >= t1) & (t < t2), origins=C.origins(cache))
    cal = D.window(d, t1, t2)
    sizes = tuple(int(x) for x in a.sizes.split(","))
    console.print(f"[bold]the inspection MILP at {cp_name}, grown to a plant[/] (time limit {a.time_limit:g} s):")
    res = scale.run(cal, len(fams), econ, sizes, a.solvers.split(",") if a.solvers else None, a.time_limit,
                    log=console.print)
    solvers = list(res[0]["solvers"])
    rows = []
    for r in res:
        cells = []
        for sname in solvers:
            v = r["solvers"][sname]
            gap = v["gap_to_best"]
            cells.append(f"{v['seconds']:.2f} s, {v['status']}" + ("" if gap is None or gap < 1e-6 else
                                                                    f", +{gap:.2%}"))
        rows.append((f"{r['lines']}", f"{r['variables']:,}", *cells))
    table("solver time by plant size (same model, same limit)", ["lines", "variables", *solvers], rows,
          caption="+x% = objective worse than the best found at that size")
    out = Path(a.out or RUNS / "solvers")
    out.mkdir(parents=True, exist_ok=True)
    (out / "solvers.json").write_text(json.dumps({"line": ln.name, "checkpoint": cp_name, "time_limit": a.time_limit,
                                                  "sizes": res, "versions": _versions()}, indent=1, default=float))
    return 0


# ---------------------------------------------------------------- place
def cmd_place(a):
    """Server or edge node for a checkpoint's model: the line's real decision stream against the network, the
    deadline and the cost of a missed decision; compute measured here."""
    import numpy as np
    import pandas as pd
    from . import line as L
    from .data import cache as C
    from .edge import place as PL
    from .edge import scorers
    from .edge.compile import model_signal
    from .edge.stream import events_from
    md = Path(a.model)
    card = json.loads((md / "model_card.json").read_text())
    cache = _cache(a)
    ln = L.load(cache)
    cp = a.checkpoint or card["recipe"]
    if cp not in ln.checkpoints:
        raise SystemExit(f"place needs a checkpoint model, not {cp!r}")
    c = PL.config(ln, cp)
    st = pd.read_parquet(Path(cache) / C.STATIONS)
    P, _ = events_from(st, ln, ln.checkpoints[cp])
    t_from = card.get("until", -np.inf)
    dec = np.sort(P["decide"].dropna().to_numpy())
    dec = dec[dec >= t_from]
    step_s = ln.resolution * ln.unit_hours * 3600.0          # times are on the clock's grid: spread within a step
    arr = np.sort(dec * ln.unit_hours * 3600.0 + np.random.default_rng(0).uniform(0, step_s, len(dec)))
    ms = model_signal(md)
    n_feat = len(card["features"])
    record = c["record_bytes"] or 4 * n_feat
    server_bytes = record + (ms[0].bytes_per_part if ms else 0)
    edge_up = 4 * n_feat + 16
    X = _probe_rows(md, 256)
    S = _probe_signals(md, len(X))
    avail = scorers.available()
    server_b = "trt" if "trt" in avail else "ort-cpu"
    lat = {}
    for name, b, th in ((f"server ({server_b})", server_b, None), ("edge, 1 CPU core (ort-cpu)", "ort-cpu", 1)):
        lat[name] = PL.latency(scorers.make(b, md, th) if th else scorers.make(b, md), X, S)
    if "trt" in avail:
        lat["edge, NVIDIA GPU (trt)"] = lat[f"server ({server_b})"]
    value = 0.0
    if a.plan and Path(a.plan).exists():
        h = json.loads(Path(a.plan).read_text())["holdout"]["optimised (MILP + twin)"]
        parts_week = len(dec) / max((dec[-1] - dec[0]) * ln.unit_hours / 168.0, 1e-9)
        value = max(0.0, -h["cost_per_week"]) / parts_week + 0.0          # + 0.0: no "-0" when it saves nothing
    srv, edg = lat[f"server ({server_b})"][0], lat["edge, 1 CPU core (ort-cpu)"][0]
    r = PL.assess(arr, c, server_bytes, edge_up, srv, edg, value)
    pl = r["placements"]
    table(f"where should the {cp} model run? {r['parts']:,} decisions over {r['months'] * 30:.0f} days of the "
          f"line", ["", "server", "edge node"], [
              ("data sent per part", f"{server_bytes / 1e3:,.1f} kB" + (" (raw signal + record)" if ms else ""),
               f"{edge_up / 1e3:,.1f} kB (features + decision)"),
              ("upstream per day", f"{pl['server']['upstream_mb_per_day']:,.1f} MB "
               f"({pl['server']['uplink_utilisation']:.2%} of the uplink)",
               f"{pl['edge']['upstream_mb_per_day']:,.1f} MB, when the network is up"),
              ("decision delay p50 / p99", f"{pl['server']['p50_s'] * 1e3:,.1f} ms / {pl['server']['p99_s']:,.2f} s",
               f"{pl['edge']['p50_s'] * 1e3:,.1f} ms / {pl['edge']['p99_s'] * 1e3:,.1f} ms"),
              (f"late (> {c['deadline_s']:g} s) per month", f"{pl['server']['late_per_month']:,.1f}",
               f"{pl['edge']['late_per_month']:,.1f}"),
              ("cost of late decisions / month", f"{pl['server']['miss_cost_per_month']:,.0f}",
               f"{pl['edge']['miss_cost_per_month']:,.0f}"),
              ("node / month", "0 (the plant's server)", f"{c['edge_cost_per_month']:,.0f}"),
              ("total / month", f"{pl['server']['cost_per_month']:,.0f}", f"{pl['edge']['cost_per_month']:,.0f}")],
          caption=f"network: {c['uplink_mbps']:g} Mbit/s, round trip {c['rtt_ms'][0]:g} / {c['rtt_ms'][1]:g} ms "
                  f"(p50 / p99), {c['outages_per_month']:g} outages a month of ~{c['outage_minutes']:g} min; "
                  f"undecided parts: {c['undecided']}"
                  + (f" (downtime {c['downtime_per_hour']:,.0f} / h)" if c["undecided"] == "stop" else
                     f" (value per decision {value:,.2f}, from the plan)") + ". Assumptions from the line spec.")
    peak = np.max(np.bincount(((arr - arr[0]) // 60).astype(int))) / 60.0      # busiest minute, parts/s
    caps = []
    for name, (_, p50, p99) in lat.items():
        caps.append((name, f"{p50 * 1e3:.2f} / {p99 * 1e3:.2f}", f"{1 / p50:,.0f}", f"{1 / p50 / peak:,.0f}x"))
    table("compute per decision (one part per call, measured here)", ["backend", "p50 / p99 ms", "parts/s",
                                                                      "headroom over the busiest minute"], caps,
          caption="the server's compute is not what decides placement: " + (
              "one CPU core already keeps up with this line" if 1 / np.median(edg) > peak else
              "one CPU core does not keep up with this line: the edge node needs a GPU"))
    verdict = ("[bold]edge node[/]" if r["verdict"] == "edge" else "[bold]plant server[/]")
    console.print(f"verdict for {cp}: {verdict} is cheaper per month"
                  + (": " + "; ".join(r["reasons"]) if r["reasons"] else ": nothing forces it to the cell"))
    out = Path(a.out or RUNS / "place")
    out.mkdir(parents=True, exist_ok=True)
    (out / "place.json").write_text(json.dumps({
        "line": ln.name, "checkpoint": cp, **r, "server_bytes": server_bytes, "edge_up_bytes": edge_up,
        "signal": None if not ms else ms[0].name, "compute": {k: {"p50_ms": v[1] * 1e3, "p99_ms": v[2] * 1e3}
                                                              for k, v in lat.items()},
        "busiest_minute_parts_per_s": float(peak), "versions": _versions()}, indent=1, default=float))
    return 0


# ---------------------------------------------------------------- twin
def cmd_twin(a):
    """Robustness of the plan's policies to its assumptions: thousands of drawn economics in one Warp launch on the
    hold-out weeks; then the same scenarios timed on the GPU, one CPU core and every CPU core."""

    import numpy as np
    import pandas as pd
    from . import accel
    from .data import cache as C
    from .twin import data as D
    from .twin import inspection as T
    from .twin import robust as R
    cache = _cache(a)
    ln, cp_name, cp = _checkpoint_of(a, cache)
    econ = _economics(a, ln, cp_name)
    plan_p = Path(a.plan or Path(a.scored).parent.parent / "plan" / "plan.json")
    if not plan_p.exists():
        raise SystemExit(f"no plan at {plan_p}: run `lineloop plan` first (or give --plan)")
    plan = json.loads(plan_p.read_text())
    pols = {k: v for k, v in R.policies_from_plan(plan).items() if not k.startswith("today")}
    scored = pd.read_parquet(a.scored)
    t = scored["t"].to_numpy() * ln.unit_hours
    t1, t2 = (float(x) * ln.unit_hours for x in plan["split_ticks"].split(","))
    d, fams = D.prepare(scored, _station_starts(cache), ln, cp, k=len(plan["families"]) - 1,
                        fit_on=(t >= t1) & (t < t2), origins=C.origins(cache))
    hold = D.window(d, t2, np.inf)
    nf = len(fams)
    dev = accel.warp_device(a.device)
    n = a.scenarios or (8192 if dev.startswith("cuda") else 1024)
    res, draws, secs = R.run(hold, nf, pols, econ, n=n, ranges=ln.uncertainty, device=dev)
    rows = []
    for k, r in res.items():
        top = sorted(r["sensitivity"].items(), key=lambda kv: -abs(kv[1]))[:3]
        be = r["break_even_value_per_defect_median"]
        rows.append((k, f"{r['saves_share']:.0%}", f"{r['net_median']:,.0f}",
                     f"{r['net_p05']:,.0f} to {r['net_p95']:,.0f}", "-" if be is None else f"{be:,.0f}",
                     ", ".join(f"{kk} {v:+.2f}" for kk, v in top)))
    table(f"robustness on the hold-out weeks: {n:,} drawn economics per policy at {cp_name} ({dev})",
          ["policy", "saves money in", "net / week (median)", "5 % to 95 %", "break-even value / defect",
           "moved most by (rank corr.)"], rows,
          caption="every assumption drawn from its [uncertainty] range; same scenarios and seeds for every policy; "
                  "negative net = saving against today's line")
    # timing: the same scenarios on each device (a slice of them on the CPU)
    S = len(pols) * n
    grid = R.draw(econ, n, ln.uncertainty)
    thr = np.concatenate([np.broadcast_to(R._thr(v["thr"], nf, T.n_shifts(econ["shift_h"])),
                                          (n, nf, T.n_shifts(econ["shift_h"]))) for v in pols.values()])
    cap = np.concatenate([np.broadcast_to(np.broadcast_to(v["cap"], (T.n_shifts(econ["shift_h"]),)),
                                          (n, T.n_shifts(econ["shift_h"]))) for v in pols.values()])
    sc = T.scenarios(nf, thr, cap, np.tile(grid["svc_mean_h"], len(pols)), np.tile(grid["svc_cv"], len(pols)),
                     np.tile(grid["detect"], len(pols)), np.tile(grid["max_wait_h"], len(pols)),
                     shift_h=econ["shift_h"])
    parts = len(hold["y"])
    timing, trows = {}, []

    def record(name, n_sc, secs, note=""):
        timing[name] = {"scenarios": n_sc, "seconds": secs, "part_steps_per_s": n_sc * parts / secs}
        trows.append((name, f"{n_sc:,}", f"{secs:.3f}", f"{n_sc * parts / secs:,.0f}", note))

    def tiled(n_sc):
        """n_sc scenarios spread evenly over the robustness set (every policy in its share; repeated if n_sc is
        larger), so the GPU and the CPU time the same mix of work."""
        idx = np.round(np.linspace(0, S - 1, n_sc)).astype(int) if n_sc <= S else np.arange(n_sc) % S
        return {kk: (v[idx] if np.ndim(v) else v) for kk, v in sc.items()}

    def sized(run, n0, target_s):
        """Grow the scenario count until one timed run takes at least target_s (short runs time overheads)."""
        n_sc, secs = n0, run(n0)
        while secs < target_s and n_sc < 64 * S:
            n_sc = int(min(n_sc * max(2.0, 1.2 * target_s / max(secs, 1e-6)), 64 * S))
            secs = run(n_sc)
        return n_sc, secs

    if accel.has("warp") and dev.startswith("cuda"):
        T.simulate(hold, _first(sc, 1), dev)
        n_g, s_g = sized(lambda k: T.simulate(hold, tiled(k), dev)[2], S, a.time_s / 4)
        record("GPU (" + dev + ")", n_g, s_g)
    T.simulate(hold, _first(sc, 1), "cpu")
    n_c, s_c = sized(lambda k: T.simulate(hold, tiled(k), "cpu")[2], max(8, int(S * a.cpu_fraction)), a.time_s)
    record("CPU, 1 core", n_c, s_c, "Warp's CPU backend runs a launch on one thread")
    import os
    n_cpu = os.cpu_count() or 1
    tries = {p: T.simulate_cpu_cores(hold, tiled(n_c), procs=p)[2] for p in sorted({n_cpu, max(n_cpu // 2, 1)})}
    procs = min(tries, key=tries.get)                  # with SMT, one process per physical core is often faster
    record(f"CPU, {procs} processes", n_c, tries[procs], "scenarios split; the faster of "
           + " and ".join(f"{p} ({t:.2f} s)" for p, t in tries.items()) + " processes")
    table("line twin throughput: the same kernel, the same scenarios", ["device", "scenarios", "seconds",
                                                                       "part-steps/s", ""], trows,
          caption=f"{parts:,} hold-out parts per scenario")
    names = list(timing)
    if names[0].startswith("GPU"):
        g = timing[names[0]]["part_steps_per_s"]
        sp = {k: g / v["part_steps_per_s"] for k, v in timing.items() if not k.startswith("GPU")}
        console.print("GPU speed-up: " + ", ".join(f"{v:,.1f}x against {k}" for k, v in sp.items()))
        timing["speedup"] = sp
    out = Path(a.out or RUNS / "twin")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps({
        "line": ln.name, "checkpoint": cp_name, "window": "hold-out", "parts": parts, "scenarios_per_policy": n,
        "ranges": {**R.RANGES, **ln.uncertainty}, "robustness": res, "devices": timing, "versions": _versions()},
        indent=1, default=float))
    return 0


def _first(sc, n):
    """The first n scenarios (scalars such as shift_h are shared)."""
    import numpy as np
    return {k: (v[:n] if np.ndim(v) else v) for k, v in sc.items()}


# ---------------------------------------------------------------- demo
def _splits(cache, checkpoint, q=(0.7, 0.85)):
    """Train / calibration / hold-out cut-offs: quantiles of the parts' decision times at the checkpoint."""
    import numpy as np
    import pandas as pd
    from . import line as L
    from .data import cache as C
    from .quality import features as F
    ln = L.load(cache)
    st = pd.read_parquet(Path(cache) / C.STATIONS, columns=["Id", "is_train", "date_hash"] +
                         [c for c in C.station_columns(cache) if c.endswith("_t0") or c.endswith("_t1")])
    P = F.part_table(st, F.stations(ln, st.columns, checkpoint), pd, ln, ln.checkpoint_station(checkpoint))
    nd = -int(np.floor(np.log10(ln.resolution * 10)))           # on a grid of ten clock steps
    return [round(float(np.nanquantile(P["decide"], x)), nd) for x in q]


def cmd_demo(a):
    """The whole loop on one line, one command: data -> cuDF features -> XGBoost (CUDA) -> TensorRT edge node ->
    Warp twin x cuOpt policy -> hold-out scoreboard. A line without data but with a [simulate] table is simulated
    first, so any spec runs on a laptop."""
    from . import accel
    from . import line as L
    from .edge import scorers
    if not (a.synthetic or a.line):
        raise SystemExit("give --line <spec or name>, e.g. --line battery-module (see lines/)")
    ln = L.read("bosch-synthetic" if a.synthetic else a.line)
    runs = Path(a.out or RUNS / "demo" / ln.name)
    cache = Path(a.cache) if a.cache else ln.cache

    def step(stage, title):
        console.rule(f"[bold]{stage}[/]  {title}")

    step("0 · Load", f"the line: {ln.name}")
    from .data import simulate
    has_data = ln.source.exists() and any(ln.source.iterdir())
    if ln.simulate and has_data and not simulate.is_current(ln, ln.source, a.parts):
        console.print(f"[yellow]{ln.source} was simulated from another version of the spec (or another --parts): "
                      f"simulating it again[/]")
        has_data = False
    stamp, built = ln.source / "simulated.txt", cache / "simulated.txt"
    stale = stamp.exists() and (not built.exists() or built.read_text() != stamp.read_text())
    if not has_data or stale or not (cache / L.LINE_FILE).exists():
        if not has_data:
            if not ln.simulate:
                raise SystemExit(f"no data for line {ln.name!r} in {ln.source}, and the spec has no [simulate] table")
            simulate.write(ln, parts=a.parts, log=console.print)
        _ingest(ln, ln.source, cache)
        if stamp.exists():                       # the cache now matches this simulated source
            built.write_text(stamp.read_text())
    ln = L.load(cache)
    _show(ln)
    from .data import cache as C
    org = C.origins(cache)
    if org:
        import pandas as pd
        share = pd.Series(list(org.values())).value_counts(normalize=True)
        console.print("failures by the station where the defect arose (simulated ground truth): " + ", ".join(
            f"{k} {v:.0%}" for k, v in share.items()))
    if not ln.checkpoints:
        raise SystemExit(f"line {ln.name!r} has no checkpoints: add one to its spec to run the live loop")
    cp = next(iter(ln.checkpoints))
    rounds = str(a.rounds or (200 if ln.simulate else 800))     # simulated lines are small: fewer rounds
    c = ["--cache", str(cache)]
    t1, t2 = _splits(cache, cp)
    console.print("[bold]run order[/]: 1 Predict -> 2 Score live -> 3 Decide -> 4 Stress-test -> 5 Place, at "
                  f"checkpoint {cp} ({ln.checkpoints[cp]})")
    console.print(f"[bold]time split[/] (decision time at {cp}): train t < {t1} · calibrate {t1} <= t < {t2} · "
                  f"hold-out t >= {t2}; nothing is tuned on the hold-out")
    step("0 · Load", "accelerators found here")
    main(["doctor"])
    if not a.skip_cv:
        step("1 · Predict", "benchmark: end-of-line failure prediction (XGBoost, 5-fold CV); a yardstick, not "
                            "deployable: by then the final test has the answer")
        main(["quality", "cv", "--recipe", "offline", *c, "--rounds", rounds, "--out", str(runs / "offline")])
    step("1 · Predict", f"the checkpoint model at {cp}: XGBoost on live features, trained on the train window, "
                        f"checked forward on everything after it")
    main(["quality", "fit", "--recipe", cp, *c, "--rounds", rounds, "--until", str(t1), "--out", str(runs / cp)])
    sig = [sg for sg in ln.signal_specs() if ln.position(sg.station) < ln.position(ln.checkpoints[cp])]
    if sig:
        console.print(f"[bold]the same model without the raw {sig[0].name} signals[/] (what a server that only "
                      f"has the factory system's records could learn):")
        main(["quality", "fit", "--recipe", cp, *c, "--rounds", rounds, "--until", str(t1),
              "--drop", f"^{sig[0].station}\\.{sig[0].name}\\.", "--out", str(runs / f"{cp}-no-signal")])
    step("2 · Score live", "compile: the trees (and any signal stage) -> one ONNX tensor program -> TensorRT")
    main(["edge", "compile", "--model", str(runs / cp)])
    main(["edge", "bench", "--model", str(runs / cp), "--batches", "1,64,1024", "--seconds", "1",
          "--out", str(runs / "edge-bench")])
    backend = next(b for b in ("trt", "fil", "ort-cpu", "xgb-cpu") if b in scorers.available())
    step("2 · Score live", f"the edge node replays the line's events after t = {t1}, scoring each part as it "
                           f"arrives ({backend})")
    main(["edge", "replay", "--model", str(runs / cp), *c, "--t-from", str(t1), "--backend", backend,
          "--out", str(runs / "replay")])
    step("3 · Decide", f"the inspection policy: {accel.mip_solver()} MILP checked by the Warp twin on the "
                       f"calibration window, judged on the hold-out")
    main(["plan", "--scored", str(runs / "replay" / "scored.parquet"), *c, "--split", f"{t1},{t2}",
          "--reps", str(a.reps), "--sweep-value", "50,100,150,300,600,900", "--out", str(runs / "plan")])
    step("4 · Stress-test", "the twin replays the plan under thousands of drawn costs and cell parameters")
    main(["twin", "--scored", str(runs / "replay" / "scored.parquet"), *c, "--plan", str(runs / "plan" / "plan.json"),
          "--out", str(runs / "twin")])
    step("5 · Place", f"should the {cp} model run on a plant server or at the cell?")
    main(["place", "--model", str(runs / cp), *c, "--plan", str(runs / "plan" / "plan.json"),
          "--out", str(runs / "place")])
    _summary(runs, ln, cp)


def _summary(runs, ln, cp):
    def load(p):
        p = runs / p
        return json.loads(p.read_text()) if p.exists() else None
    cv, card, rep, plan, tw, pl = (load("offline/summary.json"), load(f"{cp}/model_card.json"),
                                   load("replay/summary.json"), load("plan/plan.json"), load("twin/sweep.json"),
                                   load("place/place.json"))
    rows = []
    if cv:
        rows.append(("1 Predict · benchmark, end of line (shuffled CV)",
                     f"MCC {cv['mcc_best']:.4f}, AUC {cv['auc']:.4f}"))
    if card and card.get("forward"):
        f = card["forward"]
        rows.append((f"1 Predict · {cp} model, forward in time", f"AUC {f['auc']:.4f} on {f['parts']:,} later parts"))
        ns = load(f"{cp}-no-signal/model_card.json")
        if ns and ns.get("forward"):
            rows.append(("1 Predict · ... without the raw signals", f"AUC {ns['forward']['auc']:.4f}: the signals add "
                                                            f"{f['auc'] - ns['forward']['auc']:+.4f}"))
    if rep:
        rows.append(("2 Score live · edge node", f"{rep['events_per_s']:,.0f} events/s, "
                                                 f"{rep['tick_ms_p50']:.2f} ms per tick (p50) on {rep['backend']}"))
    if plan:
        h = plan["holdout"]
        opt = h["optimised (MILP + twin)"]
        base = [k for k in h if k.startswith("best constant")][0]
        gr = h.get("best per-family policy (greedy search in the twin)")
        rows.append(("3 Decide · policy on hold-out weeks", f"net {opt['cost_per_week']:,.0f}/week vs "
                     f"{h[base]['cost_per_week']:,.0f} (best constant)"
                     + (f", {gr['cost_per_week']:,.0f} (greedy per family)" if gr else "")
                     + f" and 0 (today); caught {opt['caught'] / opt['weeks']:.1f} of "
                     f"{opt.get('defects_present', opt['defects']) / opt['weeks']:.1f} findable defects/week"))
    if tw:
        r = tw["robustness"].get("optimised (MILP + twin)")
        if r:
            rows.append(("4 Stress-test · under drawn economics", f"saves money in {r['saves_share']:.0%} of "
                         f"{r['scenarios']:,} scenarios; median net {r['net_median']:,.0f}/week"))
        d = tw["devices"]
        rows.append(("4 Stress-test · twin throughput", ", ".join(f"{k}: {v['part_steps_per_s']:,.0f} part-steps/s"
                                                        for k, v in d.items() if "part_steps_per_s" in v)))
    if pl:
        rows.append((f"5 Place · the {cp} model", f"{pl['verdict']}: " + ("; ".join(pl["reasons"]) or
                                                                         "nothing forces it to the cell")))
    table(f"Line Loop demo: {ln.name}", ["stage", "result"], rows, caption=f"outputs in {runs}")


# ---------------------------------------------------------------- report
def cmd_report(a):
    from . import report
    if a.collect:
        got = report.collect(a.runs, a.results)
        console.print(f"collected {len(got)} summaries into {a.results}: {', '.join(got)}")
    secs = report.sections(a.results)
    for title, cols, rows, note in secs:
        table(title, cols, rows, caption=note)
    (Path(a.results) / "SCOREBOARD.md").write_text(report.markdown(secs))
    console.print(f"written {Path(a.results) / 'SCOREBOARD.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lineloop", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="accelerators and fallbacks").set_defaults(fn=cmd_doctor)

    def line_args(p):
        p.add_argument("--line", default=None, help="line spec: a .toml path or a name in lines/")
        p.add_argument("--cache", type=Path, default=None, help="cache folder (default: the spec's data.cache)")

    b = sub.add_parser("line", help="a line spec: show it, simulate a history of it, ingest its data")
    b.add_argument("action", choices=["show", "simulate", "ingest"])
    line_args(b)
    b.add_argument("--src", type=Path, default=None, help="ingest: data folder (default: the spec's data.source)")
    b.add_argument("--limit", type=int, default=None, help="ingest: parts (or rows per Bosch file), for smoke runs")
    b.add_argument("--keep-parquet", dest="keep_parquet", action="store_true",
                   help="ingest (bosch format): reuse the parquet files, only resolve the line again")
    b.add_argument("--out", type=Path, default=None, help="simulate: folder (default: the spec's data.source)")
    b.add_argument("--parts", type=int, default=None, help="simulate: parts (default: the spec's [simulate])")
    b.add_argument("--seed", type=int, default=None)
    b.add_argument("--spec-only", dest="spec_only", action="store_true",
                   help="show: the spec as written, even if a resolved line.json exists")
    b.set_defaults(fn=cmd_line)

    q = sub.add_parser("quality", help="failure prediction: cv, fit, parity")
    q.add_argument("action", choices=["cv", "fit", "parity", "sweep"])
    q.add_argument("--recipe", default="offline", help="offline | live | a checkpoint of the line")
    line_args(q)
    q.add_argument("--out", type=Path, default=None)
    q.add_argument("--folds", type=int, default=5)
    q.add_argument("--drop", default=None, help="ablation: leave out features whose name matches this regex")
    q.add_argument("--split", default="stratified", choices=["stratified", "time"],
                   help="cv folds: shuffled and stratified (as Kaggle CV), or forward in time (what a live model "
                        "can expect: train on the past, score the next block)")
    q.add_argument("--rounds", type=int, default=800)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--sample", type=float, default=1.0, help="fraction of labelled parts (quick runs)")
    q.add_argument("--device", default="auto", help="auto | cuda | cpu")
    q.add_argument("--frames", default="auto", help="auto | cudf | pandas")
    q.add_argument("--params", default=None, help="JSON overrides of the XGBoost parameters")
    q.add_argument("--bags", type=int, default=1, help="cv: models per fold, differing only in seed, averaged")
    q.add_argument("--until", type=float, default=None, help="fit: only parts decided before this time")
    q.add_argument("--grid", default=None, help="sweep: JSON file, a list of parameter overrides (may set rounds)")
    q.set_defaults(fn=cmd_quality)

    e = sub.add_parser("edge", help="edge node: compile, bench, replay")
    e.add_argument("action", choices=["compile", "bench", "replay"])
    e.add_argument("--model", type=Path, required=True, help="a quality fit output folder (model.json)")
    e.add_argument("--backends", default="all")
    e.add_argument("--backend", default="xgb-cpu", help="replay: scorer")
    e.add_argument("--batches", default="1,16,256,4096")
    e.add_argument("--seconds", type=float, default=2.0)
    e.add_argument("--threads", type=int, default=None)
    line_args(e)
    e.add_argument("--t-from", dest="t_from", type=float, default=None, help="replay: warm state before this time")
    e.add_argument("--t-until", dest="t_until", type=float, default=None)
    e.add_argument("--out", type=Path, default=None)
    e.set_defaults(fn=cmd_edge)

    pl = sub.add_parser("plan", help="optimise the inspection policy (cuOpt MILP + Warp twin) and test it")
    pl.add_argument("--scored", type=Path, required=True, help="edge replay output (scored.parquet)")
    line_args(pl)
    pl.add_argument("--checkpoint", default=None, help="default: the replayed model's recipe")
    pl.add_argument("--split", default=None, help="calibration from,to (data time units); hold-out after "
                                                  "(default: the 70 %% and 85 %% quantiles of decision times)")
    pl.add_argument("--families", type=int, default=6)
    pl.add_argument("--solver", default="auto")
    pl.add_argument("--device", default="auto")
    pl.add_argument("--rho", type=float, default=0.85)
    pl.add_argument("--target-bypass", dest="target_bypass", type=float, default=0.02)
    pl.add_argument("--reps", type=int, default=32)
    pl.add_argument("--estimate", default="expected", choices=["expected", "realised"],
                    help="defects per cell from the model's probabilities (default) or the window's counts")
    pl.add_argument("--sweep-value", dest="sweep_value", default=None,
                    help="comma list of value_per_defect: optimise and test a policy for each")
    pl.add_argument("--out", type=Path, default=None)
    from .optimize.inspection import ECONOMICS
    for k, v in ECONOMICS.items():
        pl.add_argument(f"--{k.replace('_', '-')}", dest=k, type=float, default=None,
                        help=f"default: the line spec's [economics], else {v}")
    pl.set_defaults(fn=cmd_plan)

    tw = sub.add_parser("twin", help="robustness of the plan to its assumptions; GPU vs CPU timing")
    line_args(tw)
    tw.add_argument("--scored", required=True, help="the edge replay's scored.parquet")
    tw.add_argument("--plan", default=None, help="plan.json (default: ../plan/plan.json next to the replay)")
    tw.add_argument("--checkpoint", default=None)
    tw.add_argument("--scenarios", type=int, default=None, help="per policy (default 8192 on a GPU, 1024 on CPU)")
    tw.add_argument("--device", default="auto")
    tw.add_argument("--cpu-fraction", dest="cpu_fraction", type=float, default=1 / 64,
                    help="share of the scenarios to start the CPU timing from (grown until it is long enough)")
    tw.add_argument("--time-s", dest="time_s", type=float, default=4.0,
                    help="each CPU timing runs at least this long (the GPU's a quarter of it)")
    tw.add_argument("--out", type=Path, default=None)
    tw.set_defaults(fn=cmd_twin)

    sv = sub.add_parser("solvers", help="cuOpt vs HiGHS on the inspection MILP grown to a plant")
    line_args(sv)
    sv.add_argument("--scored", required=True, help="the edge replay's scored.parquet")
    sv.add_argument("--plan", required=True, help="plan.json (its calibration window and families)")
    sv.add_argument("--checkpoint", default=None)
    sv.add_argument("--sizes", default="1,4,16,64,256", help="lines in the plant")
    sv.add_argument("--solvers", default=None, help="default: every installed one")
    sv.add_argument("--time-limit", dest="time_limit", type=float, default=120.0)
    sv.add_argument("--out", type=Path, default=None)
    sv.set_defaults(fn=cmd_solvers)

    pc = sub.add_parser("place", help="server or edge node for a checkpoint model: deadline, network, outages, cost")
    line_args(pc)
    pc.add_argument("--model", required=True, help="the checkpoint model's folder (quality fit + edge compile)")
    pc.add_argument("--plan", default=None, help="plan.json, for the value of a decision (undecided = pass)")
    pc.add_argument("--checkpoint", default=None, help="default: the model's recipe")
    pc.add_argument("--out", type=Path, default=None)
    pc.set_defaults(fn=cmd_place)

    dm = sub.add_parser("demo", help="the whole loop on one line (simulated first if it has no data)")
    line_args(dm)
    dm.add_argument("--synthetic", action="store_true", help="same as --line bosch-synthetic")
    dm.add_argument("--parts", type=int, default=None, help="parts to simulate (default: the spec's)")
    dm.add_argument("--rounds", type=int, default=None, help="boosting rounds (default 200 simulated, 800 real)")
    dm.add_argument("--reps", type=int, default=32)
    dm.add_argument("--skip-cv", dest="skip_cv", action="store_true")
    dm.add_argument("--out", type=Path, default=None, help="default: runs/demo/<line>")
    dm.set_defaults(fn=cmd_demo)

    rp = sub.add_parser("report", help="render results/SCOREBOARD.md (and --collect summaries from runs/)")
    rp.add_argument("--collect", action="store_true")
    rp.add_argument("--runs", type=Path, default=RUNS)
    rp.add_argument("--results", type=Path, default=REPO / "results")
    rp.set_defaults(fn=cmd_report)

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
