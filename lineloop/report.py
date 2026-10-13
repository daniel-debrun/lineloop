"""python -m lineloop report: collect run summaries into results/ and render the scoreboard.

Only aggregate numbers are copied (the Bosch data and anything per part stay in runs/ and data/, never in git):
out-of-fold scores, scored parts and models are left behind. The scoreboard is rendered from these JSON files, so
every number in results/SCOREBOARD.md and the README is program output.
"""
import json
import shutil
from pathlib import Path

COLLECT = {   # results/<target> <- runs/<source>
    "quality/cv-kaggle.json": "quality/v5/config0/summary.json",
    "quality/cv-causal.json": "quality/causal/summary.json",
    "quality/cv-l3-entry.json": "quality/l3-entry/summary.json",
    "quality/parity-kaggle.json": "quality/parity-kaggle/parity.json",
    "quality/parity-l3-entry.json": "quality/parity-l3-entry/parity.json",
    "edge/model_card.json": "edge/l3-entry/model_card.json",
    "edge/compile.json": "edge/l3-entry/compile.json",
    "edge/bench.json": "edge/bench/bench.json",
    "edge/replay.json": "edge/replay/summary.json",
    "plan/plan.json": "plan/plan.json",
    "twin/sweep.json": "twin/sweep.json",
    "quality/cv-causal-forward.json": "quality/causal-forward/summary.json",
    "quality/cv-l3-entry-forward.json": "quality/l3-entry-forward/summary.json",
    "quality/ablation-no-id.json": "ablation/l3-entry-no-id/model_card.json",
    "quality/ablation-no-order.json": "ablation/l3-entry-no-order/model_card.json",
    "place/place.json": "place/place.json",
    "solvers/solvers.json": "solvers/solvers.json",
}
DROP = {"features", "feature_names", "dropped"}          # long lists that add nothing to a scoreboard


DEMO = ("offline/summary.json", "{cp}/model_card.json", "{cp}-no-signal/model_card.json", "{cp}/compile.json",
        "edge-bench/bench.json", "replay/summary.json", "plan/plan.json", "twin/sweep.json", "place/place.json")


def _copy(s, d):
    j = json.loads(Path(s).read_text())
    if isinstance(j, dict):
        for k in [k for k in DROP & set(j) if isinstance(j[k], list)]:
            j[k] = f"{len(j[k])} names (see runs/)"
    d.parent.mkdir(parents=True, exist_ok=True)
    d.write_text(json.dumps(j, indent=1))


def collect(runs, results):
    """The Bosch study's summaries (COLLECT), and every simulated line's demo summaries (runs/demo/<line>/ ->
    results/<line>/): aggregate numbers only."""
    got = []
    for dst, src in COLLECT.items():
        s = Path(runs) / src
        if s.exists():
            _copy(s, Path(results) / dst)
            got.append(dst)
    for d in sorted((Path(runs) / "demo").glob("*")):
        cps = [p.parent.name for p in d.glob("*/model_card.json") if not p.parent.name.endswith("-no-signal")]
        for cp in cps[:1]:
            for rel in DEMO:
                s = d / rel.format(cp=cp)
                if s.exists():
                    _copy(s, Path(results) / d.name / rel.format(cp=cp))
                    got.append(f"{d.name}/{rel.format(cp=cp)}")
    return got


def _load(results, p):
    p = Path(results) / p
    return json.loads(p.read_text()) if p.exists() else None


def sections(results):
    """[(title, columns, rows, note)] for the scoreboard."""
    R = results
    out = []
    lb = _load(R, "quality/leaderboard.json")
    cv = {k: _load(R, f"quality/cv-{k}.json") for k in ("kaggle", "causal", "l3-entry")}
    if cv["kaggle"] and lb:
        at = lb["at_rank"]
        k = cv["kaggle"]
        rows = [("Line Loop, competition recipe (5-fold CV)", f"{k['mcc_best']:.4f}",
                 f"[{k['mcc_best_ci95'][0]:.4f}, {k['mcc_best_ci95'][1]:.4f}]", f"{k['mcc_honest']:.4f}",
                 f"{k['auc']:.4f}")]
        for r in ("1", "10", "12", "25", "50", "68", "137"):
            tag = {"12": " (gold cut)", "68": " (silver cut)", "137": " (bronze cut)"}.get(r, "")
            rows.append((f"public leaderboard #{r}{tag}", f"{at[r]:.4f}", "", "", ""))
        rows.sort(key=lambda x: -float(x[1]))
        out.append(("Bosch failure prediction against the Kaggle leaderboard (MCC)",
                    ["entry", "MCC", "95 % CI", "MCC, threshold from other folds", "AUC"], rows,
                    "Line Loop numbers are cross-validation on the 1,183,747 labelled parts, not a leaderboard "
                    "submission (no test-set measurements were used or submitted). Leaderboard: public board, "
                    f"{lb['teams']} teams, downloaded {lb['downloaded']}. Selection history: results/quality/runs.md."))
    rows = []
    fwd = {k: _load(R, f"quality/cv-{k}-forward.json") for k in ("causal", "l3-entry")}
    for name, label, c in (("kaggle", "competition: whole route, any neighbour part", cv["kaggle"]),
                           ("causal", "live, end of line: only what was known when the part finished", cv["causal"]),
                           ("causal", "  the same, forward in time", fwd["causal"]),
                           ("l3-entry", "live, before line 3: own data up to line 3 + live context", cv["l3-entry"]),
                           ("l3-entry", "  the same, forward in time", fwd["l3-entry"])):
        if c:
            f = c.get("split") == "time"
            rows.append((label, "forward: 5 blocks" if f else "shuffled, 5 folds", f"{c['auc']:.4f}",
                         f"{c['mcc_best']:.4f}", f"[{c['mcc_best_ci95'][0]:.4f}, {c['mcc_best_ci95'][1]:.4f}]",
                         f"{c['mcc_honest']:.4f}"))
    if rows:
        out.append(("What going live costs: shuffled folds against forward in time",
                    ["recipe", "folds", "AUC", "MCC (best threshold)", "95 % CI",
                     "MCC, threshold from other (shuffled) / earlier (forward) folds"], rows,
                    "Shuffled folds train on parts made after the ones they score, which no live model can; forward "
                    "folds score each block of time with a model trained on the blocks before it. Causal recipes "
                    "are reproduced bit for bit by the streaming edge node (tests/test_edge_stream.py)."))
    card = _load(R, "edge/model_card.json")
    abl = [(n, _load(R, f"quality/ablation-{k}.json")) for n, k in (
        ("without the part-number features (_id, Id-neighbour I_)", "no-id"),
        ("without any production-order or neighbour feature (S_, E_, I_, start_rank, nbr_ysum)", "no-order"))]
    if card and card.get("forward") and any(a for _, a in abl):
        rows = [("all features", card["forward"]["parts"], f"{card['forward']['auc']:.4f}", "")]
        for n, a in abl:
            if a and a.get("forward"):
                rows.append((n, a["forward"]["parts"], f"{a['forward']['auc']:.4f}",
                             f"{a['forward']['auc'] - card['forward']['auc']:+.4f}"))
        out.append(("Does the line-3 model read the process, or the part order? (forward in time)",
                    ["l3-entry model", "later parts", "AUC", "change"], rows,
                    "Trained on parts decided before t = 1171.0, scored on every later part; only the features "
                    "differ."))
    rows = []
    for name in ("kaggle", "l3-entry"):
        par = _load(R, f"quality/parity-{name}.json")
        if par:
            cells = f"{par['cells']:,}" if "cells" in par else "1,379,065,255 (v1 features)"
            rows.append((par["recipe"], cells, f"{par['nan_mismatch']} / {par['max_abs_diff']:g}",
                         f"{par['pandas']['features_s']:.1f} s", f"{par['cudf']['features_s']:.1f} s",
                         f"{par['pandas']['features_s'] / par['cudf']['features_s']:.1f}x"))
    if rows:
        out.append(("RAPIDS cuDF vs pandas: the same feature matrix, built on GPU and CPU",
                    ["recipe", "cells compared", "missing mismatches / max abs diff", "pandas", "cuDF", "speed-up"],
                    rows, "Builds include sorting 2.37 M parts in three orders, group counts and the float32 matrix "
                          "fill; the competition recipe was checked on its first (v1) feature set."))
    comp, bench = _load(R, "edge/compile.json"), _load(R, "edge/bench.json")
    if bench:
        rows = []
        for b, v in bench["backends"].items():
            for r in v["runs"]:
                rows.append((b, r["batch"], f"{r['p50_ms']:.3f}", f"{r['p99_ms']:.3f}", f"{r['parts_per_s']:,.0f}",
                             f"{v['max_abs_diff_vs_xgb_cpu']:.1e}"))
        note = None
        if comp:
            note = (f"Model: {comp['trees']} trees of depth {comp['depth']}, {comp['features']} features, compiled "
                    f"to a {comp['onnx_bytes'] / 1e6:.1f} MB ONNX tensor program. Timings include host<->device "
                    "copies. GPU: " + str(bench["versions"].get("gpu")))
        out.append(("Edge inference backends (the l3-entry model)",
                    ["backend", "batch", "p50 ms", "p99 ms", "parts/s", "max |dp| vs XGBoost CPU"], rows, note))
    rep = _load(R, "edge/replay.json")
    if rep:
        out.append(("Edge node replaying the line, forward in time", ["metric", "value"], [
            ("events replayed (starts, QC results, decisions)", f"{rep['events']:,}"),
            ("labelled parts scored live / failures", f"{rep['scored_parts']:,} / {rep['failures']:,}"),
            ("AUC / MCC (best threshold) on parts after the training cut-off",
             f"{rep['auc']:.4f} / {rep['mcc_best']:.4f}"),
            ("throughput", f"{rep['events_per_s']:,.0f} events/s ({rep['backend']})"),
            ("decision latency per tick p50 / p99", f"{rep['tick_ms_p50']:.2f} / {rep['tick_ms_p99']:.2f} ms")],
            "A tick is 0.01 Bosch time units (about 6 min); the node spends milliseconds of it."))
    plan = _load(R, "plan/plan.json")
    if plan:
        rows = []
        for k, e in plan["holdout"].items():
            rows.append((k, f"{e['cost_per_week']:,.0f} ± {e['cost_sd']:,.0f}", f"{e['inspector_hours_per_week']:.0f}",
                         f"{e['inspected'] / e['weeks']:,.0f}", f"{e['caught'] / e['weeks']:.1f} / "
                         f"{e.get('defects_present', e['defects']) / e['weeks']:.1f} / {e['defects'] / e['weeks']:.1f}",
                         f"{e['work_saved'] / e['weeks']:,.0f}", f"{e['bypass_rate']:.1%}"))
        loop = plan["loop"]
        note = (f"{plan['solver']} MILP ({loop[-1]['size']['vars']} variables, {loop[-1]['size']['rows']} rows) x Warp "
                f"twin on {plan['device']}, {len(loop)} loop iteration(s). Economics (assumed): "
                + ", ".join(f"{k} {v}" for k, v in plan["economics"].items()))
        out.append(("Early inspection before line 3, hold-out weeks (per week)",
                    ["policy", "net cost", "inspector h", "inspections", "defects caught / findable / all",
                     "L3 visits saved", "bypass"], rows, note))
        if plan.get("break_even"):
            out.append(("Break-even: optimised policy by value of a defect caught early (hold-out weeks)",
                        ["value / defect", "net cost / week", "inspector h", "defects caught / all"],
                        [(f"{r['value_per_defect']:,.0f}", f"{r['cost_per_week']:,.0f} ± {r['cost_sd']:,.0f}",
                          f"{r['inspector_hours_per_week']:.0f}",
                          f"{r['caught'] / r['weeks']:.1f} / {r['defects'] / r['weeks']:.1f}")
                         for r in plan["break_even"]], None))
    tw = _load(R, "twin/sweep.json")
    if tw and "robustness" in tw:
        rows = []
        for k, r in tw["robustness"].items():
            be = r["break_even_value_per_defect_median"]
            top = sorted(r["sensitivity"].items(), key=lambda kv: -abs(kv[1]))[:2]
            rows.append((k, f"{r['saves_share']:.0%}", f"{r['net_median']:,.0f}",
                         f"{r['net_p05']:,.0f} to {r['net_p95']:,.0f}", "-" if be is None else f"{be:,.0f}",
                         ", ".join(f"{kk} {v:+.2f}" for kk, v in top)))
        out.append((f"Does the plan survive its assumptions? {tw['scenarios_per_policy']:,} drawn economics per policy "
                    "(hold-out weeks)", ["policy", "saves money in", "net / week, median", "5 % to 95 %",
                                         "break-even value / defect", "moved most by"], rows,
                    "Every assumption drawn log-uniformly from its range (lineloop.twin.robust.RANGES); the same "
                    "scenarios and seeds for every policy."))
    if tw:
        rows = [(d, f"{v['scenarios']:,}", f"{v['seconds']:.3f}",
                 f"{v.get('part_steps_per_s', v.get('scenario_parts_per_s')):,.0f}")
                for d, v in tw["devices"].items() if isinstance(v, dict) and "seconds" in v]
        sp = tw["devices"].get("speedup")
        out.append(("Line twin (NVIDIA Warp): one kernel launch, every scenario a thread",
                    ["device", "scenarios", "seconds", "part-steps/s"], rows,
                    ("GPU speed-up: " + ", ".join(f"{v:,.0f}x against {k}" for k, v in sp.items())) if
                    isinstance(sp, dict) else None))
    pl = _load(R, "place/place.json")
    if pl:
        a, b = pl["placements"]["server"], pl["placements"]["edge"]
        out.append((f"Where should the {pl['checkpoint']} model run? (lineloop place)", ["", "server", "edge node"], [
            ("data per part", f"{pl['server_bytes'] / 1e3:,.1f} kB", f"{pl['edge_up_bytes'] / 1e3:,.1f} kB"),
            ("decision delay p50 / p99", f"{a['p50_s'] * 1e3:,.1f} ms / {a['p99_s']:,.2f} s",
             f"{b['p50_s'] * 1e3:,.2f} ms / {b['p99_s'] * 1e3:,.2f} ms"),
            ("late decisions per month", f"{a['late_per_month']:,.1f}", f"{b['late_per_month']:,.1f}"),
            ("cost per month (late decisions + node)", f"{a['cost_per_month']:,.0f}", f"{b['cost_per_month']:,.0f}")],
            f"Verdict: {pl['verdict']}" + (": " + "; ".join(pl["reasons"]) if pl["reasons"] else "") +
            ". Network, deadline and costs are assumptions in the spec's [edge] table."))
    sv = _load(R, "solvers/solvers.json")
    if sv:
        names = list(sv["sizes"][0]["solvers"])
        rows = []
        for r in sv["sizes"]:
            cells = []
            for n in names:
                v = r["solvers"][n]
                g = v["gap_to_best"]
                cells.append(f"{v['seconds']:.2f} s, {v['status']}" + ("" if g is None or g < 1e-6 else f", +{g:.2%}"))
            rows.append((r["lines"], f"{r['variables']:,}", *cells))
        out.append(("cuOpt vs HiGHS: the inspection MILP grown from one line to a plant sharing a crew",
                    ["lines", "variables", *names], rows,
                    f"Same model, {sv['time_limit']:g} s limit; +x% = objective worse than the best found at that "
                    "size."))
    return out


def markdown(secs):
    lines = ["# Scoreboard", "", "Rendered by `python -m lineloop report` from the JSON files in this folder.", ""]
    for title, cols, rows, note in secs:
        lines += [f"## {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
        if note:
            lines += ["", note]
        lines.append("")
    return "\n".join(lines)


def copy_tree_json(src, dst):
    for p in Path(src).rglob("*.json"):
        t = Path(dst) / p.relative_to(src)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, t)
