"""The README's figures, drawn from the committed results (results/*.json) and the line specs.

  python tools/figures.py            # -> docs/img/<name>.svg (needs matplotlib)

Every figure has a white background and black text, so it reads the same on a light or a dark page.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "docs" / "img"
R = REPO / "results"

MODES = {
    "light": {"bg": "#ffffff", "card": "#f6f8fa", "border": "#c9d1d9", "ink": "#000000", "ink2": "#000000",
              "muted": "#000000", "grid": "#e6e8eb", "axis": "#9a9a9a", "dim": "#c3c2b7", "s1": "#2a78d6",
              "s2": "#eb6834", "tint": "#d6e6fa"},
}
SANS = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def load(p):
    return json.loads((R / p).read_text())


# ------------------------------------------------------------------ matplotlib plumbing
def _plt(c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 12.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "text.color": c["ink"], "axes.labelcolor": c["ink2"], "axes.edgecolor": c["axis"],
        "xtick.color": c["muted"], "ytick.color": c["muted"], "xtick.labelcolor": c["ink2"],
        "ytick.labelcolor": c["ink2"], "axes.grid": True, "grid.color": c["grid"], "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 13.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "legend.frameon": False, "legend.fontsize": 12, "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "xtick.major.size": 0, "ytick.major.size": 0,
    })
    return plt


def _save(fig, name, mode):
    """Every chart spans its full figure width (an invisible guide), so all render at one text size in the
    README, whatever their content's extent."""
    from matplotlib.lines import Line2D
    fig.add_artist(Line2D([-0.15, 1.06], [0.5, 0.5], transform=fig.transFigure, alpha=0))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (f"{name}.svg" if mode == "light" else f"{name}-dark.svg")
    fig.savefig(p, format="svg", bbox_inches="tight", pad_inches=0.12, metadata={"Date": None})
    import matplotlib.pyplot as plt
    plt.close(fig)
    s = p.read_text()
    s = s.replace("'DejaVu Sans'", SANS).replace('"DejaVu Sans"', SANS).replace("DejaVu Sans", SANS)
    s = re.sub(r"<!-- Created with matplotlib.*?-->\n?", "", s)
    p.write_text(s)
    return p


def _note(ax, text, c, y=-0.22):
    ax.text(0, y, text, transform=ax.transAxes, color=c["muted"], fontsize=11, va="top")


def _fnote(fig, text, c):
    import textwrap
    fig.text(0.125, -0.06, textwrap.fill(text, 100), color=c["muted"], fontsize=11, va="top")


# ------------------------------------------------------------------ the whole loop, one picture
def overview(mode):
    c = MODES[mode]
    W, H = 1000, 420
    bat = "results/battery-module/"
    card = json.loads((REPO / bat / "post-weld/model_card.json").read_text())["forward"]["auc"]
    nosig = json.loads((REPO / bat / "post-weld-no-signal/model_card.json").read_text())["forward"]["auc"]
    plan = json.loads((REPO / bat / "plan/plan.json").read_text())["holdout"]["optimised (MILP + twin)"]
    tw = json.loads((REPO / bat / "twin/sweep.json").read_text())
    rob = tw["robustness"]["optimised (MILP + twin)"]
    pl = json.loads((REPO / bat / "place/place.json").read_text())
    stages = [
        ("1", "Predict", ["learn which parts fail, from", "route, timing, measurements", "and the parts around them"],
         "XGBoost on cuDF features", "quality fit", f"AUC {card:.2f} after welding"),
        ("2", "Score live", ["compiles the model and its", "signal stage; an edge node", "scores each part on arrival"],
         "TensorRT · ONNX Runtime", "edge compile, replay", f"signals add +{card - nosig:.3f} AUC"),
        ("3", "Decide", ["a MILP picks thresholds and", "staffing; the Warp twin", "checks every plan"],
         "cuOpt · HiGHS + Warp", "plan", f"saves {-plan['cost_per_week'] / 1e3:,.1f}k / week"),
        ("4", "Stress-test", ["the twin replays the plan", "under thousands of drawn", "costs and cell parameters"],
         "Warp (GPU or CPU)", "twin", f"saves in {rob['saves_share']:.0%} of {tw['scenarios_per_policy']:,}"),
        ("5", "Place", ["server or edge node? from", "the deadline, the network", "and a late decision's cost"],
         "measured, then priced", "place", f"{pl['verdict']}: {pl['placements']['edge']['cost_per_month']:,.0f} vs "
                                           f"{pl['placements']['server']['cost_per_month']:,.0f}/mo"),
    ]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="{SANS}">',
           f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c["axis"]}"/></marker></defs>',
           f'<rect width="{W}" height="{H}" fill="{c["bg"]}"/>']

    def text(x, y, s, size=12, color=None, weight=400, anchor="start"):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color or c["ink"]}" font-weight="{weight}" '
                   f'text-anchor="{anchor}">{s}</text>')

    def box(x, y, w, h, fill=None, stroke=None, r=8):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill or c["card"]}" '
                   f'stroke="{stroke or c["border"]}" stroke-width="1"/>')

    def arrow(x1, y1, x2, y2, dashed=False):
        out.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{c["axis"]}" stroke-width="1.5" fill="none" '
                   f'marker-end="url(#a)"/>')

    # row 1: what you provide -> the cache
    text(20, 30, "YOU PROVIDE", 11, c["muted"], 600)
    ins = [("line spec", "lines/my-line.toml: stations, checkpoints, clock, costs"),
           ("history", "visits · results · measurements, from any factory system")]
    for k, (title, line) in enumerate(ins):
        x = 20 + k * 380
        box(x, 42, 366, 58)
        text(x + 14, 64, title, 13.5, weight=600)
        text(x + 14, 86, line, 11, c["ink2"])
        if k == 0:
            pass
    arrow(386, 71, 398, 71)
    arrow(766, 71, 788, 71)
    box(790, 42, 190, 58, c["bg"])
    text(804, 64, "cache", 13.5, weight=600)
    text(804, 86, "$ line ingest", 10.5, c["ink2"])
    # row 2: the five stages
    x0, w, gap, y0, h = 20, 180, 15, 150, 214
    text(x0, 138, "THEN, IN THIS ORDER  (one command runs them all: demo;  numbers: the simulated battery line)", 11,
         c["muted"], 600)
    out.append(f'<path d="M885,100 L885,118 L110,118 L110,{y0 - 2}" stroke="{c["axis"]}" stroke-width="1.5" '
               f'fill="none" marker-end="url(#a)"/>')
    for i, (num, title, lines, lib, cmd, result) in enumerate(stages):
        x = x0 + i * (w + gap)
        box(x, y0, w, h)
        out.append(f'<rect x="{x}" y="{y0}" width="{w}" height="4" rx="2" fill="{c["s1"]}"/>')
        out.append(f'<circle cx="{x + 24}" cy="{y0 + 32}" r="12" fill="{c["tint"]}" stroke="{c["s1"]}"/>')
        text(x + 24, y0 + 36.5, num, 12.5, c["ink"], 700, "middle")
        text(x + 44, y0 + 37, title, 15, weight=700)
        for j, ln in enumerate(lines):
            text(x + 14, y0 + 68 + j * 17, ln, 11.5, c["ink2"])
        out.append(f'<line x1="{x + 14}" y1="{y0 + 128}" x2="{x + w - 14}" y2="{y0 + 128}" stroke="{c["border"]}"/>')
        text(x + 14, y0 + 148, lib, 11, c["muted"], 600)
        out.append(f'<text x="{x + 14}" y="{y0 + 166}" font-size="10.5" fill="{c["ink2"]}" '
                   f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">$ {cmd}</text>')
        text(x + 14, y0 + 194, result, 12.5, c["ink"], 700)
        if i:
            arrow(x - gap, y0 + 32, x - 2, y0 + 32)
    text(20, H - 30, "Every step prints a scoreboard and writes JSON: every number in these docs is program "
                     "output (results/).", 11, c["muted"])
    text(20, H - 12, "Each step runs on an NVIDIA library, or on a CPU with the same results.", 11, c["muted"])
    out.append("</svg>")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / ("overview.svg" if mode == "light" else "overview-dark.svg")
    p.write_text("\n".join(out) + "\n")
    return p


class _Svg:
    """A tiny SVG writer for the diagrams (text, boxes, arrows, in the page's light or dark ink)."""

    def __init__(self, c, W, H):
        self.c, self.W, self.H = c, W, H
        self.out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
                    f'font-family="{SANS}">',
                    f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c["axis"]}"/></marker></defs>',
                    f'<rect width="{W}" height="{H}" fill="{c["bg"]}"/>']

    def text(self, x, y, s, size=12, color=None, weight=400, anchor="start", mono=False):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        fam = ' font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"' if mono else ""
        self.out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color or self.c["ink"]}" '
                        f'font-weight="{weight}" text-anchor="{anchor}"{fam}>{s}</text>')

    def box(self, x, y, w, h, fill=None, stroke=None, r=8):
        self.out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill or self.c["card"]}" '
                        f'stroke="{stroke or self.c["border"]}" stroke-width="1"/>')

    def diamond(self, cx, cy, w, h):
        c = self.c
        self.out.append(f'<path d="M{cx},{cy - h / 2} L{cx + w / 2},{cy} L{cx},{cy + h / 2} L{cx - w / 2},{cy} z" '
                        f'fill="{c["bg"]}" stroke="{c["border"]}" stroke-width="1"/>')

    def arrow(self, *pts):
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        self.out.append(f'<path d="{d}" stroke="{self.c["axis"]}" stroke-width="1.5" fill="none" '
                        f'marker-end="url(#a)"/>')

    def save(self, name, mode):
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / (f"{name}.svg" if mode == "light" else f"{name}-dark.svg")
        p.write_text("\n".join(self.out + ["</svg>"]) + "\n")
        return p


# ------------------------------------------------------------------ the time split every stage respects
def timeline(mode):
    c = MODES[mode]
    g = _Svg(c, 1000, 392)
    bc, bp = load("edge/model_card.json"), load("plan/plan.json")
    tc, tp = load("battery-module/post-weld/model_card.json"), load("battery-module/plan/plan.json")
    sizes = {"bosch": (bc["train_parts"], bp["brute_force"]["parts"], load("twin/sweep.json")["parts"]),
             "battery": (tc["train_parts"], tp["brute_force"]["parts"],
                         load("battery-module/twin/sweep.json")["parts"])}
    g.text(20, 28, "ONE TIMELINE, THREE WINDOWS: NOTHING IS TUNED ON THE WEEKS IT IS JUDGED ON", 11, c["muted"], 600)
    x0, lab_w, col_w, gap = 20, 250, 236, 8
    cols = [("train", "the past: the model learns here"), ("calibrate", "the plan is chosen here"),
            ("hold-out", "everything is judged here")]
    for k, (name, sub) in enumerate(cols):
        x = x0 + lab_w + k * (col_w + gap)
        g.box(x, 42, col_w, 66, c["card"])
        g.text(x + 12, 64, f"{k + 1} · {name}", 13.5, weight=700)
        g.text(x + 12, 82, sub, 11, c["ink2"])
        b, t = sizes["bosch"][k], sizes["battery"][k]
        g.text(x + 12, 99, f"Bosch {b:,} parts · battery {t:,}", 10.5, c["muted"])
        if k:
            g.text(x - gap / 2, 75, "→", 14, c["muted"], anchor="middle")
    g.text(x0, 64, "time →", 12, c["muted"])
    g.text(x0, 82, "cut at the 70 % and 85 %", 11, c["ink2"])
    g.text(x0, 99, "quantiles of decision times", 11, c["ink2"])
    rows = [("1 Predict", "quality fit --until t1", [(0, "trains"), (1, "checked forward"), (2, "checked forward")]),
            ("2 Score live", "edge replay --t-from t1", [(0, "history loaded"), (1, "scores live"),
                                                        (2, "scores live")]),
            ("3 Decide", "plan --split t1,t2", [(1, "MILP + twin plan here"), (2, "plan judged")]),
            ("4 Stress-test", "twin", [(2, "8,192 drawn economics")]),
            ("5 Place", "place", [(1, "real decision stream"), (2, "real decision stream")])]
    for r, (name, cmd, cells) in enumerate(rows):
        y = 130 + r * 50
        g.text(x0, y + 18, name, 13, weight=700)
        g.text(x0, y + 35, "$ " + cmd, 10.5, c["ink2"], mono=True)
        for k, role in cells:
            x = x0 + lab_w + k * (col_w + gap)
            strong = role not in ("history loaded",)
            fill = c["tint"] if strong else c["card"]
            g.box(x, y + 6, col_w, 32, fill, c["s1"] if strong else c["border"], 6)
            g.text(x + col_w / 2, y + 27, role, 11.5, c["ink"], 600 if strong else 400, "middle")
    g.text(x0, 392 - 8, "t1 and t2 come from the data; every stage reads the same split, so a number reported on "
                        "the hold-out was never used to choose anything.", 11, c["muted"])
    return g.save("timeline", mode)


# ------------------------------------------------------------------ what the twin simulates, part by part
def twin_diagram(mode):
    c = MODES[mode]
    g = _Svg(c, 1000, 404)
    g.text(20, 28, "THE TWIN: WHAT HAPPENS TO EACH PART AT THE CHECKPOINT, IN ONE SIMULATED SCENARIO", 11, c["muted"],
           600)
    # main flow, left to right
    g.box(20, 58, 170, 74)
    g.text(34, 82, "a part arrives", 13, weight=700)
    g.text(34, 100, "real time, real live score,", 11, c["ink2"])
    g.text(34, 116, "real outcome (from replay)", 11, c["ink2"])
    d1 = (300, 95)
    g.diamond(*d1, 190, 90)
    g.text(d1[0], d1[1] - 6, "score ≥ threshold for its", 11, c["ink"], 600, "middle")
    g.text(d1[0], d1[1] + 10, "route and shift slot?", 11, c["ink"], 600, "middle")
    g.arrow((190, 95), (204, 95))
    d2 = (510, 95)
    g.diamond(*d2, 190, 90)
    g.text(d2[0], d2[1] - 6, "an inspector on shift", 11, c["ink"], 600, "middle")
    g.text(d2[0], d2[1] + 10, "free within max wait?", 11, c["ink"], 600, "middle")
    g.arrow((395, 95), (414, 95))
    g.text(404, 86, "yes", 10.5, c["muted"], anchor="middle")
    g.box(626, 62, 150, 66)
    g.text(640, 86, "inspect", 13, weight=700)
    g.text(640, 104, "lognormal time;", 11, c["ink2"])
    g.text(640, 119, "the inspector is busy", 11, c["ink2"])
    g.arrow((605, 95), (624, 95))
    g.text(614, 86, "yes", 10.5, c["muted"], anchor="middle")
    d3 = (874, 95)
    g.diamond(*d3, 186, 92)
    g.text(d3[0], d3[1] - 6, "defect already there,", 11, c["ink"], 600, "middle")
    g.text(d3[0], d3[1] + 10, "and found (detect %)?", 11, c["ink"], 600, "middle")
    g.arrow((776, 95), (780, 95))
    # outcomes
    g.box(784, 182, 196, 66, c["bg"], c["s1"])
    g.text(798, 206, "caught early", 13, weight=700)
    g.text(798, 224, "saves its value + the work", 11, c["ink2"])
    g.text(798, 239, "after the checkpoint", 11, c["ink2"])
    g.arrow((874, 140), (874, 180))
    g.text(882, 164, "yes", 10.5, c["muted"])
    g.box(204, 182, 470, 66, c["bg"])
    g.text(218, 206, "goes on, as today", 13, weight=700)
    g.text(218, 224, "not flagged · bypassed (the cell was full) · missed · or its defect is made later:", 11,
           c["ink2"])
    g.text(218, 239, "a failing part is found at the final test, with all the line's work spent", 11, c["ink2"])
    g.arrow((300, 140), (300, 180))
    g.text(308, 164, "no", 10.5, c["muted"])
    g.arrow((510, 140), (510, 180))
    g.text(518, 164, "no (bypass)", 10.5, c["muted"])
    g.arrow((830, 118), (830, 150), (680, 150), (676, 182))
    g.text(756, 145, "no", 10.5, c["muted"])
    # real vs assumed, and the scale
    g.box(20, 278, 470, 96)
    g.text(34, 300, "real, from the data", 12.5, weight=700)
    for j, ln in enumerate(["arrival times and live scores (the edge replay), final-test outcomes,",
                            "where each defect arose (if recorded), the work each part went on to get"]):
        g.text(34, 320 + j * 17, ln, 11, c["ink2"])
    g.box(510, 278, 470, 96)
    g.text(524, 300, "assumed, from the spec (and varied by the stress-test)", 12.5, weight=700)
    for j, ln in enumerate(["thresholds and inspectors per shift slot (the plan), inspection time,",
                            "detection rate, maximum wait, and every cost"]):
        g.text(524, 320 + j * 17, ln, 11, c["ink2"])
    g.text(20, 396, "One GPU thread runs one scenario over every part of a window; thousands of scenarios run in "
                    "one launch, on NVIDIA Warp (or the same kernel on the CPU).", 11, c["muted"])
    return g.save("twin", mode)


# ------------------------------------------------------------------ 1. predict (Bosch: forward in time, ablation)
def predict(mode):
    c = MODES[mode]
    plt = _plt(c)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.75})
    rows = [("competition recipe, shuffled CV", load("quality/cv-kaggle.json"), False),
            ("live, shuffled CV", load("quality/cv-causal.json"), False),
            ("live, forward in time", load("quality/cv-causal-forward.json"), True),
            ("line-3 entry, forward in time", load("quality/cv-l3-entry-forward.json"), True)]
    for i, (lab, d, fwd) in enumerate(rows):
        y = len(rows) - 1 - i
        col = c["s1"] if fwd else c["dim"]
        a.barh(y, d["mcc_best"], height=0.5, color=col, linewidth=0)
        lo, hi = d["mcc_best_ci95"]
        a.plot([lo, hi], [y, y], color=c["ink2"], linewidth=1.2)
        a.text(hi + 0.012, y, f"{d['mcc_best']:.3f}", va="center", color=c["ink"], fontsize=12)
    a.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    a.set_xlim(0, 0.6)
    a.grid(axis="y", visible=False)
    a.set_xlabel("MCC (best threshold), with 95 % interval")
    a.set_title("Bosch: what going live really costs")

    card, no_id, no_order = (load("edge/model_card.json"), load("quality/ablation-no-id.json"),
                             load("quality/ablation-no-order.json"))
    rows = [("all features", card["forward"]["auc"]), ("without part-number features", no_id["forward"]["auc"]),
            ("without any order or\nneighbour feature", no_order["forward"]["auc"])]
    for i, (lab, v) in enumerate(rows):
        y = len(rows) - 1 - i
        b.barh(y, v - 0.5, left=0.5, height=0.5, color=c["s1"], linewidth=0)
        b.text(v + 0.006, y, f"{v:.3f}", va="center", color=c["ink"], fontsize=12)
    b.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    b.set_xlim(0.5, 0.92)
    b.axvline(0.75, color=c["ink2"], linewidth=1)
    b.set_ylim(-0.85, 2.5)
    b.text(0.745, -0.62, "pre-registered bar 0.75", color=c["ink2"], fontsize=11, ha="right")
    b.grid(axis="y", visible=False)
    b.set_xlabel("AUC on 354,824 later parts (0.5 = chance)")
    b.set_title("Line-3 warning: does it read the process?")
    _fnote(fig, "Left: shuffled folds train on parts made after the ones they score; forward folds don't. "
                "Right: the same model, trained before the cut-off, with feature families removed.", c)
    return _save(fig, "predict", mode)


# ------------------------------------------------------------------ 2. score live: what the edge node sees
def signals(mode):
    from lineloop import line as L
    from lineloop.data import simulate
    c = MODES[mode]
    plt = _plt(c)
    sg = L.read("battery-module").signal_specs()[0]
    W = sg.per_part
    good = np.zeros((1, W))
    poor = good.copy()
    poor[0, 0] = 3.2
    seed = np.array([7])
    g, p = simulate.waveforms(sg, good, seed)[0, 0], simulate.waveforms(sg, poor, seed)[0, 0]
    t = np.arange(sg.samples) / sg.rate_hz * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.7), gridspec_kw={"wspace": 0.12})
    names = {"power": "laser power", "photodiode": "photodiode (spatter)", "depth": "penetration depth"}
    for k, ax in enumerate(axes):
        ax.plot(t, g[k], color=c["s1"], linewidth=1.4, label="a sound weld")
        ax.plot(t, p[k], color=c["s2"], linewidth=1.4, label="a weld that will fail inside")
        ax.set_title(names.get(sg.channels[k], sg.channels[k]), fontsize=12)
        ax.set_xlabel("ms")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.suptitle(f"What the edge node sees: one of a module's {W} welds, {len(sg.channels)} channels x "
                 f"{sg.samples} samples", x=0.125, ha="left", y=1.2, fontsize=11.5, fontweight="bold",
                 color=c["ink"])
    fig.text(0.125, 1.08, f"{sg.bytes_per_part / 1e3:,.0f} kB a module, which the factory system never stores. "
                          "Same weld noise in both: only the weld's quality differs.", color=c["ink2"],
             fontsize=11.5)
    return _save(fig, "signals", mode)


# ------------------------------------------------------------------ 3. simulate: robustness + speed
def simulate_fig(mode):
    c = MODES[mode]
    plt = _plt(c)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.1), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.7})
    tw = load("battery-module/twin/sweep.json")
    names = {"best constant policy (top 15.0%, 3/shift)": "best single threshold",
             "best per-family policy (greedy search in the twin)": "greedy search per family",
             "optimised (MILP + twin)": "optimised (MILP + twin)"}
    rob = tw["robustness"]
    for i, (k, lab) in enumerate(names.items()):
        r = rob[k]
        y = len(names) - 1 - i
        lo, hi, med = -r["net_p95"] / 1e3, -r["net_p05"] / 1e3, -r["net_median"] / 1e3
        col = c["s1"] if k.startswith("optimised") else c["dim"]
        a.plot([lo, hi], [y, y], color=col, linewidth=5, solid_capstyle="round",
               alpha=0.6 if k.startswith("optimised") else 0.8)
        a.plot([med], [y], "o", color=col, markersize=8, markeredgecolor=c["bg"], markeredgewidth=2)
    a.axvline(0, color=c["ink2"], linewidth=1)
    a.set_yticks(range(len(names)), [f"{lab}\nsaves money in {rob[k]['saves_share']:.0%}"
                                     for k, lab in names.items()][::-1])
    a.set_xlim(-15, 215)
    a.grid(axis="y", visible=False)
    a.set_xlabel("saving / week, thousands")
    a.set_title(f"The plan under {tw['scenarios_per_policy']:,} drawn economics")
    d = tw["devices"]
    bosch = load("twin/sweep.json")["devices"]
    labels = [("1 CPU core", "CPU, 1 core"), ("every CPU core (best)", None), ("RTX 5090", None)]

    def get(dev, kind):
        if kind == 0:
            return 1.0
        if kind == 1:
            return 1 / min(v for k, v in dev["speedup"].items() if not k.endswith("1 core")) * \
                dev["speedup"]["CPU, 1 core"]
        return dev["speedup"]["CPU, 1 core"]
    for i in range(3):
        y = 2 - i
        for j, (dev, col, lab) in enumerate(((d, c["s1"], "battery line"), (bosch, c["s2"], "Bosch"))):
            v = get(dev, i)
            b.barh(y + (0.14 if j == 0 else -0.14), v, height=0.24, color=col, linewidth=0,
                   label=lab if i == 0 else None)
            b.text(v + 12, y + (0.14 if j == 0 else -0.14), f"{v:,.0f}x" if v >= 10 else f"{v:.1f}x",
                   va="center", color=c["ink"], fontsize=11)
    b.set_yticks(range(3), [x[0] for x in labels][::-1])
    b.set_xlim(0, 860)
    b.grid(axis="y", visible=False)
    b.set_xlabel("times one CPU core")
    b.set_title("Twin speed against one CPU core")
    b.legend(loc="center right")
    _fnote(fig, "Left: battery line, hold-out weeks; dot = median, bar = 5 % to 95 % of scenarios, with every cost, "
                "the inspection time and the detection rate drawn from its range. Right: twin throughput on the same "
                "scenarios.", c)
    return _save(fig, "simulate", mode)


# ------------------------------------------------------------------ 4. decide: break-even on both lines
def decide(mode):
    c = MODES[mode]
    plt = _plt(c)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw={"wspace": 0.35})
    for ax, path, title in ((a, "battery-module/plan/plan.json", "Battery line (simulated), after welding"),
                            (b, "plan/plan.json", "Bosch (real data), line-3 entry")):
        pl = load(path)
        be = pl["break_even"]
        x = [r["value_per_defect"] for r in be]
        y = [-r["cost_per_week"] / 1e3 for r in be]
        ax.axhline(0, color=c["ink2"], linewidth=1)
        ax.fill_between(x, 0, y, color=c["s1"], alpha=0.10, linewidth=0)
        ax.plot(x, y, color=c["s1"])
        ax.plot(x, y, "o", color=c["s1"], markersize=6, markeredgecolor=c["bg"], markeredgewidth=2)
        ax.set_xlabel("value of a defect caught early (an assumption)")
        ax.set_ylabel("saving / week, thousands")
        ax.set_title(title)
        lo = min(min(y), 0)
        hi = max(max(y), 0)
        pad = (hi - lo) * 0.22 or 1
        ax.set_ylim(lo - pad, hi + pad)
        k = int(np.argmax(np.abs(y)))
        ax.annotate(f"{y[k]:+,.1f}k at {x[k]:,.0f}", (x[k], y[k]), textcoords="offset points",
                    xytext=(-8, 10 if y[k] > 0 else -16), ha="right", color=c["ink"], fontsize=11.5)
    _fnote(fig, "The optimiser re-plans for each value and is scored on hold-out weeks. On real data it declines up "
                "to 150; the plans it would make at 300 and 600 lose money.", c)
    return _save(fig, "decide", mode)


# ------------------------------------------------------------------ 5. place: server or edge node
def place(mode):
    c = MODES[mode]
    plt = _plt(c)
    fig, ax = plt.subplots(figsize=(10, 2.9))
    cases = [("battery, after welding\n(interlocked cover press)", "battery-module/place/place.json"),
             ("battery, before welding\n(unchecked module goes on)", "battery-module-preweld/place/place.json"),
             ("Bosch, line-3 entry\n(policy never flags)", "place/place.json")]
    for i, (lab, path) in enumerate(cases):
        p = load(path)["placements"]
        y = len(cases) - 1 - i
        for dy, key, col, name in ((0.16, "server", c["s2"], "plant server: late decisions"),
                                   (-0.16, "edge", c["s1"], "edge node: its cost")):
            v = p[key]["cost_per_month"]
            ax.barh(y + dy, max(v, 0), height=0.28, color=col, linewidth=0, label=name if i == 0 else None)
            ax.text(max(v, 0) + 60, y + dy, f"{v:,.0f}", va="center", color=c["ink"], fontsize=11.5)
        win = load(path)["verdict"]
        ax.text(1.02, y, "→ edge node" if win == "edge" else "→ plant server", va="center", ha="left",
                color=c["ink"], fontsize=10.5, fontweight="bold", transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(cases)), [x[0] for x in cases][::-1])
    ax.set_xlim(0, 9800)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("cost per month (assumed network, outages, downtime and node cost from the line spec)")
    ax.set_title("Where should the checkpoint's model run? The cheaper placement wins")
    ax.legend(loc="lower right")
    return _save(fig, "place", mode)


# ------------------------------------------------------------------ cuOpt vs HiGHS by plant size
def solvers(mode):
    c = MODES[mode]
    plt = _plt(c)
    sv = load("solvers/solvers.json")
    sizes = [r["lines"] for r in sv["sizes"]]
    x = np.arange(len(sizes))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw={"wspace": 0.3})
    for name, col in (("cuopt", c["s1"]), ("highs", c["s2"])):
        t = [r["solvers"][name]["seconds"] for r in sv["sizes"]]
        g = [100 * (r["solvers"][name]["gap_to_best"] or 0) for r in sv["sizes"]]
        lab = {"cuopt": "cuOpt (GPU)", "highs": "HiGHS (CPU)"}[name]
        a.plot(x, t, color=col, label=lab)
        a.plot(x, t, "o", color=col, markersize=7, markeredgecolor=c["bg"], markeredgewidth=2)
        b.plot(x, g, color=col, label=lab)
        b.plot(x, g, "o", color=col, markersize=7, markeredgecolor=c["bg"], markeredgewidth=2)
    a.axhline(sv["time_limit"], color=c["ink2"], linewidth=1)
    a.text(0, sv["time_limit"] * 1.35, f"time limit {sv['time_limit']:g} s", color=c["ink2"], fontsize=11)
    a.set_yscale("log")
    for ax in (a, b):
        ax.set_xticks(x, [f"{s}" for s in sizes])
        ax.set_xlabel("lines in the plant (sharing one inspection crew)")
    a.set_ylabel("seconds")
    a.set_title("Solve time, the same model")
    b.set_ylabel("% worse than the best plan")
    b.set_title("Plan quality")
    b.legend(loc="upper right")
    return _save(fig, "solvers", mode)


FIGURES = (overview, timeline, twin_diagram, predict, signals, simulate_fig, decide, place, solvers)

if __name__ == "__main__":
    for f in FIGURES:
        print(f("light").relative_to(REPO))
