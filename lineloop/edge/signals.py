"""In-station signals and the edge node's signal stage: raw waveforms in, a few numbers per part out.

A welder, a press or a nut-runner records waveforms while it works (current, force, displacement, at kHz rates).
Factory systems rarely keep them: they are large, and they live on the cell's network. A line spec declares them
with [[signal]] tables:

  name       e.g. "weld"
  stations   where the signal is recorded (parallel alternatives: whichever the part visits)
  channels   channel names, e.g. ["current", "voltage", "displacement"]
  rate_hz    samples per second
  samples    samples per repeat (one weld)
  per_part   repeats per part (welds per module)

The signal stage turns a part's raw block [per_part, channels, samples] into FEATURES per channel and repeat, then
summarises the repeats (mean, max and min), so a part gets 3 x 3 x channels numbers named
"<signal>.<channel>.<feature>_<summary>". They are recorded as measurements of the signal's last station, so the
rest of the framework (features, checkpoints, training) treats them like any other measurement; only the edge node
ever sees the raw waveforms.

The stage is written twice: `features_numpy` (the reference, and what a line's history holds) and `onnx_nodes`
(the same arithmetic as ONNX ops: Slice, Conv, Sub, Mul, ReduceMean / Max / Min, Concat), which the compiler puts
in front of the tree program, so TensorRT runs raw signal to decision in one engine.
"""
from dataclasses import dataclass

import numpy as np

FEATURES = ("mid", "hf", "end")     # mean of the middle 80 %; high-frequency energy; mean of the last 5 %
SUMMARIES = ("mean", "max", "min")  # over a part's repeats
BOX = 5                             # high-pass = x - moving average of BOX samples (zero padded at the ends)


@dataclass
class Signal:
    name: str
    stations: list
    channels: list
    rate_hz: float
    samples: int
    per_part: int

    @property
    def station(self):
        """The station its features are recorded under (the last listed)."""
        return self.stations[-1]

    @property
    def bytes_per_part(self):
        return 4 * self.per_part * len(self.channels) * self.samples         # float32

    @property
    def shape(self):
        return (self.per_part, len(self.channels), self.samples)

    def feature_names(self):
        per = [f"{c}.{f}" for c in self.channels for f in FEATURES]
        return [f"{self.name}.{p}_{s}" for s in SUMMARIES for p in per]

    def columns(self):
        """Measurement columns in the cache (station.name, as lineloop.data.events names them)."""
        return [f"{self.station}.{n}" for n in self.feature_names()]

    def windows(self):
        S = self.samples
        return (int(0.1 * S), int(0.9 * S)), (int(0.95 * S), S)


def from_spec(d):
    s = Signal(name=str(d["name"]), stations=list(d["stations"]), channels=list(d["channels"]),
               rate_hz=float(d["rate_hz"]), samples=int(d["samples"]), per_part=int(d.get("per_part", 1)))
    if s.samples < 40 or s.per_part < 1 or not s.channels or not s.stations:
        raise ValueError(f"signal {s.name!r}: needs stations, channels, samples >= 40 and per_part >= 1")
    return s


def features_numpy(sig, raw):
    """raw [n, per_part, channels, samples] -> [n, len(feature_names)] float32 (the reference)."""
    raw = np.asarray(raw, np.float32)
    (m0, m1), (e0, e1) = sig.windows()
    mid = raw[..., m0:m1].mean(axis=-1, dtype=np.float32)
    pad = np.pad(raw, [(0, 0)] * 3 + [(BOX // 2, BOX // 2)])
    box = sum(pad[..., k:k + raw.shape[-1]] for k in range(BOX)) * np.float32(1 / BOX)
    hp = raw - box
    hf = (hp * hp).mean(axis=-1, dtype=np.float32)
    end = raw[..., e0:e1].mean(axis=-1, dtype=np.float32)
    per = np.stack([mid, hf, end], axis=-1).reshape(raw.shape[0], raw.shape[1], -1)   # [n, W, C * 3]
    return np.concatenate([per.mean(axis=1), per.max(axis=1), per.min(axis=1)], axis=1).astype(np.float32)


def onnx_nodes(sig, inp="sig", out="sigfeat"):
    """(nodes, initializers): ONNX ops computing features_numpy from input `inp` [batch, W, C, S] to `out`
    [batch, n_features]."""
    from onnx import helper, numpy_helper
    W, C, S = sig.shape
    (m0, m1), (e0, e1) = sig.windows()
    i64 = lambda *v: np.array(v, np.int64)                                   # noqa: E731
    init = [numpy_helper.from_array(i64(-1, 1, S), "sg_flat"),
            numpy_helper.from_array(np.full((1, 1, BOX), 1 / BOX, np.float32), "sg_box"),
            numpy_helper.from_array(i64(m0), "sg_m0"), numpy_helper.from_array(i64(m1), "sg_m1"),
            numpy_helper.from_array(i64(e0), "sg_e0"), numpy_helper.from_array(i64(e1), "sg_e1"),
            numpy_helper.from_array(i64(-1), "sg_axl"),
            numpy_helper.from_array(i64(-1, W, C), "sg_bwc"),
            numpy_helper.from_array(i64(-1, W, C * len(FEATURES)), "sg_bwf")]
    n = [helper.make_node("Reshape", [inp, "sg_flat"], ["sg_x"]),                      # [B*W*C, 1, S]
         helper.make_node("Slice", ["sg_x", "sg_m0", "sg_m1", "sg_axl"], ["sg_xm"]),
         helper.make_node("ReduceMean", ["sg_xm"], ["sg_mid"], axes=[-1], keepdims=0),
         helper.make_node("Conv", ["sg_x", "sg_box"], ["sg_avg"], pads=[BOX // 2, BOX // 2]),
         helper.make_node("Sub", ["sg_x", "sg_avg"], ["sg_hp"]),
         helper.make_node("Mul", ["sg_hp", "sg_hp"], ["sg_hp2"]),
         helper.make_node("ReduceMean", ["sg_hp2"], ["sg_hf"], axes=[-1], keepdims=0),
         helper.make_node("Slice", ["sg_x", "sg_e0", "sg_e1", "sg_axl"], ["sg_xe"]),
         helper.make_node("ReduceMean", ["sg_xe"], ["sg_end"], axes=[-1], keepdims=0)]
    for f in FEATURES:                                                                  # [B*W*C, 1] -> [B, W, C]
        n.append(helper.make_node("Reshape", [f"sg_{f}", "sg_bwc"], [f"sg_{f}3"]))
        n.append(helper.make_node("Unsqueeze", [f"sg_{f}3", "sg_axl"], [f"sg_{f}4"]))
    n += [helper.make_node("Concat", [f"sg_{f}4" for f in FEATURES], ["sg_per4"], axis=-1),   # [B, W, C, 3]
          helper.make_node("Reshape", ["sg_per4", "sg_bwf"], ["sg_per"]),                    # [B, W, C*3]
          helper.make_node("ReduceMean", ["sg_per"], ["sg_mean"], axes=[1], keepdims=0),
          helper.make_node("ReduceMax", ["sg_per"], ["sg_max"], axes=[1], keepdims=0),
          helper.make_node("ReduceMin", ["sg_per"], ["sg_min"], axes=[1], keepdims=0),
          helper.make_node("Concat", ["sg_mean", "sg_max", "sg_min"], [out], axis=1)]
    return n, init
