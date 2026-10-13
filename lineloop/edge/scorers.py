"""Interchangeable inference backends for the compiled failure model, and a latency / throughput benchmark.

  backend      runs on                          from
  xgb-cpu      any CPU                          model.json (the reference every other backend is checked against)
  xgb-cuda     NVIDIA GPU                       model.json, XGBoost's CUDA predictor
  fil          NVIDIA GPU (x86, Jetson Thor)    model.json via cuML's Forest Inference Library
  trt          NVIDIA GPU incl. Jetson Orin     model.onnx -> TensorRT engine (lineloop.edge.compile)
  ort-cpu      any CPU                          model.onnx via ONNX Runtime (the portable fallback)

Every scorer takes a host float32 matrix and returns host probabilities, so timings include the copies an edge
node really pays (events arrive on the CPU; decisions leave on the CPU). A model with a signal stage
(lineloop.edge.compile) also takes the raw signal block S: ort-cpu and trt compute its features in the compiled
graph; xgb and fil, which run the model as trained, get them from the reference (lineloop.edge.signals).
"""
import time
from pathlib import Path

import numpy as np

BACKENDS = ("xgb-cpu", "xgb-cuda", "fil", "trt", "ort-cpu")


class Scorer:
    name = "?"
    signal = None                           # (Signal, feature columns) when the model has a signal stage

    def __call__(self, X, S=None):
        raise NotImplementedError

    def _with_signal(self, X, S):
        """X with the signal features filled in from S by the reference (for backends without the stage)."""
        if self.signal is None or S is None:
            return X
        from .signals import features_numpy
        X = np.array(X, np.float32, copy=True)
        X[:, self.signal[1]] = features_numpy(self.signal[0], S)
        return X


class XGB(Scorer):
    def __init__(self, model_dir, device="cpu", threads=None):
        import xgboost as xgb
        self.name = f"xgb-{device}"
        self.bst = xgb.Booster(model_file=str(Path(model_dir) / "model.json"))
        self.bst.set_param({"device": device, **({"nthread": threads} if threads else {})})
        self.device = device

    def __call__(self, X, S=None):
        X = self._with_signal(X, S)
        if self.device == "cuda":
            import cupy as cp
            return cp.asnumpy(self.bst.inplace_predict(cp.asarray(X)))
        return self.bst.inplace_predict(X)


class FIL(Scorer):
    name = "fil"

    def __init__(self, model_dir, batch_size=1024):
        from cuml.fil import ForestInference
        self.m = ForestInference.load(str(Path(model_dir) / "model.json"), is_classifier=True,
                                      model_type="xgboost_json", precision="single")
        self.batch_size = batch_size

    def tune(self, X):
        self.m.optimize(data=X[: self.batch_size], batch_size=self.batch_size, predict_method="predict_proba")

    def __call__(self, X, S=None):
        import cupy as cp
        p = self.m.predict_proba(cp.asarray(self._with_signal(X, S)))
        if p.ndim == 2:                       # (n, 2) in older cuML, (n, 1) = P(class 1) in 25.x; the bench's
            p = p[:, -1]                      # parity check against XGBoost confirms which
        return cp.asnumpy(p)


class ORT(Scorer):
    name = "ort-cpu"

    def __init__(self, model_dir, threads=None):
        import onnxruntime as ort
        o = ort.SessionOptions()
        if threads:
            o.intra_op_num_threads = threads
        self.s = ort.InferenceSession(str(Path(model_dir) / "model.onnx"), o, providers=["CPUExecutionProvider"])

    def __call__(self, X, S=None):
        feed = {"x": np.ascontiguousarray(X, np.float32)}
        if self.signal is not None:
            feed["sig"] = np.ascontiguousarray(S, np.float32)
        return self.s.run(["p"], feed)[0]


class TRT(Scorer):
    name = "trt"

    def __init__(self, model_dir, max_batch=4096, signal=None):
        import cupy as cp
        import tensorrt as trt
        from . import compile as C
        self.signal = signal
        if signal is not None:
            max_batch = 256                                         # raw blocks are large: keep the buffers small
        d = Path(model_dir)
        eng = d / f"model.b{max_batch}.engine"
        if not eng.exists():                                        # engines are per device: build here
            C.build_engine(d / "model.onnx", eng, max_batch=max_batch)
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(eng.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.max_batch = max_batch
        self.n_feat = self.engine.get_tensor_shape("x")[1]
        self.x = cp.empty((max_batch, self.n_feat), cp.float32)
        self.sig_shape = tuple(self.engine.get_tensor_shape("sig")[1:]) if self.signal is not None else None
        self.sig = cp.empty((max_batch, *self.sig_shape), cp.float32) if self.sig_shape else None
        self.p = cp.empty((max_batch,), cp.float32)
        self.host_p = None

    def __call__(self, X, S=None):
        import cupy as cp
        out = np.empty(len(X), np.float32)
        for i in range(0, len(X), self.max_batch):
            b = X[i:i + self.max_batch]
            n = len(b)
            with self.stream:
                self.x[:n].set(np.ascontiguousarray(b, np.float32), stream=self.stream)
                self.ctx.set_input_shape("x", (n, self.n_feat))
                self.ctx.set_tensor_address("x", self.x.data.ptr)
                if self.sig is not None:
                    self.sig[:n].set(np.ascontiguousarray(S[i:i + n], np.float32), stream=self.stream)
                    self.ctx.set_input_shape("sig", (n, *self.sig_shape))
                    self.ctx.set_tensor_address("sig", self.sig.data.ptr)
                self.ctx.set_tensor_address("p", self.p.data.ptr)
                self.ctx.execute_async_v3(self.stream.ptr)
                out[i:i + n] = cp.asnumpy(self.p[:n], stream=self.stream)
            self.stream.synchronize()
        return out


def make(name, model_dir, threads=None):
    from .compile import model_signal
    sig = model_signal(model_dir)
    if name == "xgb-cpu":
        s = XGB(model_dir, "cpu", threads)
    elif name == "xgb-cuda":
        s = XGB(model_dir, "cuda")
    elif name == "fil":
        s = FIL(model_dir)
    elif name == "trt":
        s = TRT(model_dir, signal=sig)
    elif name == "ort-cpu":
        s = ORT(model_dir, threads)
    else:
        raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")
    s.signal = sig
    return s


def available():
    from .. import accel
    out = ["xgb-cpu"]
    if accel.has("xgboost") and accel.has("cupy"):
        out.append("xgb-cuda")
    if accel.has("fil"):
        out.append("fil")
    if accel.has("tensorrt") and accel.has("cupy"):
        out.append("trt")
    try:
        import onnxruntime  # noqa: F401
        out.append("ort-cpu")
    except ImportError:
        pass
    return out


def bench(scorer, X, batch_sizes=(1, 16, 256, 4096), seconds=2.0, warmup=3, S=None):
    """Latency per call (p50 / p99, ms) and throughput (parts/s) per batch size, on rows of X (and of the raw
    signal blocks S, for a model with a signal stage)."""
    rows = []
    for b in batch_sizes:
        b = min(b, len(X))
        offs = [(k * b) % (len(X) - b + 1) for k in range(8)]
        xs = [np.ascontiguousarray(X[o:o + b]) for o in offs]
        ss = [None if S is None else np.ascontiguousarray(S[o:o + b]) for o in offs]
        for k in range(warmup):
            scorer(xs[k % 8], ss[k % 8])
        lat, t_end, k = [], time.perf_counter() + seconds, 0
        while time.perf_counter() < t_end or len(lat) < 5:
            t0 = time.perf_counter()
            scorer(xs[k % 8], ss[k % 8])
            lat.append(time.perf_counter() - t0)
            k += 1
        lat = np.array(lat)
        rows.append({"batch": b, "calls": len(lat), "p50_ms": float(np.median(lat) * 1e3),
                     "p99_ms": float(np.percentile(lat, 99) * 1e3), "parts_per_s": float(b / np.median(lat))})
    return rows
