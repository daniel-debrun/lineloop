"""Compile an XGBoost model into an ONNX tensor program that TensorRT (Jetson, RTX) and ONNX Runtime (any CPU) run.

Tree ensembles are branchy code; GPUs and TensorRT want dense tensor ops. Every tree is padded to a perfect binary
tree of the ensemble's depth D (a leaf above depth D is copied into all of its padded descendants, so the padded
splits cannot change the answer), and evaluation becomes D rounds of:

    node    = level_offset + idx                       per (part, tree)
    feature = Gather(FEAT, tree * N + node)
    value   = GatherElements(x, feature)
    left    = (value < threshold) or (IsNaN(value) and default_left)   # XGBoost's rule, missing -> default
    idx     = 2 * idx + not(left)

then margin = sum of Gather(LEAF, tree * 2^D + idx) + base margin, probability = sigmoid(margin). Only Gather,
GatherElements, IsNaN, Less, And, Or, Not, Cast, Add, Mul, ReduceSum and Sigmoid are used, all supported by
TensorRT 10 and ONNX Runtime. Indices are int32 throughout. Comparisons stay float32: thresholds are never
rounded, so the compiled model takes the same branch as XGBoost on every input (checked by `verify`).

A model that uses a signal's features (lineloop.edge.signals) gets a second input, the raw block
sig [batch, per_part, channels, samples]: the signal stage computes the features in the graph and writes them
into their columns of x (Concat, then Gather of a column permutation) before the trees run. The features then
agree with the reference to float32 rounding (~1e-7), not bit for bit, so a value sitting on a split threshold
can take the other branch; `edge compile` reports how often that changes the score.
"""
import json
from pathlib import Path

import numpy as np


def _base_margin(learner):
    raw = learner["learner_model_param"]["base_score"]
    b = float(json.loads(raw)[0]) if raw.startswith("[") else float(raw)
    obj = learner["objective"]["name"]
    if obj != "binary:logistic":
        raise ValueError(f"only binary:logistic models are compiled (got {obj})")
    return float(np.log(b / (1 - b)))


def trees_from_xgboost(model_json):
    """[(left, right, feature, threshold, default_left, value)] per tree + base margin, from save_model JSON."""
    m = json.loads(Path(model_json).read_text()) if not isinstance(model_json, dict) else model_json
    learner = m["learner"]
    gb = learner["gradient_booster"]
    if gb["name"] != "gbtree":
        raise ValueError("only gbtree boosters are compiled")
    trees = []
    for t in gb["model"]["trees"]:
        trees.append((np.array(t["left_children"]), np.array(t["right_children"]), np.array(t["split_indices"]),
                      np.array(t["split_conditions"], dtype=np.float32), np.array(t["default_left"], dtype=bool),
                      np.array(t["base_weights"], dtype=np.float32)))
    n_feat = int(learner["learner_model_param"]["num_feature"])
    names = learner.get("feature_names") or [f"f{i}" for i in range(n_feat)]
    return trees, _base_margin(learner), n_feat, names


def _depth(left, right):
    d, stack = 0, [(0, 0)]
    while stack:
        n, k = stack.pop()
        if left[n] == -1:
            d = max(d, k)
        else:
            stack += [(left[n], k + 1), (right[n], k + 1)]
    return d


def tensorize(trees, depth=None):
    """Dense arrays (FEAT int32 [T, 2^D-1], THR f32, DL bool, LEAF f32 [T, 2^D]) for perfect trees of depth D."""
    D = depth or max(_depth(t[0], t[1]) for t in trees)
    D = max(D, 1)
    N, L, T = 2 ** D - 1, 2 ** D, len(trees)
    feat = np.zeros((T, N), np.int32)
    thr = np.full((T, N), np.inf, np.float32)     # padded split: finite -> left, NaN -> default_left (True)
    dl = np.ones((T, N), bool)
    leaf = np.zeros((T, L), np.float32)
    for ti, (left, right, sidx, cond, dleft, _) in enumerate(trees):
        stack = [(0, 0, 0)]                       # (xgb node, level, index within level)
        while stack:
            n, lev, i = stack.pop()
            if left[n] == -1:                     # leaf: XGBoost keeps its value in split_conditions
                span = 2 ** (D - lev)
                leaf[ti, i * span:(i + 1) * span] = cond[n]
                continue
            if lev >= D:
                raise ValueError(f"tree {ti} deeper than {D}")
            k = 2 ** lev - 1 + i
            feat[ti, k], thr[ti, k], dl[ti, k] = sidx[n], cond[n], dleft[n]
            stack += [(left[n], lev + 1, 2 * i), (right[n], lev + 1, 2 * i + 1)]
    return {"feat": feat, "thr": thr, "dl": dl, "leaf": leaf, "depth": D}


def predict_numpy(t, base, X):
    """Reference evaluation of the tensor program in NumPy (what the ONNX graph computes)."""
    X = np.asarray(X, np.float32)
    B, T = X.shape[0], t["feat"].shape[0]
    N, L = t["feat"].shape[1], t["leaf"].shape[1]
    idx = np.zeros((B, T), np.int32)
    toff = np.arange(T, dtype=np.int64)[None, :]
    for lev in range(t["depth"]):
        flat = toff * N + (2 ** lev - 1) + idx
        f = t["feat"].ravel()[flat]
        v = np.take_along_axis(X, f, axis=1)
        left = np.where(np.isnan(v), t["dl"].ravel()[flat], v < t["thr"].ravel()[flat])
        idx = 2 * idx + (~left).astype(np.int32)
    margin = t["leaf"].ravel()[toff * L + idx].sum(axis=1, dtype=np.float32) + np.float32(base)
    return 1.0 / (1.0 + np.exp(-margin.astype(np.float64)))


def to_onnx(t, base, n_feat, path, opset=17, signal=None):
    """Write the tensor program as an ONNX model: input x [batch, n_feat] float32, output p [batch] float32.
    signal: (Signal, [feature column per signal feature]) adds the input sig and the signal stage."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    pre, x_in, inputs = [], "x", [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", n_feat])]
    pre_init = []
    if signal is not None:
        from . import signals as G
        sg, cols = signal
        nodes_s, init_s = G.onnx_nodes(sg, "sig", "sigfeat")
        perm = np.arange(n_feat, dtype=np.int64)
        perm[np.asarray(cols, np.int64)] = n_feat + np.arange(len(cols))
        pre = nodes_s + [helper.make_node("Concat", ["x", "sigfeat"], ["xs_cat"], axis=1),
                         helper.make_node("Gather", ["xs_cat", "SIGPERM"], ["xin"], axis=1)]
        pre_init = init_s + [numpy_helper.from_array(perm, "SIGPERM")]
        inputs.append(helper.make_tensor_value_info("sig", TensorProto.FLOAT, ["batch", *sg.shape]))
        x_in = "xin"
    T, N = t["feat"].shape
    L = t["leaf"].shape[1]
    init = [
        numpy_helper.from_array(t["feat"].ravel().astype(np.int32), "FEAT"),
        numpy_helper.from_array(t["thr"].ravel(), "THR"),
        numpy_helper.from_array(t["dl"].ravel(), "DL"),
        numpy_helper.from_array(t["leaf"].ravel(), "LEAF"),
        numpy_helper.from_array((np.arange(T, dtype=np.int32) * N)[None, :], "TOFF_N"),
        numpy_helper.from_array((np.arange(T, dtype=np.int32) * L)[None, :], "TOFF_L"),
        numpy_helper.from_array(np.array([base], np.float32), "BASE"),
        numpy_helper.from_array(np.array([T], np.int64), "T"),
        numpy_helper.from_array(np.array([0], np.int64), "I0"),
        numpy_helper.from_array(np.array([1], np.int64), "I1"),
        numpy_helper.from_array(np.array([1], np.int64), "AX1"),
        numpy_helper.from_array(np.array(2, np.int32), "TWO"),
    ]
    nodes = pre + [
        helper.make_node("Shape", [x_in], ["xs"]),
        helper.make_node("Slice", ["xs", "I0", "I1"], ["B"]),
        helper.make_node("Concat", ["B", "T"], ["bt"], axis=0),
        helper.make_node("ConstantOfShape", ["bt"], ["idx0"],
                         value=helper.make_tensor("z", TensorProto.INT32, [1], [0])),
    ]
    idx = "idx0"
    for lev in range(t["depth"]):
        s = f"_{lev}"
        init.append(numpy_helper.from_array(np.array(2 ** lev - 1, np.int32), f"OFF{s}"))
        nodes += [
            helper.make_node("Add", [idx, f"OFF{s}"], [f"node{s}"]),
            helper.make_node("Add", [f"node{s}", "TOFF_N"], [f"flat{s}"]),
            helper.make_node("Gather", ["FEAT", f"flat{s}"], [f"f{s}"], axis=0),
            helper.make_node("GatherElements", [x_in, f"f{s}"], [f"v{s}"], axis=1),
            helper.make_node("Gather", ["THR", f"flat{s}"], [f"thr{s}"], axis=0),
            helper.make_node("Gather", ["DL", f"flat{s}"], [f"dl{s}"], axis=0),
            helper.make_node("IsNaN", [f"v{s}"], [f"nan{s}"]),
            helper.make_node("Less", [f"v{s}", f"thr{s}"], [f"lt{s}"]),
            helper.make_node("And", [f"nan{s}", f"dl{s}"], [f"nl{s}"]),
            helper.make_node("Or", [f"lt{s}", f"nl{s}"], [f"left{s}"]),     # NaN < t is false, so this is exact
            helper.make_node("Not", [f"left{s}"], [f"right{s}"]),
            helper.make_node("Cast", [f"right{s}"], [f"r{s}"], to=TensorProto.INT32),
            helper.make_node("Mul", [idx, "TWO"], [f"dbl{s}"]),
            helper.make_node("Add", [f"dbl{s}", f"r{s}"], [f"idx{s}"]),
        ]
        idx = f"idx{s}"
    nodes += [
        helper.make_node("Add", [idx, "TOFF_L"], ["lflat"]),
        helper.make_node("Gather", ["LEAF", "lflat"], ["leaves"], axis=0),
        helper.make_node("ReduceSum", ["leaves", "AX1"], ["msum"], keepdims=0),
        helper.make_node("Add", ["msum", "BASE"], ["margin"]),
        helper.make_node("Sigmoid", ["margin"], ["p"]),
    ]
    g = helper.make_graph(nodes, "lineloop_trees", inputs,
                          [helper.make_tensor_value_info("p", TensorProto.FLOAT, ["batch"])], init + pre_init)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], producer_name="lineloop.edge.compile")
    m.ir_version = 8
    onnx.checker.check_model(m)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(path))
    return path


def model_signal(model_dir):
    """(Signal, feature columns) if the model in model_dir uses a signal's features (signals.json), else None."""
    p = Path(model_dir) / "signals.json"
    if not p.exists():
        return None
    from . import signals as G
    d = json.loads(p.read_text())
    return G.from_spec(d["signal"]), d["columns"]


def write_signal_manifest(model_dir, line, names):
    """signals.json next to a model when its features include a signal's (the compiler and the scorers read it).
    One signal per model."""
    used = [(sg, [names.index(c) if c in names else -1 for c in sg.columns()]) for sg in line.signal_specs()]
    used = [(sg, cols) for sg, cols in used if any(c >= 0 for c in cols)]
    p = Path(model_dir) / "signals.json"
    if not used:
        p.unlink(missing_ok=True)
        return None
    if len(used) > 1 or any(c < 0 for c in used[0][1]):
        raise ValueError("a model may use one signal, with all of its features")
    sg, cols = used[0]
    spec = next(d for d in line.signals if d["name"] == sg.name)
    p.write_text(json.dumps({"signal": spec, "columns": cols}, indent=1))
    return sg, cols


def compile_model(model_json, out_dir):
    """model.json -> out_dir/model.onnx + out_dir/compile.json; returns the manifest."""
    trees, base, n_feat, names = trees_from_xgboost(model_json)
    t = tensorize(trees)
    out_dir = Path(out_dir)
    sig = model_signal(out_dir)
    to_onnx(t, base, n_feat, out_dir / "model.onnx", signal=sig)
    man = {"trees": len(trees), "depth": t["depth"], "features": n_feat, "base_margin": base,
           "onnx_bytes": (out_dir / "model.onnx").stat().st_size, "feature_names": names,
           "signal": None if sig is None else {"name": sig[0].name, "shape": list(sig[0].shape),
                                               "bytes_per_part": sig[0].bytes_per_part,
                                               "features": len(sig[1])}}
    (out_dir / "compile.json").write_text(json.dumps(man, indent=1))
    return man


def build_engine(onnx_path, engine_path, max_batch=4096, opt_batch=256, workspace_gb=2):
    """Serialise a TensorRT engine for this GPU (engines are device-specific: build on the Jetson itself)."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if hasattr(trt.BuilderFlag, "TF32"):        # full float32 in the signal stage's Conv: TF32 rounds to 10 bits,
        cfg.clear_flag(trt.BuilderFlag.TF32)    # enough to move a feature across a tree's split
    n_feat = net.get_input(0).shape[1]
    prof = builder.create_optimization_profile()
    prof.set_shape("x", (1, n_feat), (opt_batch, n_feat), (max_batch, n_feat))
    for k in range(1, net.num_inputs):                     # the raw signal block, if the model has a signal stage
        inp = net.get_input(k)
        rest = tuple(inp.shape[1:])
        prof.set_shape(inp.name, (1, *rest), (opt_batch, *rest), (max_batch, *rest))
    cfg.add_optimization_profile(prof)
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError("TensorRT engine build failed")
    data = bytes(blob)                      # IHostMemory (TensorRT 10 / 11) -> bytes
    Path(engine_path).write_bytes(data)
    return {"engine_bytes": len(data), "max_batch": max_batch, "trt": trt.__version__}
