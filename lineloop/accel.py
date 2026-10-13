"""Which NVIDIA accelerators this machine can use, and the CPU fallback each component takes without them.

Every heavy step picks its backend here, so the same command runs on an RTX workstation, a Jetson and a laptop:

  component            GPU path                       CPU fallback
  feature engineering  RAPIDS cuDF                    pandas
  model training       XGBoost device="cuda"          XGBoost hist on CPU threads
  edge inference       TensorRT engine, cuML FIL      ONNX Runtime CPU, XGBoost CPU
  line twin            NVIDIA Warp kernel on cuda:0   the same Warp kernel on the CPU (LLVM)
  optimisation         NVIDIA cuOpt (GPU MILP)        HiGHS

Imports are lazy: probing cuDF costs about two seconds, so callers ask only for what they use.
"""
import ctypes
import shutil
import subprocess
from functools import lru_cache
from importlib import metadata
from importlib.util import find_spec


@lru_cache(maxsize=None)
def cuda_driver():
    """True when libcuda loads (a GPU and driver are present)."""
    try:
        ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        return False


@lru_cache(maxsize=None)
def gpu():
    """{name, memory_mib, driver, compute_cap} of GPU 0 via nvidia-smi, or None. Jetson has no nvidia-smi."""
    if not cuda_driver() or not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        name, mem, drv, cc = [s.strip() for s in out.splitlines()[0].split(",")]
        return {"name": name, "memory_mib": int(float(mem)), "driver": drv, "compute_cap": cc}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _ver(dist):
    for d in (dist, f"{dist}-cu12", f"{dist}-cu13"):
        try:
            return metadata.version(d)
        except metadata.PackageNotFoundError:
            continue
    return None


@lru_cache(maxsize=None)
def has(lib):
    """Can `lib` run on the GPU here? lib: cudf | cupy | xgboost | fil | cuopt | warp | tensorrt | torch | ort."""
    if not cuda_driver():
        return False
    try:
        if lib == "cudf":
            return find_spec("cudf") is not None
        if lib == "cupy":
            import cupy
            return cupy.cuda.runtime.getDeviceCount() > 0
        if lib == "xgboost":
            import xgboost
            return bool(xgboost.build_info().get("USE_CUDA"))
        if lib == "fil":
            return find_spec("cuml") is not None
        if lib == "cuopt":
            return find_spec("cuopt") is not None
        if lib == "warp":
            import warp as wp
            wp.config.quiet = True
            wp.init()
            return bool(wp.get_cuda_devices())
        if lib == "tensorrt":
            return find_spec("tensorrt") is not None
        if lib == "torch":
            import torch
            return torch.cuda.is_available()
        if lib == "ort":
            import onnxruntime
            return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:  # a broken optional install must not stop the CPU path
        return False
    raise ValueError(f"unknown accelerator {lib!r}")


def frame_lib(prefer="auto"):
    """The dataframe module for feature engineering: cudf on a GPU, else pandas."""
    if prefer == "cudf" or (prefer == "auto" and has("cudf")):
        import cudf
        return cudf
    import pandas
    return pandas


def xgb_device(prefer="auto"):
    if prefer in ("cuda", "cpu"):
        return prefer
    return "cuda" if has("xgboost") else "cpu"


def warp_device(prefer="auto"):
    if prefer not in ("auto", None):
        return prefer
    return "cuda:0" if has("warp") else "cpu"


def mip_solver(prefer="auto"):
    if prefer != "auto":
        return prefer
    if has("cuopt"):
        return "cuopt"
    if find_spec("highspy") is not None:
        return "highs"
    raise RuntimeError("no MIP solver: pip install highspy (CPU) or cuopt-cu12 (GPU)")


def report():
    """Rows (component, nvidia library, version, status) for `lineloop doctor`."""
    g = gpu()
    rows = [("GPU", g["name"] if g else ("CUDA driver, no nvidia-smi (Jetson?)" if cuda_driver() else "none"),
             g["driver"] if g else "", "ok" if cuda_driver() else "cpu only")]
    for comp, lib, dist in [("features", "cudf", "cudf"), ("training", "xgboost", "xgboost"),
                            ("edge inference", "tensorrt", "tensorrt"), ("edge inference", "fil", "cuml"),
                            ("edge inference", "ort", "onnxruntime-gpu"), ("line twin", "warp", "warp-lang"),
                            ("optimisation", "cuopt", "cuopt")]:
        v = _ver(dist) or (_ver("onnxruntime") if lib == "ort" else None)
        status = "gpu" if has(lib) else ("installed, cpu" if v else "missing")
        rows.append((comp, lib, v or "", status))
    return rows
