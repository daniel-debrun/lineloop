# Edge node for NVIDIA Jetson Orin (JetPack 6.x, L4T r36: CUDA 12.6, TensorRT 10.3). Build on the Jetson or with
# `docker buildx build --platform linux/arm64`. Run with the NVIDIA runtime:
#   docker run --rm --runtime nvidia -v $PWD/models:/models -v $PWD/data:/data lineloop-edge:jetson \
#     edge replay --model /models/l3-entry --cache /data/cache/bosch --backend trt --t-from 1300
# The TensorRT engine is built on first use from model.onnx, on the device itself (engines are per-GPU).
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip && rm -rf /var/lib/apt/lists/*
# TensorRT's Python bindings come with JetPack (python3-libnvinfer); CuPy has aarch64 CUDA 12 wheels.
RUN pip3 install "cupy-cuda12x>=13" "numpy<2.3" "pandas>=2.2" "pyarrow>=15" "scipy>=1.11" "scikit-learn>=1.4" \
        "xgboost>=2.1" "onnx>=1.16" "onnxruntime>=1.18" "warp-lang>=1.5" "highspy>=1.7" "rich>=13"
WORKDIR /opt/lineloop
COPY pyproject.toml README.md ./
COPY lineloop ./lineloop
RUN pip3 install --no-deps -e .
ENTRYPOINT ["python3", "-m", "lineloop"]
CMD ["doctor"]
