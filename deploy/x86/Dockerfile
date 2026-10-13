# Edge node for an x86 box. Two targets:
#   cpu  python slim, ONNX Runtime / XGBoost on CPU (what a gateway PC without a GPU runs)
#   gpu  + TensorRT, CuPy, cuML FIL, cuOpt wheels (needs the NVIDIA Container Toolkit on the host: --gpus all)
#   docker build -f deploy/x86/Dockerfile --target cpu -t lineloop-edge:cpu .
#   docker build -f deploy/x86/Dockerfile --target gpu -t lineloop-edge:gpu .
FROM python:3.12-slim AS cpu
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /opt/lineloop
COPY pyproject.toml README.md ./
COPY lineloop ./lineloop
RUN pip install .
ENTRYPOINT ["python", "-m", "lineloop"]
CMD ["doctor"]

FROM cpu AS gpu
COPY deploy/requirements-gpu.txt /tmp/
RUN pip install -r /tmp/requirements-gpu.txt --extra-index-url https://pypi.nvidia.com
