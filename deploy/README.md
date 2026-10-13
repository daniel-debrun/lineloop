# Deploy

## Run on the Bosch data (NVIDIA GPU)

1. Accept the competition rules on Kaggle and download the data:
   `kaggle competitions download -c bosch-production-line-performance`
2. Install the NVIDIA stack and run the pipeline:

```bash
deploy/vm/setup.sh                                   # pinned NVIDIA stack: deploy/requirements-gpu.txt
python -m lineloop doctor                            # which GPU libraries were found
python -m lineloop line ingest --line bosch --src <folder with the Kaggle zips>        # -> data/cache/bosch
python -m lineloop quality cv --line bosch --recipe offline --params '{"eta": 0.02}' --rounds 1300   # the competition score
tools/gpu_pipeline.sh data/cache/bosch               # live and forward-in-time models, ablations, edge compile /
                                                     # benchmark / replay, plan, twin, placement, solver scaling
python -m lineloop report --collect                  # results/SCOREBOARD.md
```

Tested on an RTX 5090, driver 580, CUDA 12.8 wheels.

## Targets

| target | file | what runs there | verified |
|---|---|---|---|
| GPU workstation / cloud (x86, RTX or datacenter GPU) | [`vm/setup.sh`](vm/setup.sh), [`requirements-gpu.txt`](requirements-gpu.txt) | everything: cuDF features, XGBoost training on CUDA, TensorRT / FIL inference, Warp twin, cuOpt | RTX 5090, driver 580, 2026-10-08 |
| Jetson Orin (JetPack 6.x) | [`jetson/Dockerfile`](jetson/Dockerfile) | the edge node: event stream, live features, the raw-signal stage and trees as one TensorRT engine built on the device | **never run**: no Jetson was available. The programs were verified on TensorRT 11.4 (RTX 5090); JetPack 6 ships TensorRT 10.3, which supports every op they use (Gather, GatherElements, Conv, Reduce*, logic ops), but that is not tested. On the battery line one CPU core keeps up with the node, so a CPU-only box would also do. |
| x86 gateway without a GPU | [`x86/Dockerfile`](x86/Dockerfile) `--target cpu` | the edge node on ONNX Runtime / XGBoost CPU, Warp twin on CPU, HiGHS | not built here |
| x86 with a GPU | [`x86/Dockerfile`](x86/Dockerfile) `--target gpu` | as the workstation, in a container | not built here (needs the NVIDIA Container Toolkit) |

Models and data are mounted, never baked in: the Bosch data may not be redistributed, and a model trained on it
stays with the data.
