# Runtime overlay

The measured deployment used runtime mounts rather than a rebuilt image. Every node had the following shared paths:

| Host path | Container mount | Purpose |
|---|---|---|
| `/mnt/beegfs/sglang-v41-python` | `/v41` | V4.1 SGLang Python package tree |
| `/mnt/beegfs/sglang-v41-csrc` | `/v41csrc` | JIT CUDA/C++ source tree |
| `/mnt/beegfs/sglang-v41-jitinclude` | `/v41jitinclude` | SM121 JIT headers/templates |
| `/mnt/beegfs/dsv41-spark-adapter-hybrid` | `/v41adapter` | Files from this repository's `patch/` directory |

The base image was `lmsysorg/sglang:dev-v4f-2dgx-v2`, image ID `sha256:67873eb93b994736ab534111f79b5aa93d2575b973ebf9d276c40d548ff9afec`. It reported SGLang `0.0.0.dev1+g452239a74`, PyTorch `2.13.0+cu130`, and an ARM64 userspace. Keep the image ID fixed: the tag alone is mutable.

The launch command copies all Python files from `/v41` over `/sgl-workspace/sglang/python/sglang`, copies the JIT trees, compiles `patch/row_store.cpp` inside the ARM64 container, and prepends the adapter to `PYTHONPATH`. The large upstream Python/JIT snapshot is not duplicated here. Preserve the exact snapshot that accompanies the pinned image, or replace it with an upstream release that has separately passed the same validation suite.

To publish the adapter tree on shared storage:

```bash
install -d /mnt/beegfs/dsv41-spark-adapter-hybrid
install -m 0644 patch/engram_backend.py patch/sitecustomize.py \
  patch/row_store.cpp /mnt/beegfs/dsv41-spark-adapter-hybrid/
```

Before launch, verify the image identity and runtime on every node:

```bash
docker image inspect lmsysorg/sglang:dev-v4f-2dgx-v2 \
  --format '{{.Id}} {{.Architecture}}'
docker run --rm --entrypoint python3 lmsysorg/sglang:dev-v4f-2dgx-v2 \
  -c 'import platform,sglang,torch; print(platform.machine(),sglang.__version__,torch.__version__)'
```

The launcher intentionally does not pull images. Distribute a verified image archive through shared storage if a node lacks the pinned image.

## FlashInfer 0.7 follow-up

The optimized four-node follow-up uses
[`Dockerfile.flashinfer070`](Dockerfile.flashinfer070).  Prepare its build
context by copying the `flashinfer` package and matching
`flashinfer_python-0.7.0rc1.dist-info` from a locally available, compatible
ARM64 image.  Build on one node, save the derived image to shared storage, and
load that archive on the other nodes.  Do not mix package versions between
ranks.

Install the optimized SM120 module into the shared V4.1 Python overlay before
launch (adjust the destination to the equivalent module path in your pinned
snapshot):

```bash
install -m 0644 patches/flash_mla_sm120.py \
  /path/to/sglang-v41-python/kernels/ops/attention/flash_mla_sm120.py
```

Launch with the derived image, `--mla-backend hybrid`,
`--chunked-prefill-size 1024`, `--flashinfer-autotune`, and the
`DSV41_PREFILL_TRITON=1` environment set by the hybrid adapter.  The exact measured behavior is documented in
[`docs/FLASHINFER-070-OPTIMIZATION-20260911.md`](../docs/FLASHINFER-070-OPTIMIZATION-20260911.md).
