"""Launch the user-specified DSV4.1 reference recipe on DGX Spark nodes.

The reference image is amd64/SM120-only, so this launcher keeps the Spark
ARM64 image and overlays the reference's V4.1 Python implementation.  The
Engram adapter is compiled inside each ARM64 container.
"""
from __future__ import annotations
import pathlib
import shlex
import subprocess
import sys

HOSTS = [f"192.168.32.{n}" for n in range(98, 106)]
IMAGE = "lmsysorg/sglang:dev-v4f-2dgx-v2"
MODEL = "/mnt/beegfs/models/DeepSeek-V4.1-Flash"
OVERLAY = "/mnt/beegfs/sglang-v41-python"
ADAPTER = "/mnt/beegfs/dsv41-spark-adapter"
CSRC = "/mnt/beegfs/sglang-v41-csrc"
JITINCLUDE = "/mnt/beegfs/sglang-v41-jitinclude"
MASTER = "192.168.32.98"
ROOT = pathlib.Path(__file__).resolve().parent

COPY = r'''for f in $(find /v41 -type f -name '*.py'); do rel=${f#/v41/}; mkdir -p /sgl-workspace/sglang/python/sglang/$(dirname "$rel"); cp /v41/$rel /sgl-workspace/sglang/python/sglang/$rel; done; if [ -f /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4_vl.py ]; then sed -i 's/^EntryClass = \[DeepseekV4ForCausalLM\]/EntryClass = []/' /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4_vl.py; fi'''


def ssh(host: str, command: str) -> None:
    subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command], check=True)


def command(size: int, rank: int, port: int) -> str:
    name = f"dsv41-spark{size}-20260910"
    env = [
        "NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1",
        "NCCL_SOCKET_IFNAME=enp1s0f1np1",
        "GLOO_SOCKET_IFNAME=enp1s0f1np1",
        "NCCL_IB_GID_INDEX=5",
        f"SGLANG_HOST_IP={HOSTS[rank]}",
        "SGLANG_OPT_DEEPGEMM_HC_PRENORM=0",
        "DSV41_SOURCE=/models/dsv41",
        "OFFLOAD_MODE=nvme",
        "DSV41_CACHE_GIB=32",
    ]
    args = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", "/models/dsv41", "--served-model-name", "deepseek-v4.1-flash",
        "--trust-remote-code", "--load-format", "safetensors",
        "--tp-size", str(size), "--ep-size", str(size),
        "--attention-backend", "dsv4", "--moe-runner-backend", "flashinfer_mxfp4",
        "--mem-fraction-static", "0.70" if size == 4 else "0.82",
        "--chunked-prefill-size", "2048", "--context-length", "409600",
        "--max-running-requests", "4", "--cuda-graph-max-bs-decode", "4",
        "--min-free-slots-delay", "1", "--random-seed", "0",
        "--speculative-algorithm", "DSPARK", "--speculative-dspark-block-size", "5",
        "--tool-call-parser", "deepseekv41", "--reasoning-parser", "deepseek-v41",
        "--host", "0.0.0.0", "--port", str(port),
        "--dist-init-addr", f"{MASTER}:{29540 + size}",
        "--nnodes", str(size), "--node-rank", str(rank),
    ]
    env_args = " ".join("-e " + shlex.quote(x) for x in env)
    inner = (
        "set -e; mkdir -p /tmp/dsv41-adapter; cp -a /v41adapter/. /tmp/dsv41-adapter; "
        "g++ -O2 -Wall -Wextra -Werror -std=c++17 -shared -fPIC -pthread "
        "/tmp/dsv41-adapter/row_store.cpp -o /tmp/dsv41-adapter/librow_store.so; "
        + COPY + "; mkdir -p /sgl-workspace/sglang/python/sglang/kernels/jit/csrc /sgl-workspace/sglang/python/sglang/kernels/jit/include; cp -a /v41csrc/. /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/; cp -a /v41jitinclude/. /sgl-workspace/sglang/python/sglang/kernels/jit/include/; export PYTHONPATH=/tmp/dsv41-adapter:/sgl-workspace/sglang/python; exec "
        + shlex.join(args)
    )
    return shlex.join([
        "docker", "run", "-d", "--name", name, "--entrypoint", "bash", "--gpus", "all",
        "--network", "host", "--ipc", "host", "--privileged", "--ulimit", "memlock=-1",
        "-v", f"{MODEL}:/models/dsv41:ro", "-v", f"{OVERLAY}:/v41:ro", "-v", f"{ADAPTER}:/v41adapter:ro", "-v", f"{CSRC}:/v41csrc:ro", "-v", f"{JITINCLUDE}:/v41jitinclude:ro",
        *sum((["-e", x] for x in env), []), IMAGE, "-lc", inner,
    ])


def launch(size: int) -> None:
    if size not in (4, 8):
        raise SystemExit("size must be 4 or 8")
    port = 8100 + size
    name = f"dsv41-spark{size}-20260910"
    for host in HOSTS[:size]:
        ssh(host, f"docker rm -f {name} >/dev/null 2>&1 || true")
    for rank, host in list(enumerate(HOSTS[:size]))[1:] + [(0, HOSTS[0])]:
        cmd = command(size, rank, port)
        (ROOT / f"spark{size}-rank{rank}.command.sh").write_text(cmd + "\n")
        ssh(host, cmd)
    print(f"LAUNCHED {name} port={port}")


if __name__ == "__main__":
    launch(int(sys.argv[1]))
