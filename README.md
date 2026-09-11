# DeepSeek-V4.1-Flash on 4/8 × DGX Spark with SGLang

Run DeepSeek-V4.1-Flash across four or eight DGX Spark / GB10 nodes with SGLang. The measured configuration keeps the published model precision, uses TP=EP=node count and PP1, enables DSpark block 5 and decode CUDA graphs, and stages each rank's Engram rows on node-local storage.

This repository contains the launcher, Spark-specific patches, validation suite, raw benchmark receipts, and the final HTML/PDF report. It records the exact experimental deployment measured on 2026-09-11; it is not an upstream SGLang release.

## 1. Measured status

| Item | 4 Spark | 8 Spark |
|---|---:|---:|
| Parallel layout | TP4 / EP4 / PP1 | TP8 / EP8 / PP1 |
| C1 count decode, 3-run median | **71.04 tok/s** | **78.32 tok/s** |
| C1 code decode, 3-run median | **57.92 tok/s** | **67.03 tok/s** |
| C1 prose decode, 3-run median | **27.37 tok/s** | **28.58 tok/s** |
| C8 count aggregate decode | 284.16 tok/s | 293.84 tok/s |
| C8 code aggregate decode | 207.87 tok/s | 216.19 tok/s |
| C8 prose aggregate decode | 87.81 tok/s | 84.21 tok/s |
| 2,950-token client TTFT | 11.502 s | 7.062 s |
| 5,853-token client TTFT | 23.141 s | 13.182 s |
| Validation | 108 matrix requests + 21 checks passed | 108 matrix requests + 21 checks passed |

The eight-node deployment improved median C1 decode by 4.4%–15.7% over four nodes. It improved effective prefill by 1.63–1.76×, while C8 aggregate decode changed by about -4% to +4%. For single-stream decode, eight nodes are fastest in this data; four nodes deliver substantially better performance per node.

The count prompt is a useful decode ceiling, not a general quality score. A separate six-run eight-node count test produced 69.64, 86.15, 85.34, 85.99, 84.80, and 85.47 tok/s, with a median of **85.40 tok/s**. All six outputs counted 1–80 correctly. The first run shows material run-to-run variation, so the main comparison above uses the fixed three-repeat protocol.

### FlashInfer 0.7 optimization follow-up

A four-node follow-up replaced FlashInfer 0.6.18 with 0.7.0rc1 and routed the
short decode path directly to its DSV4 split-K kernel while retaining Triton
for large prefill batches.  The final C1 medians were **77.66 tok/s** for count,
**58.17 tok/s** for code and **16.86 tok/s** for prose.  Warmed effective
prefill improved to **347.6/351.2 tok/s** at 2,950/5,853 prompt tokens.  The
prose result varies with DSpark acceptance and is therefore reported alongside,
rather than attributed to, the attention kernel.  See
[`docs/FLASHINFER-070-OPTIMIZATION-20260911.md`](docs/FLASHINFER-070-OPTIMIZATION-20260911.md)
for the patch, rejected chunk-size test and full interpretation.

## 2. What differs from the stock image

The measured service started from `lmsysorg/sglang:dev-v4f-2dgx-v2`, image ID `sha256:67873eb93b994736ab534111f79b5aa93d2575b973ebf9d276c40d548ff9afec`, reporting SGLang `0.0.0.dev1+g452239a74` and PyTorch `2.13.0+cu130`. The launcher overlays the V4.1 Python implementation plus the SM121 JIT source/header set at container start.

The Spark adaptation adds four behavior changes:

1. [`patch/engram_backend.py`](patch/engram_backend.py) locates ARM64 CUDA runtime libraries and accepts only locally staged Engram shards whose model-index hash, safetensors header, and rank range validate against the checkpoint.
2. [`patch/sitecustomize.py`](patch/sitecustomize.py) keeps FlashInfer for short decode queries and routes the longer prefill path through the SM120 Triton sparse-MLA implementation. The FlashInfer 0.7 follow-up additionally overlays [`patches/flash_mla_sm120.py`](patches/flash_mla_sm120.py) to call the DSV4 decode kernel directly. An explicitly armed diagnostic can compare both outputs; it is disabled during timing.
3. [`launch/recover_eager.py`](launch/recover_eager.py) forces sequential bounded-memory weight loading and clones CPU expert views before H2D. This avoids GB10 unified-memory loader deaths seen with the stock asynchronous path.
4. The final launch uses DSpark block 5, decode graphs for batch sizes 1–8, local Engram storage, `max-total-tokens=8192`, and an 8,192-token context limit.

The original measurement did not rebuild the Docker image. The FlashInfer 0.7
follow-up uses [`build/Dockerfile.flashinfer070`](build/Dockerfile.flashinfer070)
with an ARM64 package extracted from a compatible local image.  See
[`build/README.md`](build/README.md) for the required tree layout and
[`docs/RUNBOOK-SGLANG-SPARK.md`](docs/RUNBOOK-SGLANG-SPARK.md) for the measured
launch procedure.

## 3. Launch

Requirements:

- four or eight ARM64 DGX Spark / GB10 nodes with Docker and NVIDIA Container Toolkit;
- the same SGLang image on every node;
- a shared DeepSeek-V4.1-Flash checkpoint;
- the prepared V4.1 Python/JIT trees described in [`build/README.md`](build/README.md);
- passwordless SSH from the control node and one consistent RoCE interface/GID configuration;
- rank-local Engram rows created and verified with the scripts in [`tools/`](tools/).

Edit the host, interface, rendezvous, checkpoint and shared-tree constants in [`launch/launch_spark_v41.py`](launch/launch_spark_v41.py). The final measured shape can then be rendered first and launched after review:

```bash
# Render commands and run the non-mutating preflight.
python3 launch/recover_eager.py --size 4 --probe \
  --hosts spark01 spark04 spark05 spark06 --master 10.0.0.1

# Launch the measured 4-node shape.
python3 launch/recover_eager.py --size 4 --launch \
  --name dsv41-spark4-hybrid --dspark --graphs --concurrency 8 \
  --local-engram --force-copy --mla-backend hybrid \
  --hosts spark01 spark04 spark05 spark06 --master 10.0.0.1

# Launch the measured 8-node shape.
python3 launch/recover_eager.py --size 8 --launch \
  --name dsv41-spark8-hybrid --dspark --graphs --concurrency 8 \
  --local-engram --force-copy --mla-backend hybrid \
  --hosts spark01 spark02 spark03 spark04 spark05 spark06 spark07 spark08 \
  --master 10.0.0.1
```

The launcher probes every host before changing containers. On launch, it saves logs from same-name experiment containers, stops and renames only those containers, then starts nonzero ranks before rank zero. Generated commands and logs go to `results/live/`.

## 4. Validation and benchmark

The suite treats prefill and decode as separate metrics. Decode throughput uses the server-reported completion token count and elapsed time from the first content token through EOS; SSE chunk count is never used as token count. Prefill is reported as client-observed prompt tokens divided by TTFT, so it includes scheduling, network, and first-token overhead.

```bash
python3 bench/finish_spark_bench.py http://spark01:8104 \
  --label spark4-hybrid
```

Before the speed sweep, the suite runs nine functional prompts, nine prompt-length checks, and three unique-prefix retrieval checks. It then runs three C1 repeats and the C1–C8 matrix. The measured raw receipts are under [`results/sglang-final/`](results/sglang-final/), and their internal consistency can be checked offline:

```bash
python3 tools/verify_final_data.py
```

## 5. Aggregate decode matrix

Each cell is delivered completion tokens per second across all active streams. C1 values here are the single matrix run; the headline C1 values above are the three-run medians.

| Nodes / prompt | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 / count | 63.67 | 112.96 | 147.21 | 190.45 | 219.47 | 246.76 | 265.82 | 284.16 |
| 4 / code | 51.41 | 83.79 | 103.55 | 134.64 | 152.86 | 186.21 | 187.36 | 207.87 |
| 4 / prose | 22.96 | 36.49 | 42.77 | 58.62 | 57.19 | 71.15 | 81.36 | 87.81 |
| 8 / count | 67.03 | 142.27 | 182.27 | 216.48 | 251.79 | 249.77 | 280.72 | 293.84 |
| 8 / code | 60.89 | 93.35 | 122.46 | 164.04 | 172.57 | 203.55 | 226.05 | 216.19 |
| 8 / prose | 28.35 | 40.01 | 55.64 | 65.75 | 72.70 | 79.21 | 87.99 | 84.21 |

The publication-ready comparison, including charts and the broader experiment history, is available as [HTML](results/DGX-Spark-DeepSeek-V4.1-Flash-Spark-4-8机性能报告-20260910.html) and [PDF](results/DGX-Spark-DeepSeek-V4.1-Flash-Spark-4-8机性能报告-20260910.pdf).

## 6. Four- and eight-node vLLM cross-check

The referenced vLLM deployment was also reproduced locally on four Spark nodes. Its three-run median C1 decode was **91.98 tok/s** for counting, **75.54 tok/s** for code, and **30.79 tok/s** for prose. Its C8 aggregate decode was 341.26, 237.45, and 95.55 tok/s, and effective prefill reached 1,502.7–1,664.4 tok/s for the measured 2,950/5,853-token inputs. The 18 functional/length checks, three cold retrieval checks, and all 108 matrix requests passed. See [`docs/VLLM-CROSSCHECK-20260911.md`](docs/VLLM-CROSSCHECK-20260911.md) and [`results/vllm-crosscheck/`](results/vllm-crosscheck/).

The eight-node TP8/EP8/PP1 follow-up reached three-run C1 medians of
**108.03/83.24/37.53 tok/s** and C8 aggregate decode of
**364.37/273.91/103.38 tok/s** for count/code/prose.  Effective prefill was
1,461.6/1,685.3 tok/s, essentially flat versus four nodes.  Its final
`gpu-memory-utilization=0.65` avoids the driver workspace OOM observed at 0.80
while retaining sufficient KV capacity for the C1-C8 test.  The same 21
correctness/retrieval checks and 108 matrix requests passed; compact receipts
are in [`results/vllm-crosscheck-8node/`](results/vllm-crosscheck-8node/).

## 7. Repository layout

| Path | Contents |
|---|---|
| `build/` | Runtime overlay layout and provenance |
| `launch/` | Multi-node command renderer, preflight, launch and recovery |
| `patch/` | ARM64/local-Engram adapter and hybrid-attention hook |
| `bench/` | Fixed prompts, correctness gates and C1–C8 benchmark |
| `tools/` | Engram staging/verification and GPU diagnostics |
| `docs/` | Runbook, findings, migration notes and third-party notices |
| `results/` | Self-contained report, summaries and raw request receipts |

## 8. Credits and license

The local Engram adapter began from [0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000). The prompt set and parts of the benchmark/reporting workflow are adapted from [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark), fixed at commit `ca662ac35193c69ace9cee37f13a94abf2eff0fc`. SGLang-derived runtime code remains under Apache-2.0; deployment and adapter changes are provided under [`LICENSE`](LICENSE). Model weights and container images are not redistributed.
