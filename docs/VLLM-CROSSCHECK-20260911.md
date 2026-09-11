# Four- and eight-Spark vLLM cross-check

This is a local cross-check of the launch published by
[`tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark`](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark),
fixed at commit `ca662ac35193c69ace9cee37f13a94abf2eff0fc`. It was measured on the same
four DGX Spark nodes and with the same prompt set as the SGLang results in this repository.

An eight-node TP8/EP8/PP1 follow-up used the same image, model patches,
DSpark configuration, prompt set and scheduler limits.  Its
`gpu-memory-utilization` was reduced from 0.80 to 0.65: at 0.80, vLLM profiled
about 39 GiB of KV cache per node and the NVIDIA driver reported
`NV_ERR_NO_MEMORY` when FlashInfer requested temporary autotune workspace.  The
0.65 setting leaves ample KV capacity for C1-C8 and completed without OOM.

## Environment

| Item | Value |
|---|---|
| Nodes | 4 × DGX Spark / GB10, ARM64 |
| Parallelism | TP4 / PP1; expert parallel follows TP |
| Interconnect | 200 GbE RoCE; one data interface per node |
| vLLM | `0.1.dev20904+g179dd0fa9` |
| PyTorch | `2.13.0+cu130` |
| FlashInfer | `0.7.0rc1` |
| Image ID | `sha256:10cd74096eb272a2d9f1ba36e232d891d9c772399f8d63c9f00e52ba5af189a0` |
| Context / scheduler | max model length 300,000; max sequences 8; max batched tokens 8,192 |
| Decode | DSpark block 5; full and piecewise CUDA graphs |
| Engram | verified rank-local sparse copies; full checkpoint remains on BeeGFS |
| Memory | GPU utilization 0.80; 112 GiB container memory ceiling |

The repository's exact nightly image could not build its FlashInfer overlay on the available ARM64
base. The successful run used the repository's ARM64 fallback image, upgraded FlashInfer to 0.7.0rc1,
prebuilt the MXFP8 and sparse-MLA kernels, and ported the seven repository patches to the fallback
image's newer `deepseek_v4_1` module layout. The Engram patch was three-way merged so the current
data-parallel/ubatch interfaces and the repository's disk stager coexist. All four ranks loaded both
Engram layers from node-local storage.

## Single-stream decode

These are three independent requests per prompt. Decode TPS uses the server-reported completion token
count, subtracts the first token, and measures from first token through EOS. TTFT and end-to-end output
TPS remain separate.

| Prompt | vLLM decode median | vLLM TTFT median | vLLM end-to-end output median | SGLang 4-node decode median |
|---|---:|---:|---:|---:|
| Count 1–80 | **91.98 tok/s** | 0.192 s | 86.50 tok/s | 71.04 tok/s |
| Python code | **75.54 tok/s** | 0.253 s | 69.31 tok/s | 57.92 tok/s |
| Explanatory prose | **30.79 tok/s** | 0.223 s | 29.24 tok/s | 27.37 tok/s |

On these prompts, the vLLM run improves median single-stream decode over the measured four-node
SGLang run by 29.5%, 30.4%, and 12.5%, respectively. These are prompt-specific measurements, not a
general model quality score.

## C1–C8 aggregate decode

| Prompt | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Count 1–80 | 81.54 | 146.89 | 174.77 | 222.07 | 261.91 | 291.03 | 320.38 | **341.26** |
| Python code | 64.68 | 107.58 | 138.60 | 195.54 | 186.84 | 217.03 | 239.00 | **237.45** |
| Explanatory prose | 31.96 | 40.44 | 58.44 | 66.67 | 79.82 | 82.27 | 95.66 | **95.55** |

Values are aggregate delivered completion tokens per second. At C8, this vLLM run exceeds the
four-node SGLang measurement by 20.1% for counting, 14.2% for code, and 8.8% for prose. Per-stream
decode at C8 is 46.09, 32.64, and 13.42 tok/s; the distinction matters because aggregate throughput
rises while each individual stream slows down.

## Prefill and validation

| Prompt tokens | Client TTFT | Effective prefill | SGLang 4-node effective prefill |
|---:|---:|---:|---:|
| 2,950 | 1.963 s | **1,502.7 tok/s** | 256.5 tok/s |
| 5,853 | 3.517 s | **1,664.4 tok/s** | 252.9 tok/s |

Effective prefill is prompt tokens divided by client-observed TTFT; it includes scheduling, network,
and first-token overhead. Three cold unique-prefix retrieval checks at 1,502, 2,922, and 5,914 prompt
tokens all returned their hidden record codes exactly. The 18-request functional and length suite also
passed its automated checks. The C1–C8 matrix completed all 24 cells and 108 requests, and every
content check passed. All four containers remained running with `OOMKilled=false` after the sweep.

Compact results and validation receipts are in [`results/vllm-crosscheck/`](../results/vllm-crosscheck/).
Full container logs and raw SSE receipts remain in the local experiment archive because they include
machine-specific paths and are not required to reproduce the summarized measurements.

## Eight-node result

The eight-node service passed the same 18 functional/length checks, all three
unique-prefix retrieval checks at 1,502/2,922/5,914 prompt tokens, and all 108
matrix requests.  All eight containers remained running with
`OOMKilled=false` after the sweep.

| Prompt | 4-node C1 median | 8-node C1 median | Change | 4-node C8 aggregate | 8-node C8 aggregate | Change |
|---|---:|---:|---:|---:|---:|---:|
| Count 1–80 | 91.98 | **108.03** | +17.4% | 341.26 | **364.37** | +6.8% |
| Python code | 75.54 | **83.24** | +10.2% | 237.45 | **273.91** | +15.4% |
| Explanatory prose | 30.79 | **37.53** | +21.9% | 95.55 | **103.38** | +8.2% |

The fixed three-repeat 8-node C1 series was 108.03/109.09/94.89 tok/s for
counting, 84.76/83.24/83.00 for code, and 41.97/37.38/37.53 for prose.  Median
TTFT was 0.181/0.227/0.152 seconds.  These data show a useful C1 decode gain,
but still far below linear scaling from doubling the nodes because TP8/EP8
adds collectives to each decode step and FlashInfer all-reduce is unavailable
at world size 8 in this build.

| Prompt tokens | 4-node effective prefill | 8-node effective prefill | Change |
|---:|---:|---:|---:|
| 2,950 | 1,502.7 tok/s | 1,461.6 tok/s | -2.7% |
| 5,853 | 1,664.4 tok/s | 1,685.3 tok/s | +1.3% |

Prefill is effectively flat from four to eight nodes for these shapes.  The
added tensor-parallel communication offsets the smaller per-rank compute.
Compact eight-node receipts are in
[`results/vllm-crosscheck-8node/`](../results/vllm-crosscheck-8node/).
