# FlashInfer 0.7 decode optimization on four DGX Spark nodes

This experiment isolates the main SGLang decode gap observed against the
four-node vLLM reference.  It uses the same four-node TP4/EP4/PP1 topology,
DSpark block 5, local Engram rows, 8,192-token context and KV limits, and
decode CUDA graphs for batch sizes 1 through 8.

## Change

The base image ships FlashInfer 0.6.18.  A derived image replaces only its
Python FlashInfer package with ARM64 FlashInfer 0.7.0rc1.  The SGLang SM120
adapter in `patches/flash_mla_sm120.py` then:

1. calls `sparse_mla_sm120_decode_dsv4` directly for decode batches of at most
   64 tokens, avoiding the generic dispatcher crossing into an unsupported
   DSV4 prefill specialization during CUDA graph capture;
2. keeps the validated Triton sparse-MLA implementation for prefill query
   batches above 64 tokens when `DSV41_PREFILL_TRITON=1`;
3. preserves SGLang's existing 256-to-64-token page conversion and separate
   main/extra cache buffers.

The production measurement uses chunked prefill size 1,024.  Increasing it to
8,192 reduced warmed prefill to roughly 78-89 tok/s because the current Triton
kernel scales poorly with the larger query batch, so that candidate was
rejected.

## Validation

The final hybrid service passed all functional, prompt-length and unique-prefix
retrieval checks.  Decode CUDA graph capture succeeded for batch sizes 1-8.
The 2,950- and 5,853-token prefill shapes were explicitly warmed before their
reported measurements to exclude first-shape Triton compilation.

## Results

| Metric | FlashInfer 0.6.18 baseline | 0.7.0rc1 hybrid | Change |
|---|---:|---:|---:|
| C1 count decode, median | 71.04 tok/s | 77.66 tok/s | +9.3% |
| C1 code decode, median | 57.92 tok/s | 58.17 tok/s | +0.4% |
| C1 prose decode, median | 27.37 tok/s | 16.86 tok/s | -38.4% |
| 2,950-token effective prefill | 256.5 tok/s | 347.6 tok/s | +35.5% |
| 5,853-token effective prefill | 252.9 tok/s | 351.2 tok/s | +38.9% |

The prose regression is correlated with DSpark acceptance rather than a
uniform decode-kernel slowdown: observed accepted lengths/rates were about
2.08-3.92 tokens and 21-58% for prose, versus 5.5-5.88 tokens and 90-97% for
counting.  A chunk-8192 comparison after sustained traffic measured C1 medians
of 77.77/63.40/30.16 tok/s for count/code/prose, confirming that prompt and
speculative-acceptance variance is material.  The fixed final run is retained
as the reproducible result; category medians must not be collapsed into one
claim about the attention kernel.

Final C8 aggregate decode was 281.51 tok/s for count, 204.97 tok/s for code,
and 77.25 tok/s for prose.  FlashInfer 0.7 improves the strongest single-stream
decode case and warmed prefill, but this SGLang path still trails the reproduced
four-node vLLM result (91.98/75.54/30.79 tok/s for the same three C1 prompts).

## Remaining limits

- TP=EP across nodes requires expert collectives for every decode step; doubling
  the node count adds communication latency and cannot provide linear C1 gains.
- FlashInfer all-reduce support and fused shared-expert paths are topology and
  implementation dependent.
- DSpark speed is highly prompt dependent because rejected draft tokens still
  consume verification work.
- First use of a new Triton prefill shape includes JIT compile/autotune time and
  should be reported separately from warmed throughput.
