## vllm-repo-spark8-065 (2026-09-11T05:54:43Z)

8 DGX Spark; TP8 PP1 inferred EP8; vLLM repo adaptation; DSpark k=5; max context 300000; GPU memory utilization 0.65; local Engram; 200GbE RoCE.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (2 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 54.02 | 57.99 | 0.22 |
| C2 | 96.34 | 54.34 | 0.291 |
| C3 | 128.52 | 47.17 | 0.306 |
| C4 | 141.27 | 39.62 | 0.39 |
| C5 | 155.41 | 34.64 | 0.36 |
| C6 | 173.59 | 31.77 | 0.391 |
| C7 | 188.59 | 29.87 | 0.407 |
| C8 | 188.65 | 26.49 | 0.479 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|
| coding | 78.41 | 76.85 | 66.65 | 57.1 | 49.45 | 46.65 | 42.76 | 38.39 |
| prose | 37.57 | 31.83 | 27.69 | 22.13 | 19.82 | 16.88 | 16.98 | 14.59 |
| ceiling_count | 100.37 | 91.96 | 78.67 | 73.24 | 64.67 | 59.43 | 54.97 | 49.83 |

### Effective prefill (unique prefix; shape warmup recorded separately)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 2.018 | 1461.6 |
| 4000 | 5853 | 3.473 | 1685.3 |
