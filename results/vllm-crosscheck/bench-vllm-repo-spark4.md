## vllm-repo-spark4 (2026-09-11T02:01:13Z)

4 DGX Spark; TP4 PP1 inferred EP; vLLM repo adaptation; DSpark k=5; max context 300000; GPU memory utilization 0.80; local Engram; 200GbE RoCE.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (2 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 48.32 | 52.8 | 0.287 |
| C2 | 74.01 | 40.56 | 0.363 |
| C3 | 98.52 | 35.83 | 0.339 |
| C4 | 131.1 | 37.07 | 0.343 |
| C5 | 133.33 | 29.49 | 0.385 |
| C6 | 149.65 | 27.59 | 0.39 |
| C7 | 167.33 | 26.16 | 0.423 |
| C8 | 166.5 | 23.03 | 0.416 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|
| coding | 71.55 | 58.84 | 51.04 | 55.62 | 41.35 | 39.67 | 37.63 | 32.64 |
| prose | 34.06 | 22.29 | 20.62 | 18.52 | 17.63 | 15.5 | 14.68 | 13.42 |
| ceiling_count | 88.68 | 79.24 | 63.65 | 60.32 | 57.58 | 52.7 | 49.15 | 46.09 |

### Effective prefill (unique prefix; shape warmup recorded separately)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 1.963 | 1502.7 |
| 4000 | 5853 | 3.517 | 1664.4 |
