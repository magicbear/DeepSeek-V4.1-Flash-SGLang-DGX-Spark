## spark4-fi070-hybrid-c1024-final (2026-09-11T04:45:14Z)

SGLang FI 0.7 direct DSV4 decode; Triton prefill; chunk 1024; TP4 EP4 PP1; DSpark5; graphs BS1-8; explicit prefill warmup.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (2 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 39.02 | 42.03 | 0.28 |
| C2 | 62.66 | 36.23 | 0.539 |
| C3 | 75.09 | 29.07 | 0.663 |
| C4 | 94.75 | 28.82 | 0.827 |
| C5 | 113.94 | 27.24 | 0.885 |
| C6 | 123.05 | 25.77 | 1.134 |
| C7 | 138.26 | 24.63 | 1.112 |
| C8 | 141.11 | 21.79 | 1.218 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|
| coding | 65.0 | 55.13 | 45.54 | 41.92 | 40.4 | 39.88 | 38.27 | 32.54 |
| prose | 19.07 | 17.34 | 12.6 | 15.71 | 14.09 | 11.65 | 10.99 | 11.04 |
| ceiling_count | 72.96 | 67.75 | 60.55 | 61.23 | 53.34 | 52.71 | 46.6 | 42.83 |

### Effective prefill (unique prefix; shape warmup recorded separately)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 8.486 | 347.6 |
| 4000 | 5853 | 16.667 | 351.2 |
