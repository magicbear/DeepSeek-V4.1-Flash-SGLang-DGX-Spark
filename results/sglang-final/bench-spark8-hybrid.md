## spark8-hybrid (2026-09-10T19:18:36Z)

spark8-hybrid; TP=EP=node count, PP1, DSpark5, decode graphs BS1-8, local Engram, context/KV 8192.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (2 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 44.62 | 47.73 | 0.236 |
| C2 | 66.68 | 40.32 | 0.621 |
| C3 | 89.05 | 33.84 | 0.523 |
| C4 | 114.89 | 32.63 | 0.545 |
| C5 | 122.63 | 28.26 | 0.709 |
| C6 | 141.38 | 28.58 | 0.843 |
| C7 | 157.02 | 27.07 | 0.853 |
| C8 | 150.2 | 22.2 | 0.873 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|
| coding | 65.62 | 58.82 | 47.95 | 46.69 | 40.14 | 41.47 | 39.38 | 32.39 |
| prose | 29.83 | 21.82 | 19.72 | 18.57 | 16.38 | 15.69 | 14.75 | 12.02 |
| ceiling_count | 70.87 | 75.23 | 68.01 | 60.1 | 57.81 | 47.11 | 45.86 | 42.14 |

### Effective prefill (unique prefix; shape warmup recorded separately)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 7.062 | 417.7 |
| 4000 | 5853 | 13.182 | 444.0 |
