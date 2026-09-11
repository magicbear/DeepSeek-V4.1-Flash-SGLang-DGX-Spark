## spark4-hybrid (2026-09-10T19:02:38Z)

spark4-hybrid; TP=EP=node count, PP1, DSpark5, decode graphs BS1-8, local Engram, context/KV 8192.

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (2 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 37.19 | 39.87 | 0.293 |
| C2 | 60.14 | 33.8 | 0.481 |
| C3 | 73.16 | 29.8 | 0.796 |
| C4 | 96.63 | 28.99 | 0.717 |
| C5 | 105.03 | 25.45 | 0.891 |
| C6 | 128.68 | 25.62 | 0.952 |
| C7 | 134.36 | 23.28 | 1.117 |
| C8 | 147.84 | 21.98 | 1.14 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|
| coding | 55.7 | 48.38 | 43.7 | 40.48 | 36.92 | 37.58 | 33.18 | 31.53 |
| prose | 24.04 | 19.23 | 15.91 | 17.51 | 13.98 | 13.67 | 13.38 | 12.42 |
| ceiling_count | 68.08 | 61.79 | 54.9 | 53.68 | 51.44 | 47.98 | 45.29 | 41.88 |

### Effective prefill (unique prefix; shape warmup recorded separately)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 11.502 | 256.5 |
| 4000 | 5853 | 23.141 | 252.9 |
