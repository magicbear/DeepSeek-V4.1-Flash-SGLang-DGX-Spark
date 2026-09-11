# Results

- `sglang-final/bench-spark{4,8}-hybrid.json`: complete C1–C8 result sets.
- `sglang-final/raw-spark{4,8}-hybrid/`: raw request receipts referenced by the matrices.
- `sglang-final/scaling-summary.json`: headline 4/8 comparison.
- `sglang-final/final-data-verification.json`: expected coverage counts.
- `vllm-crosscheck/`: compact four-node vLLM matrix, repeated C1, correctness, retrieval and environment records.
- `vllm-crosscheck-8node/`: sanitized eight-node vLLM C1, C1-C8, prefill and validation summary.
- `sglang-fi070-4node/`: compact four-node FlashInfer 0.7 hybrid follow-up, including repeated C1, C1-C8 matrix and warmed prefill.
- `DGX-Spark-DeepSeek-V4.1-Flash-Spark-4-8机性能报告-20260910.{html,pdf}`: publication-ready report.
- `live/`: output directory for new launches and benchmark runs; its generated contents are ignored by Git.

The report and checked-in receipts are immutable measurements from 2026-09-11. New runs should use a new label under `live/` before being reviewed and promoted.
