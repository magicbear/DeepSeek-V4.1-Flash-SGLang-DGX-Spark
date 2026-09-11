"""Verify the checked-in 4/8-node matrices and every referenced raw receipt."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "sglang-final"
report = {}

for nodes in (4, 8):
    matrix_path = RESULTS / f"bench-spark{nodes}-hybrid.json"
    matrix = json.loads(matrix_path.read_text())
    assert matrix["status"] == "completed"
    assert matrix["valid_for_performance_comparison"] is True
    expected = {
        (category, concurrency)
        for category in ("coding", "prose", "ceiling_count")
        for concurrency in range(1, 9)
    }
    assert {(batch["category"], batch["c"]) for batch in matrix["batches"]} == expected
    assert len(matrix["batches"]) == 24

    requests = 0
    raw_paths = set()
    for batch in matrix["batches"]:
        assert len(batch["requests"]) == batch["c"]
        assert batch["agg_tok_s"] > 0
        for request in batch["requests"]:
            assert request["output_check"] is True
            raw_path = RESULTS / request["raw_file"]
            assert raw_path.is_file()
            raw = json.loads(raw_path.read_text())
            assert raw["result"]["text"] == request["text"]
            assert raw["result"]["completion_tokens"] == request["completion_tokens"]
            assert math.isfinite(request["decode_tok_s"]) and request["decode_tok_s"] > 0
            expected_tps = (request["completion_tokens"] - 1) / (
                request["finish_elapsed_s"] - request["ttft_s"]
            )
            assert abs(request["decode_tok_s"] - expected_tps) < 1e-7
            raw_paths.add(raw_path)
            requests += 1

    assert requests == 108
    assert len(matrix["prefill"]) == 2
    for prefill in matrix["prefill"]:
        assert prefill["prompt_tokens"] in (2950, 5853)
        assert prefill["ttft_s"] > 0 and prefill["prefill_tok_s"] > 0
        raw_path = RESULTS / prefill["raw_file"]
        assert raw_path.is_file()
        raw_paths.add(raw_path)

    for warmup in matrix.get("prefill_shape_warmups", []):
        raw_path = RESULTS / warmup["raw_file"]
        assert raw_path.is_file()
        raw_paths.add(raw_path)

    report[str(nodes)] = {
        "matrix_cells": 24,
        "requests": requests,
        "raw_receipts": len(raw_paths),
        "prefill_points": 2,
        "passed": True,
    }

published = json.loads((RESULTS / "final-data-verification.json").read_text())
assert all(published[str(nodes)]["passed"] for nodes in (4, 8))
print(json.dumps(report, indent=2))
