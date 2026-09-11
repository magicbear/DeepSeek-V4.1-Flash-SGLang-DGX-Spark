#!/usr/bin/env python3
"""Single-GPU flip probe (Bluey). Alternates 10 s blocks, twice, recording every sample with epoch time:
  gemv_duty : decode-shaped GEMV (6 x 5120 @ 5120 x 16384 bf16) in ~70 ms bursts, 30 ms idle
  gemv_cont : same GEMV back-to-back
  mm_duty   : SM-bound fp16 matmul (2048^3) in ~70 ms bursts, 30 ms idle
  copy_cont : 64 MB device copy (all-SM, memory-bound)
If gemv and mm drop together while copy holds, the slow mode is an SM clock drop, not memory.
Output: one line per sample 'S <epoch> <kind> <value>' plus per-block summaries."""
import time

import torch

torch.cuda.set_device(0)
W = torch.randn(5120, 16384, dtype=torch.bfloat16, device="cuda")
a = torch.randn(6, 5120, dtype=torch.bfloat16, device="cuda")
A = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")
B = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")
x = torch.empty(32 * 2**20, dtype=torch.bfloat16, device="cuda")
y = torch.empty_like(x)


def timed(fn, n):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / n


KINDS = {
    "gemv_duty": (lambda: a @ W, 90, 2 * W.numel() / 1e6, True, "GB/s"),
    "gemv_cont": (lambda: a @ W, 8, 2 * W.numel() / 1e6, False, "GB/s"),
    "mm_duty": (lambda: A @ B, 40, 2 * 2048**3 / 1e9, True, "TFLOPS"),
    "copy_cont": (lambda: y.copy_(x), 8, 4 * x.numel() / 1e6, False, "GB/s"),
}
for f, *_ in KINDS.values():
    for _ in range(5):
        f()
torch.cuda.synchronize()
print(f"START {time.time():.3f}", flush=True)
for rnd in range(2):
    for kind, (fn, n, work, duty, unit) in KINDS.items():
        vals, t0 = [], time.time()
        while time.time() - t0 < 10:
            ms = timed(fn, n)
            v = work / ms  # GB/s (bytes/1e6/ms) or TFLOPS (flop/1e9/ms)
            vals.append(v)
            print(f"S {time.time():.3f} {kind} {v:.1f}", flush=True)
            if duty:
                time.sleep(0.03)
        s = sorted(vals)
        print(f"BLOCK r{rnd} {kind:9} {unit:6} n={len(s):4d} min={s[0]:6.1f} p10={s[len(s) // 10]:6.1f} "
              f"p50={s[len(s) // 2]:6.1f} max={s[-1]:6.1f}", flush=True)
print(f"END {time.time():.3f}", flush=True)
