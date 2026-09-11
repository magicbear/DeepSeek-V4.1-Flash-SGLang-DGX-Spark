# Spark patches

- `engram_backend.py` implements exact safetensors-backed Engram row reads, ARM64 CUDA runtime discovery, bounded per-rank caching, and validation of node-local shard copies.
- `row_store.cpp` implements the native row store used by the Python adapter.
- `sitecustomize.py` installs the Engram replacement, fixes the SM120/121 DeepGEMM metadata guard, and selects Triton for sparse-MLA prefill while retaining FlashInfer for short decode queries.

Copy all three files together. `sitecustomize.py` imports `engram_backend.py`, which loads the compiled `librow_store.so` from the same directory.
