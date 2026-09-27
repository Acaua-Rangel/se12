# Tasks: Weight offload (stream lossless-compressed weights over PCIe)

> feature: weight-offload

<!--
  Runs after model-integration (reuses CompressedLinear and the order-matched
  kernels). T-015 → T-016 share src/lws/offload.py.
  All Python code follows P-013 (hexagonal) and P-014 (object calisthenics).
-->

## T-015 — Layer placement and streaming pipeline [pending]

- Refs: US-016, AC-029
- Files: src/lws/domain/offload/__init__.py, src/lws/domain/offload/placement.py, src/lws/application/ports/runtime/weight_streamer.py, src/lws/application/runtime/load_offloaded_model.py, src/lws/adapters/cuda/weight_streamer.py, src/lws/adapters/transformers/offloaded_layers.py, src/lws/offload.py, tests/test_spec_offload.py, src/lws/adapters/transformers/compressed_linear.py
- Model: claude-opus-5-5
- Effort: high
- Notes: src/lws/adapters/transformers/compressed_linear.py is a read-only dependency (runs after model-integration T-007). Placement and streamed-bytes prediction are pure domain, tested on CPU with fake layer sizes (design.md table as test cases). The CUDA streamer owns pinned host blocks, two staging buffers, the copy stream and events; the transformers adapter swaps streamed layers for thin modules that wait on the copy event and then call the same kernel path as resident layers. VRAM cap via `torch.cuda.set_per_process_memory_fraction` in the composition root. The bit-identity test (BF16 vs SE12 streamed, order-matched kernels) is marked `gpu` and `model`.

## T-016 — Offload benchmark and llama.cpp reference [pending]

- Refs: US-016, AC-030, AC-031
- Files: src/lws/application/runtime/run_offload_benchmark.py, src/lws/adapters/external/__init__.py, src/lws/adapters/external/llama_cpp_runner.py, src/lws/bench/offload.py, tests/test_spec_offload_bench.py, src/lws/offload.py
- Model: claude-sonnet-5
- Effort: medium
- Notes: src/lws/offload.py is a read-only dependency (runs after T-015). Reuse the benchmark statistics, `BenchmarkTimer` and `ReportStore` (do not edit them); report carries @principle:P-006. The llama.cpp adapter runs `llama-bench` / `llama-cli` as a subprocess with the matching `-ngl`, records version and full command lines, and returns "not runnable: <reason>" instead of raising when the build or the GGUF conversion fails (ASM-021). The "≤ 0.80× when transfer-bound" rule is a domain rule over the recorded numbers.
