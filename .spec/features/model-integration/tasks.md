# Tasks: Model integration (drop-in compressed Linear layers)

> feature: model-integration

<!--
  All Python code follows P-013 (hexagonal) and P-014 (object calisthenics).
  `lws.runtime` is the composition root that exposes `load_packed`.
-->

## T-007 — CompressedLinear and packed-model loader [pending]

- Refs: US-008, US-014, AC-015, AC-016, AC-025
- Files: src/lws/adapters/transformers/compressed_embedding.py, src/lws/domain/dispatch/__init__.py, src/lws/domain/dispatch/kernel_path.py, src/lws/application/ports/runtime/__init__.py, src/lws/application/ports/runtime/verdict_store.py, src/lws/application/runtime/__init__.py, src/lws/application/runtime/load_packed_model.py, src/lws/adapters/transformers/__init__.py, src/lws/adapters/transformers/compressed_linear.py, src/lws/adapters/transformers/model_builder.py, src/lws/adapters/filesystem/verdict_store.py, src/lws/runtime/__init__.py, tests/test_spec_loader.py
- Model: claude-sonnet-5
- Effort: high
- Notes: The transformers adapter builds the model on the meta device, then materializes only non-Linear params + SE12 buffers. Modes: "exact" (decode + F.linear), "fused" (T-005 kernel) and "auto". The dispatch rule (prefill → exact; decode batch size → fused | exact from the verdicts of the current GPU slug) is pure domain, evaluated ONCE at load time into a per-layer table, so `CompressedLinear.forward` never calls domain code per token; the dispatch test with fake verdicts for this and another slug runs on CPU and carries @principle:P-011. Compute profile comes from the domain device rules, never from a GPU name. `CompressedLinear.forward` is a hot path (P-014 rules 3/5/8/9 relaxed) and must stay CUDA-graph capturable (no host sync, no `.item()`, no CuPy pool allocation). `CompressedEmbedding` gathers the tiles of the requested token rows through the `WeightDecoder` port (row-tile decode from T-004) and serves as the LM head through the GEMV path. Also expose `load_reference(model_dir)` (original weights through the raw-BF16 kernel) and the `order="matched"` option of the fused mode, both consumed by T-008 for AC-033. Tests marked `gpu` and `model` where they need them.

## T-008 — Generation equivalence harness [pending]

- Refs: US-009, AC-017, AC-033
- Files: src/lws/domain/equivalence/__init__.py, src/lws/domain/equivalence/divergence.py, src/lws/application/ports/runtime/text_generator.py, src/lws/application/runtime/check_equivalence.py, src/lws/adapters/transformers/generator.py, tests/test_spec_equivalence.py, tests/prompts/equivalence_prompts.jsonl, src/lws/runtime/__init__.py
- Model: claude-sonnet-5
- Effort: medium
- Notes: src/lws/runtime/__init__.py is a read-only dependency (runs after T-007). The comparison rule (identical, or first divergence at a near-tie < 0.05, and the ≥ 95% threshold) is pure domain (`lws.domain.equivalence`); the use case feeds it through the `TextGenerator` port; the transformers adapter does greedy decoding and returns tokens plus top-2 logits. The prompt set is fixed and versioned in the repo. AC-033 reuses the same harness with the reference runtime and requires zero divergences.

## T-009 — End-to-end benchmark [pending]

- Refs: US-010, AC-018
- Files: src/lws/application/runtime/run_end_to_end.py, src/lws/bench/e2e.py, tests/test_spec_e2e_bench.py, src/lws/runtime/__init__.py
- Model: claude-sonnet-5
- Effort: medium
- Notes: src/lws/runtime/__init__.py is a read-only dependency (runs after T-008). Reuse the benchmark statistics (`lws.domain.benchmark`), the `BenchmarkTimer` and `ReportStore` ports and their adapters by import (do not edit them). Graph capture through transformers' static cache (`cache_implementation="static"`, hybrid cache for Gemma 2) or a manual `torch.cuda.CUDAGraph` of one decode step, inside the transformers adapter. Check VRAM fit with the domain device rules before loading each variant; a variant that does not fit is recorded, not crashed on. `lws.bench.e2e` is the composition root.
