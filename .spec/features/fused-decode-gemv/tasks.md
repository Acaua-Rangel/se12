# Tasks: GPU decode and fused decode+GEMV kernels (CUDA C, any NVIDIA GPU sm_60+)

> feature: fused-decode-gemv

<!--
  Lane ordering: T-011 → T-004 share src/lws/adapters/cuda/nvrtc_compiler.py;
  T-005 → T-012 → T-006 share src/lws/adapters/cuda/gemv.py.
  All Python code follows P-013 (hexagonal) and P-014 (object calisthenics);
  CUDA C kernels are adapter assets and are exempt from P-014.
-->

## T-011 — Device doctor and arch-aware NVRTC compiler [concluida]

- Refs: US-012, AC-021, AC-022
- Files: src/lws/domain/device/__init__.py, src/lws/domain/device/properties.py, src/lws/domain/device/decisions.py, src/lws/application/ports/gpu/__init__.py, src/lws/application/ports/gpu/gpu_probe.py, src/lws/application/ports/gpu/kernel_compiler.py, src/lws/application/gpu/__init__.py, src/lws/application/gpu/diagnose_device.py, src/lws/adapters/cuda/__init__.py, src/lws/adapters/cuda/gpu_probe.py, src/lws/adapters/cuda/nvrtc_compiler.py, src/lws/device.py, tests/test_spec_device.py
- Model: claude-opus-5-5
- Effort: high
- Notes: The `GpuProbe` adapter is the ONLY code that reads hardware facts (principle P-012): device selection (`LWS_DEVICE`), properties via torch + `cupy.cuda.runtime.getDeviceProperties` (SM count, L2, memory clock, bus width), torch smoke test, NVRTC supported targets. It returns a `DeviceProperties` value object (a composition — identity, compute, memory — to respect P-014 rule 8). All decisions are pure domain rules over it: GPU slug (`<lowercased name, non-alnum → '-'>-sm<major><minor>`), compile target (SASS → PTX fallback → refuse), compute profile (`bf16-native` if cc ≥ 8.0 else `fp32-act`), VRAM fit per feature/model profile — so the CPU tests cover sm_52/60/75/86/90/120 with fake properties and no GPU. The `KernelCompiler` adapter compiles with CuPy RawModule using the chosen target, honors `LWS_FORCE_PTX=1`, caches binaries under `$LWS_CACHE_DIR` (default `~/.cache/lws`) keyed by source hash + NVRTC version + target, and launches on torch's current stream via `cupy.cuda.ExternalStream`. `lws.device` is the composition root of the doctor. The compile-target test carries @principle:P-008. GPU tests marked `gpu`.

## T-004 — CUDA C decode-only kernel [concluida]

- Refs: US-004, AC-009, AC-022
- Files: src/lws/adapters/cuda/kernels/se12_decode.cu, src/lws/adapters/cuda/kernels/se12_common.cuh, src/lws/application/ports/gpu/weight_decoder.py, src/lws/adapters/cuda/se12_decoder.py, tests/test_spec_gpu_decode.py, src/lws/adapters/cuda/nvrtc_compiler.py
- Model: claude-opus-5-5
- Effort: high
- Notes: src/lws/adapters/cuda/nvrtc_compiler.py is a read-only dependency (runs after T-011). Needs weight-codec done (`lws.domain.se12` is the oracle). The `WeightDecoder` port has this CUDA adapter and a CPU fake backed by the domain codec. The adapter crosses torch ↔ CuPy via DLPack and can decode either a whole tensor or only a given set of row tiles (used by the compressed embedding lookup, weight-codec Q-002). se12_common.cuh holds the device decode function reused by T-005 (codebook pre-shifted to `exp << 23`, see design.md). No arch-specific code needed here. Tests marked `gpu`; the bit-exact test also carries @principle:P-004 and is parametrized over the SASS and the forced-PTX target (AC-022's last clause).

## T-005 — Raw-BF16 baseline kernel and fused SE12 decode + GEMV kernel [concluida]

- Refs: US-005, AC-010, AC-011, AC-032
- Files: src/lws/adapters/cuda/kernels/gemv_raw_bf16.cu, src/lws/adapters/cuda/kernels/gemv_se12.cu, src/lws/domain/launch/__init__.py, src/lws/domain/launch/heuristic.py, src/lws/application/ports/gpu/matvec_kernel.py, src/lws/adapters/cuda/gemv.py, tests/test_spec_fused_gemv.py, src/lws/adapters/cuda/kernels/se12_common.cuh
- Model: claude-opus-5-5
- Effort: max
- Notes: src/lws/adapters/cuda/kernels/se12_common.cuh is a read-only dependency (runs after T-004). Write BOTH kernels from the same skeleton (principle P-009): tune the raw one first, then swap its load step for load+decode. Activations FP32 or BF16 (exact upcast), output always FP32 (principle P-005). Any `__CUDA_ARCH__`-guarded load path must exist in both kernels, with a generic path for sm_60. The default launch configuration is a pure domain rule (`lws.domain.launch`) over `DeviceProperties` and the shape; the `MatVecKernel` adapter applies it (or a tuned one when present). FP64 reference in VRAM-sized row chunks. Kernels must be capture-safe (no host sync, no allocation; output buffer passed in). The launch path is a hot path: P-014 rules 3/5/8/9 are relaxed there (constitution). The test asserting identical launch structure carries @principle:P-009. Reduction order is fully determined by the launch configuration (no atomics, fixed warp-reduction tree), so the same configuration gives bit-identical outputs (AC-032). gemv_se12.cu also takes a compile-time `LWS_EXTRA_DECODE_OPS` (default 0) used only by performance-model AC-027 — k dependent integer ops per weight in inline PTX that leave the decoded value unchanged.

## T-012 — Per-GPU launch tuner [concluida]

- Refs: US-013, AC-024
- Files: src/lws/domain/launch/search_space.py, src/lws/application/ports/gpu/tuning_cache.py, src/lws/application/gpu/tune_launch.py, src/lws/adapters/filesystem/__init__.py, src/lws/adapters/filesystem/tuning_cache.py, src/lws/tune.py, tests/test_spec_tune.py, src/lws/adapters/cuda/gemv.py
- Model: claude-sonnet-5
- Effort: medium
- Notes: src/lws/adapters/cuda/gemv.py is a read-only dependency (runs after T-005). Search space and best-choice rule are domain; the use case runs candidates through the `MatVecKernel` port and saves through the `TuningCache` port (JSON file adapter). Same search space for raw and fused, each keeps its own best. Writes `tune-<gpu-slug>.json`; a file with another slug is ignored with a warning. `lws.tune` is the composition root (`python -m lws.tune`). The heuristic-only test runs with an empty cache dir.

## T-006 — Microbenchmark, per-GPU verdict and cross-GPU summary [pending]

- Refs: US-006, US-007, AC-012, AC-013, AC-014, AC-023
- Files: src/lws/domain/benchmark/__init__.py, src/lws/domain/benchmark/statistics.py, src/lws/domain/benchmark/verdict.py, src/lws/domain/benchmark/summary.py, src/lws/application/ports/gpu/benchmark_timer.py, src/lws/application/ports/gpu/report_store.py, src/lws/application/gpu/run_microbenchmark.py, src/lws/application/gpu/summarize_reports.py, src/lws/adapters/cuda/benchmark_timer.py, src/lws/adapters/filesystem/report_store.py, src/lws/bench/__init__.py, src/lws/bench/micro.py, src/lws/bench/summary.py, tests/test_spec_microbench.py, src/lws/adapters/cuda/gemv.py
- Model: claude-sonnet-5
- Effort: high
- Notes: src/lws/adapters/cuda/gemv.py is a read-only dependency (runs after T-012). Domain: latency statistics (median/p10/p90), bandwidth ratio, the GO / NO-GO / INCONCLUSIVE rule and the summary table — all testable on CPU with fake timings. The `BenchmarkTimer` CUDA adapter owns CUDA events, graph capture of stacked mode, L2 flush (buffer ≥ 2× queried L2) and the peak-copy measurement; the report metadata (P-006) is assembled by the use case from `DeviceProperties` + timer facts, so T-009 reuses it through the same ports. cuBLAS BF16 baseline only when the compute profile is `bf16-native`; the verdict uses the faster baseline (test carries @principle:P-009). Report metadata test carries @principle:P-006; the "rejected, not timed" test carries @principle:P-007; the test that verdicts are stored and read back keyed by GPU slug carries @principle:P-011. Keep only one variant's weights resident at a time. Output names use the GPU slug (micro-<slug>.json, verdict-<slug>.json).
