# Tasks: Performance model (predict when lossless streaming speeds up decode)

> feature: performance-model

<!--
  Runs after fused-decode-gemv (needs its kernels and microbenchmark).
  T-013 → T-014 share src/lws/domain/performance/roofline.py.
  All Python code follows P-013 (hexagonal) and P-014 (object calisthenics).
-->

## T-013 — Calibration microkernels and roofline model [pending]

- Refs: US-015, AC-026
- Files: src/lws/domain/performance/__init__.py, src/lws/domain/performance/roofline.py, src/lws/application/ports/gpu/calibration_probe.py, src/lws/application/gpu/calibrate_device.py, src/lws/adapters/cuda/calibration_probe.py, src/lws/adapters/cuda/kernels/calibration.cu, src/lws/bench/calibrate.py, tests/test_spec_performance_model.py, src/lws/adapters/cuda/gemv.py
- Model: claude-opus-5-5
- Effort: high
- Notes: src/lws/adapters/cuda/gemv.py is a read-only dependency (runs after fused-decode-gemv T-006). The model (design.md) is pure domain and is tested on CPU with synthetic calibrations (known inputs → known ratio and crossover). calibration.cu holds INT32/FP32 throughput microkernels; cache-resident runs reuse the raw and fused kernels through the `MatVecKernel` port. Calibration report carries @principle:P-006. `lws.bench.calibrate` is the composition root.

## T-014 — Decode-cost sweep and cross-GPU prediction [pending]

- Refs: US-015, AC-027, AC-028
- Files: src/lws/domain/performance/validation.py, src/lws/application/gpu/validate_model.py, src/lws/bench/predict.py, tests/test_spec_prediction.py, src/lws/domain/performance/roofline.py
- Model: claude-opus-5-5
- Effort: high
- Notes: src/lws/domain/performance/roofline.py is a read-only dependency (runs after T-013). Uses the `LWS_EXTRA_DECODE_OPS` knob of gemv_se12.cu (added in fused-decode-gemv T-005); every k variant must pass the bit-exact check before timing (test carries @principle:P-007). Error metrics (MAPE, crossover difference) and the "prediction written before measurement" check (compare report timestamps) are domain rules. `lws.bench.predict` is the composition root (subcommands `sweep` and `cross`).
