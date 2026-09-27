# Design: performance model

> feature: performance-model

## Model

Per GEMV call with W weights, batch B, on one GPU:

```
t_mem(format)     = bytes_per_weight(format) · W / BW_stream
t_compute(format) = W · c(format, B)          # c = measured compute-bound cost per weight
t(format)         ≈ max(t_mem, t_compute) + (1 − η) · min(t_mem, t_compute)
ratio             = t(SE12) / t(baseline)
```

- `bytes_per_weight`: 2 for raw BF16, 12.03/8 for SE12 (weight-codec design.md).
- `BW_stream`: AC-013 peak copy bandwidth.
- `c(format, B)`: measured by running the kernel on cache-resident data (a
  small set of tiles re-read many times), so memory is not the limit (AC-026).
- `η ∈ [0, 1]`: overlap efficiency — how much of the shorter phase hides under
  the longer one. Fitted ONCE per GPU on the raw kernel only (never on the
  fused one), then reused: the fused prediction is out-of-sample.

The crossover is where `t_compute(SE12)` grows past the saved memory time. The
model is deliberately small: every term is measured, one parameter (η) is
fitted, and the fit never sees the fused kernel.

## Controlled experiment (AC-027)

The fused kernel exposes a compile-time macro `LWS_EXTRA_DECODE_OPS` (default
0). For k > 0 each weight goes through k dependent integer operations whose
result is folded back so that the decoded value is unchanged (e.g. an
`x ^ (x ^ y)`-style chain on a register that feeds the bit reconstruction,
written in inline PTX so NVRTC cannot remove it). The variant is compiled only
by the benchmark, is never selected by the runtime, and must pass the AC-009
bit-exact check before being timed (principle P-007).

Sweeping k moves the fused kernel from memory-bound to compute-bound on the
SAME GPU, which traces the full ratio curve — the stand-in for many GPUs with
different instructions per byte when only two GPUs are available.

## Cross-GPU prediction (AC-028)

Calibration is per GPU (cheap: minutes). The model's prediction for GPU B uses
only B's calibration and device report; the fitted η of A is NOT reused on B
(η is fitted on B's raw kernel, which needs no fused measurement). The
prediction file is written before B's microbenchmark runs, and the summary
compares them.

## Code layout (P-013 / P-014)

| Concern | Layer | Where |
|---|---|---|
| roofline formula, η fit, error metrics, crossover | domain | `lws.domain.performance` |
| calibration and validation use cases | application | `lws.application.gpu` |
| instruction-throughput microkernels, cache-resident runs | adapter | `lws.adapters.cuda` (+ `kernels/calibration.cu`) |
| CLI: calibration | entrypoint | `lws.bench.calibrate` |
| CLI: `sweep` and `cross` | entrypoint | `lws.bench.predict` |
