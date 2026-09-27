# Spec: Performance model (predict when lossless streaming speeds up decode)

> feature: performance-model

## Context

This is the paper's thesis (docs/paper-plan.md, contribution C1). Published
lossless systems report isolated wins and losses — Split12 beat BF16 GEMV on an
A40 but never on B300/H100 with CUDA-core kernels; DFloat11 is slower at batch
1 when the model fits — without a model that explains WHY a given GPU wins or
loses. This feature builds that model: from a handful of quantities measured on
the GPU (achievable bandwidth, instruction throughput, decode cost per weight)
it predicts the fused ÷ baseline latency ratio for any shape and batch size,
and therefore the GO / NO-GO boundary.

Hardware constraint (independent researcher, zero budget): only Kaggle's P100
and T4 are available. With two GPUs the model is validated by (1) a
**controlled experiment** that sweeps the decode cost on the same GPU by
injecting extra integer work per weight, tracing the whole ratio curve and its
crossover through 1.0, and (2) **cross-GPU prediction**: calibrate on one GPU,
predict the other before looking at its measurements. Reports from other GPUs
(future or contributed) plug into the same summary without code changes.

## Stories

### US-015 — Predict the speedup from measured hardware facts

As the researcher, I want a performance model whose inputs are measured on the
GPU and whose output is the expected fused ÷ baseline ratio, so that the paper
explains every GO and NO-GO instead of only reporting them.

#### AC-026 — Calibration measures the model's inputs on the GPU

- **Given** a supported GPU
- **When** I run `python -m lws.bench.calibrate --out reports/calibration-<gpu-slug>.json`
- **Then** it records, with the metadata required by principle P-006: achievable streaming bandwidth (the AC-013 peak copy), sustained INT32 and FP32 instruction throughput from dedicated microkernels, and the compute-bound cost per weight of the raw and fused kernels (measured on cache-resident data so memory is not the limit) for batch sizes 1, 4 and 16
- **And** each quantity is the median of at least 100 timed runs after at least 25 warmups, with p10 and p90

#### AC-027 — Controlled experiment: predicted and measured ratio agree across a decode-cost sweep

- **Given** the calibration of one GPU and the fused kernel compiled with k extra dependent integer operations per weight, k ∈ {0, 2, 4, 8, 12, 16, 24, 32} (a test-only compile-time knob; every variant still passes the AC-009 bit-exact check)
- **When** the stacked-mode benchmark (AC-012) measures the fused ÷ raw ratio at batch size 1 for the MLP shapes and the LM head, for every k
- **Then** the report lists predicted and measured ratio per k and shape, and the mean absolute percentage error over the sweep is at most 10% on that GPU
- **And** it reports the predicted and the measured crossover k* (the smallest k whose ratio exceeds 1.0) and their difference

#### AC-028 — Cross-GPU prediction before measurement

- **Given** the calibration of GPU A and the device report (AC-021) plus calibration of GPU B, and no benchmark results of GPU B
- **When** I run `python -m lws.bench.predict cross --from <A> --to <B>`
- **Then** it writes the predicted ratio and verdict (GO / NO-GO) for every shape and batch 1, 4, 16 on B, with a timestamp earlier than B's microbenchmark report
- **And** once B's microbenchmark exists, the summary (AC-023) shows predicted vs measured ratio per shape and whether the verdict matched
- **And** with Kaggle's P100 and T4, both directions (P100 → T4 and T4 → P100) are reported

## Out of scope

- Cycle-accurate or simulator-based models; the model is analytic (roofline-style) on purpose.
- Tensor-core kernels (future work, docs/paper-plan.md F5).
- Attention / KV-cache time: the model covers the Linear layers only; end-to-end impact is measured by AC-018.

## Assumptions

| ID | Assumption | Status | Resolution |
|---|---|---|---|
| ASM-016 | An analytic max(memory time, compute time) model plus a measured overlap term is enough to reach ≤ 10% error on GEMV at batch ≤ 16 | open | — (AC-027 checks it) |
| ASM-017 | Injected dependent integer operations raise the fused kernel's compute cost linearly in k and are not removed by NVRTC (inline PTX / volatile dependency) | open | — |
| ASM-018 | P100 and T4 differ enough in instructions per byte (~6 vs ~13, fused-decode-gemv design.md) to make cross-prediction a meaningful out-of-sample test | open | — |

## Open questions

| ID | Question | Status | Answer |
|---|---|---|---|
| Q-008 | Should community-contributed reports (other GPUs, run with scripts/verify.sh) be accepted into the paper's dataset, and under which acceptance rules beyond the P-006 metadata? | answered | Not in the first paper (product owner): validation is the controlled experiment plus P100/T4; external reports are future work |
