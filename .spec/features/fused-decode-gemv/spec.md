# Spec: GPU decode and fused decode+GEMV kernels (CUDA C, any NVIDIA GPU sm_60+)

> feature: fused-decode-gemv

## Context

The hypothesis of the whole project lives here: at batch size 1 the GPU
spends most of each token waiting for weights to arrive from VRAM. If the
kernel reads SE12 weights (≈ 75% of the bytes) and rebuilds the exact BF16
values in registers right before the multiply, the time saved on the memory
bus may exceed the time spent decoding. Published lossless systems (DFloat11,
tile-level rANS) report being SLOWER than plain BF16 at batch 1 when the model
already fits in VRAM, because variable-length decode is expensive — so this
feature must measure honestly and is allowed to conclude "no-go". The closest
public experiment, Split12 (weight-codec design.md, "External evidence"),
points both ways: a dense 12-bit prototype ran at 0.733× BF16 GEMV time on an
A40 but without its escape correction, while exact scalar kernels never beat
BF16 on B300 and H100.

Target hardware: ANY NVIDIA GPU with compute capability ≥ 6.0, from a home
card (GTX 1060, RTX 3060, RTX 4090, RTX 5090) to a datacenter H100. The ratio
between compute and bandwidth varies ~8× across that range (design.md), so the
answer to "is it faster?" is expected to differ per GPU: the verdict is per GPU
and per batch size (principle P-011). Kernels are one portable CUDA C source
compiled with NVRTC for the detected architecture (principle P-008) and
compared against the best lossless BF16 path on the same GPU (principle P-009).

Reference data points (not requirements — any sm_60+ GPU can produce proof):
Kaggle P100 and T4 (free, no native BF16), one consumer Ampere/Ada card
(native BF16, high compute per byte), and one A100/H100 (native BF16, lowest
compute per byte).

## Stories

### US-012 — Know whether my GPU can run it, and how

As the researcher (or anyone cloning the repo), I want one command that tells
me whether my GPU is supported and which toolchain path it will use, so that
failures on an unusual GPU are explained before any kernel runs.

#### AC-021 — Device doctor reports capabilities and a support verdict

- **Given** a machine with an NVIDIA GPU and driver
- **When** I run `python -m lws.device` (optionally with `LWS_DEVICE=<index>`)
- **Then** it prints and writes `reports/device-<gpu-slug>.json` with: name, GPU slug, compute capability, SM count, total and free VRAM, L2 size, theoretical bandwidth (memory clock × bus width), driver, CUDA runtime, torch version and its CUDA arch list, cupy and NVRTC versions, NVRTC-supported targets, the chosen compile target (SASS or PTX fallback), the compute profile (`bf16-native` or `fp32-act`) and which features and model profiles fit in the free VRAM
- **And** it exits 0 when the GPU is supported, and non-zero with an actionable message (what is missing and how to fix it) when compute capability is below 6.0, when the installed torch cannot run a kernel on the device, or when NVRTC can target neither the device architecture nor any lower PTX architecture
- **And** the decision logic is tested on CPU with fake device properties covering sm_60, sm_75, sm_86, sm_90, sm_120 and sm_52

#### AC-022 — Kernels compile for the detected architecture, with a PTX fallback

- **Given** the kernel sources and the selected device
- **When** the loader compiles them
- **Then** it produces SASS for `sm_XY` of the device when NVRTC supports it, otherwise PTX for the highest supported `compute_XY` ≤ the device, and the kernel handle records which one was used
- **And** compiled binaries are cached on disk keyed by source hash, NVRTC version and target, so a second process does not recompile
- **And** with `LWS_FORCE_PTX=1` the PTX path is used and the AC-009 bit-exact test still passes on the same GPU

### US-004 — Rebuild exact BF16 weights on the GPU

As the researcher, I want a GPU kernel that decodes SE12 tensors back to BF16
in VRAM, so that I have a correctness oracle on the GPU and a fallback
"decode-then-multiply" path.

#### AC-009 — GPU decode matches the original bits exactly

- **Given** SE12 tensors produced by the reference packer (synthetic edge cases and, when available, every tensor of the real model)
- **When** the GPU decode kernel rebuilds them in VRAM on any supported GPU
- **Then** every rebuilt tensor is identical bit for bit to the original BF16 tensor (compared as int16)

### US-005 — Multiply directly from compressed weights

As the researcher, I want a kernel that computes `y = x · Wᵀ` reading only the
SE12 streams, so that full-size BF16 weights never cross the memory bus.

#### AC-010 — Fused result is as accurate as the raw-BF16 kernel

- **Given** random activations in FP32 and in BF16, and the target model's weight shapes
- **When** the fused kernel computes the product from SE12 weights and the raw-BF16 baseline kernel computes it from the original BF16 weights, both writing FP32
- **Then** the fused kernel's maximum absolute error against an FP64 reference (computed from the exact original weights) is no larger than 2× the raw-BF16 kernel's error against the same reference
- **And** the FP64 reference is computed in row chunks sized from the free VRAM, so the check runs on a GPU with 4 GB free
- **And** the fused kernel never writes decoded weights to global memory (its only inputs are the SE12 streams and the activations)

#### AC-011 — Works for every Linear shape of the target model and batch sizes 1 to 16

- **Given** every distinct Linear weight shape of the target model, including the tied embedding used as LM head (256000 × 2304 for Gemma 2 2B)
- **When** the fused kernel runs with batch sizes 1, 2, 4, 8 and 16
- **Then** every combination passes the accuracy check of AC-010
- **And** shapes whose K is not a multiple of the tile size are handled without reading out of bounds (checked with compute-sanitizer memcheck on one shape; needs the CUDA toolkit on the machine that produces the proof)

#### AC-032 — Order-matched fused output is bit-identical to the raw-BF16 kernel

- **Given** the raw and fused kernels launched with the same launch configuration (same reduction order), for every Linear shape of the target model, batch sizes 1, 2, 4, 8 and 16, and FP32 and BF16 activations
- **When** both compute the product, the raw kernel from the original BF16 weights and the fused kernel from the SE12 streams
- **Then** the FP32 outputs are identical bit for bit (compared as int32)
- **And** the microbenchmark reports the speed cost of the order-matched configuration versus each kernel's independently tuned best (AC-024)

### US-013 — The same kernel is tuned for whatever GPU runs it

As the researcher, I want launch parameters derived from the GPU at hand, so
that a card with 10 SMs and one with 132 SMs are both used well without code
changes.

#### AC-024 — Launch configuration comes from the device, with an optional per-GPU tuning cache

- **Given** a GPU with no tuning file
- **When** the raw and fused kernels run
- **Then** their launch configuration comes from a heuristic over the queried SM count, max threads per SM and the shape — no GPU name or fixed SM count in the code
- **And** `python -m lws.tune` grid-searches the same space for raw and fused, keeps each one's best configuration per (shape, batch), and writes `tune-<gpu-slug>.json` to the cache directory; only configurations that pass the AC-010 accuracy check are saved
- **And** a tuning file whose GPU slug differs from the current GPU is ignored with a warning
- **And** deleting the tuning file changes speed only, never results beyond the AC-010 tolerance
- **And** the tuner also saves one order-matched configuration per (shape, batch) — the best configuration shared by raw and fused — used whenever bit-identical outputs are required (AC-032)

### US-006 — Measure speed honestly

As the researcher, I want a microbenchmark that compares the fused kernel with
the best lossless baseline on the same shapes and GPU, so that the go/no-go
decision rests on recorded numbers.

#### AC-012 — Benchmark report with full measurement metadata

- **Given** the target model's Linear shapes and batch sizes 1, 4 and 16
- **When** I run the microbenchmark (`python -m lws.bench.micro --out reports/micro-<gpu-slug>.json`)
- **Then** for every shape/batch it records median, p10 and p90 latency (µs) and effective bandwidth (GB/s), in stacked mode (the same projection of every layer back to back, captured in one CUDA graph) and in isolated mode, for: the raw-BF16 kernel, GPU decode + raw-BF16 kernel, the fused kernel, cuBLAS BF16 via torch `F.linear` on GPUs with native BF16, and — as reference only — torch FP32 `F.linear` on upcast weights
- **And** the report records every field required by principle P-006, warmup count (≥ 25) and timed runs (≥ 100), measured with CUDA events after an L2 flush with a buffer at least twice the queried L2 size
- **And** a kernel that fails its exactness/accuracy check on the benchmark inputs is reported as "rejected" and not timed

#### AC-013 — The baseline is really limited by memory bandwidth

- **Given** the benchmark results at batch size 1
- **When** the report computes the raw-BF16 kernel's effective bandwidth on the MLP shapes in stacked mode
- **Then** it is at least 70% of the GPU's measured peak copy bandwidth (device-to-device copy of min(1 GiB, 25% of free VRAM))
- **And** if it is not, the report states that the baseline is not tuned enough for a valid comparison and the verdict for that GPU is "INCONCLUSIVE"

### US-007 — Go / no-go decision, per GPU

As the researcher, I want a clear verdict on the central hypothesis for each
GPU I measure, so that I only build the integration if compression buys speed
somewhere, and the runtime knows where.

#### AC-014 — Per-GPU verdict: fused kernel at least 5% faster than the best lossless baseline

- **Given** the microbenchmark results in stacked mode for the target model's MLP projections (gate, up, down) and the LM head, on one GPU
- **When** the verdict is computed
- **Then** the GPU is "GO" at batch size 1 when, for each of those shapes, the fused kernel's median latency is at most 0.95× the median of the faster lossless baseline on that GPU (principle P-009); otherwise "NO-GO" (or "INCONCLUSIVE" per AC-013)
- **And** the same comparison is recorded for batch sizes 4 and 16, and all verdicts are written to `reports/verdict-<gpu-slug>.json` with the GPU slug, compute capability and the ratio per shape and batch
- **And** the report prints "GO", "NO-GO" or "INCONCLUSIVE" with the measured ratio per shape and the GPU it was measured on

#### AC-023 — Cross-GPU summary

- **Given** microbenchmark reports from one or more GPUs in `reports/`
- **When** I run `python -m lws.bench.summary reports/ --out reports/summary.md`
- **Then** it writes one row per GPU with name, compute capability, measured peak bandwidth, raw-kernel bandwidth as % of peak, fused ÷ baseline ratio per shape at batch 1, 4 and 16, and the verdict
- **And** reports missing any metadata required by principle P-006 are listed as rejected instead of being summarized

## Out of scope

- Triton, CUTLASS or Hopper-specific paths (TMA, wgmma) — principle P-008.
- Tensor-core fused GEMM for batch sizes where CUDA-core GEMV is compute-bound (expected at batch ≥ 4 on A100/H100/P100); the runtime uses the exact path there (P-011).
- Prefill (large-batch GEMM) optimization; prefill may keep using decode + GEMM.
- Multi-GPU (tensor/pipeline parallel). A multi-GPU machine is measured one device at a time (`LWS_DEVICE`).
- GPUs below sm_60 (Maxwell and older): current CUDA 12 / PyTorch builds are dropping them.
- Non-NVIDIA GPUs (Q-007).

## Assumptions

| ID | Assumption | Status | Resolution |
|---|---|---|---|
| ASM-003 | For every sm_60+ GPU there is a PyTorch CUDA 12.x wheel that runs on it (cu126 for sm_60/70, current builds for sm_75–sm_90, cu128+ for sm_100/120) and a `cupy-cuda12x` whose NVRTC can target it (natively or by PTX fallback) in the same environment | open | — (AC-021 checks it on each machine) |
| ASM-004 | At batch size 1 the Linear layers are memory-bandwidth bound on every supported GPU (AC-013 checks it per GPU) | open | — |
| ASM-005 | The SE12 decode costs roughly ≤ 8 integer instructions per weight, which fits the batch-1 budget of GPUs with ≳ 6 instructions per byte (design.md table); A100-class GPUs (~5) may expose decode time — an honest NO-GO there is a valid result | open | — |
| ASM-012 | CuPy kernels launched on torch's current stream can be captured in a CUDA graph on every supported GPU, so launch overhead does not hide the kernel time on fast GPUs | open | — |

## Open questions

| ID | Question | Status | Answer |
|---|---|---|---|
| Q-003 | Which exact GPU will run the experiments? | answered | Any NVIDIA GPU with cc ≥ 6.0. Reference set: Kaggle P100 and T4, one consumer Ampere/Ada card, one A100/H100. The roadmap gate needs GO on at least one GPU; the runtime uses the fused path only where GO was measured |
| Q-004 | Is 5% the right "go" threshold, or should the bar be higher (e.g. 10%) to justify the added complexity? | answered | 5% (product owner); the performance model (AC-027) lets the paper discuss the exact margin anyway |
| Q-007 | Should AMD (ROCm/HIP), Intel or Apple GPUs be supported later? The CUDA C source would need a HIP build (warp size 64 on CDNA) or a Metal port | answered | Future work only (product owner): cited in the paper as an extension, nothing in the current plan |
