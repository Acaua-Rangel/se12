# Design: fused SE12 decode + GEMV (CUDA C, any NVIDIA GPU sm_60+)

> feature: fused-decode-gemv

## Why GEMV and not GEMM

At batch size B ≤ 16, `y[B, N] = x[B, K] · W[N, K]ᵀ` does ~2·B FLOPs per
weight: far below the compute/bandwidth balance of most GPUs. The kernel is a
streaming reduction over W; the win can only come from moving fewer bytes —
and only while the decode stays hidden under the memory traffic.

## Hardware range (spec-sheet values, approximate — the report records measured ones)

| GPU | cc | VRAM | BW (GB/s) | FP32 (TFLOPS) | L2 | native BF16 | FP32 instr/byte |
|---|---|---|---|---|---|---|---|
| GTX 1060 6GB | 6.1 | 6 GB | 192 | 4.4 | 1.5 MB | no | ~11 |
| Tesla P100 | 6.0 | 16 GB | 732 | 9.3 | 4 MB | no | ~6 |
| V100 | 7.0 | 16/32 GB | 900 | 15.7 | 6 MB | no | ~9 |
| T4 | 7.5 | 16 GB | 320 | 8.1 | 4 MB | no | ~13 |
| RTX 3060 | 8.6 | 12 GB | 360 | 12.7 | 3 MB | yes | ~18 |
| RTX 4090 | 8.9 | 24 GB | 1008 | 82.6 | 72 MB | yes | ~41 |
| RTX 5090 | 12.0 | 32 GB | 1792 | 105 | 96 MB | yes | ~29 |
| A100 80GB SXM | 8.0 | 80 GB | 2039 | 19.5 | 40 MB | yes | ~5 |
| H100 SXM | 9.0 | 80 GB | 3350 | 67 | 50 MB | yes | ~10 |

"FP32 instr/byte" = (FP32 TFLOPS / 2) / bandwidth: how many FMA-class
instructions the GPU can issue per byte it reads. Rough per-weight cost:

```
raw BF16 : (1 unpack + B FMA) / 2 bytes    → batch 1 ≈ 1 instr/byte
SE12     : (~8 decode + B FMA) / 1.5 bytes → batch 1 ≈ 6 instr/byte
```

Consequences:

- Consumer Ampere/Ada/Blackwell cards have the most slack: best GO candidates.
- HBM datacenter GPUs (A100, P100, H100) are the tightest: decode may become
  exposed even at batch 1, and at batch ≥ 4 CUDA-core GEMV turns compute-bound.
  That is why the verdict is per GPU and per batch size (P-011) and the runtime
  falls back to the exact path where there is no GO.
- INT32 throughput differs per generation (shared FP32/INT32 lanes on sm_86+,
  separate pipes on sm_70/75/80/90). The report measures, it does not assume.

## Compute profiles (chosen at runtime by the device rules in `lws.domain.device`)

| Profile | When | Activations | Weights in VRAM | Kernel output |
|---|---|---|---|---|
| `bf16-native` | cc ≥ 8.0 | BF16 | BF16 (original) / SE12 | FP32 → cast to BF16 by torch |
| `fp32-act` | cc < 8.0 | FP32 | BF16 (original) / SE12 | FP32 |

Both kernels accept FP32 or BF16 activations (BF16 → FP32 is exact:
`bits << 16`) and always accumulate and write FP32 (P-005). The profile only
decides what the model feeds them and what the baselines are (P-009).

## Speed-of-light model (write it in the report, per GPU)

```
t_raw    ≈ 2·N·K / BW
t_se12   ≈ 1.503·N·K / BW  +  t_decode_exposed
speedup  ≤ 16 / 12.03 ≈ 1.33×   (when decode is fully hidden)
```

Gemma 2 2B LM head (256000 × 2304 ≈ 590M weights):

| GPU | raw | SE12 (decode hidden) |
|---|---|---|
| GTX 1060 | ~6.1 ms | ~4.6 ms |
| P100 | ~1.6 ms | ~1.2 ms |
| H100 SXM | ~0.35 ms | ~0.27 ms |

A 2304 × 2304 projection is ~10.6 MB: ~3 µs on an H100 — the same order as a
kernel launch, and it fits entirely in the H100's 50 MB L2. Isolated-shape
timings are therefore meaningless on fast GPUs; the verdict uses **stacked
mode** (below).

## Kernel shape

- One CUDA block per group of SE12 tiles along N (tile = 16 rows × 256 cols,
  fixed by the packed format — never tuned per GPU, P-010).
- Each warp streams its rows with 128-bit loads (`uint4`): 16 bytes of `sm`
  and 8 bytes of `ec` per 16 weights. Warp size 32 is valid on every NVIDIA
  GPU; the kernel still reads it from `warpSize`-derived constants, not a
  literal scattered through the code.
- Decode in registers: `code = (ec >> s) & 0xF`; the 15-entry codebook is held
  in shared memory **pre-shifted to FP32 exponent position** (`exp << 23`), so
  the common path is extract + one lookup + OR with sign/mantissa bits;
  escape path via tile escape mask + `__popc`; fallback tiles via one bitmap
  bit (rare branch). Rebuild `float` with
  `__int_as_float((sign<<31)|(exp<<23)|(mant<<16))`.
- FP32 FMA into per-thread accumulators; warp reduction with `__shfl_down_sync`.
- Double buffering: issue loads for the next chunk before decoding the current
  one. Bytes in flight per SM needed to saturate bandwidth grow with the GPU
  (≈ bandwidth × latency / SM count), so the unroll depth is a launch
  parameter, not a constant.
- Optional arch paths behind `#if __CUDA_ARCH__ >= 800` (e.g. `cp.async` into
  shared memory, L2 eviction hints). Raw and fused kernels share the SAME load
  path on each architecture (P-009) — only load+decode differs.

## Code layout (constitution P-013 / P-014)

| Concern | Layer | Where |
|---|---|---|
| compile target, compute profile, VRAM fit, GPU slug | domain | `lws.domain.device` |
| launch heuristic and tuning search space | domain | `lws.domain.launch` |
| latency statistics, GO / NO-GO / INCONCLUSIVE rule | domain | `lws.domain.benchmark` |
| reading device properties, NVRTC, kernel launch, CUDA events/graphs | adapter | `lws.adapters.cuda` |
| CUDA C sources (exempt from object calisthenics) | adapter asset | `lws.adapters.cuda.kernels` |
| tuning cache, reports, verdicts on disk | adapter | `lws.adapters.filesystem` |

Every hardware-dependent decision is a pure function of a `DeviceProperties`
value object, so it is tested on CPU with fake GPUs; only the adapters touch
the real device.

## Launch configuration

- Default: heuristic from queried SM count, max threads per SM and shape
  (enough blocks to cover every SM several times, unroll sized for bytes in
  flight). No GPU name appears in the code (P-012).
- Optional tuner: grid search over rows per block × unroll × threads per
  block, SAME search space for raw and fused, each keeps its best. Results go
  to `$LWS_CACHE_DIR/tune-<gpu-slug>.json` (default `~/.cache/lws`); a file
  whose slug does not match the current GPU is ignored with a warning.

## Toolchain

`src/lws/adapters/cuda/kernels/*.cu` are plain CUDA C, shipped as package
data, loaded as strings and compiled with `cupy.RawModule` by the
`KernelCompiler` adapter, using the target chosen by `lws.domain.device`:

1. Detect `(major, minor)` of the selected device (`LWS_DEVICE`, default 0).
2. If NVRTC supports `sm_{major}{minor}` → compile SASS for it (no driver JIT,
   works with older drivers under CUDA minor-version compatibility).
3. Else → PTX for the highest `compute_XY` NVRTC supports that is ≤ the device;
   the driver JITs it. Recorded as "PTX fallback" in every report.
4. Else (device below sm_60, or no usable target) → refuse with a clear message.

Compiled binaries are cached on disk keyed by (source hash, NVRTC version,
target). `LWS_FORCE_PTX=1` forces step 3 so the fallback is testable on any
GPU. Tensors cross between torch and CuPy through DLPack (zero copy); kernels
are launched on torch's current stream (`cupy.cuda.ExternalStream`) and never
allocate or synchronize, so they can be captured in CUDA graphs.

Toolchain versions that matter (the installer picks them, the doctor
`python -m lws.device` checks them):

| GPU range | PyTorch wheel that ships kernels for it | NVRTC able to target it |
|---|---|---|
| sm_60 – sm_70 (Pascal, Volta) | CUDA 12.x builds that still ship sm_60/70 (e.g. cu126) | CUDA 12.x (dropped in CUDA 13) |
| sm_75 – sm_90 | any current CUDA 12.x build | CUDA 12.x |
| sm_100, sm_120 (Blackwell) | cu128 or newer | CUDA ≥ 12.8 |

The installer never trusts this table blindly: it runs a smoke test on the
device and switches wheel only if the test fails.

## Benchmark method (any GPU)

- L2 flush before each timed run by writing a buffer ≥ 2× the queried L2 size.
- **Stacked mode** (used for the verdict): the kernel runs over the SAME
  projection of every decoder layer back to back (e.g. 26 distinct gate
  matrices for Gemma 2 2B), captured in one CUDA graph, so launch overhead and
  L2 reuse disappear as they do in the real model. Isolated-shape numbers are
  reported too, but only as a diagnostic.
- CUDA events around the graph replay; warmup ≥ 25, timed runs ≥ 100.
- Peak copy bandwidth measured with a device-to-device copy of
  `min(1 GiB, 25% of free VRAM)`.

## VRAM awareness

- FP64 reference products are computed in row chunks sized from
  `torch.cuda.mem_get_info()` (FP64 of a full LM head would be 4.7 GB).
- The benchmark keeps only one variant's weights resident at a time.
- Kernel-level features need ~4 GB of free VRAM with Gemma 2 2B shapes; the
  doctor reports which features and model profiles fit (see weight-codec Q-001).

## Accuracy note

The weights are bit-exact, but a different FP32 reduction order gives slightly
different outputs. AC-010 therefore compares both kernels against an FP64
reference instead of demanding identical outputs.
