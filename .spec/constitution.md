# Constitution — v1.7.0

<!--
  Non-negotiable project principles. They are not style: they are constraints.
  P-xxx = principle (traceability ID, like US/AC/T).
  Levels: [MUST] mandatory · [RECOMMENDED] strong · [MAY] allowed/explicit.
  Every [MUST] needs an executable check — otherwise the audit flags
  "principle without a check" (PRINCIPLE_NO_CHECK). Formats:
    - verification(gate): satisfied by the audit itself (only for "meta" principles)
    - verification(test): @principle:P-xxx
    - verification(forbidden): `regex` in `glob`
    - verification(required): `regex` in `glob`
-->

## P-001 [MUST] Every requirement has an executable proof

No feature is declared done without the audit in CI mode exiting clean (exit 0).
This principle is checked by the audit mechanism itself (AC_NO_TEST,
AC_NO_PROOF, TASK_DONE_NO_PROOF) — it needs no extra test from you.

- verification(gate): intrinsic to the audit

## P-002 [RECOMMENDED] Secrets never in code

Keys and tokens (e.g. the Hugging Face token) come from environment variables,
never hard-coded.

- verification(forbidden): `(api[_-]?key|password|secret|hf_token)\s*[:=]\s*['"][^'"]{8,}` in `src/**/*.py`

## P-003 [MUST] The original model files are never modified

The compressor only READS the original `.safetensors` files and writes the
packed copy to a different directory. Proven by hashing the originals before
and after packing (sha256 must be identical).

- verification(test): @principle:P-003

## P-004 [MUST] Every codec path is bit-exact

Any path that turns packed weights back into BF16 (CPU reference, GPU
decode-only kernel, fused kernel's internal decode) must reproduce the
original 16-bit patterns exactly — including ±0, subnormals, ±inf and NaN
payloads. Comparison is done on the raw bits (`view(torch.int16)`), never
with a float tolerance.

- verification(test): @principle:P-004

## P-005 [MUST] No lossy numeric shortcut inside the codec or kernels

The project's whole claim is "100% of the original precision" of the WEIGHTS.
Quantization, FP8/INT casts and BF16→FP16 conversion are forbidden in library
code. Kernels always accumulate and write FP32; casting the output to the
model's compute dtype (BF16 on GPUs with native BF16, see P-009) is done by
torch outside the kernel, exactly as the original model rounds its own
activations — never applied to weights. vLLM calls its pluggable weight
formats "quantization methods": registering SE12 through that API is allowed;
calling any quantize function or casting to a lossy dtype is not.

- verification(forbidden): `(float8_e4m3|float8_e5m2|torch\.int4|\.half\(\)|\bquantize(_\w+)?\s*\()` in `src/**/*.py`
- verification(forbidden): `(__float2half|__float2bfloat16_rn|__half2float|__nv_fp8)` in `src/**/*.cu`

## P-006 [MUST] Performance claims come only from recorded measurements

Every benchmark report records: GPU name, GPU slug, compute capability, SM
count, VRAM, L2 size, measured peak copy bandwidth, driver, CUDA runtime,
NVRTC, torch and cupy versions, compile target (SASS `sm_XY` or PTX fallback),
compute profile, SM/memory clocks, temperature and power limit before and after
the run, ECC and MIG state, whether a display is attached, CUDA graphs on/off,
shape, batch size, warmup count, number of runs and the median / p10 / p90
latency. A speed number without that metadata is not a result.

- verification(test): @principle:P-006

## P-007 [MUST] Correctness before speed

A benchmark refuses to time a kernel whose bit-exactness check fails on the
same inputs — a fast wrong kernel is never reported.

- verification(test): @principle:P-007

## P-008 [MUST] One portable CUDA C source for every NVIDIA GPU from sm_60 to sm_120

Target: any NVIDIA GPU with compute capability ≥ 6.0 — consumer (GTX 10xx,
GTX 16xx / RTX 20xx, RTX 30xx, RTX 40xx, RTX 50xx) and datacenter (P100, V100,
T4, A100, L4, H100, B200). Kernels are CUDA C (`src/lws/adapters/cuda/kernels/*.cu`),
compiled at runtime with NVRTC through CuPy for the architecture DETECTED at
runtime: native SASS (`sm_XY`) when the installed NVRTC supports it, otherwise
PTX for the highest `compute_XY` it supports that is ≤ the device (the driver
JITs it). Architecture-specific code (e.g. `cp.async` on sm_80+) is allowed
only behind `#if __CUDA_ARCH__ >= ...` with a generic path always present in
the same file. A Triton or CUTLASS port is justified only on sm_80+ hardware
and only after a recorded benchmark of the CUDA C version on that hardware.

- verification(test): @principle:P-008

## P-009 [MUST] Speed is compared against the best LOSSLESS BF16 path on the same GPU

The primary baseline is always a raw-BF16 GEMV kernel with the same code
skeleton as the fused kernel (reads 16-bit weights, upcasts them exactly to
FP32, FP32 FMA) — the only differences between the two are the bytes read and
the decode. On GPUs with native BF16 (sm_80+), cuBLAS BF16 (torch `F.linear`
with BF16 weights and BF16 activations) is ALSO a baseline, and the verdict is
taken against the FASTER of the two. Converting weights to FP16 is not a
baseline: it is lossy.

- verification(test): @principle:P-009

## P-010 [MUST] Pack once, run on any GPU

The packed SE12 format contains no GPU-dependent field (no architecture, no
launch configuration, no tuning). The same packed files are decoded bit-exactly
on every supported GPU; per-GPU tuning lives in a separate cache file keyed by
the GPU slug and can be deleted at any time without changing results.

- verification(test): @principle:P-010

## P-011 [MUST] Speed verdicts are per GPU, and the runtime obeys them

A GO measured on one GPU says nothing about another. Every verdict is stored
with the GPU slug it was measured on, and the runtime's automatic mode only
uses the fused kernel on a GPU (and batch size) with a recorded GO; everywhere
else it uses the exact decode + matmul path.

- verification(test): @principle:P-011

## P-012 [MUST] No hard-coded GPU facts in library code

Architecture flags, L2 size, SM count, bandwidth and VRAM are queried at
runtime through the GPU probe port (`lws.adapters.cuda.gpu_probe`) and turned
into decisions by pure rules (`lws.domain.device`), never written as constants
or chosen by GPU model name.

- verification(forbidden): `(-arch=(sm|compute)_\d|["'](Tesla )?(P100|V100|T4|A100|H100|RTX ?\d{4})["'])` in `src/**/*.py`

## P-013 [MUST] Hexagonal architecture (ports and adapters) for all Python code

Dependencies point inward only: entrypoints → adapters → application → domain.

| Layer | Package | May import | Holds |
|---|---|---|---|
| domain | `lws.domain.*` | stdlib, numpy, `lws.domain` | pure rules: SE12 codec, entropy, device decisions (compile target, compute profile, VRAM fit), launch heuristic, benchmark statistics and verdict, kernel-path dispatch |
| application | `lws.application.*` | the above + `lws.application` | use cases (analyze, pack, diagnose device, tune, benchmark, load, generate, chat) and ports as `typing.Protocol` in `lws.application.ports.<area>` |
| adapters | `lws.adapters.<technology>` | anything | the ONLY code that imports torch, cupy, safetensors, transformers, huggingface_hub, vllm, fastapi; implements ports (`safetensors`, `huggingface`, `cuda`, `transformers`, `vllm`, `filesystem`, `http`, `external`); CUDA C sources live in `lws.adapters.cuda.kernels` |
| entrypoints | `lws.analyze`, `lws.pack`, `lws.device`, `lws.tune`, `lws.bench.*`, `lws.runtime`, `lws.server` | anything | composition roots: parse arguments, wire adapters into a use case, nothing else |

Rules: every port has at least one adapter and one in-memory fake used by the
tests; use cases receive ports by constructor injection, never import an
adapter; imports are absolute. Scope: every `.py` under `src/lws`. Not
applicable: CUDA C kernels (adapter assets), `tests/`, `conftest.py`,
`scripts/`.

- verification(forbidden): `^\s*(from|import)\s+(torch|cupy|safetensors|transformers|huggingface_hub|vllm|fastapi|uvicorn|httpx)\b` in `src/lws/domain/**/*.py`
- verification(forbidden): `^\s*(from|import)\s+(torch|cupy|safetensors|transformers|huggingface_hub|vllm|fastapi|uvicorn|httpx)\b` in `src/lws/application/**/*.py`
- verification(forbidden): `^\s*(from|import)\s+lws\.(application|adapters|analyze|pack|device|tune|bench|runtime|server)\b` in `src/lws/domain/**/*.py`
- verification(forbidden): `^\s*(from|import)\s+lws\.(adapters|analyze|pack|device|tune|bench|runtime|server)\b` in `src/lws/application/**/*.py`
- verification(test): @principle:P-013

## P-014 [MUST] Object Calisthenics wherever it applies

The nine rules, read for Python, and where each one applies:

| # | Rule (Python reading) | domain + application | adapters + entrypoints |
|---|---|---|---|
| 1 | One level of indentation per function (at most one nested compound statement; comprehensions do not count) | yes | yes |
| 2 | No `else` / `elif` (including `for`/`while`/`try` … `else`): guard clauses, early return, dict dispatch or polymorphism; conditional expressions are allowed | yes | yes |
| 3 | Wrap all primitives and strings: public functions and methods neither take nor return bare `int`/`float`/`str`/`bool`/`ndarray`; they use single-field frozen value objects (`TileShape`, `EscapeBudget`, `ComputeCapability`, `GpuSlug`, `BatchSize`, `Microseconds`, …). Vectorized numpy INSIDE a method works on raw arrays — wrapping is at the boundary, never per weight | yes | relaxed |
| 4 | First-class collections: a class that holds a collection holds nothing else | yes | relaxed |
| 5 | One dot per line: at most one attribute access chained from `self` or a local (`self.codebook.lookup(x)` ✗); module-qualified names (`numpy.int16`) do not count | yes | relaxed |
| 6 | No abbreviations: banned identifiers include `cfg conf mgr tmp idx cnt num buf val res arr ptr calc util info obj ctx`; allowed glossary: BF16 FP32 FP64 SE12 GEMV GPU VRAM SM L2 NVRTC PTX SASS CUDA KV LM MLP API HTTP JSON SHA256 and the SE12 stream names `sm` `ec` `esc` (weight-codec design.md) | yes | yes |
| 7 | Small entities: a class has at most 50 lines, a package at most 10 modules | yes | yes |
| 8 | At most two instance variables per class (dataclass fields count): bigger records are compositions (e.g. a device report = identity + resources) | yes | relaxed |
| 9 | No getters, setters or properties: no `@property`, no `get_*`/`set_*` methods, never assign another object's attributes; reading a field of an immutable value object is allowed; behaviour lives with the data (tell, don't ask) | yes | relaxed |

"Relaxed" means the rule yields where the wrapped library dictates the shape
(`nn.Module` attributes, FastAPI/pydantic schemas, torch/numpy fluent calls,
per-token hot paths such as `CompressedLinear.forward`). Any other deviation in
an adapter is marked on the line with `# calisthenics: allow <rule> — <reason>`
and listed by the fitness test; domain and application have no exceptions.
Correctness and speed principles (P-004 to P-012) win over this one when they
conflict. Not applicable: CUDA C kernels (procedural GPU code shaped by
fused-decode-gemv design.md), `tests/`, `conftest.py`, `scripts/`.

- verification(forbidden): `^\s*(else\s*:|elif\s)` in `src/lws/**/*.py`
- verification(test): @principle:P-014

## P-015 [RECOMMENDED] Every number in the paper is regenerated from a recorded report

Tables and figures of the paper (docs/paper-plan.md) are produced by a script
that reads only `reports/*.json` files carrying the P-006 metadata — never
typed by hand. Every experiment runs on free, public hardware (Kaggle) from a
public notebook, so a reviewer can reproduce it at zero cost; the artifact
(code, reports, notebooks) gets a permanent DOI at submission.
