# Spec: Weight offload (stream lossless-compressed weights over PCIe)

> feature: weight-offload

## Context

Paper contribution C2 (docs/paper-plan.md). When a model does not fit in VRAM,
the missing layers are streamed from host memory over PCIe on every token.
PCIe (≈ 12–16 GB/s on Gen3 x16, as on Kaggle's P100/T4) is ~25–60× slower
than VRAM on those GPUs, so the GPU has hundreds of spare instructions per streamed byte: the
SE12 decode is practically free and the saving tends to the format's bound.
On top of that, under a fixed VRAM budget SE12 keeps MORE weights resident, so
the two effects compound (design.md): the streamed bytes per token can drop
well below the 0.75 format ratio.

Prior art: DFloat11 reports large throughput gains over CPU offloading of an
uncompressed model when compression makes the model fit (capacity effect;
verify the exact numbers in the paper before citing). The distinct question
here is the regime where the model does NOT fit even compressed, so compressed
weights themselves are streamed — and a fixed-rate format makes each layer a
fixed-size, randomly addressable block, which keeps the copy pipeline simple.

Setup that costs nothing: Kaggle P100/T4 (16 GB VRAM, ~30 GB host RAM) with
Gemma 2 9B (BF16 ≈ 18.5 GB, SE12 ≈ 13.9 GB). Smaller consumer cards (6, 8,
12 GB) are emulated by capping the VRAM available to the process, which the
report states explicitly.

## Stories

### US-016 — Run a model larger than the VRAM budget by streaming its weights

As a user with a GPU too small for the model, I want the missing layers
streamed from host memory with copies overlapped with compute, in BF16 or in
SE12, so that I can run the model and compare both formats on the same
pipeline.

#### AC-029 — One streaming pipeline, two formats, identical results

- **Given** the original BF16 model, its packed SE12 copy and a VRAM budget smaller than the model (e.g. `--vram-budget 8GiB`)
- **When** I load each with `lws.offload.load(model_dir, vram_budget=..., weight_format="bf16" | "se12")`
- **Then** a pure placement rule decides which layers stay resident and which are streamed, filling the budget in the same layer order for both formats, and the load log prints resident and streamed bytes per format
- **And** streamed layers are copied from pinned host memory on a dedicated stream, double buffered, and overlapped with the compute of the previous layer
- **And** for the same prompts and greedy decoding, the logits of the two formats are bit-identical when both run the order-matched kernels (AC-032, AC-033)

#### AC-030 — Streaming SE12 is faster than streaming BF16 when transfer-bound

- **Given** Gemma 2 9B on a 16 GB GPU with VRAM budgets of 6, 8, 10 and 12 GiB
- **When** I run `python -m lws.bench.offload --out reports/offload-<gpu-slug>.json` with a 128-token prompt and 128 generated tokens at batch size 1
- **Then** for each budget and format it records time per output token (median, p10, p90), streamed bytes per token, measured host-to-device bandwidth, overlap efficiency (copy time hidden under compute) and the predicted time streamed bytes ÷ measured bandwidth, with the metadata required by principle P-006
- **And** at every budget where BF16 is transfer-bound, SE12's median time per token is at most 0.80× BF16's, and the report compares the measured ratio with the predicted ratio of streamed bytes

#### AC-031 — Capacity point and external reference

- **Given** the full 16 GB of the GPU
- **When** the offload benchmark runs with no budget cap
- **Then** it reports the case where SE12 fits entirely in VRAM while BF16 must stream, with both times per token
- **And** it records an external reference run of llama.cpp with the same model in BF16 and the same number of layers offloaded (tool version, command line, GGUF conversion command), or — when llama.cpp cannot run that configuration on this GPU — the report states why instead of a number

## Out of scope

- Lossy offload formats (INT8/INT4/FP8) and activation sparsity schemes — they change the model.
- Disk (NVMe) offload and CPU compute of offloaded layers.
- Network weight transfer (RL weight sync, multi-node): discussed in the paper, not measured on Kaggle.
- Batch sizes above 16 (throughput-oriented offload such as FlexGen's large-batch schedule).

## Assumptions

| ID | Assumption | Status | Resolution |
|---|---|---|---|
| ASM-019 | Kaggle allows pinning ~18.5 GB of its ~30 GB host RAM for the BF16 9B weights without the session being killed | open | — |
| ASM-020 | Capping VRAM with `torch.cuda.set_per_process_memory_fraction` plus the placement budget is a fair emulation of a smaller GPU for this workload (same SMs and bandwidth, less memory) — the paper states it as emulation | open | — |
| ASM-021 | llama.cpp runs Gemma 2 9B in BF16 with partial GPU offload on sm_60/sm_75 (BF16 matmul upcast to FP32 on those GPUs) | open | — |
| ASM-022 | Kaggle disk has room for the BF16 model input, the SE12 copy (~14 GB) and a BF16 GGUF (~18.5 GB) in the same session, using ephemeral space outside the ~20 GB output directory | open | — |

## Open questions

| ID | Question | Status | Answer |
|---|---|---|---|
| Q-009 | FlexGen is the classic offload reference, but it targets OPT-family models and throughput-oriented large batches. Keep it as related work only, or port its schedule to Gemma 2 as a second baseline? | answered | Related work only (product owner); the external baseline is llama.cpp (AC-031) |
