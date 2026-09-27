# Spec: Model integration (drop-in compressed Linear layers)

> feature: model-integration

## Context

Built after fused-decode-gemv is audited, whatever its verdict: GO or NO-GO,
the integration is needed for bit-exact end-to-end serving (paper
contribution C3) and for weight offload (C2), which do not depend on the
kernel beating BF16 from VRAM. The runtime loads the packed model into a normal Hugging
Face transformers model, swapping each eligible `nn.Linear` for a layer that
keeps only SE12 streams in VRAM. Everything else (attention, KV cache,
sampling, tokenizer) stays untouched — the model treats weights as passive
data.

The runtime must work on any GPU the kernels support. It picks the compute
profile from the device (`bf16-native` on sm_80+, `fp32-act` below — see
fused-decode-gemv design.md), uses the fused kernel only where a GO verdict was
recorded for that GPU (principle P-011), and even where fused is NO-GO the
packed model still saves ~25% of weight VRAM — which lets a model run on a GPU
where the BF16 original does not fit.

Warning from prior art (Split12, weight-codec design.md "External evidence"):
a +27% decode gain on 8× B300 turned into −64.6% against a matched production
SGLang baseline, and greedy tokens diverged after 26 of 32 from reduction
order alone. A kernel-level GO does not imply an end-to-end gain; this feature
must prove it with AC-017 and AC-018.

## Stories

### US-008 — Load the packed model as a drop-in replacement

As the researcher, I want to load the packed model with one call and get a
regular transformers model back, so that existing generation code runs
unchanged.

#### AC-015 — Every eligible Linear layer is replaced and weight memory shrinks

- **Given** the packed target model
- **When** I load it with `lws.runtime.load_packed(packed_dir, mode=...)`
- **Then** every eligible `nn.Linear` is a `CompressedLinear`, the tied embedding is a `CompressedEmbedding` (decodes only the tiles of the looked-up rows; the LM head uses the GEMV kernels), and none keeps a full BF16 weight in VRAM
- **And** the VRAM taken by weights is at most 80% of the original model's weight VRAM (measured with torch.cuda memory stats)

#### AC-016 — "Exact" mode produces the very same logits as the original model

- **Given** the original model and the packed model loaded in "exact" mode (GPU decode of each layer right before the same matmul the original uses), both with the compute profile `lws.device` picks for this GPU
- **When** both run the same prompts with the same seed
- **Then** the logits of every step are identical bit for bit

### US-014 — Use the fastest exact path on whatever GPU I have

As a user with any supported GPU, I want the loader to choose the kernel path
from what was measured on my GPU, so that I never get a slower path than the
original just because the fused kernel lost on my hardware.

#### AC-025 — "auto" mode uses the fused kernel only where a GO was recorded for this GPU

- **Given** the packed model and, in `reports/` (or `LWS_VERDICT_DIR`), zero or more `verdict-<gpu-slug>.json` files
- **When** I load it with `lws.runtime.load_packed(packed_dir, mode="auto")`
- **Then** each `CompressedLinear` uses the fused kernel for a batch size b only when the smallest measured batch size ≥ b (1, 4 or 16) is GO for the current GPU slug, and the exact path otherwise (including every b > 16)
- **And** prefill (a forward pass with more than one new token per sequence) always uses the exact path — decode the layer once, then the same matmul as the original — whatever the verdict (Q-005)
- **And** with no verdict file for the current GPU slug (e.g. a verdict from another GPU only), every layer uses the exact path
- **And** the chosen path per batch size and the verdict file used are logged once at load time

### US-009 — Fast mode keeps the answers

As the researcher, I want the "fused" mode to generate the same text as the
original model, so that the speed gain does not change what the model says.

#### AC-017 — Greedy generations match, and any divergence is only at a near-tie

- **Given** a fixed set of at least 50 prompts and greedy decoding of 128 tokens
- **When** the original model and the packed model in "fused" mode generate
- **Then** at least 95% of the generations are token-for-token identical
- **And** at every first divergence, the original model's top-2 logits differ by less than 0.05 (a near-tie that the reduction order is allowed to flip), and the report lists each divergence

#### AC-033 — Order-matched fused mode reproduces the reference runtime bit for bit

- **Given** the reference runtime `lws.runtime.load_reference(model_dir)` (original BF16 weights, every eligible Linear through the raw-BF16 kernel) and the packed model loaded with `mode="fused", order="matched"` (fused kernel with the order-matched configuration of AC-032), both with the same compute profile
- **When** both run the AC-017 prompt set with greedy decoding of 128 tokens
- **Then** the logits of every step are identical bit for bit and 100% of the generations are token-for-token identical
- **And** the report puts this next to AC-017's comparison against the stock transformers path, so the paper can state which divergences come from reduction order and which (none) from the weights

### US-010 — End-to-end speed

As the researcher, I want tokens-per-second numbers for the full model, so
that I know whether the kernel-level gain survives the whole decode step.

#### AC-018 — End-to-end benchmark report

- **Given** the original model, the packed model in "exact" mode and in "fused" mode
- **When** I run `python -m lws.bench.e2e --out reports/e2e-<gpu-slug>.json` with a 128-token prompt and 256 generated tokens at batch sizes 1, 4 and 16
- **Then** the report records tokens/s, time per output token (median, p90) and peak VRAM for each variant, with the decode step captured in a CUDA graph and without it, and the same measurement metadata required by the microbenchmark (principle P-006)
- **And** when a variant does not fit in this GPU's VRAM, the report records "does not fit" with the required and available bytes instead of failing, and still measures the variants that fit

## Out of scope

- Changes to attention, KV cache or sampling.
- Integration into vLLM / llama.cpp (a later step, see local-api Q-006).
- Training or fine-tuning.

## Assumptions

| ID | Assumption | Status | Resolution |
|---|---|---|---|
| ASM-006 | The target model's text decoder uses standard `nn.Linear` modules in transformers, so replacement by module type works without model-specific code | open | — |
| ASM-007 | The fused-decode-gemv verdict (AC-014) is "GO" on at least one GPU before this feature starts | resolved | No longer a gate: exact end-to-end serving and weight offload justify the integration even with NO-GO on every measured GPU |
| ASM-010 | On GPUs without native BF16 (cc < 8.0) both the original and the packed model keep weights in 16-bit form in VRAM but run activations in FP32 (weights upcast exactly); on cc ≥ 8.0 both run BF16 activations as the original model normally does — either way the original and the packed model use the same profile, so the comparison stays fair | open | — |
| ASM-013 | On the `bf16-native` profile, where each layer's FP32 output is rounded to BF16, a different reduction order still keeps ≥ 95% of greedy generations identical (AC-017) | open | — |
| ASM-014 | The Gemma 2 decode step (with its static/hybrid KV cache in transformers) can be captured in a CUDA graph, so Python launch overhead does not hide the gain on fast GPUs such as the H100 | open | — |
| ASM-011 | Gemma 2 runs with `attn_implementation="eager"` (needed for its attention logit soft-capping) and its final logit soft-capping is left untouched by the runtime | open | — |

## Open questions

| ID | Question | Status | Answer |
|---|---|---|---|
| Q-005 | Should prefill (long prompts) use "exact" mode automatically and only decoding use "fused" mode, or one mode for the whole run? | answered | Prefill always exact, decode follows "auto" (product owner) — AC-025 |
