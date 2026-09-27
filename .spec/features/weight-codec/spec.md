# Spec: Weight codec (SE12 lossless fixed-rate format)

> feature: weight-codec
> status: draft

## Context

Batch-1 LLM decoding reads every weight from VRAM once per token, so it is
limited by memory bandwidth, not math. BF16 weights waste bits: the 8-bit
exponent carries only ~2.5–3 bits of real information, while sign+mantissa
are close to random. This feature measures that redundancy on the target
model and builds an offline compressor whose format ("SE12", split-exponent,
12 bits per weight) can later be decoded on the GPU with a handful of bitwise
operations — without changing a single bit of any weight.

Prior art (read before implementing): DFloat11 (Huffman on BF16 exponents,
~70% size, bit-exact, NeurIPS 2025 — arXiv 2504.11651), ZipNN (exponent
entropy coding for storage), SplitZip (4-bit exponent codebook + escapes, used
for KV cache — arXiv 2605.01708), "Approaching Shannon Bound with Lossless LLM
Weight Compression" (tile-level rANS fused into GEMM — arXiv 2606.15789),
Split12 (byte-split BF16, 12.005 bits/weight, bit-exact on all 59,509 tensors
of GLM-5.2 — `brianbell-x/weight-compression`; the closest public experiment,
summarized with its caveats in design.md "External evidence").
The deliberate difference here: a FIXED-RATE format (no variable-length codes)
so that every tile is randomly addressable and decode is pure bit
manipulation, trading ~1 bit/weight of ratio for decode speed.

## Stories

### US-001 — Know the compression ceiling before writing kernels

As the researcher, I want an entropy report of the target model's weights, so
that I know how many bits per weight are really needed before investing in
GPU kernels.

#### AC-001 — Entropy report per tensor and for the whole model

- **Given** a directory with the original BF16 `.safetensors` model
- **When** I run the analyzer on it (`python -m lws.analyze <model_dir> --out reports/entropy.json`)
- **Then** the report lists, for every BF16 tensor: element count, exponent entropy (bits), sign+mantissa entropy (bits), the 16-bit Shannon bound, and the share of weights covered by the 15 most frequent exponents
- **And** it shows the model-wide element-weighted averages of the same numbers

#### AC-002 — Tensors that are not BF16 are skipped, not crashed on

- **Given** a model file that also contains tensors in other dtypes (FP32, INT, bool)
- **When** I run the analyzer
- **Then** those tensors appear in the report under "not eligible" with their dtype
- **And** the analyzer exits successfully

#### AC-034 — Entropy survey across model families

- **Given** the Hugging Face repo ids of the original BF16 checkpoints of Gemma 2 2B, Llama 3.1 8B, Qwen2.5 7B, Mistral 7B and one open MoE model (e.g. OLMoE-1B-7B), or local directories
- **When** I run `python -m lws.survey <repo_id or dir>... --out reports/entropy-survey.json` on a machine without GPU
- **Then** the report has one row per model with element-weighted exponent entropy, sign+mantissa entropy, top-15 exponent coverage, projected SE12 bits per weight and share of fallback tiles (from the reference codec), plus the same numbers split by tensor role (attention, MLP or expert, embedding)
- **And** each model is streamed tensor by tensor and, for repo ids, shard by shard (download one shard, analyze it, delete it), so the survey runs on a PC with 16 GB of RAM and free disk for a single shard; the report records the repo revision (commit hash) of every model

### US-002 — Compress a weight tensor without losing a single bit

As the researcher, I want a reference encoder/decoder for the SE12 format, so
that every later GPU kernel has a trusted, bit-exact oracle to compare against.

#### AC-003 — Round trip is bit-exact, including float edge cases

- **Given** synthetic BF16 tensors containing +0, -0, subnormals, ±inf, NaNs with different payloads, the largest and smallest exponents, and random normal values
- **When** they are encoded to SE12 and decoded back with the reference codec
- **Then** the decoded tensor is identical bit for bit to the input (compared as int16)

#### AC-004 — Round trip is bit-exact on every tensor of the real model

- **Given** the original target model (path in the `LWS_MODEL_DIR` environment variable)
- **When** every eligible tensor is encoded and decoded with the reference codec
- **Then** every tensor is identical bit for bit to the original
- **And** the run prints how many tensors and weights were checked

#### AC-005 — Tiles with too many rare exponents fall back to raw storage

- **Given** a tensor built so that one tile holds more rare exponents than the tile's escape budget
- **When** it is encoded
- **Then** that tile is stored raw (uncompressed BF16) and flagged as a fallback tile
- **And** the round trip is still bit-exact
- **And** the encoder reports the number of fallback tiles

### US-003 — Produce a compressed copy of the model, never touching the original

As the researcher, I want a packer that writes a compressed copy of the whole
model to a separate folder, so that the original files stay pristine and the
runtime can load the packed version.

#### AC-006 — The packed model is at most 78% of the original weight bytes

- **Given** the original target model
- **When** I pack it (`python -m lws.pack <model_dir> <out_dir>`)
- **Then** the summary shows original bytes, packed bytes and the ratio for all eligible tensors
- **And** the ratio (packed ÷ original, metadata included) is at most 0.78

#### AC-007 — Original files are byte-for-byte untouched

- **Given** the sha256 of every file in the original model directory taken before packing
- **When** packing finishes
- **Then** every original file still has the same sha256
- **And** the packed output lives only inside the output directory
- **And** tensors that are not eligible (or excluded by configuration) are copied unchanged into the packed model

#### AC-008 — Packed files declare their format version, and unknown versions are refused

- **Given** a packed model
- **When** its metadata is read
- **Then** it contains the format name "SE12", the format version, the tile shape and, per tensor, the exponent codebook and the fallback-tile index
- **And** loading a file whose format version is unknown fails with an error message naming the found and the supported versions

## Out of scope

- Variable-length entropy coding (Huffman / ANS) — known to reach ~11 bits but costs decode speed; may be a later experiment.
- Compressing the KV cache or activations.
- Any lossy technique (quantization, pruning, low-rank).
- GPU kernels (feature fused-decode-gemv).

## Assumptions

| ID | Assumption | Status | Resolution |
|---|---|---|---|
| ASM-001 | In the target model, the 15 most frequent exponents of each weight matrix cover at least 99.5% of its weights, so escapes are rare and SE12 lands near 12.1 bits/weight | open | — (AC-001's report confirms or kills it) |
| ASM-002 | Tensors that are not 2-D Linear weights (norm scales, biases) are small enough to keep raw without hurting the ratio | open | — |
| ASM-009 | The Gemma 2 checkpoint is available on the machine that runs the proof — local disk, Kaggle Models input, or a Hugging Face download with the license accepted and the token in the `HF_TOKEN` environment variable (principle P-002) — and `LWS_MODEL_DIR` points to its BF16 `.safetensors` | open | — |
| ASM-015 | The packer streams one tensor at a time, so packing needs host RAM of only a few times the largest tensor (the Gemma 2 2B embedding, ≈ 1.2 GB) and runs on a home PC; packing needs no GPU | open | — |

## Open questions

| ID | Question | Status | Answer |
|---|---|---|---|
| Q-001 | Which model is the first target? | answered | Gemma 2 (product owner). **Reference profile, run on every GPU:** 2B (`google/gemma-2-2b-it`, BF16 ≈ 5.2 GB, SE12 ≈ 3.9 GB) — same shapes everywhere so results compare across GPUs. **Optional profiles by VRAM:** 9B (BF16 ≈ 18.5 GB → needs ≥ 24 GB; SE12 ≈ 14 GB → fits 16 GB) and 27B (BF16 ≈ 54 GB → needs 80 GB; SE12 ≈ 41 GB → fits 48 GB). Kernel-level features need ~4 GB free; the packed 2B needs ≥ 6 GB, the BF16 2B baseline ≥ 8 GB. `lws.device` reports which profiles fit |
| Q-002 | Is the tied embedding / LM head compressed too? In Gemma 2 2B it is one 256000 × 2304 matrix (~590M weights, ~23% of the bytes read per token), so leaving it raw caps the possible gain | answered | Yes (product owner). It is packed like any 2-D weight; as LM head it goes through the GEMV kernels, as embedding the runtime decodes only the tiles of the looked-up rows |
