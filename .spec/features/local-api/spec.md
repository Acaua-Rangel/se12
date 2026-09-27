# Spec: Local OpenAI-compatible API

> feature: local-api

## Context

Once the packed model runs with identical answers, expose it on localhost with
the OpenAI chat-completions shape, so that local agent orchestrators and SDKs
can use it without any change on their side. Built only after
model-integration is audited; not part of the first paper.

Two backends behind the same protocol (Q-006):

- **vLLM** where the installed vLLM supports the GPU: SE12 is registered as an
  out-of-tree linear method (vLLM's pluggable weight-format API), so the server
  inherits continuous batching, CUDA graphs and vLLM's OpenAI server.
- **FastAPI fallback** everywhere else (e.g. sm_60 such as the P100, or no
  vLLM installed): a minimal server over `lws.runtime` in `mode="auto"`
  (model-integration AC-025).

Dtype rule for both: the compute profile of the GPU decides — BF16
activations on `bf16-native`, FP32 on `fp32-act`. FP16 is never used: it would
be a lossy conversion (principle P-005). On a T4 this means vLLM runs with
`dtype="float32"`.

## Stories

### US-011 — Use the compressed model from any OpenAI-compatible client

As a developer running local agents, I want a local endpoint that speaks the
OpenAI chat-completions protocol, so that my existing SDK code points to it by
changing only the base URL.

#### AC-019 — Chat completion answers in the OpenAI format

- **Given** the server started with a packed model on 127.0.0.1, with either backend
- **When** a client sends `POST /v1/chat/completions` with a messages list
- **Then** it receives HTTP 200 with a body containing `id`, `object: "chat.completion"`, `choices[0].message.content` and `usage` token counts
- **And** on the FastAPI backend, with temperature 0 the content equals what `lws.runtime` generates directly for the same prompt

#### AC-020 — Streaming works token by token

- **Given** the server running, with either backend
- **When** a client sends the same request with `"stream": true`
- **Then** it receives Server-Sent Events chunks with `object: "chat.completion.chunk"` ending with `data: [DONE]`
- **And** concatenating the chunks gives the same text as the non-streaming answer

### US-017 — The best available backend is chosen for my GPU

As a user on any supported GPU, I want `python -m lws.server` to pick vLLM
when it can run on my GPU and fall back to the minimal server otherwise, so
that the same command works from a P100 to an H100.

#### AC-035 — Backend and dtype selection, with a lossless vLLM path

- **Given** a packed model and a GPU
- **When** I run `python -m lws.server <packed_dir>` (optionally `--backend vllm|fastapi`)
- **Then** a pure selection rule picks vLLM when vLLM is importable and supports the GPU's compute capability, and FastAPI otherwise, and logs the backend, the reason and the activation dtype (BF16 on `bf16-native`, FP32 on `fp32-act`, never FP16)
- **And** forcing `--backend vllm` on an unsupported GPU exits with a message naming the GPU, its compute capability and the fallback
- **And** on the vLLM backend every eligible linear layer is served from SE12 buffers through the registered linear method (no full BF16 weight in VRAM), and with temperature 0 its greedy generations on the AC-017 prompt set satisfy the AC-017 rule (≥ 95% identical, divergences only at near-ties) against `lws.runtime`

## Out of scope

- Authentication, multi-user scheduling policy beyond what vLLM provides.
- Tensor parallelism (vLLM multi-GPU) and continuous batching on the FastAPI fallback.
- Tool/function-calling schema (can be added later).

## Assumptions

| ID | Assumption | Status | Resolution |
|---|---|---|---|
| ASM-008 | The server is single-user and bound to localhost only, so no authentication is needed | open | — |
| ASM-023 | vLLM's plugin mechanism lets an out-of-tree package register a weight-format ("quantization") config whose linear method loads SE12 tensors from the packed safetensors and calls the lws kernels | open | — |
| ASM-024 | vLLM runs Gemma 2 with `dtype="float32"` on sm_75 with an attention backend that supports its logit soft-capping | open | — (if not, the T4 uses the FastAPI fallback and the report says why) |

## Open questions

| ID | Question | Status | Answer |
|---|---|---|---|
| Q-006 | Own minimal FastAPI server, or register CompressedLinear as a custom weight format inside vLLM so it inherits batching and its OpenAI server? | answered | Both (product owner): vLLM where it supports the GPU, FastAPI fallback elsewhere (AC-035) |
