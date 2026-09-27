# Tasks: Local OpenAI-compatible API

> feature: local-api

<!--
  All Python code follows P-013 (hexagonal) and P-014 (object calisthenics).
  T-018 → T-010 share src/lws/adapters/vllm/se12_linear_method.py; `lws.server`
  (T-010) is the composition root that wires both backends.
-->

## T-018 — vLLM backend: SE12 as a registered linear method [pending]

- Refs: US-017, AC-035
- Files: src/lws/domain/serving/__init__.py, src/lws/domain/serving/backend.py, src/lws/adapters/vllm/__init__.py, src/lws/adapters/vllm/se12_linear_method.py, src/lws/adapters/vllm/server_launcher.py, tests/test_spec_vllm_backend.py
- Model: claude-opus-5-5
- Effort: high
- Notes: Only after model-integration is audited. Backend and dtype selection (vLLM supported? compute profile → BF16 or FP32, never FP16) is a pure domain rule over `DeviceProperties` plus "vLLM importable / minimum compute capability", tested on CPU with fake devices. The vLLM adapter registers an SE12 weight-format config through vLLM's plugin API, loads SE12 streams in its weight loader and calls the kernels through the existing ports; it never imports a quantize function (principle P-005). vLLM goes in an optional extra (`pip install -e ".[vllm]"`) because it pins its own torch; the tests that need it are marked `gpu` and `model` and skip when vLLM is absent.

## T-010 — FastAPI fallback server and backend wiring [pending]

- Refs: US-011, US-017, AC-019, AC-020, AC-035
- Files: src/lws/application/runtime/complete_chat.py, src/lws/adapters/http/__init__.py, src/lws/adapters/http/openai_api.py, src/lws/server/__init__.py, src/lws/server/__main__.py, tests/test_spec_server.py, src/lws/adapters/vllm/se12_linear_method.py
- Model: claude-sonnet-5
- Effort: medium
- Notes: src/lws/adapters/vllm/se12_linear_method.py is a read-only dependency (runs after T-018). The chat use case (messages → prompt → generation, token usage) depends only on the `TextGenerator` port; the FastAPI adapter maps the OpenAI request/response and SSE chunk schemas (pydantic schemas are where P-014 rules 3/8/9 are relaxed). `lws.server` asks the domain rule for the backend and starts either the vLLM launcher or the FastAPI app. Protocol tests (AC-019/AC-020) run against both backends when available, using FastAPI's TestClient with the fake `TextGenerator` for the fallback, plus one `gpu`+`model` test for the equivalence clauses.
