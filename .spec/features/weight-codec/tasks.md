# Tasks: Weight codec (SE12 lossless fixed-rate format)

> feature: weight-codec

<!--
  pyproject.toml, conftest.py (pytest -> TAP bridge), onpspec.config.json and
  src/lws/__init__.py ship with the starter bundle and are mapped to T-001.
  A file listed only to force ordering is marked "(read-only dependency)" in
  Notes: onp-spec plan parallelizes by disjoint files, so sharing a file is
  how a dependent task is kept in the same lane, after its dependency.
  All Python code follows P-013 (hexagonal) and P-014 (object calisthenics);
  tests/test_principle_architecture.py, written in T-001, enforces both on
  every later task.
-->

## T-001 — Hexagonal skeleton, architecture fitness test and entropy analyzer [pending]

- Refs: US-001, AC-001, AC-002
- Files: src/lws/__init__.py, src/lws/domain/__init__.py, src/lws/domain/entropy/__init__.py, src/lws/domain/entropy/report.py, src/lws/application/__init__.py, src/lws/application/ports/__init__.py, src/lws/application/ports/codec/__init__.py, src/lws/application/ports/codec/weight_source.py, src/lws/application/codec/__init__.py, src/lws/application/codec/analyze_model.py, src/lws/adapters/__init__.py, src/lws/adapters/safetensors/__init__.py, src/lws/adapters/safetensors/weight_source.py, src/lws/analyze.py, tests/fakes.py, tests/test_principle_architecture.py, tests/test_spec_analyze.py
- Model: claude-opus-5-5
- Effort: high
- Notes: First write tests/test_principle_architecture.py, an AST-based fitness test over src/lws that enforces P-013 (import direction per layer, no relative imports, every Protocol in `lws.application.ports` has an adapter and a fake in tests/fakes.py) and P-014 (the nine rules per layer as the constitution tables them, `# calisthenics: allow` markers accepted only in adapters and printed in the test output); it carries @principle:P-013 and @principle:P-014 and must stay green for every later task. Then the analyzer: exponent/sign-mantissa histograms and entropy in `lws.domain.entropy` (numpy only, value objects such as `Bits` and `ElementCount`); the `WeightSource` port streams named tensors; the safetensors adapter reads shards lazily (torch framework, no full-model load) and hands int16 views to the domain as numpy arrays; `lws.analyze` only parses arguments and wires the adapter into the use case. Tests use small synthetic safetensors files built in tmp_path and the in-memory fake source; a separate test marked `model` runs on LWS_MODEL_DIR and only writes the report (skips without it).

## T-002 — SE12 reference codec (CPU) [pending]

- Refs: US-002, AC-003, AC-004, AC-005
- Files: src/lws/domain/se12/__init__.py, src/lws/domain/se12/codebook.py, src/lws/domain/se12/tile.py, src/lws/domain/se12/codec.py, tests/test_spec_se12_roundtrip.py, src/lws/adapters/safetensors/weight_source.py
- Model: claude-sonnet-5
- Effort: high
- Notes: src/lws/adapters/safetensors/weight_source.py is a read-only dependency (runs after T-001; used by the AC-004 real-model test). Domain code: numpy integer ops only, no torch. Follow design.md exactly (tile shape, escape budget, fallback bitmap); `TileShape`, `EscapeBudget`, `Codebook` and `Bf16Weights` (first-class collection over the int16 array) are value objects. The bit-exact test also carries @principle:P-004. AC-004 test is marked `model` (skips without LWS_MODEL_DIR — a skip is not proof, so run it on the GPU box).

## T-003 — Packed container and packer CLI [pending]

- Refs: US-003, AC-006, AC-007, AC-008
- Files: src/lws/domain/se12/container.py, src/lws/application/ports/codec/packed_store.py, src/lws/application/codec/pack_model.py, src/lws/adapters/safetensors/packed_store.py, src/lws/pack.py, tests/test_spec_pack.py, src/lws/domain/se12/codec.py
- Model: claude-sonnet-5
- Effort: medium
- Notes: src/lws/domain/se12/codec.py is a read-only dependency (runs after T-002). The container (format name, version rule, per-tensor metadata) is domain; reading/writing safetensors is the `PackedStore` adapter. Hash test carries @principle:P-003. Eligibility rules (Q-002) live in one place, in the pack use case. Pack tensor by tensor (never the whole model in RAM — ASM-015). The test that packing twice gives identical bytes and that the metadata holds no device/arch/tuning key carries @principle:P-010. The tied embedding is eligible (Q-002).

## T-017 — Entropy survey across model families [pending]

- Refs: US-001, AC-034
- Files: src/lws/domain/entropy/survey.py, src/lws/application/codec/survey_models.py, src/lws/adapters/huggingface/__init__.py, src/lws/adapters/huggingface/shard_stream_source.py, src/lws/survey.py, tests/test_spec_survey.py, src/lws/domain/se12/codec.py
- Model: claude-sonnet-5
- Effort: medium
- Notes: src/lws/domain/se12/codec.py is a read-only dependency (runs after T-003). Reuses the entropy domain (T-001) and the reference codec for projected bits and fallback share. Tensor role comes from a name-pattern table per architecture kept in one place in the domain. The Hugging Face adapter is a second `WeightSource`: it lists the repo's safetensors shards at a pinned revision, downloads one at a time with `huggingface_hub`, yields its tensors and deletes the file before the next one (peak disk = one shard). Gated checkpoints (Gemma, Llama) use `HF_TOKEN` from the environment (principle P-002). Synthetic tests on CPU; the real survey test is marked `model` and reads the directories from `LWS_SURVEY_DIRS` (skips without it). CPU only — runs on the local machine, no Kaggle quota.
