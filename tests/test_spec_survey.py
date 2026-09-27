"""Tests for the entropy survey across model families (US-001, AC-034)."""

from __future__ import annotations

import json
import os
import pathlib

import numpy
import pytest
import safetensors.torch
import torch

from lws import survey
from lws.application.codec.survey_models import ModelIdentity, NamedModelSource, survey_models
from lws.domain.entropy.survey import SurveyInput, TensorRole, survey_model
from lws.domain.weights import Bf16Weights, ShapedWeights, TensorShape
from tests.fakes import FakeWeightSource


def _concentrated_bf16(rows: int, cols: int, seed: int, exponents: tuple[int, int] = (120, 135)) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    low, high = exponents
    n = rows * cols
    exponent_bytes = rng.choice(numpy.arange(low, high, dtype=numpy.uint8), size=n)
    sign = rng.integers(0, 2, size=n).astype(numpy.uint16)
    mantissa = rng.integers(0, 0x80, size=n).astype(numpy.uint16)
    bits = (sign << 15) | (exponent_bytes.astype(numpy.uint16) << 7) | mantissa
    return bits.astype(numpy.int16)


def _input(name: str, rows: int, cols: int, seed: int) -> SurveyInput:
    pattern = _concentrated_bf16(rows, cols, seed)
    weights = Bf16Weights(bit_pattern=pattern)
    shape = TensorShape(dims=(rows, cols))
    return SurveyInput(name=name, shaped=ShapedWeights(weights=weights, shape=shape))


def _norm_input(name: str, length: int, seed: int) -> SurveyInput:
    """A genuinely 1-D tensor (dims has length 1), like a norm scale — not
    SE12-eligible (weight-codec ASM-002/Q-002), unlike a (1, N) 2-D matrix."""
    pattern = _concentrated_bf16(1, length, seed)
    weights = Bf16Weights(bit_pattern=pattern)
    shape = TensorShape(dims=(length,))
    return SurveyInput(name=name, shaped=ShapedWeights(weights=weights, shape=shape))


def _sample_items() -> list[SurveyInput]:
    return [
        _input("model.embed_tokens.weight", 256, 512, 1),
        _input("model.layers.0.self_attn.q_proj.weight", 64, 256, 2),
        _input("model.layers.0.mlp.gate_proj.weight", 128, 512, 3),
        _norm_input("model.layers.0.input_layernorm.weight", 64, 4),
    ]


def test_survey_model_reports_the_five_required_numbers_overall_and_per_role():
    """@spec:AC-034"""
    report = survey_model(_sample_items())
    overall = report.overall
    assert overall.entropy.exponent.entropy.value >= 0.0
    assert overall.entropy.byte_pair.sign_mantissa_entropy.value >= 0.0
    assert 0.0 <= overall.entropy.exponent.top_coverage.value <= 1.0
    assert overall.se12.bits_per_weight.value > 0.0
    assert 0.0 <= overall.se12.fallback_share.value <= 1.0

    roles_present = {entry.role for entry in report.by_role.items}
    assert roles_present == {TensorRole.EMBEDDING, TensorRole.ATTENTION, TensorRole.MLP_OR_EXPERT, TensorRole.OTHER}

    by_role = {entry.role: entry.statistics for entry in report.by_role.items}
    embedding_stats = by_role[TensorRole.EMBEDDING]
    assert embedding_stats.se12.bits_per_weight.value == pytest.approx(12.0, abs=0.2)
    other_stats = by_role[TensorRole.OTHER]
    assert other_stats.se12.bits_per_weight.value == 0.0  # 1-D tensor: not SE12-eligible


def test_survey_model_projected_bits_matches_the_reference_codec():
    """@spec:AC-034 — the projection is measured from the reference codec
    (T-002), not the design.md back-of-envelope 12.03 estimate."""
    items = [_input("layer.weight", 64, 512, 9)]
    report = survey_model(items)
    bits_per_weight = report.overall.se12.bits_per_weight.value
    assert 11.9 <= bits_per_weight <= 12.2


def test_survey_models_use_case_streams_from_a_fake_weight_source():
    """@spec:AC-034 — uses the SAME WeightSource port as the analyzer (T-001); no new fake needed."""
    from lws.application.ports.codec.weight_source import EligibleTensor

    items = _sample_items()
    fake_tensors = [EligibleTensor(name=item.name, tensor=item.shaped) for item in items]
    source = FakeWeightSource(fake_tensors)
    named_source = NamedModelSource(identity=ModelIdentity(label="fake-model", revision="abc123"), source=source)

    report = survey_models([named_source])

    assert len(report.entries) == 1
    entry = report.entries[0]
    assert entry.identity.label == "fake-model"
    assert entry.identity.revision == "abc123"
    assert entry.report.overall.se12.bits_per_weight.value > 0.0


def test_entrypoint_writes_a_json_report_for_a_local_model(tmp_path: pathlib.Path):
    """@spec:AC-034 — the CLI entrypoint wires the safetensors adapter and writes JSON."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    rows, cols = 32, 256
    pattern = _concentrated_bf16(rows, cols, seed=5)
    tensor = torch.from_numpy(pattern.copy()).view(torch.bfloat16).reshape(rows, cols)
    tensors = {"model.layers.0.self_attn.q_proj.weight": tensor}
    safetensors.torch.save_file(tensors, str(model_dir / "shard.safetensors"))
    out_path = tmp_path / "reports" / "entropy-survey.json"

    exit_code = survey.main([str(model_dir), "--out", str(out_path)])

    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    entries = payload["entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["identity"]["label"] == str(model_dir)
    assert entry["identity"]["revision"] is None
    by_role = entry["report"]["by_role"]["items"]
    roles = {item["role"] for item in by_role}
    assert "attention" in roles  # enum serialized to its plain value, not "TensorRole.ATTENTION"


@pytest.mark.model
def test_survey_real_models_from_local_directories():
    """@spec:AC-034 — runs on LWS_SURVEY_DIRS (colon-separated directories) only; skips (not proof) otherwise."""
    raw = os.environ.get("LWS_SURVEY_DIRS")
    if not raw:
        pytest.skip("LWS_SURVEY_DIRS not set")
    directories = [entry for entry in raw.split(os.pathsep) if entry]
    exit_code = survey.main([*directories, "--out", "reports/entropy-survey.json"])
    assert exit_code == 0
    payload = json.loads(pathlib.Path("reports/entropy-survey.json").read_text())
    assert len(payload["entries"]) == len(directories)
