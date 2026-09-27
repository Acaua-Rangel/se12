"""Tests for the entropy analyzer (US-001).

@spec:AC-001 @spec:AC-002
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy
import pytest
import safetensors.torch
import torch

from lws import analyze
from lws.adapters.safetensors.weight_source import SafetensorsWeightSource
from lws.application.codec.analyze_model import analyze_model
from lws.application.ports.codec.weight_source import EligibleTensor, IneligibleSourceTensor
from lws.domain.weights import Bf16Weights, RawBytes, RawTensor, RawTensorPayload, ShapedWeights, TensorShape
from tests.fakes import FakeWeightSource


def _bf16_bit_pattern(values: list[float]) -> numpy.ndarray:
    tensor = torch.tensor(values, dtype=torch.bfloat16)
    return tensor.view(torch.int16).numpy().copy()


def _eligible(name: str, pattern: numpy.ndarray) -> EligibleTensor:
    weights = Bf16Weights(bit_pattern=pattern)
    shape = TensorShape(dims=(pattern.size,))
    return EligibleTensor(name=name, tensor=ShapedWeights(weights=weights, shape=shape))


def _ineligible(name: str, dtype_name: str, element_count: int = 1) -> IneligibleSourceTensor:
    raw_array = numpy.zeros(element_count, dtype=numpy.uint8)
    payload = RawTensorPayload(bytes_value=RawBytes(value=raw_array), shape=TensorShape(dims=(element_count,)))
    return IneligibleSourceTensor(name=name, tensor=RawTensor(dtype_name=dtype_name, payload=payload))


def test_analyze_model_reports_every_required_number():
    """@spec:AC-001"""
    pattern = _bf16_bit_pattern([1.0, -1.0, 2.0, -2.0, 0.5, 3.0, 4.0, 5.0])
    source = FakeWeightSource([_eligible("layer.weight", pattern)])
    report = analyze_model(source)
    per_tensor = report.per_tensor.items
    assert len(per_tensor) == 1
    named = per_tensor[0]
    assert named.name == "layer.weight"
    assert named.report.element_count.value == 8
    assert named.report.statistics.exponent.entropy.value >= 0.0
    assert named.report.statistics.byte_pair.sign_mantissa_entropy.value >= 0.0
    assert named.report.statistics.byte_pair.full_entropy.value >= 0.0
    assert 0.0 <= named.report.statistics.exponent.top_coverage.value <= 1.0
    summary = report.summary
    assert summary.weighted_average.exponent.entropy.value >= 0.0
    assert summary.ineligible.items == ()


def test_analyze_model_lists_ineligible_tensors_and_does_not_crash():
    """@spec:AC-002"""
    pattern = _bf16_bit_pattern([1.0, 2.0])
    source = FakeWeightSource(
        [
            _eligible("w", pattern),
            _ineligible("norm.scale", "torch.float32"),
            _ineligible("mask", "torch.bool"),
        ]
    )
    report = analyze_model(source)
    ineligible_names = {item.name: item.dtype for item in report.summary.ineligible.items}
    assert ineligible_names == {"norm.scale": "torch.float32", "mask": "torch.bool"}
    assert len(report.per_tensor.items) == 1


def test_top15_coverage_is_one_when_five_or_fewer_exponents_are_used():
    """@spec:AC-001 — a small, controlled distribution makes the coverage number checkable by hand."""
    pattern = _bf16_bit_pattern([1.0] * 100 + [2.0] * 100)
    source = FakeWeightSource([_eligible("t", pattern)])
    report = analyze_model(source)
    named = report.per_tensor.items[0]
    assert named.report.statistics.exponent.top_coverage.value == pytest.approx(1.0)


def test_entrypoint_writes_a_json_report(tmp_path: pathlib.Path):
    """@spec:AC-001 @spec:AC-002 — the CLI entrypoint wires the real adapter and writes JSON."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tensors = {
        "layer.weight": torch.tensor([1.0, -1.0, 2.0, 3.0], dtype=torch.bfloat16),
        "layer.bias": torch.tensor([1, 2], dtype=torch.int32),
    }
    safetensors.torch.save_file(tensors, str(model_dir / "shard.safetensors"))
    out_path = tmp_path / "reports" / "entropy.json"

    exit_code = analyze.main([str(model_dir), "--out", str(out_path)])

    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    per_tensor = payload["per_tensor"]["items"]
    assert len(per_tensor) == 1
    assert per_tensor[0]["name"] == "layer.weight"
    ineligible = payload["summary"]["ineligible"]["items"]
    assert ineligible == [{"name": "layer.bias", "dtype": "torch.int32"}]


@pytest.mark.model
def test_analyze_real_model_writes_a_report(tmp_path: pathlib.Path):
    """@spec:AC-001 — runs on LWS_MODEL_DIR only; skips (not proof) otherwise."""
    model_dir = os.environ.get("LWS_MODEL_DIR")
    if not model_dir:
        pytest.skip("LWS_MODEL_DIR not set")
    out_path = tmp_path / "entropy.json"
    exit_code = analyze.main([model_dir, "--out", str(out_path)])
    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert len(payload["per_tensor"]["items"]) > 0
